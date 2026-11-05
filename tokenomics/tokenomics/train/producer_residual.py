"""Producer residual on the M5 base (add-m5-potential-game-gnn task 2.1, design D2).

The base is `entity_planner_buyahead_k4_rf3` (EXP-0079, design D8): buy-ahead k = 4 plus the early
inventory expansion. A parameter vector theta, shared by every producer and every input row, corrects
three of its decisions from features of the producer's own view and its own past cycles:

- buffer per input: k_g = 4 * exp(clip(theta_k . phi_g, -K_CLIP, K_CLIP)) cycles of recent use;
- extra buying: extra_g = round(max(0, theta_b . phi_g)) units on top of the planner's target;
- expansion: requested when theta_e . psi > 0, in addition to the base's own triggers (headroom 0,
  or headroom below the blocked shortfall), when configured, not running and fundable.

Both the buffer and the extra buying act through the input row's demand pressure (as
`baselines.buyahead_view`), so inventory headroom, the per-cycle replenish ceiling and the fund gate
bind exactly as in the base. Restoration, upgrade and the input priority are the base's. theta = 0
gives k_g = 4, extra 0 and no extra expansion: the base exactly.

Information set (ADR-0058): the producer's own `ProducerEntityView` (current own state: stock,
inbound, reservations, headroom; telemetry EMAs of its own past completions) and an agent-local
history of its own past cycles (an EMA of each input's blocked shortfall), kept per episode in a
`weakref.WeakKeyDictionary` keyed by the system object (M4 7.22 review: never by address). Nothing
of another agent's same-cycle routing or bids. The fund level (`investment_fund`) is read only for
the base's existing affordability gate.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..domain.producer_planner import DEFAULT_INPUT_STOCK_TARGET
from ..domain.producer_view import ProducerEntityView
from ..domain.producing_agent import ProducingAgentCommand
from ..eval.baselines import (
    _action_from_producing_command,
    blocked_shortfall,
    plan_reservations_first,
    recent_input_use,
)

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv

Policy = Callable[["TokenomicsAECEnv", str], np.ndarray]

BASE_PRODUCER = "entity_planner_buyahead_k4_rf3"
BASE_K = 4.0
K_CLIP = 2.0  # k_g in [4 e^-2, 4 e^2] ~ [0.54, 29.6] cycles
HISTORY_ALPHA = 0.3  # the telemetry's EMA weight
INPUT_FEATURES = ("bias", "stock_cover", "inbound_cover", "blocked_cover", "log_use",
                  "headroom_frac", "shortfall_ema_cover", "latency", "depth")
PRODUCER_FEATURES = ("bias", "headroom_frac", "shortfall_cover", "shortfall_ema_cover",
                     "fill_frac")
N_IN, N_P = len(INPUT_FEATURES), len(PRODUCER_FEATURES)
THETA_DIM = 2 * N_IN + N_P  # theta = [theta_k (N_IN), theta_b (N_IN), theta_e (N_P)]
TTL_SCALE = 2.0  # latency normalised by the default order TTL
DEPTH_SCALE = 3.0


def split_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (THETA_DIM,):
        raise ValueError(f"theta must have shape ({THETA_DIM},), got {theta.shape}")
    return theta[:N_IN], theta[N_IN:2 * N_IN], theta[2 * N_IN:]


def input_features(view: ProducerEntityView, shortfall_ema: dict) -> dict:
    """phi_g per input product g (INPUT_FEATURES order), from the producer's own view and its own
    past-cycle shortfall EMA. Every cover term is normalised by 1 + recent use of g."""
    use = recent_input_use(view)
    lim = max(1, view.globals.inventory_limit)
    headroom = max(0, view.globals.inventory_headroom) / lim
    latency: dict = {}
    depth: dict = {}
    for (xi, gi), _n in view.relations.product_input_required.items():
        g = view.input_rows[gi].input_product
        row = view.product_rows[xi]
        latency[g] = max(latency.get(g, 0.0), float(row.recent_completion_latency))
        depth[g] = max(depth.get(g, 0), int(row.product_depth))
    out = {}
    for r in view.input_rows:
        g = r.input_product
        u = 1.0 + use.get(g, 0.0)
        out[g] = np.array([
            1.0, r.inventory / u, r.outstanding_inbound / u, r.blocked_reservation_qty / u,
            math.log1p(use.get(g, 0.0)), headroom, shortfall_ema.get(g, 0.0) / u,
            latency.get(g, 0.0) / TTL_SCALE, depth.get(g, 0) / DEPTH_SCALE])
    return out


def producer_features(view: ProducerEntityView, shortfall_ema: dict) -> np.ndarray:
    """psi (PRODUCER_FEATURES order)."""
    lim = max(1, view.globals.inventory_limit)
    use = 1.0 + sum(recent_input_use(view).values())
    short = sum(blocked_shortfall(view).values())
    return np.array([1.0, max(0, view.globals.inventory_headroom) / lim, short / use,
                     sum(shortfall_ema.values()) / use, view.globals.inventory_units / lim])


def residual_view(view: ProducerEntityView, theta_k: np.ndarray, theta_b: np.ndarray,
                  phi: dict) -> ProducerEntityView:
    """The view the base plans its replenishment from. The planner's target per input is
    max(DEFAULT_INPUT_STOCK_TARGET, pressure + blocked) and it buys target - (stock + inbound),
    clamped at 0. Each row's pressure is raised so that the target becomes
    max(T, have) + extra_g with T = max(floor, pressure + blocked + ceil(k_g x recent use)) and
    have = stock + inbound: the buffer at k_g, then extra_g units on top of the planner's want
    (design D2), never absorbed by the floor or by stock already above T (review m5-2-1). At
    theta = 0 this is `buyahead_view(view, 4)` exactly."""
    use = recent_input_use(view)
    rows = []
    for r in view.input_rows:
        f = phi[r.input_product]
        k = BASE_K * math.exp(max(-K_CLIP, min(K_CLIP, float(theta_k @ f))))
        extra = int(math.floor(max(0.0, float(theta_b @ f)) + 0.5))
        pressure = r.downstream_demand_pressure + math.ceil(k * use.get(r.input_product, 0.0))
        if extra:  # extra_g on top of the planner's want, also when stock exceeds the target
            target = max(DEFAULT_INPUT_STOCK_TARGET, pressure + r.blocked_reservation_qty)
            have = r.inventory + r.outstanding_inbound
            pressure = max(target, have) + extra - r.blocked_reservation_qty
        rows.append(replace(r, downstream_demand_pressure=pressure))
    return replace(view, input_rows=rows)


def plan_residual(view: ProducerEntityView, theta: np.ndarray, shortfall_ema: dict, *,
                  max_replenish: int, replenish_cost_per_unit, funding_budget,
                  inventory_expansion_increases: int, inventory_expansion_cost
                  ) -> ProducingAgentCommand:
    """The base's command with theta's corrections (module docstring)."""
    theta_k, theta_b, theta_e = split_theta(theta)
    zero = not (theta_k.any() or theta_b.any())
    plan_view = None if zero else residual_view(view, theta_k, theta_b,
                                                input_features(view, shortfall_ema))
    # the base (rf3): buy-ahead k = 4 (or theta's targets), its early expansion; input priority
    # scored from the unmodified view
    cmd = plan_reservations_first(
        view, ahead_cycles=BASE_K, max_replenish=max_replenish,
        replenish_cost_per_unit=replenish_cost_per_unit, funding_budget=funding_budget,
        kept_room=0, inventory_expansion_increases=inventory_expansion_increases,
        inventory_expansion_cost=inventory_expansion_cost, blocked_first=False, keep_room=False,
        expand_early=True, plan_view=plan_view)
    if cmd.do_inventory_expansion or not theta_e.any():
        return cmd
    psi = producer_features(view, shortfall_ema)
    if float(theta_e @ psi) > 0 and view.globals.active_investments[2] == 0.0 \
            and inventory_expansion_increases > 0 and _affordable(view, cmd, funding_budget,
                                                                 replenish_cost_per_unit,
                                                                 inventory_expansion_cost):
        return replace(cmd, do_inventory_expansion=True)
    return cmd


def _affordable(view: ProducerEntityView, cmd: ProducingAgentCommand, funding_budget,
                replenish_cost_per_unit, expansion_cost) -> bool:
    """The planner's spending order: restoration, replenishment, upgrade, then expansion."""
    if funding_budget is None or not expansion_cost or expansion_cost <= 0:
        return True
    budget = float(max(0, funding_budget))
    if cmd.do_restoration:
        budget -= next((int(r.restore_cost) for r in view.capacity_rows if r.restore_cost > 0), 0)
    if replenish_cost_per_unit:
        budget -= sum(cmd.replenishments.values()) * float(replenish_cost_per_unit)
    if cmd.do_upgrade:
        budget -= next((int(r.upgrade_cost) for r in view.capacity_rows if r.upgrade_cost > 0), 0)
    return budget >= expansion_cost


def producer_residual_policy(theta: np.ndarray) -> Policy:
    """A producing policy playing the residual with `theta` (THETA_DIM values, shared by every
    producer). Episode state (each producer's per-input shortfall EMA) is weak-keyed by the system
    and read before the current cycle enters it."""
    theta = np.asarray(theta, dtype=np.float64).copy()
    split_theta(theta)  # shape check
    state: "weakref.WeakKeyDictionary[object, dict]" = weakref.WeakKeyDictionary()

    def policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        _, pid = env._role[agent]
        mechanics = env.spec_.producer_mechanics[pid]
        system = env.system
        view = system.producer_entity_view(pid)
        per_system = state.setdefault(system, {})
        cycle = system.cycle_counter
        last_cycle, ema = per_system.get(pid, (None, {}))
        if last_cycle is not None and cycle < last_cycle:
            last_cycle, ema = None, {}
        cmd = plan_residual(view, theta, ema, max_replenish=env.spec_.max_replenish[pid],
                            replenish_cost_per_unit=mechanics.replenish_cost_per_unit,
                            funding_budget=system.investment_fund,
                            inventory_expansion_increases=mechanics.inventory_expansion_increases,
                            inventory_expansion_cost=mechanics.inventory_expansion_cost)
        if cycle != last_cycle:
            short = blocked_shortfall(view)
            per_system[pid] = (cycle, {g: HISTORY_ALPHA * short.get(g, 0)
                                       + (1.0 - HISTORY_ALPHA) * ema.get(g, 0.0)
                                       for g in set(short) | set(ema)})
        return _action_from_producing_command(env.spec_, pid, cmd)

    policy.theta = theta  # type: ignore[attr-defined]
    policy.state = state  # type: ignore[attr-defined]
    return policy
