"""PettingZoo AEC environment for the tokenomics simulation (ADR-0001/0002).

One cycle = a full pass over agents: all producing agents act (investment
decisions), then all ordering agents act (token distribution). After the last
producing agent, the env runs `start_ordering`; after the last ordering agent it
runs `complete_cycle`, emits the shared reward, and advances the cycle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector

from ..domain import (
    ScheduledConsumer,
    ScriptedConsumer,
    build_system,
    load_config,
    product_stream_agents,
)
from ..domain.common import ConsumerId, stream_id
from ..domain.config import Configuration
from ..domain.consumer import Consumer, ScheduleEntry
from ..domain.producing_agent import ProducerInfo
from ..domain.system import CycleResult
from .decode import decode_ordering, decode_producing
from .encode import (
    encode_global,
    encode_ordering,
    encode_ordering_enriched,
    encode_ordering_stream,
    encode_producing,
    encode_producing_entity,
)
from .demand_digest import demand_digest
from .demand_drift import draw_schedule_entry, drift_schedules  # noqa: F401 (re-export)
from .demand_patterns import pattern_schedules
from .shock_schedule import build_schedule, shock_digest
from .side_effects import SideEffectTracker, capacity_ratio
from .spaces import build_spec

# Action handed to step(): both roles emit a raw float vector (Box); None = a
# dead-agent step. ADR-0026 made the producing action continuous too.
Action = np.ndarray | None

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "phase1.json"


# Shared reward weights (ADR-0002, recalibrated in ADR-0042).
# Consumer fulfilment is the terminal objective. Instrumental producer-order terms
# and direct rejection penalties are intentionally excluded at scale: rejection is
# penalized implicitly as forgone fulfilment, and only live systems keep completing
# units. Consumer completion is weighted by fulfilled UNITS, not order counts.
R_CONSUMER_PER_UNIT = 10.0  # the objective: delivered consumer units
# Inventory holding penalty (ADR-0025 B6) NORMALIZED per producer (ADR-0042): the
# over-stocking counter-pressure, applied in the env (tokens carry no value). Total
# held inventory grows with producer count, so the penalty is divided by the number
# of producers — keeping its per-producer weight comparable across phase1 and scale.
R_HOLDING_PER_UNIT = 0.1


def cycle_reward(result: CycleResult, held: int, num_producers: int = 1) -> float:
    """Shared per-cycle training reward (ADR-0002/0025/0042). Consumer fulfilment is
    the SOLE objective, weighted by delivered UNITS (+10/unit), minus a per-producer-
    normalized inventory holding penalty (B6 overstock pressure).

    Finding B (ADR-0042): the former instrumental terms — investment/replenishment
    completed (+1), in-progress (−1), consumer rejection (−5/unit), producer-order
    rejection (−5) — are dropped. At producer scale they scaled with producer count
    and demand, swamping fulfilment and making do-nothing the return-maximizer.
    Rejecting demand is already penalised implicitly: the forgone +10/unit. Survival
    is rewarded implicitly: only a live system can keep completing orders."""
    return (
        R_CONSUMER_PER_UNIT * result.consumer_completed_units
        - R_HOLDING_PER_UNIT * held / max(1, num_producers)
    )


class TokenomicsAECEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "tokenomics_aec_v0", "is_parallelizable": False}

    def __init__(
        self,
        config: Configuration | None = None,
        horizon: int = 100,
        render_mode: str | None = None,
        enriched_orderer_obs: bool = False,
        product_stream_ordering: bool = False,
        corrected_stream_k: bool = False,
        entity_producer_obs: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or load_config(DEFAULT_CONFIG)
        self.config.validate()
        # ADR-0052 / EXP-0007A: opt-in product-stream ordering agents (one per
        # (source, product)). Off by default so legacy checkpoints keep their exact
        # agent layout + observation contract.
        self.product_stream_ordering = product_stream_ordering
        # PRODUCT_AGNOSTIC O0: opt in to the corrected product-stream contract (v4),
        # whose backlog bound is per-stream (k_stream) instead of the global legacy k.
        # Requires product_stream_ordering; off by default so v3 checkpoints stay valid.
        self.corrected_stream_k = corrected_stream_k
        # EXP-0006A: opt in to the enriched (capacity-aware) ordering obs. Off by
        # default so M2-v1 / v1 checkpoints keep their exact observation contract.
        # EXP-0007A: product-stream agents use the simplified per-stream obs (v3,
        # ADR-0052); it supersedes the enriched flag since the obs is per-stream.
        if product_stream_ordering:
            orderer_obs_version = 4 if corrected_stream_k else 3
        elif enriched_orderer_obs:
            orderer_obs_version = 2
        else:
            orderer_obs_version = 1
        # ADR-0053 / EXP-0010A: opt in to the entity-factored producer obs (rows +
        # relations, version 2). Off by default so flat_v1 producer checkpoints keep
        # their exact observation contract; the command/auction are untouched.
        self.entity_producer_obs = entity_producer_obs
        producer_obs_version = 2 if entity_producer_obs else 1
        self.spec_ = build_spec(
            self.config,
            orderer_obs_version=orderer_obs_version,
            producer_obs_version=producer_obs_version,
        )
        self.horizon = horizon
        self.render_mode = render_mode

        prod_ids = self.spec_.producer_ids
        if product_stream_ordering:
            order_ids = [stream_id(src, prod) for src, prod in product_stream_agents(self.config)]
        else:
            order_ids = [c.id for c in self.config.consumers] + list(prod_ids)
        self._prod_agents = [f"prod_{pid}" for pid in prod_ids]
        self._order_agents = [f"order_{aid}" for aid in order_ids]
        self.possible_agents = self._prod_agents + self._order_agents

        # env agent name -> (role, domain id)
        self._role: dict[str, tuple[str, str]] = {}
        for pid in prod_ids:
            self._role[f"prod_{pid}"] = ("prod", pid)
        for aid in order_ids:
            self._role[f"order_{aid}"] = ("order", aid)
        self._last_prod_agent = self._prod_agents[-1]

        self._prod_obs_space = self.spec_.producing_obs_space()
        self._prod_act_space = self.spec_.producing_action_space()
        self._order_obs_space = self.spec_.ordering_obs_space()
        self._order_act_space = self.spec_.ordering_action_space()

        self.system = None
        self.current_cycle = 0

    # --- spaces ---
    def observation_space(self, agent: str) -> spaces.Space:
        return self._prod_obs_space if self._role[agent][0] == "prod" else self._order_obs_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._prod_act_space if self._role[agent][0] == "prod" else self._order_act_space

    # --- lifecycle ---
    def _build_consumers(self, seed: int | None) -> dict[str, Consumer]:
        """Build this episode's consumers. Randomized consumers (B9) get a seeded
        per-cycle (product, quantity) schedule drawn here in the env layer so the
        domain stays RNG-free (CLAUDE.md); same seed -> identical schedule, so the
        episode demand replays deterministically. Non-randomized consumers keep the
        canonical fixed-product ScriptedConsumer path untouched. An explicit config
        `schedule` (design D8) is replayed as-is and draws nothing from the RNG, so
        every episode sees the same demand whatever the seed.

        Also records each consumer's schedule as built in `self._demand_schedules`, the
        input of the `demand_digest` stat (design D5, env/demand_digest.py)."""
        rng = np.random.default_rng(seed)
        # Demand drift (add-shocks-i-drift D4): every B9-randomized consumer drifts, drawn on
        # its own `drift` stream (env/demand_drift.py), so the legacy stream above is never
        # read for them and every shock stream is untouched.
        drift = self.config.demand_drift
        drifted: dict[ConsumerId, list[ScheduleEntry]] = {}
        if drift is not None:
            if seed is None:  # refused whatever N is, as for shock generators (D3)
                raise ValueError("demand drift needs an integer episode seed; "
                                 "reset(seed=None) would not be reproducible")
            drifting = [c for c in self.config.consumers if c.is_randomized]
            if drifting:
                drifted = drift_schedules(drifting, drift.j, int(seed), self.horizon)
        # Demand pattern (add-m3-development-audit D11): every B9-randomized consumer follows it,
        # drawn on the pattern's own stream (env/demand_patterns.py); validation keeps it and
        # drift apart, so it takes the drift's place and never reads the legacy stream.
        pattern = self.config.demand_pattern
        if pattern is not None:
            if seed is None:
                raise ValueError("a demand pattern needs an integer episode seed; "
                                 "reset(seed=None) would not be reproducible")
            patterned_consumers = [c for c in self.config.consumers if c.is_randomized]
            if patterned_consumers:
                drifted = pattern_schedules(patterned_consumers, pattern, int(seed), self.horizon)
        consumers: dict[str, Consumer] = {}
        self._demand_schedules: dict[ConsumerId, tuple[ScheduleEntry, ...]] = {}
        for c in self.config.consumers:
            if c.schedule:
                # Copy: ScheduledConsumer holds a mutable list (future_fork rewrites
                # randomized suffixes in place); the frozen config must stay untouched.
                consumers[c.id] = ScheduledConsumer(c.id, list(c.schedule))
                self._demand_schedules[c.id] = tuple(c.schedule)
                continue
            # Exact legacy default (no pool, fixed quantity 1) keeps the canonical
            # ScriptedConsumer path bit-identical.
            if not c.products and c.quantity_min == 1 and c.quantity_max == 1:
                consumers[c.id] = ScriptedConsumer(c.id, c.product)
                self._demand_schedules[c.id] = ((c.product, 1),)
                continue
            pool = c.product_pool
            schedule: list[ScheduleEntry]
            if c.id in drifted:
                schedule = drifted[c.id]
            elif c.is_randomized:
                schedule = [
                    draw_schedule_entry(rng, pool, c.quantity_min, c.quantity_max)
                    for _ in range(self.horizon)
                ]
            else:
                # Deterministic but non-default: a singleton pool and/or a fixed
                # quantity > 1. Honor exactly what was configured (read from the
                # validated `product_pool`, never the unvalidated `product` field).
                schedule = [(pool[0], c.quantity_min)]
            consumers[c.id] = ScheduledConsumer(c.id, schedule)
            self._demand_schedules[c.id] = tuple(schedule)
        return consumers

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        # Shock schedule (ADR-0055, add-shocks-i-drift D3): drawn here, before any action,
        # from config + episode seed only, on per-kind streams that never touch the demand
        # RNG; the domain replays it. Private: no observation reads it (unannounced).
        self._shock_schedule = build_schedule(self.config, seed, self.horizon)
        self.system = build_system(
            self.config, self._build_consumers(seed),
            product_stream_ordering=self.product_stream_ordering,
            shock_schedule=self._shock_schedule,
        )
        self.current_cycle = 0
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.episode_stats = {
            "cycles": 0,
            "dead": False,
            "consumer_completed": 0,
            "consumer_completed_units": 0,  # #3: delivered consumer units
            "investment_completed": 0,
            "replenishment_completed": 0,
            "consumer_rejected": 0,
            "consumer_rejected_units": 0,  # #3: unmet consumer units
            "investment_rejected": 0,
            "replenishment_rejected": 0,
            "in_progress": 0,
            "holding_units": 0,
            "return": 0.0,
            # Per-consumer accounting (design D2, add-measurement-gate-harness): consumer id
            # -> units over the played-cycle cohort. Every consumer is listed from the start,
            # so one that never orders reports 0 ordered units.
            "consumer_ordered_units": {cid: 0 for cid in self.system.consumers},
            "consumer_fulfilled_units": {cid: 0 for cid in self.system.consumers},
            # Consumer units completed in each played cycle, in order (add-shocks-i-drift D6):
            # the input of the resilience rows (eval/resilience.py), compared against the
            # shock-free twin. Its sum equals `consumer_completed_units`.
            "consumer_fulfilled_units_trace": [],
        }
        # Demand-schedule digest (design D5): sha256 hex of every consumer's schedule as
        # built above. The gate compares it across contracts per (world, episode seed);
        # what is and is not digested is documented in env/demand_digest.py.
        if set(self._demand_schedules) != set(self.system.consumers):
            raise ValueError(
                f"demand digest covers {sorted(self._demand_schedules)}, "
                f"episode consumers are {sorted(self.system.consumers)}"
            )
        self.episode_stats["demand_digest"] = demand_digest(self._demand_schedules)
        # Shock-schedule digest + event count (D3): fixed at reset, compared across subjects.
        self.episode_stats["shock_digest"] = shock_digest(self._shock_schedule)
        self.episode_stats["shock_events"] = len(self._shock_schedule)
        # Side-effect metrics (design D3): raw quantities only, no X/Y verdict. The key
        # list and every edge case (None on a zero denominator) are documented in
        # env/side_effects.py; written now so they exist before the first cycle.
        self._side_effects = SideEffectTracker(
            r_start=capacity_ratio(self.system.producing_agents.values())
        )
        self._side_effects.write(self.episode_stats)
        # capacity trajectory: mean current capacity per producer at each cycle end
        self.capacity_trace: list[float] = []

    def _units_placed_for(self, cycle: int) -> dict[ConsumerId, int]:
        """Consumer units per consumer of the orders first played in `cycle` (design D2).
        Read just before that cycle's `complete_cycle`: every such order is still open
        then (orders only resolve inside `complete_cycle`), and the next cycle's orders do
        not exist yet. Counting at play time, not at placement, is what keeps the final
        `start_cycle`'s never-played orders (truncation or death) out of the cohort."""
        units: dict[ConsumerId, int] = {}
        for o in self.system.orders.values():
            req = o.consumer_request
            if req is not None and o.placement_cycle == cycle:
                units[req.consumer_id] = units.get(req.consumer_id, 0) + o.quantity
        return units

    def _accumulate_consumer_units(self, placed: dict[ConsumerId, int], result: CycleResult) -> None:
        """Ordered = whole quantity of every played order, resolved or still open at the
        end; fulfilled = quantity of completed orders. Completion is whole-order (ADR-0025:
        a partly delivered order that expires is rejected for its full quantity), so this
        matches `consumer_completed_units` summed over consumers."""
        ordered = self.episode_stats["consumer_ordered_units"]
        fulfilled = self.episode_stats["consumer_fulfilled_units"]
        for cid, q in placed.items():
            ordered[cid] += q
        for out in result.consumer_outcomes:
            if out.completed:
                fulfilled[out.consumer_id] += out.quantity

    def _accumulate_episode_stats(self, result: CycleResult, reward: float, held: int) -> None:
        s = self.episode_stats
        s["consumer_completed"] += result.consumer_completed
        s["consumer_completed_units"] += result.consumer_completed_units
        s["investment_completed"] += result.investment_completed
        s["replenishment_completed"] += result.replenishment_completed
        s["consumer_rejected"] += result.consumer_rejected
        s["consumer_rejected_units"] += result.consumer_rejected_units
        s["investment_rejected"] += result.investment_rejected
        s["replenishment_rejected"] += result.replenishment_rejected
        s["in_progress"] += result.in_progress
        s["holding_units"] += held
        s["return"] += reward
        # mean total current capacity per producer (collapse warning trace), from the
        # end-of-cycle snapshot taken before the next cycle's shocks (add-shocks-i-drift D2),
        # so cap_final / cap_min never see an unplayed cycle; without shocks = live state.
        caps = [sum(snap.capacities.values()) for snap in result.end_capacities]
        self.capacity_trace.append(sum(caps) / len(caps))
        s["consumer_fulfilled_units_trace"].append(result.consumer_completed_units)

    def _producer_infos(self) -> dict[str, ProducerInfo]:
        return {pid: p.info() for pid, p in self.system.producing_agents.items()}

    def _open_consumer_demand(self) -> dict[int, tuple[int, int]]:
        """Aggregate open consumer orders by product -> (unresolved qty, available
        tokens) (#2). Consumer orders are placed before producers act, so this is a
        real demand signal at observation time, not hidden state."""
        demand: dict[int, tuple[int, int]] = {}
        for o in self.system.orders.values():
            if o.consumer_request is None:
                continue
            q, t = demand.get(o.product, (0, 0))
            demand[o.product] = (q + o.unresolved_quantity(), t + o.available_tokens())
        return demand

    def observe(self, agent: str) -> np.ndarray:
        role, did = self._role[agent]
        cp = min(1.0, self.current_cycle / self.horizon)
        # Producer infos (one `info()` per producer) are built only for the two
        # encoders that read them, not for every observation (task 8.4).
        if role == "prod":
            if self.spec_.producer_obs_version >= 2:
                return encode_producing_entity(self.spec_, self.system.producer_entity_view(did), cp)
            view = self.system.producing_agents[did].view()
            return encode_producing(
                self.spec_, view, self._producer_infos(), self._open_consumer_demand(), cp
            )
        if self.spec_.orderer_obs_version >= 3:
            return encode_ordering_stream(self.spec_, self.system.product_stream_view(did), cp)
        if self.spec_.orderer_obs_version >= 2:
            return encode_ordering_enriched(self.spec_, self.system.orderer_economy_view(did), cp)
        incoming = self.system.ordering_agents[did].incoming
        return encode_ordering(self.spec_, incoming, self._producer_infos(), cp)

    def global_state(self) -> np.ndarray:
        """Centralized-critic input (ADR-0004): full system state, shared across
        all agents. Independent of whose turn it is."""
        cp = min(1.0, self.current_cycle / self.horizon)
        views = {pid: p.view() for pid, p in self.system.producing_agents.items()}
        return encode_global(self.spec_, views, self._producer_infos(), self._open_consumer_demand(), cp)

    def step(self, action: Action) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0
        role, did = self._role[agent]

        if role == "prod":
            # decode owns all gating now (ADR-0026): already-running investments
            # and joint replenishment headroom are handled in decode_producing.
            view = self.system.producing_agents[did].view()
            cmd = decode_producing(self.spec_, np.asarray(action), view)
            self.system.producing_agent_action(did, cmd)
            if agent == self._last_prod_agent:
                self.system.start_ordering()
        else:
            view = self.system.ordering_agents[did].view(self.system.producer_infos)
            cmd = decode_ordering(self.spec_, np.asarray(action), view)
            self.system.ordering_agent_action(did, cmd)

        if self._agent_selector.is_last():
            # system.cycle_counter = the cycle complete_cycle is about to play.
            placed = self._units_placed_for(self.system.cycle_counter)
            result = self.system.complete_cycle()
            # inventory held into the next cycle = the holding liability (ADR-0025).
            held = sum(p.inventory_units() for p in self.system.producing_agents.values())
            reward = cycle_reward(result, held, len(self.system.producing_agents))
            self.current_cycle += 1
            self.rewards = {a: reward for a in self.agents}
            self._accumulate_episode_stats(result, reward, held)
            self._accumulate_consumer_units(placed, result)

            dead = self.system.is_dead()
            truncate = self.current_cycle >= self.horizon
            if dead:
                self.terminations = {a: True for a in self.agents}
                self.episode_stats["dead"] = True
            if truncate:
                self.truncations = {a: True for a in self.agents}
            self.episode_stats["cycles"] = self.current_cycle
            # Post-cycle capacity (after degradation and investment completions), so the
            # death cycle's collapse lands in r_end. Read from the domain's end-of-cycle
            # snapshot, taken before the next cycle's shocks (add-shocks-i-drift D2), so a hit
            # in an unplayed cycle never reaches r_end; without shocks it equals live state.
            self._side_effects.observe_cycle(result, capacity_ratio(result.end_capacities))
            self._side_effects.write(self.episode_stats)
            self.infos = {a: {"cycle": self.current_cycle} for a in self.agents}
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def render(self) -> None:
        if self.render_mode is None:
            return
        s = self.system
        caps = {pid: dict(p.capacities) for pid, p in s.producing_agents.items()}
        print(f"cycle={self.current_cycle} caps={caps} open_orders={len(s.orders)}")

    def close(self) -> None:
        pass
