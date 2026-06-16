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

from ..domain import ScheduledConsumer, ScriptedConsumer, build_system, load_config
from ..domain.config import Configuration
from ..domain.consumer import Consumer, ScheduleEntry
from ..domain.producing_agent import ProducerInfo
from ..domain.system import CycleResult
from .decode import decode_ordering, decode_producing
from .encode import encode_global, encode_ordering, encode_producing
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
        self, config: Configuration | None = None, horizon: int = 100, render_mode: str | None = None
    ) -> None:
        super().__init__()
        self.config = config or load_config(DEFAULT_CONFIG)
        self.config.validate()
        self.spec_ = build_spec(self.config)
        self.horizon = horizon
        self.render_mode = render_mode

        prod_ids = self.spec_.producer_ids
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
        canonical fixed-product ScriptedConsumer path untouched."""
        rng = np.random.default_rng(seed)
        consumers: dict[str, Consumer] = {}
        for c in self.config.consumers:
            # Exact legacy default (no pool, fixed quantity 1) keeps the canonical
            # ScriptedConsumer path bit-identical.
            if not c.products and c.quantity_min == 1 and c.quantity_max == 1:
                consumers[c.id] = ScriptedConsumer(c.id, c.product)
                continue
            pool = c.product_pool
            if c.is_randomized:
                schedule: list[ScheduleEntry] = [
                    (pool[int(rng.integers(len(pool)))],
                     int(rng.integers(c.quantity_min, c.quantity_max + 1)))
                    for _ in range(self.horizon)
                ]
            else:
                # Deterministic but non-default: a singleton pool and/or a fixed
                # quantity > 1. Honor exactly what was configured (read from the
                # validated `product_pool`, never the unvalidated `product` field).
                schedule = [(pool[0], c.quantity_min)]
            consumers[c.id] = ScheduledConsumer(c.id, schedule)
        return consumers

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.system = build_system(self.config, self._build_consumers(seed))
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
        }
        # capacity trajectory: mean current capacity per producer at each cycle end
        self.capacity_trace: list[float] = []

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
        # mean total current capacity per producer (collapse warning trace)
        caps = [sum(p.capacities.values()) for p in self.system.producing_agents.values()]
        self.capacity_trace.append(sum(caps) / len(caps))

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
        infos = self._producer_infos()
        if role == "prod":
            view = self.system.producing_agents[did].view()
            return encode_producing(self.spec_, view, infos, self._open_consumer_demand(), cp)
        incoming = self.system.ordering_agents[did].incoming
        return encode_ordering(self.spec_, incoming, infos, cp)

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
            result = self.system.complete_cycle()
            # inventory held into the next cycle = the holding liability (ADR-0025).
            held = sum(p.inventory_units() for p in self.system.producing_agents.values())
            reward = cycle_reward(result, held, len(self.system.producing_agents))
            self.current_cycle += 1
            self.rewards = {a: reward for a in self.agents}
            self._accumulate_episode_stats(result, reward, held)

            dead = self.system.is_dead()
            truncate = self.current_cycle >= self.horizon
            if dead:
                self.terminations = {a: True for a in self.agents}
                self.episode_stats["dead"] = True
            if truncate:
                self.truncations = {a: True for a in self.agents}
            self.episode_stats["cycles"] = self.current_cycle
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
