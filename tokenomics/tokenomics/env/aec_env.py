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

from ..domain import build_system, load_config
from ..domain.config import Configuration
from ..domain.producing_agent import ProducerInfo
from ..domain.system import CycleResult
from .decode import decode_ordering, decode_producing
from .encode import encode_global, encode_ordering, encode_producing
from .spaces import build_spec

# Action handed to step(): producing = Discrete int, ordering = logit vector,
# None = a dead-agent step.
Action = int | np.ndarray | None

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "phase1.json"

# Shared reward weights (ADR-0002, reweighted in Phase 1.5).
# Consumer fulfilment is the terminal objective; investment is instrumental
# (keep capacity alive) so it earns only a small positive — the critic learns
# its indirect value. Rejections are penalized; in-progress is a mild drag.
R_CONSUMER_COMPLETED = 10.0
R_INVESTMENT_COMPLETED = 1.0
R_IN_PROGRESS = -1.0
R_REJECTED = -5.0


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
    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.system = build_system(self.config)
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
            "investment_completed": 0,
            "consumer_rejected": 0,
            "investment_rejected": 0,
            "in_progress": 0,
            "return": 0.0,
        }
        # capacity trajectory: mean current capacity per producer at each cycle end
        self.capacity_trace: list[float] = []

    def _accumulate_episode_stats(self, result: CycleResult, reward: float) -> None:
        s = self.episode_stats
        s["consumer_completed"] += result.consumer_completed
        s["investment_completed"] += result.investment_completed
        s["consumer_rejected"] += result.consumer_rejected
        s["investment_rejected"] += result.investment_rejected
        s["in_progress"] += result.in_progress
        s["return"] += reward
        caps = [p.capacity for p in self.system.producing_agents.values()]
        self.capacity_trace.append(sum(caps) / len(caps))

    def _producer_infos(self) -> dict[str, ProducerInfo]:
        return {pid: p.info() for pid, p in self.system.producing_agents.items()}

    def observe(self, agent: str) -> np.ndarray:
        role, did = self._role[agent]
        cp = min(1.0, self.current_cycle / self.horizon)
        infos = self._producer_infos()
        if role == "prod":
            view = self.system.producing_agents[did].view()
            return encode_producing(self.spec_, view, infos, cp)
        incoming = self.system.ordering_agents[did].incoming
        return encode_ordering(self.spec_, incoming, infos, cp)

    def global_state(self) -> np.ndarray:
        """Centralized-critic input (ADR-0004): full system state, shared across
        all agents. Independent of whose turn it is."""
        cp = min(1.0, self.current_cycle / self.horizon)
        views = {pid: p.view() for pid, p in self.system.producing_agents.items()}
        return encode_global(self.spec_, views, self._producer_infos(), cp)

    def step(self, action: Action) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0
        role, did = self._role[agent]

        if role == "prod":
            view = self.system.producing_agents[did].view()
            cmd = decode_producing(int(action))
            # mask investments already in flight (domain would otherwise reject)
            if view.restoration_running:
                cmd.do_restoration = False
            if view.upgrade_running:
                cmd.do_upgrade = False
            self.system.producing_agent_action(did, cmd)
            if agent == self._last_prod_agent:
                self.system.start_ordering()
        else:
            incoming = self.system.ordering_agents[did].incoming
            cmd = decode_ordering(self.spec_, np.asarray(action), incoming)
            self.system.ordering_agent_action(did, cmd)

        if self._agent_selector.is_last():
            result = self.system.complete_cycle()
            reward = (
                R_CONSUMER_COMPLETED * result.consumer_completed
                + R_INVESTMENT_COMPLETED * result.investment_completed
                + R_IN_PROGRESS * result.in_progress
                + R_REJECTED * result.rejected
            )
            self.current_cycle += 1
            self.rewards = {a: reward for a in self.agents}
            self._accumulate_episode_stats(result, reward)

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
        caps = {pid: p.capacity for pid, p in s.producing_agents.items()}
        print(f"cycle={self.current_cycle} caps={caps} open_orders={len(s.orders)}")

    def close(self) -> None:
        pass
