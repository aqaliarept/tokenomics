"""Per-consumer, played-cycle accounting in the env (design D2, add-measurement-gate-harness;
side-effect-guards "Per-consumer accounting").

`episode_stats` reports `consumer_ordered_units` / `consumer_fulfilled_units`, each a dict
consumer id -> units. An order counts as ordered once the cycle it was placed for has been
played, so the orders the final `start_cycle` places for the next, never-played cycle are
excluded. Orders still open when the episode ends stay ordered and unfulfilled.

Every scenario also checks the closure: ordered = fulfilled + rejected + still open (played
cohort), and fulfilled summed over consumers = the legacy `consumer_completed_units`.
"""

from __future__ import annotations

import numpy as np

from tokenomics.domain.common import ConsumerId, Product, Tokens
from tokenomics.domain.config import (
    Configuration,
    ConsumerConfig,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
from tokenomics.domain.consumer import Consumer
from tokenomics.domain.requests import ConsumerRequest
from tokenomics.env import TokenomicsAECEnv
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy

P, Q = 1, 2


def _config(*consumers: ConsumerConfig, cap: int = 100, degradation: int = 5) -> Configuration:
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(P, {"capacity-1": 2}), ProcessSheet(Q, {"capacity-1": 3})],
        producer_configs=[
            ProducingAgentConfig(
                "p1", {"capacity-1": cap}, degradation,
                Restoration(P, 50, "capacity-1"), Upgrade(P, 40, "capacity-1"),
                produces=frozenset({P, Q}),
            ),
        ],
        consumers=list(consumers),
    )
    cfg.validate()
    return cfg


def _scheduled(cid: str, *entries: tuple[Product, int]) -> ConsumerConfig:
    return ConsumerConfig(cid, entries[0][0], schedule=tuple(entries))


def _run(env: TokenomicsAECEnv, prod: str = "always_restore", order: str = "proportional") -> dict:
    """Play one episode (seed 0) to its end with the named baselines."""
    env.reset(seed=0)
    for a in env.possible_agents:
        env.action_space(a).seed(0)
    prod_policy: Policy = PRODUCING_BASELINES[prod]
    order_policy: Policy = ORDERING_BASELINES[order]
    for agent in env.agent_iter(max_iter=env.horizon * len(env.possible_agents) * 3 + 100):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        policy = prod_policy if env._role[agent][0] == "prod" else order_policy
        env.step(policy(env, agent))
    return env.episode_stats


def _open_units(env: TokenomicsAECEnv, played: bool) -> dict[ConsumerId, int]:
    """Units of consumer orders still open at the end, split by whether their placement
    cycle was played (<= cycles played) or is the never-played next cycle."""
    units: dict[ConsumerId, int] = {}
    for o in env.system.orders.values():
        req = o.consumer_request
        if req is None or (o.placement_cycle <= env.current_cycle) != played:
            continue
        units[req.consumer_id] = units.get(req.consumer_id, 0) + o.quantity
    return units


def _assert_closure(env: TokenomicsAECEnv, stats: dict) -> None:
    ordered, fulfilled = stats["consumer_ordered_units"], stats["consumer_fulfilled_units"]
    assert set(ordered) == set(fulfilled) == set(env.system.consumers)
    assert sum(fulfilled.values()) == stats["consumer_completed_units"]
    open_played = _open_units(env, played=True)
    assert sum(ordered.values()) == (
        stats["consumer_completed_units"] + stats["consumer_rejected_units"]
        + sum(open_played.values())
    )
    for cid in ordered:
        assert 0 <= fulfilled[cid] <= ordered[cid]


def test_stats_list_units_per_consumer() -> None:
    cfg = _config(_scheduled("c1", (P, 1)), _scheduled("c2", (Q, 2)))
    env = TokenomicsAECEnv(config=cfg, horizon=4)
    stats = _run(env)
    # 4 played cycles: c1 orders 1 unit per cycle, c2 2 units per cycle.
    assert stats["consumer_ordered_units"] == {"c1": 4, "c2": 8}
    assert set(stats["consumer_fulfilled_units"]) == {"c1", "c2"}
    assert stats["consumer_completed_units"] > 0
    _assert_closure(env, stats)


def test_horizon_one_counts_only_the_played_cycle() -> None:
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 2), (P, 5))), horizon=1)
    stats = _run(env)
    assert stats["cycles"] == 1
    assert stats["consumer_ordered_units"] == {"c1": 2}
    # The final start_cycle placed cycle 2's order (x5); it exists but is never counted.
    assert _open_units(env, played=False) == {"c1": 5}
    _assert_closure(env, stats)


def test_unplayed_next_cycle_orders_excluded_cyclic_schedule() -> None:
    # Cyclic schedule P x1, Q x3: cycles 1..3 order 1 + 3 + 1 = 5 units. The final
    # start_cycle places cycle 4's Q x3, which would make it 8 if counted.
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 1), (Q, 3))), horizon=3)
    stats = _run(env)
    assert stats["consumer_ordered_units"] == {"c1": 5}
    assert _open_units(env, played=False) == {"c1": 3}
    _assert_closure(env, stats)


def test_demand_unresolved_at_horizon_counts_ordered_not_fulfilled() -> None:
    # 200 units of P need 400 capacity-1; the lone producer holds 100, so no order can
    # ever clear. The last played cycle's order is still in progress (TTL 2) at the end.
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 200))), horizon=3)
    stats = _run(env)
    assert stats["consumer_ordered_units"] == {"c1": 600}
    assert stats["consumer_fulfilled_units"] == {"c1": 0}
    last = [o for o in env.system.orders.values()
            if o.consumer_request is not None and o.placement_cycle == 3]
    assert [o.quantity for o in last] == [200]
    _assert_closure(env, stats)
    assert stats["consumer_rejected_units"] < 600  # the open order is not a rejection


def test_early_death_counts_cycles_played_up_to_death() -> None:
    # 50% degradation of max per cycle with no restoration: capacity hits 0 in cycle 2.
    # Cycle k orders k units, so an off-by-one cohort shows in the totals.
    schedule = tuple((P, k) for k in range(1, 13))
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", *schedule), degradation=50), horizon=12)
    stats = _run(env, prod="noop")
    assert stats["dead"] is True
    t = stats["cycles"]
    assert t == 2
    assert stats["consumer_ordered_units"] == {"c1": 1 + 2}
    assert _open_units(env, played=False) == {"c1": 3}  # cycle t+1's order, never played
    _assert_closure(env, stats)


class _SilentConsumer:
    """Receives emission but never orders."""

    def __init__(self, consumer_id: ConsumerId) -> None:
        self._id = consumer_id

    def id(self) -> ConsumerId:
        return self._id

    def emit(self, tokens: Tokens) -> None:
        pass

    def order(self) -> list[ConsumerRequest]:
        return []


class _SilentC2Env(TokenomicsAECEnv):
    def _build_consumers(self, seed: int | None) -> dict[str, Consumer]:
        consumers = super()._build_consumers(seed)
        consumers["c2"] = _SilentConsumer("c2")
        return consumers


def test_zero_order_consumer_reported_with_zero_units() -> None:
    cfg = _config(_scheduled("c1", (P, 1)), _scheduled("c2", (P, 1)))
    env = _SilentC2Env(config=cfg, horizon=3)
    stats = _run(env)
    assert stats["consumer_ordered_units"] == {"c1": 3, "c2": 0}
    assert stats["consumer_fulfilled_units"]["c2"] == 0
    _assert_closure(env, stats)


def test_legacy_scripted_consumers_phase1() -> None:
    """Default phase1 world (ScriptedConsumer path, quantity 1 per order) with a random
    orderer, horizon 12: the closure holds and every consumer is listed. Nothing is
    fulfilled this early in phase1 (12 ordered: 10 rejected, 2 still open), so this case
    pins the ordered/rejected/open split; the fixtures above exercise fulfilment."""
    env = TokenomicsAECEnv(horizon=12)
    stats = _run(env, prod="restore_when_low", order="random")
    assert list(stats["consumer_ordered_units"]) == [c.id for c in env.config.consumers]
    assert all(v == stats["cycles"] for v in stats["consumer_ordered_units"].values())
    assert stats["consumer_rejected_units"] == 10
    _assert_closure(env, stats)


def test_reset_clears_per_consumer_units() -> None:
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 1))), horizon=2)
    first = _run(env)
    assert first["consumer_ordered_units"] == {"c1": 2}
    second = _run(env)
    assert second["consumer_ordered_units"] == {"c1": 2}
    assert first["consumer_ordered_units"] is not second["consumer_ordered_units"]
    assert np.isfinite(second["return"])
