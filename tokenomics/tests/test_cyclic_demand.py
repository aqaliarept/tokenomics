"""Deterministic cyclic demand: `ConsumerConfig.schedule` (design D8,
add-measurement-gate-harness; stochastic-environment "Deterministic cyclic demand").

A consumer with an explicit (product, quantity) schedule replays it one entry per
cycle, identically in every episode whatever the episode seed. Consumers without it
keep their legacy path (tests/test_legacy_invariance.py pins the numbers).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tokenomics.domain import build_system
from tokenomics.domain.config import (
    Configuration,
    ConsumerConfig,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
    load_config,
)
from tokenomics.domain.consumer import ScheduledConsumer, ScriptedConsumer
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.future_fork import fork_future_demand
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.scale import build_scale_config, config_to_dict, permute_scale_config

P, Q = 1, 2

# (consumer id, product, quantity, placement cycle) of one placed consumer order.
Placed = tuple[str, int, int, int]


def _config(*consumers: ConsumerConfig) -> Configuration:
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(P, {"capacity-1": 2}), ProcessSheet(Q, {"capacity-1": 3})],
        producer_configs=[
            ProducingAgentConfig(
                "p1", {"capacity-1": 100}, 5,
                Restoration(P, 50, "capacity-1"), Upgrade(P, 40, "capacity-1"),
                produces=frozenset({P, Q}),
            ),
        ],
        consumers=list(consumers),
    )
    cfg.validate()
    return cfg


def _placed_demand(cfg: Configuration, horizon: int, seed: int) -> list[Placed]:
    """Every consumer order placed in a played cycle, sorted by (cycle, consumer).
    Scans `system.orders` after every env step, so no order resolves unseen."""
    env = TokenomicsAECEnv(config=cfg, horizon=horizon)
    env.reset(seed=seed)
    for a in env.possible_agents:
        env.action_space(a).seed(seed)
    prod = PRODUCING_BASELINES["always_restore"]
    order = ORDERING_BASELINES["proportional"]
    seen: dict[int, Placed] = {}

    def scan() -> None:
        for oid, o in env.system.orders.items():
            if o.consumer_request is not None:
                assert o.placement_cycle is not None, f"consumer order {oid} untagged"
                seen[oid] = (o.consumer_request.consumer_id, o.product, o.quantity, o.placement_cycle)

    scan()
    for agent in env.agent_iter(max_iter=horizon * len(env.possible_agents) * 3 + 100):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
        else:
            policy = prod if env._role[agent][0] == "prod" else order
            env.step(policy(env, agent))
        scan()
    assert env.episode_stats["cycles"] == horizon
    played = [p for p in seen.values() if p[3] <= horizon]
    return sorted(played, key=lambda p: (p[3], p[0]))


def test_schedule_repeats() -> None:
    # Scenario "Schedule repeats": [(P,1),(Q,2)] over 5 cycles -> P1, Q2, P1, Q2, P1.
    cfg = _config(ConsumerConfig("c1", product=P, schedule=((P, 1), (Q, 2))))
    demand = _placed_demand(cfg, horizon=5, seed=0)
    assert [(prod, qty) for _, prod, qty, _ in demand] == [(P, 1), (Q, 2), (P, 1), (Q, 2), (P, 1)]
    assert [cycle for *_, cycle in demand] == [1, 2, 3, 4, 5]


def test_episode_seed_has_no_effect() -> None:
    # Scenario "Episode seed has no effect": all consumers scheduled; seeds 0 and 9 give
    # identical demand. c2 also carries B9 bounds, which the schedule overrides (D8).
    cfg = _config(
        ConsumerConfig("c1", product=P, schedule=((P, 1), (Q, 2))),
        ConsumerConfig("c2", product=Q, products=(P, Q), quantity_min=1, quantity_max=5,
                       schedule=((Q, 3), (Q, 1), (P, 2))),
    )
    d0 = _placed_demand(cfg, horizon=8, seed=0)
    d9 = _placed_demand(cfg, horizon=8, seed=9)
    assert d0 == d9
    c2 = [(prod, qty) for cid, prod, qty, _ in d0 if cid == "c2"]
    assert c2 == [(Q, 3), (Q, 1), (P, 2)] * 2 + [(Q, 3), (Q, 1)]


def test_b9_control_is_seed_sensitive() -> None:
    # Control for the test above: the same B9 bounds WITHOUT a schedule do vary by seed,
    # so the equality there is the schedule's doing, not an insensitive harness.
    cfg = _config(ConsumerConfig("c2", product=Q, products=(P, Q), quantity_min=1, quantity_max=5))
    assert _placed_demand(cfg, horizon=8, seed=0) != _placed_demand(cfg, horizon=8, seed=9)


def test_legacy_consumer_path_untouched() -> None:
    # Scenario "Legacy fixed demand untouched" (numbers: tests/test_legacy_invariance.py):
    # no schedule -> the canonical ScriptedConsumer, and no "schedule" key in the dict.
    cfg = _config(ConsumerConfig("c1", product=P))
    env = TokenomicsAECEnv(config=cfg, horizon=5)
    env.reset(seed=0)
    assert isinstance(env.system.consumers["c1"], ScriptedConsumer)
    assert "schedule" not in config_to_dict(cfg)["consumers"][0]


def test_scheduled_consumer_built_from_config() -> None:
    cfg = _config(ConsumerConfig("c1", product=P, schedule=((Q, 2), (P, 1))))
    c = cfg.consumers[0]
    assert c.product_pool == (Q, P)
    assert not c.is_randomized
    env = TokenomicsAECEnv(config=cfg, horizon=5)
    env.reset(seed=3)
    built = env.system.consumers["c1"]
    assert isinstance(built, ScheduledConsumer)
    assert built._schedule == [(Q, 2), (P, 1)]
    # The domain-default builder honours the schedule too (pure data, no RNG).
    assert build_system(cfg).consumers["c1"]._schedule == [(Q, 2), (P, 1)]


def test_future_fork_leaves_schedule_alone() -> None:
    # A schedule is fixed demand: the counterfactual fork must not resample it.
    cfg = _config(ConsumerConfig("c1", product=P, schedule=((P, 1), (Q, 2))))
    env = TokenomicsAECEnv(config=cfg, horizon=6)
    env.reset(seed=0)
    clone = fork_future_demand(env, task_id="t", decision_digest=1, continuation_id=2)
    assert clone.system.consumers["c1"]._schedule == [(P, 1), (Q, 2)]


@pytest.mark.parametrize(
    "schedule, match",
    [
        (((P, 1), (99, 1)), "demands product 99 with no process sheet"),
        (((P, 1), (Q, 0)), "schedule quantity must be >= 1, got 0"),
        (((P, -2),), "schedule quantity must be >= 1, got -2"),
    ],
)
def test_schedule_validation(schedule: tuple[tuple[int, int], ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _config(ConsumerConfig("c1", product=P, schedule=schedule))


def test_load_config_and_round_trip(tmp_path: Path) -> None:
    cfg = _config(
        ConsumerConfig("c1", product=P, schedule=((P, 1), (Q, 2))),
        ConsumerConfig("c2", product=Q),
    )
    d = config_to_dict(cfg)
    assert d["consumers"][0]["schedule"] == [[P, 1], [Q, 2]]
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(d))
    loaded = load_config(path)
    assert loaded.consumers[0].schedule == ((P, 1), (Q, 2))
    assert loaded.consumers[1].schedule == ()
    assert loaded == cfg
    assert config_to_dict(loaded) == d


def test_load_config_rejects_bad_schedule(tmp_path: Path) -> None:
    d = config_to_dict(_config(ConsumerConfig("c1", product=P)))
    d["consumers"][0]["schedule"] = [[P, 1], [Q, 0]]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="schedule quantity must be >= 1"):
        load_config(path)


def test_schedule_is_part_of_config_identity() -> None:
    base = ConsumerConfig("c1", product=P)
    assert replace(base, schedule=((P, 1),)) != base


def test_permute_relabels_schedule() -> None:
    # Slot relabeling (ADR-0049) must move the schedule with the products it names, and
    # leave the B9 pool mapped from `products`, not collapsed to the schedule's pool.
    base = build_scale_config(0)
    c0 = replace(base.consumers[0], schedule=((20, 1), (21, 2)))
    cfg = replace(base, consumers=[c0, *base.consumers[1:]])
    out = permute_scale_config(cfg, product_perm=[2, 0, 3, 1], producer_perm=[0, 1, 2, 3])
    assert out.consumers[0].schedule == ((22, 1), (20, 2))
    assert out.consumers[0].products == (22, 20, 23, 21)
    # Unscheduled consumers relabel exactly as before the schedule field existed.
    legacy = permute_scale_config(base, product_perm=[2, 0, 3, 1], producer_perm=[0, 1, 2, 3])
    assert out.consumers[1:] == legacy.consumers[1:]
    assert all(c.schedule == () for c in legacy.consumers)
