"""Per-episode demand-schedule digest (design D5, add-measurement-gate-harness;
gate-harness "Identical demand across observation contracts").

The gate compares `episode_stats["demand_digest"]` across contracts per (world, episode
seed) (task 8.1). These tests pin the producer side: same world + seed -> same digest
under the legacy and joint contracts, seed-sensitive only where demand is randomized,
fixed at reset, and stable across processes (no `hash()`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tokenomics.domain.config import (
    Configuration,
    ConsumerConfig,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
    load_config,
)
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.aec_env import DEFAULT_CONFIG
from tokenomics.env.demand_digest import demand_digest
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.runner import run_episode

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HORIZON = 8
P, Q = 1, 2


def _digest(env: TokenomicsAECEnv, seed: int) -> str:
    env.reset(seed=seed)
    return env.episode_stats["demand_digest"]


def _legacy(config: Configuration) -> TokenomicsAECEnv:
    return TokenomicsAECEnv(config=config, horizon=HORIZON)


def _scheduled_config() -> Configuration:
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
        consumers=[
            ConsumerConfig("c1", product=P, schedule=((P, 1), (Q, 2))),
            # B9 bounds as well: the explicit schedule overrides them (D8), so no RNG.
            ConsumerConfig("c2", product=Q, products=(P, Q), quantity_min=1, quantity_max=5,
                           schedule=((Q, 3), (P, 2))),
        ],
    )
    cfg.validate()
    return cfg


def test_same_seed_same_digest_across_contracts() -> None:
    # Scenario basis of "Digest mismatch": a B9 world (seeded demand) and the scripted
    # phase1 world give the same digest under the legacy and joint contracts.
    for cfg in (load_config(FIXTURES / "random_demand.json"), load_config(DEFAULT_CONFIG)):
        for seed in (0, 3):
            legacy = _digest(_legacy(cfg), seed)
            joint = _digest(joint_stack_env(cfg, HORIZON), seed)
            assert legacy == joint
            assert len(legacy) == 64 and int(legacy, 16) >= 0  # sha256 hex


def test_b9_digest_is_seed_sensitive() -> None:
    cfg = load_config(FIXTURES / "random_demand.json")
    digests = {_digest(_legacy(cfg), seed) for seed in range(4)}
    assert len(digests) == 4


def test_scheduled_world_digest_ignores_seed() -> None:
    # All consumers carry an explicit schedule (task 2.2): demand, and so the digest, is
    # the same in every episode whatever the seed.
    env = _legacy(_scheduled_config())
    assert len({_digest(env, seed) for seed in (0, 1, 9)}) == 1


def test_digest_matches_built_schedules() -> None:
    # The digest is exactly `demand_digest` over each consumer's schedule as configured:
    # sorted consumer ids, and a scripted consumer counts as its (product, 1) per cycle.
    sched = _scheduled_config()
    assert _digest(_legacy(sched), 0) == demand_digest(
        {"c2": [(Q, 3), (P, 2)], "c1": [(P, 1), (Q, 2)]}
    )
    phase1 = load_config(DEFAULT_CONFIG)
    assert _digest(_legacy(phase1), 0) == demand_digest(
        {c.id: [(c.product, 1)] for c in phase1.consumers}
    )


def test_digest_canonical_and_content_sensitive() -> None:
    base = {"a": [(1, 1), (2, 3)], "b": [(2, 1)]}
    assert demand_digest(base) == demand_digest({"b": [(2, 1)], "a": [(1, 1), (2, 3)]})
    assert demand_digest(base) != demand_digest({"a": [(1, 1), (2, 2)], "b": [(2, 1)]})
    assert demand_digest(base) != demand_digest({"a": [(2, 3), (1, 1)], "b": [(2, 1)]})
    assert demand_digest(base) != demand_digest({"a": [(1, 1), (2, 3)], "c": [(2, 1)]})


def test_digest_fixed_at_reset_independent_of_play() -> None:
    # Policy actions and the cycle count do not move the digest: it is the one written
    # at reset, also in the run_episode output, and recomputed identically by a fresh env.
    cfg = load_config(FIXTURES / "random_demand.json")
    env = _legacy(cfg)
    at_reset = _digest(env, 2)
    for prod, order in (("noop", "even_split"), ("always_restore", "random")):
        stats = run_episode(_legacy(cfg), PRODUCING_BASELINES[prod], ORDERING_BASELINES[order], seed=2)
        assert stats["demand_digest"] == at_reset
    assert _digest(_legacy(cfg), 2) == at_reset
    assert _digest(env, 2) == at_reset  # re-reset of the same instance


def test_digest_stable_across_processes() -> None:
    # A fresh interpreter with a different PYTHONHASHSEED computes the same digest, so
    # digests from separate gate worker processes compare equal.
    cfg_path = FIXTURES / "random_demand.json"
    code = (
        "from tokenomics.domain.config import load_config;"
        "from tokenomics.env import TokenomicsAECEnv;"
        f"e = TokenomicsAECEnv(config=load_config({str(cfg_path)!r}), horizon={HORIZON});"
        "e.reset(seed=1); print(e.episode_stats['demand_digest'])"
    )
    env_vars = {**os.environ, "PYTHONHASHSEED": "12345"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env_vars, capture_output=True, text=True,
        check=True, timeout=60,
    )
    assert out.stdout.strip() == _digest(_legacy(load_config(cfg_path)), 1)
