"""EXP-0037 value-routed orderer: the residual router probe.

The residual router keeps fixed `blend_0.5` as the base/floor and only *bends* routing
away from current throughput when a learned per-producer score says a lower-capacity
producer is strategically better (inventory, inbound, TTL, recipe position, scarcity).
This is deliberately NOT the EXP-0012 throughput teacher nor the EXP-0032 noisy
per-cycle search vector (HANDOVER non-goals): the base is blend_0.5, the label is the
marginal value of bumping a producer's logit, and acceptance is deployed rollout vs
blend_0.5 — never table error.

Kept tiny (small scale configs, seconds-long) so it stays local per the compute rule;
the real s29/s77/s45 gate runs on a pod.
"""

from __future__ import annotations

import numpy as np

from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.decode import decode_ordering
from tokenomics.eval.baselines import ordering_blend, ordering_even_split
from tokenomics.eval.scale import build_scale_config
from tokenomics.eval.curriculum import TaskRecord
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.value_router import (
    _one_cycle_bump,
    evaluate_router_gate,
    marginal_value_labels,
    residual_router,
)


def _scale_task(seed: int, depth: int, episode_seed: int = 0) -> TaskRecord:
    """A self-contained gate task from a scale generator-ref (no dependence on experiment
    archives), so the harness test replays a real economy at anchored seeds."""
    return TaskRecord(
        task_id=f"scale-d{depth}-s{seed}", task_family="scale",
        generator_version="test", serializer_version="test",
        world_seed=seed, episode_seed=episode_seed,
        world_spec={"generator": "build_scale_config", "params": {"depth": depth}},
    )


def _first_contested_order(env: TokenomicsAECEnv, agent: str):
    """Return (stream_index k, order_id, eligible producer_ids) for the first order held
    by `agent` that has >= 2 eligible producers — the only orders where routing weight is
    a real choice."""
    _, aid = env._role[agent]
    view = env.system.ordering_agents[aid].view(env.system.producer_infos)
    for k, oid in enumerate(sorted(view.incoming)):
        elig = list(view.eligible[oid])
        if len(elig) >= 2:
            return k, oid, elig, view
    return None


def _routed_quantity(env: TokenomicsAECEnv, agent: str, action, oid: str, pid: str) -> int:
    _, aid = env._role[agent]
    view = env.system.ordering_agents[aid].view(env.system.producer_infos)
    cmd = decode_ordering(env.spec_, action, view)
    return sum(so.quantity for so in cmd.orders[oid] if so.producer_id == pid)


def _env_at_ordering(cfg, seed: int = 0, factory=None) -> tuple[TokenomicsAECEnv, str]:
    """Advance a fresh env to the first cycle where an ordering agent must act, and
    return (env, that agent) so an ordering policy can be called directly and compared.
    `factory(cfg, horizon)` overrides the env builder (e.g. `joint_stack_env`)."""
    env = factory(cfg, 100) if factory else TokenomicsAECEnv(config=cfg, horizon=100)
    env.reset(seed=seed)
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    prod = PRODUCING_BASELINES["stock_aware"]
    for agent in env.agent_iter(max_iter=10000):
        role = env._role[agent][0]
        if role == "order":
            return env, agent
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        env.step(prod(env, agent))
    raise AssertionError("no ordering agent reached")


def test_zero_residual_reproduces_the_base_blend():
    """The whole safety story: a residual router with an all-zero score reduces exactly to
    fixed blend_0.5 (the floor). No score => no override."""
    cfg = build_scale_config(0, depth=2)
    env, agent = _env_at_ordering(cfg)
    base = ordering_blend(0.5)

    def zero_score(e: TokenomicsAECEnv, a: str) -> np.ndarray:
        return np.zeros(e.spec_.ordering_action_dim, dtype=np.float32)

    router = residual_router(base, zero_score, clip=1.0)
    np.testing.assert_array_equal(router(env, agent), base(env, agent))


def test_positive_residual_routes_more_quantity_to_the_targeted_producer():
    """A learned score is only useful if it actually bends routing. Over an even_split base
    (equal shares among eligibles), a positive residual on one producer's (stream, producer)
    cell must route it strictly more quantity than the base does — through the real decode."""
    # depth 3 gives an order with several eligible producers and quantity to split.
    cfg = build_scale_config(0, depth=3)
    env, agent = _env_at_ordering(cfg)
    found = _first_contested_order(env, agent)
    assert found is not None, "need an order with >= 2 eligible producers"
    k, oid, elig, _view = found
    target = elig[0]

    def bump_target(e: TokenomicsAECEnv, a: str) -> np.ndarray:
        sc = np.zeros((e.spec_.k_stream, e.spec_.num_producers), dtype=np.float32)
        sc[k, e.spec_.producer_ids.index(target)] = 10.0  # will be clipped to `clip`
        return sc.flatten()

    base = ordering_even_split
    router = residual_router(base, bump_target, clip=3.0)

    base_q = _routed_quantity(env, agent, base(env, agent), oid, target)
    routed_q = _routed_quantity(env, agent, router(env, agent), oid, target)
    assert routed_q > base_q


def test_bump_perturbs_only_the_labeled_agent():
    """A marginal label must isolate ONE ordering agent's decision. In the joint-stack env there
    are many product-stream ordering agents acting the same cycle; a bump built for agent A must
    leave any other agent B's routing exactly at the base — otherwise the label conflates A's
    decision with same-index nudges to B (confounded marginal value)."""
    cfg = build_scale_config(0, depth=3)
    env, a = _env_at_ordering(cfg, factory=joint_stack_env)
    others = [x for x in env.agents if env._role[x][0] == "order" and x != a]
    assert others, "need a second ordering agent to prove isolation"
    b = others[0]
    base = ordering_blend(0.5)

    bump = _one_cycle_bump(base, a, k=0, col=0, delta=5.0, at_cycle=env.current_cycle)
    assert not np.array_equal(bump(env, a), base(env, a))       # A is nudged
    np.testing.assert_array_equal(bump(env, b), base(env, b))   # B is untouched


def test_zero_delta_bump_yields_zero_marginal_labels():
    """The teacher measures the marginal value of a routing nudge; a zero-magnitude nudge is a
    no-op, so the perturbed rollout equals the base rollout and every label is exactly 0. Pins
    that the label is a true *difference* against the blend_0.5 base (no spurious offset)."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    cfg = build_scale_config(0, depth=3)
    env, agent = _env_at_ordering(cfg)
    found = _first_contested_order(env, agent)
    assert found is not None
    k, _oid, elig, _view = found
    cols = [env.spec_.producer_ids.index(p) for p in elig]

    labels = marginal_value_labels(
        env, PRODUCING_BASELINES["stock_aware"], ordering_blend(0.5),
        agent=agent, k=k, eligible_cols=cols, delta=0.0, lookahead=4,
    )
    np.testing.assert_array_equal(labels, np.zeros(env.spec_.num_producers, dtype=np.float32))


def test_marginal_labels_are_deterministic():
    """Domain is deterministic and the teacher clones the env per candidate (never mutates it),
    so labels are reproducible — the offline training set must be stable across collection runs."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    cfg = build_scale_config(0, depth=3)
    env, agent = _env_at_ordering(cfg)
    k, _oid, elig, _view = _first_contested_order(env, agent)
    cols = [env.spec_.producer_ids.index(p) for p in elig]
    prod, base = PRODUCING_BASELINES["stock_aware"], ordering_blend(0.5)

    a = marginal_value_labels(env, prod, base, agent=agent, k=k, eligible_cols=cols, delta=1.0, lookahead=4)
    b = marginal_value_labels(env, prod, base, agent=agent, k=k, eligible_cols=cols, delta=1.0, lookahead=4)
    np.testing.assert_array_equal(a, b)


def test_gate_scores_zero_residual_exactly_at_the_blend_floor():
    """The gate must be honest: a residual router with a zero score deploys as fixed blend_0.5,
    so per task the learned rollout equals the floor rollout (delta 0, does NOT beat the floor,
    same death rate). This pins the harness's seed-anchoring + rollout comparison end to end."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    tasks = [_scale_task(0, depth=2), _scale_task(1, depth=2)]

    def zero_score(e: TokenomicsAECEnv, a: str) -> np.ndarray:
        return np.zeros(e.spec_.ordering_action_dim, dtype=np.float32)

    rows = evaluate_router_gate(
        tasks, PRODUCING_BASELINES["stock_aware"], zero_score,
        clip=1.0, eval_seeds=2, horizon=40,
    )
    assert [r["task_id"] for r in rows] == ["scale-d2-s0", "scale-d2-s1"]
    for r in rows:
        assert r["delta_vs_blend"] == 0.0
        assert r["beats_floor"] is False
        assert r["learned_cu"] == r["blend_cu"]
        assert r["learned_dead"] == r["blend_dead"]
