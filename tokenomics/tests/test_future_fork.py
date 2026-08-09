"""EXP-0040 future-demand fork invariants (HANDOVER "Required Validity Tests").

The fork must hold the entire deployable pre-action state fixed (graph, open orders,
current demand, fixed consumers) and move ONLY the unconsumed randomized future suffix,
deterministically from a continuation digest. These tests pin every one of those
invariants before the collector is built.
"""

from __future__ import annotations

import copy

import numpy as np

from tokenomics.domain.config import (
    Configuration,
    ConsumerConfig,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
from tokenomics.domain.consumer import ScheduledConsumer, ScriptedConsumer
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.future_fork import continuation_seed, fork_future_demand

TASK = "exp0040-test-task"
DIGEST = 1234567890123456789


def _contested_env(seed: int = 29, horizon: int = 30) -> TokenomicsAECEnv:
    """A topology where product 1 has TWO eligible producers (p1a, p1b) so ordering
    decisions are contested (>=2 candidates, like the EXP-0039 population), driven by a
    randomized consumer (product in {1,2}, quantity in [1,4]) so a future suffix exists."""
    cfg = Configuration(
        cycle_emission=2000,
        process_sheets=[ProcessSheet(p, {f"capacity-{p}": 2}) for p in (1, 2)],
        producer_configs=[
            ProducingAgentConfig(
                "p1a", {"capacity-1": 100}, 5,
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                produces=frozenset({1})),
            ProducingAgentConfig(
                "p1b", {"capacity-1": 100}, 5,
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                produces=frozenset({1})),
            ProducingAgentConfig(
                "p2", {"capacity-2": 100}, 5,
                Restoration(2, 50, "capacity-2"), Upgrade(2, 40, "capacity-2"),
                produces=frozenset({2})),
        ],
        consumers=[ConsumerConfig("c1", product=1, products=(1, 2), quantity_min=1, quantity_max=4)],
    )
    env = TokenomicsAECEnv(config=cfg, horizon=horizon, product_stream_ordering=True)
    env.reset(seed=seed)
    return env


def _drive_to_contested_decision(env: TokenomicsAECEnv):
    """Step the fixed blend_0.5-style traversal until an ordering agent faces a contested
    decision (>=2 eligible producers) at cycle >= 1 (so some schedule is already consumed).
    Returns (aid, oid, eligible cols) positioned BEFORE that agent acts."""
    from tokenomics.eval.baselines import ordering_blend

    base = ordering_blend(0.5)
    max_iter = env.horizon * len(env.possible_agents) * 3 + 100
    for agent in env.agent_iter(max_iter=max_iter):
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        role, aid = env._role[agent]
        if role == "prod":
            env.step(None if env.action_space(agent) is None else _noop_prod(env, agent))
            continue
        if env.current_cycle >= 1:
            view = env.system.ordering_agents[aid].view(env.system.producer_infos)
            for oid in sorted(view.incoming):
                elig = view.eligible[oid]
                if len(elig) >= 2:
                    cols = [env.spec_.producer_ids.index(p) for p in elig]
                    return aid, oid, list(elig), cols
        env.step(base(env, agent))
    raise AssertionError("no contested ordering decision reached")


def _noop_prod(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
    return np.zeros(env.action_space(agent).shape, dtype=np.float32)


def _encode_at(env: TokenomicsAECEnv, aid: str, oid, elig):
    from tokenomics.domain.graph_view import build_economy_graph_view
    from tokenomics.eval.graph_value_router_dump import encode_graph

    return encode_graph(build_economy_graph_view(env.system, aid, oid, list(elig)))


def _schedules(env: TokenomicsAECEnv) -> dict:
    return {cid: list(c._schedule) for cid, c in env.system.consumers.items()
            if isinstance(c, ScheduledConsumer)}


# --- seed determinism (never Python hash()) -----------------------------------------
def test_continuation_seed_is_deterministic_and_id_sensitive():
    assert continuation_seed(TASK, DIGEST, 0) == continuation_seed(TASK, DIGEST, 0)
    assert continuation_seed(TASK, DIGEST, 0) != continuation_seed(TASK, DIGEST, 1)
    assert continuation_seed(TASK, DIGEST, 0) != continuation_seed("other", DIGEST, 0)
    assert continuation_seed(TASK, DIGEST, 0) != continuation_seed(TASK, DIGEST + 1, 0)


# --- graph / current-state identity -------------------------------------------------
def test_fork_leaves_encoded_graph_byte_identical():
    env = _contested_env()
    aid, oid, elig, _cols = _drive_to_contested_decision(env)
    before = _encode_at(env, aid, oid, elig)
    forked = fork_future_demand(env, task_id=TASK, decision_digest=DIGEST, continuation_id=0)
    after = _encode_at(forked, aid, oid, elig)
    for name in before.__dict__:
        b, a = getattr(before, name), getattr(after, name)
        if isinstance(b, np.ndarray):
            assert np.array_equal(b, a), f"graph array {name} changed under fork"


def test_fork_preserves_open_orders_and_current_demand():
    env = _contested_env()
    _drive_to_contested_decision(env)
    orders_before = copy.deepcopy(env.system.orders)
    forked = fork_future_demand(env, task_id=TASK, decision_digest=DIGEST, continuation_id=3)
    assert list(forked.system.orders.keys()) == list(orders_before.keys())
    for oid, o in orders_before.items():
        fo = forked.system.orders[oid]
        assert (fo.product, fo.quantity, fo.tokens) == (o.product, o.quantity, o.tokens)


# --- suffix surgery: consumed prefix preserved, only tail moves ---------------------
def test_fork_preserves_consumed_prefix_and_replaces_only_tail():
    env = _contested_env()
    _drive_to_contested_decision(env)
    c = env.system.consumers["c1"]
    cyc = c._cycle
    assert cyc >= 1  # some demand already consumed
    src = list(c._schedule)
    forked = fork_future_demand(env, task_id=TASK, decision_digest=DIGEST, continuation_id=5)
    fc = forked.system.consumers["c1"]._schedule
    assert fc[:cyc] == src[:cyc]                      # consumed entries preserved verbatim
    assert fc[cyc:] != src[cyc:]                      # tail resampled (differs)
    assert all(p in {1, 2} and 1 <= q <= 4 for p, q in fc[cyc:])  # legal pool/range


def test_fork_does_not_mutate_source_env():
    env = _contested_env()
    _drive_to_contested_decision(env)
    before = _schedules(env)
    fork_future_demand(env, task_id=TASK, decision_digest=DIGEST, continuation_id=7)
    assert _schedules(env) == before


# --- determinism + continuation independence ----------------------------------------
def test_fork_is_bitwise_deterministic_for_fixed_continuation():
    env = _contested_env()
    _drive_to_contested_decision(env)
    a = fork_future_demand(env, task_id=TASK, decision_digest=DIGEST, continuation_id=2)
    b = fork_future_demand(env, task_id=TASK, decision_digest=DIGEST, continuation_id=2)
    assert _schedules(a) == _schedules(b)


def test_different_continuations_give_different_suffixes():
    env = _contested_env()
    _drive_to_contested_decision(env)
    s0 = _schedules(fork_future_demand(env, task_id=TASK, decision_digest=DIGEST, continuation_id=0))
    s1 = _schedules(fork_future_demand(env, task_id=TASK, decision_digest=DIGEST, continuation_id=1))
    assert s0 != s1


# --- fixed-demand consumers are untouched -------------------------------------------
def test_fork_leaves_fixed_consumers_unchanged():
    env = TokenomicsAECEnv(horizon=10)  # canonical phase1: ScriptedConsumer(s)
    env.reset(seed=1)
    # advance a couple of cycles
    for _ in range(20):
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        space = env.action_space(agent)
        env.step(np.zeros(space.shape, dtype=np.float32))
    fixed_before = {cid: (c.id(), type(c).__name__)
                    for cid, c in env.system.consumers.items()}
    forked = fork_future_demand(env, task_id=TASK, decision_digest=DIGEST, continuation_id=0)
    for cid, c in forked.system.consumers.items():
        assert isinstance(c, ScriptedConsumer)
        assert (c.id(), type(c).__name__) == fixed_before[cid]
