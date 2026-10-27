"""The optimized orderer observation equals the pre-optimization one (task 8.4;
docs/DIAGNOSTICS.md "120-producer joint-stack observation cost").

Two caches were added:
- `_EmaSeries.sum_by_producer` reads a per-producer index rebuilt after each fold, instead of
  scanning every EMA key;
- the stream views of one ordering phase share their producer rows (`StreamViewCache`).

The reference oracle is the pre-optimization code. `_scan_sum` is the old scan, verbatim.
`_reference_view` is `build_product_stream_view` without the phase cache, with every telemetry
sum taken by `_scan_sum`. Observations are compared as full encoded arrays with
`np.array_equal`. Whole episodes are compared on actions and every episode statistic.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from tokenomics.domain import orderer_view, telemetry
from tokenomics.domain.config import Configuration, load_config
from tokenomics.domain.system import System, SystemState
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.encode import encode_ordering_stream
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from tokenomics.eval.families import build_family_config
from tokenomics.eval.fixtures import build_forkjoin_config
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.runner import run_episode
from tokenomics.eval.scale import build_scale_config

from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SERIES = ("_routed", "_accepted", "_completed", "_rejected", "_failed")


def _scan_sum(series: telemetry._EmaSeries, producer_id: object) -> float:
    """The pre-8.4 `sum_by_producer`, verbatim: a scan of every EMA key."""
    return sum(v for (pid, _x), v in series._ema.items() if pid == producer_id)


@contextlib.contextmanager
def _reference_mode() -> Iterator[None]:
    """Run the pre-8.4 code path: scanned telemetry sums and no phase cache."""
    orig_sum = telemetry._EmaSeries.sum_by_producer
    orig_view = System.product_stream_view

    def view(self: System, aid: object) -> orderer_view.ProductStreamView:
        agent = self.ordering_agents[aid]
        incoming = {oid: self.orders[oid] for oid in agent.incoming}
        return orderer_view.build_product_stream_view(
            self.stream_products[aid], incoming, self.producing_agents,
            self.telemetry, self.process_sheets, self.orders, None,
        )

    telemetry._EmaSeries.sum_by_producer = _scan_sum  # type: ignore[method-assign]
    System.product_stream_view = view  # type: ignore[method-assign]
    try:
        yield
    finally:
        telemetry._EmaSeries.sum_by_producer = orig_sum  # type: ignore[method-assign]
        System.product_stream_view = orig_view  # type: ignore[method-assign]


def _reference_obs(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
    _, did = env._role[agent]
    cp = min(1.0, env.current_cycle / env.horizon)
    with _reference_mode():
        view = env.system.product_stream_view(did)
    return encode_ordering_stream(env.spec_, view, cp)


def _family(family: str, producers: int, regime: str = "calm") -> Configuration:
    return build_family_config(1000, family=family, producers=producers, regime=regime)


# --- the telemetry index equals the scan, bit for bit ---


def test_sum_by_producer_index_equals_scan_every_cycle() -> None:
    """After every cycle of a 120-producer episode, every producer's total in every series
    equals the scan: same value, same type (int 0 for a producer without keys)."""
    env = joint_stack_env(_family("sparse_wide", 120), 6)
    tel = env.system.telemetry if env.system else None
    env.reset(seed=1000)
    checked = 0
    pol_p, pol_o = PRODUCING_BASELINES["entity_planner"], ORDERING_BASELINES["proportional"]
    cycle = -1
    for agent in env.agent_iter(max_iter=10_000):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        if env.current_cycle != cycle:
            cycle = env.current_cycle
            tel = env.system.telemetry
            for name in SERIES:
                series = getattr(tel, name)
                for pid in list(env.system.producing_agents) + ["no-such-producer"]:
                    got, want = series.sum_by_producer(pid), _scan_sum(series, pid)
                    assert got == want and type(got) is type(want), (name, pid, got, want)
                    checked += 1
        role = env._role[agent][0]
        env.step(pol_p(env, agent) if role == "prod" else pol_o(env, agent))
    assert cycle >= 5 and checked > 0


def test_index_invalidated_by_fold() -> None:
    s = telemetry._EmaSeries(0.3)
    s.add(("p", 1), 2.0)
    s.fold()
    assert s.sum_by_producer("p") == _scan_sum(s, "p") == 0.6
    s.add(("p", 2), 1.0)
    s.add(("q", 1), 5.0)
    assert s.sum_by_producer("p") == 0.6  # accumulators do not touch the EMA until fold
    s.fold()
    assert s.sum_by_producer("p") == _scan_sum(s, "p")
    assert s.sum_by_producer("q") == _scan_sum(s, "q") == 1.5
    assert s.sum_by_producer("none") == 0 and type(s.sum_by_producer("none")) is int


@pytest.mark.parametrize("values", [
    [1e16, 1.0, 1.0],          # running + loses both 1.0s; compensated sum keeps them
    [0.1] * 10,                # ordinary magnitudes: running + gives 0.9999999999999999
    [0.3, 0.6, 0.1, 1e-17, 0.7, 0.2],
])
def test_index_matches_builtin_sum_semantics(values: list[float]) -> None:
    """The index must reproduce builtin `sum` exactly, including Python >= 3.12's
    compensated float summation. Values interleave with another producer's so grouping,
    not just order, is exercised."""
    s = telemetry._EmaSeries(0.3)
    s._ema = {}
    for i, v in enumerate(values):
        s._ema[("p", i)] = v
        s._ema[("q", i)] = v / 3
    s._by_producer = None
    for pid in ("p", "q"):
        got, want = s.sum_by_producer(pid), _scan_sum(s, pid)
        assert got.hex() == want.hex(), (pid, got, want)
    t = 0.0
    for v in values:
        t += v
    assert s.sum_by_producer("p") != t or sum(values) == t  # the case is adversarial
    # through the public path: add + fold
    r = telemetry._EmaSeries(1.0)
    for i, v in enumerate(values):
        r.add(("p", i), v)
    r.fold()
    assert r.sum_by_producer("p").hex() == _scan_sum(r, "p").hex()


# --- full observation arrays, every orderer observation, >= 10 cycles at 120 producers ---


def _check_observations(config: Configuration, horizon: int,
                        stride: int = 1) -> tuple[int, int, set[str], set[str]]:
    """Run an episode; at every stride-th orderer observation compare the optimized encoded
    obs with the reference. Returns (compared observations, cycles played, orderer agents
    compared at least once, all orderer agents)."""
    env = joint_stack_env(config, horizon)
    pol_p, pol_o = PRODUCING_BASELINES["entity_planner"], ORDERING_BASELINES["proportional"]
    env.reset(seed=1000)
    compared = seen = 0
    covered: set[str] = set()
    orderers = {a for a in env.possible_agents if env._role[a][0] == "order"}
    for agent in env.agent_iter(max_iter=1_000_000):
        obs, _, term, trunc, _ = env.last()
        if env._role[agent][0] == "order":
            if seen % stride == 0:
                covered.add(agent)
                ref = _reference_obs(env, agent)
                assert obs.dtype == ref.dtype and np.array_equal(obs, ref), (
                    agent, env.current_cycle)
                compared += 1
            seen += 1
        if term or trunc:
            env.step(None)
            continue
        role = env._role[agent][0]
        env.step(pol_p(env, agent) if role == "prod" else pol_o(env, agent))
    return compared, env.current_cycle, covered, orderers


@pytest.mark.parametrize("family", ["dense", "sparse_wide"])
def test_observations_equal_reference_120_producers(family: str) -> None:
    """10 cycles on a 120-producer family world. The reference is costly (it is the slow
    path), so every 11th orderer observation is compared. 11 is coprime with both orderer
    counts (dense 504, sparse 300), and the episode makes 11 passes over the agents (10
    cycles plus the terminal pass), so the compared agents rotate through every residue.
    The test asserts every orderer was compared at least once. Includes the cached
    ORDERING-phase observations and the terminal uncached ones."""
    compared, cycles, covered, orderers = _check_observations(_family(family, 120), 10, stride=11)
    assert cycles == 10 and compared > 100
    assert covered == orderers


@pytest.mark.parametrize("config_fn", [
    lambda: _family("dense", 12),
    lambda: _family("sparse_wide", 12, "shocked"),
    lambda: build_scale_config(0),
    lambda: build_forkjoin_config(0),
    lambda: load_config(FIXTURES / "competition.json"),
], ids=["dense12", "sparse12-shocked", "scale0", "forkjoin0", "competition"])
def test_observations_equal_reference_small_worlds(config_fn: Callable[[], Configuration]) -> None:
    """Every orderer observation, 12 cycles, on 12-producer family worlds and legacy fixture
    worlds run under the joint-stack contract."""
    compared, cycles, covered, orderers = _check_observations(config_fn(), 12)
    assert compared > 0 and cycles >= 1 and covered == orderers


# --- whole episodes: actions and every episode statistic ---


def _recording(policy: Policy, log: list[np.ndarray]) -> Policy:
    def wrapped(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        a = np.asarray(policy(env, agent))
        log.append(a.copy())
        return a
    return wrapped


def _episode(config: Configuration, prod: str, order: str, *, reference: bool,
             horizon: int = 12) -> tuple[dict, list[np.ndarray]]:
    log: list[np.ndarray] = []
    ctx = _reference_mode() if reference else contextlib.nullcontext()
    with ctx:
        stats = run_episode(joint_stack_env(config, horizon),
                            _recording(PRODUCING_BASELINES[prod], log),
                            _recording(ORDERING_BASELINES[order], log), seed=1000)
    return stats, log


@pytest.mark.parametrize("config_fn", [
    lambda: _family("dense", 12),
    lambda: _family("sparse_wide", 12, "shocked"),
    lambda: build_scale_config(0),
    lambda: build_forkjoin_config(0),
], ids=["dense12", "sparse12-shocked", "scale0", "forkjoin0"])
@pytest.mark.parametrize("prod,order", [("entity_planner", "proportional"),
                                        ("random", "random")])
def test_episode_actions_and_stats_equal_reference(
    config_fn: Callable[[], Configuration], prod: str, order: str,
) -> None:
    config = config_fn()
    opt_stats, opt_actions = _episode(config, prod, order, reference=False)
    ref_stats, ref_actions = _episode(config, prod, order, reference=True)
    assert opt_stats == ref_stats
    assert len(opt_actions) == len(ref_actions)
    assert all(a.dtype == b.dtype and np.array_equal(a, b)
               for a, b in zip(opt_actions, ref_actions, strict=True))


def test_portable_orderer_actions_equal_reference() -> None:
    """The EXP-0012 scorer policy reads `system.product_stream_view` directly, so it gets the
    phase cache too. A scorer with fixed random weights must act identically with and without
    it (no checkpoint needed)."""
    torch = pytest.importorskip("torch")
    from tokenomics.train.orderer_scorer import OrdererScoreModel, scorer_ordering_policy

    torch.manual_seed(0)
    model = OrdererScoreModel()
    config = _family("sparse_wide", 12)

    def run(reference: bool) -> tuple[dict, list[np.ndarray]]:
        log: list[np.ndarray] = []
        ctx = _reference_mode() if reference else contextlib.nullcontext()
        with ctx:
            stats = run_episode(joint_stack_env(config, 8),
                                _recording(PRODUCING_BASELINES["entity_planner"], log),
                                _recording(scorer_ordering_policy(model), log), seed=1000)
        return stats, log

    (s1, a1), (s2, a2) = run(False), run(True)
    assert s1 == s2 and all(np.array_equal(x, y) for x, y in zip(a1, a2, strict=True))


# --- the cache lives only while producer rows cannot change ---


def test_cache_dropped_at_every_producer_mutation() -> None:
    """A memo exists from start_cycle until the first producer acts, and for the whole
    ordering phase; it is never present right after a producer action or production."""
    env = joint_stack_env(_family("sparse_wide", 12), 3)
    env.reset(seed=1000)
    sysm = env.system
    assert sysm._stream_view_cache is not None  # start_cycle ran in reset
    pol_p = PRODUCING_BASELINES["entity_planner"]
    pol_o = ORDERING_BASELINES["proportional"]
    after_prod_action = set()
    ordering_cached = set()
    for agent in env.agent_iter(max_iter=10_000):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        role = env._role[agent][0]
        if role == "order":
            ordering_cached.add(sysm._stream_view_cache is not None)
        env.step(pol_p(env, agent) if role == "prod" else pol_o(env, agent))
        if role == "prod" and sysm.state == SystemState.ORDERS_PLACEMENT:
            after_prod_action.add(sysm._stream_view_cache is None)
    assert after_prod_action == {True} and ordering_cached == {True}


def test_shared_rows_are_frozen() -> None:
    env = joint_stack_env(_family("sparse_wide", 12), 2)
    env.reset(seed=1000)
    for agent in env.agent_iter(max_iter=10_000):
        if env._role[agent][0] == "order":
            break
        env.step(PRODUCING_BASELINES["entity_planner"](env, agent))
    view = env.system.product_stream_view(env._role[agent][1])
    row = next(iter(view.producers.values()))
    with pytest.raises(Exception):
        row.total_load = 99  # type: ignore[misc]
