"""Per-order residual orderer (task 4.3 of add-m4-orderer-redesign, design D5 / D9, ADR-0058 /
ADR-0059).

Episode fixtures play the pinned 12-producer Atlas world ws-atlas-v1-12-0060 at its first
confirm seed, replicate 0, `shocks-i-smoke`, horizon 12, `entity_planner` (test_agent_history's
job). The normaliser is fitted on the rows of one A0 episode of that job (a fixture fit; task
5.1 fits the real one).
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from tokenomics.env import aec_env
from tokenomics.env.agent_history import env_agent_id
from tokenomics.env.spaces import CHOICE_V1, LOGITS_V1, decode_pins
from tokenomics.train import per_order_scorer as pos
from tokenomics.train.per_order_scorer import (AGENT, ARMS, B, HAS_COMPLETED_COL, LATENCY_COL,
                                               N_PHI, N_PSI, SHARED_ONLY, BaseScorer, Decision,
                                               FeatureNormaliser, NonFiniteChoice,
                                               PerOrderPolicy, RawRows)

from test_agent_history import ATLAS_V2, HORIZON, ORDERER, _job, _lfs_present

EXP0012_SHA = "d25388bfa33ea3fc82aebbac48c75c886ed5c091f6016f1ce36eeb4293744a8c"
HIST = slice(31, 41)  # 9 history features + has_completed
OWN = slice(41, 43)

Cmds = list[str]
Log = list[tuple[int, str, int, list[Decision]]]  # (cycle, agent, product, decisions)


# --- arms, normaliser (no episode) ---


def test_arm_sizes_sigma_and_init() -> None:
    """phi = 31 + 9 + has-completed + 2 = 43, psi 13. A1 = 43 + 1 = 44, A2 = 44 + 4 (43 + 13)
    = 268 (R = 2: 156); sigma 0.05 per parameter; A2's init has u ~ N(0, 0.1) from the fixed
    seed and v = 0; A0 has no parameters."""
    assert (N_PHI, N_PSI) == (43, 13)
    assert pos.PHI_FEATURES[HIST] == pos.HISTORY_FEATURES + ("has_completed",)
    assert pos.PHI_FEATURES[OWN] == ("assigned_units", "assigned_orders")
    assert {k: a.n_params for k, a in ARMS.items()} == {"A0": 0, "A1": 44, "A2": 268,
                                                        "A2-R2": 156}
    for a in ARMS.values():
        assert a.sigma().shape == (a.n_params,) and np.all(a.sigma() == 0.05)
    t = ARMS["A2"].initial_theta()
    w, tt, u, v = ARMS["A2"].split(t)
    assert not w.any() and tt == 0.0 and not v.any() and u.std() == pytest.approx(0.1, rel=0.2)
    assert np.array_equal(t, ARMS["A2"].initial_theta())  # fixed seed
    assert np.array_equal(ARMS["A1"].split(t[:44])[0], w)  # A1 is A2's prefix
    with pytest.raises(ValueError, match="shape"):
        ARMS["A1"].split(np.zeros(45))


def test_normaliser_zero_spread_clip_and_hash() -> None:
    """Scale = IQR; IQR 0 -> std; std 0 -> 1 (the zero-spread rule). Values clip at +-5. The
    hash is stable, round-trips and moves with any value."""
    rng = np.random.default_rng(0)
    phi = rng.normal(3.0, 2.0, (200, N_PHI))
    phi[:, 0] = 7.0  # constant: IQR 0, std 0 -> 1
    phi[:, 1] = 0.0
    phi[:5, 1] = 1.0  # 2.5 % ones: IQR 0, std > 0
    r = RawRows()
    for i in range(0, 200, 20):
        r.add(phi[i:i + 20], rng.normal(0, 1, N_PSI))
    nm = FeatureNormaliser.fit(r)
    assert nm.n_rows == 10
    assert (nm.phi_median[0], nm.phi_scale[0]) == (7.0, 1.0)
    assert nm.phi_scale[1] == pytest.approx(np.std(phi[:, 1]))
    q25, q50, q75 = np.percentile(phi[:, 2], [25, 50, 75])
    assert (nm.phi_median[2], nm.phi_scale[2]) == (q50, q75 - q25)
    z = nm.apply_phi(phi)
    assert np.array_equal(z[:, 2], np.clip((phi[:, 2] - q50) / (q75 - q25), -5, 5))
    assert np.all(z[:, 0] == 0.0) and np.all(np.isfinite(z))
    assert np.all(np.abs(nm.apply_phi(phi * 1e6)) <= 5.0)
    assert np.all(np.abs(nm.apply_psi(np.full(N_PSI, 1e9))) <= 5.0)
    assert nm.sha256 == FeatureNormaliser.from_dict(json.loads(json.dumps(nm.to_dict()))).sha256
    assert nm == FeatureNormaliser.from_dict(nm.to_dict())
    moved = dataclasses.replace(nm, phi_scale=(2.0,) + nm.phi_scale[1:])
    assert moved.sha256 != nm.sha256
    with pytest.raises(ValueError, match="scales > 0"):
        FeatureNormaliser((0.0,) * N_PHI, (0.0,) * N_PHI, (0.0,) * N_PSI, (1.0,) * N_PSI)
    with pytest.raises(ValueError, match="at least one"):
        FeatureNormaliser.fit(RawRows())
    with pytest.raises(ValueError, match="feature names"):
        FeatureNormaliser.from_dict({**nm.to_dict(), "phi_features": ["x"] * N_PHI})


# --- episode fixtures ---


@pytest.fixture(scope="module")
def job() -> dict:
    pytest.importorskip("torch")
    if not (_lfs_present(ATLAS_V2) and _lfs_present(ORDERER)):
        pytest.skip("EXP-0052 atlas / EXP-0012 checkpoint absent")
    import torch

    torch.set_num_threads(1)
    return _job()


@pytest.fixture(scope="module")
def base(job: dict) -> BaseScorer:
    return BaseScorer.load(ORDERER)


@contextmanager
def _decode_spy(cmds: Cmds, k_stream: list[int]) -> Iterator[None]:
    """Record every ordering decode; assert every slotted order has exactly one sub-order with
    the whole quantity and tokens (overflow orders beyond k_stream are the legacy split's)."""
    real = aec_env.decode_ordering

    def spy(spec, action, view):  # noqa: ANN001, ANN202 - test wrapper
        cmd = real(spec, action, view)
        if spec.ordering_decode == CHOICE_V1:
            for oid in sorted(view.incoming)[:spec.k_stream]:
                subs = cmd.orders[oid]
                info = view.incoming[oid]
                assert len(subs) == 1, (oid, subs)
                assert (subs[0].quantity, subs[0].tokens) == (info.quantity, info.tokens)
            k_stream[0] = max(k_stream[0], len(view.incoming) - spec.k_stream)
        cmds.append(repr(sorted((oid, [(s.producer_id, s.quantity, s.tokens) for s in subs])
                                for oid, subs in cmd.orders.items())))
        return cmd

    aec_env.decode_ordering = spy
    try:
        yield
    finally:
        aec_env.decode_ordering = real


@dataclasses.dataclass
class Run:
    stats: dict
    cmds: Cmds
    log: Log
    overflow: int
    draws: list[tuple[bytes, float]]  # every keyed draw: (weights bytes, u)


@contextmanager
def _draw_spy(draws: list[tuple[bytes, float]]) -> Iterator[None]:
    """Record the weights and uniform of every inverse-CDF draw (TS's and the policy's)."""
    from tokenomics.eval import routing_diagnostics as rd

    real = rd.inverse_cdf

    def spy(weights: np.ndarray, u: float) -> int:
        draws.append((np.asarray(weights, dtype=np.float64).tobytes(), u))
        return real(weights, u)

    rd.inverse_cdf = pos.inverse_cdf = spy
    try:
        yield
    finally:
        rd.inverse_cdf = pos.inverse_cdf = real


def _logged(policy: PerOrderPolicy, log: Log) -> Callable:
    def call(env: object, agent: str) -> np.ndarray:
        policy.record = []
        action = policy(env, agent)  # type: ignore[arg-type]
        aid = env_agent_id(env, agent)
        log.append((env.system.cycle_counter, agent,  # type: ignore[attr-defined]
                    env.system.stream_products[aid], policy.record))  # type: ignore[attr-defined]
        policy.record = None
        return action

    return call


def _play(job: dict, order: Callable | None = None, policy: PerOrderPolicy | None = None,
          decode: str = CHOICE_V1, wrap: Callable[[Callable], Callable] | None = None) -> Run:
    from tokenomics.eval import decoder_check as dc
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.routing_audit import job_config
    from tokenomics.eval.routing_diagnostics import set_draw_key
    from tokenomics.eval.runner import run_episode

    cfg = job_config(job)
    env = pos.per_order_env(cfg, HORIZON) if decode == CHOICE_V1 else joint_stack_env(cfg, HORIZON)
    set_draw_key(env, dc.job_draw_key(job))
    log: Log = []
    if policy is not None:
        order = _logged(policy, log)
    assert order is not None
    if wrap is not None:
        order = wrap(order)
    cmds: Cmds = []
    over = [0]
    draws: list[tuple[bytes, float]] = []
    with _decode_spy(cmds, over), _draw_spy(draws):
        run_episode(env, PRODUCING_BASELINES[job["producing"]], order, seed=job["episode"],
                    action_seed=job["action_seed"])
    return Run(dict(env.episode_stats), cmds, log, over[0], draws)  # type: ignore[attr-defined]


@pytest.fixture(scope="module")
def fit_rows(job: dict, base: BaseScorer) -> RawRows:
    p = PerOrderPolicy(base, "A0")
    p.sink = RawRows()
    _play(job, policy=p)
    return p.sink


@pytest.fixture(scope="module")
def normaliser(fit_rows: RawRows) -> FeatureNormaliser:
    return FeatureNormaliser.fit(fit_rows)


def _theta(arm: str, seed: int, scale: float = 1.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, ARMS[arm].n_params)


@pytest.fixture(scope="module")
def a0(job: dict, base: BaseScorer, normaliser: FeatureNormaliser) -> Run:
    return _play(job, policy=PerOrderPolicy(base, "A0", normaliser=normaliser))


@pytest.fixture(scope="module")
def a2(job: dict, base: BaseScorer, normaliser: FeatureNormaliser) -> Run:
    return _play(job, policy=PerOrderPolicy(base, "A2", _theta("A2", 1), normaliser))


# --- D5 scenarios ---


def test_zero_residual_equals_the_decoder_check(job: dict, a0: Run) -> None:
    """A0 under choice-v1 gives the same sub-orders and episode statistics as TS (the decoder
    check's keyed sampled-whole decode over the EXP-0012 scorer) under logits-v1, drawing from
    byte-identical weights and uniforms."""
    from test_agent_history import _policies

    _, ts = _policies(job)
    ref = _play(job, order=ts, decode=LOGITS_V1)
    assert len(a0.cmds) == len(ref.cmds) > 50
    assert a0.cmds == ref.cmds
    assert a0.stats == ref.stats
    assert len(a0.draws) > 100 and a0.draws == ref.draws
    assert a0.overflow == 0


@pytest.mark.parametrize("arm,theta", [
    ("A1", lambda: np.zeros(44)), ("A2", lambda: np.zeros(268)),
    ("A2", lambda: ARMS["A2"].initial_theta()), ("A2-R2", lambda: ARMS["A2-R2"].initial_theta())])
def test_zero_residual_arms_equal_a0(job: dict, base: BaseScorer, normaliser: FeatureNormaliser,
                                     a0: Run, arm: str, theta: Callable) -> None:
    run = _play(job, policy=PerOrderPolicy(base, arm, theta(), normaliser))
    assert run.cmds == a0.cmds and run.stats == a0.stats and run.draws == a0.draws
    assert all(np.all(d.delta == 0.0) for *_, ds in run.log for d in ds)


def test_a2_with_v_zero_equals_a1(job: dict, base: BaseScorer,
                                  normaliser: FeatureNormaliser) -> None:
    t1 = _theta("A1", 3, 0.5)
    t2 = np.concatenate([t1, _theta("A2", 4)[44:44 + 4 * 43], np.zeros(4 * 13)])
    r1 = _play(job, policy=PerOrderPolicy(base, "A1", t1, normaliser))
    r2 = _play(job, policy=PerOrderPolicy(base, "A2", t2, normaliser))
    assert r1.cmds == r2.cmds and r1.stats == r2.stats and r1.draws == r2.draws
    d1 = [d.delta.tobytes() for *_, ds in r1.log for d in ds]
    assert d1 == [d.delta.tobytes() for *_, ds in r2.log for d in ds]
    assert any(np.any(d.delta != 0) for *_, ds in r1.log for d in ds)


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_no_split_emitted_and_delta_bounded(job: dict, base: BaseScorer,
                                            normaliser: FeatureNormaliser, scale: float) -> None:
    """Every routed (slotted) order has exactly one sub-order with its whole quantity and
    tokens (asserted inside the decode spy), every action row is one-hot, and |Delta| <= B,
    across random theta."""
    run = _play(job, policy=PerOrderPolicy(base, "A2", _theta("A2", 7, scale), normaliser))
    decisions = [d for *_, ds in run.log for d in ds]
    assert len(decisions) > 50
    deltas = np.concatenate([d.delta for d in decisions])
    assert np.all(np.abs(deltas) <= B) and np.abs(deltas).max() > (1.9 if scale > 1 else 0.1)
    assert run.overflow == 0


def test_own_earlier_orders_visible(a2: Run) -> None:
    """Within a turn, decision j's own-turn input holds the units / orders the agent's earlier
    decisions of the same turn assigned, per eligible producer."""
    seen = 0
    for _, _, _, ds in a2.log:
        acc: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for j, d in enumerate(ds):
            exp = np.array([acc[i] for i in d.eligible])
            assert np.array_equal(d.own_turn, exp)
            if j and d.own_turn.any():
                seen += 1
            acc[d.chosen][0] += d.quantity
            acc[d.chosen][1] += 1.0
    assert seen >= 5


def test_has_completed_indicator_separates_no_completion(job: dict, base: BaseScorer,
                                                         normaliser: FeatureNormaliser) -> None:
    """Latency reads 0.0 before any completion (4.2's default), so phi carries has_completed:
    0 for a routed pair without a completion (whose latency reads 0.0), 1 once completed. A
    failures-only pair and a pair that completes in its routing cycle (latency 0) therefore
    differ in phi, which a linear A1 term can use."""
    p = PerOrderPolicy(base, "A2", _theta("A2", 1), normaliser)
    p.sink = RawRows()
    run = _play(job, policy=p)
    raw, _ = p.sink.arrays()
    phi = np.vstack([d.phi for *_, ds in run.log for d in ds])
    done, routed = raw[:, HAS_COMPLETED_COL], raw[:, 39] == 0.0  # never_routed off
    assert set(np.unique(done)) == {0.0, 1.0}
    no_completion = (done == 0) & routed
    fast = (done == 1) & (raw[:, LATENCY_COL] == 0.0)
    assert no_completion.sum() >= 10 and fast.sum() >= 10
    assert np.all(raw[no_completion, LATENCY_COL] == 0.0)  # indistinguishable on latency ...
    assert np.all(raw[done == 1, 33] > 0)  # (completed EMA > 0 once completed)
    hc = np.unique(phi[:, HAS_COMPLETED_COL])
    assert len(hc) == 2  # ... but not on the indicator, after normalisation
    assert np.all(phi[no_completion, HAS_COMPLETED_COL] == hc.min())
    assert np.all(phi[fast, HAS_COMPLETED_COL] == hc.max())


def _snapshots(job: dict, policy: PerOrderPolicy, cycles: tuple[int, ...]) -> dict[int, object]:
    """Env copies at the first ordering turn of each cycle (the start of its ordering phase)."""
    snaps: dict[int, object] = {}

    def snapper(order: Callable) -> Callable:
        def call(e: object, agent: str) -> np.ndarray:
            c = e.system.cycle_counter  # type: ignore[attr-defined]
            if c in cycles and c not in snaps:
                snaps[c] = copy.deepcopy(e)
            return order(e, agent)

        return call

    _play(job, policy=policy, wrap=snapper)
    assert sorted(snaps) == sorted(cycles)
    return snaps


def _ordering(snap: object) -> list[str]:
    return [a for a in snap.agents if snap._role[a][0] == "order"]  # type: ignore[attr-defined]


def _held(snap: object, agent: str) -> set[int]:
    s = snap.system  # type: ignore[attr-defined]
    return {s.orders[o].product for o in s.ordering_agents[env_agent_id(snap, agent)].incoming}


def _decide(policy: PerOrderPolicy, e: object, agent: str) -> tuple[np.ndarray, list[Decision]]:
    policy.record = []
    action = policy(e, agent)  # type: ignore[arg-type]
    rec, policy.record = policy.record, None
    return action, rec


def _dec_bytes(ds: list[Decision]) -> list[tuple]:
    return [(d.slot, d.order_id, d.quantity, tuple(d.eligible), d.b.tobytes(),
             None if d.phi is None else d.phi.tobytes(),
             None if d.psi is None else d.psi.tobytes(), d.own_turn.tobytes(),
             d.delta.tobytes(), d.u, d.chosen) for d in ds]


def _move(action: np.ndarray, mask: np.ndarray, how: str) -> np.ndarray:
    """A valid whole choice derived from `action`: the receiver rotated to the next eligible
    producer, or every order to its first eligible producer."""
    a = action.reshape(mask.shape).copy()
    for k in range(mask.shape[0]):
        idx = np.flatnonzero(mask[k])
        if not idx.size or not a[k].any():
            continue
        cur = int(np.argmax(a[k]))
        a[k] = 0.0
        if how == "rotate":
            a[k, idx[(list(idx).index(cur) + 1) % idx.size]] = 1.0
        else:
            a[k, idx[0]] = 1.0
    return a.reshape(-1)


def test_adr0058_earlier_agents_leave_inputs_and_suborders_unchanged(
        job: dict, base: BaseScorer, normaliser: FeatureNormaliser) -> None:
    """ADR-0058: from the start of the ordering phase of cycles 2 / 4 / 6, the ordering agents
    before agent a act with the policy's keyed actions, their receivers rotated, everything to
    the first eligible producer, or a cyclic shift of the receiver rotation. a's phi, psi, b_i,
    own-turn features, Delta, draws, action, sub-orders and history are byte-identical."""
    from tokenomics.env.decode import decode_ordering, eligibility_mask

    policy = PerOrderPolicy(base, "A2", _theta("A2", 1), normaliser)
    snaps = _snapshots(job, policy, (2, 4, 6))
    differ = 0
    for snap in snaps.values():
        ordering = _ordering(snap)
        target = next(a for a in reversed(ordering) if _held(snap, a) and any(
            _held(snap, b) & _held(snap, a) for b in ordering[:ordering.index(a)]))
        earlier = ordering[:ordering.index(target)]

        def replay(variant: str) -> dict:
            e = copy.deepcopy(snap)
            for b in earlier:
                assert e.agent_selection == b  # type: ignore[attr-defined]
                act = policy(e, b)  # type: ignore[arg-type]
                if variant != "keyed":
                    view = e.system.ordering_agents[env_agent_id(e, b)].view(  # type: ignore[attr-defined]
                        e.system.producer_infos)  # type: ignore[attr-defined]
                    act = _move(act, eligibility_mask(e.spec_, view), variant)  # type: ignore[attr-defined]
                e.step(act)  # type: ignore[attr-defined]
            assert e.agent_selection == target  # type: ignore[attr-defined]
            aid = env_agent_id(e, target)
            action, rec = _decide(policy, e, target)
            view = e.system.ordering_agents[aid].view(e.system.producer_infos)  # type: ignore[attr-defined]
            cmd = decode_ordering(e.spec_, action, view)  # type: ignore[attr-defined]
            return {"decisions": _dec_bytes(rec), "action": action.tobytes(),
                    "subs": repr(sorted((o, [(s.producer_id, s.quantity, s.tokens) for s in ss])
                                        for o, ss in cmd.orders.items())),
                    "history": e.agent_history.snapshot(),  # type: ignore[attr-defined]
                    "bids": sorted((pid, b.order_id, b.quantity) for pid, p in
                                   e.system.producing_agents.items() for b in p.bids)}  # type: ignore[attr-defined]

        ref = replay("keyed")
        assert ref["decisions"] and all(d[5] is not None for d in ref["decisions"])
        for variant in ("rotate", "first"):
            got = replay(variant)
            differ += int(got.pop("bids") != ref["bids"])
            assert got == {k: v for k, v in ref.items() if k != "bids"}
    assert differ >= 3


def test_adr0058_turn_decision_equals_start_of_ordering_decision(
        job: dict, base: BaseScorer, normaliser: FeatureNormaliser) -> None:
    """ADR-0058, totals and turn position: every order-holding ordering agent's decision at its
    own turn (after every earlier agent placed its bids) equals the decision taken at the start
    of the ordering phase (no agent stepped). A leak that keeps per-producer totals over a
    product, e.g. the cycle's total placed units or bid count (a proxy of the AEC position),
    breaks it; receiver permutations alone cannot see such a leak."""
    policy = PerOrderPolicy(base, "A2", _theta("A2", 1), normaliser)
    snaps = _snapshots(job, policy, (2, 4, 6))
    checked = with_bids = 0
    for snap in snaps.values():
        e = copy.deepcopy(snap)
        for agent in _ordering(snap):
            assert e.agent_selection == agent  # type: ignore[attr-defined]
            if _held(snap, agent):
                start = _decide(policy, copy.deepcopy(snap), agent)
                turn = _decide(policy, e, agent)
                assert turn[0].tobytes() == start[0].tobytes()
                assert _dec_bytes(turn[1]) == _dec_bytes(start[1])
                checked += 1
                with_bids += any(p.bids for p in  # type: ignore[attr-defined]
                                 e.system.producing_agents.values())  # type: ignore[attr-defined]
            e.step(policy(e, agent))  # type: ignore[attr-defined]
    assert checked >= 30 and with_bids >= checked - 3


def test_fixed_decision_order(job: dict, base: BaseScorer,
                              normaliser: FeatureNormaliser) -> None:
    """Orders are decided in ascending order id (= slot order); the same state, parameters and
    keys replay byte-identical decisions and sub-orders."""
    policy = PerOrderPolicy(base, "A2", _theta("A2", 1), normaliser)
    snap = _snapshots(job, policy, (3,))[3]
    multi = 0
    for agent in _ordering(snap):
        runs = [_decide(policy, copy.deepcopy(snap), agent) for _ in range(2)]
        assert runs[0][0].tobytes() == runs[1][0].tobytes()
        assert _dec_bytes(runs[0][1]) == _dec_bytes(runs[1][1])
        ds = runs[0][1]
        assert [d.slot for d in ds] == sorted(d.slot for d in ds)
        assert [d.order_id for d in ds] == sorted(d.order_id for d in ds)
        multi += len(ds) >= 2
    assert multi >= 1


def test_empty_histories_still_differ(a0: Run) -> None:
    """Cycle 1: agents of one product with empty histories hold identical history inputs, draw
    from different keyed uniforms, and some pairs choose different producers."""
    by_product: dict[int, list[Decision]] = defaultdict(list)
    for cycle, _, product, ds in a0.log:
        if cycle == 1 and ds:
            by_product[product].append(ds[0])
    pairs = differ = 0
    for ds in by_product.values():
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                x, y = ds[i], ds[j]
                assert x.eligible == y.eligible
                assert x.phi[:, HIST].tobytes() == y.phi[:, HIST].tobytes()
                assert x.u != y.u
                pairs += 1
                differ += x.chosen != y.chosen
    assert pairs >= 3 and differ >= 1


def test_relabelled_agents(job: dict, base: BaseScorer, normaliser: FeatureNormaliser,
                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap two ordering agents' labels (each label now addresses the other's orders and
    history) and reassign their sampling keys accordingly: each agent's decision is unchanged.
    No agent identity enters the model."""
    policy = PerOrderPolicy(base, "A2", _theta("A2", 1), normaliser)
    snap = _snapshots(job, policy, (3,))[3]
    holders = [a for a in _ordering(snap) if _held(snap, a)]
    checked = 0
    for a1, a2 in zip(holders[::2], holders[1::2]):
        ref = {a: _decide(policy, copy.deepcopy(snap), a) for a in (a1, a2)}
        e = copy.deepcopy(snap)
        e._role[a1], e._role[a2] = e._role[a2], e._role[a1]  # type: ignore[attr-defined]
        swap = {a1: a2, a2: a1}
        real = pos.keyed_uniform
        with monkeypatch.context() as m:
            m.setattr(pos, "keyed_uniform",
                      lambda key, agent, cycle, slot: real(key, swap.get(agent, agent), cycle,
                                                           slot))
            got = {a: _decide(policy, e, swap[a]) for a in (a1, a2)}
        for a in (a1, a2):
            assert got[a][0].tobytes() == ref[a][0].tobytes()
            assert _dec_bytes(got[a][1]) == _dec_bytes(ref[a][1])
        checked += 1
        if checked == 3:
            break
    assert checked >= 2


def test_keys_exclude_the_member(job: dict, base: BaseScorer,
                                 normaliser: FeatureNormaliser) -> None:
    """The antithetic members theta +- eps draw from identical uniforms at the same state."""
    t0, eps = _theta("A2", 1, 0.3), _theta("A2", 2, 0.05)
    plus = PerOrderPolicy(base, "A2", t0 + eps, normaliser)
    minus = PerOrderPolicy(base, "A2", t0 - eps, normaliser)
    snap = _snapshots(job, plus, (2,))[2]
    n = 0
    for agent in _ordering(snap):
        up = [d.u for d in _decide(plus, copy.deepcopy(snap), agent)[1]]
        um = [d.u for d in _decide(minus, copy.deepcopy(snap), agent)[1]]
        assert up == um
        n += len(up)
    assert n >= 10


def test_shared_only_ablation_identical_history_inputs(job: dict, base: BaseScorer,
                                                       normaliser: FeatureNormaliser) -> None:
    """Shared-only: within a cycle every agent of a product sees the same history inputs (the
    mean over the product's agents), while agent-local history differs."""
    def hist_sets(run: Run) -> dict[tuple[int, int], set[bytes]]:
        out: dict[tuple[int, int], set[bytes]] = defaultdict(set)
        for cycle, _, product, ds in run.log:
            for d in ds:
                order = np.argsort(d.eligible)
                out[(cycle, product)].add(d.phi[order][:, HIST].tobytes())
        return out

    theta = _theta("A2", 1)
    shared = _play(job, policy=PerOrderPolicy(base, "A2", theta, normaliser, SHARED_ONLY))
    local = _play(job, policy=PerOrderPolicy(base, "A2", theta, normaliser, AGENT))
    s, l_ = hist_sets(shared), hist_sets(local)
    assert all(len(v) == 1 for v in s.values())
    assert sum(len(v) > 1 for v in l_.values()) >= 5
    assert sum(1 for c, _, _, ds in shared.log if ds and c > 1) >= 10


def test_hidden_shock_state_does_not_change_actions(job: dict, base: BaseScorer,
                                                    normaliser: FeatureNormaliser,
                                                    a2: Run) -> None:
    """Shock sentinel: with every piece of shock state behind the sentinel during each policy
    call, the actions, sub-orders and statistics equal the plain run's."""
    from test_shock_reactive import _hidden

    hidden = _play(job, policy=PerOrderPolicy(base, "A2", _theta("A2", 1), normaliser),
                   wrap=_hidden)
    assert a2.stats["shock_events"] > 0
    assert hidden.cmds == a2.cmds and hidden.stats == a2.stats


@pytest.mark.parametrize("bad", ["tau_zero", "tau_inf", "nan"])
def test_non_finite_guard(job: dict, base: BaseScorer, normaliser: FeatureNormaliser,
                          bad: str) -> None:
    """A theta whose scores or temperature are non-finite raises before decode, never emits
    NaN and never hangs."""
    snap = _snapshots(job, PerOrderPolicy(base, "A0", normaliser=normaliser), (2,))[2]
    theta = np.zeros(44)
    if bad == "tau_zero":
        theta[43] = -800.0  # tau = exp(-800) = 0
    elif bad == "tau_inf":
        theta[43] = 800.0
    else:
        theta[:43] = np.where(np.arange(43) % 2, 1e308, -1e308)  # inf - inf
    agent = next(a for a in _ordering(snap) if _held(snap, a))
    p = PerOrderPolicy(base, "A1", theta, normaliser)
    with pytest.raises(NonFiniteChoice):
        p(copy.deepcopy(snap), agent)  # type: ignore[arg-type]


# --- composite checkpoint, base, env helper, pins ---


def test_frozen_base_loads(job: dict, base: BaseScorer) -> None:
    """The legacy-versioned EXP-0012 scorer loads as the frozen base: its registered SHA-256,
    eval mode, no trainable parameter, and the scores the legacy policy emits."""
    from tokenomics.train.orderer_scorer import load_scorer, orderer_portable_schema

    assert base.sha256 == EXP0012_SHA
    assert not base.model.training
    assert not any(p.requires_grad for p in base.model.parameters())
    assert orderer_portable_schema().decoder_version == pos.BASE_DECODER_VERSION
    assert orderer_portable_schema().architecture_version == pos.BASE_ARCHITECTURE_VERSION
    legacy = load_scorer(str(ORDERER))
    for a, b_ in zip(legacy.state_dict().values(), base.model.state_dict().values()):
        assert a.equal(b_)


def test_composite_checkpoint_roundtrip_and_refusals(job: dict, base: BaseScorer,
                                                     normaliser: FeatureNormaliser,
                                                     tmp_path: Path) -> None:
    """Round trip in a choice-v1 env (same theta, sigma, normaliser hash, decisions). Refused,
    naming the decoder version, in a logits-v1 env; refused for a legacy scorer file, a base of
    another SHA-256 and an edited theta / normaliser."""
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.routing_audit import job_config

    cfg = job_config(job)
    policy = PerOrderPolicy(base, "A2", _theta("A2", 1), normaliser)
    path = tmp_path / "composite.json"
    pos.save_composite(policy, path)
    ck = json.loads(path.read_text())
    assert ck["decoder_version"] == "per-order-choice-v1"
    assert ck["architecture_version"] == "orderer-per-order-residual-v1"
    assert ck["base"]["sha256"] == EXP0012_SHA and ck["normaliser_sha256"] == normaliser.sha256
    assert ck["arm"] == {"name": "A2", "residual": True, "rank": 4, "n_params": 268}
    assert ck["sigma"] == [0.05] * 268

    env = pos.per_order_env(cfg, HORIZON)
    got = pos.load_composite(path, ORDERER, env)
    assert np.array_equal(got.theta, policy.theta) and got.normaliser == normaliser
    snap = _snapshots(job, policy, (2,))[2]
    for agent in _ordering(snap)[:6]:
        assert _dec_bytes(_decide(got, copy.deepcopy(snap), agent)[1]) == \
            _dec_bytes(_decide(policy, copy.deepcopy(snap), agent)[1])

    legacy_env = joint_stack_env(cfg, HORIZON)
    assert legacy_env.ordering_decode == LOGITS_V1
    with pytest.raises(ValueError, match="per-order-choice-v1"):
        pos.load_composite(path, ORDERER, legacy_env)
    with pytest.raises(ValueError, match="needs ordering_decode='choice-v1'"):
        policy(legacy_env, _ordering(snap)[0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not a per-order-composite-v1"):
        pos.load_composite(ORDERER, ORDERER, env)
    for edit in ("base", "theta", "normaliser", "decoder", "arm", "bound", "sigma", "u_init"):
        bad = json.loads(path.read_text())
        if edit == "base":
            bad["base"]["sha256"] = "0" * 64
        elif edit == "theta":
            bad["theta"][0] += 1.0
        elif edit == "normaliser":
            bad["normaliser"]["clip"] = 4.0
        elif edit == "decoder":
            bad["decoder_version"] = "per-order-choice-v0"
        elif edit == "arm":
            bad["arm"]["name"] = "A3"
        elif edit == "bound":
            bad["bound"] = 3.0
        elif edit == "sigma":
            bad["sigma"][0] = 0.1
        else:
            bad["u_init"]["seed"] = 1
        p2 = tmp_path / f"bad_{edit}.json"
        p2.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            pos.load_composite(p2, ORDERER, env)


def test_env_helper_and_pins(job: dict, base: BaseScorer, normaliser: FeatureNormaliser) -> None:
    """The helper builds a choice-v1 env with history attached; the pins carry the decode pin,
    so a choice-v1 run never resumes onto a logits-v1 out-dir."""
    from tokenomics.eval.routing_audit import job_config

    env = pos.per_order_env(job_config(job), HORIZON)
    assert env.ordering_decode == CHOICE_V1 and hasattr(env, "agent_history")
    policy = PerOrderPolicy(base, "A2", _theta("A2", 1), normaliser)
    pins = pos.per_order_pins(policy)
    assert pins["ordering_decode"] == CHOICE_V1 and decode_pins(CHOICE_V1).items() <= pins.items()
    assert pins != decode_pins(LOGITS_V1)
    assert pins["per_order"]["normaliser_sha256"] == normaliser.sha256
    assert pins["per_order"]["base_sha256"] == EXP0012_SHA
    with pytest.raises(ValueError, match="normaliser"):
        PerOrderPolicy(base, "A1")
    with pytest.raises(ValueError, match="history_mode"):
        PerOrderPolicy(base, "A0", history_mode="pooled")


def test_missing_history_or_key_refused(job: dict, base: BaseScorer) -> None:
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.routing_audit import job_config

    env = joint_stack_env(job_config(job), HORIZON, ordering_decode=CHOICE_V1)
    env.reset(seed=job["episode"])
    agent = next(a for a in env.agents if env._role[a][0] == "order")
    with pytest.raises(ValueError, match="attach_agent_history"):
        PerOrderPolicy(base, "A0")(env, agent)
    pos.attach_agent_history(env)
    with pytest.raises(RuntimeError, match="set_draw_key"):
        PerOrderPolicy(base, "A0")(env, agent)

