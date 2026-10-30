"""Agent-local history (task 4.2 of add-m4-orderer-redesign, design D5, ADR-0058).

The episode fixtures play the pinned 12-producer Atlas world ws-atlas-v1-12-0060 (train split,
EXP-0056's world; the first one of the Atlas's shocked 12-producer worlds scanned whose TS
episode holds all five domain causes among the agents' own sub-orders) at its first confirm seed
1295251592, replicate 0, with `shocks-i-smoke`, horizon 12, `entity_planner` and TS (the keyed
sampled-whole router over the EXP-0012 scorer) as the ordering policy.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import random
import string
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tokenomics.domain import system as system_mod
from tokenomics.domain.metrics import producer_requested_load
from tokenomics.domain.producing_agent import ProducingAgent
from tokenomics.domain.system import BidOutcome, CycleResult, OrderResolution
from tokenomics.env import agent_history as ah
from tokenomics.env.agent_history import (FEATURES, N_FEATURES, AgentHistory,
                                          attach_agent_history, env_agent_id, stream_producers)

REPO = Path(__file__).resolve().parents[2]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
WORLD = "ws-atlas-v1-12-0060"
EPISODE = 1295251592
HORIZON = 12
CAUSES = ("outage", "no_fit", "displaced", "dropped", "ttl")
MODULE = Path(ah.__file__)


def _lfs_present(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs_world = pytest.mark.skipif(not (_lfs_present(ATLAS_V2) and _lfs_present(ORDERER)),
                                 reason="EXP-0052 atlas / EXP-0012 checkpoint absent")


# --- static ---


def test_module_never_names_a_cause_field() -> None:
    """The module's source holds no `reason`, `REJECT_` or `FAIL_` identifier, attribute or
    string (AST), nor the text anywhere, comments and docstrings included."""
    src = MODULE.read_text()
    banned = ("reason", "REJECT_", "FAIL_")
    for tok in banned:
        assert tok.lower() not in src.lower(), tok
    for node in ast.walk(ast.parse(src)):
        names: list[str] = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.alias):
            names += [node.name, node.asname or ""]
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.keyword):
            names.append(node.arg or "")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.append(node.value)
        for n in names:
            assert not any(tok.lower() in n.lower() for tok in banned), n


# --- hand-built fixtures ---


def _owners(**incoming: list[str]) -> SimpleNamespace:
    return SimpleNamespace(ordering_agents={
        aid: SimpleNamespace(incoming={oid: None for oid in oids})
        for aid, oids in incoming.items()})


def _out(pid: str, oid: str, kind: str, q: int, routed: int, cause: str | None = None
         ) -> BidOutcome:
    return BidOutcome(pid, oid, "consumer", "x", q, q, kind, cause, routed)


def _play(cycles: list[tuple[SimpleNamespace, tuple[BidOutcome, ...], tuple[str, ...]]]
          ) -> AgentHistory:
    h = AgentHistory(alpha=0.3)
    for t, (owners, outs, resolved) in enumerate(cycles, start=1):
        h.observe_owners(owners)  # type: ignore[arg-type]
        h.fold(t, CycleResult(score=0, bid_outcomes=outs, order_resolutions=tuple(
            OrderResolution(o, "consumer", "x", 1, True, 1) for o in resolved)))
    return h


def test_feature_definitions_on_a_hand_fixture() -> None:
    """EMAs (α 0.3, idle pairs decay), latency (completed only, first value unbiased, held,
    then EMA), rate with the 1-unit prior over raw totals, log recency and the flag."""
    a = _owners(a=["o1", "o2"])
    h = _play([
        (a, (_out("p1", "o1", "accepted", 4, 1),), ()),
        (a, (_out("p1", "o1", "completed", 4, 1), _out("p1", "o2", "rejected", 2, 2)), ("o1",)),
        (a, (), ()),
    ])
    f = dict(zip(FEATURES, h.features("a", "p1")))
    exp = {"routed_ema": 0.7 * (0.3 * 2 + 0.7 * 0.3 * 4), "accepted_ema": 0.7 * 0.7 * 0.3 * 4,
           "completed_ema": 0.7 * 0.3 * 4, "failed_ema": 0.0, "rejected_ema": 0.7 * 0.3 * 2,
           "completion_latency": 1.0, "completion_rate": 4.5 / 7, "log_recency": np.log(3.0),
           "never_routed": 0.0}
    for k, v in exp.items():
        assert f[k] == pytest.approx(v, abs=1e-12), k
    assert h.totals("a", "p1") == {"routed": 6, "accepted": 4, "completed": 4, "failed": 0,
                                   "rejected": 2}
    never = dict(zip(FEATURES, h.features("a", "p9")))
    assert never == pytest.approx({
        "routed_ema": 0.0, "accepted_ema": 0.0, "completed_ema": 0.0, "failed_ema": 0.0,
        "rejected_ema": 0.0, "completion_latency": 0.0, "completion_rate": 0.5,
        "log_recency": float(np.log(5.0)), "never_routed": 1.0}, abs=1e-12)
    assert h.rows("a", ["p1", "p9"]).shape == (2, N_FEATURES)
    # a failed sub-order never moves the latency; a later 0-latency completion EMAs it
    h.observe_owners(_owners(a=["o3", "o4"]))  # type: ignore[arg-type]
    h.fold(4, CycleResult(score=0, bid_outcomes=(
        _out("p1", "o3", "accepted", 1, 4), _out("p1", "o3", "completed", 1, 4),
        _out("p1", "o2", "failed", 2, 2))))
    assert dict(zip(FEATURES, h.features("a", "p1")))["completion_latency"] == pytest.approx(0.7)
    h.observe_owners(_owners(a=["o4"]))  # type: ignore[arg-type]
    h.fold(5, CycleResult(score=0, bid_outcomes=(_out("p1", "o4", "failed", 3, 5),)))
    assert dict(zip(FEATURES, h.features("a", "p1")))["completion_latency"] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="already folded"):
        h.fold(5, CycleResult(score=0))


def test_rejected_only_pair_has_default_latency() -> None:
    """Before any completion the latency is the defined default 0.0, and the flag is off once
    routed (a rejection is a routing)."""
    h = _play([(_owners(a=["o1"]), (_out("p1", "o1", "rejected", 3, 1),), ())])
    f = dict(zip(FEATURES, h.features("a", "p1")))
    assert f["completion_latency"] == 0.0 and f["never_routed"] == 0.0
    assert f["completion_rate"] == pytest.approx(0.5 / 4)


def test_outage_not_revealed() -> None:
    """An outage rejection and a no-fit rejection, and a TTL failure during an outage and a
    dropped reservation, give identical history: each pair is counted as rejected / failed."""
    a = _owners(a=["o1"])
    for x, y, kind in (("outage", "no_fit", "rejected"), ("outage", "displaced", "rejected")):
        hx = _play([(a, (_out("p1", "o1", kind, 3, 1, x),), ())])
        hy = _play([(a, (_out("p1", "o1", kind, 3, 1, y),), ())])
        assert hx.snapshot() == hy.snapshot() and hx.totals("a", "p1")["rejected"] == 3
    hs = [_play([(a, (_out("p1", "o1", "accepted", 3, 1),), ()),
                 (a, (_out("p1", "o1", "failed", 3, 1, cause),), ("o1",))])
          for cause in ("ttl", "dropped")]
    assert hs[0].snapshot() == hs[1].snapshot() and hs[0].totals("a", "p1")["failed"] == 3
    assert hs[0].rows("a", ["p1"]).tobytes() == hs[1].rows("a", ["p1"]).tobytes()


def test_outcomes_are_attributed_to_their_holder_only() -> None:
    """Two agents at the same producer: each history holds only its own sub-orders; an order
    without a recorded holder is counted as unattributed, never guessed."""
    h = _play([(_owners(a=["o1"], b=["o2"]),
                (_out("p1", "o1", "accepted", 2, 1), _out("p1", "o2", "rejected", 5, 1),
                 _out("p1", "zz", "rejected", 7, 1)), ())])
    assert h.totals("a", "p1")["routed"] == 2 and h.totals("b", "p1")["routed"] == 5
    assert h.unattributed == 1
    assert h.features("a", "p1") != h.features("b", "p1")


# --- episode fixtures ---


@dataclasses.dataclass
class Run:
    stats: dict
    after_fold: dict[int, bytes]  # cycle -> snapshot right after its fold
    turns: list[dict]  # one per ordering turn
    causes: Counter
    history: AgentHistory


def _job() -> dict:
    from tokenomics.eval import decoder_check as dc
    from tokenomics.eval.provenance import sha256_file
    from tokenomics.eval.roster import load_roster
    from tokenomics.eval.routing_audit import build_jobs
    from tokenomics.eval.teacher_audit import load_atlas, world_view

    view = {v["world"]: v for v in map(world_view, load_atlas(ATLAS_V2)["worlds"])}[WORLD]
    assert view["size"] == 12 and view["shocks"]
    table = [dc.decoder_arm(dc.TS, load_roster(dc.DEFAULT_ROSTER))]
    pin = {"orderer": {"path": str(ORDERER), "sha256": sha256_file(str(ORDERER))}}
    jobs = build_jobs([dc.world_entry(view, "shocks-i-smoke")], table, 1, HORIZON, None, pin)
    assert jobs[0]["episode"] == EPISODE  # the confirm's first episode (generation order)
    (job,) = [j for j in jobs if j["episode"] == EPISODE and j["replicate"] == 0]
    return job


@pytest.fixture(scope="module")
def job() -> dict:
    pytest.importorskip("torch")
    if not (_lfs_present(ATLAS_V2) and _lfs_present(ORDERER)):
        pytest.skip("EXP-0052 atlas / EXP-0012 checkpoint absent")
    return _job()


def _env(job: dict, attach: bool = True) -> object:
    from tokenomics.eval import decoder_check as dc
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.routing_audit import job_config
    from tokenomics.eval.routing_diagnostics import set_draw_key

    env = joint_stack_env(job_config(job), HORIZON)
    set_draw_key(env, dc.job_draw_key(job))
    if attach:
        attach_agent_history(env)
    return env


def _policies(job: dict) -> tuple[Callable, Callable]:
    from tokenomics.eval import decoder_check as dc

    prod, order, _ = dc.decoder_policies(job)
    return prod, order


def _eligible(env: object, agent: str) -> list[str]:
    return stream_producers(env.system, env_agent_id(env, agent),  # type: ignore[attr-defined]
                            env.spec_.producer_ids)  # type: ignore[attr-defined]


def _shared_rows(env: object, aid: str) -> str:
    """The agent's whole product-stream producer rows."""
    v = env.system.product_stream_view(aid)  # type: ignore[attr-defined]
    return repr((v.producers, v.producer_products))


POOLED = ("eligible", "recent_routed_qty", "recent_completed_qty", "recent_rejected_qty",
          "recent_success_rate", "recent_throughput", "recent_completion_latency",
          "recent_failure_latency")


def _pooled_rows(env: object, aid: str) -> str:
    """The rows every agent of a product shares: producer stats and the pooled
    producer-product telemetry. The quantity-dependent producer-product fields
    (accept / complete / makeable now, bottleneck ratios) are scaled by the agent's own open
    backlog, so they are own-order inputs, not shared ones."""
    v = env.system.product_stream_view(aid)  # type: ignore[attr-defined]
    return repr((v.producers, {pid: tuple(getattr(s, f) for f in POOLED)
                               for pid, s in v.producer_products.items()}))


def _play_episode(job: dict, wrap_order: Callable[[Callable], Callable] | None = None,
                  wrap_history: Callable[[object, AgentHistory], None] | None = None) -> Run:
    from tokenomics.eval.runner import run_episode

    env = _env(job)
    h: AgentHistory = env.agent_history  # type: ignore[attr-defined]
    after_fold: dict[int, bytes] = {}
    causes: Counter = Counter()
    fold = h.fold

    def recording_fold(cycle: int, result: CycleResult) -> None:
        for o in result.bid_outcomes:  # the test may read the cause, the module never does
            if o.order_id in h._owners and o.reason is not None:
                causes[o.reason] += 1
        fold(cycle, result)
        after_fold[cycle] = h.snapshot()

    h.fold = recording_fold  # type: ignore[method-assign]
    if wrap_history is not None:
        wrap_history(env, h)
    prod, order = _policies(job)
    turns: list[dict] = []

    def traced(e: object, agent: str) -> np.ndarray:
        aid = env_agent_id(e, agent)
        system = e.system  # type: ignore[attr-defined]
        before = h.snapshot()
        elig = _eligible(e, agent)
        rows = h.rows(aid, elig)
        loads = {pid: producer_requested_load(p) for pid, p in system.producing_agents.items()}
        action = (wrap_order(order) if wrap_order else order)(e, agent)
        turns.append({"cycle": system.cycle_counter, "agent": agent, "aid": aid,
                      "product": system.stream_products[aid], "eligible": elig,
                      "snapshot": before, "rows": rows.tobytes(), "loads": loads,
                      "view_loads": {pid: s.requested_load for pid, s in
                                     system.product_stream_view(aid).producers.items()},
                      "shared": _pooled_rows(e, aid), "h_cycle": h.cycle})
        assert h.snapshot() == before  # the turn writes nothing
        return action

    run_episode(env, prod, traced, seed=job["episode"], action_seed=job["action_seed"])
    return Run(dict(env.episode_stats), after_fold, turns, causes, h)  # type: ignore[attr-defined]


@pytest.fixture(scope="module")
def plain(job: dict) -> Run:
    return _play_episode(job)


def _rand(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(1, 12)))


def _scramble(values: list[str], rng: random.Random) -> list[str]:
    """Same length (the domain's strict zip holds): a seeded permutation, then about half the
    entries replaced by random strings. Nothing is removed."""
    out = list(values)
    rng.shuffle(out)
    return [_rand(rng) if rng.random() < 0.5 else v for v in out]


@contextmanager
def _scrambled_causes(monkeypatch: pytest.MonkeyPatch, seed: int) -> Iterator[Counter]:
    rng = random.Random(seed)
    changed: Counter = Counter()
    produce = ProducingAgent.produce

    def scrambled_produce(self: ProducingAgent, *args: object, **kwargs: object) -> object:
        r = produce(self, *args, **kwargs)
        rej, fail = _scramble(r.rejection_reasons, rng), _scramble(r.failure_reasons, rng)
        assert len(rej) == len(r.rejected) and len(fail) == len(r.failed)
        changed["lists"] += int(rej != r.rejection_reasons) + int(fail != r.failure_reasons)
        return dataclasses.replace(r, rejection_reasons=rej, failure_reasons=fail)

    real = system_mod.BidOutcome

    def scrambled_outcome(*args: object, **kwargs: object) -> BidOutcome:
        o = real(*args, **kwargs)  # type: ignore[arg-type]
        new = rng.choice([*CAUSES, None, _rand(rng)])
        changed["outcomes"] += int(new != o.reason)
        return dataclasses.replace(o, reason=new)

    with monkeypatch.context() as m:
        m.setattr(ProducingAgent, "produce", scrambled_produce)
        m.setattr(system_mod, "BidOutcome", scrambled_outcome)
        yield changed


@needs_world
def test_reasons_scrambled(job: dict, plain: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every cause occurs among the agents' own sub-orders; with the cause values substituted
    (in the ProductionResult lists and in BidOutcome) every agent's history, after every fold
    and at every ordering turn, is byte-identical to the unscrambled episode's."""
    assert all(plain.causes[c] >= 1 for c in CAUSES), plain.causes
    assert plain.history.unattributed == 0 and plain.stats["shock_events"] > 0
    for seed in (0, 1):
        with _scrambled_causes(monkeypatch, seed) as changed:
            got = _play_episode(job)
        assert changed["lists"] > 10 and changed["outcomes"] > 100, changed
        assert set(got.causes) - set(CAUSES)  # the scrambled run saw substituted values
        assert got.after_fold == plain.after_fold
        assert [(t["snapshot"], t["rows"]) for t in got.turns] == [
            (t["snapshot"], t["rows"]) for t in plain.turns]
        assert got.history.snapshot() == plain.history.snapshot()
        assert got.stats == plain.stats


@needs_world
def test_folded_only_after_complete_cycle(plain: Run) -> None:
    """What an agent sees in cycle t is exactly the state folded after cycle t − 1's
    `complete_cycle` (empty in cycle 1): nothing from cycle t's ordering or production."""
    assert plain.turns and set(plain.after_fold) == set(range(1, plain.stats["cycles"] + 1))
    empty = AgentHistory(alpha=0.3).snapshot()
    for t in plain.turns:
        assert t["h_cycle"] == t["cycle"] - 1
        assert t["snapshot"] == (plain.after_fold[t["cycle"] - 1] if t["cycle"] > 1 else empty)
    h = plain.history
    assert all(h._pairs[k].last_routed <= h.cycle for k in h.pairs())


@needs_world
def test_routed_units_equal_placed_bids_and_routing_cycles(job: dict) -> None:
    """Routed = accepted + rejected outcome units equals the units of the bids placed in each
    cycle; accepted / rejected outcomes are stamped with the played cycle, completions with an
    earlier or equal one (the domain field the latency reads)."""
    placed = Counter()
    seen: list[tuple[int, BidOutcome]] = []

    def wrap(env: object, h: AgentHistory) -> None:
        observe, fold = h.observe_owners, h.fold

        def observing(system: object) -> None:  # called just before the auction
            placed["units"] += sum(b.quantity for p in system.producing_agents.values()  # type: ignore[attr-defined]
                                   for b in p.bids)
            observe(system)  # type: ignore[arg-type]

        def folding(cycle: int, result: CycleResult) -> None:
            seen.extend((cycle, o) for o in result.bid_outcomes)
            fold(cycle, result)

        h.observe_owners, h.fold = observing, folding  # type: ignore[method-assign]

    run = _play_episode(job, wrap_history=wrap)
    routed = sum(run.history.totals(a, p)["routed"] for a, p in run.history.pairs())
    assert routed == placed["units"] > 0
    for cycle, o in seen:
        if o.outcome in ("accepted", "rejected"):
            assert o.routed_cycle == cycle
        else:
            assert 1 <= o.routed_cycle <= cycle


@needs_world
def test_different_histories_different_inputs(plain: Run) -> None:
    """Two agents of the same product with different past routing at a producer get different
    history rows for it, while their shared (pooled) producer rows are identical."""
    found = 0
    by_turn: dict[tuple[int, str], list[dict]] = {}
    for t in plain.turns:
        by_turn.setdefault((t["cycle"], t["product"]), []).append(t)
    for turns in by_turn.values():
        for i, x in enumerate(turns):
            for y in turns[i + 1:]:
                assert x["eligible"] == y["eligible"]
                rx = np.frombuffer(x["rows"]).reshape(-1, N_FEATURES)
                ry = np.frombuffer(y["rows"]).reshape(-1, N_FEATURES)
                if not np.array_equal(rx, ry):
                    assert x["shared"] == y["shared"]
                    found += 1
    assert found >= 10


@needs_world
def test_last_cycles_load_only(plain: Run) -> None:
    """Requested load read during the ordering phase is the previous auction's value and does
    not change while ordering agents route in the current cycle."""
    per_cycle: dict[int, list[dict]] = {}
    for t in plain.turns:
        per_cycle.setdefault(t["cycle"], []).append(t)
    assert len(per_cycle) >= 10
    for turns in per_cycle.values():
        first = turns[0]["loads"]
        assert any(v > 0 for v in first.values()) or turns[0]["cycle"] == 1
        for t in turns:
            assert t["loads"] == first
            assert all(t["view_loads"][pid] == first[pid] for pid in t["view_loads"])


@needs_world
def test_earlier_ordering_agents_same_cycle_routing_is_invisible(job: dict) -> None:
    """ADR-0058: snapshot the env as the ordering phase of cycles 2 / 4 / 6 starts; replay the
    ordering agents before agent a with their TS actions, a cyclic permutation of them, the
    reversed order, and `even_split`. a's history (whole state and its rows), shared rows and
    TS sub-orders are byte-identical."""
    from tokenomics.env.decode import decode_ordering
    from tokenomics.eval.baselines import ORDERING_BASELINES
    from tokenomics.eval.runner import run_episode

    env = _env(job)
    prod, order = _policies(job)
    snaps: dict[int, object] = {}

    def snapper(e: object, agent: str) -> np.ndarray:
        c = e.system.cycle_counter  # type: ignore[attr-defined]
        if c in (2, 4, 6) and c not in snaps:
            snaps[c] = copy.deepcopy(e)
        return order(e, agent)

    run_episode(env, prod, snapper, seed=job["episode"], action_seed=job["action_seed"])
    assert sorted(snaps) == [2, 4, 6]
    differ = 0
    for snap in snaps.values():
        ordering = [a for a in snap.agents if snap._role[a][0] == "order"]  # type: ignore[attr-defined]
        held = {a: {snap.system.orders[o].product  # type: ignore[attr-defined]
                    for o in snap.system.ordering_agents[env_agent_id(snap, a)].incoming}  # type: ignore[attr-defined]
                for a in ordering}
        target = next(a for a in reversed(ordering) if held[a] and any(
            held[b] & held[a] for b in ordering[:ordering.index(a)]))
        earlier = ordering[:ordering.index(target)]

        def replay(act: Callable[[object, str, int], np.ndarray]) -> tuple[dict, list]:
            e = copy.deepcopy(snap)
            acts = []
            for i, b in enumerate(earlier):
                assert e.agent_selection == b  # type: ignore[attr-defined]
                acts.append(np.asarray(act(e, b, i)))
                e.step(acts[-1])  # type: ignore[attr-defined]
            assert e.agent_selection == target  # type: ignore[attr-defined]
            aid = env_agent_id(e, target)
            h: AgentHistory = e.agent_history  # type: ignore[attr-defined]
            view = e.system.ordering_agents[aid].view(e.system.producer_infos)  # type: ignore[attr-defined]
            action = np.asarray(order(e, target))
            cmd = decode_ordering(e.spec_, action, view)  # type: ignore[attr-defined]
            bids = sorted((pid, b.order_id, b.quantity)
                          for pid, p in e.system.producing_agents.items() for b in p.bids)  # type: ignore[attr-defined]
            return {"history": h.snapshot(), "rows": h.rows(aid, _eligible(e, target)).tobytes(),
                    "shared": _shared_rows(e, aid), "action": action.tobytes(),
                    "subs": repr({oid: [(s.producer_id, s.quantity, s.tokens) for s in subs]
                                  for oid, subs in sorted(cmd.orders.items())}),
                    "bids": bids}, acts

        ref, keyed = replay(lambda e, b, i: order(e, b))
        n = len(keyed)
        for variant in (lambda e, b, i: keyed[(i + 1) % n], lambda e, b, i: keyed[n - 1 - i],
                        lambda e, b, i: ORDERING_BASELINES["even_split"](e, b)):
            got, _ = replay(variant)
            differ += int(got.pop("bids") != ref["bids"])
            assert got == {k: v for k, v in ref.items() if k != "bids"}
    assert differ >= 3  # the earlier agents' routing really changed


@needs_world
def test_hidden_shock_state_does_not_change_history(job: dict, plain: Run) -> None:
    """With every piece of shock state behind the sentinel (any access raises) while the
    history records holders, folds and is read, the history equals the plain run's."""
    from test_shock_reactive import _hidden

    def hide(fn: Callable) -> Callable:
        envs: list[object] = []

        def call(*args: object) -> object:
            return _hidden(lambda env, agent: fn(*args))(envs[0], "")

        call.envs = envs  # type: ignore[attr-defined]
        return call

    def wrap(env: object, h: AgentHistory) -> None:
        for name in ("observe_owners", "fold", "rows", "snapshot"):
            f = hide(getattr(h, name))
            f.envs.append(env)  # type: ignore[attr-defined]
            setattr(h, name, f)

    got = _play_episode(job, wrap_history=wrap)
    assert got.after_fold == plain.after_fold
    assert [t["rows"] for t in got.turns] == [t["rows"] for t in plain.turns]
    assert got.stats == plain.stats


@needs_world
def test_history_resets_at_reset_and_forks_keep_their_own(job: dict, plain: Run) -> None:
    """A second reset clears the history; replaying the episode on the same env gives the
    same history as the first time. A deepcopy (fork) folds into its own history and leaves
    the original's untouched; both reach the same end state."""
    from tokenomics.eval.runner import run_episode

    env = _env(job)
    prod, order = _policies(job)
    run_episode(env, prod, order, seed=job["episode"], action_seed=job["action_seed"])
    h: AgentHistory = env.agent_history  # type: ignore[attr-defined]
    assert h.snapshot() == plain.history.snapshot() and h.pairs()
    env.reset(seed=7)  # type: ignore[attr-defined]
    assert h.cycle == 0 and not h.pairs() and h.snapshot() == AgentHistory(0.3).snapshot()
    assert env.agent_history is h  # type: ignore[attr-defined]
    assert h.alpha == env.system.telemetry._alpha  # type: ignore[attr-defined]

    forks: list[object] = []
    frozen: list[bytes] = []

    def forker(e: object, agent: str) -> np.ndarray:
        if e.system.cycle_counter == 5 and not forks:  # type: ignore[attr-defined]
            forks.append(copy.deepcopy(e))
            frozen.append(e.agent_history.snapshot())  # type: ignore[attr-defined]
            fork = forks[0]
            assert fork.agent_history is not e.agent_history  # type: ignore[attr-defined]
            # play the fork to its end: the original's history must not move
            while not (fork.terminations[fork.agent_selection]  # type: ignore[attr-defined]
                       or fork.truncations[fork.agent_selection]):  # type: ignore[attr-defined]
                a = fork.agent_selection  # type: ignore[attr-defined]
                role = fork._role[a][0]  # type: ignore[attr-defined]
                fork.step(prod(fork, a) if role == "prod" else order(fork, a))  # type: ignore[attr-defined]
            assert e.agent_history.snapshot() == frozen[0]  # type: ignore[attr-defined]
        return order(e, agent)

    run_episode(env, prod, forker, seed=job["episode"], action_seed=job["action_seed"])
    assert forks and h.snapshot() == plain.history.snapshot()
    assert forks[0].agent_history.snapshot() == h.snapshot()  # type: ignore[attr-defined]


@needs_world
def test_attaching_changes_no_episode_statistic(job: dict, plain: Run) -> None:
    """The history is opt-in and observational: the same episode without it has identical
    statistics."""
    from tokenomics.eval.runner import run_episode

    env = _env(job, attach=False)
    prod, order = _policies(job)
    run_episode(env, prod, order, seed=job["episode"], action_seed=job["action_seed"])
    assert not hasattr(env, "agent_history")
    assert dict(env.episode_stats) == plain.stats  # type: ignore[attr-defined]
