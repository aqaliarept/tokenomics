"""Task 6.2 (add-m3-development-audit design D5): the teacher audit driver and the pluggable
scenario source / floor / candidate set of `eval/bounded_rollout_orderer.py`."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from tokenomics.env.shock_posterior import PosteriorSample, ShockPosterior
from tokenomics.eval import teacher_audit as ta
from tokenomics.eval.atlas import content_hash
from tokenomics.eval.baselines import PRODUCING_BASELINES
from tokenomics.eval.bounded_rollout_orderer import (
    BoundedScenarioOrderer,
    ClairvoyantSource,
    DeterministicSource,
    WholeOrderCandidates,
    deterministic_future,
    resolve_stream,
)
from tokenomics.eval.gate import DigestMismatch
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.provenance import sha256_file
from tokenomics.eval.roster import load_roster
from tokenomics.eval.routing_audit import build_jobs
from tokenomics.eval.routing_diagnostics import DIAGNOSTIC_ROUTERS
from tokenomics.eval.runner import run_episode
from tokenomics.eval.task_archive import TaskRecord, replay
from tokenomics.eval.world_space import config_from_record

REPO = Path(__file__).resolve().parents[2]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
PROBE = REPO / "openspec/changes/add-m3-development-audit/probes/exp0052_check.py"
PROD = PRODUCING_BASELINES["entity_planner"]
ROUND_ROBIN = DIAGNOSTIC_ROUTERS["round_robin"]
# The sample `probes/exp0052_check.py --reclassified` prints ("new"), in class rank order.
EXPECTED_SAMPLE = [
    ("hard", "ws-atlas-v1-12-0060"), ("hard", "ws-atlas-v1-12-0176"),
    ("hard", "ws-atlas-v1-12-0005"), ("hard", "ws-atlas-v1-12-0348"),
    ("open", "ws-atlas-v1-12-0111"), ("open", "ws-atlas-v1-12-0344"),
    ("open", "ws-atlas-v1-12-0230"), ("open", "ws-atlas-v1-12-0308"),
    ("open", "ws-atlas-v1-12-0392"), ("open", "ws-atlas-v1-12-0254"),
    ("open", "ws-atlas-v1-12-0155"), ("open", "ws-atlas-v1-12-0408"),
]
SHOCKED_WORLD = "ws-atlas-v1-12-0060"  # train, hard, burst demand, shocks
CYCLIC_WORLD = "ws-atlas-v1-12-0176"  # train, hard, cyclic demand, no shocks
# EXP-0042's first registered task (`_pod_inputs/gate_archive.jsonl`, sha256 bc6ca4eb...)
S29 = ('{"demand_trace":[],"episode_seed":29,"event_trace":[],"expected_claim_boundary":'
       '"in_claim","failure_class":"policy_weakness","generator_version":"mine-v1",'
       '"initial_state_spec":{},"legal":true,"notes":"mined: score=0.4345 gap=133.333u win=1.0",'
       '"reference_status":"solved","replayable":true,"serializer_version":"v1",'
       '"shock_trace":[],"task_family":"random","task_id":"mine-build_scale_config-'
       'capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-'
       's29","world_seed":29,"world_spec":{"generator":"build_scale_config","params":'
       '{"capacity_scale":0.6,"depth":3,"primal_capacity_scale":0.3,"producers_per_layer":2,'
       '"quantity_max":14}}}')
# Pre-change decisions (commit 427c959) of the EXP-0042 default search on S29: depth 1,
# horizon 12, seed 3299, entity_planner producer; sha256 over (cycle, agent, action bytes).
S29_GOLDEN = "25c221183860f0779cb19291f19c95af0dc6b7c3b4dbaeab66a4ac68ba33e468"

needs_atlas = pytest.mark.skipif(not ATLAS_V2.exists(), reason="EXP-0052 atlas_v2.json absent")


def _atlas() -> dict:
    return json.loads(ATLAS_V2.read_text())


def _entry(atlas: dict, wid: str) -> dict:
    return next(w for w in atlas["worlds"] if w["record"]["world_id"] == wid)


def _manifest(tmp_path: Path, edits: dict[str, dict] | None = None,
              extra: list[dict] | None = None) -> Path:
    """A small atlas-v2 manifest from real EXP-0052 entries (`edits` overwrite entry fields)."""
    atlas = _atlas()
    ids = [SHOCKED_WORLD, CYCLIC_WORLD, "ws-atlas-v1-12-0111", "ws-atlas-v1-12-0321",
           "ws-atlas-v1-12-0235", "ws-atlas-v1-12-0091", "ws-atlas-v1-12-0089"]
    worlds = [copy.deepcopy(_entry(atlas, w)) for w in ids]
    for w in worlds:
        w.update((edits or {}).get(w["record"]["world_id"], {}))
    doc = {k: v for k, v in atlas.items() if k not in ("worlds", "content_hash")}
    doc["worlds"] = worlds + (extra or [])
    doc["content_hash"] = content_hash(doc)
    path = tmp_path / "atlas_v2.json"
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture(scope="module")
def orderer_ckpt(tmp_path_factory: pytest.TempPathFactory) -> str:
    pytest.importorskip("torch")
    from tokenomics.train.orderer_scorer import OrdererScoreModel, save_scorer

    path = str(tmp_path_factory.mktemp("ckpt") / "orderer_scorer.pt")
    save_scorer(OrdererScoreModel(hidden=16), path)
    return path


def _job(view: dict, arm: str, ckpt: str, horizon: int, smoke: bool = True) -> dict:
    e = ta.world_entry(view, "shocks-i-smoke" if smoke and view["shocks"] else None)
    job = build_jobs([e], [ta.teacher_arm(arm)], 1, horizon, None,
                     {"orderer": {"path": ckpt, "sha256": sha256_file(ckpt)}})[0]
    job["teacher"] = arm
    return job


# --- the sample ---


@needs_atlas
def test_pre_registered_sample_equals_the_probe() -> None:
    """D5's sample from EXP-0052's reclassified atlas equals the probe's "new" sample."""
    atlas = ta.load_atlas(ATLAS_V2)
    assert ta.pre_registered_sample(atlas) == EXPECTED_SAMPLE
    src = PROBE.read_text()
    ns: dict = {"POC_PATTERNS": set(ta.POC_PATTERNS)}
    exec("def poc_a_sample" + src.split("def poc_a_sample")[1].split("\ndef ")[0], ns)
    probe_view = {v["world"]: {**v, "cls": v["class"]} for v in map(ta.world_view, atlas["worlds"])}
    assert ns["poc_a_sample"](probe_view, "cls") == EXPECTED_SAMPLE
    for _, w in EXPECTED_SAMPLE:
        v = ta.world_view(_entry(atlas, w))
        assert v["split"] == "train" and v["size"] == 12 and ta.refusal(v) is None
    # the confirm's first episode seed (EXP-0052 episodes.csv)
    assert ta.first_confirm_seed(ta.world_view(_entry(atlas, SHOCKED_WORLD))) == 1295251592


STEP2_SAMPLE = [
    ("hard", "ws-atlas-v1-12-0060"), ("hard", "ws-atlas-v1-12-0176"),
    ("hard", "ws-atlas-v1-12-0005"), ("hard", "ws-atlas-v1-12-0348"),
    ("open", "ws-atlas-v1-12-0111"), ("open", "ws-atlas-v1-12-0344"),
    ("open", "ws-atlas-v1-12-0308"), ("open", "ws-atlas-v1-12-0392"),
]


@needs_atlas
def test_step2_reduction_sample() -> None:
    """D5 step 2 (sizing.md §EXP-0053, the EXP-0053 sample): the same rule at 8 world-episodes,
    4 hard + 4 open; 12-0408 leaves by the rule."""
    atlas = ta.load_atlas(ATLAS_V2)
    assert ta.pre_registered_sample(atlas, ta.STEP2) == STEP2_SAMPLE
    assert [w for _, w in ta.pre_registered_sample(atlas, ta.STEP2)] == [
        v["world"] for v in ta.select_worlds(atlas, reduction=ta.STEP2)]
    assert ta.pre_registered_sample(atlas) == EXPECTED_SAMPLE  # the full sample is unchanged
    with pytest.raises(ValueError, match="unknown reduction"):
        ta.pre_registered_sample(atlas, "step9")
    with pytest.raises(ValueError, match="exclusive"):
        ta.select_worlds(atlas, [SHOCKED_WORLD], reduction=ta.STEP2)


class _PastChecks(Exception):
    pass


@needs_atlas
def test_step2_reduction_allowed_in_a_real_run(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """A real run with `--reduction step2` passes the real-run checks (it stops at the
    checkpoint check, which runs after them); any other filter is still refused."""
    def past(*_a: object, **_k: object) -> None:
        raise _PastChecks

    monkeypatch.setattr(ta, "checked_orderer", past)
    monkeypatch.setattr(ta, "lp_available", lambda: True)
    roster = load_roster(str(ta.DEFAULT_ROSTER))
    kw = dict(atlas_path=ATLAS_V2, orderer_ckpt="o.pt", roster=roster, source_commit="a" * 40,
              log=lambda s: None)
    with pytest.raises(_PastChecks):
        ta.run_teacher_audit(out_dir=tmp_path / "step2", reduction=ta.STEP2, **kw)
    with pytest.raises(_PastChecks):  # the full sample, as before
        ta.run_teacher_audit(out_dir=tmp_path / "full", **kw)
    for extra, flag in (({"max_worlds": 2}, "--max-worlds"), ({"arms": ["floor", "d2"]}, "--arms"),
                        ({"horizon": 12}, "horizon 12"), ({"baseline": []}, "--baseline"),
                        ({"max_search_actions": 3}, "--max-search-actions"),
                        ({"replicates": 1}, "--replicates"),
                        ({"shock_profile": "shocks-i-smoke"}, "--shock-profile")):
        with pytest.raises(ValueError, match=flag):
            ta.run_teacher_audit(out_dir=tmp_path / "x", reduction=ta.STEP2, **kw, **extra)
    with pytest.raises(ValueError, match="--worlds is set"):  # an explicit world list, no reduction
        ta.run_teacher_audit(out_dir=tmp_path / "w", worlds=[SHOCKED_WORLD], **kw)


@needs_atlas
def test_step2_reduction_recorded_in_spec(tmp_path: Path) -> None:
    """spec.json pins the reduction and records its sample and the deviation note."""
    out = tmp_path / "rec"
    code = ta.run_teacher_audit(atlas_path=ATLAS_V2, out_dir=out, orderer_ckpt="unused.pt",
                                roster=load_roster(str(ta.DEFAULT_ROSTER)), dry_run=True,
                                reduction=ta.STEP2, max_worlds=1, arms=["floor"], baseline=[],
                                horizon=3, log=lambda s: None)
    assert code == 0
    spec = json.loads((out / "spec.json").read_text())
    assert spec["pins"]["reduction"] == "step2"
    red = spec["pre_registered"]["reduction"]
    assert (red["name"], red["size"], red["hard"]) == ("step2", 8, 4)
    assert "deviation" in red["note"] and "`ub` kept" in red["note"]
    assert [x["world"] for x in spec["pins"]["sample"]] == [SHOCKED_WORLD]
    assert spec["pins"]["sample"][0]["pre_registered"] is True


@needs_atlas
def test_moving_hit_world_offered_to_the_teacher(tmp_path: Path) -> None:
    """Scenario "Moving-hit world offered to the teacher": hit and switch worlds with the best
    room are not in the sample, and an explicit request for one is refused."""
    path = _manifest(tmp_path, {"ws-atlas-v1-12-0321": {"room": 0.99, "confirm_class": "open"},
                                "ws-atlas-v1-12-0235": {"room": 0.98, "confirm_class": "open"}})
    atlas = ta.load_atlas(path)
    sample = [w for _, w in ta.pre_registered_sample(atlas)]
    assert "ws-atlas-v1-12-0321" not in sample and "ws-atlas-v1-12-0235" not in sample
    assert "ws-atlas-v1-12-0111" in sample
    for w in ("ws-atlas-v1-12-0321", "ws-atlas-v1-12-0235"):
        with pytest.raises(ValueError, match="hidden state"):
            ta.select_worlds(atlas, [w])


@needs_atlas
def test_held_out_sealed(tmp_path: Path) -> None:
    """Scenario "Held-out sealed": held-out and validation worlds never enter the sample, even
    when they rank first, and are refused when named; so is a world above 42 producers."""
    big = copy.deepcopy(_entry(_atlas(), "ws-atlas-v1-12-0155"))
    big["record"] = {**big["record"], "world_id": "ws-fake-120", "size": 120}
    path = _manifest(tmp_path, {"ws-atlas-v1-12-0091": {"gain": 9999.0},
                                "ws-atlas-v1-12-0089": {"room": 0.999}}, extra=[big])
    atlas = ta.load_atlas(path)
    sample = ta.pre_registered_sample(atlas)
    assert sample == [("hard", SHOCKED_WORLD), ("hard", CYCLIC_WORLD),
                      ("open", "ws-atlas-v1-12-0111")]
    assert all(ta.world_view(_entry(atlas, w))["split"] == "train" for _, w in sample)
    with pytest.raises(ValueError, match="held-out split"):
        ta.select_worlds(atlas, ["ws-atlas-v1-12-0091"])
    with pytest.raises(ValueError, match="validation split"):
        ta.select_worlds(atlas, ["ws-atlas-v1-12-0089"])
    with pytest.raises(ValueError, match="120 producers"):
        ta.select_worlds(atlas, ["ws-fake-120"])


# --- search parts ---


def test_one_scenario_zero_variance_rule() -> None:
    """D4 "Deterministic futures": one scenario, SE 0 -> a unique strictly positive gain."""
    assert resolve_stream(np.array([[3.0, 0.0]]), (5, 7)).choice == 5
    assert resolve_stream(np.array([[0.0, 0.0]]), (5, 7)).reason == "deterministic_zero"
    assert resolve_stream(np.array([[-1.0, -2.0]]), (5, 7)).reason == "confident_base"
    tie = resolve_stream(np.array([[2.0, 2.0]]), (5, 7))
    assert not tie.resolved and tie.reason == "mean_tie"


@needs_atlas
def test_candidate_set_is_whole_orders_plus_trained() -> None:
    """The candidate set: the whole order to each of the E eligible producers plus the trained
    orderer's row; the floor's own receiver is one of the E and is dropped from the search."""
    view = ta.world_view(_entry(_atlas(), CYCLIC_WORLD))
    env = joint_stack_env(config_from_record(view["record"]), 4)
    env.reset(seed=ta.first_confirm_seed(view))
    trained_row = np.full(env.spec_.k_stream * env.spec_.num_producers, 0.5, dtype=np.float32)
    cands = WholeOrderCandidates(trained=lambda e, a: trained_row)
    seen = 0
    for agent in env.agent_iter(max_iter=10_000):
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        if env._role[agent][0] == "prod":
            env.step(PROD(env, agent))
            continue
        floor = ROUND_ROBIN(env, agent).reshape(env.spec_.k_stream, env.spec_.num_producers)
        for k, _, elig, rows, labels in cands.candidates(env, agent):
            seen += 1
            assert len(rows) == len(elig) + 1 and labels[-1] == "trained"
            assert labels[:-1] == [f"whole:{c}" for c in elig]
            for j, row in enumerate(rows[:-1]):
                kept = [c for c in elig if row[c] > -10]
                assert kept == [elig[j]]  # the whole order to one producer
            assert np.array_equal(rows[-1], trained_row.reshape(floor.shape)[k])
            assert sum(np.array_equal(r, floor[k]) for r in rows[:-1]) == 1
        from tokenomics.eval.bounded_rollout_orderer import SearchTelemetry

        for s in cands(env, agent, floor.copy(), SearchTelemetry(), None):
            assert s.n_candidates == len(s.eligible_cols)  # E + 1 minus the floor's row
            assert all(not np.array_equal(r, floor[s.stream_index]) for r in s.rows)
        env.step(ROUND_ROBIN(env, agent))
        if seen >= 6:
            break
    assert seen >= 6


@needs_atlas
def test_bound_in_a_report_ub_equals_d2_on_deterministic_future() -> None:
    """On a deterministic-future world the clairvoyant search equals the teacher action for
    action (D5), and the bound is flagged as hidden information, apart from the teacher arms
    (scenario "Bound in a report")."""
    view = ta.world_view(_entry(_atlas(), CYCLIC_WORLD))
    cfg = config_from_record(view["record"])
    assert deterministic_future(cfg)
    seed = ta.first_confirm_seed(view)
    trained = lambda e, a: np.zeros(e.spec_.k_stream * e.spec_.num_producers,  # noqa: E731
                                    dtype=np.float32)

    def play(source: object) -> tuple[list[bytes], list[tuple]]:
        search = BoundedScenarioOrderer(
            task_id=view["world"], episode_seed=seed, prod=PROD, depth=2, base=ROUND_ROBIN,
            scenarios=source, candidates=WholeOrderCandidates(trained),  # type: ignore[arg-type]
            record_decisions=True, max_search_actions=12)
        actions: list[bytes] = []

        def order(env: object, agent: str) -> np.ndarray:
            a = search(env, agent)
            actions.append(np.asarray(a, dtype=np.float32).tobytes())
            return a

        run_episode(joint_stack_env(cfg, 4), PROD, order, seed)
        return actions, [(d.cycle, d.agent, d.searched, d.candidates, d.scenarios,
                          d.rollouts, d.overrides, d.capped) for d in search.decisions]

    d2, ub = play(DeterministicSource()), play(ClairvoyantSource())
    assert d2 == ub
    assert sum(r[4] for r in d2[1]) > 0  # it searched
    assert ClairvoyantSource().hidden_information and not DeterministicSource().hidden_information
    arm = ta.teacher_arm(ta.UB)
    assert arm["hidden_information"] and arm["side"] == ta.BOUND and arm["side"] != ta.TEACHER
    assert ta.teacher_arm(ta.D2)["side"] == ta.TEACHER
    assert not ta.teacher_arm(ta.D2)["hidden_information"]
    assert all(c.startswith("clairvoyant_") for c in ta.RATIO_COLUMNS if "ub" in c)
    # `ub` is planned only where the future is hidden
    shocked = ta.world_view(_entry(_atlas(), SHOCKED_WORLD))
    assert [a["subject"] for a in ta.world_arms(view, [], ta.TEACHER_ARMS)] == [
        "teacher/floor", "teacher/d2"]
    assert ta.world_arms(shocked, [], ta.TEACHER_ARMS)[-1]["subject"] == "bound/ub"


@needs_atlas
def test_unresolved_decision_acts_as_round_robin(orderer_ckpt: str,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Every cycle unresolved -> every teacher decision is the floor's: the d2 episode equals
    the floor episode, each decision row records the unresolved posterior, none searched."""
    view = ta.world_view(_entry(_atlas(), SHOCKED_WORLD))
    floor_row, _ = ta.run_teacher_episode(_job(view, ta.FLOOR, orderer_ckpt, 6))
    monkeypatch.setattr(ShockPosterior, "sample",
                        lambda self, h, rng: PosteriorSample(h.cycle if hasattr(h, "cycle")
                                                             else 0, (), 0, 0, True))
    row, decisions = ta.run_teacher_episode(_job(view, ta.D2, orderer_ckpt, 6))
    assert decisions and all(d["unresolved"] is True for d in decisions)
    assert all(d.get("searched_streams") is None and d["acceptance_rate"] == 0.0
               for d in decisions)
    assert row["unresolved_fraction"] == "1.0" and row["searched_streams"] == "0"
    for col in ("units", "demand_digest", "shock_digest", "fulfilled_units"):
        assert row[col] == floor_row[col]


def test_exp0042_archive_behaviour_unchanged() -> None:
    """The default floor, candidates and scenario source decide exactly as before task 6.2 on
    one archived task (depth 1, horizon 12, the same seed)."""
    rec = TaskRecord.from_json_line(S29)
    planner = BoundedScenarioOrderer(task_id=rec.task_id, episode_seed=3299, prod=PROD, depth=1)
    h = hashlib.sha256()

    def order(env: object, agent: str) -> np.ndarray:
        a = planner(env, agent)
        h.update(f"{env.current_cycle}|{agent}|".encode())  # type: ignore[attr-defined]
        h.update(np.asarray(a, dtype=np.float32).tobytes())
        return a

    stats = run_episode(joint_stack_env(replay(rec), 12), PROD, order, 3299)
    assert h.hexdigest() == S29_GOLDEN
    assert stats["consumer_completed_units"] == 39
    assert planner.telemetry.override_streams == 2
    assert planner.telemetry.rollout_calls == 1576


def test_traced_env_refused() -> None:
    """DIAGNOSTICS 22: the routing tracer's `complete_cycle` hook is shared by a deepcopy, so
    the search refuses a traced env before any fork, and the live env is left untouched."""
    from tokenomics.eval.routing_trace import RoutingTracer

    rec = TaskRecord.from_json_line(S29)
    env = joint_stack_env(replay(rec), 4)
    RoutingTracer().attach(env)
    env.reset(seed=3299)
    fork = copy.deepcopy(env)
    assert fork.system.complete_cycle is env.system.complete_cycle  # the hazard itself
    planner = BoundedScenarioOrderer(task_id=rec.task_id, episode_seed=3299, prod=PROD, depth=1)
    for agent in env.agent_iter(max_iter=10_000):
        if env._role[agent][0] == "prod":
            env.step(PROD(env, agent))
            continue
        cycle = env.current_cycle
        with pytest.raises(ValueError, match="DIAGNOSTICS 22"):
            planner(env, agent)
        assert env.current_cycle == cycle and env.agent_selection == agent
        assert planner.telemetry.rollout_calls == 0
        break


# --- the driver ---


def _audit(tmp_path: Path, atlas: Path, ckpt: str, out: str, **kw: object) -> int:
    return ta.run_teacher_audit(
        atlas_path=atlas, out_dir=tmp_path / out, orderer_ckpt=ckpt,
        roster=load_roster(str(ta.DEFAULT_ROSTER)), dry_run=True, log=lambda s: None, **kw)


@needs_atlas
def test_digest_mismatch_in_the_audit(tmp_path: Path, orderer_ckpt: str) -> None:
    """Scenario "Digest mismatch in the audit": a teacher episode whose demand digest differs
    from the floor's on the same world-episode fails the run, naming world, episode and arm."""
    path = _manifest(tmp_path)

    def tampered(job: dict) -> dict:
        row = ta.run_audit_job(job)
        if job["subject"] == "teacher/d2":
            row["demand_digest"] = "0" * 64
        return row

    with pytest.raises(DigestMismatch, match=rf"{SHOCKED_WORLD} episode 1295251592: teacher/d2"):
        _audit(tmp_path, path, orderer_ckpt, "mismatch", worlds=[SHOCKED_WORLD],
               arms=["floor", "d2"], baseline=[], horizon=6, shock_profile="shocks-i-smoke",
               max_search_actions=1, run_job_fn=tampered)
    spec = json.loads((tmp_path / "mismatch" / "spec.json").read_text())
    assert spec["status"] == "invalid"
    rows = list(csv.DictReader((tmp_path / "mismatch" / "contract_failures.csv").open()))
    assert [r["check"] for r in rows] == ["digest_pairing"]


@needs_atlas
def test_real_run_refused(tmp_path: Path, orderer_ckpt: str) -> None:
    path = _manifest(tmp_path)
    with pytest.raises(ValueError, match="registered EXP-0052|source commit|--worlds"):
        ta.run_teacher_audit(atlas_path=path, out_dir=tmp_path / "real",
                             orderer_ckpt=orderer_ckpt, roster=load_roster(str(ta.DEFAULT_ROSTER)),
                             worlds=[SHOCKED_WORLD], log=lambda s: None)
    with pytest.raises(ValueError, match="hidden state"):
        _audit(tmp_path, path, orderer_ckpt, "hit", worlds=["ws-atlas-v1-12-0321"])


def _row(world: str, subject: str, units: float, episode: int = 1) -> dict[str, str]:
    return {"world": world, "episode": str(episode), "subject": subject, "replicate": "0",
            "units": repr(float(units)), "dead": "False", "bound_b2": "500.0",
            "bound_base": "400.0", "ordered_units": "", "fulfilled_units": ""}


def _ratios(cls: str, d2: float, ub: float | None, *, bar: float = 100.0,
            reach: float = 150.0, nf: float = 10.0) -> dict:
    """`world_ratios` on one synthetic world-episode: a blind bar arm, a reach arm, the floor,
    `d2` and (unless None) `ub`."""
    w = "w"
    arms = [{"subject": "x/bar", "side": "blind", "random": False},
            {"subject": "x/reach", "side": "reach", "random": False}]
    dicts = [_row(w, "x/bar", bar), _row(w, "x/reach", reach), _row(w, "teacher/floor", bar),
             _row(w, "teacher/d2", d2)] + ([_row(w, "bound/ub", ub)] if ub is not None else [])
    view = {"world": w, "class": cls, "size": 12, "choice": "dense", "pattern": "burst",
            "shocks": True, "noise_floor": nf}
    (out,) = ta.world_ratios(dicts, {w: view}, arms)
    return out


def test_world_ratios_readings() -> None:
    """D5's per-world readings: open = `d2` beats the bar by MORE than the noise floor, else
    fork-limited when `ub` does, else ceiling-limited; hard = `d2` against the best reach arm
    (not the bar)."""
    at_floor = _ratios("open", 110, None)
    assert at_floor["d2_beats_bar"] is False and at_floor["reading"] == "ceiling_limited"
    assert at_floor["bar"] == 100 and at_floor["reach"] == 150 and at_floor["ceiling"] == 400
    assert at_floor["d2_minus_bar"] == 10 and at_floor["d2_over_bar"] == 1.1
    assert at_floor["clairvoyant_ub_units"] is None
    assert _ratios("open", 111, None)["reading"] == "search_headroom"
    assert _ratios("open", 111, 105)["reading"] == "search_headroom"
    fork = _ratios("open", 105, 120)
    assert fork["reading"] == "fork_limited" and fork["clairvoyant_ub_beats_bar"] is True
    assert fork["clairvoyant_ub_hidden_information"] is True
    assert _ratios("open", 105, 110)["reading"] == "ceiling_limited"
    assert _ratios("open", 105, 105)["reading"] == "ceiling_limited"
    hard = _ratios("hard", 120, None)
    assert hard["reading"] == "candidate_or_depth_limited" and hard["d2_ge_reach"] is False
    assert hard["d2_beats_bar"] is True  # beating the bar is not the hard reading
    assert _ratios("hard", 150, None)["reading"] == "beyond_trained"
    assert _ratios("hard", 151, 90)["reading"] == "beyond_trained"
    assert _ratios("open", 90, None)["d2_below_bar"] is True
    assert _ratios("open", 115, None)["d2_clears_margin_ref"] is True  # shocked: 1.10
    assert _ratios("open", 109, None)["d2_clears_margin_ref"] is False


def test_floor_reproduction_check() -> None:
    """`teacher/floor` must reproduce the `entity_planner/round_robin` blind arm's units."""
    rr, fl = "entity_planner/round_robin", "teacher/floor"
    assert ta.floor_failures([_row("w", rr, 100), _row("w", fl, 100)]) == []
    (f,) = ta.floor_failures([_row("w", rr, 100), _row("w", fl, 99), _row("v", fl, 7)])
    assert f.check == ta.FLOOR_CHECK and f.error_type == "FloorMismatch"
    assert "world w episode 1" in f.message


def test_class_readings() -> None:
    def r(cls: str, d2: float, reach: float, reading: str, beats: bool) -> dict:
        return {"world": f"w{d2}", "class": cls, "d2_units": d2, "d2_beats_bar": beats,
                "reading": reading, "d2_below_bar": d2 < 100, "d2_dead": False,
                "floor_dead": False, "d2_unresolved_fraction": 0.25}
    out = ta.class_readings([r("open", 150, 0, "search_headroom", True),
                             r("open", 90, 0, "fork_limited", False),
                             r("hard", 120, 130, "candidate_or_depth_limited", False)])
    assert out["open"]["search_headroom"] and out["open"]["d2_beats_bar"] == 1
    assert out["open"]["readings"]["fork_limited"] == 1
    assert out["hard"]["candidate_or_depth_limited"] == ["w120"]
    assert out["all"]["below_bar"] == ["w90"] and out["all"]["mean_unresolved_fraction"] == 0.25


@needs_atlas
def test_dry_run_layout(tmp_path: Path, orderer_ckpt: str) -> None:
    """`teacher_audit --dry-run`: 1 atlas world at 12 from a dry-run manifest, 1 episode,
    horizon 12, shocks-i-smoke, arms floor + d2, --workers 1 -> the EXP layout in under 2 min,
    teacher_decisions.csv non-empty, marked "not an acceptance read"."""
    path = _manifest(tmp_path)
    out = tmp_path / "EXP-dry"
    started = time.perf_counter()
    code = ta.main(["--atlas", str(path), "--orderer-ckpt", orderer_ckpt, "--out-dir", str(out),
                    "--dry-run", "--worlds", SHOCKED_WORLD, "--arms", "floor,d2",
                    "--horizon", "12", "--shock-profile", "shocks-i-smoke",
                    "--max-search-actions", "20", "--workers", "1"])
    assert time.perf_counter() - started < 120
    assert code == 0
    for f in ("spec.json", "episodes.csv", "ratios.csv", "teacher_decisions.csv",
              "contract_failures.csv", "RESULT.md"):
        assert (out / f).exists(), f
    decisions = list(csv.DictReader((out / "teacher_decisions.csv").open()))
    assert decisions and {d["arm"] for d in decisions} == {"d2"}
    assert {d["scenario_source"] for d in decisions} == {"posterior"}
    assert all(d["acceptance_rate"] != "" for d in decisions)
    assert any(d["seconds_per_searched_stream"] not in ("", None) for d in decisions)
    spec = json.loads((out / "spec.json").read_text())
    assert spec["kind"] == "dry-run" and spec["status"] == "complete"
    assert spec["acceptance_read"] is False
    assert spec["pre_registered"]["teacher_arms"] == ["floor", "d2"]
    assert spec["pre_registered"]["reference_arms"]["ub"]["hidden_information"] is True
    assert "not an acceptance read" in (out / "RESULT.md").read_text()
    episodes = list(csv.DictReader((out / "episodes.csv").open()))
    subjects = {e["subject"] for e in episodes}
    assert {"teacher/floor", "teacher/d2", "entity_planner/round_robin",
            "entity_planner/trained_orderer"} <= subjects and "bound/ub" not in subjects
    ratios = list(csv.DictReader((out / "ratios.csv").open()))
    assert len(ratios) == 1 and ratios[0]["world"] == SHOCKED_WORLD
    assert ratios[0]["bar"] != "" and ratios[0]["d2_units"] != ""
    assert (out / "contract_failures.csv").read_text().count("\n") == 1  # header only


# --- PoC-C: role-swap arms and attribution (task 8.1, design D6) ---


REPO_CKPTS = {
    "producer": REPO / "experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt",
    "orderer": REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"}


def _lfs_present(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs_ckpts = pytest.mark.skipif(not all(_lfs_present(p) for p in REPO_CKPTS.values()),
                                 reason="git-lfs checkpoints absent")


def _units(s: float, a: float, b: float, t: float) -> dict[str, float]:
    return {"S": s, "A": a, "B": b, "T": t}


def test_orderer_swap_closes_more() -> None:
    """c_O 0.6, c_P 0.2 -> orderer; G, interaction by D6's arithmetic."""
    out = ta.role_attribution({"w": _units(1000, 1060, 1020, 1100)})
    assert out["G"] == 100 and out["c_O"] == pytest.approx(0.6)
    assert out["c_P"] == pytest.approx(0.2) and out["reading"] == "orderer"
    assert out["interaction"] == 1100 - 1060 - 1020 + 1000
    assert ta.role_attribution({"w": _units(1000, 1020, 1060, 1100)})["reading"] == "producer"


def test_closures_within_the_band() -> None:
    """|c_O - c_P| < 0.15 is a tie (goes to the user); exactly 0.15 is not."""
    out = ta.role_attribution({"w": _units(1000, 1050, 1040, 1100)})
    assert out["c_O"] == pytest.approx(0.5) and out["c_P"] == pytest.approx(0.4)
    assert out["reading"] == "tie" and out["band"] == 0.15
    edge = ta.role_attribution({"w": _units(1000, 1015, 1000, 1100)})
    assert edge["closure_difference"] == ta.TIE_BAND and edge["reading"] == "orderer"


def test_gap_not_positive_is_undefined() -> None:
    for t in (1000, 900):
        out = ta.role_attribution({"w": _units(1000, 1050, 1040, t)})
        assert out["reading"] == "undefined" and out["G"] == t - 1000
        assert out["c_O"] is None and out["c_P"] is None


def test_attribution_sums_over_worlds_per_size() -> None:
    """G, c_O, c_P are ratios of sums over worlds (not means of per-world ratios); a world
    lacking an arm stays out of the sums and marks the arm missing."""
    out = ta.role_attribution({"w1": _units(1000, 1100, 1000, 1100),
                               "w2": _units(500, 500, 600, 800),
                               "w3": {"S": 1, "A": 2, "B": 3}})
    assert out["worlds"] == 2 and out["G"] == 400
    assert out["c_O"] == pytest.approx(100 / 400) and out["c_P"] == pytest.approx(100 / 400)
    assert out["reading"] == "tie" and out["missing_arms"] == "T"
    dicts = [{"world": "w1", "subject": ta.ROLE_SUBJECT[r], "units": str(u)}
             for r, u in _units(1000, 1060, 1020, 1100).items()]
    rows = ta.attribution_rows(dicts, {"w1": {"size": 12}})
    assert [r["scope"] for r in rows] == ["size", "world"]
    assert rows[0]["reading"] == "orderer" and rows[1]["world"] == "w1"
    inc = ta.role_attribution({"w": {"S": 1.0, "A": 2.0}})
    assert inc["reading"] == "incomplete" and inc["S"] is None and inc["G"] is None


def test_role_arms() -> None:
    arms = {r: ta.role_arm(r) for r in ta.ROLE_ARMS}
    assert arms["S"]["subject"] == "portable" and arms["S"]["producing"] == "portable"
    assert arms["B"]["subject"] == "entity_planner/trained_orderer"
    assert arms["B"]["producing"] == "entity_planner"
    assert {k: v for k, v in arms["T"].items() if k != "role"} == ta.teacher_arm(ta.D2)
    assert arms["A"]["teacher"] == "d2" and arms["A"]["producing"] == "portable"
    assert arms["A"]["subject"] == "teacher/d2_portable"
    assert len({a["subject"] for a in arms.values()}) == 4


@needs_atlas
def test_role_swap_sample_is_poc_a_shocked_as_run() -> None:
    """D6: the shocked worlds of EXP-0053's step-2 sample (3, not the full sample's 6)."""
    assert ta.role_swap_sample(_atlas()) == [
        ("hard", "ws-atlas-v1-12-0060"), ("open", "ws-atlas-v1-12-0111"),
        ("open", "ws-atlas-v1-12-0344")]


def test_teacher_path_fingerprint_reads_only_the_teacher_path() -> None:
    """Driver code of this module outside `TEACHER_PATH_DEFS` does not move the fingerprint; a
    T-path definition or any other package file does."""
    src = (Path(ta.__file__)).read_text()
    files = {"eval/teacher_audit.py": src.encode(), "domain/x.py": b"A = 1\n"}
    ref = ta.teacher_path_fingerprint(files)
    extra = src + "\n\ndef poc_c_only() -> None:\n    pass\n"
    assert ta.teacher_path_fingerprint({**files, "eval/teacher_audit.py": extra.encode()}) == ref
    moved = src.replace("DEPTH = 2", "DEPTH = 3", 1)
    assert ta.teacher_path_fingerprint({**files, "eval/teacher_audit.py": moved.encode()}) != ref
    assert ta.teacher_path_fingerprint({**files, "domain/x.py": b"A = 2\n"}) != ref


def test_exp0053_teacher_path_reference() -> None:
    """The registered EXP-0053 fingerprint is the teacher-path fingerprint of its commit."""
    try:
        files = ta.git_package_files(ta.EXP0053_COMMIT)
    except Exception:  # noqa: BLE001 - no git or no history: nothing to re-derive
        pytest.skip("git history with EXP-0053's commit absent")
    assert ta.teacher_path_fingerprint(files) == ta.EXP0053_TEACHER_PATH


def _source_run(tmp_path: Path, horizon: int, units: str = "2617.0") -> Path:
    src = tmp_path / "EXP-0053"
    (src / "artifacts").mkdir(parents=True)
    spec = {"source": {"commit": ta.EXP0053_COMMIT},
            "pins": {"horizon": horizon, "shock_profile": "shocks-i-smoke",
                     "max_search_actions": 3, "depth": ta.DEPTH,
                     "checkpoints": {"orderer": "o" * 64}, "atlas": {"content_hash": None}}}
    (src / "spec.json").write_text(json.dumps(spec))
    with (src / "artifacts" / "episodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["world", "episode", "subject", "units"])
        w.writeheader()
        w.writerow({"world": SHOCKED_WORLD, "episode": 1295251592, "subject": "teacher/d2",
                    "units": units})
    return src


@needs_atlas
@pytest.mark.parametrize("matches", [True, False])
def test_t_reuse_refused_on_fingerprint_mismatch(tmp_path: Path, matches: bool,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """T rows of the source run stand in for T only when the teacher-path fingerprint equals
    EXP-0053's; on a mismatch T is rerun (a search job) and the reason is in spec.json."""
    path = _manifest(tmp_path)
    atlas_hash = json.loads(path.read_text())["content_hash"]
    src = _source_run(tmp_path, 12)
    spec = json.loads((src / "spec.json").read_text())
    spec["pins"]["atlas"]["content_hash"] = atlas_hash
    (src / "spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(ta, "checked_inputs", lambda out, paths, log: {
        r: {"path": str(p), "sha256": ("o" if r == "orderer" else "p") * 64}
        for r, p in paths.items()})
    monkeypatch.setattr(ta, "teacher_path_fingerprint", lambda files: (
        ta.EXP0053_TEACHER_PATH if matches else "f" * 64))
    seen: list[dict] = []

    def fake(job: dict) -> dict:
        seen.append(job)
        return ta.job_failure(job, RuntimeError("not run")).to_dict()

    out = tmp_path / "EXP-c"
    ta.run_role_swap_audit(atlas_path=path, out_dir=out, prod_ckpt="p.pt", orderer_ckpt="o.pt",
                           horizon=12, worlds=[SHOCKED_WORLD], arms=["S", "T"],
                           shock_profile="shocks-i-smoke", max_search_actions=3,
                           reuse_t=src, dry_run=True, run_job_fn=fake, log=lambda m: None)
    t = next(j for j in seen if j["subject"] == "teacher/d2")
    reuse = json.loads((out / "spec.json").read_text())["t_reuse"]
    if matches:
        assert t["reused_row"]["units"] == "2617.0" and "teacher" not in t
        assert reuse["reused"] and reuse["problems"] == []
    else:
        assert "reused_row" not in t and t["teacher"] == "d2"
        assert not reuse["reused"] and any("fingerprint" in p for p in reuse["problems"])


def test_t_reuse_problems_pins() -> None:
    spec = {"source": {"commit": ta.EXP0053_COMMIT},
            "pins": {"horizon": 100, "shock_profile": None, "max_search_actions": None,
                     "depth": ta.DEPTH, "checkpoints": {"orderer": "o"},
                     "atlas": {"content_hash": "h"}}}
    kw = dict(horizon=100, shock_profile=None, max_search_actions=None, orderer_sha="o",
              atlas_hash="h")
    assert ta.t_reuse_problems(spec, ta.EXP0053_TEACHER_PATH, **kw) == []
    assert any("horizon" in p for p in ta.t_reuse_problems(
        spec, ta.EXP0053_TEACHER_PATH, **{**kw, "horizon": 12}))
    assert any("orderer" in p for p in ta.t_reuse_problems(
        spec, ta.EXP0053_TEACHER_PATH, **{**kw, "orderer_sha": "x"}))
    other = {**spec, "source": {"commit": "a" * 40}}
    assert any("commit" in p for p in ta.t_reuse_problems(other, ta.EXP0053_TEACHER_PATH, **kw))
    assert any("fingerprint" in p for p in ta.t_reuse_problems(spec, "f" * 64, **kw))
    for key, value, word in (("shock_profile", "shocks-i-smoke", "shock profile"),
                             ("max_search_actions", 3, "search cap"),
                             ("atlas_hash", "g", "atlas")):
        got = ta.t_reuse_problems(spec, ta.EXP0053_TEACHER_PATH, **{**kw, key: value})
        assert len(got) == 1 and word in got[0], (key, got)
    deeper = {**spec, "pins": {**spec["pins"], "depth": ta.DEPTH + 1}}
    got = ta.t_reuse_problems(deeper, ta.EXP0053_TEACHER_PATH, **kw)
    assert len(got) == 1 and "depth" in got[0]


@needs_atlas
def test_role_swap_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unshocked world is refused; a real run needs the registered checkpoints."""
    path = _manifest(tmp_path)
    monkeypatch.setattr(ta, "checked_inputs", lambda out, paths, log: {
        r: {"path": str(p), "sha256": "0" * 64} for r, p in paths.items()})
    kw = dict(atlas_path=path, out_dir=tmp_path / "o", prod_ckpt="p.pt", orderer_ckpt="o.pt")
    with pytest.raises(ValueError, match="no shocks"):
        ta.run_role_swap_audit(**kw, worlds=[CYCLIC_WORLD], dry_run=True)
    with pytest.raises(ValueError, match="checkpoint .* is not the registered"):
        ta.run_role_swap_audit(**kw, source_commit="a" * 40)


@needs_atlas
def test_role_swap_real_run_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real PoC-C run needs the registered atlas, a 40-hex commit, horizon 100 and no dry-run
    filter: each one alone is refused, naming it."""
    monkeypatch.setattr(ta, "checked_inputs", lambda out, paths, log: {
        r: {"path": str(p), "sha256": ta.ROLE_CHECKPOINTS[r]} for r, p in paths.items()})
    kw = dict(atlas_path=ATLAS_V2, out_dir=tmp_path / "o", prod_ckpt="p.pt",
              orderer_ckpt="o.pt", source_commit="a" * 40)
    cases = [({"worlds": [SHOCKED_WORLD]}, "--worlds is set"),
             ({"max_worlds": 1}, "--max-worlds is set"),
             ({"arms": ["S"]}, "--arms is set"),
             ({"shock_profile": "shocks-i-smoke"}, "--shock-profile is set"),
             ({"max_search_actions": 3}, "--max-search-actions is set"),
             ({"horizon": 12}, "horizon 12 != 100"),
             ({"source_commit": ""}, "no full 40-hex source commit"),
             ({"atlas_path": _manifest(tmp_path)}, "not the registered EXP-0052")]
    for edit, msg in cases:
        with pytest.raises(ValueError, match="not a valid PoC-C run") as err:
            ta.run_role_swap_audit(**{**kw, **edit})
        assert msg in str(err.value), (edit, str(err.value))
        inner = str(err.value).split("(", 1)[1].rsplit("); pass", 1)[0]
        assert ";" not in inner, str(err.value)  # that problem alone


@needs_atlas
def test_arm_a_plays_the_portable_producer(tmp_path: Path, orderer_ckpt: str,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Arm A's episode and the search's rollouts run the job's portable producer; T's run
    `entity_planner` and never load a producer checkpoint."""
    import tokenomics.eval.bounded_rollout_orderer as bro

    calls = {"producer": 0}
    real = ta.trained_policy

    def portable(env: object, agent: str):  # noqa: ANN202 - a Policy
        calls["producer"] += 1
        return PROD(env, agent)

    def spy_policy(job: dict, role: str):  # noqa: ANN202
        return portable if role == "producer" else real(job, role)

    seen: list[object] = []
    orig = bro.BoundedScenarioOrderer

    class Spy(orig):  # type: ignore[misc, valid-type]
        def __init__(self, *a: object, **k: object) -> None:
            seen.append(k["prod"])
            super().__init__(*a, **k)

    monkeypatch.setattr(ta, "trained_policy", spy_policy)
    monkeypatch.setattr(bro, "BoundedScenarioOrderer", Spy)
    view = ta.world_view(_entry(_atlas(), SHOCKED_WORLD))
    e = ta.world_entry(view, "shocks-i-smoke")
    pins = {"orderer": {"path": orderer_ckpt, "sha256": sha256_file(orderer_ckpt)},
            "producer": {"path": "p.pt", "sha256": "0" * 64}}
    for role in (ta.ROLE_A, ta.ROLE_T):
        job = build_jobs([e], [ta.role_arm(role)], 1, 6, None, pins)[0]
        job["teacher"] = ta.D2
        job["max_search_actions"] = 2
        before = calls["producer"]
        ta.run_teacher_episode(job, max_search_actions=2)
        if role == ta.ROLE_A:
            assert seen[-1] is portable and calls["producer"] > before
        else:
            assert seen[-1] is PROD and calls["producer"] == before


@needs_atlas
@needs_ckpts
def test_role_swap_dry_run_layout(tmp_path: Path) -> None:
    """`--poc c --dry-run`, arms S, A, B: 1 shocked atlas world at 12 from a dry-run manifest, 1
    episode, horizon 12, shocks-i-smoke, --workers 1 -> zero contract failures in under 2 min,
    checkpoint SHA-256 pins in spec.json, attribution.csv written (incomplete without T)."""
    path = _manifest(tmp_path)
    out = tmp_path / "EXP-c-dry"
    started = time.perf_counter()
    code = ta.main(["--poc", "c", "--atlas", str(path), "--prod-ckpt", str(REPO_CKPTS["producer"]),
                    "--orderer-ckpt", str(REPO_CKPTS["orderer"]), "--out-dir", str(out),
                    "--dry-run", "--worlds", SHOCKED_WORLD, "--arms", "S,A,B",
                    "--horizon", "12", "--shock-profile", "shocks-i-smoke",
                    "--max-search-actions", "3", "--workers", "1"])
    assert time.perf_counter() - started < 120
    assert code == 0
    assert (out / "contract_failures.csv").read_text().count("\n") == 1  # header only
    spec = json.loads((out / "spec.json").read_text())
    assert spec["kind"] == "dry-run" and spec["status"] == "complete"
    assert spec["pins"]["checkpoints"] == ta.ROLE_CHECKPOINTS
    assert {r: c["sha256"] for r, c in spec["inputs"].items()} == ta.ROLE_CHECKPOINTS
    episodes = list(csv.DictReader((out / "episodes.csv").open()))
    assert {e["subject"] for e in episodes} == {"portable", "teacher/d2_portable",
                                                "entity_planner/trained_orderer"}
    decisions = list(csv.DictReader((out / "teacher_decisions.csv").open()))
    assert decisions and {d["subject"] for d in decisions} == {"teacher/d2_portable"}
    rows = list(csv.DictReader((out / "attribution.csv").open()))
    assert rows[0]["scope"] == "size" and rows[0]["reading"] == "incomplete"
    assert rows[0]["missing_arms"] == "T"
    assert "not an acceptance read" in (out / "RESULT.md").read_text()


def test_teacher_path_defs_closed() -> None:
    """Every module-level name of `teacher_audit.py` loaded by a `TEACHER_PATH_DEFS` definition
    is itself in `TEACHER_PATH_DEFS` (transitively), so the fingerprint cannot miss one."""
    import ast

    tree = ast.parse(Path(ta.__file__).read_text())
    defs = {n: node for node in tree.body for n in ta._defined_names(node)}
    closure, todo = set(), list(ta.TEACHER_PATH_DEFS)
    while todo:
        name = todo.pop()
        if name in closure or name not in defs:
            continue
        closure.add(name)
        todo += [x.id for x in ast.walk(defs[name])
                 if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id in defs]
    assert closure <= set(ta.TEACHER_PATH_DEFS), sorted(closure - set(ta.TEACHER_PATH_DEFS))


# --- DIAGNOSTICS 19 split check (task 11.2, design D13) ---

from collections.abc import Callable  # noqa: E402

from tokenomics.env.allocator import allocate_suborders  # noqa: E402
from tokenomics.eval.baselines import Policy  # noqa: E402
from tokenomics.eval.roster import build_roster_v2  # noqa: E402
from tokenomics.eval.routing_diagnostics import (  # noqa: E402
    TRAINED_WRAPPERS,
    _slots,
    decoded_units,
    ordering_forced_whole,
    ordering_split_rotating,
    stable_hash,
)

# Rows of PoC-C's S and B arms (EXP-0054) on SHOCKED_WORLD, horizon 12, shocks-i-smoke, the
# registered checkpoints, recorded before task 11.2 (commit 8208a09): sha256 of the sorted row.
ROLE_ROW_GOLDEN = {"S": "b0ada33c45d43fae63205affc00e9ca9ec7cd099bb864e42da64d6deb938b484",
                   "B": "42e71b5a99d294139798d96a67f560be60e8a933226f45572628d2e356bc1a4f"}


def _random_scores(seed: int) -> Policy:
    """A Policy with fresh normal scores per call."""
    rng = np.random.default_rng(seed)

    def policy(env: object, agent: str) -> np.ndarray:
        sp = env.spec_  # type: ignore[attr-defined]
        return rng.normal(0.0, 3.0, sp.k_stream * sp.num_producers).astype(np.float32)

    return policy


def _slot_units(env: object, agent: str, action: np.ndarray) -> dict[int, list[int]]:
    sp = env.spec_  # type: ignore[attr-defined]
    raw = np.asarray(action, dtype=np.float64).reshape(sp.k_stream, sp.num_producers)
    return {k: decoded_units(raw[k, elig], q) for k, _, q, elig in _slots(env, agent) if elig}


def _play_checked(check: Callable[[object, str], int], horizon: int = 8,
                  order_policy: Policy = ROUND_ROBIN) -> int:
    """Play SHOCKED_WORLD (shocks-i-smoke) with `entity_planner` and an orderer that calls
    `check(env, agent)` and routes `order_policy`; returns the number of checked slots."""
    view = ta.world_view(_entry(_atlas(), SHOCKED_WORLD))
    e = ta.world_entry(view, "shocks-i-smoke")
    job = build_jobs([e], [ta.split_arm(ta.SPLIT_RR, load_roster(ta.DEFAULT_ROSTER))], 1,
                     horizon, None)[0]
    env = joint_stack_env(ta.job_config(job), horizon)
    n = [0]

    def order(env_: object, agent: str) -> np.ndarray:
        n[0] += check(env_, agent)
        return order_policy(env_, agent)

    run_episode(env, PROD, order, seed=job["episode"], action_seed=None)
    return n[0]


def test_decoded_units_match_decode() -> None:
    """`decoded_units` is the decode's own split (softmax + largest remainder)."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        n, q = int(rng.integers(1, 8)), int(rng.integers(1, 20))
        scores = rng.normal(0, 3, n)
        subs = allocate_suborders([f"p{i}" for i in range(n)], scores, np.ones(n, bool), q, q)
        got = {s.producer_id: s.quantity for s in subs}
        assert decoded_units(scores, q) == [got.get(f"p{i}", 0) for i in range(n)]


@needs_atlas
def test_forced_whole_never_splits_and_picks_argmax() -> None:
    """`trained_whole`: every slot's order goes whole to the argmax of the base's scores over
    its eligible producers; ties go to the lowest index."""
    base = _random_scores(0)
    whole = ordering_forced_whole(lambda env, agent: base_action[0])
    base_action = [None]

    def check(env: object, agent: str) -> int:
        base_action[0] = base(env, agent)
        sp = env.spec_  # type: ignore[attr-defined]
        raw = np.asarray(base_action[0], np.float64).reshape(sp.k_stream, sp.num_producers)
        out = _slot_units(env, agent, whole(env, agent))
        for k, _, q, elig in _slots(env, agent):
            if elig:
                want = [0] * len(elig)
                want[int(np.argmax(raw[k, elig]))] = q
                assert out[k] == want
        return len(out)

    assert _play_checked(check) > 20
    tie = ordering_forced_whole(lambda env, agent: np.zeros(
        env.spec_.k_stream * env.spec_.num_producers, np.float32))

    def check_tie(env: object, agent: str) -> int:
        out = _slot_units(env, agent, tie(env, agent))
        assert all(u[0] == sum(u) for u in out.values())
        return len(out)

    assert _play_checked(check_tie) > 20


@needs_atlas
def test_split_rotating_holds_profile_and_rotation() -> None:
    """`trained_split_rotating` keeps the base's split profile (the multiset of part sizes) and
    `round_robin`'s receiver rule: the parts, largest first, go to `round_robin`'s pick and then
    the next producers in rotation order, and a whole base is routed exactly as `round_robin`
    routes it."""
    base = _random_scores(1)
    base_action = [None]
    rotating = ordering_split_rotating(lambda env, agent: base_action[0])
    whole_then_rotating = ordering_split_rotating(
        ordering_forced_whole(lambda env, agent: base_action[0]))
    split_seen = [0]

    def check(env: object, agent: str) -> int:
        base_action[0] = base(env, agent)
        mine = _slot_units(env, agent, base_action[0])
        out = _slot_units(env, agent, rotating(env, agent))
        rr = _slot_units(env, agent, ROUND_ROBIN(env, agent))
        cycle = env.system.cycle_counter  # type: ignore[attr-defined]
        for k, _, _, elig in _slots(env, agent):  # type: ignore[arg-type]
            if not elig:
                continue
            n = len(elig)
            start = (cycle + stable_hash(agent, k)) % n
            assert int(np.argmax(rr[k])) == start
            profile = sorted(mine[k], reverse=True)
            assert [out[k][(start + i) % n] for i in range(n)] == profile
            split_seen[0] += sum(u > 0 for u in mine[k]) > 1
        np.testing.assert_array_equal(whole_then_rotating(env, agent), ROUND_ROBIN(env, agent))
        return len(mine)

    assert _play_checked(check) > 20 and split_seen[0] > 5


@needs_atlas
def test_lead_receivers_concurrent_spread_and_rotation() -> None:
    """`LeadReceivers` only reads (the episode equals an unwrapped one), and it sees both sides
    of the receiver rule: a constant-score argmax sends every concurrent order to one producer
    and never rotates (top share 1, change 0); `round_robin` spreads and rotates."""
    const = ordering_forced_whole(lambda env, agent: np.zeros(
        env.spec_.k_stream * env.spec_.num_producers, np.float32))
    cells = {}
    for name, policy in (("const", const), ("rr", ROUND_ROBIN)):
        lead = ta.LeadReceivers(policy)
        units = []
        for order in (lead, policy):
            view = ta.world_view(_entry(_atlas(), SHOCKED_WORLD))
            e = ta.world_entry(view, "shocks-i-smoke")
            job = build_jobs([e], [ta.split_arm(ta.SPLIT_RR, load_roster(ta.DEFAULT_ROSTER))],
                             1, 10, None)[0]
            env = joint_stack_env(ta.job_config(job), 10)
            run_episode(env, PROD, order, seed=job["episode"], action_seed=None)
            units.append(env.episode_stats["consumer_fulfilled_units"])
        assert units[0] == units[1]
        cells[name] = lead.cells()
    assert set(cells["rr"]) == {c for c in ta.LEAD_COLUMNS}
    assert int(cells["const"]["lead_concurrent_groups_consumer"]) > 0
    assert float(cells["const"]["lead_concurrent_top_share_consumer"]) == 1.0
    assert float(cells["const"]["lead_receiver_change_share_consumer"]) == 0.0
    assert float(cells["rr"]["lead_concurrent_top_share_consumer"]) < 0.9
    assert float(cells["rr"]["lead_receiver_change_share_consumer"]) > 0.5


def test_split_wrappers_outside_the_roster() -> None:
    from tokenomics.eval.baselines import ORDERING_BASELINES

    assert set(TRAINED_WRAPPERS) == {ta.TRAINED_WHOLE, ta.TRAINED_SPLIT_ROTATING}
    assert not set(TRAINED_WRAPPERS) & (set(DIAGNOSTIC_ROUTERS) | set(ORDERING_BASELINES))


def test_split_arms() -> None:
    """RR / RW are the atlas confirm's arms as EXP-0053 ran them (RW R = 2); TR is PoC-C's B."""
    table = {a: ta.split_arm(a, load_roster(ta.DEFAULT_ROSTER)) for a in ta.SPLIT_ARMS}
    assert {a: t["subject"] for a, t in table.items()} == ta.SPLIT_SUBJECT
    assert table["TR"]["subject"] == ta.ROLE_SUBJECT[ta.ROLE_B]
    assert table["RW"]["random"] and table["RW"]["replicates"] == 2
    assert not any(table[a]["random"] for a in ("TR", "TW", "RS", "RR"))
    assert all(table[a]["trained"] for a in ("TR", "TW", "RS"))
    assert not table["RR"]["trained"] and not table["RW"]["trained"]


def _split_jobs(ckpt: str, horizon: int = 12) -> list[dict]:
    view = ta.world_view(_entry(_atlas(), SHOCKED_WORLD))
    e = ta.world_entry(view, "shocks-i-smoke")
    table = [ta.split_arm(a, load_roster(ta.DEFAULT_ROSTER)) for a in ta.SPLIT_ARMS]
    jobs = build_jobs([e], table, 1, horizon, None,
                      {"orderer": {"path": ckpt, "sha256": sha256_file(ckpt)}})
    for j in jobs:
        j["split_arm"] = next(a["split_arm"] for a in table if a["subject"] == j["subject"])
    return jobs


@needs_atlas
def test_split_arms_deterministic_per_seed(orderer_ckpt: str) -> None:
    """Every arm's row is identical when its job is played twice (RW per replicate seed)."""
    jobs = _split_jobs(orderer_ckpt)
    assert len(jobs) == 6
    for j in jobs:
        first, second = ta.run_split_job(dict(j)), ta.run_split_job(dict(j))
        assert "units" in first and first == second, j["subject"]


@needs_atlas
@needs_ckpts
def test_role_arms_unchanged_golden() -> None:
    """PoC-C's S and B rows equal the pre-11.2 rows; the split check's TR row equals B on every
    gate and trace column (the lead-receiver wrapper only reads)."""
    view = ta.world_view(_entry(_atlas(), SHOCKED_WORLD))
    e = ta.world_entry(view, "shocks-i-smoke")
    pins = {r: {"path": str(p), "sha256": sha256_file(p)} for r, p in REPO_CKPTS.items()}
    rows = {}
    for role in ("S", "B"):
        job = build_jobs([e], [ta.role_arm(role)], 1, 12, None, pins)[0]
        rows[role] = ta.run_audit_job(job)
        digest = hashlib.sha256(json.dumps(rows[role], sort_keys=True).encode()).hexdigest()
        assert digest == ROLE_ROW_GOLDEN[role], role
    tr = next(j for j in _split_jobs(str(REPO_CKPTS["orderer"])) if j["split_arm"] == "TR")
    split_row = ta.run_split_job(tr)
    shared = [c for c in ta.EPISODE_COLUMNS if c in rows["B"]]
    assert len(shared) > 40
    assert {c: split_row[c] for c in shared} == {c: rows["B"][c] for c in shared}


def _world_units(tr: float, tw: float, rs: float, rr: float) -> dict[str, float]:
    return {"TR": tr, "TW": tw, "RS": rs, "RR": rr, "RW": rr}


@pytest.mark.parametrize("worlds, reading", [
    # whole recovers >= 2/3 of the gap (and > floor) on 2 of 3 worlds -> splitting (f)
    ([(2000, 2600, 2100, 2800), (2000, 2700, 2000, 2900), (2000, 2050, 2000, 2400)], "f"),
    # only the rotation contrast does -> receiver rotation (b)
    ([(2000, 2050, 2600, 2800), (2000, 1900, 2700, 2900), (2000, 2600, 2100, 2800)], "b"),
    ([(2000, 2600, 2700, 2800), (2000, 2700, 2650, 2900), (2000, 2000, 2000, 2400)], "f+b"),
    # each single change recovers half the gap (additive): only both together -> joint
    ([(2000, 2400, 2400, 2800), (2000, 2450, 2450, 2900), (2000, 2200, 2200, 2400)], "joint"),
    # complements: whole alone loses, the split on rotation ~ TR, both together win -> joint
    ([(2000, 1900, 2010, 2800), (2000, 1950, 1990, 2900), (2000, 2000, 2000, 2600)], "joint"),
    # the worlds disagree (f, b, joint) -> neither
    ([(2000, 2700, 2000, 2800), (2000, 2000, 2700, 2800), (2000, 2300, 2300, 2800)], "neither"),
    # the trained orderer does not trail round_robin beyond the floor on 2 worlds
    ([(2000, 2100, 2100, 2100), (2000, 2000, 2000, 2150), (2000, 2600, 2000, 2800)], "no_gap"),
    # a gain under the floor never recovers, even at >= 2/3 of a small gap (so not f)
    ([(2000, 2160, 2000, 2200), (2000, 2160, 2000, 2200), (2000, 2600, 2000, 2800)], "joint"),
])
def test_split_reading(worlds: list[tuple[float, ...]], reading: str) -> None:
    rows = [{"scope": "world", **ta.split_world(_world_units(*w), 164.3)} for w in worlds]
    assert ta.split_reading(rows) == reading


def test_split_reading_boundaries() -> None:
    """On a whole-number floor: a gap equal to the floor does not clear it; a gain equal to the
    floor does not recover; a gain of exactly 2/3 of the gap does, one unit less does not."""
    floor = 100.0
    at = ta.split_world(_world_units(2000, 2100, 2000, 2100), floor)
    assert not at["gap_clears"] and not at["whole_recovers"] and not at["joint"]
    gain_at_floor = ta.split_world(_world_units(2000, 2100, 2000, 2150), floor)
    assert gain_at_floor["gap_clears"] and not gain_at_floor["whole_recovers"]
    exact = ta.split_world(_world_units(2000, 2200, 2000, 2300), floor)
    assert exact["whole_recovers"] and not exact["joint"]
    short = ta.split_world(_world_units(2000, 2199, 2000, 2300), floor)
    assert not short["whole_recovers"] and short["joint"]


def test_split_world_arithmetic() -> None:
    r = ta.split_world(_world_units(2000, 2600, 2300, 2900), 164.3)
    assert r["gap"] == 900 and r["whole_gain"] == 600 and r["rotation_gain"] == 300
    assert r["whole_recovery"] == pytest.approx(2 / 3) and r["whole_recovers"]
    assert not r["rotation_recovers"]
    assert r["split_cost_at_rotation"] == 600 and r["receiver_gain_whole"] == 300
    assert r["interaction"] == r["split_cost_at_rotation"] - r["whole_gain"] == 0
    assert r["split_cost_at_rotation_share"] == pytest.approx(2 / 3)
    assert r["receiver_gain_whole_share"] == pytest.approx(1 / 3) and not r["joint"]


def _split_dicts(world: str, units: dict[str, float]) -> list[dict[str, str]]:
    return [{"world": world, "episode": "7", "subject": ta.SPLIT_SUBJECT[a], "replicate": "0",
             "units": str(u), "demand_digest": "d", "shock_digest": "s"}
            for a, u in units.items()]


def test_split_check_rows_incomplete_and_sums() -> None:
    dicts = _split_dicts("w1", _world_units(2000, 2600, 2100, 2800))
    dicts += _split_dicts("w2", {"TR": 1, "TW": 2, "RR": 3})
    rows = ta.split_check_rows(dicts, {"w1": 164.3, "w2": 164.3})
    assert [r["scope"] for r in rows] == ["world", "all"]
    assert rows[-1]["reading"] == "incomplete" and rows[-1]["missing_arms"] == "w2:RS"
    rows = ta.split_check_rows(dicts, {"w1": 164.3})
    assert rows[-1]["gap"] == 800 and rows[-1]["whole_recovers"] == 1
    assert rows[-1]["reading"] == "no_gap"  # one world cannot reach 2 of 3


def test_reference_pairing(tmp_path: Path) -> None:
    """A world-episode absent from the reference or with other digests is a problem; units of
    rows both played are compared."""
    mine = _split_dicts("w1", {"TR": 2155, "RR": 2429})
    ref = tmp_path / "exp0053"
    ref.mkdir()
    fields = list(mine[0])
    with (ref / "episodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows([{**mine[0]}, {**mine[1], "units": "2430"}])
    units, problems = ta.reference_pairing(mine, {"exp0053": ref})
    assert problems == []
    assert [(u["subject"], u["equal"]) for u in units] == [
        (ta.SPLIT_SUBJECT["TR"], True), (ta.SPLIT_SUBJECT["RR"], False)]
    other = [{**d, "shock_digest": "x"} for d in mine] + _split_dicts("w9", {"TR": 1})
    _, problems = ta.reference_pairing(other, {"exp0053": ref})
    assert any("w9" in p and "no row" in p for p in problems)
    assert sum("digests differ" in p for p in problems) == 2
    with (ref / "episodes.csv").open("a", newline="") as fh:
        csv.DictWriter(fh, fields).writerow({**mine[0], "replicate": "1", "demand_digest": "e"})
    _, problems = ta.reference_pairing(mine, {"exp0053": ref})
    assert problems == ["exp0053: its rows of w1 episode 7 disagree on digests"]


@needs_atlas
def test_split_check_real_run_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real split check needs the registered atlas and orderer, a 40-hex commit, horizon 100,
    a reference run and no dry-run filter: each one alone is refused, naming it."""
    monkeypatch.setattr(ta, "checked_orderer", lambda out, path, log: {
        "path": path, "sha256": ta.ROLE_CHECKPOINTS["orderer"]})
    kw = dict(atlas_path=ATLAS_V2, out_dir=tmp_path / "o", orderer_ckpt="o.pt",
              roster=load_roster(ta.DEFAULT_ROSTER), source_commit="a" * 40,
              references=[tmp_path / "exp0053"])
    cases = [({"worlds": [SHOCKED_WORLD]}, "--worlds is set"),
             ({"max_worlds": 1}, "--max-worlds is set"),
             ({"arms": ["TR"]}, "--arms is set"),
             ({"shock_profile": "shocks-i-smoke"}, "--shock-profile is set"),
             ({"horizon": 12}, "horizon 12 != 100"),
             ({"source_commit": ""}, "no full 40-hex source commit"),
             ({"references": []}, "no --reference run"),
             ({"roster": build_roster_v2()}, "is not frozen"),
             ({"atlas_path": _manifest(tmp_path)}, "not the registered EXP-0052")]
    for edit, msg in cases:
        with pytest.raises(ValueError, match="not a valid split-check run") as err:
            ta.run_split_check(**{**kw, **edit})
        assert msg in str(err.value), (edit, str(err.value))
        inner = str(err.value).split("(", 1)[1].rsplit("); pass", 1)[0]
        assert ";" not in inner, str(err.value)
    monkeypatch.setattr(ta, "checked_orderer", lambda out, path, log: {
        "path": path, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="orderer checkpoint 000000000000 is not the registered"):
        ta.run_split_check(**kw)
    with pytest.raises(ValueError, match="no shocks"):
        ta.run_split_check(**{**kw, "worlds": [CYCLIC_WORLD], "dry_run": True})


def test_split_check_flags_refused() -> None:
    base = ["--atlas", "a.json", "--orderer-ckpt", "o.pt", "--out-dir", "o"]
    for extra in (["--prod-ckpt", "p.pt"], ["--reuse-t", "r"], ["--reduction", "step2"],
                  ["--max-search-actions", "3"], ["--baseline", ""], ["--replicates", "1"]):
        with pytest.raises(SystemExit):
            ta.main(["--poc", "split", *base, *extra])
    with pytest.raises(SystemExit):
        ta.main(["--poc", "c", *base, "--prod-ckpt", "p.pt", "--reference", "r"])


@needs_atlas
@needs_ckpts
def test_split_check_dry_run_layout(tmp_path: Path) -> None:
    """`--poc split --dry-run`: 1 shocked world, 1 episode, horizon 12, shocks-i-smoke, every
    arm, --workers 1 -> the layout in under 2 min, traced rows with lead-receiver columns, the
    reading row; a second run paired with the first as its reference reproduces every row, and
    one with a tampered reference digest is invalid."""
    path = _manifest(tmp_path)
    out = tmp_path / "EXP-split-dry"
    args = ["--poc", "split", "--atlas", str(path), "--orderer-ckpt", str(REPO_CKPTS["orderer"]),
            "--dry-run", "--worlds", SHOCKED_WORLD, "--horizon", "12",
            "--shock-profile", "shocks-i-smoke", "--workers", "1"]
    started = time.perf_counter()
    assert ta.main([*args, "--out-dir", str(out)]) == 0
    assert time.perf_counter() - started < 120
    assert (out / "contract_failures.csv").read_text().count("\n") == 1
    spec = json.loads((out / "spec.json").read_text())
    assert spec["kind"] == "dry-run" and spec["status"] == "complete"
    assert spec["pins"]["checkpoints"] == {"orderer": ta.ROLE_CHECKPOINTS["orderer"]}
    assert spec["pins"]["sample"][0]["floor"] == pytest.approx(164.33, abs=0.01)
    episodes = list(csv.DictReader((out / "episodes.csv").open()))
    assert sorted((e["arm"], e["replicate"]) for e in episodes) == [
        ("RR", "0"), ("RS", "0"), ("RW", "0"), ("RW", "1"), ("TR", "0"), ("TW", "0")]
    for e in episodes:
        assert e["trace_split_share_consumer"] != "" and e["lead_pairs_consumer"] != ""
        if e["arm"] in ("TW", "RR", "RW"):
            assert float(e["trace_split_share_consumer"]) == 0.0
    manip = list(csv.DictReader((out / "manipulation.csv").open()))
    assert sorted(r["arm"] for r in manip) == ["RR", "RS", "RW", "TR", "TW"]
    check = list(csv.DictReader((out / "split_check.csv").open()))
    assert [r["scope"] for r in check] == ["world", "all"] and check[-1]["reading"]
    assert "not an acceptance read" in (out / "RESULT.md").read_text()

    again = tmp_path / "EXP-split-again"
    assert ta.main([*args, "--out-dir", str(again), "--reference", str(out)]) == 0
    spec = json.loads((again / "spec.json").read_text())
    assert len(spec["reproduction"]) == 6 and all(r["equal"] for r in spec["reproduction"])

    bad = tmp_path / "bad_ref"
    bad.mkdir()
    rows = [{**r, "shock_digest": "0" * 64} for r in csv.DictReader((out / "episodes.csv").open())]
    with (bad / "episodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert ta.main([*args, "--out-dir", str(tmp_path / "EXP-split-bad"),
                    "--reference", str(bad)]) == ta.CONTRACT_INVALID


def test_lead_receivers_concurrent_groups_by_hand() -> None:
    """`lead_concurrent_top_share_*` by hand: groups are per class x cycle x product x eligible
    set, need >= 2 orders and >= 2 eligible producers, and score the most common receiver's
    share; rotation pairs are per stream across cycles."""
    lead = ta.LeadReceivers(ROUND_ROBIN)
    c, p = ta.CONSUMER_CLASS, ta.PRODUCER_CLASS
    # cycle 1, product x, eligible (0, 1): 2:1 -> 2/3 (min would give 1/3)
    for agent, r in (("a", 0), ("b", 0), ("c", 1)):
        lead.record(agent, c, 1, "x", (0, 1), r)
    # same cycle and product, another eligible set: its own group, 1:1 -> 1/2
    lead.record("d", c, 1, "x", (2, 3), 2)
    lead.record("e", c, 1, "x", (2, 3), 3)
    # same producer in two cycles: two single-order groups, not one group of 2 -> no group
    lead.record("f", c, 2, "y", (0, 1), 0)
    lead.record("f", c, 3, "y", (0, 1), 0)
    # one eligible producer: never a group
    lead.record("g", c, 1, "z", (4,), 4)
    lead.record("h", c, 1, "z", (4,), 4)
    # producer orders in the same cycle/product/set: their own class, 1:1 -> 1/2
    lead.record("p1", p, 1, "x", (0, 1), 0)
    lead.record("p2", p, 1, "x", (0, 1), 1)
    cells = lead.cells()
    assert cells["lead_concurrent_groups_consumer"] == "2"
    assert float(cells["lead_concurrent_top_share_consumer"]) == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert cells["lead_concurrent_groups_producer"] == "1"
    assert float(cells["lead_concurrent_top_share_producer"]) == pytest.approx(1 / 2)
    # rotation: only stream f has a pair (cycle 2 -> 3, same receiver), so change share 0
    assert cells["lead_pairs_consumer"] == "1"
    assert float(cells["lead_receiver_change_share_consumer"]) == 0.0
    lead.record("f", c, 4, "y", (0, 1), 1)
    assert float(lead.cells()["lead_receiver_change_share_consumer"]) == pytest.approx(1 / 2)
    # producer-order rotation is its own class: p1 changes, p3 keeps -> 1 of 2
    lead.record("p1", p, 2, "x", (0, 1), 1)
    lead.record("p3", p, 2, "w", (0, 1), 0)
    lead.record("p3", p, 3, "w", (0, 1), 0)
    cells = lead.cells()
    assert cells["lead_pairs_producer"] == "2"
    assert float(cells["lead_receiver_change_share_producer"]) == pytest.approx(1 / 2)
    assert cells["lead_pairs_consumer"] == "2"  # unchanged by producer orders


@needs_atlas
def test_lead_receivers_call_plumbing() -> None:
    """`LeadReceivers.__call__` passes each slot's class, cycle, agent, product, eligible set and
    lead receiver to `record`: on a `round_robin` episode its groups and cells equal an
    independent recomputation, with groups over several cycles and both classes, several
    agents per product and producer-order rotation pairs."""
    from tokenomics.domain.system import order_kind
    from tokenomics.eval.routing_trace import order_class

    lead = ta.LeadReceivers(ROUND_ROBIN)
    seen: list[tuple[str, str, int, str, tuple[int, ...], int]] = []

    def order(env: object, agent: str) -> np.ndarray:
        action = lead(env, agent)
        sp = env.spec_  # type: ignore[attr-defined]
        raw = np.asarray(action, np.float64).reshape(sp.k_stream, sp.num_producers)
        for k, oid, q, elig in _slots(env, agent):  # type: ignore[arg-type]
            if elig and q > 0:
                o = env.system.orders[oid]  # type: ignore[attr-defined]
                lead_r = elig[int(np.argmax(decoded_units(raw[k, elig], q)))]
                seen.append((agent, order_class(order_kind(o)),
                             int(env.system.cycle_counter), o.product,  # type: ignore[attr-defined]
                             tuple(elig), lead_r))
        return action

    view = ta.world_view(_entry(_atlas(), SHOCKED_WORLD))
    e = ta.world_entry(view, "shocks-i-smoke")
    job = build_jobs([e], [ta.split_arm(ta.SPLIT_RR, load_roster(ta.DEFAULT_ROSTER))], 1, 10,
                     None)[0]
    env = joint_stack_env(ta.job_config(job), 10)
    run_episode(env, PROD, order, seed=job["episode"], action_seed=None)
    ref = ta.LeadReceivers(ROUND_ROBIN)
    for rec in seen:
        ref.record(*rec)
    assert lead.groups == ref.groups and lead.cells() == ref.cells()
    keys = list(lead.groups)
    assert len({k[1] for k in keys}) > 1 and {k[0] for k in keys} == {
        ta.CONSUMER_CLASS, ta.PRODUCER_CLASS}
    agents_per_product: dict[str, set[str]] = {}
    for agent, product in lead.last:
        agents_per_product.setdefault(product, set()).add(agent)
    assert max(len(a) for a in agents_per_product.values()) > 1
    cells = lead.cells()
    assert int(cells["lead_pairs_producer"]) > 0 and int(cells["lead_pairs_consumer"]) > 0
    by_cls: dict[str, list[int]] = {ta.CONSUMER_CLASS: [0, 0], ta.PRODUCER_CLASS: [0, 0]}
    last: dict[tuple[str, str], int] = {}
    for agent, cls, _, product, _, r in seen:
        if (agent, product) in last:
            by_cls[cls][0] += 1
            by_cls[cls][1] += int(r != last[(agent, product)])
        last[(agent, product)] = r
    for cls, (pairs, changes) in by_cls.items():
        assert cells[f"lead_pairs_{cls}"] == str(pairs)
        assert float(cells[f"lead_receiver_change_share_{cls}"]) == pytest.approx(changes / pairs)


def test_manipulation_rows() -> None:
    """Realized levels per world and arm, replicate means, and the difference from TR's."""
    def row(arm: str, rep: int, share: float, change: float) -> dict[str, str]:
        return {"world": "w", "subject": ta.SPLIT_SUBJECT[arm], "replicate": str(rep),
                "trace_split_share_consumer": str(share), "trace_split_degree_consumer": "1.0",
                "lead_concurrent_top_share_consumer": "1.0",
                "lead_receiver_change_share_consumer": str(change)}

    rows = ta.manipulation_rows([row("RS", 0, 0.30, 0.8), row("TR", 0, 0.36, 0.44),
                                 row("TW", 0, 0.0, 0.0), row("RW", 0, 0.0, 0.7),
                                 row("RW", 1, 0.0, 0.8)])
    assert [r["arm"] for r in rows] == ["TR", "TW", "RS", "RW"]
    by = {r["arm"]: r for r in rows}
    assert by["RS"]["d_trace_split_share_consumer"] == pytest.approx(-0.06)
    assert by["TW"]["d_lead_receiver_change_share_consumer"] == pytest.approx(-0.44)
    assert by["RW"]["replicates"] == 2
    assert by["RW"]["lead_receiver_change_share_consumer"] == pytest.approx(0.75)
    assert by["TW"]["held_by_construction"].startswith("receiver rule")
    assert by["RS"]["held_by_construction"].startswith("split rule")
    assert by["TR"]["d_trace_split_share_consumer"] == 0.0
