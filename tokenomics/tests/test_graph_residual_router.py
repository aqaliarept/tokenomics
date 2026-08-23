"""EXP-0041 residual router: the confidence gate, route integrity, and closed-loop Gate B.

The deployment claim is a SAFETY claim as much as a performance one: the policy must fall back
to `blend_0.5` byte-for-byte whenever the ensemble is not confident, must never move mass to an
ineligible producer or lose a unit, must keep routing across multiple producers, and must clear
a paired per-episode floor. These tests pin each of those, and check that a result which should
fail cannot pass the gate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenomics.eval import graph_residual_router as rr  # noqa: E402
from tokenomics.eval.baselines import PRODUCING_BASELINES, ordering_blend  # noqa: E402

from test_graph_view import _scale_task, first_contested_decision  # noqa: E402


class _StubGNN(torch.nn.Module):
    """A graph-arm stand-in that scores candidates by a fixed rank pattern, so the confidence
    rule can be tested without training anything."""

    def __init__(self, direction: float = 1.0, offset: float = 0.0, scale: float = 1.0) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.direction, self.offset, self.scale = direction, offset, scale

    def forward(self, batch) -> "torch.Tensor":
        n = int(batch.cand_producer.shape[0])
        v = torch.arange(n, dtype=torch.float32) * self.direction * self.scale + self.offset
        return v


def _agreeing(n_models: int = 3, scale: float = 1.0) -> list:
    return [_StubGNN(1.0, offset=float(i), scale=scale) for i in range(n_models)]


# --- the residual rule ----------------------------------------------------------------------
def test_residual_is_zero_unless_the_seeds_agree_and_the_margin_clears():
    """The gate is unanimity AND margin. Disagreeing seeds, or a margin under the threshold, give
    an exactly-zero residual — which is the blend action byte-for-byte, not a small nudge."""
    up = [np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0]), np.array([-1.0, 0.0, 1.0])]
    conf = rr.Adapter(alpha=0.5, margin_threshold=1.0)
    res, info = rr.ensemble_residual(up, conf)
    assert info["unanimous"] and info["confident"] and info["margin"] == pytest.approx(1.0)
    assert np.allclose(res, 0.5 * np.clip(np.array([-1.0, 0.0, 1.0]), -1, 1))

    split = [np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0, 0.0]), np.array([0.0, 1.0, 2.0])]
    res, info = rr.ensemble_residual(split, conf)
    assert not info["unanimous"] and not info["confident"] and np.all(res == 0.0)

    tight = rr.Adapter(alpha=0.5, margin_threshold=1.5)  # margin 1.0 < threshold
    res, info = rr.ensemble_residual(up, tight)
    assert info["unanimous"] and not info["confident"] and np.all(res == 0.0)

    zero_alpha = rr.Adapter(alpha=0.0, margin_threshold=0.0)
    assert np.all(rr.ensemble_residual(up, zero_alpha)[0] == 0.0)


def test_residual_is_centered_clipped_and_bounded_by_alpha():
    """Per-seed centering means a constant offset on one model cannot tilt the ensemble, and the
    fixed clip of 1.0 bounds the residual by `alpha` however large the raw scores are."""
    base = [np.array([0.0, 4.0]), np.array([0.0, 4.0]), np.array([0.0, 4.0])]
    shifted = [base[0] + 100.0, base[1], base[2] - 7.5]
    cfg = rr.Adapter(alpha=0.25, margin_threshold=0.0)
    assert np.allclose(rr.ensemble_residual(base, cfg)[0], rr.ensemble_residual(shifted, cfg)[0])
    huge = [np.array([0.0, 1e6])] * 3
    res, _ = rr.ensemble_residual(huge, cfg)
    assert np.abs(res).max() <= cfg.alpha + 1e-12
    nonfinite = [np.array([0.0, np.inf])] * 3
    res, info = rr.ensemble_residual(nonfinite, cfg)
    assert np.all(res == 0.0) and info.get("nonfinite")


# --- deploy-time behaviour --------------------------------------------------------------------
def _router(models, adapter, audit=None) -> rr.EnsembleRouter:
    return rr.EnsembleRouter("mpnn_4x64", models, None, adapter=adapter, audit=audit)


def test_policy_matches_blend_exactly_when_the_ensemble_is_not_confident():
    """A non-confident decision (or alpha=0) must reproduce the blend logits bit-for-bit, so the
    floor is always reachable and a mistrained ensemble can only ever cost the confident cases."""
    env, aid, oid, elig = first_contested_decision()
    agent = env.agent_selection
    base = ordering_blend(0.5)
    disagreeing = [_StubGNN(1.0), _StubGNN(-1.0), _StubGNN(1.0)]
    pol = rr.adapter_policy(base, _router(disagreeing, rr.Adapter(0.5, 0.0)))
    assert np.array_equal(pol(env, agent), np.asarray(base(env, agent), dtype=np.float32))
    pol0 = rr.adapter_policy(base, _router(_agreeing(), rr.Adapter(0.0, 0.0)))
    assert np.array_equal(pol0(env, agent), np.asarray(base(env, agent), dtype=np.float32))


def test_residual_touches_only_contested_streams_and_eligible_producers():
    """Structural masking: an ineligible producer's column and any uncontested stream stay at
    exactly zero, so the residual can never create routing mass the allocator would have to
    discard."""
    env, aid, oid, elig = first_contested_decision()
    agent = env.agent_selection
    _, aid_ = env._role[agent]
    view = env.system.ordering_agents[aid_].view(env.system.producer_infos)
    router = _router(_agreeing(), rr.Adapter(0.5, 0.0))
    mat = router.residual(env, agent).reshape(env.spec_.k_stream, env.spec_.num_producers)
    touched = 0
    for k, order_id in enumerate(sorted(view.incoming)):
        if k >= env.spec_.k_stream:
            break
        eligible = view.eligible[order_id]
        for col, pid in enumerate(env.spec_.producer_ids):
            if pid not in eligible or len(eligible) < 2:
                assert mat[k, col] == 0.0
        touched += int(len(eligible) >= 2 and np.any(mat[k] != 0.0))
    assert touched >= 1  # at least the contested stream really was moved
    assert np.isfinite(mat).all()


def test_route_audit_detects_a_collapse_to_a_single_producer():
    """Gate B condition 6 must be able to FAIL. A bounded alpha keeps blend's multi-producer
    split; an absurd alpha collapses routing onto one producer, and the audit counts exactly
    that (which is why the pre-registered grid bounds alpha at 0.5)."""
    env, aid, oid, elig = first_contested_decision()
    agent = env.agent_selection
    base = ordering_blend(0.5)

    safe_audit = rr.RouteAudit()
    rr.adapter_policy(base, _router(_agreeing(), rr.Adapter(0.25, 0.0), safe_audit))(env, agent)
    assert safe_audit.n_decisions >= 1 and safe_audit.violations() == 0

    wild_audit = rr.RouteAudit()
    rr.adapter_policy(base, _router(_agreeing(), rr.Adapter(60.0, 0.0), wild_audit))(env, agent)
    assert wild_audit.multi_producer_violations >= 1
    assert wild_audit.conservation_violations == 0  # the allocator still conserves quantity
    assert wild_audit.eligibility_violations == 0


def test_per_task_diagnostic_arm_cannot_be_deployed():
    """The per-task arm exists only to attribute a pooling effect; deploying it would be an
    untested policy family, so construction refuses it."""
    with pytest.raises(ValueError):
        rr.EnsembleRouter("mpnn_4x64_per_task", _agreeing(), None, adapter=rr.Adapter(0.1, 0.0))
    with pytest.raises(ValueError):
        rr.EnsembleRouter("mpnn_4x64", [], None, adapter=rr.Adapter(0.1, 0.0))


# --- validation-only adapter selection ------------------------------------------------------
def _grid_rows(spec: dict, tasks=("a", "b", "c"), seeds=(0, 1)) -> list[dict]:
    """`spec` maps policy -> {task: (cu, dead)}; expands to per-episode rollout rows."""
    rows = []
    for policy, by_task in spec.items():
        for task in tasks:
            cu, dead = by_task[task]
            for s in seeds:
                rows.append({"task_id": task, "trajectory_seed": s, "policy": policy,
                             "cu": float(cu), "dead": dead, "route_decisions": 5,
                             "route_conservation_violations": 0, "route_eligibility_violations": 0,
                             "route_multi_producer_violations": 0, "route_nonfinite": 0})
    return rows


def test_adapter_selection_rejects_unsafe_pairs_and_breaks_ties_toward_the_smaller_alpha():
    """Rule order: a pair that loses >2% on any task or adds a death is rejected outright, even
    if its mean is the best; among survivors the highest mean wins; a tie within 0.5% goes to the
    smaller alpha."""
    rows = _grid_rows({
        "blend_0.5": {"a": (100, 0), "b": (100, 0), "c": (100, 0)},
        "gnn_a": {"a": (130, 0), "b": (130, 0), "c": (95, 0)},     # -5% on c -> unsafe
        "gnn_b": {"a": (110, 0), "b": (110, 0), "c": (110, 1)},    # extra death -> unsafe
        "gnn_c": {"a": (106, 0), "b": (106, 0), "c": (106, 0)},    # safe, ties with gnn_d
        "gnn_d": {"a": (106, 0), "b": (106, 0), "c": (106, 0)},
    })
    meta = {"gnn_a": {"alpha": 0.5, "margin_threshold": 0.0},
            "gnn_b": {"alpha": 0.25, "margin_threshold": 0.0},
            "gnn_c": {"alpha": 0.5, "margin_threshold": 1.0},
            "gnn_d": {"alpha": 0.1, "margin_threshold": 0.0}}
    out = rr.select_adapter(rows, meta=meta, candidates=list(meta))
    assert out["selected"] == "gnn_d"  # smaller alpha wins the 0.5% tie
    assert out["evaluated"]["gnn_a"]["safe"] is False
    assert out["evaluated"]["gnn_b"]["safe"] is False


def test_adapter_selection_stops_when_nothing_beats_blend():
    """If no pair beats blend on at least 2 of 3 validation tasks, selection returns None and the
    experiment stops BEFORE any final-test rollout."""
    rows = _grid_rows({
        "blend_0.5": {"a": (100, 0), "b": (100, 0), "c": (100, 0)},
        "gnn_a": {"a": (100, 0), "b": (99.5, 0), "c": (101, 0)},
    })
    out = rr.select_adapter(rows, meta={"gnn_a": {"alpha": 0.1, "margin_threshold": 0.0}},
                            candidates=["gnn_a"])
    assert out["selected"] is None and "beats blend" in out["reason"]
    with pytest.raises(ValueError):  # the floor must be measured on the same episodes
        rr.select_adapter([r for r in rows if r["policy"] != "blend_0.5"], candidates=["gnn_a"])


# --- closed-loop Gate B -------------------------------------------------------------------------
def _test_rows(learned: dict, blend: dict, *, violations: int = 0, dead: int = 0,
               seeds_by_task: "dict | None" = None) -> list[dict]:
    rows = []
    for policy, by_task in (("gnn_residual", learned), ("blend_0.5", blend)):
        for task, per_seed in by_task.items():
            seeds = (seeds_by_task or {}).get(task) or list(range(len(per_seed)))
            for s, cu in zip(seeds, per_seed):
                rows.append({"task_id": task, "trajectory_seed": s, "policy": policy,
                             "cu": float(cu),
                             "dead": dead if policy == "gnn_residual" else 0,
                             "route_decisions": 10, "route_nonfinite": 0,
                             "route_conservation_violations": 0,
                             "route_eligibility_violations": 0,
                             "route_multi_producer_violations": (
                                 violations if policy == "gnn_residual" else 0)})
    return rows


def test_gate_b_passes_a_safe_improvement_and_fails_every_individual_shortfall():
    """A genuine, safe improvement passes; each pre-registered failure mode independently flips
    the verdict. `pass` is never a judgement call. (Synthetic tasks, so the registered-scope
    requirement is switched off here — its own test covers it.)"""
    def gate(rows, **kw):
        return rr.gate_b(rows, policy="gnn_residual", require_registered=False, **kw)

    blend = {"a": [100, 100], "b": [100, 100], "c": [100, 100]}
    good = {"a": [110, 110], "b": [108, 108], "c": [106, 106]}
    v = gate(_test_rows(good, blend))
    assert v["pass"] and v["avg_rel_improvement"] == pytest.approx(0.08)

    small = {"a": [102, 102], "b": [102, 102], "c": [102, 102]}  # +2% < 5%
    assert not gate(_test_rows(small, blend))["conditions"]["avg_rel_improvement_5pct"]

    lopsided = {"a": [130, 130], "b": [130, 130], "c": [90, 90]}  # one task -10%
    lop = gate(_test_rows(lopsided, blend))
    assert not lop["pass"] and not lop["conditions"]["no_task_worse_5pct"]

    one_bad_episode = {"a": [140, 85], "b": [120, 120], "c": [115, 115]}  # -15% on one episode
    ep = gate(_test_rows(one_bad_episode, blend))
    assert not ep["pass"] and not ep["conditions"]["no_episode_worse_10pct"]

    deaths = gate(_test_rows(good, blend, dead=1))
    assert not deaths["pass"] and not deaths["conditions"]["no_extra_deaths"]

    broken = gate(_test_rows(good, blend, violations=3))
    assert not broken["pass"] and not broken["conditions"]["route_integrity_and_three_tasks"]

    one_win = {"a": [130, 130], "b": [99, 99], "c": [99, 99]}
    assert not gate(_test_rows(one_win, blend))["conditions"]["beats_blend_2of3"]
    assert not gate([])["pass"]


def test_gate_b_requires_the_registered_episodes_paired_with_blend():
    """Same lesson as Gate A: coverage is exact or it is not coverage. A missing registered
    episode, a duplicated row standing in for one, an unregistered task set, and an unpaired
    learned episode must each fail — the old check only compared task NAMES and counted three."""
    blend = {"a": [100, 100, 100], "b": [100, 100, 100], "c": [100, 100, 100]}
    good = {"a": [110, 110, 110], "b": [108, 108, 108], "c": [106, 106, 106]}
    want = {"a": [0, 1, 2], "b": [0, 1, 2], "c": [0, 1, 2]}
    full = _test_rows(good, blend)

    def gate(rows, **kw):
        kw.setdefault("expected_tasks", list(want))
        kw.setdefault("expected_seeds_by_task", want)
        return rr.gate_b(rows, policy="gnn_residual", require_registered=False, **kw)

    v = gate(full)
    assert v["pass"] and v["coverage"]["ok"]

    two = gate(_test_rows({t: cu[:2] for t, cu in good.items()},
                          {t: cu[:2] for t, cu in blend.items()}))
    assert not two["pass"] and not two["conditions"]["route_integrity_and_three_tasks"]

    duped = gate([*full, dict(full[0])])
    assert not duped["pass"] and duped["coverage"]["duplicate_rows"] == 1

    other = gate(full, expected_tasks=["a", "b", "z"],
                 expected_seeds_by_task={"a": [0, 1, 2], "b": [0, 1, 2], "z": [0, 1, 2]})
    assert not other["pass"]

    unpaired = [r for r in full
                if not (r["policy"] == "blend_0.5" and r["task_id"] == "a"
                        and r["trajectory_seed"] == 2)]
    lonely = gate(unpaired)
    assert not lonely["pass"] and not lonely["coverage"]["paired_with_blend"]


def test_gate_b_defaults_to_the_registered_scope_instead_of_inferring_it_from_the_rows():
    """Leaving `expected_*` unset used to mean "believe whatever the rows contain", so two
    arbitrary episodes per task returned PASS. The default is now the pre-registered final-test
    scope: the real tasks and their real episodes pass, and everything narrower is refused before
    a single consumer-unit comparison is made."""
    from tokenomics.eval import graph_generalization_dump as gg

    registered = {t: list(gg.SPLIT_REGISTRY[t]["test"]) for t in gg.TASK_IDS}
    blend = {t: [100.0] * 3 for t in gg.TASK_IDS}
    good = {t: [110.0] * 3 for t in gg.TASK_IDS}
    full = _test_rows(good, blend, seeds_by_task=registered)
    assert rr.gate_b(full, policy="gnn_residual")["pass"]

    one_each = _test_rows({t: v[:1] for t, v in good.items()},
                          {t: v[:1] for t, v in blend.items()},
                          seeds_by_task={t: v[:1] for t, v in registered.items()})
    thin = rr.gate_b(one_each, policy="gnn_residual")
    assert not thin["pass"] and not thin["conditions"]["route_integrity_and_three_tasks"]

    off_scope = rr.gate_b(full, policy="gnn_residual", expected_tasks=list(gg.TASK_IDS)[:2],
                          expected_seeds_by_task={t: registered[t] for t in list(gg.TASK_IDS)[:2]})
    assert not off_scope["pass"] and not off_scope["conditions"]["registered_scope"]

    # rows from a task that is not a registered one no longer define their own scope
    synthetic = rr.gate_b(_test_rows({"a": [110.0]}, {"a": [100.0]}), policy="gnn_residual")
    assert not synthetic["pass"] and not synthetic["conditions"]["route_integrity_and_three_tasks"]
    assert synthetic["coverage"]["expected_episode_seeds"] == {
        t: sorted(v) for t, v in registered.items()}


def test_rollout_stages_refuse_an_unregistered_configuration():
    """The closed-loop number must come from the SAME economy, episodes, horizon, producer, fit
    and CODE the labels did. Previously any archive/producer/task/seed/horizon was accepted, so a
    rollout could silently be a different experiment from the offline read; and a fit that merely
    stamped `require_shape: true` could license one no matter how it had actually been run."""
    from tokenomics.eval.graph_value_router_dump import environment_manifest, source_fingerprint
    from tokenomics.train import graph_generalization_probe as gp

    task = _scale_task(0, depth=3)
    registry = {task.task_id: {"train": [0, 1], "validation": [2], "test": [3]}}
    inputs = {"archive": {"sha256": "a" * 64}, "producer_ckpt": {"sha256": "b" * 64}}
    source = source_fingerprint("a" * 40)
    fit_spec = {"require_shape": True, "model_seeds": list(gp.MODEL_SEEDS),
                "arms": {a: {} for a in gp.DEFAULT_ARMS}, "train_size": 8,
                "curve_sizes": list(gp.NESTED_TRAIN_SIZES), "training": dict(gp.TRAINING),
                "source": source,
                "dump_spec": {"horizon": 100, "inputs": inputs, "source": source,
                              "env": environment_manifest(),
                              "split_manifest": {"registry": registry}}}
    ok = dict(records=[task], seeds_by_task={task.task_id: [2]}, horizon=100,
              model_seeds=list(gp.MODEL_SEEDS), split="validation", input_manifest=inputs)
    rr.assert_rollout_shape(fit_spec, **ok)

    for change in ({"seeds_by_task": {task.task_id: [2, 3]}},   # a test episode sneaks in
                   {"horizon": 50},                              # not the dump's horizon
                   {"model_seeds": [0]},                         # not the registered ensemble
                   {"input_manifest": {"archive": {"sha256": "c" * 64},
                                       "producer_ckpt": {"sha256": "b" * 64}}},
                   {"input_manifest": None},
                   {"records": []}):
        with pytest.raises(ValueError):
            rr.assert_rollout_shape(fit_spec, **{**ok, **change})

    # ...and every way the FIT itself can be off-contract while still claiming to be certified
    for bad_fit in ({"require_shape": False},
                    {"model_seeds": [0]},
                    {"train_size": 2},
                    {"curve_sizes": [8]},
                    {"arms": {"row_mlp_h32": {}}},
                    {"training": {**gp.TRAINING, "max_epochs": 1}},
                    {"source": source_fingerprint(None)},
                    {"dump_spec": {**fit_spec["dump_spec"],
                                   "source": {"tree_sha256": "0" * 64, "commit": "a" * 40}}}):
        with pytest.raises(ValueError):
            rr.assert_rollout_shape({**fit_spec, **bad_fit}, **ok)


def test_a_forged_gate_a_cannot_authorize_a_test_rollout(tmp_path):
    """`gate_a.json` is not evidence — the verdict is RECOMPUTED from the prediction rows it
    names, whose SHA-256 it recorded. A hand-written `pass: true`, a swapped prediction file, and
    (on the real path) an unbound gate file must all be refused before an episode is played."""
    from tokenomics.train import graph_generalization_probe as gp

    failing = _gate_pred_rows(regret=0.5)          # at the random floor: Gate A must fail
    pred = tmp_path / "test_predictions.csv"
    gp.write_csv(str(pred), failing, gp.PRED_FIELDS)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"roots": {}, "sha256": {}}))
    honest = gp.gate_a(failing, arm="mpnn_4x64")
    assert not honest["pass"]

    forged = {"arm": "mpnn_4x64", "tasks": list(_REGISTERED_TASKS), "model_seeds": [0, 1, 2],
              "test_seeds_by_task": _registered_test_seeds(),
              "predictions": "test_predictions.csv",
              "predictions_sha256": gp.sha256_of(str(pred)),
              "manifest_sha256": gp.sha256_of(str(manifest)),
              "verdict": {"pass": True, "conditions": {}}}
    gate = tmp_path / "gate_a.json"
    gate.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="recomputing Gate A"):
        rr._verified_gate_a(forged, str(gate), manifest_path=str(manifest),
                            recompute=gp.gate_a, require_binding=True)

    swapped = dict(forged, predictions_sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        rr._verified_gate_a(swapped, str(gate), manifest_path=str(manifest),
                            recompute=gp.gate_a, require_binding=True)

    unbound = {"verdict": {"pass": True, "conditions": {}}}
    with pytest.raises(ValueError, match="not bound"):
        rr._verified_gate_a(unbound, str(gate), manifest_path=str(manifest),
                            recompute=gp.gate_a, require_binding=True)

    unscoped = {k: v for k, v in forged.items() if k != "test_seeds_by_task"}
    with pytest.raises(ValueError, match="scope"):  # cannot re-derive what it never recorded
        rr._verified_gate_a(unscoped, str(gate), manifest_path=str(manifest),
                            recompute=gp.gate_a, require_binding=True)

    smoke = dict(forged, require_shape=False)
    with pytest.raises(ValueError, match="require_shape=False"):
        rr._verified_gate_a(smoke, str(gate), manifest_path=str(manifest),
                            recompute=gp.gate_a, require_binding=True)

    # an honest gate file recomputes to the same verdict and is accepted
    passing = _gate_pred_rows(regret=0.0)
    gp.write_csv(str(pred), passing, gp.PRED_FIELDS)
    truthful = dict(forged, predictions_sha256=gp.sha256_of(str(pred)),
                    verdict=gp.gate_a(passing, arm="mpnn_4x64"))
    assert truthful["verdict"]["pass"]
    assert rr._verified_gate_a(truthful, str(gate), manifest_path=str(manifest),
                               recompute=gp.gate_a, require_binding=True)["pass"]


def _registered_test_seeds() -> dict:
    from tokenomics.eval import graph_generalization_dump as gg

    return {t: list(gg.SPLIT_REGISTRY[t]["test"]) for t in gg.TASK_IDS}


def _registered_tasks() -> tuple:
    from tokenomics.eval import graph_generalization_dump as gg

    return gg.TASK_IDS


_REGISTERED_TASKS = _registered_tasks()


def _gate_pred_rows(*, regret: float) -> list[dict]:
    """Test predictions over the REGISTERED final-test scope (three tasks x their three episodes
    x three model seeds); `regret` is the fraction of decisions the scored arm ranks backwards
    (0.5 == the uniform-random floor)."""
    seeds = _registered_test_seeds()
    rows = []
    for arm, r in (("mpnn_4x64", regret), ("row_mlp_h32", 0.5)):
        for task in _REGISTERED_TASKS:
            for ms in ("0", "1", "2", "ensemble"):
                for ts in seeds[task]:
                    for d in range(10):
                        pred = [1.0, 0.0] if d < int(round(r * 10)) else [0.0, 1.0]
                        for ci in range(2):
                            rows.append({"arm": arm, "model_seed": ms, "split": "test",
                                         "task_id": task, "trajectory_seed": ts,
                                         "decision_index": d, "candidate_index": ci,
                                         "producer_id": f"p{ci}", "pred": pred[ci],
                                         "primary_label": float(ci), "flat": 0})
    return rows


def test_the_verdict_stage_recomputes_both_gates_instead_of_reading_their_pass_fields(tmp_path):
    """The standalone verdict stage is artifact-only, but not trust-only. `gate_b.json` names the
    rollout rows it came from and their SHA-256, so a hand-written `pass: true` — or an edited
    rows file underneath an honest one — cannot mint PASS_RESIDUAL_WITHIN_TOPOLOGY."""
    seeds = _registered_test_seeds()
    rows = _test_rows({t: [110.0] * 3 for t in _REGISTERED_TASKS},
                      {t: [100.0] * 3 for t in _REGISTERED_TASKS}, seeds_by_task=seeds)
    rows_path = tmp_path / "test_rows.csv"
    rr._write_csv(str(rows_path), rows)
    doc = {"policy": "gnn_residual", "blend_policy": "blend_0.5", "rows": "test_rows.csv",
           "rows_sha256": rr._sha256(str(rows_path)), "tasks": list(_REGISTERED_TASKS),
           "test_seeds_by_task": seeds, "verdict": rr.gate_b(rows, policy="gnn_residual")}
    path = tmp_path / "gate_b.json"
    path.write_text(json.dumps(doc))
    assert doc["verdict"]["pass"]
    assert rr._verified_gate_b(doc, str(path), require_binding=True)["pass"]

    forged = {"verdict": {"pass": True, "conditions": {}}}
    with pytest.raises(ValueError, match="not bound"):
        rr._verified_gate_b(forged, str(path), require_binding=True)

    # the rows are swapped for a worse rollout after the verdict was written
    rr._write_csv(str(rows_path), _test_rows({t: [90.0] * 3 for t in _REGISTERED_TASKS},
                                             {t: [100.0] * 3 for t in _REGISTERED_TASKS},
                                             seeds_by_task=seeds))
    with pytest.raises(ValueError, match="sha256"):
        rr._verified_gate_b(doc, str(path), require_binding=True)

    honest_but_failing = dict(doc, rows_sha256=rr._sha256(str(rows_path)))
    with pytest.raises(ValueError, match="recomputing Gate B"):
        rr._verified_gate_b(honest_but_failing, str(path), require_binding=True)


def test_verdict_json_is_emitted_as_the_single_mechanical_result(tmp_path):
    """The handover requires exactly one result string; it must exist as an artifact, not as an
    inference a reader makes from three gate files."""
    out = str(tmp_path / "rollout")
    doc = rr.write_verdict(out, gate_a={"pass": True}, adapter={"selected": "gnn_alpha0.1_margin0"},
                           gate_b={"pass": True})
    assert doc["verdict"] == "PASS_RESIDUAL_WITHIN_TOPOLOGY"
    assert json.load(open(f"{out}/verdict.json")) == doc
    assert rr.write_verdict(out, gate_a={"pass": True}, adapter={"selected": None},
                            gate_b=None)["verdict"] == "OFFLINE_ONLY"
    assert rr.write_verdict(out, gate_a={"pass": False}, adapter=None,
                            gate_b=None)["verdict"] == "NO_GENERALIZATION"


def test_mechanical_verdict_covers_every_gate_combination():
    """One field decides the experiment. Offline success alone is `OFFLINE_ONLY`; a failed
    offline gate is `NO_GENERALIZATION` whatever the rollouts would have said."""
    assert rr.verdict_from_gates(True, True, True) == "PASS_RESIDUAL_WITHIN_TOPOLOGY"
    assert rr.verdict_from_gates(True, True, False) == "OFFLINE_ONLY"
    assert rr.verdict_from_gates(True, False, None) == "OFFLINE_ONLY"
    assert rr.verdict_from_gates(False, True, True) == "NO_GENERALIZATION"
    assert rr.verdict_from_gates(True, True, True, cancelled_by_exp0040=True) == \
        "CANCELLED_BY_EXP0040"


# --- end-to-end smoke ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted(tmp_path_factory) -> tuple:
    """The smallest real stack the router stages need: one task with train/validation/test
    episodes, a frozen 2-arm fit, and the task record to replay. Built once per module."""
    from tokenomics.eval import graph_generalization_dump as gg
    from tokenomics.train import graph_generalization_probe as gp

    root = tmp_path_factory.mktemp("router_e2e")
    task = _scale_task(0, depth=3)
    reg = {task.task_id: {"train": (0, 1), "validation": (2,), "test": (3,)}}
    dump = str(root / "dump")
    gg.run_generalization_dump([task], prod=PRODUCING_BASELINES["stock_aware"],
                               target_mode="single_suffix", delta=1.0, lookahead=1, horizon=6,
                               out_dir=dump, registry=reg, strict_splits=False,
                               require_license=False, log=lambda *a, **k: None)
    fit = str(root / "fit")
    gp.run_fit(dump, fit, arms=("row_mlp_h32", "graph_0hop_h32"), model_seeds=(0,),
               training={"max_epochs": 4, "min_epochs": 1, "validation_every": 2,
                         "early_stop_patience": 1, "batch_decisions_per_task": 4},
               train_size=2, curve_sizes=(2,), require_shape=False, log=lambda *a, **k: None)
    return str(root), dump, fit, task


def test_selection_then_test_rollout_runs_the_immutable_order_end_to_end(fitted):
    """Smoke of stages 3-4-6 on real episodes: the validation grid rolls the adapter out against
    the blend floor, freezes `(alpha, margin)` plus a content-hash manifest, and the test stage —
    only after a passing Gate A — replays the FROZEN adapter on the untouched test episodes and
    writes rows that recompute to the same Gate B verdict."""
    root, dump, fit, task = fitted
    out = f"{root}/adapter"
    res = rr.run_validation_selection(
        [task], fit, out, prod=PRODUCING_BASELINES["stock_aware"], arm="graph_0hop_h32",
        model_seeds=(0,), horizon=6, alphas=(0.25,), margins=(0.0,), require_shape=False,
        log=lambda *a, **k: None)
    doc = json.load(open(f"{out}/adapter.json"))
    assert doc == res["adapter"] and doc["clip"] == 1.0
    grid = [r for r in res["rows"]]
    assert {r["policy"] for r in grid} == {"blend_0.5", "gnn_alpha0.25_margin0",
                                           "row_alpha0.25_margin0"}
    assert all(r["route_conservation_violations"] == 0 and r["route_eligibility_violations"] == 0
               for r in grid)
    assert {r["trajectory_seed"] for r in grid} == {2}  # validation episodes only

    gate = f"{root}/gate_a.json"
    with open(gate, "w") as f:  # stand-in for the real Gate A read (its own tests cover it)
        json.dump({"verdict": {"pass": True, "conditions": {}}}, f)
    test_out = f"{root}/rollout"
    if doc["selected"] is None:  # the honest stop path: nothing cleared validation safety
        with pytest.raises(ValueError):
            rr.run_test_rollout([task], fit, out, test_out,
                                prod=PRODUCING_BASELINES["stock_aware"], gate_a_path=gate,
                                model_seeds=(0,), horizon=6, require_shape=False,
                                log=lambda *a, **k: None)
        return
    verdict = rr.run_test_rollout([task], fit, out, test_out,
                                  prod=PRODUCING_BASELINES["stock_aware"], gate_a_path=gate,
                                  model_seeds=(0,), horizon=6, require_shape=False,
                                  log=lambda *a, **k: None)
    typed = rr._read_rollout_rows(f"{test_out}/test_rows.csv")
    assert {r["trajectory_seed"] for r in typed} == {3}  # the untouched test episode
    assert rr.gate_b(typed, policy="gnn_residual",
                     require_registered=False)["conditions"] == verdict["conditions"]
    gate_b_doc = json.load(open(f"{test_out}/gate_b.json"))
    assert gate_b_doc["verdict"] == verdict
    # the emitted gate document is bound to the rows it was computed from
    assert gate_b_doc["rows"] == "test_rows.csv"
    assert rr._verified_gate_b(gate_b_doc, f"{test_out}/gate_b.json",
                               require_binding=False)["conditions"] == verdict["conditions"]


# --- run-order enforcement ------------------------------------------------------------------------
def test_test_rollout_refuses_without_a_passing_gate_a(tmp_path):
    """Step 6 may only run after Gate A passes. A failing (or absent) Gate A aborts before a
    single final-test episode is played."""
    gate = tmp_path / "gate_a.json"
    gate.write_text(json.dumps({"verdict": {"pass": False, "conditions": {}}}))
    with pytest.raises(ValueError):
        rr.run_test_rollout([_scale_task(0)], str(tmp_path), str(tmp_path), str(tmp_path / "out"),
                            prod=PRODUCING_BASELINES["stock_aware"], gate_a_path=str(gate),
                            require_shape=False)
