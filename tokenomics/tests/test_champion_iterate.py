"""EXP-0007B champion-iteration driver (`tokenomics.train.champion_iterate`).

The driver orchestrates the existing orderer expert-iteration pieces (`orderer_improve`)
into one repeatable loop: load the champion registry -> preflight the champion checkpoint
-> train a candidate -> evaluate candidate + champion on the held-out suite -> gate ->
write an experiment record -> emit a promotion recommendation. It must NEVER overwrite the
champion (ALPHAZERO_EXPERT_ITERATION_APPROACH.md: promotion is human-approved).

These tests exercise the orchestration through its public interface with injected
train/eval functions, so the loop is fast and cloud-free (real training is Runpod-only,
CLAUDE.md). The heavy `run_orderer_improve`/`eval_m2` wiring is the production default,
tested separately by the existing orderer-improve suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tokenomics.train.champion_iterate import (
    ChampionRegistry,
    GateDecision,
    PreflightResult,
    _parse_teacher,
    champion_iterate,
    check_checkpoint,
    evaluate_gates,
    preflight,
    write_experiment_record,
)


def _rows(cu_by_seed: dict[int, float], *, env_by_seed=None, dead=0.0):
    """eval_m2-shaped rows: consumer units per held-out seed, env bar below it by default
    (so every seed clears the strict envelope) unless `env_by_seed` overrides."""
    out = []
    for s, cu in cu_by_seed.items():
        env = (env_by_seed or {}).get(s, cu - 1.0)
        out.append({
            "seed": s, "m2_cu": cu, "m2_ci": 0.0, "env_cu": env,
            "ratio_env": (cu / env if env else float("nan")), "m2_dead": dead,
        })
    return out

# The synced package dir (DEFAULT_LOCAL_DIR -> /workspace/tokenomics on the pod) is the
# registry's repo_root: tests/ -> package dir. The champion is staged inside it (P1).
_SYNC_DIR = Path(__file__).resolve().parents[1]
_V3_CHAMPION = _SYNC_DIR / "m2_inputs" / "m2_stream_stochastic.pt"


def _valid_registry_dict() -> dict:
    """The checked-in v3 product-stream champion registry (EXP-0007A)."""
    return {
        "champion_id": "m2_stream_stochastic",
        "checkpoint": "experiments/EXP-0007A-product-stream-ordering/artifacts/m2_stream_stochastic.pt",
        "producer": "v4",
        "orderer_obs_version": 3,
        "product_stream": True,
        "depth": 3,
        "checkpoint_version": 6,
        "held_out_topology_seeds": [6, 7, 8, 9],
        "hard_cells": [6, 9],
        "eval_episode_seeds": 40,
        "strict_envelope": True,
        "primary_metric": "consumer_completed_units",
        "max_death_rate": 0.0,
    }


def test_registry_load_parses_the_champion_contract(tmp_path):
    """A well-formed registry round-trips into a typed `ChampionRegistry` so the driver
    has one source of truth for the active champion + gate configuration."""
    p = tmp_path / "champion.json"
    p.write_text(json.dumps(_valid_registry_dict()))

    reg = ChampionRegistry.load(p)

    assert reg.champion_id == "m2_stream_stochastic"
    assert reg.orderer_obs_version == 3
    assert reg.product_stream is True
    assert reg.held_out_topology_seeds == [6, 7, 8, 9]
    assert reg.hard_cells == [6, 9]
    assert reg.max_death_rate == 0.0


def _registry() -> ChampionRegistry:
    return ChampionRegistry(**{
        k: v for k, v in _valid_registry_dict().items()
        if k in ChampionRegistry.__dataclass_fields__
    })


def _valid_ckpt(reg: ChampionRegistry) -> dict:
    """A synthetic checkpoint that matches the registry's contract: right schema version,
    the obs-v3 fingerprint, and a *learned* stochastic head (some dims above the init floor)."""
    from tokenomics.env.spaces import config_fingerprint
    from tokenomics.eval.scale import build_scale_config

    fp = config_fingerprint(build_scale_config(0, depth=reg.depth), reg.orderer_obs_version)
    log_std = np.full(180, -2.0, dtype=np.float32)
    log_std[:28] = -0.6  # decode-active dims that learned a wider spread
    return {
        "version": reg.checkpoint_version,
        "config_fingerprint": fp,
        "order_act": 180,
        "order": {"log_std": log_std},
    }


def test_preflight_passes_a_matching_stochastic_checkpoint():
    """The happy path: a checkpoint whose schema version, obs fingerprint, and learned
    stochastic head all match the registry contract passes with no blocking reasons."""
    reg = _registry()
    res = check_checkpoint(_valid_ckpt(reg), reg)
    assert res.ok is True
    assert res.reasons == []


def test_preflight_rejects_wrong_schema_version():
    """A stale-schema checkpoint holds semantically wrong tensors (ADR-0026) — fail fast."""
    reg = _registry()
    ck = _valid_ckpt(reg)
    ck["version"] = 5
    res = check_checkpoint(ck, reg)
    assert res.ok is False
    assert any("version" in r for r in res.reasons)


def test_preflight_rejects_fingerprint_off_the_obs_contract():
    """The fingerprint must match the registry's obs version: a v1/v2 model on the same
    economy fingerprints differently (ADR-0051), so a mismatch means the wrong contract."""
    reg = _registry()
    ck = _valid_ckpt(reg)
    ck["config_fingerprint"] = "deadbeefdeadbeef"
    res = check_checkpoint(ck, reg)
    assert res.ok is False
    assert any("fingerprint" in r for r in res.reasons)


def test_preflight_rejects_a_collapsed_deterministic_head():
    """The accepted lever is routing diversity (EXP-0006B): a stochastic champion must
    carry a learned per-dim std. An all-at-init (collapsed) head is a deterministic clone."""
    reg = _registry()
    ck = _valid_ckpt(reg)
    ck["order"]["log_std"] = np.full(180, -2.0, dtype=np.float32)  # nothing grew
    res = check_checkpoint(ck, reg)
    assert res.ok is False
    assert any("stochastic" in r or "log_std" in r for r in res.reasons)


_CHECKED_IN_REGISTRY = _SYNC_DIR / "champion.json"
requires_champion = pytest.mark.skipif(
    not _V3_CHAMPION.exists(), reason="v3 champion artifact absent (git-lfs not pulled)"
)


@requires_champion
def test_preflight_passes_the_checked_in_v3_champion():
    """Guardrail: the checked-in `champion.json` + its staged artifacts actually pass
    preflight, resolved against the synced package dir as repo_root — proving the
    production Runpod path (registry + checkpoints inside the rsync payload) is loadable,
    not just the local tree."""
    reg = ChampionRegistry.load(_CHECKED_IN_REGISTRY)
    assert reg.checkpoint == "m2_inputs/m2_stream_stochastic.pt"
    assert (_SYNC_DIR / reg.producer_checkpoint).exists()  # producer staged in payload too
    res = preflight(reg, repo_root=_SYNC_DIR)
    assert res.ok, res.reasons


_HELD_OUT = {6: 510.0, 7: 540.0, 8: 530.0, 9: 505.0}
_CHAMPION = {6: 500.0, 7: 535.0, 8: 528.0, 9: 500.0}


_EXPECTED = [6, 7, 8, 9]


def test_gate_promotes_a_clean_win_over_champion_and_envelope():
    """All gates pass: zero death, every held-out seed clears the strict envelope, and the
    candidate mean beats the champion with no hard-cell regression -> PROMOTE_RECOMMENDED."""
    d = evaluate_gates(_rows(_HELD_OUT), _rows(_CHAMPION),
                       expected_seeds=_EXPECTED, hard_cells=[6, 9])
    assert d.promote is True
    assert d.reasons == []


def test_gate_blocks_when_a_held_out_seed_is_missing_from_the_eval():
    """An incomplete eval must not promote: if a held-out topology (e.g. hard cell s9) is
    absent from the rows, its envelope/death/regression checks would be silently skipped and
    the mean computed over a smaller set. The gate fails on missing coverage instead."""
    partial = {6: 510.0, 7: 540.0, 8: 530.0}  # s9 omitted
    d = evaluate_gates(_rows(partial), _rows(_CHAMPION),
                       expected_seeds=_EXPECTED, hard_cells=[6, 9])
    assert d.promote is False
    assert any("9" in r and ("missing" in r or "coverage" in r) for r in d.reasons)


def test_gate_blocks_on_any_held_out_death():
    """Zero death is a hard safety gate — one held-out topology dying blocks promotion
    even if every other gate passes."""
    cand = _rows(_HELD_OUT)
    cand[2]["m2_dead"] = 0.2  # seed 8 dies
    d = evaluate_gates(cand, _rows(_CHAMPION), expected_seeds=_EXPECTED, hard_cells=[6, 9])
    assert d.promote is False
    assert any("death" in r and "8" in r for r in d.reasons)


def test_gate_blocks_when_a_seed_misses_the_strict_envelope():
    """The strict envelope is per-topology: a candidate that ties/loses the envelope on
    even one held-out seed is not promotable (the standard since EXP-0006B)."""
    cand = _rows(_HELD_OUT, env_by_seed={9: 510.0})  # s9 env above candidate -> ratio<1
    d = evaluate_gates(cand, _rows(_CHAMPION), expected_seeds=_EXPECTED, hard_cells=[6, 9])
    assert d.promote is False
    assert any("envelope ratio" in r and "9" in r for r in d.reasons)


def test_gate_blocks_a_tie_on_the_held_out_mean():
    """A lateral result is not an improvement: the documented gate requires the candidate to
    *beat* the champion mean, so an exact tie (every gate else passing) must not promote."""
    tie = dict(_CHAMPION)  # identical per-seed -> held-out mean ties exactly
    d = evaluate_gates(_rows(tie), _rows(_CHAMPION),
                       expected_seeds=_EXPECTED, hard_cells=[6, 9])
    assert d.promote is False
    assert any("mean" in r for r in d.reasons)


def test_gate_blocks_when_candidate_mean_does_not_beat_champion():
    """A candidate must beat the current champion's held-out mean, not merely the envelope —
    otherwise the loop could churn sideways."""
    weak = {6: 501.0, 7: 501.0, 8: 501.0, 9: 501.0}  # clears env, below champion mean
    d = evaluate_gates(_rows(weak), _rows(_CHAMPION), expected_seeds=_EXPECTED, hard_cells=[6, 9])
    assert d.promote is False
    assert any("mean" in r for r in d.reasons)


def test_gate_blocks_a_hard_cell_regression_below_champion():
    """No known hard cell (s6/s9) may regress below the champion even if the suite mean
    improves — the random-dominated cells are where prior policies failed."""
    cand = {6: 540.0, 7: 560.0, 8: 545.0, 9: 495.0}  # mean up, but s9 < champion 500
    d = evaluate_gates(_rows(cand), _rows(_CHAMPION), expected_seeds=_EXPECTED, hard_cells=[6, 9])
    assert d.promote is False
    assert any("hard cell s9" in r for r in d.reasons)


def _write_record(tmp_path, decision):
    out = tmp_path / "EXP-0007B-r1"
    ckpt = tmp_path / "candidate.pt"
    ckpt.write_bytes(b"fake-candidate-checkpoint")
    write_experiment_record(
        out, registry=_registry(), candidate_ckpt=ckpt,
        candidate_rows=_rows(_HELD_OUT), champion_rows=_rows(_CHAMPION),
        decision=decision,
    )
    return out


def test_record_emits_the_experiment_dir_layout(tmp_path):
    """One run writes the standard experiments/ layout (experiments/README.md contract):
    spec.json + RESULT.md + summary.csv, with the candidate checkpoint under artifacts/."""
    out = _write_record(tmp_path, GateDecision(True, []))
    assert (out / "spec.json").exists()
    assert (out / "RESULT.md").exists()
    assert (out / "summary.csv").exists()
    assert (out / "artifacts" / "checkpoint.pt").read_bytes() == b"fake-candidate-checkpoint"


def test_record_states_the_promotion_recommendation(tmp_path):
    """The record carries the machine-readable recommendation both ways: spec.json holds the
    boolean and RESULT.md prints the `PROMOTE_RECOMMENDED=...` line the driver emits."""
    out = _write_record(tmp_path, GateDecision(False, ["seed 9: envelope ratio 0.98 !> 1.00"]))
    spec = json.loads((out / "spec.json").read_text())
    assert spec["promote_recommended"] is False
    assert "seed 9" in " ".join(spec["gate_reasons"])
    assert "PROMOTE_RECOMMENDED=false" in (out / "RESULT.md").read_text()


def test_record_summary_csv_has_a_row_per_held_out_seed(tmp_path):
    """summary.csv is the machine-readable per-seed record — one row per held-out topology."""
    import csv as _csv

    out = _write_record(tmp_path, GateDecision(True, []))
    with open(out / "summary.csv") as f:
        rows = list(_csv.DictReader(f))
    assert [int(r["seed"]) for r in rows] == [6, 7, 8, 9]


def _fake_repo(tmp_path):
    """A throwaway repo_root with a registry pointing at a (fake) champion checkpoint,
    so the orchestration runs without touching the real LFS artifacts."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "champ.pt").write_bytes(b"champion")
    d = _valid_registry_dict()
    d["checkpoint"] = "champ.pt"
    (repo / "champion.json").write_text(json.dumps(d))
    return repo


def _ok_pre(reg, *, repo_root):
    return PreflightResult(True, [])


def test_champion_iterate_runs_the_full_loop_with_injected_fns(tmp_path):
    """End-to-end orchestration: load registry -> preflight -> train -> eval candidate +
    champion -> gate -> write record -> return the decision. With injected train/eval fns
    the loop is cloud-free (real training is Runpod-only, CLAUDE.md)."""
    repo = _fake_repo(tmp_path)
    seen = {}

    def fake_train(reg, candidate_ckpt, **kw):
        Path(candidate_ckpt).write_bytes(b"candidate")
        seen["train"] = True

    def fake_eval(ckpt, reg, **kw):
        # candidate beats the champion -> clean promote
        is_cand = Path(ckpt).read_bytes() == b"candidate"
        return _rows(_HELD_OUT if is_cand else _CHAMPION)

    d = champion_iterate(
        repo / "champion.json", tmp_path / "EXP-0007B-r1", repo_root=repo,
        preflight_fn=_ok_pre, train_fn=fake_train, eval_fn=fake_eval, eval_seeds=2,
    )

    assert isinstance(d, GateDecision) and d.promote is True
    assert seen["train"] is True
    assert (tmp_path / "EXP-0007B-r1" / "spec.json").exists()
    assert (tmp_path / "EXP-0007B-r1" / "RESULT.md").exists()


def test_champion_iterate_never_modifies_the_champion(tmp_path):
    """The core safety guarantee (ALPHAZERO_EXPERT_ITERATION_APPROACH.md): a run writes only
    under out_dir — the champion registry and its checkpoint are byte-identical afterwards,
    even when the candidate wins (promotion stays human-approved)."""
    repo = _fake_repo(tmp_path)
    reg_before = (repo / "champion.json").read_bytes()
    ckpt_before = (repo / "champ.pt").read_bytes()

    def fake_train(reg, candidate_ckpt, **kw):
        Path(candidate_ckpt).write_bytes(b"candidate")

    champion_iterate(
        repo / "champion.json", tmp_path / "EXP", repo_root=repo,
        preflight_fn=_ok_pre, train_fn=fake_train,
        eval_fn=lambda ckpt, reg, **kw: _rows(_HELD_OUT), eval_seeds=2,
    )

    assert (repo / "champion.json").read_bytes() == reg_before
    assert (repo / "champ.pt").read_bytes() == ckpt_before


def test_champion_iterate_aborts_before_training_when_preflight_fails(tmp_path):
    """Fail fast: a preflight mismatch (wrong obs/version/collapsed head) must abort before
    any cloud training is launched."""
    repo = _fake_repo(tmp_path)
    train_called = []

    def bad_pre(reg, *, repo_root):
        return PreflightResult(False, ["fingerprint off the obs contract"])

    with pytest.raises(RuntimeError, match="preflight"):
        champion_iterate(
            repo / "champion.json", tmp_path / "EXP", repo_root=repo,
            preflight_fn=bad_pre,
            train_fn=lambda *a, **k: train_called.append(1),
            eval_fn=lambda *a, **k: _rows(_HELD_OUT), eval_seeds=2,
        )
    assert train_called == []


def test_parse_teacher_overrides_only_named_keys():
    """`--teacher samples=24,lookahead=8` overrides the search depth/breadth while keeping
    the default candidate set — so the CLI can sweep teacher strength without retyping all."""
    t = _parse_teacher("samples=24,lookahead=8")
    assert t["samples"] == 24
    assert t["lookahead"] == 8
    assert t["candidates"] == ["even_split", "proportional", "obs_price"]


def test_registry_load_rejects_a_missing_required_field(tmp_path):
    """A registry missing the champion checkpoint is unusable — fail loudly at load,
    before any cloud work, naming the missing field."""
    d = _valid_registry_dict()
    del d["checkpoint"]
    p = tmp_path / "champion.json"
    p.write_text(json.dumps(d))

    with pytest.raises(ValueError, match="checkpoint"):
        ChampionRegistry.load(p)
