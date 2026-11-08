"""EXP-0131's held-out read: episode freshness, digest pairing, the one-shot ledger (user decision 2026-11-08)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tokenomics.train import m6_final_read as f  # noqa: E402
from tokenomics.train.producer_read import HELDOUT_STAGE  # noqa: E402


def test_read_stage_is_not_an_earlier_stage() -> None:
    assert f.FINAL_STAGE not in (HELDOUT_STAGE, f.ABANDONED_STAGE, f.TIMING_STAGE)


def _row(arm: str, demand: str, shock: str = "s") -> dict[str, str]:
    return {"arm": arm, "world": "w", "episode": "1", "replicate": "0", "demand_digest": demand, "shock_digest": shock}


def test_digest_mismatch_across_arms_is_reported() -> None:
    assert f.digest_problems([_row(f.DECIDING, "a"), _row(f.BASE, "a")]) == []
    assert f.digest_problems([_row(f.DECIDING, "a"), _row(f.BASE, "b")])
    assert f.digest_problems([_row(f.DECIDING, "a", "x"), _row(f.BASE, "a", "y")])


def _entry(status: str) -> dict[str, str | None]:
    return {"key": f.LEDGER_KEY, "status": status, "prereg": f.PREREG_V20_SHA256,
            "turn_models_sha256": f.TURN_MODELS_SHA256, "replenish_models_sha256": f.REPLENISH_MODELS_SHA256}


def test_ledger_is_one_shot(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    assert f.ledger_problems(path)
    path.write_text(json.dumps({"entries": [_entry("pending")]}))
    assert f.ledger_problems(path) == []
    path.write_text(json.dumps({"entries": [_entry("pending"), _entry("done")]}))
    assert f.ledger_problems(path)


def _job(arm: str, ep: int = 1) -> dict[str, object]:
    return {"arm": arm, "world": "w", "episode": ep, "replicate": 0}


def _partial(arm: str, ep: int = 1) -> dict[str, str]:  # no `units`: the resume check must stay blind
    return {**_row(arm, "a"), "episode": str(ep), "horizon": "100"}


LOG = f"--mode run --source-commit {f.ATTEMPT1_COMMIT}"


def test_resume_accepts_matching_rows_without_reading_units() -> None:
    assert f.resume_problems([_partial(f.DECIDING)], [_job(f.DECIDING), _job(f.BASE)], LOG, 100) == []


def test_resume_refuses_bad_pins() -> None:
    jobs = [_job(f.DECIDING)]
    assert f.resume_problems([_partial(f.DECIDING)], jobs, "--mode run --source-commit deadbeef", 100)
    assert f.resume_problems([_partial(f.DECIDING)], jobs, LOG, 200)
    assert f.resume_problems([_partial(f.DECIDING), _partial(f.DECIDING)], jobs, LOG, 100)
    assert f.resume_problems([_partial(f.BASE)], jobs, LOG, 100)


def test_job_construction_is_attempt_1s() -> None:
    assert f.job_fingerprint() == f.JOB_FINGERPRINT
