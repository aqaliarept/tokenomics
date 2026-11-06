"""EXP-0112 (pre-registration v9): reshaped exploration, E4, and a true smoke of the run."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenomics.train import m6_gnn as g  # noqa: E402
from tokenomics.train import m6_imitation as im  # noqa: E402
from tokenomics.train import m6_stage0 as s0  # noqa: E402
from tokenomics.train import m6_stage0b as sb  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "experiments/EXP-0058-es-gradient-probe/bar"
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
E10 = REPO / "experiments/EXP-0110-m6-stage0/artifacts"
MON = REPO / "experiments/EXP-0105-m6-imitation-v3/artifacts/collect/samples_monitor.pkl"


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs = pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(E10 / "start_models.pt") and _lfs(MON)
                                and (TRAINING / "training_worlds.json").exists()),
                           reason="records absent")


@needs
def test_temperatures_give_five_percent_greedy_flips() -> None:
    """The pinned T_g reproduce a 5% greedy flip rate per gate on the monitor samples."""
    pm, _om = im.load_models(E10 / "start_models.pt")
    with MON.open("rb") as fh:
        prod = pickle.load(fh)["producer"]
    with torch.no_grad():
        lg = np.stack([pm(g.v3_to_v4(f))[0].numpy() for f, _y in prod])
    for i, k in enumerate(sb.GATES):
        rate = float((1 / (1 + np.exp(np.abs(lg[:, i]) / sb.TEMPS[k]))).mean())
        assert rate == pytest.approx(0.05, abs=0.001)


def test_ridge_value_recovers_a_planted_baseline() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=(500, 8)), np.ones((500, 1))], 1)
    w = rng.normal(size=9)
    y = x @ w + rng.normal(0, 0.1, 500)
    assert np.allclose(sb.ridge_fit(x, y), w, atol=0.05)


@needs
def test_v9_dry_run(tmp_path: Path) -> None:
    """A true smoke (horizon 4): rollouts with per-decision scores, perturbations, E1–E4, the
    reading; replenish is the frozen start's greedy output, so only gates are scored."""
    assert sb.run(str(TRAINING), str(ATLAS_V2), str(E10), str(MON), str(tmp_path), 1,
                  dry_run=True) == 0
    doc = json.loads((tmp_path / "reading.json").read_text())
    assert doc["reading"] in ("signal", "absent") and doc["rule"] == sb.V9_RULE
    assert set(doc["per_size"]["12"]["D0_3"]) == {"E1", "E2", "E3", "E4"}
    assert set(doc["per_size"]["12"]["flip_rate"]) == set(sb.GATES)
    z = np.load(next((tmp_path / "npz").glob("12_*.npz")))
    assert z["producer_score"].shape[1] == s0.PROJ_K and len(z["producer_emb"]) == len(
        z["producer_cycle"])


def test_v9_is_frozen_and_pinned() -> None:
    assert sb.prereg_problems() == [] and sb.freeze()[0] == "MATCH"
