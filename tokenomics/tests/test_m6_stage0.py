"""M6 critic-free Stage 0 (pre-registration v8): the start policy, the diagnostics, a dry run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenomics.train import m6_gnn as g  # noqa: E402
from tokenomics.train import m6_imitation as im  # noqa: E402
from tokenomics.train import m6_stage0 as s0  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "experiments/EXP-0058-es-gradient-probe/bar"
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
PREV = REPO / "experiments/EXP-0100-m6-imitation-v2/artifacts/fit/models.pt"
MON = REPO / "experiments/EXP-0105-m6-imitation-v3/artifacts/collect/samples_monitor.pkl"


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs = pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(PREV)
                                and (TRAINING / "training_worlds.json").exists()),
                           reason="records absent")


@needs
def test_start_policy_is_bit_identical_to_exp0100() -> None:
    """The v4 inputs enter at zero weight: same gate logits and replenish outputs, bit for bit, on
    recorded decisions (v3 samples + the sign column vs the first 16 columns)."""
    import pickle

    pm2, _ = im.load_models(PREV)
    pm4 = s0.extend_to_v4(pm2)
    assert pm4.features == g.FEATURES_V4 and pm4.capacity_extra == 2
    assert pm4.enc["capacity"][0].in_features == 16 and not pm4.cap_extra.weight.any()
    if _lfs(MON):
        with MON.open("rb") as fh:
            prod = pickle.load(fh)["producer"][::97]
    else:
        rng = np.random.default_rng(0)
        prod = [({"global": rng.normal(size=(1, 13)).astype(np.float32),
                  "capacity": rng.normal(size=(1, 17)).astype(np.float32),
                  "product": rng.normal(size=(2, 18)).astype(np.float32),
                  "input": rng.normal(size=(1, 6)).astype(np.float32),
                  "prod_cap": [(0, 0), (1, 0)], "prod_in": [(1, 0)], "inputs": ["x"]}, None)]
    with torch.no_grad():
        for f, _y in prod:
            f4 = g.v3_to_v4(f)
            f2 = {**f, "capacity": f["capacity"][:, :16]}
            a, b = pm2(f2), pm4(f4)
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def _rows(n_seeds: int, cycles: int, seed_effect: float, noise: float, rng: np.random.Generator,
          signal: np.ndarray | None = None) -> list[dict]:
    rows = []
    for k in range(n_seeds):
        base = rng.normal(0, seed_effect)
        for i in range(s0.G_SIBLINGS):
            sc = rng.normal(size=(cycles, s0.PROJ_K))
            r = base + rng.normal(0, noise, cycles)
            if signal is not None:  # rewards that follow the score along `signal`
                r = r + sc @ signal
            rows.append({"seed": f"w|{k}", "size": 12, "sibling": i, "rewards": list(r),
                         "dead": False, "seconds": 1.0,
                         "scores": {"producer": sc.tolist(), "orderer": sc.tolist()}})
    return rows


def test_scenario_share_and_drift() -> None:
    rng = np.random.default_rng(0)
    big = s0.scenario_share(_rows(20, 10, 5.0, 0.1, rng))
    small = s0.scenario_share(_rows(20, 10, 0.0, 1.0, rng))
    assert big > 0.9 and small < 0.3
    d = s0.drift_horizon(_rows(20, 10, 0.0, 1.0, rng))
    assert d["t_star"] == 0 and s0.gamma_from(0) == 0.9 and s0.gamma_from(None) == 0.99
    assert s0.gamma_from(5) == 0.9 and s0.gamma_from(50) == pytest.approx(0.98)


def test_leave_one_out_advantages_sum_to_zero_per_seed() -> None:
    rng = np.random.default_rng(1)
    seeds = s0._by_seed(_rows(4, 6, 1.0, 1.0, rng))
    for est in ("E2", "E3"):
        adv = s0.advantages(seeds, est, 0.95)
        for v in adv.values():
            assert np.allclose(np.sum(v, axis=0), 0.0)


def test_split_half_finds_a_planted_signal() -> None:
    """Rewards that rise along one score direction give a split-half cosine above the null."""
    rng = np.random.default_rng(2)
    direction = rng.normal(size=s0.PROJ_K) * 0.5
    out = s0.split_half(_rows(24, 8, 1.0, 0.1, rng, direction), 0.95)
    assert out["E3"]["producer"]["passes"] and out["E3"]["orderer"]["passes"]
    assert out["E3"]["producer"]["cosine"] > out["E3"]["producer"]["null_q95"]


@needs
def test_start_checkpoint_round_trips(tmp_path: Path) -> None:
    """The start checkpoint reloads with its zero-weight extra layer (capacity_extra recorded)."""
    start = s0.make_start(str(PREV), tmp_path / "start.pt")
    pm, _om = im.load_models(start)
    assert pm.capacity_extra == 2 and not pm.cap_extra.weight.any()


@needs
def test_stage0_dry_run(tmp_path: Path) -> None:
    assert s0.run(str(TRAINING), str(ATLAS_V2), str(PREV), str(tmp_path), 1, dry_run=True,
                  max_seeds=1, horizon=4) == 0
    doc = json.loads((tmp_path / "reading.json").read_text())
    assert doc["reading"] in ("signal", "signal_42_only", "no_signal")
    assert set(doc["per_size"]) == {"12", "42"}
    cal = json.loads((tmp_path / "calibration.json").read_text())
    assert set(cal["per_size"]) == {"12", "42"} and cal["per_size"]["42"]["nu"] == 4.0


def test_v8_is_frozen_and_pinned() -> None:
    assert s0.prereg_problems() == [] and s0.freeze()[0] == "MATCH"
