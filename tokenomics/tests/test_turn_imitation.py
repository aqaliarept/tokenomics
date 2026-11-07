"""Task 4.3 (m6_turn_imitation): the batched head equals the per-turn head, the labels are the default decision,
a tiny fit reaches full agreement, the closed-loop reading, and a true smoke of the run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenomics.train import m6_gnn as g  # noqa: E402
from tokenomics.train import m6_turn_imitation as ti  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "experiments/EXP-0100-m6-imitation-v2/artifacts/fit/models.pt"
TRAINING = REPO / "experiments/EXP-0058-es-gradient-probe/bar"
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


def _rec(rng: np.random.Generator, kind: str) -> dict:
    nc, ni = int(rng.integers(1, 3)), int(rng.integers(0, 3))
    base = {"kind": kind, "h_cap": rng.normal(size=(nc, g.HIDDEN)).astype(np.float32),
            "h_in": rng.normal(size=(ni, g.HIDDEN)).astype(np.float32),
            "pooled": rng.normal(size=4 * g.HIDDEN).astype(np.float32)}
    if kind == "fund":
        n = int(rng.integers(1, 5))
        return {**base, "fund": rng.normal(size=(n, g.FUND_DIM)).astype(np.float32),
                "fund_cap": [(k, int(rng.integers(nc))) for k in range(n)],
                "fund_in": [(k, int(rng.integers(ni))) for k in range(n)] if ni else [],
                "y": np.arange(n, 0, -1).astype(np.float32)}
    nb, nr = int(rng.integers(1, 5)), int(rng.integers(0, 3))
    ranks = np.arange(nb + nr, 0, -1).astype(np.float32)
    return {**base, "bid": rng.normal(size=(nb, g.BID_DIM)).astype(np.float32),
            "bid_cap": [(k, int(rng.integers(nc))) for k in range(nb)], "bid_in": [],
            "claim": rng.normal(size=(nr, g.CLAIM_DIM)).astype(np.float32),
            "claim_cap": [(k, int(rng.integers(nc))) for k in range(nr)], "claim_in": [], "n_res": nr,
            "acc": (rng.random(nb) < 0.7).astype(np.float32), "y_bid": ranks[nr:], "y_claim": ranks[:nr]}


def test_batched_head_equals_per_turn_head() -> None:
    rng = np.random.default_rng(0)
    heads = g.TurnHeads()
    recs = [_rec(rng, "fund") for _ in range(5)] + [_rec(rng, "accept") for _ in range(5)]
    for kind, sel in (("fund", recs[:5]), ("bid", recs[5:]), ("claim", recs[5:])):
        batched = ti.batch_out(heads, kind, ti._cat(sel, kind))
        per = torch.cat([heads.node_out(kind, r, {"capacity": torch.from_numpy(r["h_cap"]),
                                                  "input": torch.from_numpy(r["h_in"])},
                                        torch.from_numpy(r["pooled"])) for r in sel])
        assert torch.allclose(batched, per, atol=1e-6), kind


@pytest.mark.skipif(not _lfs(MODELS), reason="records absent")
def test_a_tiny_fit_reaches_full_agreement() -> None:
    from tokenomics.train import m6_imitation as im

    rng = np.random.default_rng(1)
    recs = [_rec(rng, k) for k in ("fund", "accept") * 6]
    pm, _ = im.load_models(MODELS)
    ti.fit(pm, recs, epochs=300, lr=1e-2, batch=12, cap=None, seed=0, log=lambda *a: None)
    a = ti.agreement(pm, recs)
    assert a["accept"] == 1.0 and a["rank_accept"] == 1.0 and a["rank_fund"] == 1.0
    assert a["whole_schedule"] == 1.0 and ti.offline_pass(a)


def test_unlearnable_pairs_are_excluded_and_counted() -> None:
    """v13: replenishment vs replenishment fund pairs and feature-identical pairs leave the loss and the metric."""
    rep = np.zeros(g.FUND_DIM, np.float32); rep[ti.REPLENISH_COL] = 1.0
    res = np.zeros(g.FUND_DIM, np.float32); res[0] = 1.0
    fund = {"kind": "fund", "h_cap": np.zeros((1, g.HIDDEN), np.float32), "h_in": np.zeros((2, g.HIDDEN), np.float32),
            "pooled": np.zeros(4 * g.HIDDEN, np.float32), "fund": np.stack([res, rep, rep + 0.5 * (np.arange(g.FUND_DIM) == 5)]),
            "fund_cap": [(0, 0)], "fund_in": [(1, 0), (2, 1)], "y": np.asarray([3, 2, 1], np.float32)}
    tg = ti.turn_targets([fund], None, None)
    assert tg["counts"] == {"fund_kept": 2, "fund_rep_rep": 1, "fund_identical": 0, "accept_kept": 0,
                            "accept_identical": 0}
    bid = np.ones(g.BID_DIM, np.float32)
    acc = {"kind": "accept", "h_cap": np.zeros((1, g.HIDDEN), np.float32), "h_in": np.zeros((0, g.HIDDEN), np.float32),
           "pooled": np.zeros(4 * g.HIDDEN, np.float32), "bid": np.stack([bid, bid, 2 * bid]),
           "bid_cap": [(0, 0), (1, 0), (2, 0)], "bid_in": [], "claim": np.zeros((0, g.CLAIM_DIM), np.float32),
           "claim_cap": [], "claim_in": [], "n_res": 0, "acc": np.asarray([1, 1, 0], np.float32),
           "y_bid": np.asarray([3, 2, 1], np.float32), "y_claim": np.zeros(0, np.float32)}
    tg = ti.turn_targets([acc], None, None)
    assert tg["counts"]["accept_identical"] == 1 and tg["counts"]["accept_kept"] == 0  # bid 2 declined
    assert ti.identical_bid_pairs(acc) == (1, 0)
    acc["acc"] = np.asarray([1, 0, 0], np.float32)
    assert ti.identical_bid_pairs(acc) == (1, 1)


@pytest.mark.skipif(not _lfs(MODELS), reason="records absent")
def test_records_roundtrip_and_clipped_scoring(tmp_path: Path) -> None:
    from tokenomics.train import m6_imitation as im

    rng = np.random.default_rng(2)
    recs = [_rec(rng, k) for k in ("fund", "accept") * 5]
    ti.save_records(tmp_path / "r.npz", recs)
    back = ti.load_records(tmp_path / "r.npz")
    assert len(back) == len(recs)
    for a, b in zip(recs, back):
        assert a.keys() == b.keys()
        for k in a:
            assert (a[k] == b[k]) if isinstance(a[k], (str, int, list)) else np.array_equal(a[k], b[k]), k
    pm, _ = im.load_models(MODELS)
    torch.manual_seed(0)
    pm.add_turn_heads()
    with torch.no_grad():  # push every output past the bound: clipped scores tie, so strict order fails
        for m in pm.turn.out.values():
            m.weight.zero_(); m.bias.fill_(50.0)
    ag = ti.agreement(pm, back)
    assert ag == ti.agreement(pm, recs)
    assert ag["saturation"]["fund"] == 1.0 and ag["rank_fund"] == 0.0


def test_closed_loop_reading_rule() -> None:
    from tokenomics.train import producer_read

    def row(arm: str, w: str, rep: int, units: float, dead: bool = False) -> dict:
        return {"arm": arm, "set": "monitor", "world": w, "episode": "1", "replicate": str(rep),
                "units": repr(units), "dead": "true" if dead else "false", "r_start": "1.0", "r_end": "0.9",
                "consumer_fill_rate": '{"c0": 0.99}', "beyond_k": "0"}

    rows = [row(a, w, r, 100.0 + (1 if a == "candidate" else 0)) for a in ("candidate", "reference")
            for w in ("w1", "w2") for r in (0, 1)]
    orig = producer_read.pilot_reading
    producer_read.pilot_reading = lambda *a, **k: {"g_tr": 1.0, "F_tr": 5.0}
    try:
        assert ti.closed_loop_reading(rows, {}, "monitor")["reading"] == "pass"
        rows[0]["dead"] = "true"
        assert ti.closed_loop_reading(rows, {}, "monitor")["reading"] == "fail"
    finally:
        producer_read.pilot_reading = orig


def test_prereg_v13_matches_the_code_when_pinned() -> None:
    if ti.PREREG_V13_SHA256 is None:
        assert ti.prereg_problems()
    else:
        assert ti.prereg_problems() == []


@pytest.mark.skipif(not (_lfs(MODELS) and _lfs(ATLAS_V2) and (TRAINING / "training_worlds.json").exists()),
                    reason="records absent")
def test_true_smoke(tmp_path: Path) -> None:
    import json

    assert ti.main(["--mode", "run", "--training-dir", str(TRAINING), "--atlas", str(ATLAS_V2),
                    "--models", str(MODELS), "--out-dir", str(tmp_path), "--dry-run"]) == 0
    d = json.loads((tmp_path / "reading.json").read_text())
    assert d["collect"]["train_turns"] > 0 and d["offline"]["heldout"]["turns"] > 0
    assert d["closed_loop"]["monitor_12"]["candidate"]["rows"] == 2
    assert d["closed_loop"]["reported_42"]["reference"]["rows"] == 2
    assert (tmp_path / "models.pt").exists() and (tmp_path / "heldout_records.npz").exists()
    assert set(d["offline"]["heldout"]["pairs"]) == {"fund_kept", "fund_rep_rep", "fund_identical", "accept_kept",
                                                     "accept_identical"}
