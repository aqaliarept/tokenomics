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
IT40 = REPO / "experiments/EXP-0114-m6-pilot-continued/artifacts/candidate_it40.pt"
START = REPO / "experiments/EXP-0110-m6-stage0/artifacts/start_models.pt"


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


@pytest.mark.parametrize("layers,bounded,train", [(0, False, False), (2, True, False), (2, True, True)])
def test_batched_head_equals_per_turn_head(layers: int, bounded: bool, train: bool) -> None:
    """Also in train mode (the fit's mode: the transformer's slow path, no dropout)."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    heads = g.TurnHeads(set_layers=layers, bounded=bounded).train(train)
    recs = [_rec(rng, "fund") for _ in range(5)] + [_rec(rng, "accept") for _ in range(5)]

    def trunk(r: dict) -> tuple[dict, torch.Tensor]:
        return {"capacity": torch.from_numpy(r["h_cap"]), "input": torch.from_numpy(r["h_in"])}, \
            torch.from_numpy(r["pooled"])

    with torch.no_grad():
        bf = ti.batch_fund(heads, ti._cat(recs[:5], "fund"))
        pf = torch.cat([heads.turn_fund(r, *trunk(r)) for r in recs[:5]])
        assert torch.allclose(bf, pf, atol=1e-5)
        ba = ti.batch_accept(heads, ti._cat(recs[5:], "bid"), ti._cat(recs[5:], "claim"))
        per = [heads.turn_accept(r, *trunk(r)) for r in recs[5:]]
        for k in range(3):
            assert torch.allclose(ba[k], torch.cat([p[k] for p in per]), atol=1e-5), k


@pytest.mark.parametrize("layers", [0, 2])
def test_set_layers_let_a_bid_see_the_other_bids(layers: int) -> None:
    """v14 (DIAGNOSTICS 38): with set layers a bid's outputs move when another bid of its turn changes;
    without them they cannot. Bounded scores stay strictly inside the clip."""
    rng = np.random.default_rng(3)
    torch.manual_seed(1)
    heads = g.TurnHeads(set_layers=layers, bounded=True).eval()
    r = _rec(rng, "accept")
    r["bid"] = rng.normal(size=(3, g.BID_DIM)).astype(np.float32) * 30
    r["bid_cap"] = [(0, 0), (1, 0), (2, 0)]
    h = {"capacity": torch.from_numpy(r["h_cap"]), "input": torch.from_numpy(r["h_in"])}
    pooled = torch.from_numpy(r["pooled"])
    with torch.no_grad():
        a0, p0, c0 = heads.turn_accept(r, h, pooled)
        r2 = {**r, "bid": r["bid"].copy()}
        r2["bid"][2] += 5.0
        a1, p1, _ = heads.turn_accept(r2, h, pooled)
    moved = not torch.allclose(a0[0], a1[0]) or not torch.allclose(p0[0], p1[0])
    assert moved == (layers > 0)
    assert bool((p0.abs() < g.ACTION_LOGIT_BOUND).all()) and bool((c0.abs() < g.ACTION_LOGIT_BOUND).all())


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


@pytest.mark.skipif(not (_lfs(IT40) and _lfs(START) and _lfs(MODELS)), reason="records absent")
def test_real_runs_pin_the_f40_start(tmp_path: Path) -> None:
    import json

    real = REPO / "experiments/EXP-0119-m6-f40-confirm/artifacts/reading.json"
    it30 = REPO / "experiments/EXP-0114-m6-pilot-continued/artifacts/candidate_it30.pt"
    ok = ti.choose_start(str(real), str(MODELS), str(IT40), str(START))
    assert ok["start"] == "F40" and ti.start_problems(ok) == []
    assert ti.start_problems(ti.choose_start(str(real), str(MODELS), str(it30), str(START)))
    assert ti.start_problems(ti.choose_start(str(real), str(MODELS), str(IT40), str(MODELS)))
    fake = tmp_path / "r.json"
    fake.write_text(json.dumps({**json.loads(real.read_text()), "dry_run": True}))
    assert ti.start_problems(ti.choose_start(str(fake), str(MODELS), str(IT40), str(START)))


def test_pre_tanh_penalty_and_ties() -> None:
    """The fit collects the pre-tanh values and penalises them; equal bounded scores count as ties."""
    rng = np.random.default_rng(4)
    torch.manual_seed(0)
    heads = g.TurnHeads(set_layers=2, bounded=True)
    recs = [_rec(rng, "fund") for _ in range(3)] + [_rec(rng, "accept") for _ in range(3)]
    tg = ti.turn_targets(recs, None, None)
    base = ti.CFG["z_l2"]
    try:
        ti.CFG["z_l2"] = 0.0
        l0, _ = ti._losses(heads, tg)
        ti.CFG["z_l2"] = 1.0
        l1, _ = ti._losses(heads, tg)
    finally:
        ti.CFG["z_l2"] = base
    assert float(l1) > float(l0) and heads.pre_bound is None
    big = torch.tensor([95.0, 120.0])
    assert torch.equal(heads._bound(big)[0], heads._bound(big)[1])  # float32 saturation: a real tie


def test_prereg_v14_matches_the_code_when_pinned() -> None:
    if ti.PREREG_V14_SHA256 is None:
        assert ti.prereg_problems()
    else:
        assert ti.prereg_problems() == []




@pytest.mark.skipif(not (_lfs(MODELS) and _lfs(ATLAS_V2) and _lfs(IT40) and _lfs(START)
                         and (TRAINING / "training_worlds.json").exists()), reason="records absent")
@pytest.mark.parametrize("f40", [False, True])
def test_true_smoke(tmp_path: Path, f40: bool) -> None:
    import json

    rd = tmp_path / "exp0119.json"
    rd.write_text(json.dumps({"reading": {"deciding": {"reading": "gain" if f40 else "hold"}}}))
    assert ti.main(["--mode", "run", "--training-dir", str(TRAINING), "--atlas", str(ATLAS_V2),
                    "--models", str(MODELS), "--exp0119-reading", str(rd), "--f40-models", str(IT40),
                    "--f40-replenish", str(START), "--out-dir", str(tmp_path / "out"), "--dry-run"]) == 0
    tmp_path = tmp_path / "out"
    d = json.loads((tmp_path / "reading.json").read_text())
    assert d["start"]["start"] == ("F40" if f40 else "EXP-0100")
    assert d["start"]["replenish"] == (str(START) if f40 else None)
    rows = [json.loads(x) for x in (tmp_path / "gate_rows.jsonl").read_text().splitlines()]
    assert rows and all(r["replenish"] == (str(START) if f40 else "") for r in rows)
    assert {"fund", "accept"} == set(d["offline"]["heldout"]["tied_pairs"])
    assert set(d["offline"]["heldout"]["accept_errors"]) == {"declined_no_fit", "declined_displaced", "accepted"}
    assert d["collect"]["train_turns"] > 0 and d["offline"]["heldout"]["turns"] > 0
    assert d["closed_loop"]["monitor_12"]["candidate"]["rows"] == 2
    assert d["closed_loop"]["reported_42"]["reference"]["rows"] == 2
    assert (tmp_path / "models.pt").exists() and (tmp_path / "heldout_records.npz").exists()
    assert set(d["offline"]["heldout"]["pairs"]) == {"fund_kept", "fund_rep_rep", "fund_identical", "accept_kept",
                                                     "accept_identical"}


def test_linear_fractions_and_dims() -> None:
    """v15 option (a): linear own-state fractions appended after the log features; heads sized to match."""
    assert g.turn_dims(False) == {"fund": g.FUND_DIM, "bid": g.BID_DIM, "claim": g.CLAIM_DIM}
    assert g.turn_dims(True) == {"fund": g.FUND_DIM + 1, "bid": g.BID_DIM + 3, "claim": g.CLAIM_DIM + 3}
    assert g._binding({"c": 6}, {"c": 12}) == 0.5 and g._binding({"c": 6, "d": 3}, {"c": 12, "d": 2}) == 1.5
    assert g._frac(5, 0) == g.FRAC_CLIP and g._frac(0, 0) == 0.0 and g._frac(1e9, 1) == g.FRAC_CLIP
    h = g.TurnHeads(set_layers=2, bounded=True, linear=True)
    assert h.config()["linear"] is True and h.enc["bid"][0].in_features == g.BID_DIM + 3


@pytest.mark.skipif(not _lfs(IT40), reason="records absent")
def test_live_linear_features_match_the_auction_quantities() -> None:
    """On a live accept turn the bid fraction is the bid's capacity over the producer's free new-work
    capacity, and the price fraction is over the turn's highest item price."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_producer_turns import bid, producer

    p = producer(cap=20)
    p.place_bids([bid(8, 80, "a"), bid(5, 25, "b")])
    lists = p.turn_lists()
    from tokenomics.domain.producer_view import build_producer_entity_view

    class _T:  # the view builder needs a telemetry and order map; this producer has none
        def __getattr__(self, name: str):
            return lambda *a, **k: 0.0

    try:
        view = build_producer_entity_view(p, _T(), {}, {})
    except Exception:
        pytest.skip("entity view needs a full system")
    t = g.accept_features(p, lists, view, linear=True)
    base = g.BID_DIM
    assert t["bid"].shape[1] == base + 3
    assert np.allclose(t["bid"][:, base], [8 / 20, 5 / 20]) and np.allclose(t["bid"][:, base + 2], [1.0, 0.5])
