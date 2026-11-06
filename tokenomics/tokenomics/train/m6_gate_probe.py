"""EXP-0106: why the Stage I v3 producer over-fires its upgrade gate (EXP-0105 `short`), offline.

`python -m tokenomics.train.m6_gate_probe --collect-dir C --v2 V2.pt --v3 V3.pt --out-dir D [--stride 3]`

On EXP-0105's recorded producer decisions (k5_rf5's labels, v3 features), split by size (the training
set's 42-producer block, its 12-producer block, the 12-producer monitor set), three models:
- `v2`: EXP-0100's producer (16 capacity columns: the v3 margin column dropped, so its own features);
- `v3`: EXP-0105's producer;
- `v3_ablated`: v3 with the margin column set to its normaliser median (normalised value 0).

Per set and model: per-head confusion (label / predicted rates, accuracy), the upgrade gate's response
to the upgrade row's ceiling margin (fire rate, mean logit and label rate at margin < 0 against
margin >= 0; the OLS slope of the logit on the margin), and the producer BC loss. Also the label
balance per size and each size's share of the fit's producer decisions (each decision weighs the same
in `bc_losses`).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import m6_gnn as g
from . import m6_imitation as im

MARGIN_COL = 16  # v3's ceiling margin (the 17th capacity column)
UPGRADE_EFFECT_COL = 12  # log1p(upgrade_effect) > 0 only on the upgrade type's row
TOGGLES = ("restore", "upgrade", "expand")


def as_v2(f: dict) -> dict:
    return {**f, "capacity": f["capacity"][:, :MARGIN_COL]}


def ablated(f: dict, median: float) -> dict:
    cap = f["capacity"].copy()
    if cap.shape[0]:
        cap[:, MARGIN_COL] = median
    return {**f, "capacity": cap}


def model_stats(pm: g.ProducerGNN, samples: Sequence, adapt: Any) -> dict:
    """Confusion per head, the upgrade gate against the margin, and the producer BC loss."""
    conf = np.zeros((3, 4), dtype=np.int64)
    upg = []  # (margin, logit, label) per decision with an upgrade row
    loss = 0.0
    with torch.no_grad():
        for f, y in samples:
            ff = adapt(f)
            gates, rep = pm(ff)
            lg = gates.numpy()
            tg = (lg > 0).astype(np.float32)
            yt = y["toggles"]
            conf[:, 0] += (tg == 1) & (yt == 1)
            conf[:, 1] += (tg == 1) & (yt == 0)
            conf[:, 2] += (tg == 0) & (yt == 1)
            conf[:, 3] += (tg == 0) & (yt == 0)
            loss += float(torch.nn.functional.binary_cross_entropy_with_logits(
                gates, torch.from_numpy(yt), reduction="sum"))
            if rep.shape[0]:
                m = torch.from_numpy(y["mask"])
                loss += float((torch.nn.functional.huber_loss(
                    rep, torch.from_numpy(y["replenish"]), reduction="none", delta=0.1) * m).sum())
            cap = f["capacity"]
            rows = np.nonzero(cap[:, UPGRADE_EFFECT_COL] > 0)[0] if cap.shape[0] else []
            if len(rows):
                upg.append((float(cap[rows[0], MARGIN_COL]), float(lg[1]), float(yt[1])))
    n = max(1, len(samples))
    heads = {t: {"label_rate": (conf[i, 0] + conf[i, 2]) / n, "pred_rate": (conf[i, 0] + conf[i, 1]) / n,
                 "accuracy": (conf[i, 0] + conf[i, 3]) / n,
                 "tp_fp_fn_tn": [int(x) for x in conf[i]]} for i, t in enumerate(TOGGLES)}
    a = np.asarray(upg) if upg else np.zeros((0, 3))
    resp = {}
    for name, sel in (("margin_lt_0", a[:, 0] < 0), ("margin_ge_0", a[:, 0] >= 0)):
        s = a[sel]
        resp[name] = {"decisions": int(len(s)),
                      "label_rate": float(s[:, 2].mean()) if len(s) else None,
                      "fire_rate": float((s[:, 1] > 0).mean()) if len(s) else None,
                      "mean_logit": float(s[:, 1].mean()) if len(s) else None}
    if len(a) > 2 and np.ptp(a[:, 0]) > 0:
        slope, icpt = np.polyfit(a[:, 0], a[:, 1], 1)
        resp["logit_on_margin"] = {"slope": float(slope), "intercept": float(icpt)}
    return {"decisions": len(samples), "heads": heads, "upgrade_margin_response": resp,
            "producer_bc_loss": loss / n}


def run(collect_dir: str, v2: str, v3: str, out_dir: str, stride: int, limit: int | None) -> int:
    c = Path(collect_dir)
    meta = json.loads((c / "collect.json").read_text())
    n42 = int(meta["counts"]["train"]["producer_42"])
    with (c / "samples_train.pkl").open("rb") as fh:
        tr = pickle.load(fh)
    prod = tr["producer"]
    n_all = len(prod)
    del tr
    sets = {"42_train": prod[:n42][::stride], "12_train": prod[n42:][::stride]}
    del prod
    im.release_memory()
    with (c / "samples_monitor.pkl").open("rb") as fh:
        sets["12_monitor"] = pickle.load(fh)["producer"]
    if limit:
        sets = {k: v[:limit] for k, v in sets.items()}
    p2, _ = im.load_models(v2)
    p3, _ = im.load_models(v3)
    if p2.features != g.FEATURES_V2 or p3.features != g.FEATURES_V3:
        raise ValueError("--v2 / --v3 are not v2 / v3 producer checkpoints")
    med = float(p3.norm.c_capacity[MARGIN_COL])
    models = {"v2": (p2, as_v2), "v3": (p3, lambda f: f),
              "v3_ablated": (p3, lambda f: ablated(f, med))}
    out: dict[str, Any] = {
        "fit_share": {"42": n42 / n_all, "12": (n_all - n42) / n_all, "decisions": n_all,
                      "weighting": "each decision weighs the same in bc_losses"},
        "stride": stride, "margin_median": med,
        "capacity_rows": {k: {str(int(r)): int(cnt) for r, cnt in zip(*np.unique(
            [f["capacity"].shape[0] for f, _ in v], return_counts=True))} for k, v in sets.items()},
        "margin_lt_0_share_of_rows": {k: float(np.mean(np.concatenate(
            [f["capacity"][:, MARGIN_COL] for f, _ in v]) < 0)) for k, v in sets.items()},
        "restore_effect_0_share_of_rows": {k: float(np.mean(np.concatenate(
            [f["capacity"][:, 10] for f, _ in v]) == 0)) for k, v in sets.items()},
        "sets": {}}
    for k, v in sets.items():
        out["sets"][k] = {name: model_stats(pm, v, ad) for name, (pm, ad) in models.items()}
        print(f"[m6_gate_probe] {k}: done", flush=True)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "probe.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {m: {"upgrade_pred": s["heads"]["upgrade"]["pred_rate"],
                              "upgrade_label": s["heads"]["upgrade"]["label_rate"],
                              "lt0_fire": s["upgrade_margin_response"]["margin_lt_0"]["fire_rate"],
                              "loss": s["producer_bc_loss"]} for m, s in v.items()}
                      for k, v in out["sets"].items()}), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--collect-dir", required=True)
    p.add_argument("--v2", required=True)
    p.add_argument("--v3", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--limit", type=int, help="smoke: decisions per set")
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    return run(a.collect_dir, a.v2, a.v3, a.out_dir, a.stride, a.limit)


if __name__ == "__main__":
    sys.exit(main())
