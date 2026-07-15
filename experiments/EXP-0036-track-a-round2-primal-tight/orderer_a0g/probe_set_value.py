"""A0g: producer-count-agnostic value table (EXP-0036 HANDOVER "Proposed Model") evaluated
on the saved A0f feature traces — a local saved-artifact recompute (HANDOVER-allowed local).

Same information as the flat A0f ladder, reshaped into [context ‖ per-producer row set] and
fed to `ProducerSetValueModel` (shared producer MLP → masked mean+max pool → fixed candidate
head). Because the data is identical, held-out regret deltas vs the flat ladder isolate the
SET inductive bias. Reads emitted per task:

  set            context (stream ‖ order_queue ‖ compact) + producer rows, 2 train seeds
  set+history    context ‖ previous-4-cycle compact history
  set+history 1-seed        learning-curve point (anti-transfer check, matches A0f analysis)
  set+history pooled        ONE model fit on all three tasks' train traces (possible only
                            because the set view gives identical model shape across tasks:
                            row dim 108, context dim 230; P=6 vs 9 is just the set size) —
                            the cross-task coverage rung the flat probes could not test

Constant floors recomputed per task for the same comparison frame as analysis_followup.py.
Run from tokenomics/: ../.venv/bin/python <this file>
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from tokenomics.train.orderer_scorer import (
    CycleTrace, ProducerSetValueModel, fit_set_value_table, history_features,
    set_features_from_blocks, set_value_table_metrics)

EXP = Path("/workspaces/tokenomics/experiments/EXP-0036-track-a-round2-primal-tight")
TRACES = EXP / "orderer_a0f" / "artifacts" / "feature_traces.json.gz"
OUT = EXP / "orderer_a0g"
ITERS = 1000
K_HISTORY = 4


def infer_ptg(dims: dict[str, int]) -> tuple[int, int, int]:
    g = (dims["order_queue"] - 2) // 11
    p = dims["inventory"] // g
    t = dims["per_type_capacity"] // (4 * p)
    assert dims["relational"] == 6 * p * g and dims["inventory"] == p * g \
        and dims["per_type_capacity"] == 4 * p * t and dims["order_queue"] == 11 * g + 2
    return p, t, g


def load(rows: list[dict], ptg: tuple[int, int, int]):
    traces = [CycleTrace(seed=r["seed"], cycle=r["cycle"], cand_units=r["cand_units"],
                         choice=r["choice"], margin=r["margin"], champion_units=None,
                         state={}, features={k: np.asarray(v, dtype=np.float32)
                                             for k, v in r["features"].items()})
              for r in rows]
    p, t, g = ptg
    sets = [set_features_from_blocks(tr.features, n_producers=p, n_types=t, n_products=g)
            for tr in traces]
    ctx = np.stack([s["context"] for s in sets])
    prod = np.stack([s["producers"] for s in sets])
    hist = history_features(traces, k=K_HISTORY)
    y = np.asarray([tr.cand_units for tr in traces], dtype=np.float32)
    return traces, ctx, prod, hist, y


def fit_eval(ctx_tr, prod_tr, y_tr, evals, even: int, mask_tr=None) -> list[dict]:
    torch.manual_seed(0)  # same init across variants — regret deltas read as representation
    model = ProducerSetValueModel(context_dim=ctx_tr.shape[1],
                                  producer_dim=prod_tr.shape[2], n_choices=y_tr.shape[1])
    model.set_norm(ctx_tr, prod_tr.reshape(-1, prod_tr.shape[2]))
    fit_set_value_table(model, ctx_tr, prod_tr, y_tr, iters=ITERS, mask=mask_tr)
    return [dict(set_value_table_metrics(model, c, p, y, even, mask=m), split=name)
            for name, c, p, y, m in evals]


def regret_frac_const(y: np.ndarray, pick: int, even: int) -> float | None:
    best = y.max(axis=1)
    head = float((best - y[:, even]).sum())
    return round(float((best - y[:, pick]).sum() / head), 4) if head > 1e-9 else None


def main() -> None:
    with gzip.open(TRACES, "rt") as f:
        details = json.load(f)
    tasks = []
    for det in details:
        ptg = infer_ptg({k: len(v) for k, v in det["train"][0]["features"].items()})
        tasks.append((det, ptg, load(det["train"], ptg), load(det["heldout"], ptg)))

    rows: list[dict] = []
    for det, ptg, (tr, ctx_tr, prod_tr, hist_tr, y_tr), \
            (ho, ctx_ho, prod_ho, hist_ho, y_ho) in tasks:
        sid = det["task_id"].split("-s")[-1]
        vocab = det["vocab"]
        even = vocab.index("even_split")
        P, T, G = ptg
        print(f"\n=== s{sid}  P={P} T={T} G={G}  row_dim={prod_tr.shape[2]} "
              f"ctx_dim={ctx_tr.shape[1]}  n_train={len(tr)} n_heldout={len(ho)}")

        consts = {v: regret_frac_const(y_ho, j, even) for j, v in enumerate(vocab)}
        print("  constant floors (heldout):", consts)

        variants = [
            ("set", ctx_tr, ctx_ho, np.ones(len(tr), bool)),
            ("set+history", np.concatenate([ctx_tr, hist_tr], axis=1),
             np.concatenate([ctx_ho, hist_ho], axis=1), np.ones(len(tr), bool)),
            ("set+history 1-seed", np.concatenate([ctx_tr, hist_tr], axis=1),
             np.concatenate([ctx_ho, hist_ho], axis=1),
             np.asarray([t.seed == tr[0].seed for t in tr])),
        ]
        for name, xc_tr, xc_ho, keep in variants:
            ms = fit_eval(xc_tr[keep], prod_tr[keep], y_tr[keep],
                          [("train", xc_tr[keep], prod_tr[keep], y_tr[keep], None),
                           ("heldout", xc_ho, prod_ho, y_ho, None)], even)
            m_tr, m_ho = ms
            print(f"  {name:<20} n_fit={int(keep.sum()):>3}  "
                  f"train regret_frac {m_tr['regret_frac']}  "
                  f"heldout regret_frac {m_ho['regret_frac']}  top1 {m_ho['top1']}  "
                  f"winner_hit {m_ho['winner_hit']}")
            rows.append({"task_id": det["task_id"], "world_seed": det["world_seed"],
                         "variant": name, "n_producers": P, "producer_dim": prod_tr.shape[2],
                         "context_dim": xc_tr.shape[1], "n_fit": int(keep.sum()),
                         "n_heldout": len(ho),
                         "train_regret_frac": m_tr["regret_frac"],
                         "train_top1": m_tr["top1"],
                         "regret_frac": m_ho["regret_frac"], "top1": m_ho["top1"],
                         "winner_hit": m_ho["winner_hit"],
                         "regret_total": m_ho["regret_total"],
                         "headroom_total": m_ho["headroom_total"],
                         "const_floors": json.dumps(consts)})

    # pooled: one model on all tasks' train traces (variable P handled by padding+mask)
    even = tasks[0][0]["vocab"].index("even_split")
    pmax = max(t[2][2].shape[1] for t in tasks)

    def pad(prod: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, p, d = prod.shape
        out = np.zeros((n, pmax, d), dtype=np.float32)
        out[:, :p] = prod
        mask = np.zeros((n, pmax), dtype=np.float32)
        mask[:, :p] = 1.0
        return out, mask

    xc_tr = np.concatenate([np.concatenate([t[2][1], t[2][3]], axis=1) for t in tasks])
    padded = [pad(t[2][2]) for t in tasks]
    prod_tr = np.concatenate([p for p, _ in padded])
    mask_tr = np.concatenate([m for _, m in padded])
    y_tr = np.concatenate([t[2][4] for t in tasks])
    evals = [("train", xc_tr, prod_tr, y_tr, mask_tr)]
    for det, ptg, _, (ho, ctx_ho, prod_ho, hist_ho, y_ho) in tasks:
        p_ho, m_ho = pad(prod_ho)
        evals.append((f"heldout s{det['task_id'].split('-s')[-1]}",
                      np.concatenate([ctx_ho, hist_ho], axis=1), p_ho, y_ho, m_ho))
    print(f"\n=== pooled set+history: one model, {len(y_tr)} train rows across "
          f"{len(tasks)} tasks (P padded to {pmax} with mask)")
    for m in fit_eval(xc_tr, prod_tr, y_tr, evals, even, mask_tr=mask_tr):
        print(f"  {m['split']:<14} regret_frac {m['regret_frac']}  top1 {m['top1']}  "
              f"winner_hit {m['winner_hit']}")
        if m["split"].startswith("heldout"):
            rows.append({"task_id": m["split"].removeprefix("heldout "),
                         "world_seed": "", "variant": "set+history pooled",
                         "n_producers": pmax, "producer_dim": prod_tr.shape[2],
                         "context_dim": xc_tr.shape[1], "n_fit": len(y_tr),
                         "n_heldout": m["n"], "train_regret_frac": None,
                         "train_top1": None, "regret_frac": m["regret_frac"],
                         "top1": m["top1"], "winner_hit": m["winner_hit"],
                         "regret_total": m["regret_total"],
                         "headroom_total": m["headroom_total"], "const_floors": ""})

    with (OUT / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "spec.json").write_text(json.dumps({
        "stage": "a0g", "question": "does the set (producer-count-agnostic) value-table "
        "model of the HANDOVER 'Proposed Model' section beat the flat A0f probes on the "
        "same traces?", "traces": str(TRACES.relative_to(EXP)), "iters": ITERS,
        "k_history": K_HISTORY, "model": "ProducerSetValueModel(hidden=64, mean+max pool)",
        "target": "per-row-centered value-table MSE (fit_set_value_table)",
        "metric": "heldout regret_frac vs even_split headroom"}, indent=2) + "\n")
    print(f"\nwrote {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
