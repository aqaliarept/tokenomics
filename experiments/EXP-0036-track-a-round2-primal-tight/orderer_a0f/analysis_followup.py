"""Local recompute over the pulled A0f feature traces (EXP-0036 HANDOVER Task 1 follow-up).

Three discriminating reads the ladder alone cannot give:
1. Constant-pick floors: heldout regret_frac of always picking one vocab entry (and the
   train-majority entry). If the fitted probes sit at/above these floors, they transfer zero
   signal.
2. Two-point learning curve: fit on 1 train seed (~100 cycles) vs 2 (~200), same features.
   Regret drops with data => sample starvation/coverage; flat => label/objective wall.
3. Task 4 target variants on the same features: winner-set multi-label BCE and
   margin-weighted CE, vs the value-regression numbers from the pod run.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import torch

from tokenomics.train.orderer_scorer import (
    ValueTableModel, fit_value_table, value_table_metrics, CycleTrace, history_features)

PATH = ("/workspaces/tokenomics/experiments/EXP-0036-track-a-round2-primal-tight/"
        "orderer_a0f/artifacts/feature_traces.json.gz")
ITERS = 1000
EPS = 1e-6


def to_traces(rows):
    return [CycleTrace(seed=r["seed"], cycle=r["cycle"], cand_units=r["cand_units"],
                       choice=r["choice"], margin=r["margin"], champion_units=None,
                       state={}, features={k: np.asarray(v, dtype=np.float32)
                                           for k, v in r["features"].items()})
            for r in rows]


def matrix(traces, blocks, k_history=4):
    hist = history_features(traces, k=k_history)
    cols = [hist if b == "history" else np.stack([t.features[b] for t in traces])
            for b in blocks]
    return np.concatenate(cols, axis=1)


def tables(traces):
    return np.asarray([t.cand_units for t in traces], dtype=np.float64)


def regret_frac_of_picks(y, picks, even):
    best = y.max(axis=1)
    head = (best - y[:, even]).sum()
    reg = (best - y[np.arange(len(y)), picks]).sum()
    return round(float(reg / head), 4) if head > 1e-9 else None


class WinnerSetModel(torch.nn.Module):
    """Multi-label winner-set head (HANDOVER Task 4 variant): BCE on 'is candidate within
    EPS of the row max'; pick = argmax logit."""

    def __init__(self, dim, k, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, k))
        self.mu, self.sd = None, None

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def fit_torch(model, x, loss_fn, iters=ITERS, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(iters):
        opt.zero_grad()
        loss = loss_fn(model(x))
        loss.backward()
        opt.step()
    model.eval()
    return model


def norm_of(x):
    return (torch.from_numpy(x.mean(axis=0)),
            torch.from_numpy(x.std(axis=0).clip(min=1e-6)))


def main():
    with gzip.open(PATH, "rt") as f:
        details = json.load(f)
    for det in details:
        tid = det["task_id"].split("-s")[-1]
        vocab = det["vocab"]
        even = vocab.index("even_split")
        train = to_traces(det["train"])
        held = to_traces(det["heldout"])
        y_tr, y_ho = tables(train), tables(held)
        print(f"\n=== s{tid}  vocab={vocab}  n_train={len(train)} n_heldout={len(held)}")

        flat = float(np.mean([(np.max(r) - np.min(r)) < EPS for r in y_ho]))
        print(f"heldout all-tied rows: {flat:.2%}")

        # 1. constant floors
        for j, name in enumerate(vocab):
            rf = regret_frac_of_picks(y_ho, np.full(len(y_ho), j), even)
            print(f"  constant {name:<12} heldout regret_frac {rf}")
        maj = int(np.bincount([t.choice for t in train], minlength=len(vocab)).argmax())
        print(f"  train-majority ({vocab[maj]}) heldout regret_frac "
              f"{regret_frac_of_picks(y_ho, np.full(len(y_ho), maj), even)}")

        blocks = ("stream", "per_type_capacity", "order_queue", "inventory", "inbound",
                  "history", "relational")
        x_tr, x_ho = matrix(train, blocks), matrix(held, blocks)
        xg_tr, xg_ho = matrix(train, ("gstate",)), matrix(held, ("gstate",))

        # 2. learning curve: seed A only vs both train seeds (value regression, full ladder)
        seeds = sorted({t.seed for t in train})
        for label, idx in (("1-seed", [i for i, t in enumerate(train) if t.seed == seeds[0]]),
                           ("2-seed", list(range(len(train))))):
            torch.manual_seed(0)
            m = ValueTableModel(state_dim=x_tr.shape[1], n_choices=len(vocab))
            m.set_norm(x_tr[idx])
            fit_value_table(m, x_tr[idx], y_tr[idx].astype(np.float32), iters=ITERS)
            mm = value_table_metrics(m, x_ho, y_ho.astype(np.float32), even)
            print(f"  value-probe full-ladder {label} (n={len(idx)}): "
                  f"heldout regret_frac {mm['regret_frac']}")

        # 3. Task 4 target variants (2-seed, full ladder + gstate)
        for fname, (xa, xb) in (("ladder", (x_tr, x_ho)), ("gstate", (xg_tr, xg_ho))):
            xt = torch.from_numpy(xa)
            wins = torch.from_numpy((y_tr >= y_tr.max(axis=1, keepdims=True) - EPS)
                                    .astype(np.float32))
            torch.manual_seed(0)
            ws = WinnerSetModel(xa.shape[1], len(vocab))
            ws.mu, ws.sd = norm_of(xa)
            bce = torch.nn.BCEWithLogitsLoss()
            fit_torch(ws, xt, lambda out: bce(out, wins))
            with torch.no_grad():
                picks = ws(torch.from_numpy(xb)).argmax(dim=1).numpy()
            print(f"  winner-set BCE ({fname}): heldout regret_frac "
                  f"{regret_frac_of_picks(y_ho, picks, even)}")

            # margin-weighted CE: ties carry ~0 weight, expensive calls dominate
            tgt = torch.tensor([t.choice for t in train])
            w = torch.tensor([t.margin for t in train], dtype=torch.float32)
            w = w / max(float(w.mean()), 1e-6)
            ce = torch.nn.CrossEntropyLoss(reduction="none")
            torch.manual_seed(0)
            mw = WinnerSetModel(xa.shape[1], len(vocab))
            mw.mu, mw.sd = norm_of(xa)
            fit_torch(mw, xt, lambda out: (ce(out, tgt) * w).mean())
            with torch.no_grad():
                picks = mw(torch.from_numpy(xb)).argmax(dim=1).numpy()
            print(f"  margin-wt CE  ({fname}): heldout regret_frac "
                  f"{regret_frac_of_picks(y_ho, picks, even)}")


if __name__ == "__main__":
    main()
