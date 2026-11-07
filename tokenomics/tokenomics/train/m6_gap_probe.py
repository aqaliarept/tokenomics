"""EXP-0116: why EXP-0114's training return rises while the greedy gate's units do not (supervisor
2026-11-07; DIAGNOSTICS 34 / 35 style discriminating check; pre-registration note
`notes/exp0116-prereg.md`).

Candidates, each one arm contrast on the same keyed world-episodes:
- **Replenish drift:** in training, input buying is the frozen start's deterministic output (`pm0`). The
  gate's greedy candidate (`gnn_producing_policy(pm)`) uses the trained model's own replenish head. That head
  had no gradient, but it reads the shared trunk, which RL moved. Contrast: `start_rep` (greedy gates,
  the start's replenish = the trained policy made deterministic) against `own` (as gated).
- **Greedy vs tempered:** training plays tempered gates (v9 TEMPS). Contrast: `tempered` arms (tempered
  gates, the start's replenish, a seeded draw per row) of the candidate against the start.
- **Monitor resolution / set:** the same contrasts on the 112 training world-episodes (2 replicates).
- **Potential vs units:** each row's raw Φ terms (`PotentialTracker`), scaled as in training, per term.

Never an acceptance read. The orderer is the frozen EXP-0100 / start orderer in every arm (identical).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..env.potential import PotentialTracker, PotentialWeights
from ..env.spaces import ACTION_LOGIT_BOUND
from . import m6_gnn as g
from . import m6_imitation as im
from . import m6_stage0b as sb

PREREG_NOTE = "openspec/changes/add-m6-joint-mpg-gnn/notes/exp0116-prereg.md"
MODES = ("own", "start_rep", "tempered")
REPLICATES = (0, 1)


def arms(cands: Mapping[int, str], start: str, ref: str) -> dict[str, tuple[str, str]]:
    """name -> (producer model file, mode). `ref` = the gate's reference (EXP-0100 stack, own)."""
    out = {"ref": (ref, "own"), "start": (start, "own"), "T_start": (start, "tempered")}
    for it, path in sorted(cands.items()):
        out[f"G{it}"] = (path, "own")
        out[f"F{it}"] = (path, "start_rep")
    last = max(cands)
    out[f"T{last}"] = (cands[last], "tempered")
    return out


def plan(cands: Mapping[int, str], start: str, ref: str, mon: Sequence, train: Sequence
         ) -> list[tuple[str, str, Sequence]]:
    """(arm, set, world-episodes): every arm on the monitor; ref / start / G_last / F_last / T_start /
    T_last also on the training world-episodes."""
    a = arms(cands, start, ref)
    last = max(cands)
    on_train = {"ref", "start", "T_start", f"G{last}", f"F{last}", f"T{last}"}
    return [(n, "monitor", mon) for n in a] + [(n, "train", train) for n in a if n in on_train]


def _policy(mode: str, pm: g.ProducerGNN, pm0: g.ProducerGNN, rng: np.random.Generator) -> Any:
    def act(env: Any, agent: str) -> np.ndarray:
        pid = env._role[agent][1]
        if mode == "own":
            return g.producer_action(pm, env, pid)
        f = g.producer_features(env.system.producer_entity_view(pid), g.producer_context(env, pid),
                                pm.features)
        with torch.no_grad():
            gates, _ = pm.heads(f)
            _g0, rep0 = pm0(f)
        if mode == "tempered":
            bits = rng.random(3) < torch.sigmoid(sb.tempered_logits(gates)).numpy()
        else:
            bits = gates.numpy() > 0
        sp = env.spec_
        out = np.zeros(sp.producing_action_dim, dtype=np.float32)
        for gp, frac in zip(f["inputs"], rep0.numpy()):
            if gp in sp.input_index:
                out[sp.input_index[gp]] = float(frac) * ACTION_LOGIT_BOUND
        out[sp.num_inputs:sp.num_inputs + 3] = np.where(bits, ACTION_LOGIT_BOUND, -ACTION_LOGIT_BOUND)
        return out

    return act


def probe_job(job: dict) -> dict[str, str]:
    """One keyed episode of one arm: the gate row plus the raw Φ terms and the training-scaled Φ."""
    from ..eval.gate import row_from_stats, row_to_dict
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    env = im._episode_env(job)
    w = job["weights"]
    tr = PotentialTracker(PotentialWeights(**w))
    tr.attach(env)
    pm, om = im.load_models(job["models"])
    pm0, _ = im.load_models(job["start"])
    rng = np.random.default_rng([0, 116, int(job["episode"]) % (1 << 31), int(job["replicate"])])
    run_episode(env, _policy(job["mode"], pm, pm0, rng), g.gnn_ordering_policy(om),
                seed=job["episode"], action_seed=None)
    st = env.episode_stats
    row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                                     f"{job['arm']}@{job['set']}", job["replicate"], st))
    tot = tr.totals()
    s = job["unit_scale"]
    row.update({"arm": job["arm"], "set": job["set"], "mode": job["mode"],
                "units": repr(float(st["consumer_completed_units"])),
                **{f"phi_{k}": repr(float(v)) for k, v in tot.items()},
                "phi": repr(float(sum(tr.rewards(s)) if s is not None else math.nan)),
                "phi_units": repr(float(s * tot["units"]) if s is not None else math.nan),
                "phi_rundown": repr(-w["lam"] * tot["rundown"]),
                "phi_starving": repr(-w["mu"] * tot["starving"]),
                "phi_death": repr(-w["nu"] * tot["death"])})
    return row


def jobs_for(ts: Any, cands: Mapping[int, str], start: str, ref: str, mon: Sequence,
             train: Sequence, cal: Mapping, horizon: int) -> list[dict]:
    c12 = cal["per_size"]["12"]
    weights = {"lam": c12["lambda"], "mu": c12["mu"], "nu": float(horizon), "x": cal["x"],
               "y": cal["y"]}
    a = arms(cands, start, ref)
    out = []
    for name, sset, wes in plan(cands, start, ref, mon, train):
        path, mode = a[name]
        for j in im._jobs(ts, wes, sset, REPLICATES, horizon, path):
            bar = cal["bar"].get(f"{j['world']}|{j['episode']}")
            out.append({**j, "arm": name, "set": sset, "mode": mode, "start": start,
                        "weights": weights, "unit_scale": horizon / bar if bar else None})
    return out


# --- reading (pre-registered in PREREG_NOTE) ---


def _paired(rows: Sequence[Mapping[str, str]], a: str, b: str, sset: str, col: str = "units"
            ) -> dict[str, Any]:
    """Mean of a - b over the shared (world, episode, replicate) keys, SE over world-episode means."""
    def by(arm: str) -> dict[tuple, float]:
        return {(r["world"], r["episode"], r["replicate"]): float(r[col]) for r in rows
                if r["arm"] == arm and r["set"] == sset}
    x, y = by(a), by(b)
    keys = sorted(set(x) & set(y))
    if not keys:
        return {"n": 0}
    we: dict[tuple, list[float]] = {}
    for k in keys:
        we.setdefault(k[:2], []).append(x[k] - y[k])
    means = [statistics.fmean(v) for v in we.values()]
    se = statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else math.nan
    return {"n": len(keys), "world_episodes": len(means), "diff": statistics.fmean(means), "se": se}


def _floor(rows: Sequence[Mapping[str, str]], a: str, b: str, sset: str, cells: Mapping) -> dict:
    """M5's floors rule (the gate's g / F) for a against b on one set."""
    from .producer_read import pilot_reading

    sel = [{**r, "kind": "monitor", "generation": "0"} for r in rows
           if r["arm"] == a and r["set"] == sset]
    sel += [{**r, "kind": "monitor_base"} for r in rows if r["arm"] == b and r["set"] == sset]
    r = pilot_reading(sel, cells, 0)
    return {"g": r.get("g_tr"), "F": r.get("F_tr")}


def reading(rows: Sequence[Mapping[str, str]], last: int, cells: Mapping,
            gate_units: float | None) -> dict[str, Any]:
    G, F, T = f"G{last}", f"F{last}", f"T{last}"
    vs_ref = {a: _floor(rows, a, "ref", "monitor", cells) for a in sorted(
        {r["arm"] for r in rows if r["set"] == "monitor"} - {"ref"})}
    drift = _paired(rows, F, G, "monitor")
    rl_greedy = _floor(rows, F, "start", "monitor", cells)
    rl_tempered = _floor(rows, T, "T_start", "monitor", cells)
    train = {"rl_greedy": {**_paired(rows, F, "start", "train"), **_floor(rows, F, "start", "train", cells)},
             "rl_tempered": {**_paired(rows, T, "T_start", "train"),
                             **_floor(rows, T, "T_start", "train", cells)},
             "drift": _paired(rows, F, G, "train")}
    terms = {c: _paired(rows, T, "T_start", "train", c) for c in
             ("phi", "phi_units", "phi_rundown", "phi_starving", "phi_death")}
    f_ref = vs_ref[F]
    flags = {
        "replenish_drift": bool(drift.get("diff", 0) > 0.5 * (vs_ref[G]["F"] or math.inf)
                                and (f_ref["g"] or -math.inf) > (f_ref["F"] or math.inf)),
        "greedy_loses_it": bool((rl_tempered["g"] or -math.inf) > (rl_tempered["F"] or math.inf)
                                and (rl_greedy["g"] or math.inf) <= (rl_greedy["F"] or 0)),
        "not_units": bool(terms["phi"].get("diff", 0) > 2 * terms["phi"].get("se", math.inf)
                          and terms["phi_units"].get("diff", 0) < 0.5 * terms["phi"].get("diff", 0)
                          and (train["rl_tempered"]["g"] or 0) <= (train["rl_tempered"]["F"] or 0)),
        "monitor_power": bool(train["rl_greedy"].get("diff", 0) > 2 * train["rl_greedy"].get("se", math.inf)
                              and (rl_greedy["g"] or math.inf) <= (rl_greedy["F"] or 0)),
    }
    order = ("replenish_drift", "greedy_loses_it", "not_units", "monitor_power")
    primary = next((k for k in order if flags[k]), "unexplained")
    gate_rep = None
    if gate_units is not None:
        gu = [float(r["units"]) for r in rows if r["arm"] == G and r["set"] == "monitor"]
        gate_rep = "MATCH" if gu and abs(statistics.fmean(gu) - gate_units) < 1e-6 else "MISMATCH"
    return {"primary": primary, "flags": flags, "vs_ref_monitor": vs_ref,
            "drift_monitor": drift, "rl_greedy_monitor": rl_greedy,
            "rl_tempered_monitor": rl_tempered, "train": train, "phi_terms_train_tempered": terms,
            "gate_reproduction": gate_rep}


def main(argv: Sequence[str] | None = None) -> int:
    from ..eval.teacher_audit import load_atlas
    from .order_es_validate import monitor_cells

    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--exp0110", required=True, help="dir with start_models.pt, calibration.json")
    ap.add_argument("--ref-models", required=True)
    ap.add_argument("--candidate", action="append", required=True, help="IT=path")
    ap.add_argument("--gate-units", type=float, default=None, help="the last candidate's gate units")
    ap.add_argument("--ref-rows", default=None, help="EXP-0114 ref_rows.json (row-for-row check)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="smoke: 1 monitor + 1 train world-episode, horizon 4")
    a = ap.parse_args(argv)
    im.SOURCE_COMMIT = a.source_commit
    cands = {int(c.split("=", 1)[0]): c.split("=", 1)[1] for c in a.candidate}
    e10 = Path(a.exp0110)
    start = str(e10 / "start_models.pt")
    cal = json.loads((e10 / "calibration.json").read_text())
    ts, train, mon = im._training(a.training_dir)
    horizon = a.horizon
    if a.dry_run:
        mon, train, horizon = mon[:1], train[:1], 4
    elif not a.source_commit or len(a.source_commit) != 40:
        im._refuse(["a real run needs a 40-hex --source-commit"])
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(ts, cands, start, a.ref_models, mon, train, cal, horizon)
    rows = im._pool(probe_job, jobs, a.workers)
    with (out / "rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    cells = monitor_cells(ts.training, load_atlas(a.atlas)["spreads"])
    rd = reading(rows, max(cands), cells, None if a.dry_run else a.gate_units)
    if a.ref_rows and not a.dry_run:
        want = {(r["world"], str(r["episode"]), str(r["replicate"])): float(r["units"])
                for r in json.loads(Path(a.ref_rows).read_text())}
        got = {(r["world"], str(r["episode"]), str(r["replicate"])): float(r["units"]) for r in rows
               if r["arm"] == "ref" and r["set"] == "monitor"}
        rd["ref_reproduction"] = "MATCH" if want == got else "MISMATCH"
    valid = rd.get("gate_reproduction") in (None, "MATCH") and rd.get("ref_reproduction") in (None, "MATCH")
    rd["valid"] = valid
    from ..eval.provenance import sha256_file

    note = Path(__file__).resolve().parents[3] / PREREG_NOTE
    doc = {"reading": rd, "dry_run": a.dry_run, "source_commit": a.source_commit,
           "prereg_note_sha256": sha256_file(str(note)) if note.exists() else None,
           "jobs": len(jobs), "horizon": horizon, "candidates": cands, "start": start,
           "ref": a.ref_models}
    (out / "reading.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[gap_probe] valid {valid} primary {rd['primary']} flags {rd['flags']}", flush=True)
    return 0 if valid else 2


if __name__ == "__main__":
    sys.exit(main())
