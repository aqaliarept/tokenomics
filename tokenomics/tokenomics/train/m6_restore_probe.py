"""EXP-0109: is the v4 producer's units shortfall under-restoration? (closed loop, horizon 200)

`python -m tokenomics.train.m6_restore_probe --mode calibrate|run ...`

Arms (supervisor decision 2026-11-06, after EXP-0108):
- `v4c`: the v4 producer (EXP-0107) + the EXP-0100 orderer (EXP-0108's candidate C);
- `v2`: the EXP-0100 stack;
- `base`: k5_rf5 + MWWk;
- `v4c_bias`: v4c with the restore gate's bias shifted by b, so its restore firing on the 12-producer
  monitor samples (k5_rf5's recorded decisions) equals k5_rf5's label rate (`calibrate`, offline).

Sets: the 12-producer monitor world-episodes x replicates 0 / 1, and the four non-held-out 42-producer
worlds with EXP-0107's largest shortfall (FOUR_42), 4 episodes (stage m6-drain) x replicates 0 / 1; all at
horizon 200. Per episode: per cycle, units, mean capacity ratio and mean capacity, the producers that ask
restore / upgrade / expand, investment orders placed / funded / completed / rejected by type (read-only
tracer, `m6_drain.DrainTracer`), and the loss classes (`eval/supply_trace.SupplyObserver`). Rows stream to
`episodes.jsonl` as they finish, so a stopped run keeps its partial file.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from . import m6_gnn as g
from . import m6_imitation as im

FOUR_42 = ("ws-atlas-v1-42-0000", "ws-atlas-v1-42-0043", "ws-atlas-v1-42-0076",
           "ws-atlas-v1-42-0039")  # EXP-0107's four largest 42 shortfalls (−3,150 … −2,132)
HORIZON = 200
BINS = ((0, 50), (50, 100), (100, 150), (150, 200))
TYPES = ("RESTORATION", "UPGRADE", "INVENTORY_EXPANSION")
RULE = ("recovered share r = (units[v4c_bias] − units[v4c]) / (units[base] − units[v4c]) per set "
        "(12 monitor, 42 four); under_restoration if r >= 0.5 at both sets; partial if r >= 0.2 at "
        "both or >= 0.5 at one; else other (the shortfall is not explained by the restore rate). "
        "Descriptive: per-cycle restore asks / completions, capacity and losses per arm")


def calibrate(v4: str, monitor_pkl: str) -> dict:
    """b such that mean(restore_logit + b > 0) on the monitor samples = k5_rf5's restore label rate."""
    pm, _om = im.load_models(v4)
    with open(monitor_pkl, "rb") as fh:
        prod = pickle.load(fh)["producer"]
    logits, labels = [], []
    with torch.no_grad():
        for f, y in prod:
            gates, _rep = pm(g.v3_to_v4(f) if f["capacity"].shape[1] == 17 else f)
            logits.append(float(gates[0]))
            labels.append(float(y["toggles"][0]))
    lg, lab = np.asarray(logits), np.asarray(labels)
    target = float(lab.mean())
    b = -float(np.quantile(lg, 1.0 - target))  # fire = logit + b > 0 on a `target` share
    return {"label_rate": target, "fire_rate_before": float((lg > 0).mean()),
            "fire_rate_after": float((lg + b > 0).mean()), "bias_shift": b,
            "agreement_before": float(((lg > 0) == (lab > 0.5)).mean()),
            "agreement_after": float(((lg + b > 0) == (lab > 0.5)).mean()), "decisions": len(lg)}


def models_for(v4: str, prev: str, b: float, out: Path) -> dict[str, str]:
    cand = im.compose_candidate(v4, prev, out / "v4c_models.pt")
    pm, om = im.load_models(cand)
    with torch.no_grad():
        pm.gates.bias[0] += b
    im.save_models(pm, om, out / "v4c_bias_models.pt")
    return {"v4c": cand, "v2": prev, "base": "", "v4c_bias": str(out / "v4c_bias_models.pt")}


def restore_job(job: dict) -> dict:
    """One keyed episode: per-cycle series from the read-only tracers (module docstring)."""
    from ..eval.baselines import producing_policy
    from ..eval.routing_audit import ordering_policy
    from ..eval.runner import run_episode
    from ..eval.supply_trace import SupplyObserver
    from .m6_drain import DrainTracer, over_ceiling

    torch.set_num_threads(1)
    env = im._episode_env(job)
    tr = DrainTracer(job["x"])
    tr.attach(env)
    obs = SupplyObserver()
    obs.attach(env)  # the outer complete_cycle wrapper
    if job["arm"] == "base":
        p, o = producing_policy(im.BASE_PRODUCER), ordering_policy(im.BASE_ORDERER)
    else:
        pm, om = im.load_models(job["models"])
        p, o = g.gnn_producing_policy(pm), g.gnn_ordering_policy(om)
    run_episode(env, p, o, seed=job["episode"], action_seed=None)
    st = env.episode_stats
    cycles = sorted(tr.ratio)
    first = cycles[0] if cycles else 0
    per: dict[str, list[float]] = {k: [0.0] * len(cycles) for k in (
        "units", "ratio", "cap", "decisions", "ask_restore", "running_restore", "ask_upgrade",
        "ask_expand", *(f"{t}_{x}" for t in TYPES for x in ("placed", "funded", "done",
                                                           "rejected")))}
    idx = {c: i for i, c in enumerate(cycles)}
    for i, c in enumerate(cycles):
        per["units"][i] = tr.units[i] if i < len(tr.units) else 0.0
        per["cap"][i] = tr.cap[i] if i < len(tr.cap) else 0.0
        per["ratio"][i] = float(np.mean(list(tr.ratio[c].values()))) if tr.ratio[c] else 0.0
    for (c, _pid), a in tr.asked.items():
        i = idx.get(c)
        if i is None:
            continue
        per["decisions"][i] += 1
        per["ask_restore"][i] += a["restore"]
        per["running_restore"][i] += a["running"]
        per["ask_upgrade"][i] += a["upgrade"]
        per["ask_expand"][i] += a["expand"]
    for c, _pid, t, funded, _tok, _cost in tr.orders:
        if c in idx:
            per[f"{t}_placed"][idx[c]] += 1
            per[f"{t}_funded"][idx[c]] += funded
    for c, _pid, t, tag in tr.fates:
        if c in idx:
            per[f"{t}_{'done' if tag == 'done' else 'rejected'}"][idx[c]] += 1
    return {"set": job["set"], "arm": job["arm"], "world": job["world"],
            "episode": job["episode"], "replicate": job["replicate"], "dead": bool(st["dead"]),
            "cycles": int(st["cycles"]), "units": float(st["consumer_completed_units"]),
            "first_cycle": first, "over_ceiling": over_ceiling(env.system), "series": per,
            "losses": {k: float(v) for k, v in obs.cells().items()
                       if k.startswith("st_") and not k.startswith(("st_producer_cycles",
                                                                    "st_late_"))
                       and _is_float(v)}}


def _is_float(v: str) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def jobs(training_dir: str, atlas: Any, models: dict[str, str], max_we: int | None,
         episodes_42: int | None, horizon: int) -> list[dict]:
    from ..eval.teacher_audit import world_view
    from .m6_drain import DRAIN_STAGE
    from .order_es_worlds import X_RUNDOWN
    from .producer_read import entry_for

    ts, _train, mon = im._training(training_dir)
    mon = mon[:max_we] if max_we else mon
    out = []
    for arm, mdl in models.items():
        for j in im._jobs(ts, mon, "monitor", im.MONITOR_REPLICATES, horizon, mdl or None):
            out.append({**j, "arm": arm, "models": mdl, "set": "12", "x": float(X_RUNDOWN)})
    views = {world_view(e)["world"]: world_view(e) for e in atlas["worlds"]}
    for w in FOUR_42[:1] if max_we else FOUR_42:
        e = entry_for(views[w], episodes_42 or 4, DRAIN_STAGE)
        for ep in e.episode_seeds:
            for r in (0, 1):
                for arm, mdl in models.items():
                    out.append({"world": e.id, "size": e.size, "regime": e.regime,
                                "world_seed": e.world_seed, "generator": e.generator,
                                "episode": int(ep), "replicate": r, "horizon": horizon,
                                "arm": arm, "models": mdl, "set": "42", "x": float(X_RUNDOWN)})
    return sorted(out, key=lambda j: (j["set"] != "42", j["arm"] == "base"))


def summarise(rows: Sequence[dict]) -> dict:
    out: dict[str, Any] = {"rule": RULE, "sets": {}}
    for s in ("12", "42"):
        per_arm = {}
        for arm in ("v4c", "v2", "base", "v4c_bias"):
            rs = [r for r in rows if r["set"] == s and r["arm"] == arm]
            if not rs:
                continue

            def binned(key: str, denom: str | None = None) -> list[float | None]:
                vals = []
                for lo, hi in BINS:
                    num = sum(sum(r["series"][key][lo:hi]) for r in rs)
                    den = sum(sum(r["series"][denom][lo:hi]) for r in rs) if denom else \
                        sum(len(r["series"][key][lo:hi]) for r in rs)
                    vals.append(num / den if den else None)
                return vals

            losses: dict[str, float] = {}
            for r in rs:
                for k, v in r["losses"].items():
                    losses[k] = losses.get(k, 0.0) + v / len(rs)
            per_arm[arm] = {
                "episodes": len(rs), "deaths": sum(r["dead"] for r in rs),
                "units": fmean(r["units"] for r in rs),
                "units_per_cycle_by_bin": binned("units"),
                "ratio_by_bin": binned("ratio"), "cap_by_bin": binned("cap"),
                "restore_ask_rate_by_bin": binned("ask_restore", "decisions"),
                "restore_running_rate_by_bin": binned("running_restore", "decisions"),
                "upgrade_ask_rate_by_bin": binned("ask_upgrade", "decisions"),
                "expand_ask_rate_by_bin": binned("ask_expand", "decisions"),
                "restore_ask_rate": sum(sum(r["series"]["ask_restore"]) for r in rs)
                / max(1, sum(sum(r["series"]["decisions"]) for r in rs)),
                "orders": {t: {x: fmean(sum(r["series"][f"{t}_{x}"]) for r in rs)
                               for x in ("placed", "funded", "done", "rejected")} for t in TYPES},
                "over_ceiling": sum(r["over_ceiling"] for r in rs),
                "losses_mean": dict(sorted(losses.items(), key=lambda kv: -kv[1])[:12])}
        if {"v4c", "base", "v4c_bias"} <= set(per_arm):
            gap = per_arm["base"]["units"] - per_arm["v4c"]["units"]
            per_arm["recovered_share"] = ((per_arm["v4c_bias"]["units"] - per_arm["v4c"]["units"])
                                          / gap if gap else None)
        out["sets"][s] = per_arm
    r12 = out["sets"].get("12", {}).get("recovered_share")
    r42 = out["sets"].get("42", {}).get("recovered_share")
    if r12 is None or r42 is None:
        out["reading"] = None
    elif r12 >= 0.5 and r42 >= 0.5:
        out["reading"] = "under_restoration"
    elif (r12 >= 0.2 and r42 >= 0.2) or r12 >= 0.5 or r42 >= 0.5:
        out["reading"] = "partial"
    else:
        out["reading"] = "other"
    return out


def run(training_dir: str, atlas_path: str, v4: str, prev: str, monitor_pkl: str, out_dir: str,
        workers: int, max_we: int | None = None, episodes_42: int | None = None,
        horizon: int = HORIZON) -> int:
    from ..eval.teacher_audit import load_atlas

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cal = calibrate(v4, monitor_pkl)
    (out / "calibration.json").write_text(json.dumps(cal, indent=1) + "\n")
    print(f"[m6_restore_probe] calibration {cal}", flush=True)
    models = models_for(v4, prev, cal["bias_shift"], out)
    js = jobs(training_dir, load_atlas(atlas_path), models, max_we, episodes_42, horizon)
    rows = []
    path = out / "episodes.jsonl"
    with path.open("w") as fh:
        if workers <= 1:
            for j in js:
                rows.append(restore_job(j))
                fh.write(json.dumps(rows[-1]) + "\n")
                fh.flush()
        else:
            from .producer_es import _init_worker

            with ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                                     initializer=_init_worker) as ex:
                for fut in as_completed([ex.submit(restore_job, j) for j in js]):
                    rows.append(fut.result())
                    fh.write(json.dumps(rows[-1]) + "\n")
                    fh.flush()
    doc = {**summarise(rows), "calibration": cal, "jobs": len(js), "rows": len(rows),
           "source_commit": im.SOURCE_COMMIT, "four_42": list(FOUR_42), "horizon": horizon}
    (out / "summary.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[m6_restore_probe] reading {doc['reading']} recovered 12 "
          f"{doc['sets'].get('12', {}).get('recovered_share')} 42 "
          f"{doc['sets'].get('42', {}).get('recovered_share')}", flush=True)
    return 0 if len(rows) == len(js) else 2


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=("calibrate", "run"), default="run")
    p.add_argument("--training-dir")
    p.add_argument("--atlas")
    p.add_argument("--v4", required=True)
    p.add_argument("--prev")
    p.add_argument("--monitor-samples", required=True)
    p.add_argument("--out-dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-world-episodes", type=int)
    p.add_argument("--episodes-42", type=int)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--source-commit", default="")
    a = p.parse_args(argv)
    im.SOURCE_COMMIT = a.source_commit
    torch.set_num_threads(1)
    if a.mode == "calibrate":
        print(json.dumps(calibrate(a.v4, a.monitor_samples), indent=1))
        return 0
    return run(a.training_dir, a.atlas, a.v4, a.prev, a.monitor_samples, a.out_dir, a.workers,
               a.max_world_episodes, a.episodes_42, a.horizon)


if __name__ == "__main__":
    sys.exit(main())
