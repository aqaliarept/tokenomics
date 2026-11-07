"""EXP-0119: confirm F40 on fresh seeds before it becomes the M6 start (user decision 2026-11-07; pre-registration
note `notes/exp0119-prereg.md`).

F40 = EXP-0114's iteration-40 investment gates + the frozen start's replenish (EXP-0116's `F40` arm,
`m6_gap_probe` mode `start_rep`). The reference is the EXP-0100 stack. v11's metric and floors (M5's g / F;
`gain` iff g > F and deaths, run-down and starvation shares no worse; `hold` iff |g| ≤ F; `worse` otherwise).

Sets (fresh episode seeds, stage `m6-f40-confirm`, never played before):
- `monitor_fresh`: the 16 monitor worlds × 4 new episodes × r0 / r1;
- `validation_12`: the atlas v2 validation worlds at 12 producers (not `collapse`) × 4 episodes × r0 / r1;
- `validation_42` (reported only): the same at 42 producers.

Deciding: one pooled comparison over `monitor_fresh` ∪ `validation_12`. Each set is also reported alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import m6_gap_probe as gp
from . import m6_imitation as im

STAGE = "m6-f40-confirm"
EPISODES = 4
REPLICATES = (0, 1)


def _cell(record: Mapping, key: str, spreads: Mapping) -> Any:
    from ..eval.atlas import space_info
    from .order_es_validate import Cell, spread_from_dict

    info = space_info(record)
    sk = f"{info.size}/{info.choice}"
    return Cell(key, info.size, info.choice, info.differing,
                spread_from_dict(spreads[sk]) if sk in spreads else None)


def build_jobs(ts: Any, mon: Sequence, atlas: Mapping, arms: Mapping[str, tuple[str, str]], start: str,
               weights: Mapping, horizon: int, episodes: int, max_worlds: int | None = None
               ) -> tuple[list[dict], dict]:
    from ..eval.teacher_audit import world_view
    from ..eval.world_space import episode_seeds
    from .producer_read import entry_for

    jobs, cells = [], {}
    records = {w["world"]: w["record"] for w in ts.training["worlds"]}
    worlds = sorted({w for w, _ in mon})[:max_worlds]
    base = []
    for w in worlds:  # monitor worlds, fresh episodes (the training set's world entries)
        for ep in episode_seeds(w, episodes, STAGE):
            cells[f"{w}#{int(ep)}"] = _cell(records[w], f"{w}#{int(ep)}", atlas["spreads"])
            base.append(("monitor_fresh", w, int(ep), lambda w=w, ep=ep, r=0: im.pe.make_job(
                ts, kind="f40", producing=im.BASE_PRODUCER, theta=None, generation=0, member="f40", world=w,
                episode=int(ep), replicate=r, horizon=horizon)))
    views = [world_view(e) for e in atlas["worlds"]]
    for size, sset in ((12, "validation_12"), (42, "validation_42")):
        vs = sorted((v for v in views if v["size"] == size and v["split"] == "validation"
                     and v["class"] != "collapse"), key=lambda v: v["world"])[:max_worlds]
        for v in vs:
            e = entry_for(v, episodes, STAGE)
            for ep in e.episode_seeds:
                cells[f"{e.id}#{int(ep)}"] = _cell(v["record"], f"{e.id}#{int(ep)}", atlas["spreads"])
                base.append((sset, e.id, int(ep), lambda e=e, ep=ep, r=0: {
                    "world": e.id, "size": e.size, "regime": e.regime, "world_seed": e.world_seed,
                    "generator": e.generator, "episode": int(ep), "replicate": r, "horizon": horizon,
                    "kind": "f40"}))
    for sset, w, ep, mk in base:
        for r in REPLICATES:
            j = {**mk(), "replicate": r}
            for arm, (models, mode) in arms.items():
                jobs.append({**j, "arm": arm, "set": sset, "mode": mode, "models": models, "start": start,
                             "weights": dict(weights), "unit_scale": None,
                             "job_id": f"f40|{arm}|{w}|{ep}|{r}"})
    return jobs, cells


def reading(rows: Sequence[Mapping[str, str]], cells: Mapping) -> dict[str, Any]:
    from .m6_pilot import gate_reading
    from .order_es_validate import row_flags
    from .producer_read import pilot_reading

    def one(sets: Sequence[str]) -> dict[str, Any]:
        sel = [r for r in rows if r["set"] in sets]
        pr = pilot_reading([{**r, "kind": "monitor", "generation": "0"} for r in sel if r["arm"] == "F40"]
                           + [{**r, "kind": "monitor_base"} for r in sel if r["arm"] == "ref"], cells, 0)

        def flags(arm: str) -> dict[str, Any]:
            f = [row_flags(r) for r in sel if r["arm"] == arm]
            u = [float(r["units"]) for r in sel if r["arm"] == arm]
            return {"deaths": sum(x[0] for x in f), "rundown_share": sum(x[1] for x in f) / len(f) if f else None,
                    "starvation_share": sum(x[2] for x in f) / len(f) if f else None,
                    "units": sum(u) / len(u) if u else None, "rows": len(f)}

        gt = {"g": pr.get("g_tr"), "F": pr.get("F_tr"), "candidate": flags("F40"), "reference": flags("ref"),
              "world_episodes": pr.get("world_episodes")}
        gt["reading"] = gate_reading(gt, True) if gt["g"] is not None else None
        return gt

    return {"deciding": one(("monitor_fresh", "validation_12")), "monitor_fresh": one(("monitor_fresh",)),
            "validation_12": one(("validation_12",)), "validation_42": one(("validation_42",))}


def main(argv: Sequence[str] | None = None) -> int:
    from ..eval.teacher_audit import load_atlas

    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--exp0110", required=True)
    ap.add_argument("--ref-models", required=True)
    ap.add_argument("--candidate", required=True, help="EXP-0114 candidate_it40.pt")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dry-run", action="store_true", help="1 world per set, 1 episode, horizon 6")
    a = ap.parse_args(argv)
    im.SOURCE_COMMIT = a.source_commit
    if not a.dry_run and len(a.source_commit) != 40:
        im._refuse(["a real run needs a 40-hex --source-commit"])
    e10 = Path(a.exp0110)
    start = str(e10 / "start_models.pt")
    cal = json.loads((e10 / "calibration.json").read_text())
    horizon, episodes, mw = (6, 1, 1) if a.dry_run else (100, EPISODES, None)
    c12 = cal["per_size"]["12"]
    weights = {"lam": c12["lambda"], "mu": c12["mu"], "nu": float(horizon), "x": cal["x"], "y": cal["y"]}
    ts, _train, mon = im._training(a.training_dir)
    atlas = load_atlas(a.atlas)
    arms = {"ref": (a.ref_models, "own"), "F40": (a.candidate, "start_rep")}
    jobs, cells = build_jobs(ts, mon, atlas, arms, start, weights, horizon, episodes, mw)
    jobs.sort(key=lambda j: j["size"] != 42)  # the 42-producer episodes first (longest)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = im._pool(gp.probe_job, jobs, a.workers)
    with (out / "rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    from ..eval.provenance import sha256_file

    rd = reading(rows, cells)
    doc = {"reading": rd, "dry_run": a.dry_run, "source_commit": a.source_commit, "jobs": len(jobs),
           "candidate_sha256": sha256_file(a.candidate), "ref_sha256": sha256_file(a.ref_models),
           "start_sha256": sha256_file(start), "stage": STAGE, "episodes": episodes}
    (out / "reading.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[f40_confirm] deciding {rd['deciding']['reading']} g {rd['deciding']['g']} F {rd['deciding']['F']}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
