"""M5 reads of the producer residual: `python -m tokenomics.train.producer_read`
(add-m5-potential-game-gnn task 2.4; design D6; pre-registration `es_prereg_m5_v2.json`).

Modes:
- `pilot` (L3): g_tr and F_tr of a train run at a generation (default 20) from its monitor rows
  (the centre and rf3 at replicates 0 and 1 on the 16 monitor world-episodes). g_tr = the
  equal-weight mean over monitor world-episodes of (centre - rf3); F_tr = M4's floors rule:
  max(the atlas-spread floor at these counts, own 3 SE), own SE only with >= 2 paired observations
  on a world-episode (the n = 1 rule; order_es_validate `mean_comp` / `monitor_cells`).
  `feasible` if g_tr > F_tr, else `unresolved`.
- `validate` (L5): both seeds' candidates (their latest checkpoint and the bar's normaliser), rf3,
  `entity_planner_buyahead_k4` and `entity_planner`, all with MWWk at DrawKey(world, episode,
  replicate), replicates 0 and 1, on the loaded (rho > 0.28, not collapse) atlas v2 validation worlds
  at 12 and 42, READ_EPISODES episodes per world from M4's validation stage, observer on.
  - Per size and seed: the candidate - rf3 world mean with its floor (the same rule).
  - kill: either seed below -F_val at 12 or with extra deaths; transfer_failure: success at 12
    and either seed below -F_val at 42; success: both seeds above F_val at 12 with no extra
    deaths and run-down / starvation shares <= rf3's; plateau: anything else (precedence kill,
    transfer_failure, success, plateau).
  - The selection (v2): the seed with the larger 12-producer candidate - rf3, seed 0 on a tie.
  - `stock_heavy`: the selected candidate's mean held stock > 1.5 x rf3's on the read's rows
    (reported only).
  - The reference comparisons (k4, entity_planner vs rf3) and per-arm loss classes are reported.
- `heldout` (L6): the selected candidate, rf3 and the references on the held-out worlds (8 episodes
  from stage `m5-heldout`, replicates 0 and 1), gating on the loaded ones as L5. `confirmed` if the
  candidate - rf3 clears its floor at 12, is not below -floor at 42, with no extra deaths and
  run-down / starvation shares <= rf3's; else `not_confirmed`. A real run needs the ledger's single
  pending entry for `held-out/m5-producer` with this theta and this package tree's hash; the read
  never writes the ledger.
- `recompute`: rebuilds a read dir's `reading.json` from its `episodes.csv` (MATCH / MISMATCH).
- `ledger-pending` / `ledger-close`: the held-out ledger's two local steps (M4 design D9's
  C0 / C1 rule: the pending entry is committed before the launch; done / invalid entries are
  final).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..eval.gate_cli import CONTRACT_INVALID, Job
from ..eval.provenance import resolve_verified_commit, sha256_file, source_fingerprint
from . import producer_es as pe
from .producer_residual import BASE_PRODUCER, K5_RF5_PRODUCER, RF4_PRODUCER, Normaliser

DRIVER = "tokenomics.train.producer_read"
PILOT, VALIDATE, HELDOUT, RECOMPUTE = "pilot", "validate", "heldout", "recompute"
READ_EPISODES = 8  # es_prereg_m5 L5 (carried from M4's power rule, flagged)
READ_REPLICATES = (0, 1)
HELDOUT_STAGE = "m5-heldout"
RHO_LOADED = 0.28
STOCK_HEAVY = 1.5
# the base arm (rf4 from pre-registration v5, user decision 2026-11-05; rf3 before) and the
# reported references
BASE = {K5_RF5_PRODUCER: "k5_rf5", RF4_PRODUCER: "rf4"}.get(BASE_PRODUCER, "rf3")
REFS = {BASE: BASE_PRODUCER, "k4": "entity_planner_buyahead_k4", "ep": "entity_planner"}
SEEDS = ("cand-s0", "cand-s1")
CAND = "cand"
LEDGER_KEY = "held-out/m5-producer"
LEDGER_FORMAT = "m5-heldout-ledger-v1"
LEDGER_PATH = Path(__file__).resolve().parents[3] / "experiments" / "m5_heldout_ledger.json"
LOSS_GROUPS = {"input_late": ("b2_late",), "input_died_upstream": ("b2_lost_ttl",
               "b2_lost_ttl_outage"), "contention": ("b4_contention",),
               "capacity": ("a_held", "a_degraded", "a_max", "a_dropped"),
               "outbid": ("c_displaced",)}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _read_columns() -> tuple[str, ...]:
    from ..eval import supply_trace as st

    return tuple(dict.fromkeys((*st.COLUMNS, *pe.ES_COLUMNS)))


# --- pure readings ---


def _units(dicts: Sequence[Mapping[str, str]], key: Callable[[Mapping[str, str]], str]
           ) -> dict[str, dict[str, dict[int, dict[int, float]]]]:
    out: dict = {}
    for d in dicts:
        out.setdefault(key(d), {}).setdefault(d["arm"], {}).setdefault(
            int(d["episode"]), {})[int(d["replicate"])] = float(d["units"])
    return out


def pilot_reading(monitor_rows: Sequence[Mapping[str, str]], cells: Mapping[str, Any],
                  generation: int) -> dict[str, Any]:
    """L3 from a run's monitor rows (module docstring)."""
    from .order_es_validate import mean_comp, world_mean

    rows = []
    for d in monitor_rows:
        if d["kind"] == "monitor" and d["generation"] == str(generation):
            rows.append({**d, "arm": "center"})
        elif d["kind"] == "monitor_base":
            rows.append({**d, "arm": BASE})
    units = _units(rows, lambda d: f"{d['world']}#{d['episode']}")
    keys = sorted(k for k, v in units.items() if "center" in v and BASE in v)
    if not keys:
        return {"generation": generation, "reading": None, "reason": "no monitor rows"}
    bars = {k: world_mean(units[k][BASE]) for k in keys}
    comp = mean_comp(units, cells, "center", BASE, {"center": 2, BASE: 2}, bars, keys)
    return {"generation": generation, "g_tr": comp.diff, "F_tr": comp.floor,
            "comparison": comp.to_dict(), "world_episodes": len(keys),
            "reading": "feasible" if comp.clears else "unresolved"}


def _flags(dicts: Sequence[Mapping[str, str]], arm: str, worlds: set[str]) -> dict[str, Any]:
    from .order_es_validate import row_flags

    rs = [d for d in dicts if d["arm"] == arm and d["world"] in worlds]
    f = [row_flags(d) for d in rs]
    n = len(f)
    return {"rows": n, "deaths": sum(x[0] for x in f),
            "rundown_share": sum(x[1] for x in f) / n if n else None,
            "starvation_share": sum(x[2] for x in f) / n if n else None,
            "holding_units_per_cycle": statistics.fmean(float(d["holding_units_per_cycle"])
                                                        for d in rs) if n else None,
            "loss_classes": {g: sum(float(d.get(f"st_{c}") or 0) for d in rs for c in cs)
                             for g, cs in LOSS_GROUPS.items()}}


def _guard_ok(flags: Mapping[str, Any], base: Mapping[str, Any]) -> bool:
    return (flags["deaths"] <= base["deaths"]
            and (flags["rundown_share"] or 0) <= (base["rundown_share"] or 0)
            and (flags["starvation_share"] or 0) <= (base["starvation_share"] or 0))


def _comparisons(dicts: Sequence[Mapping[str, str]], cells: Mapping[str, Any],
                 arms: Sequence[str]) -> dict[int, dict[str, Any]]:
    """Per size: each arm - rf3 (world mean, floor) on the loaded gating worlds, and flags."""
    from .order_es_validate import mean_comp, world_mean

    units = _units(dicts, lambda d: d["world"])
    out: dict[int, dict[str, Any]] = {}
    for size in (12, 42):
        worlds = sorted(w for w, c in cells.items() if c.size == size and loaded(c)
                        and w in units)
        if not worlds:
            continue
        bars = {w: world_mean(units[w][BASE]) for w in worlds if BASE in units[w]}
        reps = {a: len(READ_REPLICATES) for a in (*arms, BASE)}
        comps = {}
        for a in arms:
            m = None if len(bars) != len(worlds) else mean_comp(units, cells, a, BASE, reps, bars,
                                                                worlds)
            comps[a] = None if m is None else m.to_dict() | {"clears": m.clears,
                                                             "below": m.diff < -m.floor}
        ws = set(worlds)
        out[size] = {"worlds": worlds, "comparisons": comps,
                     "flags": {a: _flags(dicts, a, ws) for a in (*arms, BASE)}}
    return out


def loaded(cell: Any) -> bool:
    from ..eval.atlas import COLLAPSE

    return cell.cls != COLLAPSE and cell.rho > RHO_LOADED


def validate_reading(dicts: Sequence[Mapping[str, str]], cells: Mapping[str, Any]
                     ) -> dict[str, Any]:
    arms = (*SEEDS, "k4", "ep")
    per = _comparisons(dicts, cells, arms)
    if 12 not in per or any(per[12]["comparisons"].get(s) is None for s in SEEDS):
        return {"reading": None, "reason": "incomplete: a 12-producer comparison lacks rows",
                "per_size": per}
    c12, f12 = per[12]["comparisons"], per[12]["flags"]
    kill = any(c12[s]["below"] or f12[s]["deaths"] > f12[BASE]["deaths"] for s in SEEDS)
    success = all(c12[s]["clears"] and _guard_ok(f12[s], f12[BASE]) for s in SEEDS)
    transfer = success and 42 in per and any(
        (per[42]["comparisons"].get(s) or {}).get("below") for s in SEEDS)
    reading = ("kill" if kill else "transfer_failure" if transfer else "success" if success
               else "plateau")
    selected = SEEDS[1] if c12[SEEDS[1]]["diff"] > c12[SEEDS[0]]["diff"] else SEEDS[0]
    sel_rows = [d for d in dicts if d["arm"] == selected]
    base_rows = [d for d in dicts if d["arm"] == BASE]
    hold = statistics.fmean(float(d["holding_units_per_cycle"]) for d in sel_rows)
    hold_b = statistics.fmean(float(d["holding_units_per_cycle"]) for d in base_rows)
    return {"reading": reading, "selected": selected,
            "selection_rule": "larger 12-producer candidate - rf3 world mean; seed 0 on a tie",
            "stock_heavy": {"flag": hold > STOCK_HEAVY * hold_b, "selected_holding": hold,
                            "base_holding": hold_b, "ratio": hold / hold_b if hold_b else None,
                            "role": "reported only, never deciding"},
            "per_size": per}


def heldout_reading(dicts: Sequence[Mapping[str, str]], cells: Mapping[str, Any]
                    ) -> dict[str, Any]:
    """L6 as registered in v2 (`readings.L6.confirmed`): the candidate - rf3 clears its floor at
    12; not below -floor at 42; at 12 no extra deaths and run-down / starvation shares <= rf3's
    (L5's clause); at 42 no extra deaths."""
    per = _comparisons(dicts, cells, (CAND, "k4", "ep"))
    if 12 not in per or per[12]["comparisons"].get(CAND) is None:
        return {"reading": None, "reason": "incomplete", "per_size": per}
    c12, f12 = per[12]["comparisons"][CAND], per[12]["flags"]
    below42 = 42 in per and (per[42]["comparisons"].get(CAND) or {}).get("below")
    guard42 = 42 not in per or per[42]["flags"][CAND]["deaths"] <= per[42]["flags"][BASE]["deaths"]
    ok = c12["clears"] and not below42 and _guard_ok(f12[CAND], f12[BASE]) and guard42
    return {"reading": "confirmed" if ok else "not_confirmed", "per_size": per}


# --- views, jobs ---


def read_views(atlas: Mapping, split: str, worlds: Sequence[str] | None,
               max_worlds: int | None) -> list[dict]:
    """The split's worlds at 12 and 42 (L5: validation, L6: held-out); the gating set is the
    loaded ones, the others played (L6) or skipped (L5: loaded only)."""
    from ..eval.atlas import HELDOUT as ATLAS_HELDOUT
    from ..eval.atlas import VALIDATION
    from ..eval.teacher_audit import world_view

    want = {VALIDATE: VALIDATION, HELDOUT: ATLAS_HELDOUT}[split]
    views = []
    for entry in atlas["worlds"]:
        v = world_view(entry)
        v["rho"] = float(entry["record"]["coordinates"]["rho"])
        if v["split"] == want and v["size"] in (12, 42):
            views.append(v)
    if worlds:
        views = [v for v in views if v["world"] in set(worlds)]
        if len(views) != len(set(worlds)):
            raise ValueError(f"worlds outside the {want} split are refused")
    elif split == VALIDATE:
        views = [v for v in views if v["rho"] > RHO_LOADED and v["class"] != "collapse"]
    views.sort(key=lambda v: (v["size"], v["world"]))
    return views[:max_worlds] if max_worlds is not None else views


def entry_for(view: Mapping, episodes: int, stage: str) -> Any:
    from ..eval.atlas import space_entry, space_info
    from ..eval.world_space import config_from_record, episode_seeds, point_from_record

    rec = view["record"]
    config_from_record(rec)  # the recorded config hash must rebuild
    return space_entry(point_from_record(rec), episode_seeds(space_info(rec).id, episodes, stage),
                       split=view["split"])


def read_jobs(entries: Sequence[Any], arms: Mapping[str, dict], horizon: int, kind: str
              ) -> list[Job]:
    jobs = []
    for e in entries:
        for ep in e.episode_seeds:
            for r in READ_REPLICATES:
                for arm, spec in arms.items():
                    jobs.append({
                        "kind": kind, "arm": arm, "producing": spec["producing"],
                        "theta": spec.get("theta"), "theta_sha256": spec.get("theta_sha256", ""),
                        "normaliser": spec.get("normaliser"),
                        "normaliser_sha256": spec.get("normaliser_sha256"),
                        **({"residual_base": spec["residual_base"]}
                           if "residual_base" in spec else {}),
                        "generation": "", "member": arm, "sign": 0, "pair": -1, "batch": -1,
                        "world": e.id, "size": e.size, "regime": e.regime,
                        "world_seed": e.world_seed, "generator": e.generator, "episode": int(ep),
                        "replicate": r, "horizon": horizon, "ordering": pe.ORDERING,
                        "subject": f"{kind}/{arm}", "cost": float(e.size) ** 2, "observe": True,
                        "job_id": f"{kind}|{arm}|{e.id}|{ep}|{r}"})
    return jobs


def load_candidate(run_dir: str | Path, bar_dir: str | Path, *, real: bool) -> dict:
    run = Path(run_dir)
    spec = json.loads((run / "spec.json").read_text())
    if spec.get("driver") != pe.DRIVER or spec.get("mode") != pe.TRAIN:
        raise ValueError(f"{run} is not a producer_es train run")
    done = sorted((run / "checkpoints").glob("gen-*.json"))
    if not done:
        raise ValueError(f"{run} holds no checkpoints")
    ck = pe.load_checkpoint(done[-1])
    nm, n_pins = pe.load_normaliser(bar_dir)
    problems = []
    if spec["pins"].get("normaliser_sha256") != nm.sha256:
        problems.append(f"{run} was trained under another normaliser")
    if real:
        if spec.get("status") != "complete" or spec["pins"].get("dry_run"):
            problems.append(f"{run} is not a complete real run")
        if ck["generation"] != pe.GENERATIONS["main"]:
            problems.append(f"{run} stops at generation {ck['generation']}, not "
                            f"{pe.GENERATIONS['main']}")
        if spec["pins"].get("prereg") != pe.PREREG_SHA256:
            problems.append(f"{run} was trained under pre-registration "
                            f"{str(spec['pins'].get('prereg'))[:12]}")
    if problems:
        raise ValueError("; ".join(problems))
    return {"producing": pe.RESIDUAL, "theta": [float(x) for x in ck["theta"]],
            "theta_sha256": ck["theta_sha256"], "generation": ck["generation"],
            "opt_seed": spec["pins"].get("opt_seed"), "normaliser": nm.to_dict(),
            "normaliser_sha256": nm.sha256, "run": str(run), "residual_base": BASE_PRODUCER}


def _digest_problems(dicts: Sequence[Mapping[str, str]]) -> list[str]:
    seen: dict = {}
    bad = []
    for d in dicts:
        k = (d["world"], d["episode"], d["replicate"])
        v = (d["demand_digest"], d["shock_digest"])
        if seen.setdefault(k, v) != v:
            bad.append(f"{k}: digests differ across arms")
    return bad


# --- ledger (M4 design D9's C0 / C1 rule) ---


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def read_ledger(path: Path) -> dict:
    if not path.exists():
        return {"format": LEDGER_FORMAT, "entries": []}
    doc = json.loads(path.read_text())
    if doc.get("format") != LEDGER_FORMAT:
        raise ValueError(f"{path} is not an M5 held-out ledger")
    return doc


def _entries(doc: Mapping, key: str = LEDGER_KEY) -> list[dict]:
    return [e for e in doc["entries"] if e["key"] == key]


def ledger_pending(path: Path, theta_sha256: str, atlas_hash: str,
                   head: Callable[[], str] = _git_head) -> dict:
    """Records C0 = HEAD, the package tree hash at C0, theta and the date. Refused when any entry
    exists for the key (one held-out read per key)."""
    doc = read_ledger(path)
    if _entries(doc):
        raise ValueError(f"{path} already holds an entry for {LEDGER_KEY}; refusing")
    entry = {"key": LEDGER_KEY, "status": "pending", "theta_sha256": theta_sha256,
             "atlas_hash": atlas_hash, "change": "add-m5-potential-game-gnn", "split": "held-out",
             "c0": head(), "package_source_sha256": source_fingerprint()["tree_sha256"],
             "date": time.strftime("%Y-%m-%d")}
    doc["entries"].append(entry)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return entry


def ledger_close(path: Path, status: str, exp: str, head: Callable[[], str] = _git_head) -> dict:
    if status not in ("done", "invalid"):
        raise ValueError("a ledger entry closes as done or invalid")
    doc = read_ledger(path)
    es_ = _entries(doc)
    if len(es_) != 1 or es_[0]["status"] != "pending":
        raise ValueError(f"no single pending entry for {LEDGER_KEY} (a done / invalid entry is "
                         "final)")
    es_[0].update({"status": status, "c1": head(), "exp": exp,
                   "closed": time.strftime("%Y-%m-%d")})
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return es_[0]


def ledger_problems(path: Path | None, theta_sha256: str) -> list[str]:
    """A real held-out read needs exactly one pending entry with this theta and this package
    tree (the synced tree at C1 equals C0's package: C1 changes only the ledger)."""
    if path is None:
        return ["a real held-out read needs --ledger"]
    doc = read_ledger(path)
    es_ = _entries(doc)
    if len(es_) != 1:
        return [f"the ledger holds {len(es_)} entries for {LEDGER_KEY}, not exactly one"]
    e = es_[0]
    out = []
    if e["status"] != "pending":
        out.append(f"the ledger entry is {e['status']}: a held-out read is never re-run")
    if e["theta_sha256"] != theta_sha256:
        out.append("the ledger entry names another theta")
    if e["package_source_sha256"] != source_fingerprint()["tree_sha256"]:
        out.append("the package tree differs from the ledger entry's (C0)")
    return out


# --- modes ---


def run_pilot(*, run_dir: str | Path, training_dir: str | Path, atlas_path: str | Path,
              generation: int = 20, out_path: str | Path | None = None,
              log: Callable[[str], None] = _log) -> dict:
    from ..eval.teacher_audit import load_atlas
    from .order_es_validate import monitor_cells

    ts = pe.load_training(training_dir)
    atlas = load_atlas(atlas_path)
    cells = monitor_cells(ts.training, atlas["spreads"])
    rows = pe.read_csv(Path(run_dir) / "monitor_episodes.csv")
    doc = {"run": str(run_dir), **pilot_reading(rows, cells, generation)}
    mon = {int(m["generation"]): m for m in pe.read_csv(Path(run_dir) / "monitor.csv")}
    if generation in mon and doc.get("g_tr") is not None:
        doc["monitor_csv_g_tr"] = float(mon[generation]["g_tr"])
    out = Path(out_path) if out_path else Path(run_dir) / f"pilot_g{generation}.json"
    out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    log(f"[producer_read] L3 at generation {generation}: {doc.get('reading')} "
        f"(g_tr {doc.get('g_tr')}, F_tr {doc.get('F_tr')})")
    return doc


def _cells(views: Sequence[Mapping], atlas: Mapping) -> dict:
    from .order_es_validate import world_cells

    return world_cells(views, atlas["spreads"])


def run_read(*, mode: str, out_dir: str | Path, atlas_path: str | Path, bar_dir: str | Path,
             runs: Sequence[str] = (), validation_dir: str | Path | None = None,
             ledger: str | Path | None = None, workers: int = 1, horizon: int = pe.HORIZON,
             episodes: int = READ_EPISODES, worlds: Sequence[str] | None = None,
             max_worlds: int | None = None, source_commit: str = "", dry_run: bool = False,
             run_job_fn: Callable[[Job], dict[str, str]] = pe.run_producer_job,
             log: Callable[[str], None] = _log) -> int:
    from ..eval.teacher_audit import REGISTERED_ATLAS_HASH, load_atlas

    atlas = load_atlas(atlas_path)
    real = not dry_run
    problems: list[str] = []
    if mode == VALIDATE:
        if len(runs) != len(SEEDS):
            raise ValueError("--mode validate takes both seeds' train runs (--runs s0 s1)")
        cands = {s: load_candidate(r, bar_dir, real=real) for s, r in zip(SEEDS, runs)}
        if real and [c["opt_seed"] for c in cands.values()] != [0, 1]:
            problems.append("the runs are not optimiser seeds 0 and 1 in order")
        stage = __import__("tokenomics.train.order_es_validate",
                           fromlist=["VALIDATION_STAGE"]).VALIDATION_STAGE
    else:
        if validation_dir is None:
            raise ValueError("--mode heldout needs --validation (the L5 read's dir)")
        vread = json.loads((Path(validation_dir) / "reading.json").read_text())
        vspec = json.loads((Path(validation_dir) / "spec.json").read_text())
        sel = vread.get("selected")
        if sel not in SEEDS:
            raise ValueError("the validation read selects no candidate")
        run = vspec["pins"]["runs"][SEEDS.index(sel)]
        if real:  # the L5 gate first: a held-out read never starts from a bad L5 read
            gate = []
            if vread.get("reading") != "success" or vspec["pins"].get("dry_run"):
                gate.append("L6 is gated on a real L5 `success`")
            if vspec.get("status") != "complete":  # a reading from partial rows (review m5-2-4)
                gate.append(f"the L5 read is {vspec.get('status')!r}, not complete")
            if vspec["pins"].get("prereg") != pe.PREREG_SHA256:
                gate.append("the L5 read ran under another pre-registration")
            if gate:
                raise ValueError("not a valid heldout read (" + "; ".join(gate) + ")")
        cands = {CAND: load_candidate(run, bar_dir, real=real)}
        if real:
            problems += ledger_problems(None if ledger is None else Path(ledger),
                                        cands[CAND]["theta_sha256"])
        stage = HELDOUT_STAGE
    if real:
        problems += pe.real_run_problems(horizon=horizon, source_commit=source_commit,
                                         filters={"worlds": worlds, "max_worlds": max_worlds,
                                                  "episodes": None if episodes == READ_EPISODES
                                                  else episodes})
        problems += pe.prereg_problems()
        if atlas.get("content_hash") != REGISTERED_ATLAS_HASH:
            problems.append("not the registered atlas")
        if problems:
            raise ValueError(f"not a valid {mode} read (" + "; ".join(problems)
                             + "); pass --dry-run")
    views = read_views(atlas, mode, worlds, max_worlds)
    cells = _cells(views, atlas)
    entries = [entry_for(v, episodes, stage) for v in views]
    arms = {**cands, **{k: {"producing": v} for k, v in REFS.items()}}
    out = Path(out_dir)
    pins = {"driver": DRIVER, "mode": mode, "horizon": horizon, "dry_run": dry_run,
            "episodes": episodes, "replicates": list(READ_REPLICATES), "stage": stage,
            "worlds": [v["world"] for v in views], "atlas_path": str(atlas_path),
            "atlas_sha256": sha256_file(str(atlas_path)),
            "atlas_content_hash": atlas.get("content_hash"), "prereg": pe.PREREG_SHA256,
            "runs": [c.get("run") for c in cands.values()],
            "arms": {a: {k: v for k, v in spec.items() if k != "theta"} for a, spec in arms.items()},
            "validation": None if validation_dir is None else str(validation_dir),
            "source": source_fingerprint(source_commit or None)}
    spec = pe._open_run(out, mode, pins, workers, log)
    jobs = read_jobs(entries, arms, horizon, mode)
    rows, _ = pe.play(out, "episodes.csv", jobs, workers, run_job_fn, None, log, _read_columns())
    dicts = [rows[j["job_id"]] for j in jobs if j["job_id"] in rows]
    bad = _digest_problems(dicts)
    reading = (validate_reading if mode == VALIDATE else heldout_reading)(dicts, cells)
    reading.update({"mode": mode, "dry_run": dry_run, "prereg": pe.PREREG_SHA256})
    (out / "reading.json").write_text(json.dumps(reading, indent=1, sort_keys=True, default=str)
                                      + "\n")
    return pe._finish(out, spec, len(jobs), len(dicts), bad,
                      {"reading": reading.get("reading"), "selected": reading.get("selected")},
                      log)


def run_recompute(*, out_dir: str | Path, log: Callable[[str], None] = _log) -> int:
    """Rebuild a read's reading.json from its episodes.csv and pins (the atlas file is re-read and
    must match its pinned SHA-256): MATCH (0) or MISMATCH (exit 2)."""
    import csv

    from ..eval.teacher_audit import load_atlas

    csv.field_size_limit(1 << 30)
    out = Path(out_dir)
    pins = json.loads((out / "spec.json").read_text())["pins"]
    if sha256_file(pins["atlas_path"]) != pins["atlas_sha256"]:
        raise ValueError("the read's atlas file changed since the read")
    atlas = load_atlas(pins["atlas_path"])
    cells = _cells(read_views(atlas, pins["mode"], pins["worlds"], None), atlas)
    dicts = pe.read_csv(out / "episodes.csv")
    got = (validate_reading if pins["mode"] == VALIDATE else heldout_reading)(dicts, cells)
    got.update({"mode": pins["mode"], "dry_run": pins["dry_run"], "prereg": pins["prereg"]})
    have = json.loads((out / "reading.json").read_text())
    same = json.loads(json.dumps(got, sort_keys=True, default=str)) == have
    log(f"[producer_read] recompute {'MATCH' if same else 'MISMATCH'}")
    return 0 if same else CONTRACT_INVALID


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", required=True,
                   choices=(PILOT, VALIDATE, HELDOUT, RECOMPUTE, "ledger-pending", "ledger-close"))
    p.add_argument("--run", help="pilot: the train run dir")
    p.add_argument("--runs", nargs="*", default=[], help="validate: seed 0 and seed 1 run dirs")
    p.add_argument("--validation", help="heldout: the L5 read's dir")
    p.add_argument("--training-dir", help="pilot: EXP-0058 bar/")
    p.add_argument("--bar-dir", help="validate / heldout: the producer_es bar run")
    p.add_argument("--atlas")
    p.add_argument("--out-dir")
    p.add_argument("--generation", type=int, default=pe.GENERATIONS["pilot"])
    p.add_argument("--ledger", default=None)
    p.add_argument("--theta-hash")
    p.add_argument("--status", choices=("done", "invalid"))
    p.add_argument("--exp")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--horizon", type=int, default=pe.HORIZON)
    p.add_argument("--episodes", type=int, default=READ_EPISODES)
    p.add_argument("--worlds", help="dry run: comma-separated world ids")
    p.add_argument("--max-worlds", type=int)
    p.add_argument("--source-commit", default="")
    p.add_argument("--verify-source-commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.mode == PILOT:
        doc = run_pilot(run_dir=a.run, training_dir=a.training_dir, atlas_path=a.atlas,
                        generation=a.generation)
        return 0 if doc.get("reading") else CONTRACT_INVALID
    if a.mode == RECOMPUTE:
        return run_recompute(out_dir=a.out_dir)
    if a.mode == "ledger-pending":
        from ..eval.teacher_audit import REGISTERED_ATLAS_HASH

        print(ledger_pending(Path(a.ledger or LEDGER_PATH), a.theta_hash, REGISTERED_ATLAS_HASH))
        return 0
    if a.mode == "ledger-close":
        print(ledger_close(Path(a.ledger or LEDGER_PATH), a.status, a.exp))
        return 0
    commit = resolve_verified_commit() if a.verify_source_commit else a.source_commit
    return run_read(mode=a.mode, out_dir=a.out_dir, atlas_path=a.atlas, bar_dir=a.bar_dir,
                    runs=a.runs, validation_dir=a.validation, ledger=a.ledger, workers=a.workers,
                    horizon=a.horizon, episodes=a.episodes,
                    worlds=None if a.worlds is None else a.worlds.split(","),
                    max_worlds=a.max_worlds, source_commit=commit, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
