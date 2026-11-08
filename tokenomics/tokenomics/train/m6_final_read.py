"""EXP-0131: the final held-out read of M6's result, F40 + v15 turn heads (user decision 2026-11-08: the RL track is
closed). Pre-registration v20.

EXP-0102's rule (`parity` / `better` / `short`, the side-effect clauses) against k5_rf5 + MWWk, 12 producers deciding, 42
reported. The composite plays only through `m6_onpolicy.deployed_policies`, as trained (DIAGNOSTICS 37). F40 + the
default turns is a reported second arm, so the turn heads' effect is visible.

The held-out worlds were played before (EXP-0103, EXP-0104's diagnosis of 42-0125), at the episodes of stage
`m5-heldout`. This read re-uses the worlds with episodes of a new stage, which no M6 run has played; it is a second
comparison on this set, never a fresh one-shot read of unseen worlds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from . import m6_gnn as g
from . import m6_imitation as im
from . import m6_onpolicy as op
from . import m6_power as pw
from . import m6_turn_imitation as ti
from . import producer_es as pe

PREREG_V20_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v20.json"
PREREG_V20_SHA256: str | None = "8fe01a0c4c958a33df6094c24193bbed168def6de04ce97a932ed3f3a80363ce"  # EXP-0131, frozen before data
LEDGER_KEY = "held-out/m6-f40-v15"
FINAL_STAGE = "m6-final-read"
ABANDONED_STAGE = "m6-final-heldout"  # a local smoke played 8 cycles of its first episode before the freeze
TIMING_STAGE = "m6-final-timing"
TURN_MODELS_SHA256 = "c62a3ce88befa54754cfe7407aebd0c2b3b12209ec7208d33fa9cf2674cbb2d8"  # EXP-0126 models.pt
REPLENISH_MODELS_SHA256 = "697d94c08321fffe08cb6a44bfeaba14988da21821cbed502ec78d54fb528d2c"  # EXP-0110 start_models.pt
DECIDING = "f40_v15"
SECOND = "f40_def"
BASE = "k5_rf5"
ARMS = (DECIDING, SECOND, BASE)
RULE = ("EXP-0102's rule on the loaded held-out worlds: parity if g >= -floor, no extra deaths, and run-down and "
        "starvation shares <= the base's; better if parity and g > floor; else short. Deciding: F40 + v15 heads vs "
        "k5_rf5 + MWWk at 12 producers. Reported: 42 producers and the F40 + default turns arm, on the same rule")


def final_views(atlas: Mapping) -> list[dict]:
    return im.heldout_worlds(atlas)


def seed_problems(views: Sequence[Mapping], episodes: int) -> list[str]:
    """The read's episodes must not be the ones EXP-0103 / EXP-0104 played (stage `m5-heldout`)."""
    from .producer_read import HELDOUT_STAGE, entry_for

    bad = [v["world"] for v in views for other in (HELDOUT_STAGE, ABANDONED_STAGE)
           if set(entry_for(v, episodes, FINAL_STAGE).episode_seeds) & set(entry_for(v, episodes, other).episode_seeds)]
    return [f"fresh episodes collide with an earlier stage on {bad}"] if bad else []


def registered_prereg_v20(atlas: Mapping) -> dict:
    from .producer_read import HELDOUT_STAGE, READ_EPISODES, READ_REPLICATES, RHO_LOADED

    views = final_views(atlas)
    loaded = [v["world"] for v in views if v["rho"] > RHO_LOADED and v["class"] != "collapse"]
    return {
        "format": "m6-prereg-v20", "change": "add-m6-joint-mpg-gnn", "stage": "EXP-0131: M6's final held-out read",
        "decision": "user 2026-11-08 (via the supervisor): close the RL track; M6's result is F40 + v15 heads",
        "subject": {"composite": "m6_onpolicy.load_start: F40's trunk, gates and v15 turn heads (EXP-0126 models.pt), "
                                 "the EXP-0110 start's replenish head, the frozen EXP-0100 orderer",
                    "turn_models_sha256": TURN_MODELS_SHA256, "replenish_models_sha256": REPLENISH_MODELS_SHA256,
                    "deployment": "m6_onpolicy.deployed_policies only (DIAGNOSTICS 37), in the producer-turn env"},
        "arms": {DECIDING: "F40 + v15 heads, deployed_policies (deciding)",
                 SECOND: "F40 + default turns: deployed_policies' producing and ordering policies, the turn env's "
                         "default fund and accept policies (reported)",
                 BASE: "k5_rf5 + MWWk (the base; EXP-0103's path)"},
        "held_out_reuse": {
            "worlds": "the atlas v2 held-out worlds are re-used: every M6 play of them is listed here, and this is a "
                      "second comparison on this set, never a fresh one-shot read of unseen worlds",
            "played_before": {"EXP-0103": f"all 16 held-out worlds, 8 episodes of stage {HELDOUT_STAGE}, replicates "
                                          "0 / 1, horizon 100 (the imitated EXP-0100 stack and the base)",
                              "EXP-0104": f"42-0125 only, stage {HELDOUT_STAGE}'s episodes, horizon 200, plus one-"
                                          "episode local checks on its first episode (the same stage)"},
            "episodes_this_read": f"8 episodes per world of the new stage {FINAL_STAGE}, disjoint from {HELDOUT_STAGE} "
                                  "(the driver refuses a collision)",
            "checked_2026_11_08": "no EXP-0100 to EXP-0130 record names a held-out world except EXP-0103 and EXP-0104, "
                                  "and none of this read's 128 episode seeds appears in the text records under "
                                  "experiments/",
            "abandoned_stage": f"a local dry run of this driver played 8 cycles of the first episode of stage "
                               f"{ABANDONED_STAGE} on ws-atlas-v1-12-0016 (replicates 0 / 1) before this file was "
                               f"frozen. That stage is abandoned, this read uses {FINAL_STAGE}, and dry runs now play "
                               "train-split worlds only",
            "design_leakage": "EXP-0104's diagnosis of held-out 42-0125 (the upgrade trap, DIAGNOSTICS 32) informed the "
                              "producer's later versions, so the 42-producer read is not independent of this set; "
                              "EXP-0101 found no drain at 12, and 42 is reported only"},
        "worlds": {"split": "atlas v2 held-out", "played": "every held-out world at 12 and 42",
                   "gating": f"the loaded ones (rho > {RHO_LOADED}, not collapse), as L5 / L6",
                   "atlas_content_hash": im.REGISTERED_ATLAS_HASH,
                   "ids": {str(sz): [v["world"] for v in views if v["size"] == sz] for sz in (12, 42)},
                   "loaded_ids": {str(sz): [w for w in loaded if f"-{sz}-" in w] for sz in (12, 42)},
                   "episodes": READ_EPISODES, "stage": FINAL_STAGE, "replicates": list(READ_REPLICATES),
                   "horizon": pe.HORIZON},
        "rule": RULE,
        "floor": "L5 / L6's (producer_read._comparisons: order_es_validate.mean_comp, equal-weight world mean over the "
                 "loaded worlds, max(atlas floor, own 3 SE), both arms with 2 replicates)",
        "clauses": "deaths, run-down share (r_end < X r_start), starvation share (any consumer fill < Y), X / Y = "
                   "side-effect params v2",
        "validity": "paired demand and shock digests across the arms on every world-episode-replicate, a contract "
                    "monitor on every episode; any missing row or digest difference makes the run invalid (exit 2), "
                    "never a reading from partial rows",
        "reported": ["the 42-producer reading", "the F40 + default turns arm", "holding, investment rejection, min "
                     "capacity ratio per arm", "the v15 heads minus default turns difference"],
        "operations": "one job per arm and size timed first on train-split worlds (stage m6-final-timing, no held-out "
                      "world touched); stop = 1.5 x that estimate from job start; abort if the job has not started 5 "
                      "min after pod creation, at most one retry; each job's row written as it finishes; a watchdog "
                      "pulls the run directory before any stop; <= 8 vCPU-h on M6-A",
        "gates": "reports only: the roster, registry and champion files are not changed by this run"}


def prereg_problems(atlas: Mapping) -> list[str]:
    from ..eval.provenance import sha256_file

    if PREREG_V20_SHA256 is None:
        return ["no pinned m6_prereg_v20.json (`--mode freeze`)"]
    if not PREREG_V20_PATH.exists() or sha256_file(str(PREREG_V20_PATH)) != PREREG_V20_SHA256:
        return ["m6_prereg_v20.json does not match its pinned SHA-256"]
    if json.loads(PREREG_V20_PATH.read_text()) != registered_prereg_v20(atlas):
        return ["the v20 pre-registration differs from the code / atlas"]
    return []


def freeze(atlas: Mapping) -> tuple[str, str]:
    from ..eval.provenance import sha256_file

    doc = registered_prereg_v20(atlas)
    if not PREREG_V20_PATH.exists():
        PREREG_V20_PATH.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return "FROZEN", sha256_file(str(PREREG_V20_PATH))
    return ("MATCH" if json.loads(PREREG_V20_PATH.read_text()) == doc else "MISMATCH",
            sha256_file(str(PREREG_V20_PATH)))


def ledger_problems(path: Path) -> list[str]:
    """One shot: exactly one pending entry for this subject and pre-registration, and no closed one."""
    if not path.exists():
        return [f"no held-out ledger {path} (commit a pending entry before the launch)"]
    es = [e for e in json.loads(path.read_text()).get("entries", []) if e.get("key") == LEDGER_KEY]
    pend = [e for e in es if e.get("status") == "pending" and e.get("prereg") == PREREG_V20_SHA256
            and e.get("turn_models_sha256") == TURN_MODELS_SHA256
            and e.get("replenish_models_sha256") == REPLENISH_MODELS_SHA256]
    if len(pend) != 1 or any(e.get("status") != "pending" for e in es):
        return [f"the held-out ledger does not hold exactly one pending entry (and none closed) for {LEDGER_KEY}"]
    return []


def final_job(job: dict) -> dict[str, str]:
    """One keyed episode of `job["arm"]`: the gate row, trace cells and the job's wall-clock seconds."""
    from ..eval.contract import ContractMonitor
    from ..eval.gate import row_from_stats, row_to_dict
    from ..eval.routing_audit import trace_cells
    from ..eval.routing_trace import RoutingTracer
    from ..eval.runner import run_episode

    t0 = time.perf_counter()
    if job["arm"] == BASE:
        row = im.side_effect_job({**job, "kind": "monitor_base"})
        row["arm"] = BASE
    else:
        torch.set_num_threads(1)
        env = op.turn_env(job)
        tracer = RoutingTracer()
        tracer.attach(env)
        monitor = ContractMonitor(env)
        model, om = op.load_start(job["turn_models"], job["replenish_models"])
        pol = op.deployed_policies(model, om)
        turns = ({"fund_policy": pol["fund"], "accept_policy": pol["accept"]} if job["arm"] == DECIDING else {})
        run_episode(env, pol["producing"], pol["ordering"], seed=job["episode"], action_seed=None, monitor=monitor,
                    **turns)
        st = env.episode_stats
        monitor.verify(dead=bool(st["dead"]), cycles=st["cycles"], horizon=job["horizon"])
        row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                                         f"{job['arm']}@h{job['horizon']}", job["replicate"], st))
        row.update(trace_cells(tracer))
        row.update({"arm": job["arm"], "horizon": str(job["horizon"]),
                    "units": repr(float(st["consumer_completed_units"])),
                    "fund_shares": json.dumps(tracer.fund_shares(), sort_keys=True),
                    "beyond_k": str(env.turn_beyond_k)})
    row["seconds"] = repr(time.perf_counter() - t0)
    return row


def digest_problems(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Every arm must have played the same demand and shock schedule on a world-episode-replicate."""
    seen: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    for r in rows:
        seen.setdefault((r["world"], r["episode"], r["replicate"]), set()).add((r["demand_digest"], r["shock_digest"]))
    bad = [k for k, v in seen.items() if len(v) > 1]
    return [f"digests differ across the arms on {len(bad)} world-episode-replicates, e.g. {bad[:3]}"] if bad else []


def final_reading(rows: Sequence[Mapping[str, str]], cells: Mapping) -> dict:
    """RULE per size and arm (12 / f40_v15 deciding), plus the heads' effect over the default turns."""
    from .producer_read import _comparisons, _guard_ok

    per = _comparisons(rows, cells, (DECIDING, SECOND))
    out: dict[str, Any] = {"rule": RULE, "per_size": per, "readings": {}, "turn_heads_effect": {}}
    for size, d in per.items():
        for arm in (DECIDING, SECOND):
            c, f = d["comparisons"].get(arm), d["flags"]
            if c is None:
                r = None
            else:
                ok = c["diff"] >= -c["floor"] and _guard_ok(f[arm], f[BASE])
                r = "better" if ok and c["diff"] > c["floor"] else "parity" if ok else "short"
            out["readings"][f"{size}/{arm}"] = r
        a, b = d["comparisons"].get(DECIDING), d["comparisons"].get(SECOND)
        out["turn_heads_effect"][str(size)] = None if a is None or b is None else a["diff"] - b["diff"]
    out["reading"] = out["readings"].get(f"12/{DECIDING}")
    return out


def _build_jobs(views: Sequence[Mapping], episodes: int, stage: str, horizon: int, turn_models: str,
                replenish_models: str) -> list[dict]:
    from .producer_read import READ_REPLICATES, entry_for

    jobs = []
    for v in views:
        e = entry_for(v, episodes, stage)
        for ep in e.episode_seeds:
            for r in READ_REPLICATES:
                for arm in ARMS:
                    jobs.append({"arm": arm, "world": e.id, "size": e.size, "regime": e.regime,
                                 "world_seed": e.world_seed, "generator": e.generator, "episode": int(ep),
                                 "replicate": r, "horizon": horizon, "models": turn_models, "generation": "0",
                                 "turn_models": turn_models, "replenish_models": replenish_models})
    jobs.sort(key=lambda j: (-j["size"], j["arm"] == BASE))  # longest first
    return jobs


def timing_views(atlas: Mapping) -> list[dict]:
    """The most loaded non-collapse train-split world at 12 and at 42: no held-out world is touched."""
    from ..eval.teacher_audit import world_view

    views = []
    for size in (12, 42):
        pool = []
        for entry in atlas["worlds"]:
            v = world_view(entry)
            v["rho"] = float(entry["record"]["coordinates"]["rho"])
            if v["size"] == size and v["split"] == "train" and v["class"] != "collapse":
                pool.append(v)
        views.append(max(pool, key=lambda v: (v["rho"], v["world"])))
    return views


def run_time(atlas: Mapping, turn_models: str, replenish_models: str, out: Path, horizon: int) -> int:
    """One job per arm and size on train-split worlds; the serial estimate of the read's cost."""
    from .producer_read import READ_EPISODES, READ_REPLICATES

    views = timing_views(atlas)
    seconds: dict[str, float] = {}
    for j in _build_jobs(views, 1, TIMING_STAGE, horizon, turn_models, replenish_models):
        if j["replicate"] != READ_REPLICATES[0]:
            continue
        t0 = time.perf_counter()
        final_job(j)
        seconds[f"{j['size']}/{j['arm']}"] = time.perf_counter() - t0
        print(f"[final_read] timed {j['size']}/{j['arm']} {seconds[f'{j['size']}/{j['arm']}']:.1f} s", flush=True)
    counts = {str(sz): sum(1 for v in final_views(atlas) if v["size"] == sz) for sz in (12, 42)}
    per_arm = {sz: n * READ_EPISODES * len(READ_REPLICATES) for sz, n in counts.items()}
    per = {arm: sum(seconds[f"{sz}/{arm}"] * n for sz, n in per_arm.items()) for arm in ARMS}
    doc = {"time_only": True, "worlds": [v["world"] for v in views], "seconds": seconds, "jobs_per_arm": per_arm,
           "core_seconds_per_arm": per, "core_seconds_total": sum(per.values())}
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[final_read] timing {json.dumps(doc)}", flush=True)
    return 0


def run(atlas_path: str, turn_models: str, replenish_models: str, out_dir: str, workers: int, *, ledger: str,
        dry_run: bool = False, sizes: Sequence[int] = (), max_worlds: int | None = None, episodes: int | None = None,
        horizon: int = pe.HORIZON, time_only: bool = False) -> int:
    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas
    from .producer_read import READ_EPISODES, _cells

    atlas = load_atlas(atlas_path)
    eps = episodes or READ_EPISODES
    views = final_views(atlas)
    p = seed_problems(views, READ_EPISODES)
    if sha256_file(turn_models) != TURN_MODELS_SHA256:
        p.append("the turn models are not EXP-0126's")
    if sha256_file(replenish_models) != REPLENISH_MODELS_SHA256:
        p.append("the replenish models are not the EXP-0110 start's")
    if not (dry_run or time_only):
        p += prereg_problems(atlas) + ledger_problems(Path(ledger))
        if atlas.get("content_hash") != im.REGISTERED_ATLAS_HASH:
            p.append("not the registered atlas v2")
        if sizes or max_worlds or episodes or horizon != pe.HORIZON:
            p.append("a filter is set (--sizes / --max-worlds / --episodes / --horizon)")
        if len(im.SOURCE_COMMIT) != 40:
            p.append("a real run needs a 40-hex --source-commit")
    im._refuse(p)
    out = Path(out_dir)
    if time_only:
        return run_time(atlas, turn_models, replenish_models, out, horizon)
    out.mkdir(parents=True, exist_ok=True)
    stage = FINAL_STAGE
    if dry_run:  # a smoke never plays a held-out world
        views, stage = timing_views(atlas), TIMING_STAGE
    if sizes:
        views = [v for v in views if v["size"] in set(sizes)]
    if max_worlds:
        views = [v for sz in (12, 42) for v in [x for x in views if x["size"] == sz][:max_worlds]]
    cells = _cells(views, atlas)
    jobs = _build_jobs(views, eps, stage, horizon, turn_models, replenish_models)
    rows = pw._pool(final_job, jobs, workers, out / "episode_rows.jsonl")
    problems = digest_problems(rows) + ([f"{len(rows)} rows for {len(jobs)} jobs"] if len(rows) != len(jobs) else [])
    doc = {**final_reading(rows, cells), "invalid": problems, "dry_run": dry_run, "split": "train (dry run)" if dry_run else "held-out",
           "stage": stage, "prereg": PREREG_V20_SHA256, "source_commit": im.SOURCE_COMMIT,
           "turn_models_sha256": sha256_file(turn_models), "replenish_models_sha256": sha256_file(replenish_models),
           "worlds": [v["world"] for v in views], "episodes": eps, "horizon": horizon, "rows": len(rows),
           "jobs": len(jobs)}
    if problems:
        doc["reading"] = "invalid"
    (out / "reading.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[final_read] {doc['reading']} {doc['readings']} heads {doc['turn_heads_effect']} {problems}", flush=True)
    return 2 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("freeze", "time", "run"), required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--turn-models", help="EXP-0126 models.pt")
    ap.add_argument("--replenish-models", help="EXP-0110 start_models.pt")
    ap.add_argument("--ledger", default=str(im.HELDOUT_LEDGER), help="the held-out ledger (a staged copy on the pod)")
    ap.add_argument("--out-dir")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sizes", default="", help="dry run: e.g. 12,42")
    ap.add_argument("--max-worlds", type=int, help="dry run: per size")
    ap.add_argument("--episodes", type=int, help="dry run: per world")
    ap.add_argument("--horizon", type=int, default=pe.HORIZON, help="dry run or timing smoke")
    a = ap.parse_args(argv)
    if a.mode == "freeze":
        from ..eval.teacher_audit import load_atlas

        print(*freeze(load_atlas(a.atlas)))
        return 0
    im.SOURCE_COMMIT = a.source_commit
    return run(a.atlas, a.turn_models, a.replenish_models, a.out_dir, a.workers, ledger=a.ledger, dry_run=a.dry_run,
               sizes=tuple(int(x) for x in a.sizes.split(",") if x), max_worlds=a.max_worlds, episodes=a.episodes,
               horizon=a.horizon, time_only=a.mode == "time")


if __name__ == "__main__":
    sys.exit(main())
