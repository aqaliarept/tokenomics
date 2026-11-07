"""EXP-0123: does EXP-0120's imperfect turn-head imitation cost units in closed loop? (user decision 2026-11-07;
pre-registration note `notes/exp0123-prereg.md`). Never an acceptance read; the 4.3 offline gate is unchanged.

Arms on the same keyed world-episodes (the 16 monitor worlds × fresh episodes, never played, × r0 / r1):
- `candidate`: F40 (candidate_it40's gates + the EXP-0110 start's replenish) with EXP-0120's turn heads (greedy);
- `reference`: F40 with the default turns (today's auction and funding order).

Reading (v11's metric and floors): `gain` g > F; `cost` g < −F or worse deaths / run-down / starvation; else `no_cost`.
Reported from the candidate's own episodes: at each accept turn the heads' decision against the default decision on
the same state (closed-loop accept agreement), the differing bits split by slack (DIAGNOSTICS 38) and by direction,
and the fate of the pending claims the heads create (ran later / failed at TTL; DIAGNOSTICS 36).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import m6_gnn as g
from . import m6_imitation as im
from . import m6_turn_imitation as ti

STAGE = "m6-turn-closed-loop"
EPISODES = 2
SLACK_BINS = ("<-2", "[-2,-0.5)", "[-0.5,0)", "[0,0.5)", "[0.5,2)", ">=2")


def _bin(v: float) -> str:
    return ("<-2" if v < -2 else "[-2,-0.5)" if v < -0.5 else "[-0.5,0)" if v < 0 else "[0,0.5)" if v < 0.5
            else "[0.5,2)" if v < 2 else ">=2")


def _slack(p: Any, lists: Any) -> dict[int, float]:
    """Per bid index: capacity left after the higher-priced bids (default auction) minus own need, in own
    capacities, at the binding type."""
    budget = p._new_work_budget(lists)
    rem = dict(budget)
    out = {}
    for i, b in enumerate(lists.bids):  # lists.bids are price-descending (the auction's order)
        need = max(b.capacities.values()) if b.capacities else 0
        sl = min(rem.get(t, 0) - c for t, c in b.capacities.items()) if b.capacities else 0.0
        out[i] = sl / max(need, 1)
        if p._fits(rem, b.capacities):
            p._subtract(rem, b.capacities)
    return out


def episode_job(job: dict) -> dict[str, Any]:
    from ..env.producer_turns import decode_accept
    from ..eval.gate import row_from_stats, row_to_dict
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    env = ti._turn_env(job)
    pm, om = im.load_models(job["models"])
    stats: Counter = Counter()
    kw = {}
    if job["arm"] == "candidate":
        cache = g.TurnCache()
        fund_pol, acc_pol = g.gnn_turn_policies(pm, cache)

        def accept(e: Any, agent: str) -> np.ndarray:
            pid = e._role[agent][1]
            p = e.system.producing_agents[pid]
            lists = p.turn_lists()
            act = acc_pol(e, agent)
            if lists.bids:
                mine, _ = decode_accept(act, lists)
                dflt = p.default_turn_decision()
                fits = [p._fits(p._new_work_budget(lists), b.capacities) for b in lists.bids]
                sl = _slack(p, lists)
                for i, (a, d) in enumerate(zip(mine.accept, dflt.accept)):
                    stats["bits"] += 1
                    stats["agree"] += int(a == d)
                    if a != d:
                        stats[f"differ_{'extra_accept' if a else 'extra_decline'}"] += 1
                        if fits[i]:
                            stats[f"differ_slack_{_bin(sl[i])}"] += 1
                        else:
                            stats["differ_no_fit"] += 1
            return act

        kw = {"fund_policy": fund_pol, "accept_policy": accept}
        _track_pending(env, stats)
    st = run_episode(env, ti._producer_policy(job, pm), g.gnn_ordering_policy(om), seed=job["episode"],
                     action_seed=None, **kw)
    row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                                     f"{job['arm']}@h{job['horizon']}", job["replicate"], st))
    row.update({"arm": job["arm"], "set": "monitor_fresh", "units": repr(float(st["consumer_completed_units"])),
                "beyond_k": str(env.turn_beyond_k), "replenish": job.get("replenish") or ""})
    return {"row": row, "stats": dict(stats)}


def _track_pending(env: Any, stats: Counter) -> None:
    """Wrap each producer's scheduled production to follow the pending claims the heads create: created,
    then ran in a later cycle, or failed at TTL (DIAGNOSTICS 36). Diagnostic only (eval-side wrapper)."""
    from ..domain.producing_agent import FAIL_TTL

    seen: dict[tuple[str, str], str] = {}
    original_reset = env.reset

    def reset(*a: Any, **k: Any) -> Any:
        out = original_reset(*a, **k)
        for pid, p in env.system.producing_agents.items():
            orig = p._produce_scheduled

            def wrapped(outage: bool, p: Any = p, pid: str = pid, orig: Any = orig) -> Any:
                res = orig(outage)
                for b in res.completed:
                    if seen.pop((pid, str(b.order_id)), None) == "pending":
                        stats["pending_ran_later"] += 1
                for b, why in zip(res.failed, res.failure_reasons):
                    if seen.pop((pid, str(b.order_id)), None) == "pending" and why == FAIL_TTL:
                        stats["pending_failed_ttl"] += 1
                for r in p.pending:
                    key = (pid, str(r.bid.order_id))
                    if key not in seen:
                        seen[key] = "pending"
                        stats["pending_created"] += 1
                return res

            p._produce_scheduled = wrapped
        return out

    env.reset = reset


def build_jobs(ts: Any, mon: Sequence, cand: str, ref_models: str, replenish: str, horizon: int,
               episodes: int, max_worlds: int | None, stage: str = STAGE) -> tuple[list[dict], dict]:
    from ..eval.world_space import episode_seeds
    from .m6_f40_confirm import _cell

    records = {w["world"]: w["record"] for w in ts.training["worlds"]}
    worlds = sorted({w for w, _ in mon})[:max_worlds]
    wes = [(w, int(ep)) for w in worlds for ep in episode_seeds(w, episodes, stage)]
    assert not set(wes) & set(mon)
    cells = {f"{w}#{ep}": _cell(records[w], f"{w}#{ep}", {}) for w, ep in wes}
    jobs = []
    for j in ti._jobs(ts, wes, "turn_closed_loop", cand, horizon):
        for arm, models in (("candidate", cand), ("reference", ref_models)):
            jobs.append({**j, "arm": arm, "models": models, "replenish": replenish,
                         "job_id": f"{arm}|{j['job_id']}"})
    return jobs, cells


def reading(rows: Sequence[Mapping[str, str]], cells: Mapping, stats: Sequence[Mapping[str, int]]
            ) -> dict[str, Any]:
    cl = ti.closed_loop_reading(rows, cells, "monitor_fresh")
    g_, f_, c, r = cl["g"], cl["F"], cl["candidate"], cl["reference"]
    worse = (c["deaths"] > r["deaths"] or (c["rundown_share"] or 0) > (r["rundown_share"] or 0)
             or (c["starvation_share"] or 0) > (r["starvation_share"] or 0))
    if g_ is None:
        rd = None
    elif g_ < -f_ or worse:
        rd = "cost"
    elif g_ > f_:
        rd = "gain"
    else:
        rd = "no_cost"
    tot: Counter = Counter()
    for s in stats:
        tot.update(s)
    return {"reading": rd, "g": g_, "F": f_, "candidate": c, "reference": r,
            "closed_loop_accept_agreement": tot["agree"] / tot["bits"] if tot["bits"] else None,
            "decisions": dict(tot)}


def main(argv: Sequence[str] | None = None) -> int:
    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas

    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--candidate", required=True, help="EXP-0120 models.pt (F40 trunk + turn heads)")
    ap.add_argument("--reference", required=True, help="EXP-0114 candidate_it40.pt (F40)")
    ap.add_argument("--replenish", required=True, help="EXP-0110 start_models.pt")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage", default=STAGE, help="the fresh episode set's seed stage")
    a = ap.parse_args(argv)
    if not a.dry_run and len(a.source_commit) != 40:
        im._refuse(["a real run needs a 40-hex --source-commit"])
    ts, _train, mon = im._training(a.training_dir)
    atlas = load_atlas(a.atlas)
    horizon, episodes, mw = (6, 1, 1) if a.dry_run else (100, EPISODES, None)
    jobs, cells = build_jobs(ts, mon, a.candidate, a.reference, a.replenish, horizon, episodes, mw, a.stage)
    # floors need the atlas spreads: rebuild the cells with them
    from .m6_f40_confirm import _cell
    records = {w["world"]: w["record"] for w in ts.training["worlds"]}
    cells = {k: _cell(records[k.split("#")[0]], k, atlas["spreads"]) for k in cells}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = im._pool(episode_job, jobs, a.workers)
    rows = [x["row"] for x in res]
    with (out / "rows.jsonl").open("w") as fh:
        for x in res:
            fh.write(json.dumps(x) + "\n")
    rd = reading(rows, cells, [x["stats"] for x in res if x["row"]["arm"] == "candidate"])
    doc = {"reading": rd, "dry_run": a.dry_run, "source_commit": a.source_commit, "jobs": len(jobs),
           "candidate_sha256": sha256_file(a.candidate), "reference_sha256": sha256_file(a.reference),
           "replenish_sha256": sha256_file(a.replenish), "stage": a.stage, "episodes": episodes}
    (out / "reading.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[turn_closed_loop] {rd['reading']} g {rd['g']} F {rd['F']} agreement "
          f"{rd['closed_loop_accept_agreement']} decisions {rd['decisions']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
