"""Task 6.3 timing probe for EXP-0053 (sizing.md §EXP-0053; design D5 "Sizing").

Times the teacher audit's search by composed cost, one cell per command, `--workers 1`, every
command under 2 min. Worlds come from EXP-0052's `reclassified/atlas_v2.json`, read in place.

Modes:
- `cell --world W --arm d2|ub --horizon H [--search-cycles a,b,..] [--budget S]`: one teacher
  episode built exactly as `teacher_audit.run_teacher_episode` builds it (entity_planner,
  `round_robin` floor, whole-order + trained candidates, posterior / deterministic / clairvoyant
  scenarios, recorder from reset), except that the search acts only on `--search-cycles` (the
  floor elsewhere) and stops searching once `--budget` seconds of search are spent (the rest of
  that cycle's streams are counted, not timed: `max_search_actions`). Horizon <= 12 uses
  `shocks-i-smoke` on shocked worlds; horizon 100 keeps the world's registered window. The
  episode stops after the last search cycle. Prints one JSON line per searched cycle: streams
  (every stream the search would search), timed streams, candidates per stream, decisions by
  scenarios used (4 / 8 / 16 / 1), seconds per timed stream (decision seconds / its streams,
  forks and the shared floor rollouts included), sampler seconds, unresolved decisions,
  acceptance rate.
- `sampler --world W --cycles 30,60,90`: plays `round_robin` (recorder on) at horizon 100 and
  times one posterior sampler call at each cycle (fresh cache).
- `rr --world W` (atlas world) or `--registry-world ID`: one traced `round_robin` horizon-100
  episode through the atlas / routing_audit job, for the 0.646 local-to-pod factor check against
  sizing.md §EXP-0048 (dev-sparse_wide-12-calm-03: 0.73-0.77 s).

Run from /srv/research/tokenomics/tokenomics with ../.venv/bin/python; the trained orderer is
EXP-0012's `orderer_scorer.pt`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import fmean

REPO = Path(__file__).resolve().parents[4]
ATLAS = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
CKPT = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"


class StopProbe(Exception):
    pass


def _view(world: str) -> dict:
    from tokenomics.eval import teacher_audit as ta

    atlas = ta.load_atlas(ATLAS)
    return ta.select_worlds(atlas, [world])[0]


def _job(view: dict, arm: str, horizon: int) -> dict:
    from tokenomics.eval import teacher_audit as ta
    from tokenomics.eval.provenance import sha256_file
    from tokenomics.eval.routing_audit import build_jobs

    e = ta.world_entry(view, "shocks-i-smoke" if horizon <= 12 and view["shocks"] else None)
    job = build_jobs([e], [ta.teacher_arm(arm)], 1, horizon, None,
                     {"orderer": {"path": str(CKPT), "sha256": sha256_file(CKPT)}})[0]
    job["teacher"] = arm
    return job


def cell(world: str, arm: str, horizon: int, cycles: set[int] | None, budget: float) -> None:
    from tokenomics.env.shock_posterior import (HistoryRecorder, PosteriorScenarioSource,
                                                ShockPosterior)
    from tokenomics.eval import teacher_audit as ta
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.bounded_rollout_orderer import (BoundedScenarioOrderer,
                                                         ClairvoyantSource, DeterministicSource,
                                                         PosteriorSource, WholeOrderCandidates,
                                                         deterministic_future)
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.routing_audit import job_config, ordering_policy, trained_policy
    from tokenomics.eval.runner import run_episode

    view = _view(world)
    job = _job(view, arm, horizon)
    cfg = job_config(job)
    env = joint_stack_env(cfg, horizon)
    prod = PRODUCING_BASELINES[ta.PRODUCER]
    floor = ordering_policy(ta.FLOOR_ROUTER)
    t0 = time.perf_counter()
    trained = trained_policy(job, "orderer")
    load_s = time.perf_counter() - t0
    posterior = None
    recorder = None
    if arm == ta.UB:
        source: object = ClairvoyantSource()
    elif deterministic_future(cfg):
        source = DeterministicSource()
    else:
        recorder = HistoryRecorder(keep_replay=False)
        posterior = PosteriorScenarioSource(recorder, ShockPosterior(cfg), task_id=job["world"],
                                            episode_seed=job["episode"])
        source = PosteriorSource(posterior)
    search = BoundedScenarioOrderer(
        task_id=job["world"], episode_seed=job["episode"], prod=prod, depth=ta.DEPTH,
        base=floor, scenarios=source, candidates=WholeOrderCandidates(trained),  # type: ignore[arg-type]
        record_decisions=True)
    last = max(cycles) if cycles else horizon
    spent = 0.0
    per: dict[int, dict] = {}
    sampler_s: dict[int, float] = {}

    def order(e: object, agent: str):  # noqa: ANN202 - a Policy
        nonlocal spent
        c = int(e.current_cycle)
        if c > last:
            raise StopProbe
        if cycles is not None and c not in cycles:
            return floor(e, agent)
        if posterior is not None:
            if c not in sampler_s:
                t = time.perf_counter()
                s = posterior.sample(e)
                sampler_s[c] = time.perf_counter() - t
            else:
                s = posterior.sample(e)
            if s.unresolved:
                per.setdefault(c, {"unresolved": 0})["unresolved"] = per.get(
                    c, {}).get("unresolved", 0) + 1
                return floor(e, agent)
        if spent >= budget:
            search.max_search_actions = search.telemetry.search_actions  # count, do not time
        n = len(search.decisions)
        a = search(e, agent)
        for d in search.decisions[n:]:
            spent += d.seconds
            p = per.setdefault(c, {"unresolved": 0})
            p.setdefault("decisions", []).append(d)
        return a

    prod_p, order_p = (recorder.wrap(prod), recorder.wrap(order)) if recorder else (prod, order)
    started = time.perf_counter()
    try:
        run_episode(env, prod_p, order_p, seed=job["episode"], action_seed=None)
    except StopProbe:
        pass
    wall = time.perf_counter() - started
    for c in sorted(per):
        ds = per[c].get("decisions", [])
        timed = [d for d in ds if d.searched and not d.capped]
        streams = sum(d.searched for d in ds)
        tstreams = sum(d.searched for d in timed)
        rate = None
        if posterior is not None:
            rows = [r for r in posterior.log if r.cycle == c + 1 or r.cycle == c]
            rate = rows[-1].rate if rows else None
        print(json.dumps({
            "mode": "cell", "world": world, "arm": arm, "horizon": horizon,
            "source": source.name, "cycle": c,  # type: ignore[attr-defined]
            "decisions": len(ds), "streams": streams, "timed_streams": tstreams,
            "candidates_per_stream": (sum(d.candidates for d in timed) / tstreams
                                      if tstreams else None),
            "scenario_mix": dict(Counter(d.scenarios for d in timed if d.searched)),
            "timed_seconds": sum(d.seconds for d in timed),
            "s_per_stream": sum(d.seconds for d in timed) / tstreams if tstreams else None,
            "overrides": sum(d.overrides for d in ds),
            "unresolved_decisions": per[c]["unresolved"],
            "sampler_s": sampler_s.get(c), "acceptance_rate": rate,
        }), flush=True)
    print(json.dumps({"mode": "cell-summary", "world": world, "arm": arm, "horizon": horizon,
                      "wall_s": wall, "search_s": spent, "ckpt_load_s": load_s,
                      "budget_s": budget}), flush=True)


def sampler(world: str, at: list[int]) -> None:
    from tokenomics.env.shock_posterior import (HistoryRecorder, PosteriorScenarioSource,
                                                ShockPosterior)
    from tokenomics.eval import teacher_audit as ta
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.routing_audit import job_config, ordering_policy
    from tokenomics.eval.runner import run_episode

    view = _view(world)
    job = _job(view, ta.FLOOR, 100)
    cfg = job_config(job)
    rec = HistoryRecorder(keep_replay=False)
    post = PosteriorScenarioSource(rec, ShockPosterior(cfg), task_id=job["world"],
                                   episode_seed=job["episode"])
    floor = ordering_policy(ta.FLOOR_ROUTER)
    todo = sorted(at)
    out: list[dict] = []

    def order(e: object, agent: str):  # noqa: ANN202
        c = int(e.current_cycle)
        if todo and c >= todo[0]:
            post._cache = None
            t = time.perf_counter()
            s = post.sample(e)
            out.append({"mode": "sampler", "world": world, "cycle": c,
                        "seconds": time.perf_counter() - t, "draws": s.draws,
                        "accepted": s.accepted, "unresolved": s.unresolved})
            todo.pop(0)
            if not todo:
                raise StopProbe
        return floor(e, agent)

    started = time.perf_counter()
    try:
        run_episode(joint_stack_env(cfg, 100), rec.wrap(PRODUCING_BASELINES[ta.PRODUCER]),
                    rec.wrap(order), seed=job["episode"])
    except StopProbe:
        pass
    for o in out:
        print(json.dumps(o), flush=True)
    print(json.dumps({"mode": "sampler-summary", "world": world,
                      "played_s_with_recorder": time.perf_counter() - started}), flush=True)


def rr(world: str | None, registry_world: str | None) -> None:
    from tokenomics.eval.atlas import run_atlas_job
    from tokenomics.eval.registry import DEFAULT_REGISTRY, load_registry
    from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster
    from tokenomics.eval.routing_audit import build_jobs, router_table, run_job

    roster = load_roster(str(DEFAULT_ROSTER))
    arm = [a for a in router_table(roster, ["round_robin"])]
    if world:
        from tokenomics.eval import teacher_audit as ta

        e = ta.world_entry(_view(world))
        fn = run_atlas_job
    else:
        e = next(x for x in load_registry(DEFAULT_REGISTRY).entries if x.id == registry_world)
        fn = run_job
    job = build_jobs([e], arm, 1, 100, 1)[0]
    t = time.perf_counter()
    row = fn(job)
    print(json.dumps({"mode": "rr", "world": e.id, "episode": job["episode"],
                      "seconds": time.perf_counter() - t, "units": row.get("units"),
                      "dead": row.get("dead")}), flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("cell", "sampler", "rr"))
    p.add_argument("--world")
    p.add_argument("--registry-world")
    p.add_argument("--arm", default="d2")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--search-cycles")
    p.add_argument("--budget", type=float, default=90.0)
    p.add_argument("--cycles", default="30,60,90")
    a = p.parse_args(argv)
    if a.mode == "cell":
        cyc = {int(x) for x in a.search_cycles.split(",")} if a.search_cycles else None
        cell(a.world, a.arm, a.horizon, cyc, a.budget)
    elif a.mode == "sampler":
        sampler(a.world, [int(x) for x in a.cycles.split(",")])
    else:
        rr(a.world, a.registry_world)
    return 0


if __name__ == "__main__":
    sys.exit(main())
