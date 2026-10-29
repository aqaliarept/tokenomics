"""Task 8.2 timing probe for EXP-0054 (sizing.md §EXP-0054; design D6 "Sizing").

PoC-C's arms by composed cost, as task 6.3 (`timing_63.py`) did for PoC-A, one cell per command,
`--workers 1`, every command under 2 min. Worlds come from EXP-0052's
`reclassified/atlas_v2.json`, read in place; checkpoints are the registered EXP-0035
`producer_candidate.pt` and EXP-0012 `orderer_scorer.pt`.

Modes:
- `cell --world W --role A|T --horizon H [--search-cycles a,b,..] [--budget S]`: one role
  episode built as `teacher_audit.run_teacher_episode` builds it for the role's job (A: the
  portable producer plays the live episode and every rollout; T: `entity_planner`), `round_robin`
  floor, whole-order + trained candidates, posterior / deterministic scenarios, recorder from
  reset, except that the search acts only on `--search-cycles` (the floor elsewhere) and stops
  timing once `--budget` seconds of search are spent (the rest of that cycle's streams are
  counted, not timed: `max_search_actions`). Horizon <= 12 uses `shocks-i-smoke`; horizon 100
  keeps the world's registered window. The episode stops after the last search cycle. One JSON
  line per searched cycle (streams, timed streams, candidates per stream, decisions by scenarios
  used, seconds per timed stream with forks and shared floor rollouts included, sampler seconds,
  unresolved decisions, acceptance rate) + a summary line (wall, played seconds outside the
  search, checkpoint load).
- `plain --world W --role S|B`: one whole horizon-100 episode of a search-free role arm through
  the driver's job path (`run_audit_job`), timed.
- `sampler --world W --cycles 30,60,90`: plays the portable producer + `round_robin` (recorder
  on) at horizon 100 and times one posterior sampler call at each cycle (fresh cache).

Run from /srv/research/tokenomics/tokenomics with ../.venv/bin/python.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ATLAS = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
PRODUCER = REPO / "experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt"


class StopProbe(Exception):
    pass


def _view(world: str) -> dict:
    from tokenomics.eval import teacher_audit as ta

    return ta.select_worlds(ta.load_atlas(ATLAS), [world])[0]


def _job(view: dict, role: str, horizon: int) -> dict:
    from tokenomics.eval import teacher_audit as ta
    from tokenomics.eval.provenance import sha256_file
    from tokenomics.eval.routing_audit import build_jobs

    e = ta.world_entry(view, "shocks-i-smoke" if horizon <= 12 and view["shocks"] else None)
    pins = {"producer": {"path": str(PRODUCER), "sha256": sha256_file(PRODUCER)},
            "orderer": {"path": str(ORDERER), "sha256": sha256_file(ORDERER)}}
    arm = ta.role_arm(role)
    job = build_jobs([e], [arm], 1, horizon, None, pins)[0]
    if arm.get("teacher"):
        job["teacher"] = arm["teacher"]
    return job


def _producer(job: dict):  # noqa: ANN202 - a Policy
    from tokenomics.eval import teacher_audit as ta
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.routing_audit import trained_policy

    return (trained_policy(job, "producer") if job["producing"] == ta.PORTABLE
            else PRODUCING_BASELINES[ta.PRODUCER])


def cell(world: str, role: str, horizon: int, cycles: set[int] | None, budget: float) -> None:
    from tokenomics.env.shock_posterior import (HistoryRecorder, PosteriorScenarioSource,
                                                ShockPosterior)
    from tokenomics.eval import teacher_audit as ta
    from tokenomics.eval.bounded_rollout_orderer import (BoundedScenarioOrderer,
                                                         DeterministicSource, PosteriorSource,
                                                         WholeOrderCandidates,
                                                         deterministic_future)
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.routing_audit import job_config, ordering_policy, trained_policy
    from tokenomics.eval.runner import run_episode

    view = _view(world)
    job = _job(view, role, horizon)
    assert job.get("teacher") == ta.D2, job
    cfg = job_config(job)
    env = joint_stack_env(cfg, horizon)
    t0 = time.perf_counter()
    prod = _producer(job)
    trained = trained_policy(job, "orderer")
    load_s = time.perf_counter() - t0
    floor = ordering_policy(ta.FLOOR_ROUTER)
    posterior = None
    recorder = None
    if deterministic_future(cfg):
        source: object = DeterministicSource()
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
    turns: Counter = Counter()

    def order(e: object, agent: str):  # noqa: ANN202 - a Policy
        nonlocal spent
        c = int(e.current_cycle)
        if c > last:
            raise StopProbe
        turns[c] += 1
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
                p = per.setdefault(c, {"unresolved": 0})
                p["unresolved"] += 1
                return floor(e, agent)
        if spent >= budget:
            search.max_search_actions = search.telemetry.search_actions  # count, do not time
        n = len(search.decisions)
        a = search(e, agent)
        for d in search.decisions[n:]:
            spent += d.seconds
            per.setdefault(c, {"unresolved": 0}).setdefault("decisions", []).append(d)
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
            rows = [r for r in posterior.log if r.cycle in (c, c + 1)]
            rate = rows[-1].rate if rows else None
        print(json.dumps({
            "mode": "cell", "world": world, "role": role, "horizon": horizon,
            "source": source.name, "cycle": c, "orderer_turns": turns[c],  # type: ignore[attr-defined]
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
    print(json.dumps({"mode": "cell-summary", "world": world, "role": role, "horizon": horizon,
                      "search_cycles": sorted(cycles) if cycles else None, "wall_s": wall,
                      "search_s": spent, "played_s": wall - spent, "ckpt_load_s": load_s,
                      "budget_s": budget}), flush=True)


def plain(world: str, role: str) -> None:
    from tokenomics.eval import teacher_audit as ta

    job = _job(_view(world), role, 100)
    assert not job.get("teacher"), job
    t = time.perf_counter()
    row = ta.run_audit_job(job)
    print(json.dumps({"mode": "plain", "world": world, "role": role, "episode": job["episode"],
                      "seconds": time.perf_counter() - t, "units": row.get("units"),
                      "dead": row.get("dead"), "cycles": row.get("cycles"),
                      "error": row.get("error") or row.get("failure")}), flush=True)


def sampler(world: str, at: list[int]) -> None:
    from tokenomics.env.shock_posterior import (HistoryRecorder, PosteriorScenarioSource,
                                                ShockPosterior)
    from tokenomics.eval import teacher_audit as ta
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.routing_audit import job_config, ordering_policy
    from tokenomics.eval.runner import run_episode

    job = _job(_view(world), ta.ROLE_A, 100)
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
            out.append({"mode": "sampler", "world": world, "role": "A", "cycle": c,
                        "seconds": time.perf_counter() - t, "draws": s.draws,
                        "accepted": s.accepted, "unresolved": s.unresolved})
            todo.pop(0)
            if not todo:
                raise StopProbe
        return floor(e, agent)

    started = time.perf_counter()
    try:
        run_episode(joint_stack_env(cfg, 100), rec.wrap(_producer(job)), rec.wrap(order),
                    seed=job["episode"])
    except StopProbe:
        pass
    for o in out:
        print(json.dumps(o), flush=True)
    print(json.dumps({"mode": "sampler-summary", "world": world, "role": "A",
                      "played_s_with_recorder": time.perf_counter() - started}), flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("cell", "plain", "sampler"))
    p.add_argument("--world", required=True)
    p.add_argument("--role", default="A")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--search-cycles")
    p.add_argument("--budget", type=float, default=90.0)
    p.add_argument("--cycles", default="30,60,90")
    a = p.parse_args(argv)
    if a.mode == "cell":
        cyc = {int(x) for x in a.search_cycles.split(",")} if a.search_cycles else None
        cell(a.world, a.role, a.horizon, cyc, a.budget)
    elif a.mode == "plain":
        plain(a.world, a.role)
    else:
        sampler(a.world, [int(x) for x in a.cycles.split(",")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
