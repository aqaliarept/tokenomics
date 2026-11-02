"""Extrapolate the v3 pilot and the v3 screen's two waves from the 5.2 timings (add-gate-v3 task
5.2, design D3 step 0 and waves 1-2).

Builds the real job lists with the driver's own functions (`gate_cli.build_jobs` over the Drafts
registry v3's development gate worlds, roster-only, no reserves), prices each job from
`timing_screen_v3.jsonl`, and simulates the spawn pool: jobs go to N workers in submission order
(`build_jobs` order: world, episode, member; `gate_cli._execute` submits them so), each idle
worker taking the next job. Run from `/srv/research/tokenomics/tokenomics`:

    ../.venv/bin/python ../openspec/changes/add-gate-v3/probes/extrapolate_screen_v3.py

Price models (local s per horizon-100 episode; pod s = 0.646 × local, sizing of
add-m3-development-audit §EXP-0048):
- v2 worlds: EXP-0047's own job mix timed locally by add-m3-development-audit (12 producers
  0.581 s, 42 producers 2.642 s per run, row-weighted over survived / died), × the drift factor
  measured here on dev-sparse_wide-12-calm-03 (today's surviving members against that
  measurement's 0.685 s). EXP-0047 realized exactly these per-run costs on a pod (3.7 vCPU-h).
- region worlds, **class** (the estimate): a member is priced at the measured member of its class
  on the measured world of its family and regime (`CLASS` below); producers never measured
  (always_restore, always_both, restore_when_low) at the world's slowest measured member.
- region worlds, **upper**: every member at the world's slowest measured member.
The measured region worlds are the most loaded gate worlds of their cell (k_c × m_q), so other
worlds of the cell are priced at or above their own cost.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFTS = HERE.parent / "drafts"
TIMINGS = HERE / "timing_screen_v3.jsonl"

POD = 0.646  # pod vCPU-s per local s (add-m3-development-audit sizing.md §EXP-0048)
WORKER_START = 3.5 * POD  # s per worker per pool (m4 sizing.md §EXP-0057)
PARENT = 60.0  # s of parent work per run (pins, registry replay, outputs)
V2_LOCAL = {12: 0.581, 42: 2.642}  # m3 §EXP-0048: EXP-0047 job mix, local s per run
V2_SURVIVED_THEN = 0.685  # m3 §EXP-0048: 12 producers, survived, dev-sparse_wide-12-calm-03
SETUPS = {"observed": 1.0, "planning": 10.0}  # min of non-driver pod life
SHAPES = {"cpu32": 32, "cpu16": 16}
# (family, regime) -> measured worlds: per cell the most loaded (k_c x m_q) development gate world
# of each choice level the cell holds (ws_loaded: dense and sparse_f4; ws_wide: sparse_f14). A
# member is priced at the max over the cell's worlds, so the price bounds every world of the cell
# (review round 1 of task 5.2: one world per choice level is not the cell's maximum).
MEASURED = {
    ("ws_loaded", "calm"): ("dev-ws_loaded-12-calm-00", "dev-ws_loaded-12-calm-01"),
    ("ws_loaded", "shocked"): ("dev-ws_loaded-12-shocked-00", "dev-ws_loaded-12-shocked-01"),
    ("ws_wide", "calm"): ("dev-ws_wide-42-calm-01",),
    ("ws_wide", "shocked"): ("dev-ws_wide-42-shocked-01",),
}
WHOLE = {"round_robin", "random_whole", "makeable_weighted_whole"}


def load_timings() -> dict[tuple[str, str], float]:
    """(world, member) -> local s at horizon 100 (the first, non-rerun measurement)."""
    out: dict[tuple[str, str], float] = {}
    for line in TIMINGS.read_text().splitlines():
        d = json.loads(line)
        if d["horizon"] == 100 and not d["tag"] and d["failure"] is None:
            out.setdefault((d["world"], d["member"]), d["seconds"])
    return out


def class_member(producing: str, ordering: str) -> str | None:
    """The measured member a roster member is priced at (None = the world's slowest)."""
    if producing == "entity_planner":
        return "whole" if ordering in WHOLE else "split"
    return {"noop": "noop/even_split", "random": "random/random",
            "stock_aware": "stock_aware/proportional",
            "obs_stock_aware": "stock_aware/proportional"}.get(producing)


def price(world, member, t: dict, drift: float, model: str, central: bool = False) -> float:
    """Local s of one job. `central` treats the measured drift as box slowdown, not member mix:
    v2 runs at EXP-0047's own per-run cost and region timings are divided by the drift."""
    if world.family == "sparse_wide":
        return V2_LOCAL[world.size] * (1.0 if central else drift)
    return _region_price(world, member, t, model) / (drift if central else 1.0)


def _region_price(world, member, t: dict, model: str) -> float:
    cell = MEASURED[(world.family, world.regime)]
    measured: dict[str, float] = {}
    for (w, m), s in t.items():
        if w in cell:
            measured[m] = max(measured.get(m, 0.0), s)
    if {w for (w, _) in t if w in cell} != set(cell):
        raise ValueError(f"no timings for some of {cell}")
    slowest = max(measured.values())
    if model == "upper":
        return slowest
    c = class_member(member.producing, member.ordering)
    if c == "whole":
        return max(measured["entity_planner/round_robin"],
                   measured["entity_planner/makeable_weighted_whole"])
    if c == "split":
        return max(measured["entity_planner/proportional"],
                   measured["entity_planner/makeable_split"])
    return measured[c] if c is not None else slowest


def simulate(costs: list[float], workers: int) -> tuple[float, float]:
    """(pool wall s, longest job s) for pod costs in submission order."""
    free = [WORKER_START] * workers
    for c in costs:
        i = min(range(workers), key=free.__getitem__)
        free[i] += c
    return max(free), max(costs)


def _wave_costs(name: str, segments, t: dict, drift: float, model: str, central: bool,
                new_only, v2_draft, repro) -> tuple[list[float], dict[str, float]]:
    """Pod s of every job of one wave (its segments submitted to one pool, in order)."""
    from tokenomics.eval.gate_cli import build_jobs

    costs: list[float] = []
    by_family: dict[str, float] = defaultdict(float)
    for worlds, roster in segments:
        mem = {m.id: m for m in roster.members}
        for e in worlds:
            eps = (5 if name.startswith("screen-red") and e.family != "sparse_wide"
                   and e.regime == "shocked" else None)
            for j in build_jobs([e], roster, None, episodes_per_world=eps):
                c = price(e, mem[j["subject"]], t, drift, model, central) * POD
                costs.append(c)
                by_family[e.family] += c
        if name == "screen-red12" and roster is new_only:  # the reproduction check
            vm = {m.id: m for m in v2_draft.members}
            for e in repro:
                for j in build_jobs([e], v2_draft, None, episodes_per_world=1):
                    c = price(e, vm[j["subject"]], t, drift, model, central) * POD
                    costs.append(c)
                    by_family["sparse_wide-repro"] += c
    return costs, dict(by_family)


def main() -> int:
    from tokenomics.eval.gate_cli import build_jobs
    from tokenomics.eval.registry import DEVELOPMENT, load_registry
    from tokenomics.eval.roster import Roster, RosterMember, contract_for, is_stochastic, load_roster

    t = load_timings()
    now = [s for (w, m), s in t.items() if w == "dev-sparse_wide-12-calm-03"
           and m.startswith(("entity_planner/", "stock_aware/"))]
    drift = (sum(now) / len(now)) / V2_SURVIVED_THEN
    reg = load_registry(DRAFTS / "worlds_v3_draft.json")
    gate = list(reg.gate_worlds(DEVELOPMENT))
    pilot = load_roster(DRAFTS / "roster_v3_pilot_ep.json")
    draft = load_roster(DRAFTS / "roster_v3_draft.json")
    extra = [RosterMember(p, o, contract_for(p, o), is_stochastic(p, o))
             for p, o in (("stock_aware", "round_robin"), ("stock_aware", "random"))]
    w2 = Roster(name="w2-model", replicates=draft.replicates,
                members=tuple(m for m in draft.members if m.producing == "entity_planner")
                + tuple(extra))
    region12 = [e for e in gate if e.family == "ws_loaded"]
    v2 = [e for e in gate if e.family == "sparse_wide"]
    wide = [e for e in gate if e.family == "ws_wide"]
    # a run = waves; a wave = segments (worlds, roster) submitted to one pool
    runs = {"pilot": [[(gate, pilot)]], "screen": [[(v2 + region12, draft)], [(wide, w2)]]}
    out = {"drift": drift, "pod_factor": POD, "runs": {}}
    # "screen-red1" = D3 reduction 1: shocked region worlds at 5 episodes instead of 10
    # "screen-red12" = reductions 1 and 2: also wave 1's v2-world rows of the 48 v2-draft members
    # reused from EXP-0047 (only the 40 new members run there), plus the reproduction check: one
    # world-episode per (size, regime) x all 48 v2-draft members
    from tokenomics.eval.roster import build_roster_v2

    v2_ids = set(build_roster_v2().member_ids)
    new_only = Roster(name="v3-new-members", replicates=draft.replicates,
                      members=tuple(m for m in draft.members if m.id not in v2_ids))
    v2_draft = build_roster_v2()
    repro = [next(e for e in v2 if (e.size, e.regime) == (z, g)) for z in (12, 42)
             for g in ("calm", "shocked")]
    runs["screen-red12"] = [[(v2, new_only), (region12, draft)], [(wide, w2)]]
    central = "--central" in sys.argv
    out["central"] = central
    for name, waves in [("pilot", runs["pilot"]), ("screen", runs["screen"]),
                        ("screen-red1", runs["screen"]), ("screen-red12", runs["screen-red12"])]:
        res: dict = {"waves": []}
        for model in ("class", "upper"):
            wave_costs = []
            for segments in waves:
                wave_costs.append(_wave_costs(name, segments, t, drift, model, central,
                                              new_only, v2_draft, repro))
            res[model] = {"compute_vcpu_h": sum(sum(c) for c, *_ in wave_costs) / 3600,
                          "jobs": sum(len(c) for c, *_ in wave_costs),
                          "by_family_vcpu_h": {f: v / 3600 for _, bf, *_ in wave_costs
                                               for f, v in bf.items()}}
            shapes = {}
            for shape, n in SHAPES.items():
                walls, longest = [], 0.0
                for costs, *_ in wave_costs:
                    w, lj = simulate(costs, n)
                    walls.append(w)
                    longest = max(longest, lj)
                wall = sum(walls) + PARENT
                billed = {k: n * (wall + 60 * m) / 3600 for k, m in SETUPS.items()}
                shapes[shape] = {"wall_min": wall / 60, "wave_walls_min": [w / 60 for w in walls],
                                 "longest_job_pod_s": longest,
                                 "busy": sum(sum(c) for c, *_ in wave_costs) / (n * sum(walls)),
                                 "billed": billed,
                                 "stop": 2 * billed["planning"],
                                 "stop_pod_life_min": 2 * billed["planning"] * 60 / n}
            res[model]["shapes"] = shapes
        res["waves"] = [[{"worlds": len(w), "roster": r.name, "members": len(r.members),
                          "runs_per_world_episode": sum(r.replicates_for(m) for m in r.members)}
                         for w, r in segments] for segments in waves]
        out["runs"][name] = res
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
