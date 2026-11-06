"""EXP-0104: why the imitated M6 stack drains capacity on held-out 42-0125 (DIAGNOSTICS check).

`python -m tokenomics.train.m6_drain --atlas ATLAS --models MODELS --out-dir DIR [--workers N]`

At horizon 200, the GNN stack (EXP-0100 models), k5_rf5 + MWWk, and on 42-0125 also the GNN producer + MWWk,
on the held-out world 42-0125 (EXP-0103's episodes, stage m5-heldout) and on the loaded, non-collapse
42-producer atlas worlds outside the held-out set (atlas train + validation, rho > 0.5; fresh seeds, stage
m6-drain). Per episode, a read-only tracer (instance wrappers, eval-side; the domain is untouched) records:
- consumer units and mean total capacity per producer, per cycle (does the drain bite after cycle 100?);
- every investment order at its funding cycle (`System.start_ordering`): type, producer, funded or not,
  tokens against its cost (fully / partly funded);
- every investment's fate (`investment_completed` / `investment_rejected`);
- per producer-cycle, the capacity ratio after the cycle; for the producer-cycles below X ("needy"),
  whether a restoration was running, asked that cycle, or neither ("never ordered").

The classes the check discriminates (DIAGNOSTICS 30 / 29 and the new branch): needy capacity whose
restorations are **unfunded** (fund dilution), **never ordered** (the policy does not restore), or funded
but **rejected** (the restoration product's auction or TTL), against the base on the same keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from ..env.potential import producer_ratio
from . import m6_imitation as im

HORIZON = 200
DRAIN_STAGE = "m6-drain"
HELDOUT_WORLD = "ws-atlas-v1-42-0125"
RHO_MIN = 0.5
EPISODES = {"held-out": 8, "other": 4}
REPLICATES = {"held-out": (0, 1), "other": (0,)}


class DrainTracer:
    """Read-only per-cycle record of one episode (attach before `run_episode`)."""

    def __init__(self, x: float) -> None:
        self.x = x
        self.orders: list[tuple] = []  # (cycle, pid, type, funded, tokens, cost)
        self.fates: list[tuple] = []  # (cycle, pid, type, "done" | "rejected")
        self.asked: dict[tuple[int, str], dict[str, bool]] = {}
        self.ratio: dict[int, dict[str, float]] = {}
        self.cap: list[float] = []
        self.units: list[float] = []

    def attach(self, env: Any) -> None:
        orig = env.reset

        def reset(*a: object, **k: object) -> object:
            out = orig(*a, **k)
            self._wrap(env.system)
            return out

        env.reset = reset

    def _wrap(self, system: Any) -> None:
        so, cc, act = system.start_ordering, system.complete_cycle, system.producing_agent_action

        def start_ordering() -> None:
            new = [o for o in system.orders.values()
                   if o.requires_funding() and o.investment_request is not None]
            so()
            c = system.cycle_counter
            for o in new:
                r = o.investment_request
                self.orders.append((c, r.producer_id, r.type.name, not o.requires_funding(),
                                    int(o.tokens or 0), int(r.cost or 0)))

        def complete_cycle() -> Any:
            c = system.cycle_counter
            res = cc()
            caps = []
            self.ratio[c] = {}
            for s in res.end_capacities:
                r = producer_ratio(s.capacities, s.max_capacities)
                if r is not None:
                    self.ratio[c][s.producer_id] = r
                caps.append(sum(s.capacities.values()))
            self.cap.append(sum(caps) / max(1, len(caps)))
            self.units.append(float(res.consumer_completed_units))
            return res

        def producing_agent_action(pid: str, cmd: Any) -> None:
            p = system.producing_agents[pid]
            self.asked[(system.cycle_counter, pid)] = {
                "running": bool(p.restoration_running), "restore": bool(cmd.do_restoration),
                "upgrade": bool(cmd.do_upgrade), "expand": bool(cmd.do_inventory_expansion)}
            return act(pid, cmd)

        system.start_ordering, system.complete_cycle = start_ordering, complete_cycle
        system.producing_agent_action = producing_agent_action
        for pid, p in system.producing_agents.items():
            for name, tag in (("investment_completed", "done"), ("investment_rejected", "rejected")):
                fn = getattr(p, name)

                def w(request: Any, _fn: Any = fn, _tag: str = tag, _pid: str = pid) -> None:
                    self.fates.append((system.cycle_counter, _pid, request.type.name, _tag))
                    return _fn(request)

                setattr(p, name, w)

    def summary(self) -> dict[str, Any]:
        """Per type: orders asked / funded fully / partly / unfunded / completed / rejected; per
        needy producer-cycle (ratio after the previous cycle < X): running / asked / never."""
        out: dict[str, Any] = {}
        for t in ("RESTORATION", "UPGRADE", "INVENTORY_EXPANSION"):
            os_ = [o for o in self.orders if o[2] == t]
            fs = [f for f in self.fates if f[2] == t]
            out[t.lower()] = {
                "orders": len(os_), "unfunded": sum(not o[3] for o in os_),
                "funded_full": sum(o[3] and o[4] >= o[5] for o in os_),
                "funded_partial": sum(o[3] and o[4] < o[5] for o in os_),
                "completed": sum(f[3] == "done" for f in fs),
                "rejected": sum(f[3] == "rejected" for f in fs)}
        needy = {"producer_cycles": 0, "running": 0, "asked": 0, "never": 0}
        cycles = sorted(self.ratio)
        for prev, c in zip(cycles, cycles[1:]):
            for pid, r in self.ratio[prev].items():
                if r >= self.x or (c, pid) not in self.asked:
                    continue
                a = self.asked[(c, pid)]
                needy["producer_cycles"] += 1
                needy["running" if a["running"] else "asked" if a["restore"] else "never"] += 1
        out["needy"] = needy
        funded_pids = {o[1] for o in self.orders if o[3]}
        all_pids = set().union(*(set(v) for v in self.ratio.values())) if self.ratio else set()
        out["producers_never_funded"] = len(all_pids - funded_pids)
        return out


def over_ceiling(system: Any) -> int:
    """Producers past the sustainable ceiling: degradation_for(t) > restoration.restores on the
    restoration type (DIAGNOSTICS 29 / 32; a max only grows, so the end count is the episode's)."""
    return sum(1 for p in system.producing_agents.values()
               if p.restoration is not None
               and p.degradation_for(p.restoration.type) > p.restoration.restores)


def drain_job(job: dict) -> dict:
    from ..eval.baselines import producing_policy
    from ..eval.routing_audit import ordering_policy
    from ..eval.runner import run_episode
    from . import m6_gnn as g

    torch.set_num_threads(1)
    env = im._episode_env(job)
    tr = DrainTracer(job["x"])
    tr.attach(env)
    who_p, who_o = im.ARMS[{"gnn": "monitor", "gnn_p": "swap_gnn_producer",
                            "gnn_o": "swap_gnn_orderer", "k5_rf5": "monitor_base"}[job["arm"]]]
    pm, om = im.load_models(job["models"]) if "gnn" in (who_p, who_o) else (None, None)
    p = g.gnn_producing_policy(pm) if who_p == "gnn" else producing_policy(im.BASE_PRODUCER)
    o = g.gnn_ordering_policy(om) if who_o == "gnn" else ordering_policy(im.BASE_ORDERER)
    run_episode(env, p, o, seed=job["episode"], action_seed=None)
    st = env.episode_stats
    r = [float(np.mean(list(tr.ratio[c].values()))) for c in sorted(tr.ratio) if tr.ratio[c]]
    return {"world": job["world"], "set": job["set"], "arm": job["arm"], "episode": job["episode"],
            "replicate": job["replicate"], "dead": bool(st["dead"]), "cycles": int(st["cycles"]),
            "units": float(st["consumer_completed_units"]),
            "units_halves": [sum(tr.units[:100]), sum(tr.units[100:])],
            "cap_at": {str(c): tr.cap[c] for c in (25, 50, 75, 99, 149, 199) if c < len(tr.cap)},
            "ratio_at": {str(c): r[c] for c in (0, 50, 99, 149, 199) if c < len(r)},
            "r_start": st["r_start"], "over_ceiling": over_ceiling(env.system), **tr.summary(),
            "cap_trace": tr.cap, "units_trace": tr.units}


def worlds(atlas: dict) -> list[tuple[dict, str]]:
    from ..eval.teacher_audit import world_view

    out = []
    for e in atlas["worlds"]:
        v = world_view(e)
        v["rho"] = float(e["record"]["coordinates"]["rho"])
        if v["size"] != 42:
            continue
        if v["world"] == HELDOUT_WORLD:
            out.append((v, "held-out"))
        elif v["split"] in ("train", "validation") and v["rho"] > RHO_MIN and v["class"] != "collapse":
            out.append((v, "other"))
    return sorted(out, key=lambda t: t[0]["world"])


def run(atlas_path: str, models: str, out_dir: str, workers: int, max_worlds: int | None,
        horizon: int, episodes: int | None) -> int:
    from ..eval.teacher_audit import load_atlas
    from .order_es_worlds import X_RUNDOWN
    from .producer_read import HELDOUT_STAGE, entry_for

    atlas = load_atlas(atlas_path)
    ws = worlds(atlas)
    if max_worlds:
        ws = ws[:max_worlds]
    jobs = []
    for v, kind in ws:
        e = entry_for(v, episodes or EPISODES[kind], HELDOUT_STAGE if kind == "held-out"
                      else DRAIN_STAGE)
        arms = ("gnn", "gnn_p", "k5_rf5") if kind == "held-out" else ("gnn", "k5_rf5")
        for ep in e.episode_seeds:
            for rep in REPLICATES[kind]:
                for arm in arms:
                    jobs.append({"world": e.id, "size": e.size, "regime": e.regime,
                                 "world_seed": e.world_seed, "generator": e.generator,
                                 "episode": int(ep), "replicate": rep, "horizon": horizon,
                                 "arm": arm, "set": kind, "models": models,
                                 "x": float(X_RUNDOWN), "rho": v["rho"]})
    jobs.sort(key=lambda j: j["arm"] == "k5_rf5")  # the GNN arms are slower: first
    rows = im._pool(drain_job, jobs, workers)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.jsonl").open("w") as fh:
        for r in sorted(rows, key=lambda r: (r["world"], r["arm"], r["episode"], r["replicate"])):
            fh.write(json.dumps(r) + "\n")
    summ = summarise(rows)
    (out / "summary.json").write_text(json.dumps(
        {"source_commit": im.SOURCE_COMMIT, "horizon": horizon, "worlds": [v["world"] for v, _ in ws],
         "x": float(X_RUNDOWN), "per_world_arm": summ}, indent=1) + "\n")
    print(json.dumps(summ)[:2000], flush=True)
    return 0 if len(rows) == len(jobs) else 2


def summarise(rows: Sequence[dict]) -> dict:
    out: dict[str, dict] = {}
    for key in sorted({(r["world"], r["arm"]) for r in rows}):
        rs = [r for r in rows if (r["world"], r["arm"]) == key]
        m = lambda f: fmean(f(r) for r in rs)  # noqa: E731
        out[f"{key[0]}|{key[1]}"] = {
            "episodes": len(rs), "deaths": sum(r["dead"] for r in rs), "units": m(lambda r: r["units"]),
            "units_first_100": m(lambda r: r["units_halves"][0]),
            "units_second_100": m(lambda r: r["units_halves"][1]),
            "cap_at": {c: m(lambda r, c=c: r["cap_at"].get(c, np.nan)) for c in rs[0]["cap_at"]},
            "ratio_at": {c: m(lambda r, c=c: r["ratio_at"].get(c, np.nan)) for c in rs[0]["ratio_at"]},
            **{t: {k: m(lambda r, t=t, k=k: r[t][k]) for k in rs[0][t]}
               for t in ("restoration", "upgrade", "inventory_expansion", "needy")},
            "producers_never_funded": m(lambda r: r["producers_never_funded"])}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--atlas", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--max-worlds", type=int)
    p.add_argument("--episodes", type=int)
    p.add_argument("--source-commit", default="")
    a = p.parse_args(argv)
    im.SOURCE_COMMIT = a.source_commit
    return run(a.atlas, a.models, a.out_dir, a.workers, a.max_worlds, a.horizon, a.episodes)


if __name__ == "__main__":
    sys.exit(main())
