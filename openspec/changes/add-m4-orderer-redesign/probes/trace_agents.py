"""Minimal local agent trace for pre-pod checks (a) and (b) (add-m4-orderer-redesign, design D11
item 3, task 1.3).

The recorded rows (EXP-0052 / EXP-0056 `episodes.csv`) are per-episode aggregates; (a) and (b)
need per-agent, per-turn and per-agent-producer data. This probe replays one world-episode with
one arm through the job path EXP-0056 used (`teacher_audit.world_entry` + `routing_audit`
`build_jobs` / `job_config` / `job_policies`, `joint_stack_env`, `RoutingTracer`,
`LeadReceivers`, `ContractMonitor`, `runner.run_episode`) and wraps the ordering policy from
outside the package (no edit under `tokenomics/tokenomics/`):

- `AgentTrace.__call__` calls the wrapped policy first, then reads (never writes) the acting
  ordering agent's `incoming` orders: one call = one **agent turn**;
- after each `System.complete_cycle` it reads `CycleResult.bid_outcomes`, and of each outcome only
  its kind (`accepted` / `completed` / `failed` / `rejected`), producer, order id and quantity.
  The diagnostic `reason` field is never read (ADR-0058, design D5). Routed units of a cycle =
  accepted + rejected units (every placed bid is one of the two in its auction, `produce`); a
  per-episode cross-check against the placed bids (`p.bids`, read once per cycle inside the
  `complete_cycle` wrapper, after the ordering phase) must agree.

Neutrality: the wrapper draws no random number and changes no decision; the run's units, demand
and shock digests and every scalar `trace_*` cell are compared with the recorded row.

Definitions (stated in notes/pre_pod_checks.md §a / §b):
- agent turn: one ordering-policy call for one ordering agent in one cycle (its AEC step). Under
  `joint_stack` an ordering agent is one product stream of one source (consumer or producer), so
  all orders of a turn have one product and one source class. An *active* turn holds >= 1 order;
- concurrent same-product order group: the orders routed in one cycle's ordering phase (every
  agent's `incoming`, quantity > 0) for one product, when there are >= 2 of them;
- across-agent share (group): share of such groups whose orders come from >= 2 distinct agents;
  (pair): share of the groups' order pairs whose two orders come from different agents (pooled
  over groups). Both are computed on products with >= 2 declaring producers (primary: only there
  can concurrent orders concentrate on one producer) and on all products;
- completion rate of agent a at producer i: completed / routed units over the episode;
- agent_history_dispersion: median, over cells (product x, producer i) where >= 2 agents of x
  routed >= 3 units to i, of the across-agent population SD of those agents' completion rates at
  i (D3). The sample-SD and the >= 2-declarer variants are reported too;
- routed_share_dispersion: per product with >= 2 declaring producers (primary; a one-producer
  product has TV 0 by construction) and >= 2 agents that routed >= 3 units of it, each agent's
  routed-share vector over the producers; the median over all (product, agent pair) of the
  total-variation distance (1/2) sum_i |s_a,i - s_b,i|. The per-product-median-then-median and
  the all-products variants are reported too.

Usage (from the repo root, venv at `.venv`; one world x arm per command, each < 2 min):
  .venv/bin/python openspec/changes/add-m4-orderer-redesign/probes/trace_agents.py \
      --world ws-atlas-v1-12-0060 --arm TR
  .venv/bin/python openspec/changes/add-m4-orderer-redesign/probes/trace_agents.py --report
`--world 42` picks the 42-producer world by the stated rule (`pick_42`). `--workers 1` is the only
value accepted: one process, BLAS / torch pinned to 1 thread. Results land in
`probes/trace_agents_out/<world>__<arm>.json`; `--report` writes notes §a / §b between this
script's own markers in `notes/pre_pod_checks.md` (1.2's block is left untouched).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import itertools
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median, pstdev, stdev
from typing import Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[4]
CHANGE = Path(__file__).resolve().parents[1]
OUT_DIR = CHANGE / "probes" / "trace_agents_out"
NOTES = CHANGE / "notes" / "pre_pod_checks.md"
ATLAS = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ATLAS_HASH = "220886a7cf12aaac0cc0a48dc03c57f2dc2484bb2af6185a6160ec509c8a2a23"
EXP0052_EPISODES = REPO / "experiments/EXP-0052-headroom-atlas-confirm/artifacts/episodes.csv"
EXP0056_EPISODES = REPO / "experiments/EXP-0056-split-vs-rotation/artifacts/episodes.csv"
ORDERER_CKPT = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
ORDERER_SHA = "d25388bf"  # prefix of the registered EXP-0012 SHA-256 (checked below)
HORIZON = 100
WORLDS_12 = ("ws-atlas-v1-12-0060", "ws-atlas-v1-12-0111", "ws-atlas-v1-12-0344")
ARMS = ("TR", "RR", "MWW")
ROUTER = {"TR": "trained_orderer", "RR": "round_robin", "MWW": "makeable_weighted_whole"}
MIN_UNITS = 3  # D3: "routed >= 3 units"
BEGIN = "<!-- BEGIN trace_agents.py (generated; do not edit by hand) -->"
END = "<!-- END trace_agents.py -->"
OTHER_BEGIN = "<!-- BEGIN pre_pod_offline.py"
# D11 numeric triggers
TRIG_ACROSS = 0.5
TRIG_HIST = 0.05
TRIG_SHARE = 0.10
TRIG_WORLDS = 2

Turn = tuple[int, str, str, int]  # (cycle, agent, class, orders)


def pick_42(atlas: Mapping) -> tuple[str, str]:
    """D11's loaded 42-producer `sparse_f14` train world, deterministic rule: atlas v2 train split,
    42 producers, choice `sparse_f14`, rho > 0.28, v2 class not `collapse`; the lowest world id."""
    pool = sorted(w["record"]["world_id"] for w in atlas["worlds"]
                  if w["split"] == "train" and int(w["record"]["size"]) == 42
                  and w["record"]["levers"]["choice"] == "sparse_f14"
                  and w["record"]["coordinates"]["rho"] > 0.28
                  and w["confirm_class"] != "collapse")
    return pool[0], f"lowest id of {pool}"


class AgentTrace:
    """Ordering-policy wrapper (reads only; see module docstring)."""

    def __init__(self, policy: Callable[[object, str], object]) -> None:
        self.policy = policy
        self.turns: list[Turn] = []
        self.order_agent: dict[object, str] = {}
        self.agent_product: dict[str, str] = {}
        self.agent_class: dict[str, str] = {}
        # (cycle, product) -> [agent per routed order]
        self.groups: dict[tuple[int, str], list[str]] = defaultdict(list)
        # (agent, producer) -> kind -> units
        self.units: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.unattributed = 0
        self.bid_units_placed = 0
        self.routed_units_outcomes = 0
        self.declarers: dict[str, int] = {}

    def attach(self, env: object) -> None:
        original_reset = env.reset  # type: ignore[attr-defined]

        def reset(*args: object, **kwargs: object) -> object:
            out = original_reset(*args, **kwargs)
            self._wrap(env.system)  # type: ignore[attr-defined]
            return out

        env.reset = reset  # type: ignore[attr-defined]

    def _wrap(self, system: object) -> None:
        decl: dict[str, int] = defaultdict(int)
        for p in system.producing_agents.values():  # type: ignore[attr-defined]
            for x in p.produces:
                decl[x] += 1
        self.declarers = dict(decl)
        original = system.complete_cycle  # type: ignore[attr-defined]

        def complete_cycle() -> object:
            placed = sum(b.quantity for p in system.producing_agents.values()  # type: ignore[attr-defined]
                         for b in p.bids)
            result = original()
            self._after(result, placed)
            return result

        system.complete_cycle = complete_cycle  # type: ignore[attr-defined]

    def __call__(self, env: object, agent: str) -> object:
        from tokenomics.domain.system import order_kind
        from tokenomics.eval.routing_trace import order_class

        action = self.policy(env, agent)
        system = env.system  # type: ignore[attr-defined]
        _, aid = env._role[agent]  # type: ignore[attr-defined]
        incoming = system.ordering_agents[aid].incoming
        cycle = int(system.cycle_counter)
        n = 0
        cls = self.agent_class.get(agent)
        for oid in sorted(incoming):
            info = incoming[oid]
            if info.quantity <= 0:
                continue
            order = system.orders[oid]
            c = order_class(order_kind(order))
            if cls is None:
                cls = c
            if c != cls or self.agent_product.setdefault(agent, order.product) != order.product:
                raise AssertionError(f"agent {agent} holds more than one stream")
            self.agent_class[agent] = cls
            self.order_agent[oid] = agent
            self.groups[(cycle, order.product)].append(agent)
            n += 1
        self.turns.append((cycle, agent, cls or "", n))
        return action

    def _after(self, result: object, placed: int) -> None:
        self.bid_units_placed += placed
        for o in result.bid_outcomes:  # type: ignore[attr-defined]
            kind = o.outcome  # the outcome kind only; `o.reason` is never read
            agent = self.order_agent.get(o.order_id)
            if agent is None:
                self.unattributed += 1
                continue
            cell = self.units[(agent, o.producer_id)]
            cell[kind] += o.quantity
            if kind in ("accepted", "rejected"):
                cell["routed"] += o.quantity
                self.routed_units_outcomes += o.quantity

    # --- (a) ---
    def turn_stats(self) -> dict:
        out: dict[str, dict] = {}
        for cls in ("consumer", "producer"):
            agents = {a for a, c in self.agent_class.items() if c == cls}
            all_turns = [t for t in self.turns if t[1] in agents]
            active = [t[3] for t in all_turns if t[3] > 0]
            out[cls] = {
                "agents": len(agents), "turns": len(all_turns), "active_turns": len(active),
                "orders": sum(active),
                "orders_per_active_turn": sum(active) / len(active) if active else None,
                "share_active_turns_ge2": (sum(1 for n in active if n >= 2) / len(active)
                                           if active else None),
                "max_orders_per_turn": max(active) if active else 0,
            }
        never = [t for t in self.turns if t[1] not in self.agent_class]
        out["no_order_agents"] = {"turns": len(never),
                                  "agents": len({t[1] for t in never})}
        return out

    def group_stats(self) -> dict:
        out = {}
        for scope in ("choice", "all"):
            groups = [(x, ags) for (_, x), ags in self.groups.items() if len(ags) >= 2
                      and (scope == "all" or self.declarers.get(x, 0) >= 2)]
            n = len(groups)
            across = sum(1 for _, ags in groups if len(set(ags)) >= 2)
            pairs = cross = 0
            orders = orders_x = 0
            within_only = 0
            for _, ags in groups:
                cnt = defaultdict(int)
                for a in ags:
                    cnt[a] += 1
                k = len(ags)
                pairs += k * (k - 1) // 2
                cross += k * (k - 1) // 2 - sum(c * (c - 1) // 2 for c in cnt.values())
                orders += k
                orders_x += sum(c for c in cnt.values() if c < k)
                within_only += int(len(cnt) == 1)
            by_class = defaultdict(int)
            for _, ags in groups:
                by_class["+".join(sorted({self.agent_class[a] for a in ags}))] += 1
            out[scope] = {
                "groups": n, "groups_across": across, "groups_within_only": within_only,
                "across_group_share": across / n if n else None,
                "pairs": pairs, "cross_pairs": cross,
                "across_pair_share": cross / pairs if pairs else None,
                "orders": orders, "orders_with_other_agent": orders_x,
                "across_order_share": orders_x / orders if orders else None,
                "mean_group_size": orders / n if n else None,
                "mean_agents_per_group": (sum(len(set(a)) for _, a in groups) / n) if n else None,
                "groups_by_class": dict(by_class),
            }
        return out

    # --- (b) ---
    def dispersion(self) -> dict:
        by_prod: dict[str, dict[str, dict[str, dict[str, int]]]] = defaultdict(dict)
        for (agent, pid), cell in self.units.items():
            by_prod[self.agent_product[agent]].setdefault(agent, {})[pid] = cell
        cells_p: list[float] = []
        cells_s: list[float] = []
        cells_p_choice: list[float] = []
        hist_detail = []
        tv_pairs: list[float] = []
        tv_pairs_all: list[float] = []
        tv_prod_medians: list[float] = []
        for x, agents in sorted(by_prod.items()):
            choice = self.declarers.get(x, 0) >= 2
            pids = sorted({p for cells in agents.values() for p in cells})
            for pid in pids:
                rates = [cells[pid]["completed"] / cells[pid]["routed"]
                         for cells in agents.values()
                         if pid in cells and cells[pid]["routed"] >= MIN_UNITS]
                if len(rates) >= 2:
                    cells_p.append(pstdev(rates))
                    cells_s.append(stdev(rates))
                    if choice:
                        cells_p_choice.append(pstdev(rates))
                    hist_detail.append({"product": x, "producer": pid, "agents": len(rates),
                                        "rates": [round(r, 4) for r in rates]})
            vecs = []
            for cells in agents.values():
                tot = sum(c["routed"] for c in cells.values())
                if tot >= MIN_UNITS:
                    vecs.append({p: c["routed"] / tot for p, c in cells.items()})
            tvs = [0.5 * sum(abs(u.get(p, 0.0) - v.get(p, 0.0)) for p in pids)
                   for u, v in itertools.combinations(vecs, 2)]
            tv_pairs_all += tvs
            if choice and tvs:
                tv_pairs += tvs
                tv_prod_medians.append(median(tvs))

        def med(v: Sequence[float]) -> float | None:
            return median(v) if v else None

        return {
            "agent_history_dispersion": med(cells_p),
            "agent_history_dispersion_sample_sd": med(cells_s),
            "agent_history_dispersion_choice_products": med(cells_p_choice),
            "history_cells": len(cells_p), "history_cells_choice": len(cells_p_choice),
            "history_cells_zero_sd": sum(1 for s in cells_p if s == 0.0),
            "routed_share_dispersion": med(tv_pairs),
            "routed_share_dispersion_per_product_median": med(tv_prod_medians),
            "routed_share_dispersion_all_products": med(tv_pairs_all),
            "share_pairs": len(tv_pairs), "share_products": len(tv_prod_medians),
            "share_pairs_all": len(tv_pairs_all),
            "history_detail": hist_detail,
        }


def recorded_row(path: Path, world: str, episode: int, subject: str, replicate: int
                 ) -> dict[str, str] | None:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if (r["world"] == world and r["episode"] == str(episode) and r["subject"] == subject
                    and r["replicate"] == str(replicate)):
                return r
    return None


def run(world: str, arm: str, replicate: int = 0) -> dict:
    t0 = time.perf_counter()
    from tokenomics.eval.atlas import CONFIRM, arm_table
    from tokenomics.eval.contract import ContractMonitor
    from tokenomics.eval.gate import row_from_stats, row_to_dict
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.provenance import sha256_file
    from tokenomics.eval.roster import load_roster
    from tokenomics.eval.routing_audit import (
        TRAINED_ORDERER, _trained_entry, build_jobs, job_config, job_policies, trace_cells)
    from tokenomics.eval.routing_trace import RoutingTracer
    from tokenomics.eval.runner import run_episode
    from tokenomics.eval.teacher_audit import (
        LeadReceivers, first_confirm_seed, load_atlas, world_entry, world_view)

    atlas = load_atlas(ATLAS)
    if atlas["content_hash"] != ATLAS_HASH:
        raise SystemExit(f"STOP: atlas content hash {atlas['content_hash']} != {ATLAS_HASH}")
    rule = None
    if world == "42":
        world, rule = pick_42(atlas)
    views = {v["world"]: v for v in map(world_view, atlas["worlds"])}
    view = views[world]
    if view["size"] == 42 and arm != "TR":
        raise SystemExit("D11: the 42-producer world runs TR only")
    entry = world_entry(view)
    seed = first_confirm_seed(view)
    assert entry.episode_seeds == (seed,)
    if arm == "TR":
        table = [_trained_entry(TRAINED_ORDERER)]
    else:
        table = arm_table(load_roster(), CONFIRM, [ROUTER[arm]])
    ckpt_sha = sha256_file(ORDERER_CKPT)
    if not ckpt_sha.startswith(ORDERER_SHA):
        raise SystemExit(f"STOP: orderer checkpoint sha256 {ckpt_sha}")
    jobs = [j for j in build_jobs([entry], table, 1, HORIZON, None,
                                  {"orderer": {"path": str(ORDERER_CKPT), "sha256": ckpt_sha}})
            if j["replicate"] == replicate]
    if len(jobs) != 1:
        raise SystemExit(f"STOP: {arm} has no replicate {replicate} at the confirm's R")
    job = jobs[0]

    env = joint_stack_env(job_config(job), HORIZON)
    tracer = RoutingTracer()
    tracer.attach(env)
    monitor = ContractMonitor(env)
    prod, order = job_policies(job)
    lead = LeadReceivers(order)
    trace = AgentTrace(lead)
    trace.attach(env)
    t1 = time.perf_counter()
    run_episode(env, prod, trace, seed=job["episode"], action_seed=job["action_seed"],
                monitor=monitor)
    t2 = time.perf_counter()
    stats = env.episode_stats
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=HORIZON)
    row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                                     job["subject"], job["replicate"], stats))
    row.update(trace_cells(tracer))
    row.update(lead.cells())

    checks = []
    for name, path in (("EXP-0056", EXP0056_EPISODES), ("EXP-0052", EXP0052_EPISODES)):
        ref = recorded_row(path, job["world"], job["episode"], job["subject"], replicate)
        if ref is None:
            checks.append({"ref": name, "found": False})
            continue
        keys = [k for k in ref if k in row and k not in ("shock_schedule",)
                and (k in ("units", "dead", "demand_digest", "shock_digest", "shock_events",
                           "fulfilled_trace", "fulfilled_units", "ordered_units")
                     or (k.startswith(("trace_", "lead_")) and not ref[k].startswith(("[", "{"))))]
        diff = [k for k in keys if ref[k] != row[k]]
        checks.append({"ref": name, "found": True, "compared": len(keys), "differ": diff,
                       "units_ref": ref["units"], "units": row["units"],
                       "demand_digest_equal": ref["demand_digest"] == row["demand_digest"],
                       "shock_digest_equal": ref.get("shock_digest") == row.get("shock_digest")})
    return {
        "world": job["world"], "arm": arm, "subject": job["subject"], "episode": job["episode"],
        "replicate": replicate, "action_seed": job["action_seed"], "size": job["size"],
        "world_rule": rule, "rho": view["record"]["coordinates"]["rho"],
        "choice": view["choice"], "class": view["class"], "pattern": view["pattern"],
        "shocks": view["shocks"],
        "units": row["units"], "dead": row["dead"], "demand_digest": row["demand_digest"],
        "shock_digest": row.get("shock_digest"),
        "lead_concurrent_top_share_consumer": row["lead_concurrent_top_share_consumer"],
        "checks": checks,
        "consistency": {"bid_units_placed": trace.bid_units_placed,
                        "routed_units_from_outcomes": trace.routed_units_outcomes,
                        "unattributed_outcomes": trace.unattributed},
        "turns": trace.turn_stats(), "groups": trace.group_stats(),
        "dispersion": trace.dispersion(),
        "seconds": {"setup": round(t1 - t0, 2), "episode": round(t2 - t1, 2),
                    "total": round(time.perf_counter() - t0, 2)},
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


# --- report (notes §a / §b) ---


def _f(v: object, nd: int = 3) -> str:
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def short(world: str) -> str:
    return world.replace("ws-atlas-v1-", "")


def report() -> str:
    allres = [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("*.json"))]
    res = {(r["world"], r["arm"]): r for r in allres if r["replicate"] == 0}
    sens = {(r["world"], r["arm"], r["replicate"]): r for r in allres if r["replicate"] != 0}
    lines = [BEGIN, "",
             "Generated by `/srv/research/tokenomics/openspec/changes/add-m4-orderer-redesign/"
             "probes/trace_agents.py --report` (task 1.3, design D11 item 3) from "
             "`probes/trace_agents_out/*.json`.", "",
             "**Descriptive, not a go / no-go.** EXP-0057's dispersion reading (D3) repeats (b) "
             "at scale under TS; this probe only checks D5's premise early. One world-episode per "
             "world (the confirm's first episode seed = `teacher_audit.first_confirm_seed`, equal "
             "to EXP-0056's `episode` for the three 12-producer worlds), horizon 100, "
             "`entity_planner` producer, R = 1 (replicate 0, the confirm's replicate-0 action "
             "seed). One process, `--workers 1`, BLAS / torch 1 thread.", ""]
    lines += ["### Replay and neutrality", "",
              "| world | arm | episode | units | ref | ref units | digests equal | scalar cells "
              "compared / differing | episode s | total s | max RSS MB |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (w, a), r in sorted(res.items()):
        for c in r["checks"]:
            if not c["found"]:
                lines.append(f"| {short(w)} | {a} | {r['episode']} | {r['units']} | {c['ref']} "
                             f"| no row | – | – | {r['seconds']['episode']} | "
                             f"{r['seconds']['total']} | {r['max_rss_mb']} |")
                continue
            dig = "yes" if c["demand_digest_equal"] and c["shock_digest_equal"] else "**NO**"
            lines.append(f"| {short(w)} | {a} | {r['episode']} | {r['units']} | {c['ref']} | "
                         f"{c['units_ref']} | {dig} | {c['compared']} / "
                         f"{len(c['differ']) or 0}{' ' + str(c['differ']) if c['differ'] else ''}"
                         f" | {r['seconds']['episode']} | {r['seconds']['total']} | "
                         f"{r['max_rss_mb']} |")
    lines += ["", "Consistency (per run): routed units from outcome kinds (accepted + rejected) "
              "vs units of the placed bids; unattributed outcomes.", ""]
    for (w, a), r in sorted(res.items()):
        c = r["consistency"]
        ok = (c["bid_units_placed"] == c["routed_units_from_outcomes"]
              and c["unattributed_outcomes"] == 0)
        lines.append(f"- {short(w)} {a}: {c['routed_units_from_outcomes']} vs "
                     f"{c['bid_units_placed']}, unattributed {c['unattributed_outcomes']} → "
                     f"{'equal' if ok else '**MISMATCH**'}")
    w42 = [r for r in res.values() if r["size"] == 42]
    if w42:
        r = w42[0]
        lines += ["", f"42-producer world rule: atlas v2 train, 42 producers, `sparse_f14`, "
                  f"ρ > 0.28, class not `collapse`; {r['world_rule']} → **{short(r['world'])}** "
                  f"(ρ {r['rho']:.3f}, class {r['class']}, pattern {r['pattern']}, shocks "
                  f"{r['shocks']})."]

    # §a
    lines += ["", "## §a. Orders per agent turn and the composition of concurrent same-product "
              "groups", "",
              "Definitions. **Agent turn**: one ordering-policy call for one ordering agent in one "
              "cycle (its AEC step). Under `joint_stack` an ordering agent is one product stream "
              "of one source, so a turn's orders share one product and one source class "
              "(consumer stream / producer stream = replenishment + investment orders). An "
              "*active* turn holds ≥ 1 order (quantity > 0 in `incoming`, re-bids of a rejected "
              "remainder included). **Concurrent same-product order group**: the orders routed "
              "in one cycle for one product, when there are ≥ 2. **Across-agent share**: "
              "(group) share of groups whose orders come from ≥ 2 distinct agents — the task's "
              "definition, used for the trigger; (pair) share of the groups' order pairs whose "
              "two orders come from different agents, pooled over groups — the composition "
              "measure (a group of 9 orders of one agent + 1 of another counts as across in the "
              "group version but 9 / 45 in the pair version). Primary scope: products with ≥ 2 "
              "declaring producers (only there can concurrent orders concentrate); all products "
              "in brackets.", "",
              "| world | arm | consumer: agents / active turns / orders per active turn / share "
              "≥ 2 / max | producer: agents / active turns / orders per active turn / share ≥ 2 "
              "/ max |", "|---|---|---|---|"]
    for (w, a), r in sorted(res.items()):
        t = r["turns"]
        cell = [f"{t[c]['agents']} / {t[c]['active_turns']} / "
                f"{_f(t[c]['orders_per_active_turn'], 2)} / "
                f"{_f(t[c]['share_active_turns_ge2'], 3)} / {t[c]['max_orders_per_turn']}"
                for c in ("consumer", "producer")]
        lines.append(f"| {short(w)} | {a} | {cell[0]} | {cell[1]} |")
    lines += ["", "| world | arm | groups | mean orders / agents per group | across (group) | "
              "across (pair) | across (order) | within-only groups | groups by source class |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (w, a), r in sorted(res.items()):
        g, ga = r["groups"]["choice"], r["groups"]["all"]
        lines.append(
            f"| {short(w)} | {a} | {g['groups']} [{ga['groups']}] | "
            f"{_f(g['mean_group_size'], 2)} / {_f(g['mean_agents_per_group'], 2)} | "
            f"{_f(g['across_group_share'])} [{_f(ga['across_group_share'])}] | "
            f"{_f(g['across_pair_share'])} [{_f(ga['across_pair_share'])}] | "
            f"{_f(g['across_order_share'])} | {g['groups_within_only']} | "
            f"{', '.join(f'{k} {v}' for k, v in sorted(g['groups_by_class'].items()))} |")
    lines += ["", "(order) = share of grouped orders that share their group with another agent's "
              "order."]

    # §b
    lines += ["", "## §b. Dispersion of agent-local history (read under MWW)", "",
              "MWW (`makeable_weighted_whole`, R = 1) is the A0-like sampling regime and the only "
              "arm read; TR and RR are context (under TR every agent routes to the top producer, "
              "so the other producers' rates are just the prior). Built from outcome kinds only. "
              "**Completion rate** of agent a at producer i = completed / routed units over the "
              "episode, routed = accepted + rejected units (each re-bid counts again), no prior. "
              "**agent_history_dispersion** = median over cells (product, producer) where ≥ 2 "
              "agents of the product routed ≥ 3 units to the producer of the across-agent "
              "population SD of their completion rates (sample SD and the ≥ 2-declarer cells "
              "beside it). **routed_share_dispersion** = median over all (product, agent pair), "
              "products with ≥ 2 declaring producers and agents with ≥ 3 routed units of the "
              "product, of the total-variation distance of the two routed-share vectors "
              "(per-product median then median, and all products, beside it).", "",
              "| world | arm | agent_history_dispersion (pop SD) | sample SD | ≥ 2-declarer cells "
              "| cells (zero SD) | routed_share_dispersion | per-product median | all products | "
              "pairs / products |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (w, a), r in sorted(res.items()):
        d = r["dispersion"]
        lines.append(
            f"| {short(w)} | {a} | {_f(d['agent_history_dispersion'])} | "
            f"{_f(d['agent_history_dispersion_sample_sd'])} | "
            f"{_f(d['agent_history_dispersion_choice_products'])} "
            f"({d['history_cells_choice']}) | {d['history_cells']} "
            f"({d['history_cells_zero_sd']}) | {_f(d['routed_share_dispersion'])} | "
            f"{_f(d['routed_share_dispersion_per_product_median'])} | "
            f"{_f(d['routed_share_dispersion_all_products'])} | "
            f"{d['share_pairs']} / {d['share_products']} |")

    if sens:
        lines += ["", "Sensitivity, reported apart (not the D11 reading, which is R = 1 = "
                  "replicate 0): MWW replicate 1 (the confirm's replicate-1 action seed, same "
                  "episode), because MWW's `routed_share_dispersion` sits near its 0.10 line. Neutrality vs EXP-0052's "
                  "replicate-1 row is in the replay table's sense:", "",
                  "| world | arm | replicate | units (EXP-0052 ref) | cells compared / differing "
                  "| agent_history_dispersion | routed_share_dispersion | per-product median | "
                  "across (group) |", "|---|---|---|---|---|---|---|---|---|"]
        for (w, a, rep), r in sorted(sens.items()):
            d = r["dispersion"]
            c = [c for c in r["checks"] if c["ref"] == "EXP-0052"][0]
            lines.append(f"| {short(w)} | {a} | {rep} | {r['units']} ({c.get('units_ref')}) | "
                         f"{c.get('compared')} / {len(c.get('differ') or [])} | "
                         f"{_f(d['agent_history_dispersion'])} | "
                         f"{_f(d['routed_share_dispersion'])} | "
                         f"{_f(d['routed_share_dispersion_per_product_median'])} | "
                         f"{_f(r['groups']['choice']['across_group_share'])} |")

    # triggers
    lines += ["", "## Numeric triggers (design D11 item 4)", ""]
    fired = []
    hist_low = [w for w in WORLDS_12 if (w, "MWW") in res and (
        res[(w, "MWW")]["dispersion"]["agent_history_dispersion"] is None
        or res[(w, "MWW")]["dispersion"]["agent_history_dispersion"] < TRIG_HIST)]
    share_low = [w for w in WORLDS_12 if (w, "MWW") in res and (
        res[(w, "MWW")]["dispersion"]["routed_share_dispersion"] is None
        or res[(w, "MWW")]["dispersion"]["routed_share_dispersion"] < TRIG_SHARE)]
    b_worlds = sorted(set(hist_low) | set(share_low))
    b_fire = len(b_worlds) >= TRIG_WORLDS
    lines.append(f"- (b) under MWW, worlds with `agent_history_dispersion` < {TRIG_HIST}: "
                 f"{[short(w) for w in hist_low] or 'none'}; with `routed_share_dispersion` < "
                 f"{TRIG_SHARE}: {[short(w) for w in share_low] or 'none'}; worlds failing either "
                 f"{len(b_worlds)} of 3 → **{'FIRED' if b_fire else 'not fired'}** (rule: ≥ "
                 f"{TRIG_WORLDS} of 3).")
    if b_fire:
        fired.append("b")
    a_cells = {(w, a): res[(w, a)]["groups"]["choice"]["across_group_share"]
               for w in WORLDS_12 for a in ARMS if (w, a) in res}
    a_low = [k for k, v in a_cells.items() if v is None or v < TRIG_ACROSS]
    mww_low = [w for w in WORLDS_12 if (w, "MWW") in a_low]
    a_fire = len(mww_low) >= TRIG_WORLDS
    lines.append(f"- (a) across-agent share (group, ≥ 2-declarer products) < {TRIG_ACROSS}: "
                 f"cells {[(short(w), a) for w, a in a_low] or 'none'} of {len(a_cells)}; under "
                 f"MWW on {len(mww_low)} of 3 worlds → **{'FIRED' if a_fire else 'not fired'}** "
                 f"(rule as for (b): MWW, ≥ {TRIG_WORLDS} of 3 worlds; any cell below is listed).")
    pair_low = [(short(w), a) for w in WORLDS_12 for a in ARMS if (w, a) in res
                and (res[(w, a)]["groups"]["choice"]["across_pair_share"] or 0) < TRIG_ACROSS]
    lines.append(f"- (a) pair version < {TRIG_ACROSS} (reported, not the trigger): "
                 f"{pair_low or 'none'}.")
    if a_fire:
        fired.append("a")
    mww = {(w, rep): r["dispersion"]["routed_share_dispersion"]
           for (w, a, *rest), r in list(res.items()) + list(sens.items())
           if a == "MWW" and w in WORLDS_12 for rep in [rest[0] if rest else 0]}
    reps = sorted({rep for _, rep in mww})
    near = [(short(w), rep) for (w, rep), v in sorted(mww.items())
            if v is not None and abs(v - TRIG_SHARE) <= 0.02]
    for rep in sorted({k[2] for k in sens}):
        low = [short(w) for w in WORLDS_12 if (w, "MWW", rep) in sens and (
            (sens[(w, "MWW", rep)]["dispersion"]["routed_share_dispersion"] or 0) < TRIG_SHARE
            or (sens[(w, "MWW", rep)]["dispersion"]["agent_history_dispersion"] or 0)
            < TRIG_HIST)]
        lines.append(f"- Fragility (sensitivity only, not the reading): with MWW replicate {rep} "
                     f"the (b) worlds failing either measure are {low or 'none'} "
                     f"({len(low)} of 3 → {'would fire' if len(low) >= TRIG_WORLDS else 'would not fire'}).")
    if reps:
        per_world = "; ".join(
            f"{short(w)} " + " / ".join(_f(mww.get((w, rep))) for rep in reps)
            for w in WORLDS_12)
        lines.append(f"- MWW `routed_share_dispersion` per world, replicate "
                     f"{' / '.join(map(str, reps))}: {per_world}. Within ±0.02 of {TRIG_SHARE}: "
                     f"{len(near)} of {len(mww)} (world, replicate) readings, "
                     f"{', '.join(f'{w} r{rep}' for w, rep in near) or 'none'}. The (b) share "
                     f"measure is borderline, so EXP-0057's at-scale TS reading (2 episodes × "
                     f"R = 2 per world, 23 worlds) is the one to trust.")
    lines += ["", f"**Triggers fired: {', '.join(fired) if fired else 'none'}.**", "", END]
    return "\n".join(lines)


def write_notes(block: str) -> None:
    text = NOTES.read_text()
    if BEGIN in text and END in text:
        pre, rest = text.split(BEGIN, 1)
        text = pre + block + rest.split(END, 1)[1]
    else:
        if OTHER_BEGIN not in text:
            raise SystemExit("STOP: pre_pod_offline.py's block not found in the notes")
        pre, post = text.split(OTHER_BEGIN, 1)
        text = pre + block + "\n\n" + OTHER_BEGIN + post
    NOTES.write_text(text)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--world", help="atlas world id, or 42 for the D11 42-producer world")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--replicate", type=int, default=0,
                    help="stochastic arm's replicate (0 = the D11 reading; 1 = sensitivity)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.workers != 1:
        raise SystemExit("the shared box allows --workers 1 only")
    if a.report:
        block = report()
        write_notes(block)
        print(block)
        return 0
    if not a.world or not a.arm:
        ap.error("--world and --arm (or --report)")
    r = run(a.world, a.arm, a.replicate)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / (f"{short(r['world'])}__{a.arm}"
                      + (f"__r{a.replicate}" if a.replicate else "") + ".json")
    path.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
    summary = {k: r[k] for k in ("world", "arm", "episode", "units", "dead", "seconds",
                                 "max_rss_mb", "consistency")}
    summary["checks"] = [{k: c.get(k) for k in ("ref", "found", "units_ref", "differ",
                                                "demand_digest_equal")} for c in r["checks"]]
    summary["across_group_share"] = r["groups"]["choice"]["across_group_share"]
    summary["agent_history_dispersion"] = r["dispersion"]["agent_history_dispersion"]
    summary["routed_share_dispersion"] = r["dispersion"]["routed_share_dispersion"]
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
