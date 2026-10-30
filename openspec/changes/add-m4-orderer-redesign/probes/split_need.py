"""Split-need diagnostic (add-m4-orderer-redesign, task 4.4; root-cause rule).

Question: under whole routing (TS = `trained_sampled_whole`), consumer units lost to `no_fit`
(the whole bid needs more of some recipe type than the chosen producer's auction budget holds),
would a better CHOICE of one producer have saved them, or only a SPLIT across producers?

The probe replays one world-episode x arm through EXP-0057's job path, unchanged
(`decoder_check.select_views` / `world_entry` / `decoder_arm` / `decoder_policies` /
`job_draw_key`, `routing_audit.build_jobs` / `job_config`, `joint_stack_env`, `set_draw_key`,
`RoutingTracer`, `ContractMonitor`, `runner.run_episode`) and observes from outside the package:

- each producer's `produce` is wrapped per instance (re-wrapped after every `reset`). The wrapper
  copies `capacities` and the placed bids before the call and reads the `ProductionResult` and
  the reservations after it; it changes nothing. From them, per producer and cycle:
  - auction budget B_j = capacities - capacity held by reservations carried into the auction
    (the reservations after `produce` minus those created by this cycle's accepted bids, plus
    carried reservations assembled this cycle = completed bids not accepted this cycle). This
    is `produce`'s `auction_budget`, against which `no_fit` is decided;
  - leftover L_j = B_j - capacity of the bids j accepted this cycle (what its auction left);
  - outage flag (an outage producer accepts nothing, so it is never a counterfactual receiver);
  - a replay of the price-greedy auction over B_j must give the same accepted / rejected bids
    and the same no_fit / displaced split (consistency check; mismatches are counted);
- `System.complete_cycle` is wrapped (outside the RoutingTracer's wrapper). Every rejected
  `no_fit` bid is classified at its auction (below). Each order's last rejected / failed outcome
  is kept exactly as `RoutingTracer._after_auction` keeps `last_reason`; at an unfulfilled order
  resolution whose last reason is `no_fit`, the lost units (`quantity - completed_qty`, the
  tracer's `lost_units_*_no_fit`) go to the class of that last no_fit bid.

Classes of a lost no_fit sub-order (product x, quantity q, chosen producer i, cycle t; E = the
producers declaring x in `produces`, outage producers excluded; R_j = j's own process sheet for
x times q):
- choice-fixable: some j in E, j != i, has L_j >= R_j on every type (the whole sub-order fits in
  what j's auction left, displacing nothing);
- split-needed: none does, but sum_{j in E} L_j >= R on every type (task definition: the
  eligible producers' remaining capacities together, i included, cover the per-type need);
- unfillable: not even together.
Primary counterfactual = L (leftover, no displacement). Sensitivity P (price insertion): j's
budget for the bid = B_j - capacity of j's accepted bids priced >= the bid's price at j (ties go
first); together = sum of those budgets. Also reported: unit-feasible split (sum_j
floor-units j could take in L_j >= q, a split into whole sub-orders that each fit atomically;
<= the per-type test), and a joint sensitivity where a cycle's no_fit bids (consumer and
producer streams, descending price) consume the leftovers they are assigned to, instead of each
seeing the full L.

Usage (from tokenomics/, one world x arm per command, each < 2 min):
  OMP_NUM_THREADS=1 ../.venv/bin/python \
      ../openspec/changes/add-m4-orderer-redesign/probes/split_need.py \
      --world ws-atlas-v1-12-0060 --arm TS [--replicate 0] --workers 1
  ... split_need.py --report     # writes notes/pre_pod_checks.md §split-need (own markers)
`--world 42` = task 1.3's 42-producer rule (`trace_agents.pick_42`).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_agents import pick_42, short  # noqa: E402  (reviewed 1.3 world rule)

REPO = Path(__file__).resolve().parents[4]
CHANGE = Path(__file__).resolve().parents[1]
OUT_DIR = CHANGE / "probes" / "split_need_out"
NOTES = CHANGE / "notes" / "pre_pod_checks.md"
ATLAS = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ATLAS_HASH = "220886a7cf12aaac0cc0a48dc03c57f2dc2484bb2af6185a6160ec509c8a2a23"
EXP0057_EPISODES = REPO / "experiments/EXP-0057-decoder-check/artifacts/episodes.csv"
ORDERER_CKPT = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
ORDERER_SHA = "d25388bf"
HORIZON = 100
WORLDS_12 = ("ws-atlas-v1-12-0060", "ws-atlas-v1-12-0111", "ws-atlas-v1-12-0344")
ARMS = ("TS", "MWWk")
CLASSES = ("choice", "split", "unfillable")
CLASS_NAME = {"choice": "choice-fixable", "split": "split-needed", "unfillable": "unfillable"}
STREAMS = ("consumer", "producer")
RULE_SHARE = 0.05
RULE_WORLDS = 2
BEGIN = "<!-- BEGIN split_need.py (generated; do not edit by hand) -->"
END = "<!-- END split_need.py -->"

Caps = dict[str, int]


def fits(budget: Mapping[str, int], need: Mapping[str, int]) -> bool:
    return all(budget.get(t, 0) >= c for t, c in need.items())


def minus(a: Mapping[str, int], b: Mapping[str, int]) -> Caps:
    out = dict(a)
    for t, c in b.items():
        out[t] = out.get(t, 0) - c
    return out


def add_into(acc: Caps, b: Mapping[str, int]) -> None:
    for t, c in b.items():
        acc[t] = acc.get(t, 0) + c


def max_units(budget: Mapping[str, int], per_unit: Mapping[str, int]) -> int:
    return min((budget.get(t, 0) // c for t, c in per_unit.items() if c > 0), default=0)


class SplitNeed:
    """Read-only observer (module docstring)."""

    def __init__(self) -> None:
        self.snap: dict[str, dict] = {}
        self.last: dict[object, tuple[str, dict | None]] = {}
        self.lost: dict[str, dict[str, dict[str, int]]] = {
            v: {s: defaultdict(int) for s in STREAMS} for v in ("L", "P", "units", "joint")}
        self.lost_orders: dict[str, dict[str, dict[str, int]]] = {
            v: {s: defaultdict(int) for s in STREAMS} for v in ("L", "P", "units", "joint")}
        self.lost_total = {s: 0 for s in STREAMS}
        self.qty_mismatch = 0  # lost units != the last no_fit bid's quantity
        self.replay_mismatch = 0
        self.dup_bids = 0
        self.nonuniform_sheets = 0  # classified bids whose eligible producers' sheets differ
        self.classified_rejections = {s: 0 for s in STREAMS}
        self.outage_candidates = 0
        self.examples: list[dict] = []

    def attach(self, env: object) -> None:
        original_reset = env.reset  # type: ignore[attr-defined]

        def reset(*args: object, **kwargs: object) -> object:
            out = original_reset(*args, **kwargs)
            self._wrap(env.system)  # type: ignore[attr-defined]
            return out

        env.reset = reset  # type: ignore[attr-defined]

    def _wrap(self, system: object) -> None:
        self.system = system
        self.decl: dict[object, list[str]] = defaultdict(list)
        for pid, p in system.producing_agents.items():  # type: ignore[attr-defined]
            for x in p.produces:
                self.decl[x].append(pid)
            self._wrap_producer(pid, p)
        original = system.complete_cycle  # type: ignore[attr-defined]

        def complete_cycle() -> object:
            self.snap = {}
            result = original()
            self._after(result)
            return result

        system.complete_cycle = complete_cycle  # type: ignore[attr-defined]

    def _wrap_producer(self, pid: str, p: object) -> None:
        from tokenomics.domain.producing_agent import REJECT_NO_FIT

        original = p.produce  # type: ignore[attr-defined]

        def produce(outage: bool = False) -> object:
            caps = dict(p.capacities)  # type: ignore[attr-defined]
            placed = list(p.bids)  # type: ignore[attr-defined]
            res = original(outage=outage)
            new = {id(b) for b in res.accepted}
            budget = dict(caps)
            # held through the auction: carried reservations still waiting, and those
            # assembled this cycle (completed but not accepted now); TTL-failed ones freed theirs
            held = [r.bid for r in p.reservations  # type: ignore[attr-defined]
                    if id(r.bid) not in new]
            held += [b for b in res.completed if id(b) not in new]
            for b in held:
                budget = minus(budget, b.capacities)
            leftover = dict(budget)
            accepted = []
            for b in res.accepted:
                leftover = minus(leftover, b.capacities)
                accepted.append((float(b.capacity_unit_price()), dict(b.capacities)))
            # consistency replay of the auction over the reconstructed budget (a check only)
            avail = dict(budget)
            want_acc, want_rej = [], []
            for b in sorted(placed, key=lambda b: b.capacity_unit_price(), reverse=True):
                if outage or not fits(avail, b.capacities):
                    want_rej.append((id(b), "outage" if outage else
                                     REJECT_NO_FIT if not fits(budget, b.capacities)
                                     else "displaced"))
                else:
                    avail = minus(avail, b.capacities)
                    want_acc.append(id(b))
            got_rej = [(id(b), r) for b, r in zip(res.rejected, res.rejection_reasons)]
            if want_acc != [id(b) for b in res.accepted] or want_rej != got_rej:
                self.replay_mismatch += 1
            rejected: dict[object, object] = {}
            for b, r in zip(res.rejected, res.rejection_reasons):
                if b.order_id in rejected:
                    self.dup_bids += 1
                rejected[b.order_id] = b
            self.snap[pid] = {"outage": bool(outage), "budget": budget, "leftover": leftover,
                              "accepted": accepted, "rejected": rejected,
                              "sheets": p.sheets}  # type: ignore[attr-defined]
            return res

        p.produce = produce  # type: ignore[attr-defined]

    def _need(self, pid: str, bid: object) -> tuple[Caps, Caps]:
        per = dict(self.snap[pid]["sheets"][bid.product].require)  # type: ignore[attr-defined]
        return per, {t: c * bid.quantity for t, c in per.items()}  # type: ignore[attr-defined]

    def classify(self, chosen: str, bid: object) -> dict:
        elig = [j for j in self.decl[bid.product] if not self.snap[j]["outage"]]  # type: ignore
        self.outage_candidates += len(self.decl[bid.product]) - len(elig)  # type: ignore
        needs = {j: self._need(j, bid) for j in elig}
        if len({tuple(sorted(n[0].items())) for n in needs.values()}) > 1:
            self.nonuniform_sheets += 1
        per_i, need_i = needs[chosen]
        out: dict[str, str] = {}
        # L: leftover after each producer's auction
        single_l = any(fits(self.snap[j]["leftover"], needs[j][1]) for j in elig if j != chosen)
        pool_l: Caps = {}
        for j in elig:
            add_into(pool_l, self.snap[j]["leftover"])
        together_l = fits(pool_l, need_i)
        out["L"] = "choice" if single_l else "split" if together_l else "unfillable"
        units = sum(max_units(self.snap[j]["leftover"], needs[j][0]) for j in elig)
        out["units"] = ("choice" if single_l else
                        "split" if units >= bid.quantity else "unfillable")  # type: ignore
        # P: price insertion into j's auction (bids priced >= the bid's price at j go first)
        pool_p: Caps = {}
        single_p = False
        for j in elig:
            need = needs[j][1]
            price = float(bid.tokens) / float(sum(need.values()))  # type: ignore[attr-defined]
            b = dict(self.snap[j]["budget"])
            for pr, caps in self.snap[j]["accepted"]:
                if pr >= price:
                    b = minus(b, caps)
            if j != chosen and fits(b, need):
                single_p = True
            add_into(pool_p, {t: max(0, c) for t, c in b.items()})
        out["P"] = ("choice" if single_p else
                    "split" if fits(pool_p, need_i) else "unfillable")
        return out

    def _joint(self, items: list[tuple[str, object]]) -> dict[int, str]:
        """Cycle-level sensitivity: no_fit bids by descending price claim leftovers in turn."""
        left = {j: dict(s["leftover"]) for j, s in self.snap.items() if not s["outage"]}
        out: dict[int, str] = {}
        for chosen, bid in sorted(items, key=lambda it: -float(it[1].capacity_unit_price())):
            elig = [j for j in self.decl[bid.product] if j in left]  # type: ignore[attr-defined]
            needs = {j: self._need(j, bid)[1] for j in elig}
            hit = next((j for j in elig if j != chosen and fits(left[j], needs[j])), None)
            if hit is not None:
                left[hit] = minus(left[hit], needs[hit])
                out[id(bid)] = "choice"
                continue
            pool: Caps = {}
            for j in elig:
                add_into(pool, {t: max(0, c) for t, c in left[j].items()})
            need = needs[chosen]
            if fits(pool, need):
                out[id(bid)] = "split"
                for t, c in need.items():
                    for j in elig:
                        take = min(c, max(0, left[j].get(t, 0)))
                        if take:
                            left[j][t] -= take
                            c -= take
            else:
                out[id(bid)] = "unfillable"
        return out

    def _after(self, result: object) -> None:
        from tokenomics.domain.producing_agent import REJECT_NO_FIT
        from tokenomics.eval.routing_trace import order_class

        nofit = []
        for o in result.bid_outcomes:  # type: ignore[attr-defined]
            if o.outcome == "rejected" and o.reason == REJECT_NO_FIT:
                nofit.append((o.producer_id, self.snap[o.producer_id]["rejected"][o.order_id]))
        joint = self._joint(nofit)
        for o in result.bid_outcomes:  # type: ignore[attr-defined]
            if o.outcome == "rejected":
                info = None
                if o.reason == REJECT_NO_FIT:
                    bid = self.snap[o.producer_id]["rejected"][o.order_id]
                    info = self.classify(o.producer_id, bid)
                    info["joint"] = joint[id(bid)]
                    info["qty"] = bid.quantity
                    self.classified_rejections[order_class(o.order_kind)] += 1
                self.last[o.order_id] = (str(o.reason), info)
            elif o.outcome == "failed":
                self.last[o.order_id] = (str(o.reason), None)
        for r in result.order_resolutions:  # type: ignore[attr-defined]
            last = self.last.pop(r.order_id, None)
            if r.completed or last is None or last[0] != REJECT_NO_FIT:
                continue
            cls = order_class(r.order_kind)
            lost = r.quantity - r.completed_qty
            info = last[1]
            assert info is not None
            if lost != info["qty"]:
                self.qty_mismatch += 1
            self.lost_total[cls] += lost
            for v in ("L", "P", "units", "joint"):
                self.lost[v][cls][info[v]] += lost
                self.lost_orders[v][cls][info[v]] += 1

    def result(self) -> dict:
        return {"lost_units": {v: {s: dict(d) for s, d in m.items()} for v, m in self.lost.items()},
                "lost_orders": {v: {s: dict(d) for s, d in m.items()}
                                for v, m in self.lost_orders.items()},
                "lost_total": self.lost_total, "qty_mismatch": self.qty_mismatch,
                "replay_mismatch": self.replay_mismatch, "dup_bids": self.dup_bids,
                "nonuniform_sheets": self.nonuniform_sheets,
                "classified_rejections": self.classified_rejections,
                "outage_candidates_excluded": self.outage_candidates}


def reference_rows() -> dict[tuple, dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with EXP0057_EPISODES.open(newline="") as fh:
        return {(r["world"], int(r["episode"]), r["subject"], int(r["replicate"])): r
                for r in csv.DictReader(fh)}


def world_choice() -> dict[str, str]:
    atlas = json.loads(ATLAS.read_text())
    return {w["record"]["world_id"]: w["record"]["levers"]["choice"] for w in atlas["worlds"]}


def consumer_completion(row: Mapping[str, str]) -> float:
    """Share of accepted consumer bid units that complete (bids still reserved at the horizon
    are left out): completed = units + wasted, lost = failed dropped / ttl / ttl_outage."""
    done = float(row["units"]) + float(row["trace_wasted_units_consumer"])
    failed = sum(float(row[f"trace_failed_bid_units_consumer_{k}"])
                 for k in ("dropped", "ttl", "ttl_outage"))
    return done / (done + failed) if done + failed else 0.0


def eligible_42_bounds(ref: Mapping[tuple, Mapping[str, str]]) -> list[tuple]:
    """(world, episode, lost no_fit, filled) of TS r0 at the first confirm episode, for every
    world the task text admits as the 42-producer world (pick_42's pool)."""
    from tokenomics.eval.teacher_audit import first_confirm_seed, world_view

    atlas = json.loads(ATLAS.read_text())
    out = []
    for entry in atlas["worlds"]:
        rec = entry["record"]
        if not (entry["split"] == "train" and int(rec["size"]) == 42
                and rec["levers"]["choice"] == "sparse_f14" and rec["coordinates"]["rho"] > 0.28
                and entry["confirm_class"] != "collapse"):
            continue
        ep = first_confirm_seed(world_view(entry))
        row = ref[(rec["world_id"], ep, "entity_planner/trained_sampled_whole", 0)]
        out.append((rec["world_id"], ep, float(row["trace_lost_units_consumer_no_fit"]),
                    float(row["units"])))
    return sorted(out)


def recorded_row(world: str, episode: int, subject: str, replicate: int) -> dict[str, str] | None:
    csv.field_size_limit(sys.maxsize)
    with EXP0057_EPISODES.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if (r["world"] == world and r["episode"] == str(episode) and r["subject"] == subject
                    and r["replicate"] == str(replicate)):
                return r
    return None


def run(world: str, arm: str, replicate: int) -> dict:
    t0 = time.perf_counter()
    from tokenomics.eval.contract import ContractMonitor
    from tokenomics.eval.decoder_check import (
        decoder_arm, decoder_policies, job_draw_key, select_views, world_entry)
    from tokenomics.eval.gate import row_from_stats, row_to_dict
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.provenance import sha256_file
    from tokenomics.eval.roster import load_roster
    from tokenomics.eval.routing_audit import build_jobs, job_config, trace_cells
    from tokenomics.eval.routing_diagnostics import set_draw_key
    from tokenomics.eval.routing_trace import RoutingTracer
    from tokenomics.eval.runner import run_episode
    from tokenomics.eval.teacher_audit import first_confirm_seed, load_atlas

    atlas = load_atlas(ATLAS)
    if atlas["content_hash"] != ATLAS_HASH:
        raise SystemExit(f"STOP: atlas content hash {atlas['content_hash']}")
    rule = None
    if world == "42":
        world, rule = pick_42(atlas)
    (view,) = select_views(atlas, [world])
    seed = first_confirm_seed(view)
    ckpt_sha = sha256_file(ORDERER_CKPT)
    if not ckpt_sha.startswith(ORDERER_SHA):
        raise SystemExit(f"STOP: orderer checkpoint sha256 {ckpt_sha}")
    table = [decoder_arm(arm, load_roster())]
    jobs = [j for j in build_jobs([world_entry(view)], table, 1, HORIZON, None,
                                  {"orderer": {"path": str(ORDERER_CKPT), "sha256": ckpt_sha}})
            if j["episode"] == seed and j["replicate"] == replicate]
    if len(jobs) != 1:
        raise SystemExit(f"STOP: {arm} has no job for episode {seed} replicate {replicate}")
    job = {**jobs[0], "arm": arm}

    env = joint_stack_env(job_config(job), HORIZON)
    set_draw_key(env, job_draw_key(job))
    tracer = RoutingTracer()
    tracer.attach(env)
    monitor = ContractMonitor(env)
    probe = SplitNeed()
    probe.attach(env)  # after the tracer: its complete_cycle wrapper is the outer one
    prod, order, _fid = decoder_policies(job)
    t1 = time.perf_counter()
    run_episode(env, prod, order, seed=job["episode"], action_seed=job["action_seed"],
                monitor=monitor)
    t2 = time.perf_counter()
    stats = env.episode_stats
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=HORIZON)
    row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                                     job["subject"], job["replicate"], stats))
    row.update(trace_cells(tracer))
    ref = recorded_row(job["world"], job["episode"], job["subject"], replicate)
    check: dict = {"found": ref is not None}
    if ref is not None:
        keys = [k for k in ref if k in row and k != "shock_schedule"
                and (k in ("units", "dead", "demand_digest", "shock_digest", "shock_events",
                           "fulfilled_trace", "fulfilled_units", "ordered_units")
                     or (k.startswith("trace_") and not ref[k].startswith(("[", "{"))))]
        check.update({"compared": len(keys), "differ": [k for k in keys if ref[k] != row[k]],
                      "units_ref": ref["units"],
                      "lost_no_fit_ref": {s: ref[f"trace_lost_units_{s}_no_fit"]
                                          for s in STREAMS}})
    return {
        "world": job["world"], "arm": arm, "subject": job["subject"], "episode": job["episode"],
        "replicate": replicate, "action_seed": job["action_seed"], "size": job["size"],
        "world_rule": rule, "rho": view["rho"], "class": view["class"],
        "units": row["units"], "dead": row["dead"],
        "traced_lost_no_fit": {s: row[f"trace_lost_units_{s}_no_fit"] for s in STREAMS},
        "traced_rejected_no_fit": {s: row[f"trace_rejected_bid_units_{s}_no_fit"]
                                   for s in STREAMS},
        "check": check, "probe": probe.result(),
        "seconds": {"setup": round(t1 - t0, 2), "episode": round(t2 - t1, 2),
                    "total": round(time.perf_counter() - t0, 2)},
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


# --- report ---


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def verify(r: dict) -> list[str]:
    """Problems with one run (empty = the probe is neutral and its classes add up)."""
    p = r["probe"]
    bad = []
    c = r["check"]
    if not c["found"]:
        bad.append("no EXP-0057 row")
    elif c["differ"] or c["units_ref"] != r["units"]:
        bad.append(f"differs from EXP-0057: {c['differ']}")
    for s in STREAMS:
        traced = float(r["traced_lost_no_fit"][s])
        if c.get("found") and float(c["lost_no_fit_ref"][s]) != traced:
            bad.append(f"{s} traced no_fit != EXP-0057")
        if p["lost_total"][s] != traced:
            bad.append(f"{s} probe total {p['lost_total'][s]} != traced {traced}")
        for v in ("L", "P", "units", "joint"):
            if sum(p["lost_units"][v][s].values()) != traced:
                bad.append(f"{s} {v} classes do not sum to traced")
    for k in ("replay_mismatch", "dup_bids", "qty_mismatch"):
        if p[k]:
            bad.append(f"{k} {p[k]}")
    return bad


def report() -> str:
    runs = [json.loads(f.read_text()) for f in sorted(OUT_DIR.glob("*.json"))]
    key = {(r["world"], r["arm"], r["replicate"]): r for r in runs}
    lines = [BEGIN, "",
             "## §split-need. Do whole-routing `no_fit` losses need a split or a better choice? "
             "(task 4.4)", "",
             "Generated by `/srv/research/tokenomics/openspec/changes/add-m4-orderer-redesign/"
             "probes/split_need.py --report` from `probes/split_need_out/*.json`; method, "
             "counterfactual and bias in `notes/4.4.md`. One world-episode per world (the "
             "confirm's first episode seed), horizon 100, `entity_planner` producer, EXP-0057's "
             "job path and draw key; one process, `--workers 1`, BLAS / torch 1 thread. "
             "Lost `no_fit` units = `trace_lost_units_*_no_fit` (an order resolved unfulfilled "
             "whose last bid was rejected `no_fit`), classified at that last bid's auction: "
             "**choice-fixable** = another eligible producer's leftover after its own auction "
             "holds the whole sub-order on every recipe type; **split-needed** = none does, but "
             "the eligible producers' leftovers together (chosen one included) do, per type; "
             "**unfillable** = not even together. Outage producers are never receivers.", "",
             "### Replay and neutrality", "",
             "| world | arm | rep | episode | units (EXP-0057) | trace cells compared / differing "
             "| consumer / producer lost no_fit (EXP-0057) | classes sum | auction replay "
             "mismatches | episode s | total s | max RSS MB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (w, a, rep), r in sorted(key.items()):
        c, p = r["check"], r["probe"]
        lnf = r["traced_lost_no_fit"]
        bad = verify(r)
        lines.append(
            f"| {short(w)} | {a} | {rep} | {r['episode']} | {r['units']} ({c.get('units_ref')}) "
            f"| {c.get('compared')} / {len(c.get('differ') or [])} | {lnf['consumer']} / "
            f"{lnf['producer']} ({c['lost_no_fit_ref']['consumer']} / "
            f"{c['lost_no_fit_ref']['producer']}) | "
            f"{'exact' if not bad else '**' + '; '.join(bad) + '**'} | {p['replay_mismatch']} "
            f"| {r['seconds']['episode']} | {r['seconds']['total']} | {r['max_rss_mb']} |")
    lines += ["", "### Classes (primary counterfactual L: leftover after each producer's "
              "auction)", "",
              "Units / orders per class; share = split-needed units / the arm's filled consumer "
              "units (`units`).", "",
              "| world | arm | rep | stream | lost no_fit units / orders | choice-fixable | "
              "split-needed | unfillable | split-needed / filled |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (w, a, rep), r in sorted(key.items()):
        p = r["probe"]
        for s in STREAMS:
            u, o = p["lost_units"]["L"][s], p["lost_orders"]["L"][s]
            cells = [f"{u.get(k, 0)} / {o.get(k, 0)}" for k in CLASSES]
            share = u.get("split", 0) / float(r["units"])
            lines.append(f"| {short(w)} | {a} | {rep} | {s} | {p['lost_total'][s]} / "
                         f"{sum(o.values())} | {' | '.join(cells)} | "
                         f"{_pct(share) if s == 'consumer' else '–'} |")
    lines += ["", "### Sensitivity (consumer stream; split-needed units and share of filled)", "",
              "L = primary; P = price insertion (the bid displaces j's lower-priced accepted "
              "bids); units = split must be whole sub-orders that each fit (≤ the per-type "
              "test); joint = a cycle's no_fit bids claim leftovers in price order.", "",
              "| world | arm | rep | L choice / split / unfill | P choice / split / unfill | "
              "units choice / split / unfill | joint choice / split / unfill | split share L / P "
              "/ units / joint |", "|---|---|---|---|---|---|---|---|"]
    for (w, a, rep), r in sorted(key.items()):
        p = r["probe"]
        cells, shares = [], []
        for v in ("L", "P", "units", "joint"):
            u = p["lost_units"][v]["consumer"]
            cells.append(" / ".join(str(u.get(k, 0)) for k in CLASSES))
            shares.append(_pct(u.get("split", 0) / float(r["units"])))
        lines.append(f"| {short(w)} | {a} | {rep} | {' | '.join(cells)} | "
                     f"{' / '.join(shares)} |")
    other = []
    for (w, a, rep), r in sorted(key.items()):
        p = r["probe"]
        other.append(f"{short(w)} {a} r{rep}: classified no_fit rejections consumer / producer "
                     f"{p['classified_rejections']['consumer']} / "
                     f"{p['classified_rejections']['producer']}, non-uniform eligible sheets "
                     f"{p['nonuniform_sheets']}, outage candidates excluded "
                     f"{p['outage_candidates_excluded']}")
    lines += ["", "Counts: " + "; ".join(other) + "."]

    # rule
    reading = {w: key.get((w, "TS", 0)) for w in list(WORLDS_12)}
    w42 = [r for (w, a, rep), r in key.items() if a == "TS" and rep == 0 and r["world_rule"]]
    context = [r for (w, a, rep), r in sorted(key.items()) if a == "TS" and rep == 0
               and w not in WORLDS_12 and not r["world_rule"]]

    def share(r: dict, v: str = "L") -> float:
        return r["probe"]["lost_units"][v]["consumer"].get("split", 0) / float(r["units"])

    lines += ["", "### Pre-registered rule (task 4.4)", ""]
    if any(r is None for r in reading.values()) or not w42:
        lines += ["- incomplete: the reading needs TS replicate 0 on the three 12-producer "
                  "worlds and the 42-producer world.", "", END]
        return "\n".join(lines) + "\n"
    above = [w for w, r in reading.items() if share(r) >= RULE_SHARE]
    above42 = share(w42[0]) >= RULE_SHARE
    fire = len(above) >= RULE_WORLDS or above42
    lines.append(
        f"- Reading arm TS, replicate 0, counterfactual L. Split-needed share of TS's filled "
        f"units: " + ", ".join(f"{short(w)} {_pct(share(r))}" for w, r in reading.items())
        + f"; {short(w42[0]['world'])} {_pct(share(w42[0]))}. Worlds ≥ {_pct(RULE_SHARE)}: "
        f"{len(above)} of 3 at 12 producers; 42-producer world "
        f"{'≥' if above42 else '<'} {_pct(RULE_SHARE)}.")
    lines.append(f"- Rule (≥ {_pct(RULE_SHARE)} on ≥ {RULE_WORLDS} of 3 12-producer worlds, or "
                 f"on the 42-producer world): **{'FIRED → split design to user' if fire else 'not fired → stay whole-only'}**.")
    flips = []
    for v in ("P", "units", "joint"):
        ab = [w for w, r in reading.items() if share(r, v) >= RULE_SHARE]
        f = len(ab) >= RULE_WORLDS or share(w42[0], v) >= RULE_SHARE
        flips.append(f"{v}: {'fired' if f else 'not fired'}")
    for rep in (1,):
        rs = {w: key.get((w, "TS", rep)) for w in WORLDS_12}
        r42 = key.get((w42[0]["world"], "TS", rep))
        if all(rs.values()) and r42:
            ab = [w for w, r in rs.items() if share(r) >= RULE_SHARE]
            f = len(ab) >= RULE_WORLDS or share(r42) >= RULE_SHARE
            flips.append(f"TS replicate {rep} (L): {'fired' if f else 'not fired'}")
    lines.append(f"- Same rule under the sensitivities (not the reading): {'; '.join(flips)}.")
    ref = reference_rows()
    choice = world_choice()

    def completion(r: dict) -> float:
        return consumer_completion(ref[(r["world"], r["episode"], r["subject"], r["replicate"])])

    def scaled(r: dict) -> float:
        return share(r, "joint") * completion(r)

    lines += ["", "### Context outside the pre-registered sample (never the rule's reading)", "",
              f"The 42-producer reading world {short(w42[0]['world'])} (task 1.3's `pick_42`: "
              f"lowest id among train, `sparse_f14`, ρ > 0.28, not collapse) loses "
              f"{w42[0]['traced_lost_no_fit']['consumer']} consumer units to `no_fit` under TS, so "
              f"the rule's 42-producer clause is not tested by it. EXP-0057's TS `no_fit` loss (≈ "
              f"252 per episode at 12, ≈ 901 at 42) sits mostly on a few heavy worlds outside the "
              f"sample (12-0111, 4th heaviest at 12, is in it). Those, plus 42-0126 (a rule-eligible "
              f"`sparse_f14` world with a consumer `no_fit` loss), were replayed: TS replicate 0, "
              f"first confirm episode.", "",
              "How to read the columns:",
              "- **joint** (the yardstick): a cycle's `no_fit` bids claim the leftovers in price "
              "order, so concurrent lost orders cannot all count the same spare capacity. It is "
              "conservative for choice-fixable + split-needed together, **not** for split-needed "
              "alone: an order that per-order counting rates choice-fixable can become "
              "split-needed once another order has taken that producer's leftover.",
              "- **per-order (L)**: each lost order sees the full leftovers. On heavy worlds many "
              "lost orders of one cycle and product compete for one pool, so this overstates "
              "what a split could recover.",
              "- **completion**: the share of TS's accepted consumer bid units that complete = "
              "(units + wasted) / (units + wasted + failed dropped / ttl / ttl_outage), from the "
              "run's EXP-0057 row (equal to this replay, 79 / 79 cells). The classes count "
              "acceptance, not fills: an accepted composite bid still needs its inputs. "
              "**joint × completion** is a crude scaling of the joint share by "
              "it, not a counterfactual.",
              "- Shares above 100% come from a small denominator (the episode's filled units), "
              "not from double counting: in every run the classes sum exactly to the traced "
              "`no_fit` loss.", "",
              "| world | choice | rule-eligible 42 world | filled | lost no_fit | joint choice / "
              "split / unfill | split share joint | completion | joint × completion | per-order "
              "(L) choice / split / unfill | split share L |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in [w42[0]] + context:
        u = r["probe"]["lost_units"]
        cj, cl = u["joint"]["consumer"], u["L"]["consumer"]
        ch = choice[r["world"]]
        elig = ("n/a (12)" if r["size"] == 12 else
                "yes (the reading)" if r["world_rule"] else
                "yes" if ch == "sparse_f14" else "no")
        lines.append(
            f"| {short(r['world'])} | {ch} | {elig} | {r['units']} | "
            f"{r['traced_lost_no_fit']['consumer']} | "
            f"{' / '.join(str(cj.get(k, 0)) for k in CLASSES)} | {_pct(share(r, 'joint'))} | "
            f"{_pct(completion(r))} | {_pct(scaled(r))} | "
            f"{' / '.join(str(cl.get(k, 0)) for k in CLASSES)} | {_pct(share(r))} |")
    f14 = [r for r in [w42[0]] + context if r["size"] == 42 and choice[r["world"]] == "sparse_f14"]
    c12 = [r for r in context if r["size"] == 12]
    replayed = {r["world"] for r in f14}
    bounds, bound_vals = [], []
    for w, ep, lost, filled in eligible_42_bounds(ref):
        if w not in replayed:
            bounds.append(f"{short(w)} {lost:g} / {filled:g} = {_pct(lost / filled)}")
            bound_vals.append(lost / filled)
    lines += ["",
              f"- 42-producer clause by world choice: only `sparse_f14` worlds qualify under the "
              f"task text; the heavy `sparse_f4` / `sparse_f2` worlds never could. Among the "
              f"qualifying worlds the clause depends on the pick: "
              + ", ".join(f"{short(r['world'])} joint {_pct(share(r, 'joint'))} (per-order "
                          f"{_pct(share(r))}, × completion {_pct(scaled(r))})" for r in f14)
              + ". The other rule-eligible worlds (train, 42 producers, `sparse_f14`, ρ > 0.28, "
              f"not collapse) were not replayed; their TS replicate-0 first-episode consumer "
              f"`no_fit` loss / filled units bounds any split-needed share: "
              + ", ".join(bounds) + f" ({'all' if all(b < RULE_SHARE for b in bound_vals) else '**not all**'} "
              f"< {_pct(RULE_SHARE)}).",
              f"- 12-producer context worlds ≥ {_pct(RULE_SHARE)} under joint: "
              f"{sum(share(r, 'joint') >= RULE_SHARE for r in c12)} of {len(c12)}; under joint × "
              f"completion: {sum(scaled(r) >= RULE_SHARE for r in c12)} of {len(c12)}."]
    lines += ["", END]
    return "\n".join(lines) + "\n"


def write_report() -> None:
    text = NOTES.read_text()
    block = report()
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END + "\n", 1)
        text = head + block + tail
    else:
        text = text.rstrip("\n") + "\n\n" + block
    NOTES.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.workers != 1:
        raise SystemExit("--workers 1 only (shared box)")
    if a.report:
        write_report()
        return
    if not a.world or not a.arm:
        raise SystemExit("--world and --arm are required (or --report)")
    res = run(a.world, a.arm, a.replicate)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{res['world']}__{a.arm}__r{a.replicate}.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("world", "arm", "replicate", "units", "traced_lost_no_fit",
                                         "seconds", "max_rss_mb")}))
    print(json.dumps(res["probe"]["lost_units"]["L"]), "problems:", verify(res))


if __name__ == "__main__":
    main()
