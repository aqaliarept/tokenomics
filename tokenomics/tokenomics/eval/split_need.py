"""EXP-0066 driver: `python -m tokenomics.eval.split_need` (add-m4-orderer-redesign task 4.5, the
split-need re-test; task 4.4's probe `probes/split_need.py`, ported).

Question: under whole routing, are the consumer units lost to `no_fit` (the whole bid needs more
of some recipe type than the chosen producer's auction budget holds) lost to a bad CHOICE of one
producer, or would only a SPLIT across producers have saved them? Task 4.4's sample held 13.7%
(12 producers) / 0.1% (42) of TS's `no_fit` loss; this run covers every EXP-0057 world-episode.

Sample: EXP-0057's (design D3): atlas v2 train split, rho > 0.28, not `collapse` (23 worlds at
12 producers, 14 at 42), every confirm episode seed (71 world-episodes). Arms TS
(`trained_sampled_whole`) and MWWk (`mww_keyed`), R = 2 each, EXP-0057's job path and draw keys
(`decoder_check.decoder_arm` / `world_entry` / `decoder_policies` / `job_draw_key`): 284 jobs.

Observer (read-only, outside `domain/`): each producer's `produce` is wrapped per instance and
re-wrapped after every `reset`; it copies capacities and placed bids before the call and reads the
`ProductionResult` and reservations after it, writing nothing back and drawing nothing. Per
producer and cycle: the auction budget B_j (capacities minus capacity held through the auction:
carried reservations still waiting plus those assembled this cycle) and the leftover L_j = B_j
minus j's accepted bids. A replay of the price-greedy auction over B_j must reproduce the accepted
/ rejected bids and reasons (`sn_replay_mismatch`). `System.complete_cycle` is wrapped outside the
RoutingTracer's wrapper; each order's last rejected / failed outcome is kept as the tracer keeps
`last_reason`, and at an unfulfilled resolution whose last reason is `no_fit` the lost units go to
the class of that last `no_fit` bid, classified at its own auction.

Classes (E = producers declaring the product, outage producers excluded):
- choice-fixable: another j in E holds the whole sub-order in what is left;
- split-needed: none does, but E's leftovers together (chosen one included) do, per type;
- unfillable: not even together.
Counts: **joint** (the yardstick, 4.4 review): a cycle's `no_fit` bids (both streams) claim the
leftovers in descending price order, so concurrent lost orders compete for one spare pool;
per-order **L** (each sees the full leftovers), **P** (price insertion) and **units** (whole
sub-orders that each fit) are reported, never deciding.

Pre-registered reading (task 4.5, fixed before the run): deciding arm MWWk, TS reported. Per size,
the pooled completion-corrected split-needed units (joint split-needed units x that run's completed
/ accepted consumer bid units, summed over the size's world-episodes and replicates) / the pooled
filled consumer units (`units`). `split_needed` if that share >= 5% at 12 or at 42 (the supervisor
brings a rule-gated split design to the user before group 6); else `whole_suffices` (step 2 stays
whole-only, design D5 records why).

Validity (exit 2): contract failures, missing rows, digest differences across arms, an observer
row whose classes do not sum to the traced `no_fit` loss or whose auction replay disagrees, and the
neutrality check: every row must equal its EXP-0057 row (same world, episode, subject, replicate)
on units, death, digests, fulfilled trace and every scalar `trace_*` cell. A real run needs the
registered atlas, orderer and EXP-0057 `episodes.csv` SHA-256s, horizon 100, a 40-hex source
commit and no filters; anything else needs `--dry-run`. Resumes on the same `--out-dir` when every
pin matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .contract import (
    DIGEST,
    ContractMonitor,
    ContractRunError,
    append_failure,
    ensure_failures_file,
    job_failure,
    read_failures,
    run_failure,
)
from .decoder_check import (
    EXPECTED_WORLDS,
    HORIZON,
    MWWK,
    SIZES,
    SUBJECT,
    TS,
    decoder_arm,
    decoder_policies,
    job_draw_key,
    longest_first,
    read_reference,
    select_views,
    withheld_reading,
    world_entry,
)
from .gate import DigestMismatch, row_from_stats, row_to_dict
from .gate_cli import (
    CONTRACT_INVALID,
    INVALID,
    Job,
    Key,
    _open_spec,
    _prepare_resume,
    _write_spec,
    check_rows,
    job_grid,
    job_key,
    read_rows,
)
from .provenance import resolve_verified_commit, sha256_file, source_fingerprint
from .roster import DEFAULT_ROSTER, JOINT_STACK, load_roster
from .routing_audit import (
    EPISODE_COLUMNS,
    PRODUCER,
    _is_commit,
    build_jobs,
    execute,
    expected_keys,
    job_config,
    read_episode_dicts,
    trace_cells,
    write_csv,
)
from .routing_diagnostics import set_draw_key
from .routing_trace import FAIL_REASONS, RoutingTracer
from .shock_profiles import SHOCKS_I_SMOKE
from .teacher_audit import REGISTERED_ATLAS_HASH, ROLE_CHECKPOINTS, checked_orderer, load_atlas

DRIVER = "tokenomics.eval.split_need"
EXPERIMENT = "EXP-0066"
ARMS = (TS, MWWK)
DECIDING_ARM = MWWK
REPORTED_ARMS = (TS,)
RULE_SHARE = Fraction(5, 100)  # pre-registered: >= 5% at 12 or at 42 -> split_needed
SPLIT_NEEDED, WHOLE_SUFFICES, INCOMPLETE = "split_needed", "whole_suffices", "incomplete"
NEXT = {SPLIT_NEEDED: "the supervisor brings a rule-gated split design to the user before group 6",
        WHOLE_SUFFICES: "step 2 stays whole-only; design D5 records why",
        INCOMPLETE: "none: incomplete run"}

# registered input (a real run refuses anything else)
EXP0057_EPISODES_SHA = "69656f9e2034bc2f41049591d492a70e692cd9d770b10e60d14291021d9d593c"
NEUTRALITY = "exp0057_neutrality"
CONSISTENCY = "split_need_consistency"

CHOICE, SPLIT, UNFILLABLE = "choice", "split", "unfillable"
CLASSES = (CHOICE, SPLIT, UNFILLABLE)
STREAMS = ("consumer", "producer")
JOINT, PER_ORDER, PRICE, UNITS = "joint", "L", "P", "units"
COUNTS = (JOINT, PER_ORDER, PRICE, UNITS)
CHECK_COUNTERS = ("replay_mismatch", "dup_bids", "qty_mismatch")
INFO_COUNTERS = ("nonuniform_sheets", "outage_candidates_excluded")


def _cell(count: str, stream: str, cls: str, what: str = "units") -> str:
    return f"sn_{count}_{stream}_{cls}_{what}"


SN_COLUMNS = (
    tuple(_cell(c, s, k, w) for c in COUNTS for s in STREAMS for k in CLASSES
          for w in ("units", "orders"))
    + tuple(f"sn_lost_total_{s}" for s in STREAMS)
    + tuple(f"sn_classified_rejections_{s}" for s in STREAMS)
    + tuple(f"sn_{k}" for k in CHECK_COUNTERS + INFO_COUNTERS))
COLUMNS = EPISODE_COLUMNS + ("arm",) + SN_COLUMNS

Caps = dict[str, int]


# --- pure classification ---


def fits(budget: Mapping[str, int], need: Mapping[str, int]) -> bool:
    return all(budget.get(t, 0) >= c for t, c in need.items())


def minus(a: Mapping[str, int], b: Mapping[str, int]) -> Caps:
    out = dict(a)
    for t, c in b.items():
        out[t] = out.get(t, 0) - c
    return out


def _pool(budgets: Sequence[Mapping[str, int]], clip: bool = True) -> Caps:
    """Per-type sum; `clip` counts a negative remainder as 0 (4.4's joint and P pools; its L
    pool summed raw leftovers, kept here so the port reproduces 4.4)."""
    acc: Caps = {}
    for b in budgets:
        for t, c in b.items():
            acc[t] = acc.get(t, 0) + (max(0, c) if clip else c)
    return acc


def max_units(budget: Mapping[str, int], per_unit: Mapping[str, int]) -> int:
    return min((budget.get(t, 0) // c for t, c in per_unit.items() if c > 0), default=0)


@dataclass(frozen=True)
class LostBid:
    """A rejected `no_fit` sub-order at its auction: `need[j]` / `per_unit[j]` = producer j's own
    process sheet for the product x quantity / per unit, for every eligible (declaring,
    not-in-outage) j; `chosen` is in `need`."""

    key: object
    chosen: str
    quantity: int
    price: float  # the bid's capacity-unit price at the chosen producer (joint order)
    tokens: float
    need: Mapping[str, Mapping[str, int]]
    per_unit: Mapping[str, Mapping[str, int]]


def per_order_classes(bid: LostBid, leftover: Mapping[str, Mapping[str, int]],
                      budget: Mapping[str, Mapping[str, int]],
                      accepted: Mapping[str, Sequence[tuple[float, Mapping[str, int]]]]
                      ) -> dict[str, str]:
    """L / units / P classes of one lost bid, each seeing the full leftovers (4.4's counts)."""
    elig = list(bid.need)
    need_i = bid.need[bid.chosen]
    single_l = any(fits(leftover[j], bid.need[j]) for j in elig if j != bid.chosen)
    together_l = fits(_pool([leftover[j] for j in elig], clip=False), need_i)
    out = {PER_ORDER: CHOICE if single_l else SPLIT if together_l else UNFILLABLE}
    units = sum(max_units(leftover[j], bid.per_unit[j]) for j in elig)
    out[UNITS] = CHOICE if single_l else SPLIT if units >= bid.quantity else UNFILLABLE
    single_p, budgets_p = False, []
    for j in elig:
        need = bid.need[j]
        price = bid.tokens / float(sum(need.values()))
        b = dict(budget[j])
        for pr, caps in accepted[j]:
            if pr >= price:  # j's accepted bids priced >= the inserted bid go first (ties too)
                b = minus(b, caps)
        if j != bid.chosen and fits(b, need):
            single_p = True
        budgets_p.append(b)
    out[PRICE] = (CHOICE if single_p else SPLIT if fits(_pool(budgets_p), need_i)
                  else UNFILLABLE)
    return out


def joint_classes(bids: Sequence[LostBid], leftover: Mapping[str, Mapping[str, int]]
                  ) -> dict[object, str]:
    """The yardstick: one cycle's lost bids, by descending price, claim the leftovers in turn.
    Choice goes to the first eligible j != chosen (declaration order) whose remaining leftover
    holds the whole sub-order; split takes per type, greedily, in declaration order."""
    left = {j: dict(b) for j, b in leftover.items()}
    out: dict[object, str] = {}
    for bid in sorted(bids, key=lambda b: -b.price):
        elig = [j for j in bid.need if j in left]
        hit = next((j for j in elig if j != bid.chosen and fits(left[j], bid.need[j])), None)
        if hit is not None:
            left[hit] = minus(left[hit], bid.need[hit])
            out[bid.key] = CHOICE
            continue
        need = bid.need[bid.chosen]
        if fits(_pool([left[j] for j in elig]), need):
            out[bid.key] = SPLIT
            for t, c in need.items():
                for j in elig:
                    take = min(c, max(0, left[j].get(t, 0)))
                    if take:
                        left[j][t] -= take
                        c -= take
        else:
            out[bid.key] = UNFILLABLE
    return out


# --- observer ---


class SplitNeedObserver:
    """Read-only observer of one episode (module docstring); `cells()` = the `sn_*` columns."""

    def __init__(self) -> None:
        self.snap: dict[str, dict] = {}
        self.last: dict[object, tuple[str, dict | None]] = {}
        self.units = {c: {s: defaultdict(int) for s in STREAMS} for c in COUNTS}
        self.orders = {c: {s: defaultdict(int) for s in STREAMS} for c in COUNTS}
        self.lost_total = {s: 0 for s in STREAMS}
        self.classified = {s: 0 for s in STREAMS}
        self.counters = {k: 0 for k in CHECK_COUNTERS + INFO_COUNTERS}

    def attach(self, env: object) -> None:
        """After the RoutingTracer: this `complete_cycle` wrapper is the outer one."""
        original_reset = env.reset  # type: ignore[attr-defined]

        def reset(*args: object, **kwargs: object) -> object:
            out = original_reset(*args, **kwargs)
            self._wrap(env.system)  # type: ignore[attr-defined]
            return out

        env.reset = reset  # type: ignore[attr-defined]

    def _wrap(self, system: object) -> None:
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
        from ..domain.producing_agent import REJECT_NO_FIT

        original = p.produce  # type: ignore[attr-defined]

        def produce(outage: bool = False) -> object:
            caps = dict(p.capacities)  # type: ignore[attr-defined]
            placed = list(p.bids)  # type: ignore[attr-defined]
            res = original(outage=outage)
            new = {id(b) for b in res.accepted}
            budget = dict(caps)
            # held through the auction: carried reservations still waiting, and those assembled
            # this cycle (completed but not accepted now); TTL-failed ones freed theirs
            held = [r.bid for r in p.reservations if id(r.bid) not in new]  # type: ignore
            held += [b for b in res.completed if id(b) not in new]
            for b in held:
                budget = minus(budget, b.capacities)
            leftover = dict(budget)
            accepted = []
            for b in res.accepted:
                leftover = minus(leftover, b.capacities)
                accepted.append((float(b.capacity_unit_price()), dict(b.capacities)))
            avail = dict(budget)  # consistency replay of the auction (a check only)
            want_acc, want_rej = [], []
            for b in sorted(placed, key=lambda b: b.capacity_unit_price(), reverse=True):
                if outage or not fits(avail, b.capacities):
                    want_rej.append((id(b), "outage" if outage else REJECT_NO_FIT
                                     if not fits(budget, b.capacities) else "displaced"))
                else:
                    avail = minus(avail, b.capacities)
                    want_acc.append(id(b))
            got_rej = [(id(b), r) for b, r in zip(res.rejected, res.rejection_reasons)]
            if want_acc != [id(b) for b in res.accepted] or want_rej != got_rej:
                self.counters["replay_mismatch"] += 1
            rejected: dict[object, object] = {}
            for b in res.rejected:
                if b.order_id in rejected:
                    self.counters["dup_bids"] += 1
                rejected[b.order_id] = b
            self.snap[pid] = {"outage": bool(outage), "budget": budget, "leftover": leftover,
                              "accepted": accepted, "rejected": rejected,
                              "sheets": p.sheets}  # type: ignore[attr-defined]
            return res

        p.produce = produce  # type: ignore[attr-defined]

    def _lost_bid(self, chosen: str, bid: object) -> LostBid:
        decl = self.decl[bid.product]  # type: ignore[attr-defined]
        elig = [j for j in decl if not self.snap[j]["outage"]]
        self.counters["outage_candidates_excluded"] += len(decl) - len(elig)
        per = {j: dict(self.snap[j]["sheets"][bid.product].require)  # type: ignore
               for j in elig}
        if len({tuple(sorted(v.items())) for v in per.values()}) > 1:
            self.counters["nonuniform_sheets"] += 1
        q = int(bid.quantity)  # type: ignore[attr-defined]
        return LostBid(key=id(bid), chosen=chosen, quantity=q,
                       price=float(bid.capacity_unit_price()),  # type: ignore[attr-defined]
                       tokens=float(bid.tokens),  # type: ignore[attr-defined]
                       need={j: {t: c * q for t, c in v.items()} for j, v in per.items()},
                       per_unit=per)

    def _after(self, result: object) -> None:
        from ..domain.producing_agent import REJECT_NO_FIT
        from .routing_trace import order_class

        live = {j: s for j, s in self.snap.items() if not s["outage"]}
        leftover = {j: s["leftover"] for j, s in live.items()}
        budget = {j: s["budget"] for j, s in live.items()}
        accepted = {j: s["accepted"] for j, s in live.items()}
        lost: dict[object, LostBid] = {}
        for o in result.bid_outcomes:  # type: ignore[attr-defined]
            if o.outcome == "rejected" and o.reason == REJECT_NO_FIT:
                bid = self.snap[o.producer_id]["rejected"][o.order_id]
                lost[(o.producer_id, o.order_id)] = self._lost_bid(o.producer_id, bid)
        joint = joint_classes(list(lost.values()), leftover)
        for o in result.bid_outcomes:  # type: ignore[attr-defined]
            if o.outcome == "rejected":
                info = None
                if o.reason == REJECT_NO_FIT:
                    lb = lost[(o.producer_id, o.order_id)]
                    info = per_order_classes(lb, leftover, budget, accepted)
                    info[JOINT] = joint[lb.key]
                    info["qty"] = lb.quantity
                    self.classified[order_class(o.order_kind)] += 1
                self.last[o.order_id] = (str(o.reason), info)
            elif o.outcome == "failed":
                self.last[o.order_id] = (str(o.reason), None)
        for r in result.order_resolutions:  # type: ignore[attr-defined]
            last = self.last.pop(r.order_id, None)
            if r.completed or last is None or last[0] != REJECT_NO_FIT:
                continue
            cls = order_class(r.order_kind)
            units = r.quantity - r.completed_qty
            info = last[1]
            assert info is not None
            if units != info["qty"]:
                self.counters["qty_mismatch"] += 1
            self.lost_total[cls] += units
            for c in COUNTS:
                self.units[c][cls][info[c]] += units
                self.orders[c][cls][info[c]] += 1

    def cells(self) -> dict[str, str]:
        out = {}
        for c in COUNTS:
            for s in STREAMS:
                for k in CLASSES:
                    out[_cell(c, s, k, "units")] = repr(float(self.units[c][s].get(k, 0)))
                    out[_cell(c, s, k, "orders")] = repr(float(self.orders[c][s].get(k, 0)))
        for s in STREAMS:
            out[f"sn_lost_total_{s}"] = repr(float(self.lost_total[s]))
            out[f"sn_classified_rejections_{s}"] = repr(float(self.classified[s]))
        out.update({f"sn_{k}": repr(float(v)) for k, v in self.counters.items()})
        return out


# --- worker ---


def run_split_job(job: Job) -> dict[str, str]:
    """One traced, observed episode's row (EXP-0057's job path + the observer), or its failure
    row."""
    try:
        from .joint_stack import joint_stack_env
        from .runner import run_episode

        if job["contract"] != JOINT_STACK:
            raise ValueError(f"{job['subject']}: contract {job['contract']} (D3: joint_stack)")
        env = joint_stack_env(job_config(job), job["horizon"])
        set_draw_key(env, job_draw_key(job))
        tracer = RoutingTracer()
        tracer.attach(env)
        monitor = ContractMonitor(env)
        observer = SplitNeedObserver()
        observer.attach(env)  # after the tracer: its complete_cycle wrapper is the outer one
        prod, order, _fid = decoder_policies(job)
        run_episode(env, prod, order, seed=job["episode"], action_seed=job["action_seed"],
                    monitor=monitor)
        stats = env.episode_stats
        monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=job["horizon"])
        row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"],
                                         job["episode"], job["subject"], job["replicate"],
                                         stats))
        row.update(trace_cells(tracer))
        row["arm"] = job["arm"]
        row.update(observer.cells())
        return row
    except Exception as exc:  # noqa: BLE001 - recorded as the job's failure, never fatal
        return job_failure(job, exc).to_dict()


# --- validity checks (pure) ---

NEUTRAL_CELLS = ("units", "dead", "demand_digest", "shock_digest", "shock_events",
                 "fulfilled_trace", "fulfilled_units", "ordered_units")
NEUTRALITY_COLUMNS = ("world", "episode", "arm", "subject", "replicate", "units",
                      "reference_units", "compared", "differing", "equal")


def _neutral_keys(ref: Mapping[str, str]) -> list[str]:
    """Cells compared with the EXP-0057 row: the gate outcome cells and every scalar `trace_*`
    cell of the reference (series cells are JSON; `shock_schedule` is compared through its
    digest). Taken from the reference alone, so a cell this run's row lacks is a difference,
    never a silently smaller comparison."""
    return [k for k in ref if k in NEUTRAL_CELLS
            or (k.startswith("trace_") and not str(ref[k]).startswith(("[", "{")))]


def neutrality(dicts: Sequence[Mapping[str, str]], ref: Sequence[Mapping[str, str]]
               ) -> tuple[list[dict], list[tuple[str, str]]]:
    """Every row of this run against EXP-0057's row of the same (world, episode, subject,
    replicate): every compared cell must be equal (the observer changes nothing). Returns
    (rows, problems as (check, message))."""
    theirs = {(d["world"], int(d["episode"]), d["subject"], int(d["replicate"])): d for d in ref}
    rows, problems = [], []
    for d in sorted(dicts, key=lambda d: (d["world"], int(d["episode"]), d["subject"],
                                          int(d["replicate"]))):
        k = (d["world"], int(d["episode"]), d["subject"], int(d["replicate"]))
        where = f"{k[0]} episode {k[1]} {k[2]} r{k[3]}"
        b = theirs.get(k)
        if b is None:
            problems.append((NEUTRALITY, f"{where}: no EXP-0057 row"))
            continue
        keys = _neutral_keys(b)
        differ = [c for c in keys if d.get(c) != b[c]]
        rows.append({"world": k[0], "episode": k[1], "arm": d.get("arm"), "subject": k[2],
                     "replicate": k[3], "units": float(d["units"]),
                     "reference_units": float(b["units"]), "compared": len(keys),
                     "differing": ";".join(differ), "equal": not differ})
        if any(c in ("demand_digest", "shock_digest") for c in differ):
            problems.append((DIGEST, f"{where}: digests differ from EXP-0057's"))
        elif differ:
            problems.append((NEUTRALITY, f"{where}: differs from EXP-0057 in {differ[:6]}"
                                         f"{' ...' if len(differ) > 6 else ''}"))
    return rows, problems


def consistency(dicts: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
    """Observer self-checks per row: every count's classes sum to the traced `no_fit` loss in
    both streams, the observer's total equals it, and the auction replay, duplicate-bid and
    lost-quantity counters are 0."""
    problems = []
    for d in dicts:
        where = f"{d['world']} episode {d['episode']} {d['subject']} r{d['replicate']}"
        bad = []
        for s in STREAMS:
            traced = float(d[f"trace_lost_units_{s}_no_fit"])
            if float(d[f"sn_lost_total_{s}"]) != traced:
                bad.append(f"{s} total {d[f'sn_lost_total_{s}']} != traced {traced}")
            for c in COUNTS:
                if sum(float(d[_cell(c, s, k)]) for k in CLASSES) != traced:
                    bad.append(f"{s} {c} classes do not sum to traced {traced}")
        bad += [f"{k} {d[f'sn_{k}']}" for k in CHECK_COUNTERS if float(d[f"sn_{k}"])]
        if bad:
            problems.append((CONSISTENCY, f"{where}: {'; '.join(bad)}"))
    return problems


# --- reading (pure) ---


def _done_failed(d: Mapping[str, str]) -> tuple[Fraction, Fraction]:
    """(completed, failed) accepted consumer bid units: units + wasted, and failed dropped / ttl
    / ttl_outage; bids still reserved at the horizon are in neither (4.4)."""
    done = Fraction(float(d["units"])) + Fraction(float(d["trace_wasted_units_consumer"]))
    failed = sum((Fraction(float(d[f"trace_failed_bid_units_consumer_{k}"]))
                  for k in FAIL_REASONS), Fraction(0))
    return done, failed


def completion(d: Mapping[str, str]) -> Fraction:
    """Completed / accepted consumer bid units of the run (0 when nothing was accepted)."""
    done, failed = _done_failed(d)
    return done / (done + failed) if done + failed else Fraction(0)


def corrected_split(d: Mapping[str, str]) -> Fraction:
    """Completion-corrected split-needed units = joint consumer split-needed units x completion."""
    return Fraction(float(d[_cell(JOINT, "consumer", SPLIT)])) * completion(d)


def _consumer(d: Mapping[str, str], count: str, cls: str) -> Fraction:
    return Fraction(float(d[_cell(count, "consumer", cls)]))


def pooled(dicts: Sequence[Mapping[str, str]]) -> dict:
    """Pooled consumer-stream sums over rows (world-episodes x replicates) and the shares of
    filled consumer units (None when nothing was filled)."""
    filled = sum((Fraction(float(d["units"])) for d in dicts), Fraction(0))
    corr = sum((corrected_split(d) for d in dicts), Fraction(0))
    sums = {c: {k: sum((_consumer(d, c, k) for d in dicts), Fraction(0)) for k in CLASSES}
            for c in COUNTS}
    lost = sum((Fraction(float(d["trace_lost_units_consumer_no_fit"])) for d in dicts),
               Fraction(0))

    def share(x: Fraction) -> Fraction | None:
        return x / filled if filled else None

    return {"rows": len(dicts), "filled": filled, "lost_no_fit": lost, "corrected_split": corr,
            "share_corrected": share(corr), "classes": sums,
            "share_split": {c: share(sums[c][SPLIT]) for c in COUNTS}}


def _f(x: Fraction | None) -> float | None:
    return None if x is None else float(x)


def _pooled_json(p: Mapping) -> dict:
    return {"rows": p["rows"], "filled_units": _f(p["filled"]),
            "lost_no_fit_units": _f(p["lost_no_fit"]),
            "corrected_split_units": _f(p["corrected_split"]),
            "share_corrected": _f(p["share_corrected"]),
            "classes_units": {c: {k: _f(v) for k, v in p["classes"][c].items()} for c in COUNTS},
            "share_split_uncorrected": {c: _f(v) for c, v in p["share_split"].items()}}


def split_reading(dicts: Sequence[Mapping[str, str]], sizes: Sequence[int]) -> dict:
    """The pre-registered reading on the deciding arm (MWWk): per size in `sizes`, pooled
    completion-corrected split-needed units / pooled filled consumer units; `split_needed` when
    that share >= 5% at any size (exactly 5% counts), else `whole_suffices`; `incomplete` when a
    size has no deciding row or nothing filled and no other size fires. TS is reported alongside,
    never deciding; the per-order (L) count and the uncorrected joint share are reported."""
    size_of = {d["world"]: int(d["size"]) for d in dicts}
    per_size: dict[int, dict] = {}
    fired, missing = [], []
    for s in sizes:
        arms = {}
        for arm in ARMS:
            rows = [d for d in dicts if d.get("arm") == arm and size_of[d["world"]] == s]
            arms[arm] = pooled(rows)
        dec = arms[DECIDING_ARM]["share_corrected"]
        if not arms[DECIDING_ARM]["rows"] or dec is None:
            missing.append(s)
        elif dec >= RULE_SHARE:
            fired.append(s)
        per_size[s] = {"deciding": DECIDING_ARM,
                       "at_or_above": dec is not None and dec >= RULE_SHARE,
                       **{arm: _pooled_json(p) for arm, p in arms.items()}}
    reading = SPLIT_NEEDED if fired else INCOMPLETE if missing or not sizes else WHOLE_SUFFICES
    return {"reading": reading, "next": NEXT[reading], "sizes_at_or_above": fired,
            "sizes_incomplete": missing, "threshold": float(RULE_SHARE), "per_size": per_size}


def world_rows(dicts: Sequence[Mapping[str, str]], views: Sequence[Mapping]) -> list[dict]:
    """`split_need_worlds.csv`: per world and arm, pooled over its episodes and replicates
    (reported, never deciding)."""
    out = []
    for v in views:
        for arm in ARMS:
            rows = [d for d in dicts if d["world"] == v["world"] and d.get("arm") == arm]
            if not rows:
                continue
            p = pooled(rows)
            done = sum((_done_failed(d)[0] for d in rows), Fraction(0))
            failed = sum((_done_failed(d)[1] for d in rows), Fraction(0))
            out.append({
                "size": v["size"], "world": v["world"], "choice": v["choice"], "rho": v["rho"],
                "arm": arm, "deciding": arm == DECIDING_ARM, "rows": p["rows"],
                "filled_units": _f(p["filled"]), "lost_no_fit_units": _f(p["lost_no_fit"]),
                **{f"{c}_{k}": _f(p["classes"][c][k]) for c in (JOINT, PER_ORDER) for k in CLASSES},
                "completion_pooled": _f(done / (done + failed)) if done + failed else None,
                "corrected_split_units": _f(p["corrected_split"]),
                "share_corrected": _f(p["share_corrected"]),
                "share_joint": _f(p["share_split"][JOINT]),
                "share_per_order_L": _f(p["share_split"][PER_ORDER]),
                "share_P": _f(p["share_split"][PRICE]),
                "share_units": _f(p["share_split"][UNITS])})
    return out


WORLD_COLUMNS = (
    "size", "world", "choice", "rho", "arm", "deciding", "rows", "filled_units",
    "lost_no_fit_units", *(f"{c}_{k}" for c in (JOINT, PER_ORDER) for k in CLASSES),
    "completion_pooled", "corrected_split_units", "share_corrected", "share_joint",
    "share_per_order_L", "share_P", "share_units")
EPISODE_TABLE_COLUMNS = (
    "size", "world", "episode", "arm", "replicate", "units", "lost_no_fit_units",
    *(f"{c}_{k}" for c in (JOINT, PER_ORDER) for k in CLASSES), "completion",
    "corrected_split_units", "share_corrected", "share_joint", "share_per_order_L")


def episode_rows(dicts: Sequence[Mapping[str, str]]) -> list[dict]:
    """`split_need.csv`: one row per world-episode x arm x replicate (consumer stream)."""
    out = []
    for d in sorted(dicts, key=lambda d: (int(d["size"]), d["world"], int(d["episode"]),
                                          d.get("arm", ""), int(d["replicate"]))):
        u = float(d["units"])
        out.append({
            "size": d["size"], "world": d["world"], "episode": d["episode"], "arm": d.get("arm"),
            "replicate": d["replicate"], "units": u,
            "lost_no_fit_units": float(d["trace_lost_units_consumer_no_fit"]),
            **{f"{c}_{k}": float(_consumer(d, c, k)) for c in (JOINT, PER_ORDER) for k in CLASSES},
            "completion": float(completion(d)), "corrected_split_units": float(corrected_split(d)),
            "share_corrected": float(corrected_split(d)) / u if u else None,
            "share_joint": float(_consumer(d, JOINT, SPLIT)) / u if u else None,
            "share_per_order_L": float(_consumer(d, PER_ORDER, SPLIT)) / u if u else None})
    return out


# --- driver ---

QUESTION = ("Whole routing loses consumer orders whose whole sub-order does not fit the chosen "
            "producer's remaining capacity (`no_fit`). Would a better choice of one producer have "
            "saved them, or only a split across producers? Task 4.4's sample held 13.7% / 0.1% "
            "of TS's `no_fit` loss at 12 / 42 producers; this re-test covers every EXP-0057 "
            "world-episode (task 4.5).")
RULE_TEXT = ("deciding arm MWWk (TS reported): per size, pooled completion-corrected "
             "split-needed units (joint consumer split-needed units x that run's completed / "
             "accepted consumer bid units) over all its world-episodes and replicates / pooled "
             "filled consumer units; split_needed if that share >= 5% at 12 or at 42, else "
             "whole_suffices")
RESULT_SKELETON = """# {title}

{banner}**Status: {status}.** Not an acceptance read. {n_rows} episode rows over {n_worlds}
worlds ({n_episodes} world-episodes), arms {arms}, `entity_planner` producer, horizon {horizon}.
Produced by `python -m {driver}` (see `spec.json`); reading in `reading.json`.

## Question

{question}

## Pre-registered reading (task 4.5, fixed before the run)

{rule}. `split_needed` → {next_split}; `whole_suffices` → {next_whole}. Per-world shares, the
per-order (L) count and the uncorrected joint share are reported, never deciding.

## Result

TODO (plain English first): the reading per size (`reading.json`), the per-world shares
(`split_need_worlds.csv`), the per-order vs joint counts, the neutrality check against EXP-0057
(`neutrality.csv`) and the observer self-checks.

## Conclusions

TODO.
"""
SKELETON_END = "## Conclusions\n\nTODO.\n"
DRY_BANNER = ("> **DRY RUN — not an acceptance read.** Filtered worlds or arms, another horizon "
              "or shock window, unregistered inputs or no source commit; see `spec.json`.\n\n")


def real_run_problems(*, atlas: Mapping, horizon: int, source_commit: str, orderer_sha: str,
                      reference_sha: str | None, filters: Mapping[str, object]) -> list[str]:
    problems = [msg for bad, msg in (
        (atlas.get("content_hash") != REGISTERED_ATLAS_HASH,
         f"atlas {str(atlas.get('content_hash'))[:12]} is not the registered EXP-0052 "
         f"reclassified atlas {REGISTERED_ATLAS_HASH[:12]}"),
        (not _is_commit(source_commit),
         "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
        (horizon != HORIZON, f"horizon {horizon} != {HORIZON}"),
        (orderer_sha != ROLE_CHECKPOINTS["orderer"],
         f"orderer checkpoint {orderer_sha[:12]} is not the registered "
         f"{ROLE_CHECKPOINTS['orderer'][:12]}"),
        (reference_sha != EXP0057_EPISODES_SHA,
         f"--exp0057-episodes {str(reference_sha)[:12]} is not the registered EXP-0057 "
         f"episodes.csv {EXP0057_EPISODES_SHA[:12]}"),
    ) if bad]
    problems += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    return problems


def run_split_need(*, atlas_path: str | Path, out_dir: str | Path, orderer_ckpt: str,
                   exp0057_episodes: str | Path | None = None, workers: int = 1,
                   horizon: int = HORIZON, worlds: Sequence[str] | None = None,
                   max_worlds: int | None = None, arms: Sequence[str] | None = None,
                   shock_profile: str | None = None, source_commit: str = "",
                   dry_run: bool = False,
                   run_job_fn: Callable[[Job], dict[str, str]] = run_split_job,
                   log: Callable[[str], None] = print) -> int:
    """Run (or resume) the split-need re-test into `out_dir`; return the exit status (0, or
    `CONTRACT_INVALID` on contract failures, missing rows, digest differences, an observer
    inconsistency or an EXP-0057 neutrality mismatch)."""
    if shock_profile not in (None, SHOCKS_I_SMOKE):
        raise ValueError(f"--shock-profile takes {SHOCKS_I_SMOKE} only (a dry run's window)")
    chosen_arms = tuple(arms) if arms is not None else ARMS
    if not chosen_arms or set(chosen_arms) - set(ARMS):
        raise ValueError(f"unknown split-need arms {sorted(set(chosen_arms) - set(ARMS))}; "
                         f"known: {ARMS}")
    atlas_path = Path(atlas_path)
    atlas = load_atlas(atlas_path)
    views = select_views(atlas, worlds, max_worlds)
    ref_path = Path(exp0057_episodes) if exp0057_episodes is not None else None
    ref_sha = sha256_file(ref_path) if ref_path is not None else None
    out = Path(out_dir)
    pin = checked_orderer(out, orderer_ckpt, log)
    if not dry_run:
        problems = real_run_problems(
            atlas=atlas, horizon=horizon, source_commit=source_commit,
            orderer_sha=pin["sha256"], reference_sha=ref_sha,
            filters={"worlds": worlds, "max_worlds": max_worlds, "arms": arms,
                     "shock_profile": shock_profile})
        if problems:
            raise ValueError("not a valid split-need run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
        counts = {s: sum(1 for v in views if v["size"] == s) for s in SIZES}
        if counts != EXPECTED_WORLDS:
            raise ValueError(f"the sample has {counts} worlds, D3 registers {EXPECTED_WORLDS}")
    checkpoints = {"orderer": pin}
    roster = load_roster(DEFAULT_ROSTER)
    entries = [world_entry(v, shock_profile) for v in views]
    table = [decoder_arm(a, roster) for a in chosen_arms]
    grid = job_grid(entries, None)
    expected: set[Key] = expected_keys(grid, table, 1)
    source = source_fingerprint(source_commit or None)
    pins = {
        "driver": DRIVER, "horizon": horizon, "dry_run": dry_run,
        "atlas": {"path": str(atlas_path), "sha256": sha256_file(atlas_path),
                  "content_hash": atlas["content_hash"]},
        "checkpoints": {"orderer": pin["sha256"]},
        "reference": ({"exp0057_episodes": {"path": str(ref_path), "sha256": ref_sha}}
                      if ref_path is not None else None),
        "producer": PRODUCER, "arms": list(chosen_arms),
        "subjects": {a: SUBJECT[a] for a in chosen_arms}, "arm_table": table,
        "shock_profile": shock_profile,
        "sample": [{"world": v["world"], "size": v["size"], "class": v["class"],
                    "rho": v["rho"], "choice": v["choice"], "pattern": v["pattern"],
                    "shocks": v["shocks"]} for v in views],
        "worlds": [v["record"] for v in views],
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "grid": grid,
    }
    reference = read_reference(ref_path) if ref_path is not None else None
    if reference is not None:  # fail before any job: every row must have an EXP-0057 row
        have = {(d["world"], int(d["episode"]), d["subject"], int(d["replicate"]))
                for d in reference}
        absent = sorted(expected - have)
        if absent:
            raise ValueError(f"--exp0057-episodes lacks {len(absent)} of the run's "
                             f"{len(expected)} rows, e.g. {absent[:2]}")
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else "audit",
        "acceptance_read": False, "pins": pins, "inputs": checkpoints, "source": source,
        "pre_registered": {
            "task": "add-m4-orderer-redesign 4.5 (inserted 2026-10-30, user decision "
                    "'Fair re-test')",
            "sample": "EXP-0057's: atlas v2 train split, rho > 0.28, not collapse (23 at 12, 14 "
                      "at 42), every confirm episode seed; arms TS and MWWk, R = 2, EXP-0057's "
                      "draw keys",
            "reading": RULE_TEXT, "threshold": float(RULE_SHARE), "deciding_arm": DECIDING_ARM,
            "reported_arms": list(REPORTED_ARMS), "next": NEXT,
            "yardstick": "joint count: a cycle's no_fit bids (both streams) claim the leftovers "
                         "in descending price order",
        },
    })
    ensure_failures_file(out)
    jobs = build_jobs(entries, table, 1, horizon, None, checkpoints)
    by_subject = {a["subject"]: a for a in table}
    for j in jobs:
        j["arm"] = by_subject[j["subject"]]["arm"]
    if {job_key(j["world"], j["episode"], j["subject"], j["replicate"]) for j in jobs} != expected:
        raise AssertionError("job list and pinned grid disagree")  # internal invariant
    episodes = out / "episodes.csv"
    done_keys = {job_key(r.world, r.episode, r.subject, r.replicate) for r in done}
    try:
        execute(out, spec, done_keys, read_episode_dicts(episodes), longest_first(jobs),
                workers, run_job_fn, log, columns=COLUMNS)
    except DigestMismatch as exc:  # recorded as a run row by execute; the run ends invalid
        log(f"[split_need] INVALID: {exc}")
    rows = read_rows(episodes)
    check_rows(rows, grid, expected, complete=False)
    missing = expected - {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
    dicts = read_episode_dicts(episodes)

    recorded = {f["message"] for f in read_failures(out)}
    problems = consistency(dicts)
    neutral: list[dict] = []
    if reference is not None:
        neutral, more = neutrality(dicts, reference)
        problems += more
    for check, msg in problems:
        if msg not in recorded:
            append_failure(out, run_failure(check, ValueError(msg)))
            log(f"[split_need] {check} {msg}")
    write_csv(out / "neutrality.csv", NEUTRALITY_COLUMNS, neutral)

    failures = read_failures(out)
    invalid = bool(failures or missing)
    reading = split_reading(dicts, sorted({int(v["size"]) for v in views}))
    reading = {"experiment": EXPERIMENT, "dry_run": dry_run, "rule": RULE_TEXT,
               "deciding_arm": DECIDING_ARM, "reported_arms": list(REPORTED_ARMS),
               "neutrality_checked": ref_path is not None,
               "neutral_rows": sum(r["equal"] for r in neutral), "compared_rows": len(neutral),
               **reading}
    if invalid:
        reading = {**withheld_reading({"experiment": EXPERIMENT, "dry_run": dry_run,
                                       "rules": RULE_TEXT}, len(failures), len(missing)),
                   "reading": None, "next": None}
    else:
        reading.update({"status": "valid", "readings_withheld": False})
    write_csv(out / "split_need.csv", EPISODE_TABLE_COLUMNS, episode_rows(dicts))
    write_csv(out / "split_need_worlds.csv", WORLD_COLUMNS, world_rows(dicts, views))
    (out / "reading.json").write_text(json.dumps(reading, indent=2, sort_keys=True,
                                                 default=str) + "\n")
    if invalid:
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(failures),
                          "missing_rows": len(missing)}
        status = f"INVALID ({len(failures)} contract failure(s), {len(missing)} missing row(s))"
        code = CONTRACT_INVALID
    else:
        spec["status"] = "complete"
        spec["result"] = {"status": "complete", "rows": len(rows), "contract_failures": 0,
                          "acceptance_read": False, "reading": reading["reading"],
                          "neutral_rows": reading["neutral_rows"],
                          "compared_rows": reading["compared_rows"]}
        status = "complete"
        code = 0
    _write_spec(out, spec)
    result_md = out / "RESULT.md"
    if not result_md.exists() or result_md.read_text().endswith(SKELETON_END):
        result_md.write_text(RESULT_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "", status=status,
            n_rows=len(rows), n_worlds=len(grid),
            n_episodes=sum(len(g["episodes"]) for g in grid.values()),
            arms=", ".join(chosen_arms), horizon=horizon, driver=DRIVER, question=QUESTION,
            rule=RULE_TEXT[0].upper() + RULE_TEXT[1:], next_split=NEXT[SPLIT_NEEDED],
            next_whole=NEXT[WHOLE_SUFFICES]))
    log(f"[split_need] {'DRY RUN ' if dry_run else ''}{status}"
        + (f", reading {reading['reading']}" if code == 0 else ""))
    return code


def _csv_strs(s: str | None) -> list[str] | None:
    return None if s is None else [x for x in s.split(",") if x]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--atlas", required=True, help="atlas_v2.json (EXP-0052 reclassified)")
    p.add_argument("--orderer-ckpt", required=True, help="the trained orderer (EXP-0012)")
    p.add_argument("--exp0057-episodes", help="EXP-0057's episodes.csv (neutrality; needed in a "
                                              "real run)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--source-commit", default="")
    p.add_argument("--verify-source-commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worlds", help="dry run: comma-separated atlas train world ids")
    p.add_argument("--max-worlds", type=int, help="dry run: the first N sampled worlds")
    p.add_argument("--arms", help=f"dry run: arms out of {','.join(ARMS)}")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--shock-profile", choices=(SHOCKS_I_SMOKE,))
    a = p.parse_args(argv)
    commit = resolve_verified_commit() if a.verify_source_commit else a.source_commit
    try:
        return run_split_need(
            atlas_path=a.atlas, out_dir=a.out_dir, orderer_ckpt=a.orderer_ckpt,
            exp0057_episodes=a.exp0057_episodes, workers=a.workers, horizon=a.horizon,
            worlds=_csv_strs(a.worlds), max_worlds=a.max_worlds, arms=_csv_strs(a.arms),
            shock_profile=a.shock_profile, source_commit=commit, dry_run=a.dry_run)
    except (ContractRunError, DigestMismatch):  # recorded in contract_failures.csv
        return CONTRACT_INVALID


if __name__ == "__main__":
    sys.exit(main())
