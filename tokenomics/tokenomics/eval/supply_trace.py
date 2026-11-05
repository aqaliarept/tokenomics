"""EXP-0074 driver: `python -m tokenomics.eval.supply_trace` (add-m4-orderer-redesign task 7.14,
the producer evaluation's step 2, notes/producer-eval.md).

Question: on the loaded worlds the MWW + `entity_planner` stack delivers about half of the tightest
atlas bound, and every simple orderer sits in the same place. Which supply-side failure loses the
consumer units: capacity, inputs, price or shocks? And for inputs: was no input bought, was it
bought and not delivered (and why), or could nothing upstream make it?

Sample and job path: EXP-0057's (design D3: atlas v2 train split, rho > 0.28, not `collapse`, 23
worlds at 12 producers and 14 at 42, every confirm episode seed, 71 world-episodes), arm MWWk
(`mww_keyed`, R = 2, EXP-0057's draw keys), `entity_planner`, horizon 100: 142 jobs. Every row
must equal its EXP-0057 row (neutrality: the observer changes nothing).

Observer (read-only, outside `domain/`): `System.complete_cycle` is wrapped outside the
RoutingTracer's wrapper. Before the call it snapshots every producer's state as the auction will
see it (capacities, max capacities, capacity free of reservations, input stock, what it can make
now: `domain.metrics.makeable_qty_now`), registers every live order (kind, owner, product,
quantity, investment type, first cycle seen) and the fund each producer's new orders were
granted. After the call it reads the result's `bid_outcomes` and `order_resolutions`. It writes
nothing back and draws nothing.

Each lost consumer unit (an unfulfilled consumer order's `quantity - completed_qty`) gets the
routing tracer's attribution, the reason of the order's last rejected / failed bid (`ttl` becomes
`ttl_outage` when that producer was in an outage this or the previous cycle; `never_bid` without a
bid). Each reason is then refined into one class:
- (a) capacity-short. `no_fit` (the bundle needs more of some type than the auction budget held):
  `a_held` (it fits the producer's current capacity, but reservations held the rest), `a_degraded`
  (exceeds current capacity, fits max: restoration-fixable), `a_max` (exceeds max capacity).
  `dropped` (an accepted reservation lost re-validation against degraded capacity): `a_dropped`.
- (b) input-short: `ttl` (the reservation waited for inputs until its TTL). The missing inputs are
  those the producer's stock at the start of the failing cycle could not cover. Per missing
  input g:
  - (b3) no other producer declaring g could make one unit of it (`makeable_qty_now` >= 1, not
    in an outage) in any cycle from the bid's acceptance to its failure;
  - else (b2) the producer had a replenishment order for g live in that window. Its own cause,
    one level up:
    - `late`: delivered in the failing cycle (after assembly) or later;
    - `short`: delivered before the failing cycle, yet the stock still fell short;
    - `lost_<reason>`: the replenishment order was rejected, by its last bid's reason
      (`unfunded` when it never got a bid: the fund did not back it);
    - `open`: still in progress at the episode's end;
  - else (b1) no input order for g was placed.
  A unit with several missing inputs takes b3 if any input is b3, else b1 if any is b1, else b2
  (the first missing input's cause). The count of such units is reported (`st_multi_missing`).
  (b4) `b4_contention`: no input was short at the start of the cycle; another assembly took it
  first. (b4 is not in the task's list; it is the case the list does not cover.)
- (c) outbid: `displaced` (the bundle fit the auction budget, but higher-priced bids took it):
  `c_displaced`. A `no_fit` bid lacked capacity by definition, so it is never (c).
- (d) outage: `d_outage` (rejected in an outage), `d_ttl_outage` (the reservation expired while its
  producer was out).
- `never_bid`: no bid ever (none expected for consumer orders).
Consistency (exit 2): for every reason, the classes refining it sum to the routing tracer's
`trace_lost_units_consumer_<reason>`, and the observer's check counters are 0.

Per producer and cycle (`producer_cycles.csv`, from the `st_producer_cycles` cell): capacity
ratio (sum current / sum max at the auction), input stock units, inbound units, fund granted to its
new orders, restoration and upgrade orders placed / completed / rejected, and replenishment units
placed / delivered / rejected.

Pre-registered reading (task 7.15): the top-level class with the largest pooled share of lost
consumer units at 12 producers names the producer decision to target in step 3; a runner-up
within 5 points is a tie and goes to the user. 42 producers are reported.

A real run needs the registered atlas, orderer and EXP-0057 `episodes.csv` SHA-256s, horizon 100, a
40-hex source commit and no filters; anything else needs `--dry-run`. Resumes on the same
`--out-dir` when every pin matches. DIAGNOSTIC ONLY: the classes read hidden state (outages, other
producers' stock); nothing here may feed a policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..domain.metrics import makeable_qty_now
from ..domain.producing_agent import FAIL_DROPPED, FAIL_TTL, REJECT_DISPLACED, REJECT_NO_FIT, \
    REJECT_OUTAGE
from ..domain.requests import InvestmentType
from ..domain.system import order_kind
from .contract import (
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
    RR,
    SIZES,
    TTL_SWEEP_ARMS,
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
from .baselines import EXTRA_PRODUCING
from .provenance import resolve_verified_commit, sha256_file, source_fingerprint
from .roster import DEFAULT_ROSTER, JOINT_STACK, load_roster, member_id
from .routing_audit import (
    EPISODE_COLUMNS,
    PRODUCER,
    build_jobs,
    execute,
    expected_keys,
    job_config,
    read_episode_dicts,
    trace_cells,
    write_csv,
)
from .routing_diagnostics import set_draw_key
from .routing_trace import FAIL_TTL_OUTAGE, LOSS_REASONS, RoutingTracer
from .shock_profiles import SHOCKS_I_SMOKE
from .split_need import EXP0057_EPISODES_SHA, NEUTRALITY_COLUMNS, neutrality, real_run_problems
from .teacher_audit import checked_orderer, load_atlas

DRIVER = "tokenomics.eval.supply_trace"
EXPERIMENT = "EXP-0074"
ARMS = (MWWK,)  # EXP-0074 (task 7.15)
SWEEP_ARMS = TTL_SWEEP_ARMS  # task 7.17a: MWWk and RR, EXP-0073's stacks
# registered TTL-2 reference of a sweep (task 7.17a): EXP-0073's ttl2 episodes (MWWk + RR)
EXP0073_TTL2_EPISODES_SHA = "19888e2666bda7556aceb519a77f39ea58ee295aababd916de5e4b5eb2377e02"
CONSISTENCY = "supply_trace_consistency"
NEVER_BID = "never_bid"
HISTORY_CYCLES = 8  # an order lives at most order TTL + 1 <= 5 cycles (MAX_ORDER_TTL 4)

# --- classes ---
A_HELD, A_DEGRADED, A_MAX, A_DROPPED = "a_held", "a_degraded", "a_max", "a_dropped"
B1, B3, B4 = "b1_not_ordered", "b3_upstream_unable", "b4_contention"
B2_SUBS = ("late", "short", "unfunded", "open",
           *(f"lost_{r}" for r in (REJECT_OUTAGE, REJECT_NO_FIT, REJECT_DISPLACED, FAIL_DROPPED,
                                   FAIL_TTL, FAIL_TTL_OUTAGE)))
B2 = tuple(f"b2_{s}" for s in B2_SUBS)
C_DISPLACED, D_OUTAGE, D_TTL_OUTAGE = "c_displaced", "d_outage", "d_ttl_outage"
CLASS_REASON: dict[str, str] = {
    A_HELD: REJECT_NO_FIT, A_DEGRADED: REJECT_NO_FIT, A_MAX: REJECT_NO_FIT, A_DROPPED: FAIL_DROPPED,
    B1: FAIL_TTL, B3: FAIL_TTL, B4: FAIL_TTL, **{b: FAIL_TTL for b in B2},
    C_DISPLACED: REJECT_DISPLACED, D_OUTAGE: REJECT_OUTAGE, D_TTL_OUTAGE: FAIL_TTL_OUTAGE,
    NEVER_BID: NEVER_BID,
}
CLASSES = tuple(CLASS_REASON)
# top-level groups (the reading's unit) and the producer decision each one names
GROUPS: dict[str, tuple[str, ...]] = {
    "a_capacity": (A_HELD, A_DEGRADED, A_MAX, A_DROPPED),
    "b1_not_ordered": (B1,), "b2_not_delivered": B2, "b3_upstream_unable": (B3,),
    "b4_contention": (B4,), "c_outbid": (C_DISPLACED,), "d_outage": (D_OUTAGE, D_TTL_OUTAGE),
    NEVER_BID: (NEVER_BID,),
}
DECISION = {
    "a_capacity": "capacity: restoration / upgrade (a_degraded, a_dropped), what to reserve "
                  "(a_held), order size vs capacity (a_max)",
    "b1_not_ordered": "input buying: which inputs, how many, when",
    "b2_not_delivered": "input buying lead time and the upstream / fund path (its sub-causes)",
    "b3_upstream_unable": "upstream producers' capacity and inputs (structural)",
    "b4_contention": "production choice: which reservation gets the stock",
    "c_outbid": "price / routing (not a producer decision)",
    "d_outage": "shocks (not a decision)", NEVER_BID: "none (never routed)",
}
TIE_POINTS = Fraction(5, 100)
CHECK_COUNTERS = ("history_missing", "unregistered_orders", "qty_mismatch")
INFO_COUNTERS = ("multi_missing",)
PC_FIELDS = ("cycle", "producer", "capacity_ratio", "input_stock_units", "inbound_units",
             "fund_granted", "restoration_placed", "restoration_completed",
             "restoration_rejected", "upgrade_placed", "upgrade_completed", "upgrade_rejected",
             "replenishment_units_placed", "replenishment_units_delivered",
             "replenishment_units_rejected")
PC_CELL = "st_producer_cycles"
# task 7.22: added cells only (every earlier cell stays byte-identical): the late units whose
# missing input's shortfall left after its first command following the acceptance exceeded
# the inventory headroom, and a second per-producer-cycle cell (headroom, inventory expansions).
LATE_HB = "st_b2_late_headroom_blocked"
PCX_FIELDS = ("headroom", "expansion_placed", "expansion_completed", "expansion_rejected")
PCX_CELL = "st_producer_cycles_x"
ST_COLUMNS = (*(f"st_{k}" for k in CLASSES), *(f"st_reason_{r}" for r in LOSS_REASONS),
              "st_lost_consumer_total", *(f"st_{k}" for k in CHECK_COUNTERS + INFO_COUNTERS),
              PC_CELL, LATE_HB, PCX_CELL)
COLUMNS = EPISODE_COLUMNS + ("arm",) + ST_COLUMNS
_INV_KIND = {InvestmentType.RESTORATION: "restoration", InvestmentType.UPGRADE: "upgrade",
             InvestmentType.INVENTORY_EXPANSION: "expansion"}


def _fits(have: Mapping[str, int], need: Mapping[str, int]) -> bool:
    return all(have.get(t, 0) >= c for t, c in need.items())


class SupplyObserver:
    """Read-only observer of one episode (module docstring); `cells()` = the `st_*` columns."""

    def __init__(self) -> None:
        self.units: dict[str, int] = defaultdict(int)
        self.counters = {k: 0 for k in CHECK_COUNTERS + INFO_COUNTERS}
        self.hist: dict[int, dict[str, dict]] = {}
        self.orders: dict[object, dict] = {}
        self.repl: dict[tuple[str, object], list[object]] = defaultdict(list)
        self.last: dict[object, tuple[str, str, int, int]] = {}  # reason, pid, cycle, qty
        self.accepted_at: dict[object, dict[str, int]] = defaultdict(dict)  # oid -> pid -> cycle
        self.pending: dict[object, int] = defaultdict(int)  # replenishment oid -> b2 units
        self.pending_hb: dict[object, int] = defaultdict(int)  # ... of them headroom-blocked
        self.pcx: dict[tuple[int, str], dict[str, float]] = {}
        self.pc: dict[tuple[int, str], dict[str, float]] = {}
        self.decl: dict[object, list[str]] = defaultdict(list)
        self.system: object | None = None

    # --- wiring ---
    def attach(self, env: object) -> None:
        """After the RoutingTracer: this `complete_cycle` wrapper is the outer one."""
        original_reset = env.reset  # type: ignore[attr-defined]

        def reset(*args: object, **kwargs: object) -> object:
            out = original_reset(*args, **kwargs)
            self.attach_system(env.system)  # type: ignore[attr-defined]
            return out

        env.reset = reset  # type: ignore[attr-defined]

    def attach_system(self, system: object) -> None:
        self.__init__()  # one episode per attach (a reset starts afresh)
        self.system = system
        for pid, p in system.producing_agents.items():  # type: ignore[attr-defined]
            for x in p.produces:
                self.decl[x].append(pid)
        original = system.complete_cycle  # type: ignore[attr-defined]

        def complete_cycle() -> object:
            self.before(system)
            result = original()
            self.after(result)
            return result

        system.complete_cycle = complete_cycle  # type: ignore[attr-defined]

    # --- before the auction ---
    def before(self, system: object) -> None:
        c = int(system.cycle_counter)  # type: ignore[attr-defined]
        self.cycle = c
        outages = getattr(system, "_outages_by_cycle", {}).get(c, set())
        state: dict[str, dict] = {}
        for pid, p in system.producing_agents.items():  # type: ignore[attr-defined]
            reserved = p.reserved_capacity()
            free = {t: max(0, v - reserved.get(t, 0)) for t, v in p.capacities.items()}
            state[pid] = {"caps": dict(p.capacities), "max": dict(p.max_capacities),
                          "free": free, "inv": dict(p.inventory), "outage": pid in outages,
                          "make": {x: makeable_qty_now(p, x) for x in p.produces},
                          "headroom": p.headroom(), "inbound": dict(p.outstanding_inbound)}
            self._pcx(c, pid)["headroom"] = float(p.headroom())
            cap, mx = sum(p.capacities.values()), sum(p.max_capacities.values())
            self._pc(c, pid).update({
                "capacity_ratio": cap / mx if mx else 0.0,
                "input_stock_units": float(sum(p.inventory.values())),
                "inbound_units": float(sum(p.outstanding_inbound.values()))})
        self.hist[c] = state
        for old in [k for k in self.hist if k < c - HISTORY_CYCLES]:
            del self.hist[old]
        for oid, order in system.orders.items():  # type: ignore[attr-defined]
            pid = order.producer_id()
            if oid not in self.orders:
                kind = order_kind(order)
                itype = (_INV_KIND[order.investment_request.type] if kind == "investment"
                         else None)
                self.orders[oid] = {"kind": kind, "pid": pid, "product": order.product,
                                    "qty": int(order.quantity), "itype": itype, "created": c,
                                    "resolved": None, "completed": None, "reason": None}
                if kind == "replenishment":
                    self.repl[(pid, order.product)].append(oid)
                    self._pc(c, pid)["replenishment_units_placed"] += order.quantity
                elif itype in ("restoration", "upgrade"):
                    self._pc(c, pid)[f"{itype}_placed"] += 1
                elif itype == "expansion":
                    self._pcx(c, pid)["expansion_placed"] += 1
            if pid is not None and order.funded and order.cycle_counter == 0:
                self._pc(c, pid)["fund_granted"] += float(order.tokens)  # as the tracer reads it

    def _pcx(self, c: int, pid: str) -> dict[str, float]:
        row = self.pcx.get((c, pid))
        if row is None:
            row = self.pcx[(c, pid)] = defaultdict(float)
        return row

    def _pc(self, c: int, pid: str) -> dict[str, float]:
        row = self.pc.get((c, pid))
        if row is None:
            row = self.pc[(c, pid)] = defaultdict(float)
        return row

    # --- after the auction ---
    def after(self, result: object) -> None:
        c = self.cycle
        outages = getattr(self.system, "_outages_by_cycle", {})
        in_outage = set(outages.get(c, ())) | set(outages.get(c - 1, ()))
        for o in result.bid_outcomes:  # type: ignore[attr-defined]
            if o.outcome == "accepted":
                self.accepted_at[o.order_id][o.producer_id] = c
            elif o.outcome == "rejected":
                self.last[o.order_id] = (str(o.reason), o.producer_id, c, int(o.quantity))
            elif o.outcome == "failed":
                reason = FAIL_TTL_OUTAGE if o.reason == FAIL_TTL and o.producer_id in in_outage \
                    else str(o.reason)
                self.last[o.order_id] = (reason, o.producer_id, c, int(o.quantity))
        resolutions = list(result.order_resolutions)  # type: ignore[attr-defined]
        # producer orders first: a consumer loss this cycle may read a replenishment resolved now
        for r in sorted(resolutions, key=lambda r: r.order_kind == "consumer"):
            info = self.orders.get(r.order_id)
            last = self.last.pop(r.order_id, None)
            reason = last[0] if last is not None else NEVER_BID
            if info is None:
                self.counters["unregistered_orders"] += 1
                continue
            info.update(resolved=c, completed=bool(r.completed), reason=reason)
            accepted = self.accepted_at.pop(r.order_id, {})
            if r.order_kind == "replenishment":
                key = "delivered" if r.completed else "rejected"
                self._pc(c, info["pid"])[f"replenishment_units_{key}"] += r.quantity
                units = self.pending.pop(r.order_id, 0)
                flagged = self.pending_hb.pop(r.order_id, 0)
                if units:
                    fate = self._b2_fate(info)
                    self.units[fate] += units
                    if fate == "b2_late":
                        self.units[LATE_HB] += flagged
            elif r.order_kind == "investment" and info["itype"] in ("restoration", "upgrade"):
                key = "completed" if r.completed else "rejected"
                self._pc(c, info["pid"])[f"{info['itype']}_{key}"] += 1
            elif r.order_kind == "investment" and info["itype"] == "expansion":
                key = "completed" if r.completed else "rejected"
                self._pcx(c, info["pid"])[f"expansion_{key}"] += 1
            elif r.order_kind == "consumer" and not r.completed:
                lost = r.quantity - r.completed_qty
                if lost <= 0:
                    self.counters["qty_mismatch"] += 1
                    continue
                self.units[f"reason_{reason}"] += lost
                cls = self.classify(info, last, accepted, lost)
                if cls is not None:  # None = deferred to the replenishment's resolution
                    self.units[cls] += lost
                self.units["lost_consumer_total"] += lost

    @staticmethod
    def _b2_fate(repl: Mapping) -> str:
        if repl["completed"]:
            return "b2_late"
        return "b2_unfunded" if repl["reason"] == NEVER_BID else f"b2_lost_{repl['reason']}"

    def classify(self, info: Mapping, last: tuple[str, str, int, int] | None,
                 accepted: Mapping[str, int], lost: int) -> str | None:
        """The class of `lost` consumer units of the order `info` whose last bid outcome is
        `last` (reason, producer, cycle, bid quantity); None = deferred to an in-flight replenishment."""
        if last is None:
            return NEVER_BID
        reason, pid, at, qty = last
        if reason == REJECT_OUTAGE:
            return D_OUTAGE
        if reason == FAIL_TTL_OUTAGE:
            return D_TTL_OUTAGE
        if reason == REJECT_DISPLACED:
            return C_DISPLACED
        if reason == FAIL_DROPPED:
            return A_DROPPED
        snap = self.hist.get(at, {}).get(pid)
        if snap is None:
            self.counters["history_missing"] += 1
            return NEVER_BID if reason == NEVER_BID else (
                A_MAX if reason == REJECT_NO_FIT else B4)
        sheet = self.system.producing_agents[pid].sheets[info["product"]]  # type: ignore
        if reason == REJECT_NO_FIT:
            need = {t: v * qty for t, v in sheet.require.items()}
            return A_HELD if _fits(snap["caps"], need) else (
                A_DEGRADED if _fits(snap["max"], need) else A_MAX)
        assert reason == FAIL_TTL, reason
        return self._input_class(pid, at, accepted.get(pid, at),
                                 {g: n * qty for g, n in sheet.inputs.items()}, snap, lost)

    def _input_class(self, pid: str, at: int, start: int, need: Mapping[object, int],
                     snap: Mapping, lost: int) -> str | None:
        missing = sorted((g for g, n in need.items() if snap["inv"].get(g, 0) < n), key=str)
        if not missing:
            return B4
        window = list(range(start, at + 1))
        if any(w not in self.hist for w in window):
            self.counters["history_missing"] += 1
        per: list[tuple[str, object]] = []
        for g in missing:
            able = any(not s["outage"] and s["make"].get(g, 0) >= 1
                       for w in window for j, s in self.hist.get(w, {}).items()
                       if j != pid and j in self.decl.get(g, ()))
            if not able:
                per.append((B3, None))
                continue
            live = [o for o in self.repl.get((pid, g), ())
                    if self.orders[o]["created"] <= at
                    and (self.orders[o]["resolved"] is None or self.orders[o]["resolved"] >= start)]
            if not live:
                per.append((B1, None))
                continue
            o = max(live, key=lambda o: (self.orders[o]["created"], str(o)))
            per.append(("b2", o))
        if len(missing) > 1:
            self.counters["multi_missing"] += lost
        kinds = [k for k, _ in per]
        if B3 in kinds:
            return B3
        if B1 in kinds:
            return B1
        repl_oid = per[0][1]
        r = self.orders[repl_oid]
        # task 7.22: headroom-blocked = at its first command after accepting the bid (cycle
        # start + 1; the snapshot follows that cycle's commands), part of the missing input's
        # shortfall was still not ordered and exceeded the headroom left
        g0 = next(g for g, (k, _) in zip(missing, per) if k == "b2")
        s0 = self.hist.get(min(start + 1, at), {}).get(pid)
        blocked = s0 is not None and (
            0 < need[g0] - s0["inv"].get(g0, 0) - s0["inbound"].get(g0, 0) > s0["headroom"])
        if r["resolved"] is None:  # still in flight: its fate decides the sub-cause
            self.pending[repl_oid] += lost
            self.pending_hb[repl_oid] += lost if blocked else 0
            return None
        if r["completed"]:
            late = r["resolved"] >= at
            if late and blocked:
                self.units[LATE_HB] += lost
            return "b2_late" if late else "b2_short"
        return self._b2_fate(r)

    def cells(self) -> dict[str, str]:
        units = dict(self.units)
        units["b2_open"] = units.get("b2_open", 0) + sum(self.pending.values())
        out = {f"st_{k}": repr(float(units.get(k, 0))) for k in CLASSES}
        out.update({f"st_reason_{r}": repr(float(units.get(f"reason_{r}", 0)))
                    for r in LOSS_REASONS})
        out["st_lost_consumer_total"] = repr(float(units.get("lost_consumer_total", 0)))
        out.update({f"st_{k}": repr(float(v)) for k, v in self.counters.items()})
        rows = [[c, pid, *(round(float(self.pc[(c, pid)].get(f, 0.0)), 6) for f in PC_FIELDS[2:])]
                for c, pid in sorted(self.pc, key=lambda k: (k[0], str(k[1])))]
        out[PC_CELL] = json.dumps(rows, separators=(",", ":"))
        out[LATE_HB] = repr(float(self.units.get(LATE_HB, 0)))
        rows_x = [[c, pid, *(round(float(self.pcx.get((c, pid), {}).get(f, 0.0)), 6)
                             for f in PCX_FIELDS)]
                  for c, pid in sorted(self.pc, key=lambda k: (k[0], str(k[1])))]
        out[PCX_CELL] = json.dumps(rows_x, separators=(",", ":"))
        return out


# --- worker ---


def run_supply_job(job: Job) -> dict[str, str]:
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
        observer = SupplyObserver()
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


# --- validity and reading (pure) ---


def consistency(dicts: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
    """Per row: for every reason, the classes refining it sum to the traced consumer loss of that
    reason, the observer's own per-reason sums equal the tracer's, and the check counters are 0."""
    problems = []
    for d in dicts:
        where = f"{d['world']} episode {d['episode']} {d['subject']} r{d['replicate']}"
        bad = []
        for r in LOSS_REASONS:
            traced = float(d[f"trace_lost_units_consumer_{r}"])
            classes = sum(float(d[f"st_{k}"]) for k, kr in CLASS_REASON.items() if kr == r)
            if classes != traced:
                bad.append(f"{r}: classes {classes} != traced {traced}")
            if float(d[f"st_reason_{r}"]) != traced:
                bad.append(f"{r}: observer {d[f'st_reason_{r}']} != traced {traced}")
        total = sum(float(d[f"trace_lost_units_consumer_{r}"]) for r in LOSS_REASONS)
        if float(d["st_lost_consumer_total"]) != total:
            bad.append(f"total {d['st_lost_consumer_total']} != traced {total}")
        bad += [f"{k} {d[f'st_{k}']}" for k in CHECK_COUNTERS if float(d[f"st_{k}"])]
        if bad:
            problems.append((CONSISTENCY, f"{where}: {'; '.join(bad)}"))
    return problems


def pooled(dicts: Sequence[Mapping[str, str]]) -> dict:
    """Pooled lost consumer units per class and group, and each group's share of them."""
    cls = {k: sum((Fraction(float(d[f"st_{k}"])) for d in dicts), Fraction(0)) for k in CLASSES}
    lost = sum(cls.values(), Fraction(0))
    groups = {g: sum((cls[k] for k in ks), Fraction(0)) for g, ks in GROUPS.items()}
    filled = sum((Fraction(float(d["units"])) for d in dicts), Fraction(0))
    return {"rows": len(dicts), "lost": lost, "filled": filled, "classes": cls,
            "groups": groups,
            "shares": {g: (v / lost if lost else None) for g, v in groups.items()}}


def supply_reading(dicts: Sequence[Mapping[str, str]], sizes: Sequence[int]) -> dict:
    """Task 7.15's pre-registered reading: at 12 producers the group with the largest pooled share
    of lost consumer units names the producer decision; a runner-up within 5 points is a tie
    (to the user). Other sizes are reported."""
    size_of = {d["world"]: int(d["size"]) for d in dicts}
    per_size = {}
    for s in sizes:
        p = pooled([d for d in dicts if size_of[d["world"]] == s])
        ranked = sorted(((v, g) for g, v in p["shares"].items() if v is not None), reverse=True)
        top = ranked[0][1] if ranked else None
        tie = [g for v, g in ranked[1:] if ranked and ranked[0][0] - v <= TIE_POINTS]
        per_size[s] = {
            "rows": p["rows"], "lost_units": float(p["lost"]), "filled_units": float(p["filled"]),
            "classes_units": {k: float(v) for k, v in p["classes"].items()},
            "groups_units": {g: float(v) for g, v in p["groups"].items()},
            "shares": {g: (None if v is None else float(v)) for g, v in p["shares"].items()},
            "largest": top, "within_5_points": tie,
            "reading": None if top is None else ("tie" if tie else top)}
    deciding = per_size.get(SIZES[0], {})
    reading = deciding.get("reading")
    return {"reading": reading, "decision": DECISION.get(reading) if reading else None,
            "tie_with": deciding.get("within_5_points"), "deciding_size": SIZES[0],
            "per_size": per_size, "decisions": DECISION}


def producer_cycle_rows(dicts: Sequence[Mapping[str, str]]) -> list[dict]:
    out = []
    for d in dicts:
        extra = {(r[0], r[1]): r[2:] for r in json.loads(d.get(PCX_CELL) or "[]")}
        for rec in json.loads(d[PC_CELL]):
            x = extra.get((rec[0], rec[1]))
            out.append({"world": d["world"], "episode": d["episode"], "arm": d.get("arm"),
                        "replicate": d["replicate"], **dict(zip(PC_FIELDS, rec)),
                        **(dict(zip(PCX_FIELDS, x)) if x is not None else {})})
    return out


def episode_rows(dicts: Sequence[Mapping[str, str]]) -> list[dict]:
    return [{"size": d["size"], "world": d["world"], "episode": d["episode"], "arm": d.get("arm"),
             "replicate": d["replicate"], "units": float(d["units"]),
             "lost": float(d["st_lost_consumer_total"]),
             **{k: float(d[f"st_{k}"]) for k in CLASSES}}
            for d in sorted(dicts, key=lambda d: (int(d["size"]), d["world"], int(d["episode"]),
                                                  d.get("arm", ""), int(d["replicate"])))]


EPISODE_TABLE_COLUMNS = ("size", "world", "episode", "arm", "replicate", "units", "lost",
                         *CLASSES)


def labelled_neutrality(dicts: Sequence[Mapping[str, str]], ref: Sequence[Mapping[str, str]],
                        label: str) -> tuple[list[dict], list[tuple[str, str]]]:
    """`split_need.neutrality` against `ref`, its messages naming `label` (only the reference's
    arms are compared: EXP-0057 holds both MWWk and RR, EXP-0073 ttl2 both)."""
    rows, problems = neutrality(dicts, ref)
    return rows, [(c, m.replace("EXP-0057", label)) for c, m in problems]


# --- driver ---

RULE_TEXT = ("MWWk, entity_planner: per size, the pooled share of lost consumer units per "
             "top-level class (a capacity, b1 not ordered, b2 not delivered, b3 upstream unable, "
             "b4 contention, c outbid, d outage, never_bid); the largest at 12 producers names the "
             "producer decision to target in step 3; a runner-up within 5 points is a tie and "
             "goes to the user; 42 reported")
RESULT_SKELETON = """# {title}

{banner}**Status: {status}.** Not an acceptance read. {n_rows} episode rows over {n_worlds}
worlds, arms {arms}, `entity_planner` producer, horizon {horizon}. Produced by `python -m {driver}`
(see `spec.json`); reading in `reading.json`.

## Pre-registered reading (task 7.15)

{rule}.

## Result

TODO (plain English first).

## Conclusions

TODO.
"""
SKELETON_END = "## Conclusions\n\nTODO.\n"
DRY_BANNER = ("> **DRY RUN — not an acceptance read.** Filtered worlds, another horizon or shock "
              "window, unregistered inputs or no source commit; see `spec.json`.\n\n")


def run_supply_trace(*, atlas_path: str | Path, out_dir: str | Path, orderer_ckpt: str,
                     exp0057_episodes: str | Path | None = None, workers: int = 1,
                     horizon: int = HORIZON, worlds: Sequence[str] | None = None,
                     max_worlds: int | None = None, shock_profile: str | None = None,
                     source_commit: str = "", dry_run: bool = False,
                     order_ttl: int | None = None, arms: Sequence[str] | None = None,
                     exp0073_episodes: str | Path | None = None, producer: str = PRODUCER,
                     run_job_fn: Callable[[Job], dict[str, str]] = run_supply_job,
                     log: Callable[[str], None] = print) -> int:
    """Run (or resume) the supply trace into `out_dir`; return 0, or `CONTRACT_INVALID` on
    contract failures, missing rows, digest differences, an observer inconsistency or an EXP-0057
    neutrality mismatch. `order_ttl` (task 7.17a) selects the TTL-sweep profile: every world at
    that TTL, arms within `SWEEP_ARMS` (default both), the TTL pinned in spec.json and passed to
    every job; at TTL 2 the rows are checked against EXP-0057 and EXP-0073 ttl2 (a real run
    needs both); at TTL 3 / 4 no reference applies (task 7.16 changed those rows) and none is
    taken. The 7.15 reading is reported per arm; EXP-0075 reads across the TTL runs.
    `producer` (task 7.18, sweep only) replaces `entity_planner` in every arm (subjects
    `<producer>/<router>`); references apply to `entity_planner` rows only."""
    allow_large_cells()  # before execute / read_rows re-read episodes.csv (task 7.21)
    sweep = order_ttl is not None
    if producer != PRODUCER:
        if not sweep:
            raise ValueError("--producer belongs to the TTL sweep (--order-ttl)")
        if producer not in EXTRA_PRODUCING:
            raise ValueError(f"--producer takes {PRODUCER} or one of {sorted(EXTRA_PRODUCING)}")
        if exp0057_episodes or exp0073_episodes:
            raise ValueError(f"references are {PRODUCER} rows; {producer} takes none")
    if sweep:
        chosen_arms = tuple(arms) if arms is not None else SWEEP_ARMS
        if not chosen_arms or set(chosen_arms) - set(SWEEP_ARMS):
            raise ValueError(f"--order-ttl (TTL sweep) plays arms within {SWEEP_ARMS}")
        if int(order_ttl) != 2 and (exp0057_episodes or exp0073_episodes):
            raise ValueError("references apply at TTL 2 only (task 7.16 changed TTL 3 / 4 rows)")
    else:
        if arms is not None or exp0073_episodes is not None:
            raise ValueError("--arms / --exp0073-episodes belong to the TTL sweep (--order-ttl)")
        chosen_arms = ARMS
    if shock_profile not in (None, SHOCKS_I_SMOKE):
        raise ValueError(f"--shock-profile takes {SHOCKS_I_SMOKE} only (a dry run's window)")
    atlas_path = Path(atlas_path)
    atlas = load_atlas(atlas_path)
    views = select_views(atlas, worlds, max_worlds)
    ref_path = Path(exp0057_episodes) if exp0057_episodes is not None else None
    ref_sha = sha256_file(ref_path) if ref_path is not None else None
    ref73_path = Path(exp0073_episodes) if exp0073_episodes is not None else None
    ref73_sha = sha256_file(ref73_path) if ref73_path is not None else None
    out = Path(out_dir)
    pin = checked_orderer(out, orderer_ckpt, log)
    if not dry_run:
        refs_apply = not sweep or (int(order_ttl) == 2 and producer == PRODUCER)
        problems = real_run_problems(
            atlas=atlas, horizon=horizon, source_commit=source_commit, orderer_sha=pin["sha256"],
            reference_sha=ref_sha if refs_apply else EXP0057_EPISODES_SHA,
            filters={"worlds": worlds, "max_worlds": max_worlds, "shock_profile": shock_profile,
                     "arms": arms})
        if sweep and refs_apply and ref73_sha != EXP0073_TTL2_EPISODES_SHA:
            problems.append(f"--exp0073-episodes {str(ref73_sha)[:12]} is not the registered "
                            f"EXP-0073 ttl2 episodes.csv {EXP0073_TTL2_EPISODES_SHA[:12]}")
        if problems:
            raise ValueError("not a valid supply-trace run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
        counts = {s: sum(1 for v in views if v["size"] == s) for s in SIZES}
        if counts != EXPECTED_WORLDS:
            raise ValueError(f"the sample has {counts} worlds, D3 registers {EXPECTED_WORLDS}")
    checkpoints = {"orderer": pin}
    roster = load_roster(DEFAULT_ROSTER)
    entries = [world_entry(v, shock_profile) for v in views]
    table = [decoder_arm(a, roster) for a in chosen_arms]
    if producer != PRODUCER:  # task 7.18: the same arms with another producer
        table = [{**a, "producing": producer, "subject": member_id(producer, a["ordering"])}
                 for a in table]
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
        "producer": producer, "arms": list(chosen_arms),
        "subjects": {a["arm"]: a["subject"] for a in table},
        "arm_table": table, "shock_profile": shock_profile,
        "sample": [{"world": v["world"], "size": v["size"], "class": v["class"],
                    "rho": v["rho"], "choice": v["choice"], "pattern": v["pattern"],
                    "shocks": v["shocks"]} for v in views],
        "worlds": [v["record"] for v in views],
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "grid": grid,
    }
    if sweep:  # only under the sweep: EXP-0074's pins stay as they were
        pins.update({"order_ttl": int(order_ttl), "profile": "ttl_sweep",
                     "reference_exp0073": ({"path": str(ref73_path), "sha256": ref73_sha}
                                           if ref73_path is not None else None)})
    references = [(lbl, read_reference(pth)) for lbl, pth in
                  (("EXP-0057", ref_path), ("EXP-0073 ttl2", ref73_path)) if pth is not None]
    for lbl, reference in references:  # fail before any job: every row needs a reference row
        have = {(d["world"], int(d["episode"]), d["subject"], int(d["replicate"]))
                for d in reference}
        absent = sorted(expected - have)
        if absent:
            raise ValueError(f"the {lbl} reference lacks {len(absent)} of the run's "
                             f"{len(expected)} rows, e.g. {absent[:2]}")
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else "audit",
        "acceptance_read": False, "pins": pins, "inputs": checkpoints, "source": source,
        "pre_registered": {
            "task": "add-m4-orderer-redesign 7.14 / 7.15 (inserted 2026-11-05, user decision: "
                    "producer evaluation, notes/producer-eval.md)",
            "sample": "EXP-0057's: atlas v2 train split, rho > 0.28, not collapse (23 at 12, 14 "
                      "at 42), every confirm episode seed; arm MWWk, R = 2, EXP-0057's draw keys",
            "reading": RULE_TEXT, "tie_points": float(TIE_POINTS), "groups": GROUPS,
            "decisions": DECISION,
            **({"sweep": "tasks 7.17a / 7.17 (EXP-0075): EXP-0073's stacks (MWWk R = 2, RR), "
                         "worlds and keys at this order TTL, reservation lifetime = order TTL "
                         "(task 7.16); EXP-0073's pre-registered reading (timing / scarcity / "
                         "mixed against the atlas noise floor, 12 decides, 42 reported) is read "
                         "across the TTL runs; the 7.15 rule is reported per arm"}
               if sweep else {})},
    })
    ensure_failures_file(out)
    jobs = build_jobs(entries, table, 1, horizon, None, checkpoints)
    by_subject = {a["subject"]: a for a in table}
    for j in jobs:
        j["arm"] = by_subject[j["subject"]]["arm"]
        if sweep:
            j["order_ttl"] = int(order_ttl)  # routing_audit.job_config rebuilds the world
    if {job_key(j["world"], j["episode"], j["subject"], j["replicate"]) for j in jobs} != expected:
        raise AssertionError("job list and pinned grid disagree")  # internal invariant
    episodes = out / "episodes.csv"
    done_keys = {job_key(r.world, r.episode, r.subject, r.replicate) for r in done}
    try:
        execute(out, spec, done_keys, read_episode_dicts(episodes), longest_first(jobs),
                workers, run_job_fn, log, columns=COLUMNS)
    except DigestMismatch as exc:  # recorded as a run row by execute; the run ends invalid
        log(f"[supply_trace] INVALID: {exc}")
    rows = read_rows(episodes)
    check_rows(rows, grid, expected, complete=False)
    missing = expected - {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
    dicts = read_episode_dicts(episodes)

    recorded = {f["message"] for f in read_failures(out)}
    problems = consistency(dicts)
    neutral: list[dict] = []
    neutral73: list[dict] = []
    for lbl, reference in references:
        got, more = labelled_neutrality(dicts, reference, lbl)
        if lbl == "EXP-0057":
            neutral = got
        else:
            neutral73 = got
        problems += more
    for check, msg in problems:
        if msg not in recorded:
            append_failure(out, run_failure(check, ValueError(msg)))
            log(f"[supply_trace] {check} {msg}")
    write_csv(out / "neutrality.csv", NEUTRALITY_COLUMNS, neutral)
    if sweep:
        write_csv(out / "neutrality_exp0073.csv", NEUTRALITY_COLUMNS, neutral73)

    failures = read_failures(out)
    invalid = bool(failures or missing)
    sizes = sorted({int(v["size"]) for v in views})
    reading = {"experiment": EXPERIMENT, "dry_run": dry_run, "rule": RULE_TEXT,
               "neutrality_checked": ref_path is not None,
               "neutral_rows": sum(r["equal"] for r in neutral), "compared_rows": len(neutral)}
    if sweep:  # per arm; EXP-0075's pre-registered reading is across the TTL runs
        reading.update({
            "experiment": "EXP-0075", "profile": "ttl_sweep", "order_ttl": int(order_ttl),
            "neutral_rows_exp0073": sum(r["equal"] for r in neutral73),
            "compared_rows_exp0073": len(neutral73),
            "per_arm": {a: supply_reading([d for d in dicts if d.get("arm") == a], sizes)
                        for a in chosen_arms},
            "reading": None, "note": "the 7.15 rule is reported per arm; EXP-0075 reads across "
                                     "the TTL runs"})
    else:
        reading.update(supply_reading(dicts, sizes))
    if invalid:
        reading = {**withheld_reading({"experiment": EXPERIMENT, "dry_run": dry_run,
                                       "rules": RULE_TEXT}, len(failures), len(missing)),
                   "reading": None}
    else:
        reading.update({"status": "valid", "readings_withheld": False})
    write_csv(out / "supply_trace.csv", EPISODE_TABLE_COLUMNS, episode_rows(dicts))
    write_csv(out / "producer_cycles.csv", ("world", "episode", "arm", "replicate", *PC_FIELDS,
                                            *PCX_FIELDS),
              producer_cycle_rows(dicts))
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
            n_rows=len(rows), n_worlds=len(grid), horizon=horizon, driver=DRIVER,
            arms=", ".join(chosen_arms),
            rule=RULE_TEXT[0].upper() + RULE_TEXT[1:]))
    log(f"[supply_trace] {'DRY RUN ' if dry_run else ''}{status}"
        + (f", reading {reading['reading']}" if code == 0 else ""))
    return code


def allow_large_cells() -> None:
    """Raise the csv module's (process-wide) field limit before any episodes.csv is read: the
    `st_producer_cycles` JSON cell of a 42-producer horizon-100 row is ~300k characters, over the
    default 131,072. EXP-0074-0076 only passed because `decoder_check.read_reference` (loading a
    reference) raised the limit as a side effect; EXP-0077's runs take no reference and crashed
    in `gate_cli.read_rows` (task 7.21)."""
    import csv

    csv.field_size_limit(1 << 30)


def _csv_strs(s: str | None) -> list[str] | None:
    return None if s is None else [x for x in s.split(",") if x]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--atlas", required=True, help="atlas_v2.json (EXP-0052 reclassified)")
    p.add_argument("--orderer-ckpt", required=True,
                   help="the trained orderer (EXP-0012; pinned by SHA-256, no arm plays it)")
    p.add_argument("--exp0057-episodes", help="EXP-0057's episodes.csv (neutrality; needed in a "
                                              "real run)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--source-commit", default="")
    p.add_argument("--verify-source-commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worlds", help="dry run: comma-separated atlas train world ids")
    p.add_argument("--max-worlds", type=int, help="dry run: the first N sampled worlds")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--shock-profile", choices=(SHOCKS_I_SMOKE,))
    p.add_argument("--order-ttl", type=int, choices=(1, 2, 3, 4),
                   help="the TTL-sweep profile (task 7.17a): every world at this order TTL, arms "
                        f"within {','.join(SWEEP_ARMS)}")
    p.add_argument("--arms", help=f"sweep dry run: arms out of {','.join(SWEEP_ARMS)}")
    p.add_argument("--producer", default=PRODUCER, choices=(PRODUCER, *EXTRA_PRODUCING),
                   help="sweep: the producer of every arm (task 7.18)")
    p.add_argument("--exp0073-episodes",
                   help="sweep at TTL 2: EXP-0073's ttl2 episodes.csv (neutrality, both arms)")
    a = p.parse_args(argv)
    commit = resolve_verified_commit() if a.verify_source_commit else a.source_commit
    try:
        return run_supply_trace(
            atlas_path=a.atlas, out_dir=a.out_dir, orderer_ckpt=a.orderer_ckpt,
            exp0057_episodes=a.exp0057_episodes, workers=a.workers, horizon=a.horizon,
            worlds=_csv_strs(a.worlds), max_worlds=a.max_worlds, shock_profile=a.shock_profile,
            source_commit=commit, dry_run=a.dry_run, order_ttl=a.order_ttl,
            arms=_csv_strs(a.arms), exp0073_episodes=a.exp0073_episodes, producer=a.producer)
    except (ContractRunError, DigestMismatch):  # recorded in contract_failures.csv
        return CONTRACT_INVALID


if __name__ == "__main__":
    sys.exit(main())
