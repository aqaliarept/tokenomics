"""Routing trace for the random-routing diagnosis (add-m3-development-audit D2, "Telemetry").

Per episode, it aggregates how the ordering agents' routing played out in the auctions:
- rejected and failed bid-units per reason, for consumer orders and for producer orders
  (replenishment + investment). A rejected bid is re-bid next cycle (`Order.sub_rejected`), so
  these per-cycle sums count retries: they measure auction pressure, not loss;
- true loss: the unfulfilled quantity of every order that resolves rejected, attributed to the
  reason of its last rejected or failed bid (`never_bid` if it never got one, e.g. an unfunded
  producer order). A reservation that fails at its TTL while its producer was in an outage in
  this or the previous cycle is `ttl_outage`, not `ttl`: an outage holds reservations without
  assembling them (`ProducingAgent.produce`), which otherwise reads as input starvation;
- split degree (producers per order and cycle) and orders rejected after part of them was
  delivered (completed but wasted units);
- Herfindahl concentration of routed units per product and cycle;
- receiver variation (task 3.2): a *stream* is one ordering agent's orders of one product (under
  joint_stack each ordering agent is one such stream). Each order routed in a cycle is one
  routing decision, received by the set of producers it bid on; decisions of a stream are taken
  in (cycle, order id) order, re-bids of a rejected remainder included. `receiver_change_share`
  is the share of consecutive decision pairs of a stream whose receiving sets differ (0 = the
  stream always sends to the same producers, 1 = it changes every time), consumer and producer
  orders separately;
- herding (task 3.2): per product and cycle with at least two routed orders and at least two
  declaring producers, the Herfindahl of order-weighted producer shares, each order counting 1
  spread over its receivers by quantity (1 = every order sent to one producer; 1/E = orders
  spread evenly over E producers), averaged over such groups (`herding_hhi`, groups in
  `herding_groups`). Unlike `herfindahl_mean` (units) it weights orders, not units, and skips
  single-order groups, where a whole order is trivially concentrated;
- the share of quantity-1 orders sent to their lowest-index eligible producer;
- routes to producers that cannot make one unit, and orders beyond the `k_stream` slot bound;
- each producer's investment-fund share and cut-off price, and two concentrations of the granted
  fund (reference 1/N for N producers): `fund_share_herfindahl_episode`, the Herfindahl of each
  producer's total grants over the episode (is a producer starved of investment? D2 (d) reads
  this one), and `fund_share_herfindahl`, the per-cycle Herfindahl of that cycle's grants averaged
  over cycles with grants (reported; it mostly tracks how many producers are granted per cycle);
- the capacity-ratio trace and the investment rejection rate;
- expired reservations by input stock (task 4.1, EXP-0050's deciding check):
  `ttl_lost_units_{consumer,producer}_{inputs_short,inputs_ok}` split `failed_bid_units_*_ttl`
  (reason `ttl`, not `ttl_outage`) by whether the receiving producer's input stock at routing
  time could assemble the sub-order: `domain.metrics.input_makeable_qty` read just before the
  auction of the bid's latest placement (its routing), against the bid's quantity. A failed
  reservation whose placement the tracer did not see (attached mid-episode) is in neither.

- agent-level measures for the M4 decoder check (add-m4-orderer-redesign design D3 / D9; the
  reviewed definitions of task 1.3's `probes/trace_agents.py`). An *agent* is one ordering
  agent's product stream (aid, product), the joint_stack ordering agent:
  - `lead_concurrent_top_share_*` (EXP-0056's measure, ported from `teacher_audit`'s split-check
    wrapper so that every driver reports it): each order's lead receiver is the producer that
    gets its largest part (ties to the lowest index); per group of orders routed in one cycle
    for one product over one eligible set (>= 2 producers), the share of the group led by its
    most common lead receiver, averaged over groups of >= 2 orders. Orders beyond `k_stream` are
    skipped, as the wrapper skips them;
  - `share_swing_consumer` (D9): per product with >= 2 declaring producers, the total-variation
    distance between consecutive cycles' producer shares of its consumer units, unit-weighted
    (cycle t's units) over products and consecutive cycle pairs;
  - `concurrent_cross_agent_share`: of the groups of >= 2 orders routed in one cycle for one
    product with >= 2 declaring producers, the share holding orders of >= 2 agents (the pair
    version: the share of their order pairs whose two orders come from different agents);
  - `orders_per_turn_*`: orders per active turn (one agent in one cycle holding >= 1 order);
  - `agent_history_dispersion`, `routed_share_dispersion` and its reported-only variants: see
    `history_dispersion`. Built from bid-outcome kinds only, never a reason (ADR-0058).
  `ScoreFidelity` (a policy wrapper, not part of the tracer) accumulates D3's per-decision
  score fidelity.

White-box, like the baselines: the tracer wraps `system.complete_cycle` to snapshot the placed
bids just before the auction, then reads the result's diagnostic `bid_outcomes` and
`order_resolutions` (and, for `ttl_outage`, the system's outage table). It only reads, so
episodes run identically with and without it. `attach(env)` follows every reset;
`attach_system(...)` traces a bare domain `System`.

DIAGNOSTIC ONLY: rejection reasons include `outage`, which reveals hidden shock state. Nothing
here may feed a policy or a teacher fork (design D4).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean, median, pstdev
from typing import TYPE_CHECKING, Callable

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from ..domain.common import OrderId, ProducerId, Product, is_nan
from ..domain.metrics import input_makeable_qty
from ..domain.producing_agent import FAIL_TTL
from ..domain.system import CycleResult, System, order_kind

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv

CONSUMER = "consumer"
PRODUCER = "producer"  # replenishment and investment orders
REJECT_REASONS = ("outage", "no_fit", "displaced")
FAIL_REASONS = ("dropped", "ttl", "ttl_outage")
LOSS_REASONS = REJECT_REASONS + FAIL_REASONS + ("never_bid",)
FAIL_TTL_OUTAGE = "ttl_outage"
INPUTS_SHORT = "inputs_short"  # input stock at routing could not assemble the sub-order
INPUTS_OK = "inputs_ok"
MIN_ROUTED_UNITS = 3  # D3: an agent enters a dispersion cell with >= 3 routed units
ROUTED_KINDS = ("accepted", "rejected")  # every placed bid ends its auction as one of the two
COMPLETED = "completed"

Stream = tuple[str, Product]  # (ordering agent id, product): one agent under joint_stack


def order_class(kind: str) -> str:
    """consumer orders vs everything a producer orders (inputs and investments)."""
    return CONSUMER if kind == CONSUMER else PRODUCER


def herfindahl(units: list[int]) -> float:
    """Sum of squared shares; 1.0 = all units on one producer."""
    total = sum(units)
    return sum((u / total) ** 2 for u in units) if total > 0 else 0.0


@dataclass
class _Counts:
    rejected_bid_units: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed_bid_units: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    lost_units: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_reason: dict[OrderId, str] = field(default_factory=dict)
    outages: dict[int, frozenset[ProducerId]] = field(default_factory=dict)
    cycle: int = 0
    routed_orders: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    split_orders: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    producers_per_order: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    wasted_orders: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    wasted_units: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hhi_sum: float = 0.0
    hhi_n: int = 0
    last_receivers: dict[tuple[str, Product], frozenset[ProducerId]] = field(default_factory=dict)
    receiver_pairs: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    receiver_changes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    herding_sum: float = 0.0
    herding_n: int = 0
    q1_orders: int = 0
    q1_lowest: int = 0
    routes: int = 0
    zero_throughput_routes: int = 0
    overflow_orders: int = 0
    fund_share: dict[ProducerId, float] = field(default_factory=lambda: defaultdict(float))
    fund_hhi_sum: float = 0.0
    fund_hhi_n: int = 0
    fund_granted: dict[ProducerId, int] = field(default_factory=lambda: defaultdict(int))
    cut_off_sum: dict[ProducerId, float] = field(default_factory=lambda: defaultdict(float))
    cut_off_n: dict[ProducerId, int] = field(default_factory=lambda: defaultdict(int))
    cycles: int = 0
    investment_resolved: int = 0
    investment_rejected: int = 0
    # per order and producer: (input-makeable units, bid quantity) at the latest placement
    routed_inputs: dict[OrderId, dict[ProducerId, tuple[float, int]]] = field(default_factory=dict)
    ttl_units: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # agent-level measures (add-m4-orderer-redesign D3 / D9)
    lead_groups: dict[tuple[str, int, Product, tuple[int, ...]], dict[int, int]] = field(
        default_factory=dict)
    swing_last: dict[Product, tuple[int, dict[ProducerId, float]]] = field(default_factory=dict)
    swing_sum: float = 0.0
    swing_units: int = 0
    swing_pairs: int = 0
    cc_groups: int = 0
    cc_across: int = 0
    cc_pairs: int = 0
    cc_cross_pairs: int = 0
    turn_orders: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    active_turns: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    order_stream: dict[OrderId, Stream] = field(default_factory=dict)
    stream_class: dict[Stream, str] = field(default_factory=dict)
    # stream -> producer -> [routed units, completed units]
    history: dict[Stream, dict[ProducerId, list[int]]] = field(
        default_factory=lambda: defaultdict(dict))
    declarers: dict[Product, int] = field(default_factory=dict)


def lead_receiver(units: Mapping[int, int]) -> int:
    """The producer index receiving an order's largest part, ties to the lowest index."""
    return min(units, key=lambda i: (-units[i], i))


def concurrent_top_share(groups: Iterable[Mapping[int, int]]) -> tuple[float, int]:
    """(mean over groups of >= 2 orders of the share led by the group's most common lead
    receiver, number of such groups); 0.0 without one (EXP-0056, D13)."""
    tops = [max(g.values()) / sum(g.values()) for g in groups if sum(g.values()) >= 2]
    return (fmean(tops) if tops else 0.0), len(tops)


def share_tv(a: Mapping[ProducerId, float], b: Mapping[ProducerId, float]) -> float:
    """Total-variation distance (1/2) sum_i |a_i - b_i| of two share vectors (missing = 0)."""
    return 0.5 * sum(abs(a.get(p, 0.0) - b.get(p, 0.0)) for p in set(a) | set(b))


def _shares(units: Mapping[ProducerId, int]) -> dict[ProducerId, float]:
    total = sum(units.values())
    return {p: u / total for p, u in units.items()}


def _weighted_median(pairs: Sequence[tuple[float, float]]) -> float:
    """The first value (ascending) at which the cumulative weight reaches half the total."""
    ordered = sorted(pairs)
    half = sum(w for _, w in ordered) / 2
    acc = 0.0
    for v, w in ordered:
        acc += w
        if acc >= half:
            return v
    return ordered[-1][0]


def history_dispersion(history: Mapping[Stream, Mapping[ProducerId, Sequence[int]]],
                       stream_class: Mapping[Stream, str], declarers: Mapping[Product, int],
                       min_units: int = MIN_ROUTED_UNITS) -> dict[str, float]:
    """Design D3's across-agent dispersion of agent-local history, per episode, from
    `history[stream][producer] = (routed units, completed units)` (bid-outcome kinds only,
    ADR-0058; routed = accepted + rejected, so each re-bid counts again). The definitions are
    task 1.3's reviewed ones (`probes/trace_agents.py`, notes/1.3.md):
    - `agent_history_dispersion`: median, over cells (product x, producer i) where >= 2 agents
      of x routed >= `min_units` units to i, of the population SD of those agents' completion
      rates completed / routed at i (all products);
    - `routed_share_dispersion`: median, over every (product, agent pair) of products with >= 2
      declaring producers and agents with >= `min_units` routed units of it, of the
      total-variation distance between the two agents' routed-share vectors.
    Reported only, never a reading (the 1.3 review's alternatives, carry-forward): the same
    pairs over producer-stream agents only (`_producer_streams`), each agent against the pooled
    units of the product's other qualifying agents (`_vs_rest`, median over agents), and the
    median of the pairs weighted by the product of the two agents' routed units
    (`_unit_weighted`). An empty measure is 0.0; the `*_cells` / `*_pairs` counts tell it apart."""
    by_product: dict[Product, dict[Stream, Mapping[ProducerId, Sequence[int]]]] = defaultdict(dict)
    for stream, cells in history.items():
        by_product[stream[1]][stream] = cells
    sds: list[float] = []
    pairs: list[tuple[float, float]] = []
    producer_pairs: list[float] = []
    vs_rest: list[float] = []
    for x in sorted(by_product, key=str):
        streams = by_product[x]
        pids = sorted({p for cells in streams.values() for p in cells}, key=str)
        for pid in pids:
            rates = [cells[pid][1] / cells[pid][0] for cells in streams.values()
                     if pid in cells and cells[pid][0] >= min_units]
            if len(rates) >= 2:
                sds.append(pstdev(rates))
        if declarers.get(x, 0) < 2:
            continue
        routed = {s: {p: c[0] for p, c in cells.items()} for s, cells in streams.items()}
        kept = [s for s in sorted(routed, key=str) if sum(routed[s].values()) >= min_units]
        vec = {s: _shares(routed[s]) for s in kept}
        for a, b in itertools.combinations(kept, 2):
            tv = share_tv(vec[a], vec[b])
            pairs.append((tv, float(sum(routed[a].values()) * sum(routed[b].values()))))
            if stream_class.get(a) == PRODUCER and stream_class.get(b) == PRODUCER:
                producer_pairs.append(tv)
        if len(kept) >= 2:
            for a in kept:
                rest: dict[ProducerId, int] = defaultdict(int)
                for b in kept:
                    if b != a:
                        for p, u in routed[b].items():
                            rest[p] += u
                vs_rest.append(share_tv(vec[a], _shares(rest)))
    tvs = [tv for tv, _ in pairs]
    return {
        "agent_history_dispersion": median(sds) if sds else 0.0,
        "agent_history_cells": float(len(sds)),
        "routed_share_dispersion": median(tvs) if tvs else 0.0,
        "routed_share_pairs": float(len(tvs)),
        "routed_share_dispersion_producer_streams": (
            median(producer_pairs) if producer_pairs else 0.0),
        "routed_share_pairs_producer_streams": float(len(producer_pairs)),
        "routed_share_dispersion_vs_rest": median(vs_rest) if vs_rest else 0.0,
        "routed_share_dispersion_unit_weighted": _weighted_median(pairs) if pairs else 0.0,
    }


class RoutingTracer:
    """Collects one episode's routing aggregates. `attach(env)` before `run_episode`; read
    `summary()` / `series()` after it. Re-attaching resets the counts."""

    def __init__(self) -> None:
        self._env: TokenomicsAECEnv | None = None
        self._producer_ids: list[ProducerId] = []
        self._k_stream = 0
        self._c = _Counts()

    # --- wiring ---
    def attach(self, env: "TokenomicsAECEnv") -> None:
        self._env = env
        self._c = _Counts()
        original_reset = env.reset

        def reset(*args: object, **kwargs: object) -> object:
            out = original_reset(*args, **kwargs)
            self.attach_system(env.system, env.spec_.producer_ids, env.spec_.k_stream)
            return out

        env.reset = reset  # type: ignore[method-assign]

    def attach_system(self, system: System, producer_ids: Sequence[ProducerId], k_stream: int) -> None:
        """Trace a bare `System`. `producer_ids` gives the decode's producer index order (for
        the lowest-index share); `k_stream` is the orderer slot bound. Resets the counts."""
        self._producer_ids = list(producer_ids)
        self._k_stream = k_stream
        self._c = _Counts()
        self._wrap(system)

    def _wrap(self, system: System) -> None:
        original: Callable[[], CycleResult] = system.complete_cycle

        def complete_cycle() -> CycleResult:
            self._before_auction(system)
            result = original()
            self._after_auction(result)
            return result

        system.complete_cycle = complete_cycle  # type: ignore[method-assign]

    # --- per-cycle hooks ---
    def _before_auction(self, system: System) -> None:
        c = self._c
        c.cycle = system.cycle_counter
        outages = getattr(system, "_outages_by_cycle", {})  # diagnostic read of hidden state
        c.outages[c.cycle] = frozenset(outages.get(c.cycle, ()))
        index = {pid: i for i, pid in enumerate(self._producer_ids)}
        declarers: dict[Product, list[ProducerId]] = defaultdict(list)
        for pid, p in system.producing_agents.items():
            for x in p.produces:
                declarers[x].append(pid)

        # placed bids, grouped per order and per product
        per_order: dict[object, set[ProducerId]] = defaultdict(set)
        order_units: dict[object, dict[ProducerId, int]] = defaultdict(lambda: defaultdict(int))
        per_product: dict[Product, dict[ProducerId, int]] = defaultdict(lambda: defaultdict(int))
        for pid, p in system.producing_agents.items():
            reserved = p.reserved_capacity()
            for bid in p.bids:
                c.routed_inputs.setdefault(bid.order_id, {})[pid] = (
                    input_makeable_qty(p, bid.product), bid.quantity)
                per_order[bid.order_id].add(pid)
                order_units[bid.order_id][pid] += bid.quantity
                per_product[bid.product][pid] += bid.quantity
                c.routes += 1
                unit = {t: cap // bid.quantity for t, cap in bid.capacities.items()}
                if any(p.capacities.get(t, 0) - reserved.get(t, 0) < u for t, u in unit.items()):
                    c.zero_throughput_routes += 1
        for oid, pids in per_order.items():
            order = system.orders.get(oid)
            if order is None:
                continue
            cls = order_class(order_kind(order))
            c.routed_orders[cls] += 1
            c.producers_per_order[cls] += len(pids)
            if len(pids) > 1:
                c.split_orders[cls] += 1
            eligible = declarers.get(order.product, [])
            if order.quantity == 1 and len(eligible) > 1:
                c.q1_orders += 1
                lowest = min(eligible, key=lambda q: index[q])
                if pids == {lowest}:
                    c.q1_lowest += 1
        for x, units in per_product.items():
            if len(declarers.get(x, [])) > 1:
                c.hhi_sum += herfindahl(list(units.values()))
                c.hhi_n += 1

        for oa in system.ordering_agents.values():
            c.overflow_orders += max(0, len(oa.incoming) - self._k_stream)
        self._receivers(system, per_order)
        self._herding(system, order_units, declarers)
        self._agents(system, order_units, declarers, index)

        fund = system.investment_fund
        granted: dict[ProducerId, int] = defaultdict(int)
        for order in system.orders.values():
            pid = order.producer_id()
            if pid is not None and order.funded and order.cycle_counter == 0 and fund > 0:
                c.fund_share[pid] += order.tokens / fund
                granted[pid] += order.tokens
                c.fund_granted[pid] += order.tokens
        if sum(granted.values()) > 0:  # a cycle without grants has no concentration
            c.fund_hhi_sum += herfindahl(list(granted.values()))
            c.fund_hhi_n += 1
        for pid, p in system.producing_agents.items():
            if not is_nan(p.cut_off_price):
                c.cut_off_sum[pid] += p.cut_off_price
                c.cut_off_n[pid] += 1

    def _receivers(self, system: System, per_order: dict[object, set[ProducerId]]) -> None:
        """Receiver variation per stream (module docstring)."""
        c = self._c
        for aid, oa in system.ordering_agents.items():
            for oid in sorted(oa.incoming):
                order = system.orders.get(oid)
                if order is None or oid not in per_order:
                    continue
                stream = (aid, order.product)
                receivers = frozenset(per_order[oid])
                last = c.last_receivers.get(stream)
                if last is not None:
                    cls = order_class(order_kind(order))
                    c.receiver_pairs[cls] += 1
                    c.receiver_changes[cls] += int(receivers != last)
                c.last_receivers[stream] = receivers

    def _herding(self, system: System, order_units: dict[object, dict[ProducerId, int]],
                 declarers: dict[Product, list[ProducerId]]) -> None:
        """Order-weighted concentration per product and cycle (module docstring)."""
        c = self._c
        groups: dict[Product, list[dict[ProducerId, int]]] = defaultdict(list)
        for oid, units in order_units.items():
            order = system.orders.get(oid)
            if order is not None:
                groups[order.product].append(units)
        for x, orders in groups.items():
            if len(orders) < 2 or len(declarers.get(x, [])) < 2:
                continue
            weight: dict[ProducerId, float] = defaultdict(float)
            for units in orders:
                total = sum(units.values())
                for pid, u in units.items():
                    weight[pid] += u / total
            c.herding_sum += sum((w / len(orders)) ** 2 for w in weight.values())
            c.herding_n += 1

    def _agents(self, system: System, order_units: dict[object, dict[ProducerId, int]],
                declarers: dict[Product, list[ProducerId]],
                index: dict[ProducerId, int]) -> None:
        """Agent-level measures of this cycle's routing (module docstring): every order an
        ordering agent holds this cycle (`incoming`, still set before the auction) with its
        placed bids."""
        c = self._c
        c.declarers = {x: len(pids) for x, pids in declarers.items()}
        groups: dict[Product, list[Stream]] = defaultdict(list)
        consumer_units: dict[Product, dict[ProducerId, int]] = defaultdict(lambda: defaultdict(int))
        for aid, oa in system.ordering_agents.items():
            per_class: dict[str, int] = defaultdict(int)
            for slot, oid in enumerate(sorted(oa.incoming)):
                order = system.orders.get(oid)
                if order is None or oa.incoming[oid].quantity <= 0:
                    continue
                cls = order_class(order_kind(order))
                stream = (str(aid), order.product)
                c.order_stream[oid] = stream
                c.stream_class.setdefault(stream, cls)
                per_class[cls] += 1
                groups[order.product].append(stream)
                units = order_units.get(oid)
                if not units:
                    continue
                elig = tuple(sorted(index[p] for p in declarers.get(order.product, ())
                                    if p in index))
                if slot < self._k_stream and len(elig) >= 2:
                    lead = lead_receiver({index[p]: u for p, u in units.items()})
                    group = c.lead_groups.setdefault((cls, c.cycle, order.product, elig), {})
                    group[lead] = group.get(lead, 0) + 1
                if cls == CONSUMER:
                    for pid, u in units.items():
                        consumer_units[order.product][pid] += u
            for cls, n in per_class.items():
                c.turn_orders[cls] += n
                c.active_turns[cls] += 1
        for x, streams in groups.items():
            if len(streams) < 2 or c.declarers.get(x, 0) < 2:
                continue
            counts: dict[Stream, int] = defaultdict(int)
            for s in streams:
                counts[s] += 1
            k = len(streams)
            c.cc_groups += 1
            c.cc_across += int(len(counts) >= 2)
            c.cc_pairs += k * (k - 1) // 2
            c.cc_cross_pairs += k * (k - 1) // 2 - sum(n * (n - 1) // 2 for n in counts.values())
        for x, units in consumer_units.items():
            if c.declarers.get(x, 0) < 2:
                continue
            total = sum(units.values())
            shares = _shares(units)
            last = c.swing_last.get(x)
            if last is not None and last[0] == c.cycle - 1:
                c.swing_sum += total * share_tv(last[1], shares)
                c.swing_units += total
                c.swing_pairs += 1
            c.swing_last[x] = (c.cycle, shares)

    def _history(self, result: CycleResult) -> None:
        """Agent-local history from bid-outcome kinds only (ADR-0058): `reason` is never read
        here."""
        c = self._c
        for b in result.bid_outcomes:
            stream = c.order_stream.get(b.order_id)
            if stream is None:
                continue
            cell = c.history[stream].setdefault(b.producer_id, [0, 0])
            if b.outcome in ROUTED_KINDS:
                cell[0] += b.quantity
            elif b.outcome == COMPLETED:
                cell[1] += b.quantity

    def _after_auction(self, result: CycleResult) -> None:
        c = self._c
        c.cycles += 1
        self._history(result)
        in_outage = c.outages.get(c.cycle, frozenset()) | c.outages.get(c.cycle - 1, frozenset())
        for b in result.bid_outcomes:
            cls = order_class(b.order_kind)
            if b.outcome == "rejected":
                c.rejected_bid_units[f"{cls}:{b.reason}"] += b.quantity
                c.last_reason[b.order_id] = str(b.reason)
            elif b.outcome == "failed":
                reason = FAIL_TTL_OUTAGE if b.reason == FAIL_TTL and b.producer_id in in_outage \
                    else str(b.reason)
                c.failed_bid_units[f"{cls}:{reason}"] += b.quantity
                c.last_reason[b.order_id] = reason
                routed = c.routed_inputs.get(b.order_id, {}).get(b.producer_id)
                if reason == FAIL_TTL and routed is not None:
                    stock, quantity = routed
                    state = INPUTS_SHORT if stock < quantity else INPUTS_OK
                    c.ttl_units[f"{cls}:{state}"] += b.quantity
        for r in result.order_resolutions:
            cls = order_class(r.order_kind)
            last = c.last_reason.pop(r.order_id, "never_bid")
            c.routed_inputs.pop(r.order_id, None)
            if not r.completed:
                c.lost_units[f"{cls}:{last}"] += r.quantity - r.completed_qty
            if not r.completed and r.completed_qty > 0:
                c.wasted_orders[cls] += 1
                c.wasted_units[cls] += r.completed_qty
            if r.order_kind == "investment":
                c.investment_resolved += 1
                c.investment_rejected += 0 if r.completed else 1

    # --- results ---
    def summary(self) -> dict[str, float]:
        """Flat per-episode scalars (CSV-ready)."""
        c = self._c
        out: dict[str, float] = {}
        for cls in (CONSUMER, PRODUCER):
            for reason in REJECT_REASONS:
                out[f"rejected_bid_units_{cls}_{reason}"] = float(
                    c.rejected_bid_units.get(f"{cls}:{reason}", 0))
            for reason in FAIL_REASONS:
                out[f"failed_bid_units_{cls}_{reason}"] = float(
                    c.failed_bid_units.get(f"{cls}:{reason}", 0))
            for reason in LOSS_REASONS:
                out[f"lost_units_{cls}_{reason}"] = float(c.lost_units.get(f"{cls}:{reason}", 0))
            n = c.routed_orders.get(cls, 0)
            out[f"split_share_{cls}"] = c.split_orders.get(cls, 0) / n if n else 0.0
            out[f"split_degree_{cls}"] = c.producers_per_order.get(cls, 0) / n if n else 0.0
            out[f"wasted_orders_{cls}"] = float(c.wasted_orders.get(cls, 0))
            out[f"wasted_units_{cls}"] = float(c.wasted_units.get(cls, 0))
            for state in (INPUTS_SHORT, INPUTS_OK):
                out[f"ttl_lost_units_{cls}_{state}"] = float(c.ttl_units.get(f"{cls}:{state}", 0))
            pairs = c.receiver_pairs.get(cls, 0)
            out[f"receiver_change_share_{cls}"] = (
                c.receiver_changes.get(cls, 0) / pairs if pairs else 0.0)
        out["herfindahl_mean"] = c.hhi_sum / c.hhi_n if c.hhi_n else 0.0
        out["herding_hhi"] = c.herding_sum / c.herding_n if c.herding_n else 0.0
        out["herding_groups"] = float(c.herding_n)
        out["q1_lowest_index_share"] = c.q1_lowest / c.q1_orders if c.q1_orders else 0.0
        out["q1_orders"] = float(c.q1_orders)
        out["zero_throughput_route_share"] = c.zero_throughput_routes / c.routes if c.routes else 0.0
        out["overflow_orders"] = float(c.overflow_orders)
        # Concentrations of what was granted (module docstring), never of shares of the whole
        # fund: only ~3% of it is granted per cycle, so those squares are ~0 for every router
        # (task 2.1 routing_trace.md; fixed in task 2.2).
        out["fund_share_herfindahl"] = c.fund_hhi_sum / c.fund_hhi_n if c.fund_hhi_n else 0.0
        out["fund_share_herfindahl_episode"] = herfindahl(list(c.fund_granted.values()))
        out["investment_rejection_rate"] = (
            c.investment_rejected / c.investment_resolved if c.investment_resolved else 0.0)
        out.update(self.agent_summary())
        return out

    def agent_summary(self) -> dict[str, float]:
        """The agent-level measures (module docstring; add-m4-orderer-redesign D3 / D9)."""
        c = self._c
        out: dict[str, float] = {}
        for cls in (CONSUMER, PRODUCER):
            top, n = concurrent_top_share(g for (k, *_), g in c.lead_groups.items() if k == cls)
            out[f"lead_concurrent_top_share_{cls}"] = top
            out[f"lead_concurrent_groups_{cls}"] = float(n)
            turns = c.active_turns.get(cls, 0)
            out[f"orders_per_turn_{cls}"] = c.turn_orders.get(cls, 0) / turns if turns else 0.0
            out[f"active_turns_{cls}"] = float(turns)
        out["share_swing_consumer"] = c.swing_sum / c.swing_units if c.swing_units else 0.0
        out["share_swing_pairs_consumer"] = float(c.swing_pairs)
        out["concurrent_cross_agent_share"] = c.cc_across / c.cc_groups if c.cc_groups else 0.0
        out["concurrent_cross_agent_pair_share"] = (
            c.cc_cross_pairs / c.cc_pairs if c.cc_pairs else 0.0)
        out["concurrent_groups"] = float(c.cc_groups)
        out.update(history_dispersion(c.history, c.stream_class, c.declarers))
        return out

    def fund_shares(self) -> dict[ProducerId, float]:
        """Each producer's mean per-cycle share of the whole investment fund (granted or not)."""
        c = self._c
        return {pid: s / c.cycles for pid, s in c.fund_share.items()} if c.cycles else {}

    def series(self) -> dict[str, object]:
        """Per-producer means and traces, for JSON columns."""
        c = self._c
        return {
            "fund_share_by_producer": self.fund_shares(),
            "cut_off_price_by_producer": {
                pid: c.cut_off_sum[pid] / c.cut_off_n[pid] for pid in c.cut_off_n},
            "capacity_ratio_trace": list(self._env.capacity_trace) if self._env else [],
        }


# --- score fidelity (add-m4-orderer-redesign design D3), per decision on visited states ---


def _ranks(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=np.float64)
    sorted_v = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation (average ranks); None when either side is constant."""
    ra, rb = _ranks(np.asarray(a, dtype=np.float64)), _ranks(np.asarray(b, dtype=np.float64))
    da, db = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((da * da).sum() * (db * db).sum()))
    return float((da * db).sum()) / den if den > 0 else None


def makeable_logits(makeable: Sequence[int]) -> np.ndarray:
    """The EXP-0012 scorer's distillation target (`train/orderer_scorer.teacher_logits`):
    log `makeable_qty_now` clamped to the action bound, -bound for nothing makeable. Its
    softmax is the makeable share (uniform when nothing is makeable, as `mww_keyed` draws)."""
    from ..env.spaces import ACTION_LOGIT_BOUND as bound

    m = np.asarray(makeable, dtype=np.float64)
    out = np.full(len(m), -bound)
    pos = m > 0
    out[pos] = np.clip(np.log(m[pos]), -bound, bound)
    return out


def _log_softmax(v: np.ndarray) -> np.ndarray:
    z = v - v.max()
    return z - np.log(np.exp(z).sum())


class ScoreFidelity:
    """Order policy wrapper (reads only) for D3's score fidelity on visited states: wrap the
    *scorer* (TR as deployed; TS's base, `TRAINED_WRAPPERS["trained_sampled_whole"]
    (ScoreFidelity(scorer))`). Per decision (a slot order with >= 2 eligible producers), with
    s = the scorer's logits over the eligible producers and m their `makeable_qty_now`:
    - Spearman correlation of s against log m (`makeable_logits`); skipped when s or m is
      constant;
    - top-1 agreement: the first argmax of s is one of m's argmaxes; skipped when m is constant;
    - KL(softmax(s) || softmax(log m)), the makeable share as the scorer's target defines it.
    `cells()` gives the per-episode median Spearman, the agreement share and the median KL, with
    their decision counts; `values` keeps every decision's numbers for pooling across episodes.
    The wrapped policy is called once per turn and its action returned unchanged."""

    def __init__(self, policy: Callable[..., object]) -> None:
        self.policy = policy
        self.values: dict[str, list[float]] = {"spearman": [], "top1": [], "kl": []}

    def __call__(self, env: "TokenomicsAECEnv", agent: str) -> object:
        from .routing_diagnostics import _makeables, _slots

        action = self.policy(env, agent)
        sp = env.spec_
        raw = np.asarray(action, dtype=np.float64).reshape(sp.k_stream, sp.num_producers)
        for k, oid, q, elig in _slots(env, agent):
            if len(elig) >= 2 and q > 0:
                self.record(raw[k, elig], _makeables(env, oid, elig))
        return action

    def record(self, scores: Sequence[float], makeable: Sequence[int]) -> None:
        """One decision's fidelity numbers (class docstring)."""
        s = np.asarray(scores, dtype=np.float64)
        m = np.asarray(makeable, dtype=np.float64)
        rho = spearman(s, m)  # ranks of m are those of log m (zeros lowest)
        if rho is not None:
            self.values["spearman"].append(rho)
        if m.max() > m.min():
            self.values["top1"].append(float(m[int(np.argmax(s))] == m.max()))
        lp, lq = _log_softmax(s), _log_softmax(makeable_logits(makeable))
        self.values["kl"].append(float((np.exp(lp) * (lp - lq)).sum()))

    def cells(self) -> dict[str, str]:
        v = self.values
        return {
            "score_spearman_log_makeable": repr(median(v["spearman"])) if v["spearman"] else "",
            "score_spearman_decisions": str(len(v["spearman"])),
            "score_top1_agreement": repr(fmean(v["top1"])) if v["top1"] else "",
            "score_top1_decisions": str(len(v["top1"])),
            "score_kl_to_makeable": repr(median(v["kl"])) if v["kl"] else "",
            "score_decisions": str(len(v["kl"])),
        }
