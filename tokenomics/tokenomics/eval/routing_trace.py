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
- the share of quantity-1 orders sent to their lowest-index eligible producer;
- routes to producers that cannot make one unit, and orders beyond the `k_stream` slot bound;
- each producer's investment-fund share and cut-off price;
- the capacity-ratio trace and the investment rejection rate.

White-box, like the baselines: the tracer wraps `system.complete_cycle` to snapshot the placed
bids just before the auction, then reads the result's diagnostic `bid_outcomes` and
`order_resolutions` (and, for `ttl_outage`, the system's outage table). It only reads, so
episodes run identically with and without it. `attach(env)` follows every reset;
`attach_system(...)` traces a bare domain `System`.

DIAGNOSTIC ONLY: rejection reasons include `outage`, which reveals hidden shock state. Nothing
here may feed a policy or a teacher fork (design D4).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from collections.abc import Sequence

from ..domain.common import OrderId, ProducerId, Product, is_nan
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
    q1_orders: int = 0
    q1_lowest: int = 0
    routes: int = 0
    zero_throughput_routes: int = 0
    overflow_orders: int = 0
    fund_share: dict[ProducerId, float] = field(default_factory=lambda: defaultdict(float))
    cut_off_sum: dict[ProducerId, float] = field(default_factory=lambda: defaultdict(float))
    cut_off_n: dict[ProducerId, int] = field(default_factory=lambda: defaultdict(int))
    cycles: int = 0
    investment_resolved: int = 0
    investment_rejected: int = 0


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
        per_product: dict[Product, dict[ProducerId, int]] = defaultdict(lambda: defaultdict(int))
        for pid, p in system.producing_agents.items():
            reserved = p.reserved_capacity()
            for bid in p.bids:
                per_order[bid.order_id].add(pid)
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

        fund = system.investment_fund
        for order in system.orders.values():
            pid = order.producer_id()
            if pid is not None and order.funded and order.cycle_counter == 0 and fund > 0:
                c.fund_share[pid] += order.tokens / fund
        for pid, p in system.producing_agents.items():
            if not is_nan(p.cut_off_price):
                c.cut_off_sum[pid] += p.cut_off_price
                c.cut_off_n[pid] += 1

    def _after_auction(self, result: CycleResult) -> None:
        c = self._c
        c.cycles += 1
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
        for r in result.order_resolutions:
            cls = order_class(r.order_kind)
            last = c.last_reason.pop(r.order_id, "never_bid")
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
        out["herfindahl_mean"] = c.hhi_sum / c.hhi_n if c.hhi_n else 0.0
        out["q1_lowest_index_share"] = c.q1_lowest / c.q1_orders if c.q1_orders else 0.0
        out["q1_orders"] = float(c.q1_orders)
        out["zero_throughput_route_share"] = c.zero_throughput_routes / c.routes if c.routes else 0.0
        out["overflow_orders"] = float(c.overflow_orders)
        shares = self.fund_shares()
        out["fund_share_herfindahl"] = sum(s * s for s in shares.values())
        out["investment_rejection_rate"] = (
            c.investment_rejected / c.investment_resolved if c.investment_resolved else 0.0)
        return out

    def fund_shares(self) -> dict[ProducerId, float]:
        """Each producer's mean per-cycle share of the investment fund."""
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
