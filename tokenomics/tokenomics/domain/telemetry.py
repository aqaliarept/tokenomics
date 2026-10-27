"""Persistent per-cycle EMA telemetry for the economy (EXP-0006A).

Domain-owned history behind the enriched orderer observation: recent routed /
completed / rejected flows, throughput, latency and demand, kept as exponential
moving averages so routing can react to the *current* bottleneck rather than a
lifetime average. Pure Python, no IO/RL —
it is simulation bookkeeping, like cycle accounting, not an RL artifact.

Contract: callers `record_*` raw per-cycle events, then `end_cycle()` folds each
per-cycle accumulator into its EMA via `EMA_next = alpha*current + (1-alpha)*prev`
and clears the accumulators. A cycle with no event for a series folds a 0, so the
EMA decays toward zero — recency is the point.
"""

from __future__ import annotations

from .common import Product, ProducerId


class _EmaSeries:
    """A keyed EMA: per-cycle raw accumulator + the folded moving average. `add`
    accrues this cycle; `fold` applies the EMA recurrence over the union of seen
    keys (so idle keys decay) and resets the accumulator."""

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._acc: dict[object, float] = {}
        self._ema: dict[object, float] = {}
        # producer -> sum of its EMA keys; rebuilt lazily after each fold (the only
        # mutation of `_ema`), so a producer total costs O(1), not a scan of all keys.
        self._by_producer: dict[object, float] | None = None

    def add(self, key: object, value: float) -> None:
        self._acc[key] = self._acc.get(key, 0.0) + value

    def get(self, key: object) -> float:
        return self._ema.get(key, 0.0)

    def sum_by_producer(self, producer_id: object) -> float:
        """Sum the EMA over every (producer_id, product) key — the producer total.
        Valid because the EMA recurrence is linear in its inputs.

        One pass over `_ema` (in its iteration order) groups each producer's values, then
        builtin `sum` adds each group: the same values, in the same order, through the
        same `sum` as the per-producer scan this replaced. That matters: Python >= 3.12
        `sum` compensates float error, so a running `+` would not be bit-identical. A
        producer with no keys still gets int 0. The scan per producer per stream was the
        120-producer joint-stack cost (task 8.3 sizing; docs/DIAGNOSTICS.md entry 17)."""
        if self._by_producer is None:
            groups: dict[object, list[float]] = {}
            for (pid, _x), v in self._ema.items():
                groups.setdefault(pid, []).append(v)
            self._by_producer = {pid: sum(vs) for pid, vs in groups.items()}
        return self._by_producer.get(producer_id, 0)

    def fold(self) -> None:
        for key in set(self._ema) | set(self._acc):
            current = self._acc.get(key, 0.0)
            prev = self._ema.get(key, 0.0)
            self._ema[key] = self._alpha * current + (1.0 - self._alpha) * prev
        self._acc = {}
        self._by_producer = None


class _WeightedEmaSeries:
    """A quantity-weighted EMA for latency-like values. Within a cycle, `add`
    accrues (value*weight, weight); `fold` updates a key's EMA toward the cycle's
    weighted-mean value ONLY if it saw weight this cycle — idle keys are held, not
    decayed, since a latency EMA decaying to zero would imply faster delivery."""

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._weighted_sum: dict[object, float] = {}
        self._weight: dict[object, float] = {}
        self._ema: dict[object, float] = {}

    def add(self, key: object, value: float, weight: float) -> None:
        self._weighted_sum[key] = self._weighted_sum.get(key, 0.0) + value * weight
        self._weight[key] = self._weight.get(key, 0.0) + weight

    def get(self, key: object) -> float:
        return self._ema.get(key, 0.0)

    def fold(self) -> None:
        for key, w in self._weight.items():
            if w <= 0:
                continue
            cycle_value = self._weighted_sum[key] / w
            prev = self._ema.get(key, 0.0)
            self._ema[key] = self._alpha * cycle_value + (1.0 - self._alpha) * prev
        self._weighted_sum = {}
        self._weight = {}


class EconomyTelemetry:
    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha
        self._eps = 1e-9
        self._routed = _EmaSeries(alpha)
        self._accepted = _EmaSeries(alpha)
        self._completed = _EmaSeries(alpha)
        self._rejected = _EmaSeries(alpha)
        self._failed = _EmaSeries(alpha)
        self._completion_latency = _WeightedEmaSeries(alpha)
        self._failure_latency = _WeightedEmaSeries(alpha)
        self._demand = _EmaSeries(alpha)
        self._rejected_demand = _EmaSeries(alpha)
        # (producer, product) pairs with an EMA entry in any per-pair series; rebuilt
        # lazily after each fold, the only time those EMAs gain keys (task 8.4).
        self._pairs: set[tuple[ProducerId, Product]] | None = None

    def record_routed(
        self, producer_id: ProducerId, product: Product, quantity: int, cycle: int
    ) -> None:
        """A sub-order of `quantity` units of `product` was routed to a producer
        this cycle (recorded at routing time, before production)."""
        self._routed.add((producer_id, product), quantity)

    def record_accepted(
        self,
        producer_id: ProducerId,
        product: Product,
        quantity: int,
        routed_cycle: int,
        cycle: int,
    ) -> None:
        """`quantity` units cleared a producer's auction this cycle (whether they
        complete now or reserve and wait)."""
        self._accepted.add((producer_id, product), quantity)

    def record_completed(
        self,
        producer_id: ProducerId,
        product: Product,
        quantity: int,
        routed_cycle: int,
        cycle: int,
    ) -> None:
        """`quantity` units of `product` were delivered by a producer this cycle."""
        self._completed.add((producer_id, product), quantity)
        self._completion_latency.add(
            (producer_id, product), cycle - routed_cycle, quantity
        )

    def record_rejected(
        self,
        producer_id: ProducerId,
        product: Product,
        quantity: int,
        routed_cycle: int,
        cycle: int,
    ) -> None:
        """`quantity` units never cleared the producer's auction this cycle."""
        self._rejected.add((producer_id, product), quantity)

    def record_failed(
        self,
        producer_id: ProducerId,
        product: Product,
        quantity: int,
        routed_cycle: int,
        cycle: int,
    ) -> None:
        """`quantity` units cleared earlier but their reservation dropped/TTL'd."""
        self._failed.add((producer_id, product), quantity)
        self._failure_latency.add(
            (producer_id, product), cycle - routed_cycle, quantity
        )

    def record_demand(self, product: Product, quantity: int) -> None:
        """`quantity` units of new consumer demand for `product` placed this cycle."""
        self._demand.add(product, quantity)

    def record_consumer_rejected(self, product: Product, quantity: int) -> None:
        """`quantity` units of consumer demand for `product` rejected this cycle."""
        self._rejected_demand.add(product, quantity)

    def has_pair_history(self, producer_id: ProducerId, product: Product) -> bool:
        """True iff some per-(producer, product) series holds an EMA entry for the pair.
        Without one, every per-pair read (`routed_ema`, `success_rate`, the latencies, ...)
        is the missing-key 0.0 — which lets the orderer view share one row for all such
        pairs of producers that do not declare the product."""
        if self._pairs is None:
            pairs: set[tuple[ProducerId, Product]] = set()
            for series in (self._routed, self._accepted, self._completed, self._rejected,
                           self._failed, self._completion_latency, self._failure_latency):
                pairs.update(series._ema)  # type: ignore[arg-type]
            self._pairs = pairs
        return (producer_id, product) in self._pairs

    def demand_ema(self, product: Product) -> float:
        """EMA of new consumer demand quantity per cycle for `product`."""
        return self._demand.get(product)

    def rejected_demand_ema(self, product: Product) -> float:
        """EMA of rejected consumer demand quantity per cycle for `product`."""
        return self._rejected_demand.get(product)

    def routed_ema(self, producer_id: ProducerId, product: Product) -> float:
        return self._routed.get((producer_id, product))

    def completed_ema(self, producer_id: ProducerId, product: Product) -> float:
        return self._completed.get((producer_id, product))

    def rejected_ema(self, producer_id: ProducerId, product: Product) -> float:
        return self._rejected.get((producer_id, product))

    def failed_ema(self, producer_id: ProducerId, product: Product) -> float:
        return self._failed.get((producer_id, product))

    def rejected_or_failed_ema(self, producer_id: ProducerId, product: Product) -> float:
        """EMA(rejected + failed) — the catalog's recent_rejected_qty[p,x]. Failed
        bids (a reservation that dropped/TTL'd after acceptance) are rejection
        pressure too, so the orderer must see them, not just auction rejections."""
        return self.rejected_ema(producer_id, product) + self.failed_ema(producer_id, product)

    # --- derived rates ---
    def success_rate(self, producer_id: ProducerId, product: Product) -> float:
        """EMA(completed) / max(eps, EMA(completed + rejected + failed)) — recent
        delivery reliability of this producer/product pair."""
        done = self.completed_ema(producer_id, product)
        attempted = done + self.rejected_ema(producer_id, product) + self.failed_ema(
            producer_id, product
        )
        return done / max(self._eps, attempted)

    def throughput(self, producer_id: ProducerId, product: Product) -> float:
        """EMA(completed) — recent production throughput per producer/product."""
        return self.completed_ema(producer_id, product)

    def completion_latency(self, producer_id: ProducerId, product: Product) -> float:
        """Quantity-weighted EMA of (completion_cycle - routed_cycle)."""
        return self._completion_latency.get((producer_id, product))

    def failure_latency(self, producer_id: ProducerId, product: Product) -> float:
        """Quantity-weighted EMA of (failure_cycle - routed_cycle)."""
        return self._failure_latency.get((producer_id, product))

    def producer_acceptance_rate(self, producer_id: ProducerId) -> float:
        """EMA(accepted[p]) / max(eps, EMA(routed[p])), aggregated over products."""
        return self._accepted.sum_by_producer(producer_id) / max(
            self._eps, self._routed.sum_by_producer(producer_id)
        )

    def producer_completion_rate(self, producer_id: ProducerId) -> float:
        """EMA(completed[p]) / max(eps, EMA(routed[p])), aggregated over products."""
        return self._completed.sum_by_producer(producer_id) / max(
            self._eps, self._routed.sum_by_producer(producer_id)
        )

    def producer_rejection_rate(self, producer_id: ProducerId) -> float:
        """EMA(rejected[p] + failed[p]) / max(eps, EMA(routed[p]))."""
        bad = self._rejected.sum_by_producer(producer_id) + self._failed.sum_by_producer(
            producer_id
        )
        return bad / max(self._eps, self._routed.sum_by_producer(producer_id))

    def end_cycle(self) -> None:
        """Fold this cycle's accumulators into their EMAs (call once per cycle)."""
        self._routed.fold()
        self._accepted.fold()
        self._completed.fold()
        self._rejected.fold()
        self._failed.fold()
        self._completion_latency.fold()
        self._failure_latency.fold()
        self._demand.fold()
        self._rejected_demand.fold()
        self._pairs = None
