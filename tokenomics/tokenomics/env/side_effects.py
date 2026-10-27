"""Side-effect metrics for the measurement gate (design D3, add-measurement-gate-harness;
side-effect-guards). Computed in the env from domain data; the domain stays pure.

The env reports RAW quantities only. The thresholds X (run-down) and Y (starvation) are
frozen later by calibration (design D4) and applied by `eval/gate.py`, never here, so
nothing in this module emits a pass/fail.

`episode_stats` keys written by `SideEffectTracker.write` (all present from `reset`):

Hard-check inputs
- `r_start` (float | None): capacity ratio `r` read right after `reset`.
- `r_end` (float | None): `r` after the last played cycle (the death cycle on an
  early-death episode, so death shows as run-down), read from `CycleResult.end_capacities`,
  the snapshot taken before the next cycle's shocks (add-shocks-i-drift D2). None until a
  cycle is played.
- `consumer_fill_rate` (dict consumer id -> float | None): fulfilled / ordered units over
  the played-cycle cohort (design D2). None for a consumer with 0 ordered units: it has
  no fill rate, and the starvation check only covers consumers with ordered > 0.

Monitored only
- `inversion_inverted_pairs` / `inversion_differing_pairs` (int) and `inversion_rate`
  (float | None = inverted / differing; None when no pair differs). See
  `InversionCounter` for the pair definition.
- `min_capacity_ratio` (float | None): minimum of `r` over played cycles (read after each
  cycle, so `r_start` is not included). None until a cycle is played.
- `min_spare_fraction` (float | None): minimum over producer-cycles of
  `1 - accepted bundle units / capacity at auction`. Producer-cycles with zero capacity
  are skipped (nothing to be spare); None when no producer-cycle qualifies. Accepted
  units are the bids accepted THIS cycle only: capacity still held by reservations
  carried from earlier cycles is not subtracted, so spare can be overstated
  (`ProducerAuctionRecord`; acceptable because the metric is monitored only).
- `investment_rejection_rate` (float | None): investment orders rejected /
  (completed + rejected), in orders (restoration, upgrade, inventory expansion; not
  replenishment). None when no investment order resolved.
- `consumer_rejection_rate` (float | None): consumer units rejected / (completed +
  rejected units). Orders still open at the end are not resolved, so not counted.
  None when no consumer order resolved.
- `holding_units_per_cycle` (float | None): `holding_units` / played cycles. None until a
  cycle is played. (`holding_units` itself is the existing total.)

Zero denominators always give None, never 0 or NaN, so a missing rate cannot be read as
a perfect or a failing score.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..domain.common import Capacity, CapacityType, ConsumerId, Product
from ..domain.system import ConsumerOrderOutcome, CycleResult, ProducerAuctionRecord


class HasCapacities(Protocol):
    """A producer as the capacity ratio sees it (`ProducingAgent` and the domain's
    end-of-cycle `CapacitySnapshot` satisfy this)."""

    capacities: dict[CapacityType, Capacity]
    max_capacities: dict[CapacityType, Capacity]


def ratio(num: float, den: float) -> float | None:
    """`num / den`, or None for a zero denominator (module docstring)."""
    return None if den == 0 else num / den


def capacity_ratio(producers: Iterable[HasCapacities]) -> float | None:
    """`r` (design D3): mean over producers of their mean over owned types of
    current / max capacity. Max is read live, so an upgrade raises the denominator.
    A type with max 0 has no ratio and is skipped, and a producer left with no
    ratio-bearing type is skipped; None when nothing qualifies."""
    per_producer: list[float] = []
    for p in producers:
        terms = [p.capacities[t] / m for t, m in p.max_capacities.items() if m > 0]
        if terms:
            per_producer.append(sum(terms) / len(terms))
    return sum(per_producer) / len(per_producer) if per_producer else None


def spare_fraction(record: ProducerAuctionRecord) -> float | None:
    """`1 - accepted bundle units / capacity at auction` (design D3). None at zero
    capacity: a producer with nothing left has no spare to measure (its loss already
    shows in `r`)."""
    used = ratio(record.accepted_units, record.capacity)
    return None if used is None else 1.0 - used


def fill_rates(
    ordered: Mapping[ConsumerId, int], fulfilled: Mapping[ConsumerId, int]
) -> dict[ConsumerId, float | None]:
    """Per-consumer fulfilled / ordered units; None for a consumer that ordered nothing."""
    return {cid: ratio(fulfilled[cid], q) for cid, q in ordered.items()}


@dataclass
class InversionCounter:
    """Token-priority inversion (design D3), counted incrementally over resolved
    consumer orders.

    Pairs are formed within a (placement cycle, product) group: orders placed for the
    same cycle compete over the same auctions for their whole TTL window, so a
    completed order backed by fewer tokens per unit than a rejected one means priority
    was not respected. Grouping by resolution cycle instead would pair orders from
    different windows (a cycle-c completion with a cycle-(c-1) order expiring at its
    TTL). A pair is one completed and one rejected order; it counts toward the
    denominator when their tokens per unit differ, and as inverted when the completed
    one has the lower tokens per unit. Orders still open at the end are never paired.
    Tokens per unit are compared exactly by cross-multiplying integers."""

    inverted: int = 0
    differing: int = 0
    # (placement cycle, product) -> (tokens, quantity) of completed / rejected orders
    _completed: dict[tuple[int, Product], list[tuple[int, int]]] = field(default_factory=dict)
    _rejected: dict[tuple[int, Product], list[tuple[int, int]]] = field(default_factory=dict)

    def add(self, outcome: ConsumerOrderOutcome) -> None:
        key = (outcome.placement_cycle, outcome.product)
        t, q = int(outcome.tokens), outcome.quantity
        mine, other = (
            (self._completed, self._rejected) if outcome.completed
            else (self._rejected, self._completed)
        )
        for ot, oq in other.get(key, ()):
            # sign of (completed tpu - rejected tpu), via t1/q1 vs t2/q2 -> t1*q2 vs t2*q1
            diff = t * oq - ot * q if outcome.completed else ot * q - t * oq
            if diff != 0:
                self.differing += 1
                if diff < 0:
                    self.inverted += 1
        mine.setdefault(key, []).append((t, q))

    @property
    def rate(self) -> float | None:
        return ratio(self.inverted, self.differing)


@dataclass
class SideEffectTracker:
    """Per-episode accumulator behind the D3 keys. The env creates one at `reset` and
    feeds it once per played cycle, after `complete_cycle` (capacity read post-cycle)."""

    r_start: float | None
    r_end: float | None = None
    min_capacity_ratio: float | None = None
    min_spare_fraction: float | None = None
    inversion: InversionCounter = field(default_factory=InversionCounter)

    def observe_cycle(self, result: CycleResult, r_after: float | None) -> None:
        self.r_end = r_after
        if r_after is not None and (self.min_capacity_ratio is None or r_after < self.min_capacity_ratio):
            self.min_capacity_ratio = r_after
        for rec in result.producer_auctions:
            s = spare_fraction(rec)
            if s is not None and (self.min_spare_fraction is None or s < self.min_spare_fraction):
                self.min_spare_fraction = s
        for out in result.consumer_outcomes:
            self.inversion.add(out)

    def write(self, stats: dict[str, Any]) -> None:
        """Write every D3 key into `stats`, reading the totals already accumulated there.
        Rewritten after each cycle so a copy of `episode_stats` taken at any time is
        self-consistent."""
        stats["r_start"] = self.r_start
        stats["r_end"] = self.r_end
        stats["consumer_fill_rate"] = fill_rates(
            stats["consumer_ordered_units"], stats["consumer_fulfilled_units"]
        )
        stats["inversion_inverted_pairs"] = self.inversion.inverted
        stats["inversion_differing_pairs"] = self.inversion.differing
        stats["inversion_rate"] = self.inversion.rate
        stats["min_capacity_ratio"] = self.min_capacity_ratio
        stats["min_spare_fraction"] = self.min_spare_fraction
        stats["investment_rejection_rate"] = ratio(
            stats["investment_rejected"], stats["investment_completed"] + stats["investment_rejected"]
        )
        stats["consumer_rejection_rate"] = ratio(
            stats["consumer_rejected_units"],
            stats["consumer_completed_units"] + stats["consumer_rejected_units"],
        )
        stats["holding_units_per_cycle"] = ratio(stats["holding_units"], stats["cycles"])
