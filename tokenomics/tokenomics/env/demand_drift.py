"""Seeded consumer demand: the B9 per-entry draw and smooth demand drift (ADR-0055;
add-shocks-i-drift design D4; stochastic-environment "Smooth demand drift").

`draw_schedule_entry` is the single authority for randomized-demand legality: a uniform
product from the consumer's pool, then a uniform quantity in its range. It is shared by
episode-schedule generation (`TokenomicsAECEnv._build_consumers`, which re-exports it) and
EXP-0040's future-demand fork. The draw order defines the RNG stream, so it never changes.

Drift. With `Configuration.demand_drift = DemandDrift(j)`, every B9-randomized consumer (a
pool or quantity range, no explicit schedule) drifts. N is the number of those consumers,
each ordering once per cycle. Generation is cycle-major on the episode's `drift` stream
(`derived_rng(seed, "drift")`), so a shorter horizon is a prefix of a longer one, and it
never touches the legacy B9 stream or any shock stream:
- cycle 1: each drifting consumer, in config order, draws an entry with
  `draw_schedule_entry` (the B9 distribution);
- each transition: k = floor(j·N), computed exactly with `Fraction(str(j))`; k distinct
  consumers are chosen uniformly without replacement, and each, in config order, redraws
  uniformly from its pool × quantity range minus its current entry (`redraw`, by rank, never
  materializing the domain; it keeps its entry when that set is empty).
At most k orders change per transition, and a product switch counts as one change. The
stationary marginal of redraw-from-the-others on a finite set is uniform, like B9's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from ..domain.config import ConsumerConfig
from ..domain.consumer import ScheduleEntry
from .shock_schedule import derived_rng

DRIFT_STREAM = "drift"


def draw_schedule_entry(
    rng: np.random.Generator, pool: tuple[int, ...], quantity_min: int, quantity_max: int
) -> ScheduleEntry:
    """One legal (product, quantity) entry: uniform product from `pool`, then uniform
    quantity in `[quantity_min, quantity_max]` (module docstring)."""
    return (pool[int(rng.integers(len(pool)))], int(rng.integers(quantity_min, quantity_max + 1)))


def drift_budget(j: float, n: int) -> int:
    """k = floor(j * N), exact: `Fraction(str(j))` reads j as its decimal (0.34 * 3 = 1.02)."""
    return math.floor(Fraction(str(j)) * n)


def redraw(rng: np.random.Generator, c: ConsumerConfig, current: ScheduleEntry) -> ScheduleEntry:
    """A uniform entry from the consumer's domain minus `current`, without materializing it.

    The domain is pool position x quantity, ranked position-major (rank = i*Q + q - qmin), the
    order an explicit enumeration would list. Every rank equal to `current` is excluded (a
    product listed twice in the pool appears twice). One `rng.integers(remaining)` draw picks
    the r-th remaining rank, so the result and the RNG stream equal choosing uniformly from
    the enumerated alternatives, at O(pool) memory for any quantity range. Keeps `current`
    when nothing else is legal."""
    pool = c.product_pool
    width = c.quantity_max - c.quantity_min + 1
    excluded = sorted(i * width + (current[1] - c.quantity_min)
                      for i, p in enumerate(pool) if p == current[0])
    remaining = len(pool) * width - len(excluded)
    if remaining <= 0:
        return current
    rank = int(rng.integers(remaining))
    for e in excluded:  # shift past every excluded rank at or below the target
        if e <= rank:
            rank += 1
    i, offset = divmod(rank, width)
    return (pool[i], c.quantity_min + offset)


def drift_schedules(consumers: Sequence[ConsumerConfig], j: float, seed: int,
                    horizon: int) -> dict[str, list[ScheduleEntry]]:
    """Per-cycle schedules (length `horizon`) for the drifting consumers, in config order."""
    if not consumers or horizon < 1:
        return {c.id: [] for c in consumers}
    rng = derived_rng(seed, DRIFT_STREAM)
    k = drift_budget(j, len(consumers))
    current = [draw_schedule_entry(rng, c.product_pool, c.quantity_min, c.quantity_max)
               for c in consumers]
    schedules: list[list[ScheduleEntry]] = [[e] for e in current]
    for _ in range(1, horizon):
        if k:
            for i in sorted(int(x) for x in rng.choice(len(consumers), size=k, replace=False)):
                current[i] = redraw(rng, consumers[i], current[i])
        for i, e in enumerate(current):
            schedules[i].append(e)
    return {c.id: s for c, s in zip(consumers, schedules)}
