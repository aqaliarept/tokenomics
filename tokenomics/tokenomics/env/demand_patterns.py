"""Seeded demand patterns on the B9-randomized consumers (add-m3-development-audit design D11;
stochastic-environment "Demand patterns", "Demand patterns draw on their own streams").

A pattern is demand, not a shock: the schedules built here are what the consumers order, the
demand digest hashes them, and the shock-free twin keeps them. No observation feature exposes a
pattern; agents see only the orders. The domain draws nothing (CLAUDE.md): the env calls
`pattern_schedules` at reset and injects the result as `ScheduledConsumer` schedules.

Streams. Each kind draws from its own `derived_rng(seed, "pattern-<kind>")` (ADR-0055 style),
never from the legacy B9 stream (`default_rng(seed)`), the drift stream or a shock stream, so a
pattern moves no other draw. Generation is cycle-major with a fixed number of draws per
(cycle, consumer), after a fixed set of per-episode draws, and every window lives in the config,
so a shorter horizon's schedule is a prefix of a longer one. (The legacy B9 stream is
consumer-major and is not a prefix across horizons, which is why the base draws of `burst` and
`seasonal` are made on the pattern's stream, from the same B9 distribution, as drift does.)

Kinds (parameters in `domain.config.DemandPattern`):
- burst: base (product, quantity) from the B9 distribution, then with probability p the quantity
  is multiplied by an integer factor in [min, max]; 4 draws per order (product, quantity, the
  burst test, the factor, always all four);
- hit: a seeded permutation ranks the patterned consumers' products; the ranking shifts by one
  every `hit_period` cycles (the hit moves to the next product); each consumer draws its product
  from its pool with weights rank^(-alpha), quantity uniform in its range; 2 draws per order;
- seasonal: base (product, quantity) from the B9 distribution, quantity x s(t) with
  s(t) = 1 + A sin(2 pi (t + phase) / period), t = 1, 2, ..., stochastically rounded, never below
  the consumer's `quantity_min`; 3 draws per order. The envelope depends on the config only;
- switch: a seeded half (floor(n/2), at least 1) of the products is favoured (weight 1, the rest
  `low_weight`); `switch_count` switch cycles drawn without replacement from the config window
  with gaps >= `switch_min_gap`; at each one (from that cycle on) the favoured half and its
  complement swap; quantity uniform in the consumer's range; 2 draws per order.
Tokens are untouched: a consumer spends its emission share whatever quantity it orders.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..domain.common import ConsumerId, Product
from ..domain.config import (PATTERN_BURST, PATTERN_HIT, PATTERN_SEASONAL, ConsumerConfig,
                             DemandPattern)
from ..domain.consumer import ScheduleEntry
from .shock_schedule import derived_rng


def pattern_stream(kind: str) -> str:
    return f"pattern-{kind}"


def pattern_products(consumers: Sequence[ConsumerConfig]) -> list[Product]:
    """The products a pattern ranks: every product in a patterned consumer's pool, sorted."""
    return sorted({p for c in consumers for p in c.product_pool})


def switch_cycles(rng: np.random.Generator, pattern: DemandPattern) -> list[int]:
    """`switch_count` cycles in the config window, pairwise at least `switch_min_gap` apart:
    a uniform draw without replacement from the window shrunk by the gaps, then spread back."""
    k, gap = pattern.switch_count, pattern.switch_min_gap
    span = pattern.switch_window_max - pattern.switch_window_min + 1 - (k - 1) * (gap - 1)
    picks = sorted(int(x) for x in rng.choice(span, size=k, replace=False))
    return [pattern.switch_window_min + x + i * (gap - 1) for i, x in enumerate(picks)]


def _weighted(rng: np.random.Generator, pool: tuple[Product, ...],
              weight: dict[Product, float]) -> Product:
    w = np.array([weight[p] for p in pool], dtype=np.float64)
    total = w.sum()
    probs = w / total if total > 0 else np.full(len(pool), 1.0 / len(pool))
    return pool[int(rng.choice(len(pool), p=probs))]


def seasonal_factor(pattern: DemandPattern, t: int) -> float:
    """s(t) for cycle t (1-based), from the config only."""
    return 1.0 + pattern.amplitude * math.sin(2.0 * math.pi * (t + pattern.phase) / pattern.period)


def pattern_schedules(consumers: Sequence[ConsumerConfig], pattern: DemandPattern, seed: int,
                      horizon: int) -> dict[ConsumerId, list[ScheduleEntry]]:
    """Per-cycle schedules (length `horizon`) of the patterned consumers, in config order."""
    out: dict[ConsumerId, list[ScheduleEntry]] = {c.id: [] for c in consumers}
    if not consumers or horizon < 1:
        return out
    rng = derived_rng(seed, pattern_stream(pattern.kind))
    products = pattern_products(consumers)
    n = len(products)

    # per-episode draws, fixed in number, before any cycle
    ranking: list[Product] = []
    favoured: set[Product] = set()
    switches: list[int] = []
    if pattern.kind == PATTERN_HIT:
        ranking = [products[int(i)] for i in rng.permutation(n)]
    elif pattern.kind not in (PATTERN_BURST, PATTERN_SEASONAL):  # switch
        order = [products[int(i)] for i in rng.permutation(n)]
        favoured = set(order[:max(1, n // 2)])
        switches = switch_cycles(rng, pattern)

    for t in range(1, horizon + 1):
        weight: dict[Product, float] = {}
        if pattern.kind == PATTERN_HIT:
            shift = (t - 1) // pattern.hit_period
            weight = {ranking[(r + shift) % n]: (r + 1) ** -pattern.hit_alpha for r in range(n)}
        elif switches or favoured:
            flips = sum(1 for s in switches if s <= t)
            now = favoured if flips % 2 == 0 else set(products) - favoured
            weight = {p: 1.0 if p in now else pattern.low_weight for p in products}
        for c in consumers:
            pool, qmin, qmax = c.product_pool, c.quantity_min, c.quantity_max
            if pattern.kind == PATTERN_BURST:
                product = pool[int(rng.integers(len(pool)))]
                quantity = int(rng.integers(qmin, qmax + 1))
                burst = rng.random() < pattern.burst_p
                factor = int(rng.integers(pattern.burst_factor_min, pattern.burst_factor_max + 1))
                out[c.id].append((product, quantity * factor if burst else quantity))
            elif pattern.kind == PATTERN_SEASONAL:
                product = pool[int(rng.integers(len(pool)))]
                quantity = int(rng.integers(qmin, qmax + 1))
                scaled = quantity * seasonal_factor(pattern, t)
                rounded = math.floor(scaled) + (1 if rng.random() < scaled - math.floor(scaled) else 0)
                out[c.id].append((product, max(qmin, rounded)))
            else:  # hit / switch: weighted product, B9 quantity
                product = _weighted(rng, pool, weight)
                out[c.id].append((product, int(rng.integers(qmin, qmax + 1))))
    return out
