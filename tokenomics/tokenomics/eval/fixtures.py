"""Out-of-family / scarcity / degenerate fixtures (PRODUCT_AGNOSTIC step 5).

The scale family (`build_scale_config`) is a strictly layered, single-input chain: each
composite product consumes exactly one lower product at the same within-level index, and
each level owns its own capacity type. That regularity is precisely what a portable,
id-free contract must NOT depend on. These fixtures break it deliberately so the C3
transfer rows (step 6) test the recipe axis, not a re-seeded scale economy.

Reuses the scale builder's maintenance/investment scaffolding (`_MAINT`, `_pid`,
`_stride_for_width`) so the only thing that changes is the recipe DAG shape. Pure: no
torch, deterministic for a given seed.
"""

from __future__ import annotations

import random

from ..domain.config import (Configuration, ConsumerConfig, InventoryExpansion,
                             ProcessSheet, ProducingAgentConfig, Restoration, Upgrade)
from .scale import _MAINT, _pid, _stride_for_width


def build_forkjoin_config(seed: int, *, depth: int = 3, width: int = 4,
                          inputs_per_product: int = 2,
                          num_consumers: int = 3) -> Configuration:
    """A fork/join, multi-input economy: each level-`l` composite product (l>=1) consumes
    `inputs_per_product` DISTINCT lower products (indices j, j+1, ... mod width), so every
    lower product fans out to several upper products (fork) and every upper product joins
    several inputs (join). This leaves the layered single-input scale family while staying
    feasible (every product remains makeable: layer 0 is primal, the first producer per
    level is always strong). Capacity stays one type per level — the recipe DAG is the
    out-of-family axis here; shared-capacity is a separate fixture."""
    if inputs_per_product < 1:
        raise ValueError("inputs_per_product must be >= 1")
    rng = random.Random(seed)
    W = width
    stride = _stride_for_width(W)
    n_in = min(inputs_per_product, W)  # can't join more distinct inputs than the level has

    # --- recipe DAG: multi-input join, FIXED across seeds (dims-preserving) ---
    sheets: list[ProcessSheet] = []
    for level in range(depth):
        cap_type = f"capacity-{level}"
        for j in range(W):
            require = {cap_type: rng.randint(2, 6) if level else rng.randint(2, 4)}
            if level == 0:
                sheets.append(ProcessSheet(product=_pid(0, j, stride), require=require))
            else:
                inputs = {_pid(level - 1, (j + d) % W, stride): 1 for d in range(n_in)}
                sheets.append(ProcessSheet(
                    product=_pid(level, j, stride), require=require, inputs=inputs))

    # --- producers: producers_per_layer per level, all declare their level's products ---
    producers: list[ProducingAgentConfig] = []
    for level in range(depth):
        cap_type = f"capacity-{level}"
        level_products = frozenset(_pid(level, j, stride) for j in range(W))
        for k in range(W):
            strong = (k == 0) or rng.random() < 0.6
            cap = rng.randint(80, 140) if strong else rng.randint(8, 20)
            common = dict(
                id=f"p{level}_{k}",
                capacities={cap_type: cap},
                degradation=rng.randint(3, 7),
                restoration=Restoration(require=_MAINT, restores=rng.randint(40, 70),
                                        type=cap_type, cost=rng.randint(20, 50)),
                upgrade=Upgrade(require=_MAINT, increases=rng.randint(30, 50),
                                type=cap_type, cost=rng.randint(20, 50)),
                produces=level_products,
            )
            if level == 0:
                producers.append(ProducingAgentConfig(**common))  # primal: no inventory
            else:
                producers.append(ProducingAgentConfig(
                    **common,
                    inventory_limit=rng.randint(20, 40),
                    max_replenish=rng.randint(8, 14),
                    replenish_cost_per_unit=rng.randint(2, 6),
                    inventory_expansion=InventoryExpansion(
                        require=_MAINT, increases=rng.randint(15, 25), cost=rng.randint(20, 40)),
                ))

    top = depth - 1
    top_pool = tuple(_pid(top, j, stride) for j in range(W))
    consumers = [
        ConsumerConfig(id=f"c{i}", product=top_pool[0], products=top_pool,
                       quantity_min=1, quantity_max=rng.randint(2, 3))
        for i in range(num_consumers)
    ]

    cfg = Configuration(
        cycle_emission=2000 + 200 * depth * W,
        process_sheets=sheets,
        producer_configs=producers,
        consumers=consumers,
    )
    cfg.validate()
    return cfg
