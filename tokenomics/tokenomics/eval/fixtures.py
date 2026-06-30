"""Out-of-family / scarcity / degenerate fixtures (PRODUCT_AGNOSTIC steps 5 + 9).

The scale family (`build_scale_config`) is a strictly layered, single-input chain: each
composite product consumes exactly one lower product at the same within-level index, and
each level owns its own capacity type. That regularity is precisely what a portable,
id-free contract must NOT depend on. These fixtures break it deliberately so the C3/P3
transfer rows (steps 6, 10) test the contract's failure surface, not a re-seeded scale
economy.

Three families (step 9 broadens step 5's lone fork/join probe into the reusable set the
broad C3/P3 matrix consumes):
- out-of-family recipe/topology — `build_forkjoin_config` (multi-input join),
  `build_deep_join_config` (deeper + wider join), `build_shared_capacity_config` (levels
  share capacity types), `build_uneven_strength_config` (one dominant producer per level);
- scarcity stress — `scarcity_suite` (lean fund, tight inventory, heavy backlog, harsh
  decay, lean-and-decaying running-investment start), dims-preserving transforms;
- degenerate edges — `degenerate_suite` (single eligible producer, all-primal, no
  inventory expansion, no backlog, minimal/mostly-padded).

Reuses the scale builder's maintenance/investment scaffolding (`_MAINT`, `_pid`,
`_stride_for_width`) and the `generalization.perturb_*` axes. Pure: no torch,
deterministic for a given seed.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable

from ..domain.config import (Configuration, ConsumerConfig, InventoryExpansion,
                             ProcessSheet, ProducingAgentConfig, Restoration, Upgrade)
from .generalization import (perturb_degradation, perturb_producer_capacity,
                             perturb_quantity)
from .scale import _MAINT, _pid, _stride_for_width, build_scale_config

# per-level capacity TYPE, and per-(level, producer-slot) capacity AMOUNT, drawn from the
# builder's seeded rng — the two knobs the single-input out-of-family builders vary.
CapTypeFn = Callable[[int], str]
CapFn = Callable[[random.Random, int, int], int]


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


def build_deep_join_config(seed: int, *, depth: int = 5, width: int = 4,
                           inputs_per_product: int = 3,
                           num_consumers: int = 3) -> Configuration:
    """The fork/join economy pushed further on BOTH out-of-family recipe axes: a deeper
    chain (`depth` 5 vs the step-6 probe's 3) and wider joins (`inputs_per_product` 3 vs
    2). The DAG shape is `build_forkjoin_config`'s — only the defaults are deeper/wider —
    so a transfer failure here isolates recipe-graph depth/width, not a new mechanism.
    Pure, seed-deterministic."""
    return build_forkjoin_config(seed, depth=depth, width=width,
                                 inputs_per_product=inputs_per_product,
                                 num_consumers=num_consumers)


def _build_single_input_layered(seed: int, *, depth: int, width: int, num_consumers: int,
                                cap_type_fn: CapTypeFn, cap_fn: CapFn) -> Configuration:
    """Shared scaffold for the single-input out-of-family builders: the layered,
    single-input DAG + per-level producers (declaring their level's products) + top-layer
    consumers of `build_scale_config`, but with the per-level capacity TYPE chosen by
    `cap_type_fn(level)` and the per-producer capacity AMOUNT by `cap_fn(rng, level, k)`.
    That isolates each out-of-family axis to one knob: shared-capacity varies the type
    function, uneven-strength varies the amount function, neither touches the recipe DAG.
    Pure, seed-deterministic."""
    rng = random.Random(seed)
    W = width
    stride = _stride_for_width(W)

    sheets: list[ProcessSheet] = []
    for level in range(depth):
        cap_type = cap_type_fn(level)
        for j in range(W):
            require = {cap_type: rng.randint(2, 6) if level else rng.randint(2, 4)}
            if level == 0:
                sheets.append(ProcessSheet(product=_pid(0, j, stride), require=require))
            else:
                sheets.append(ProcessSheet(
                    product=_pid(level, j, stride), require=require,
                    inputs={_pid(level - 1, j, stride): 1}))

    producers: list[ProducingAgentConfig] = []
    for level in range(depth):
        cap_type = cap_type_fn(level)
        level_products = frozenset(_pid(level, j, stride) for j in range(W))
        for k in range(W):
            common = dict(
                id=f"p{level}_{k}",
                capacities={cap_type: cap_fn(rng, level, k)},
                degradation=rng.randint(3, 7),
                restoration=Restoration(require=_MAINT, restores=rng.randint(40, 70),
                                        type=cap_type, cost=rng.randint(20, 50)),
                upgrade=Upgrade(require=_MAINT, increases=rng.randint(30, 50),
                                type=cap_type, cost=rng.randint(20, 50)),
                produces=level_products,
            )
            if level == 0:
                producers.append(ProducingAgentConfig(**common))
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


def build_uneven_strength_config(seed: int, *, depth: int = 3, width: int = 4,
                                 num_consumers: int = 3) -> Configuration:
    """A single-input layered economy with a DEDICATED producer-strength axis: every level
    has exactly one dominant producer (capacity 120-160) and the rest are weak (4-10), so
    each level's strong/weak ratio is guaranteed extreme (>= 12x) rather than the scale
    family's random split (which can come out all-strong). Routing/investment must
    concentrate on the one viable producer per level. Recipe DAG and one-type-per-level
    capacity stay in-family — strength dispersion is the only out-of-family axis. Pure,
    seed-deterministic."""
    return _build_single_input_layered(
        seed, depth=depth, width=width, num_consumers=num_consumers,
        cap_type_fn=lambda level: f"capacity-{level}",
        cap_fn=lambda rng, level, k: rng.randint(120, 160) if k == 0 else rng.randint(4, 10),
    )


def _shared_cap_type(level: int, n_types: int) -> str:
    """Capacity type for a level under capacity sharing: levels cycle through `n_types`
    types, so when `n_types < depth` several levels SHARE a type."""
    return f"capacity-{level % n_types}"


def build_shared_capacity_config(seed: int, *, depth: int = 3, width: int = 4,
                                 num_consumers: int = 3) -> Configuration:
    """A single-input layered economy whose levels SHARE capacity types (out-of-family on
    the capacity axis, not the recipe-join axis — kept single-input so the only thing that
    leaves the scale family is type sharing). With `n_types = max(1, depth - 1)` the
    layers cycle through fewer capacity types than levels, so one capacity pool serves
    several layers at once: degradation on a shared type now bites multiple levels and the
    investment fund must split that type across them. Stays feasible — the first producer
    per level is strong and every product remains makeable. Pure, seed-deterministic."""
    n_types = max(1, depth - 1)  # fewer types than levels (for depth >= 2) -> sharing

    def cap_fn(rng: random.Random, level: int, k: int) -> int:
        strong = (k == 0) or rng.random() < 0.6
        return rng.randint(80, 140) if strong else rng.randint(8, 20)

    return _build_single_input_layered(
        seed, depth=depth, width=width, num_consumers=num_consumers,
        cap_type_fn=lambda level: _shared_cap_type(level, n_types),
        cap_fn=cap_fn,
    )


# --- scarcity stress transforms (dims-preserving, like generalization.perturb_*) ---


def scale_funding(cfg: Configuration, factor: float) -> Configuration:
    """Scale the cycle emission by `factor` (<1 = a leaner investment fund, the scarcity
    axis: producer purchasing is funded from half the emission, so less funding starves
    restoration/upgrade/replenishment). Only a token amount changes, so dims are
    preserved."""
    return replace(cfg, cycle_emission=max(1, round(cfg.cycle_emission * factor)))


def scale_inventory_headroom(cfg: Configuration, factor: float) -> Configuration:
    """Scale every producer's inventory headroom — the unit limit, the per-cycle
    replenishment ceiling, and the expansion increment — by `factor` (<1 = tight buffers,
    forcing just-in-time replenishment and removing the slack that hides routing mistakes).
    Bounded below by 1 where a value is set so validate() stays happy (a positive
    max_replenish/expansion cost is required). Dims-preserving — only amounts move."""
    def _pos(v: int) -> int:
        return max(1, round(v * factor))

    producers = []
    for p in cfg.producer_configs:
        exp = (replace(p.inventory_expansion, increases=_pos(p.inventory_expansion.increases))
               if p.inventory_expansion is not None else None)
        producers.append(replace(
            p,
            inventory_limit=_pos(p.inventory_limit) if p.inventory_limit else 0,
            max_replenish=_pos(p.max_replenish) if p.max_replenish is not None else None,
            inventory_expansion=exp,
        ))
    return replace(cfg, producer_configs=producers)


def scarcity_suite(base_cfg: Configuration) -> dict[str, Configuration]:
    """Single-axis SCARCITY stress variants of `base_cfg` (ordered-plan step 9), each
    dims-preserving so a policy built on the base loads unchanged and a failure is
    attributable to the stress, not a reshape. Axes: a lean investment fund
    (`low_funding`), tight inventory buffers (`low_inventory`), heavier demand pressure
    (`high_backlog`), harsher decay (`degraded`), and a lean-and-decaying start that forces
    investment orders from cycle 1 (`running_investment` — the static-config stand-in for
    an economy already mid-investment, since in-flight orders are runtime state, not
    config). Reuses the `generalization.perturb_*` axes for decay/capacity/demand."""
    max_qty = max(c.quantity_max for c in base_cfg.consumers)
    return {
        "low_funding": scale_funding(base_cfg, 0.4),
        "low_inventory": scale_inventory_headroom(base_cfg, 0.3),
        "high_backlog": perturb_quantity(base_cfg, quantity_max=max_qty + 4),
        "degraded": perturb_degradation(base_cfg, 2.0),
        "running_investment": perturb_degradation(
            perturb_producer_capacity(base_cfg, 0.4), 2.0),
    }


# --- degenerate edge-economy fixtures (boundaries of the producer/orderer contract) ---


def degenerate_suite(seed: int, *, depth: int = 3, width: int = 4) -> dict[str, Configuration]:
    """Degenerate edge economies (ordered-plan step 9) that probe the boundaries of the
    id-free contract rather than its typical operating point. Each is a real, valid economy
    so the matrix can run it:

    - `single_producer`: one producer per level, so every product has exactly one eligible
      declarer (zero routing alternatives — the orderer's choice collapses).
    - `primal_only`: a depth-1 economy, no input products at all (the replenish/inventory
      machinery is inert).
    - `no_inventory_expansion`: every producer's inventory-expansion investment stripped
      (the limit can never grow).
    - `no_backlog`: demand pinned to a fixed single product at quantity 1 (trivially
      satisfiable — slack everywhere).
    - `minimal`: one product and one producer per level, so once encoded to the fixed-width
      entity view almost every row is padding.

    Pure, seed-deterministic."""
    base = build_scale_config(seed, depth=depth, products_per_layer=width)

    no_exp = replace(base, producer_configs=[
        replace(p, inventory_expansion=None) for p in base.producer_configs])

    top = max(c.product_pool[0] for c in base.consumers)
    no_backlog = replace(base, consumers=[
        replace(c, product=top, products=(), quantity_min=1, quantity_max=1)
        for c in base.consumers])

    return {
        "single_producer": build_scale_config(
            seed, depth=depth, products_per_layer=width, producers_per_layer=1),
        "primal_only": build_scale_config(seed, depth=1, products_per_layer=width),
        "no_inventory_expansion": no_exp,
        "no_backlog": no_backlog,
        "minimal": build_scale_config(
            seed, depth=depth, products_per_layer=1, producers_per_layer=1),
    }
