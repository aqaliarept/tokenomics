"""Scaled-economy generator (NEXT STEPS stage 5).

Builds a larger but structurally *fixed* economy — a layered supply chain of `depth`
levels, `products_per_layer` products per level, `producers_per_layer` producers per
level (all declaring their level's products → several eligible producers per product),
and `num_consumers` consumers demanding the finished top layer with episode-seeded
randomized demand (B9).

`seed` drives **bounded variation that preserves obs/action dims**: capacities,
degradation, investment amounts/costs, and demand bounds are randomized, and within a
level capacities vary so some declarers are strong and others marginal (feasible-
topology variation, like competition.json). What stays fixed — and therefore the dims
bounds (ADR-0039 dims-preserving rule, generalized) — is the recipe DAG, the
capacity-type set, the per-producer `produces` assignment (so `max_rel` and thus the
ordering `k` are seed-invariant), and the producer/consumer counts. Pure: no torch,
deterministic for a given seed."""

from __future__ import annotations

import random
from dataclasses import asdict, replace
from typing import Sequence

from ..domain.config import (Configuration, ConsumerConfig, InventoryExpansion,
                             ProcessSheet, ProducingAgentConfig, Restoration, Upgrade)
from ..domain.shocks import shock_model_to_dict


def config_to_dict(cfg: Configuration) -> dict:
    """Inverse of `domain.config.load_config`: serialize a Configuration to the JSON
    schema so a built scale topology can be pinned to a file and loaded anywhere a
    config path is accepted (`load_config`, `signal_check --config`)."""
    producers = []
    for p in cfg.producer_configs:
        d: dict = {
            "id": p.id,
            "capacities": dict(p.capacities),
            "degradation": p.degradation,
            "restoration": {"product": p.restoration.require, "restores": p.restoration.restores,
                            "type": p.restoration.type, "cost": p.restoration.cost},
            "upgrade": {"product": p.upgrade.require, "capacity": p.upgrade.increases,
                        "type": p.upgrade.type, "cost": p.upgrade.cost},
            "produces": sorted(p.produces),
            "inventoryLimit": p.inventory_limit,
        }
        if p.inventory_expansion is not None:
            d["inventoryExpansion"] = {"product": p.inventory_expansion.require,
                                       "increases": p.inventory_expansion.increases,
                                       "cost": p.inventory_expansion.cost}
        if p.replenish_cost_per_unit is not None:
            d["replenishCostPerUnit"] = p.replenish_cost_per_unit
        if p.max_replenish is not None:
            d["maxReplenish"] = p.max_replenish
        producers.append(d)
    return {
        "cycleEmission": cfg.cycle_emission,
        "processSheets": [
            {"product": ps.product, "require": dict(ps.require),
             "inputs": {str(k): v for k, v in ps.inputs.items()}}
            for ps in cfg.process_sheets
        ],
        "producerConfigs": producers,
        "consumers": [_consumer_to_dict(c) for c in cfg.consumers],
    } | ({"shocks": shock_model_to_dict(cfg.shocks)} if cfg.shocks is not None else {}) \
      | ({"demandDrift": {"j": cfg.demand_drift.j}} if cfg.demand_drift is not None else {}) \
      | ({"demandPattern": asdict(cfg.demand_pattern)} if cfg.demand_pattern is not None else {})


def _consumer_to_dict(c: ConsumerConfig) -> dict:
    d: dict = {"id": c.id, "product": c.product, "products": list(c.products),
               "quantityMin": c.quantity_min, "quantityMax": c.quantity_max}
    # Emitted only when set (design D8), so legacy configs serialize unchanged.
    if c.schedule:
        d["schedule"] = [[product, quantity] for product, quantity in c.schedule]
    return d

# product id = level*stride + index; capacity type = "capacity-{level}". Widths <=10
# keep the historical stride=10 so existing experiment fixtures/results do not drift.
_MAINT = 0  # the layer-0 primal product every restoration/upgrade/expansion requires


def _stride_for_width(width: int) -> int:
    if width <= 10:
        return 10
    stride = 10
    while stride <= width:
        stride *= 10
    return stride


def _pid(level: int, idx: int, stride: int = 10) -> int:
    return level * stride + idx


def build_scale_config(
    seed: int,
    *,
    depth: int = 3,
    products_per_layer: int = 4,
    producers_per_layer: int = 4,
    num_consumers: int = 3,
    quantity_max: int | None = None,
    capacity_scale: float = 1.0,
    primal_capacity_scale: float = 1.0,
    degradation: int | None = None,
) -> Configuration:
    """A valid layered economy (see module docstring). Defaults: 3 levels × 4 products
    = 12 products, 4 producers/level = 12 producers, 3 consumers — inside the stage-5
    range (8-16 / 10-20 / 3-5, ≥3 levels deep).

    Depth defaults to 3 as the cheapest scale smoke target. ADR-0041 corrected the
    earlier ADR-0040 diagnosis: depth 4-5 chains are fulfillable once producing
    observations expose outstanding inbound replenishment. `depth` stays a parameter
    so deeper-chain and stress findings remain reproducible.

    Track A capacity-binding axes (all opt-in; the defaults draw the exact prior rng
    stream, so `capacity_scale=1.0, quantity_max=None, degradation=None` is byte-identical
    to the original economy): `quantity_max` raises consumer demand pressure,
    `capacity_scale` multiplies every producer's owned capacity (< 1 tightens toward a
    capacity-bound ceiling), `primal_capacity_scale` multiplies ONLY layer-0 (primal)
    capacity on top of `capacity_scale` (< 1 makes the primal source the binding layer —
    the regime whose "upgrade the tight primal first" skill the Track-A round-1 replay
    traded away, EXP-0034 s54), `degradation` overrides the per-cycle decay. Each rng draw
    still happens even when overridden, so topology and every other value stay aligned."""
    rng = random.Random(seed)
    W = products_per_layer
    stride = _stride_for_width(W)

    # --- recipe DAG (FIXED, seed-independent → fixes num_inputs / num_order_products) ---
    sheets: list[ProcessSheet] = []
    for level in range(depth):
        cap_type = f"capacity-{level}"
        for j in range(W):
            require = {cap_type: rng.randint(2, 6) if level else rng.randint(2, 4)}
            if level == 0:
                sheets.append(ProcessSheet(product=_pid(0, j, stride), require=require))
            else:
                # consume the same-index product one level down (every lower product is
                # used as an input exactly once → input set is fixed across seeds).
                sheets.append(ProcessSheet(
                    product=_pid(level, j, stride), require=require,
                    inputs={_pid(level - 1, j, stride): 1},
                ))

    # --- producers: producers_per_layer per level, all declare their level's products ---
    producers: list[ProducingAgentConfig] = []
    for level in range(depth):
        cap_type = f"capacity-{level}"
        level_products = frozenset(_pid(level, j, stride) for j in range(W))
        for k in range(producers_per_layer):
            # strong vs marginal within a level (feasible-topology variation); the
            # first producer is always strong so every product stays makeable.
            strong = (k == 0) or rng.random() < 0.6
            cap = rng.randint(80, 140) if strong else rng.randint(8, 20)
            # capacity-tight axes (opt-in): uniform `capacity_scale`, plus a layer-0-only
            # `primal_capacity_scale` so the primal source can be made the binding layer.
            scale = capacity_scale * (primal_capacity_scale if level == 0 else 1.0)
            cap = max(1, int(round(cap * scale)))
            deg = rng.randint(3, 7)  # drawn even when overridden -> rng stream stays aligned
            common = dict(
                id=f"p{level}_{k}",
                capacities={cap_type: cap},
                degradation=deg if degradation is None else degradation,
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

    # --- consumers: demand the finished top layer with episode-seeded demand (B9) ---
    top = depth - 1
    top_pool = tuple(_pid(top, j, stride) for j in range(W))
    consumers = []
    for i in range(num_consumers):
        qd = rng.randint(2, 3)  # drawn even when overridden -> rng stream stays aligned
        consumers.append(ConsumerConfig(
            id=f"c{i}", product=top_pool[0], products=top_pool, quantity_min=1,
            quantity_max=(qd if quantity_max is None else quantity_max)))

    cfg = Configuration(
        cycle_emission=2000 + 200 * depth * producers_per_layer,
        process_sheets=sheets,
        producer_configs=producers,
        consumers=consumers,
    )
    cfg.validate()
    return cfg


def permute_scale_config(
    cfg: Configuration,
    *,
    product_perm: Sequence[int],
    producer_perm: Sequence[int],
) -> Configuration:
    """Relabel a scale config's slots without changing the economy (ADR-0049 Phase 2
    slot randomization). `product_perm` permutes the within-level product index (applied
    to every level, so the index-aligned recipe DAG stays isomorphic); `producer_perm`
    permutes the within-level producer slot `k`. Both are bijections of the per-level
    width, so dims, the capacity-type set, and `produces` (always a whole level) are
    preserved — only WHICH sorted id holds which config changes. That moves the obs
    producer-summary slots, the input/open-demand product slots, and the replenish
    action dims (and, run through the real env, the expert labels) off fixed positions,
    fighting slot memorization. Pure: relabel + re-validate, no mechanics touched."""
    pp = list(product_perm)
    qp = list(producer_perm)
    stride = _stride_for_width(len(pp))

    def map_prod(pid: int) -> int:
        return (pid // stride) * stride + pp[pid % stride]

    def map_producer(pid: str) -> str:
        level, k = pid[1:].split("_")
        return f"p{level}_{qp[int(k)]}"

    sheets = [
        replace(ps, product=map_prod(ps.product),
                inputs={map_prod(k): v for k, v in ps.inputs.items()})
        for ps in cfg.process_sheets
    ]
    producers = [
        replace(
            p,
            id=map_producer(p.id),
            produces=frozenset(map_prod(x) for x in p.produces),
            restoration=replace(p.restoration, require=map_prod(p.restoration.require)),
            upgrade=replace(p.upgrade, require=map_prod(p.upgrade.require)),
            inventory_expansion=(
                replace(p.inventory_expansion, require=map_prod(p.inventory_expansion.require))
                if p.inventory_expansion is not None else None
            ),
        )
        for p in cfg.producer_configs
    ]
    consumers = [
        # `products or (product,)` is the pre-schedule pool, kept so legacy configs
        # relabel bit-identically; an explicit schedule (design D8) is relabeled too.
        replace(c, product=map_prod(c.product),
                products=tuple(map_prod(x) for x in (c.products or (c.product,))),
                schedule=tuple((map_prod(x), q) for x, q in c.schedule))
        for c in cfg.consumers
    ]
    out = Configuration(
        cycle_emission=cfg.cycle_emission, process_sheets=sheets,
        producer_configs=producers, consumers=consumers,
    )
    out.validate()
    return out


def scale_topology_set(
    seeds: Sequence[int],
    *,
    depth: int = 3,
    perms: int = 0,
    perm_seed: int = 0,
    products_per_layer: int = 4,
    producers_per_layer: int = 4,
) -> list[Configuration]:
    """The Phase 2 training-config set (ADR-0049): each base `build_scale_config(seed)`
    followed by `perms` slot-permuted variants of it (distinct product- and producer-
    index permutations drawn from `perm_seed`). All variants share the same dims so
    one net spans the set; the permutations relabel slots without changing the economy
    (`permute_scale_config`), adding slot-memorization pressure on top of the multi-seed
    mechanics variation. `perms=0` returns exactly the base configs (identity)."""
    rng = random.Random(perm_seed)
    out: list[Configuration] = []
    for s in seeds:
        base = build_scale_config(s, depth=depth, products_per_layer=products_per_layer,
                                  producers_per_layer=producers_per_layer)
        out.append(base)
        for _ in range(perms):
            pp = list(range(products_per_layer))
            qp = list(range(producers_per_layer))
            rng.shuffle(pp)
            rng.shuffle(qp)
            out.append(permute_scale_config(base, product_perm=pp, producer_perm=qp))
    return out


def main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="dump build_scale_config(seed, depth) to JSON")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = build_scale_config(args.seed, depth=args.depth)
    with open(args.out, "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2)
    print(f"depth {args.depth} seed {args.seed} -> {args.out} "
          f"({len(cfg.producer_configs)} producers, {len(cfg.process_sheets)} sheets)")


if __name__ == "__main__":
    main()
