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
envelope (ADR-0039 dims-preserving rule, generalized) — is the recipe DAG, the
capacity-type set, the per-producer `produces` assignment (so `max_rel` and thus the
ordering `k` are seed-invariant), and the producer/consumer counts. Pure: no torch,
deterministic for a given seed."""

from __future__ import annotations

import random

from ..domain.config import (Configuration, ConsumerConfig, InventoryExpansion,
                             ProcessSheet, ProducingAgentConfig, Restoration, Upgrade)


def config_to_dict(cfg: Configuration) -> dict:
    """Inverse of `domain.config.load_config`: serialize a Configuration to the JSON
    schema so a built scale topology can be pinned to a file and trained via
    `smoke_ppo --config` (the only clean way to fix a depth-D base; `--topology-depth`
    alone is inert without `--topology-random`/`--validation-seeds`)."""
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
        "consumers": [
            {"id": c.id, "product": c.product, "products": list(c.products),
             "quantityMin": c.quantity_min, "quantityMax": c.quantity_max}
            for c in cfg.consumers
        ],
    }

# product id = level*10 + index; capacity type = "capacity-{level}".
_MAINT = 0  # the layer-0 primal product every restoration/upgrade/expansion requires


def _pid(level: int, idx: int) -> int:
    return level * 10 + idx


def build_scale_config(
    seed: int,
    *,
    depth: int = 3,
    products_per_layer: int = 4,
    producers_per_layer: int = 4,
    num_consumers: int = 3,
) -> Configuration:
    """A valid layered economy (see module docstring). Defaults: 3 levels × 4 products
    = 12 products, 4 producers/level = 12 producers, 3 consumers — inside the stage-5
    envelope (8-16 / 10-20 / 3-5, ≥3 levels deep).

    Depth defaults to 3 as the cheapest scale smoke target. ADR-0041 corrected the
    earlier ADR-0040 diagnosis: depth 4-5 chains are fulfillable once producing
    observations expose outstanding inbound replenishment. `depth` stays a parameter
    so deeper-chain and stress findings remain reproducible."""
    rng = random.Random(seed)
    W = products_per_layer

    # --- recipe DAG (FIXED, seed-independent → fixes num_inputs / num_order_products) ---
    sheets: list[ProcessSheet] = []
    for level in range(depth):
        cap_type = f"capacity-{level}"
        for j in range(W):
            require = {cap_type: rng.randint(2, 6) if level else rng.randint(2, 4)}
            if level == 0:
                sheets.append(ProcessSheet(product=_pid(0, j), require=require))
            else:
                # consume the same-index product one level down (every lower product is
                # used as an input exactly once → input set is fixed across seeds).
                sheets.append(ProcessSheet(
                    product=_pid(level, j), require=require,
                    inputs={_pid(level - 1, j): 1},
                ))

    # --- producers: producers_per_layer per level, all declare their level's products ---
    producers: list[ProducingAgentConfig] = []
    for level in range(depth):
        cap_type = f"capacity-{level}"
        level_products = frozenset(_pid(level, j) for j in range(W))
        for k in range(producers_per_layer):
            # strong vs marginal within a level (feasible-topology variation); the
            # first producer is always strong so every product stays makeable.
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

    # --- consumers: demand the finished top layer with episode-seeded demand (B9) ---
    top = depth - 1
    top_pool = tuple(_pid(top, j) for j in range(W))
    consumers = [
        ConsumerConfig(id=f"c{i}", product=top_pool[0], products=top_pool,
                       quantity_min=1, quantity_max=rng.randint(2, 3))
        for i in range(num_consumers)
    ]

    cfg = Configuration(
        cycle_emission=2000 + 200 * depth * producers_per_layer,
        process_sheets=sheets,
        producer_configs=producers,
        consumers=consumers,
    )
    cfg.validate()
    return cfg


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
