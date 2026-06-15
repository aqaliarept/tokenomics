"""Configuration model + loader.

Port of emulation/domain/configuration.go and producing_agent.go config types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .common import Capacity, CapacityType, Product, Tokens


@dataclass(frozen=True)
class ProcessSheet:
    """Recipe: product -> required capacity per type, plus input products it
    consumes (ADR-0025). Primal = no inputs; composite = has inputs. Exogenous,
    fixed."""

    product: Product
    require: dict[CapacityType, Capacity]
    inputs: dict[Product, int] = field(default_factory=dict)

    @property
    def is_composite(self) -> bool:
        return len(self.inputs) > 0

    @property
    def is_primal(self) -> bool:
        return not self.is_composite


@dataclass(frozen=True)
class Restoration:
    require: Product
    restores: Capacity
    type: CapacityType  # which owned capacity type it restores (ADR-0025)
    cost: Tokens | None = None  # fund target for a restoration order (ADR-0027)


@dataclass(frozen=True)
class Upgrade:
    require: Product
    increases: Capacity
    type: CapacityType  # which owned capacity type it expands (ADR-0025)
    cost: Tokens | None = None  # fund target for an upgrade order (ADR-0027)


@dataclass(frozen=True)
class InventoryExpansion:
    require: Product
    increases: int  # units the inventory-unit limit grows (ADR-0025)
    cost: Tokens | None = None  # fund target for an expansion order (ADR-0027)


@dataclass(frozen=True)
class ProducingAgentConfig:
    """A producer owns one or more capacity types (ADR-0025)."""

    id: str
    capacities: dict[CapacityType, Capacity]
    degradation: int  # DegradationRate, percent of max capacity per cycle
    restoration: Restoration
    upgrade: Upgrade
    inventory_limit: int = 0  # max input-product units held (ADR-0025); 0 = none
    inventory_expansion: InventoryExpansion | None = None
    # fund target per replenished unit (ADR-0027); None = unbounded
    replenish_cost_per_unit: Tokens | None = None
    # per-input per-cycle replenishment ceiling (ADR-0025 B5); None = inventory_limit
    max_replenish: int | None = None


@dataclass(frozen=True)
class ConsumerConfig:
    """Phase 1 scripted demand. Not present in the Go config — added here.

    Each consumer places one fixed order for `product` every cycle, funded by
    its share of the emission (cycleEmission/2/num_consumers).
    """

    id: str
    product: Product


@dataclass(frozen=True)
class Configuration:
    cycle_emission: Tokens
    process_sheets: list[ProcessSheet]
    producer_configs: list[ProducingAgentConfig]
    consumers: list[ConsumerConfig] = field(default_factory=list)

    def validate(self) -> None:
        """Port of Configuration.Validate."""
        if self.cycle_emission <= 0:
            raise ValueError(f"cycle emission must be positive, got {self.cycle_emission}")

        process_products: set[Product] = set()
        process_capacities: set[CapacityType] = set()
        for sheet in self.process_sheets:
            if not sheet.require:
                raise ValueError(f"process sheet for product {sheet.product} has no capacity requirements")
            if sheet.product in process_products:
                raise ValueError(f"duplicate product {sheet.product} in process sheets")
            process_products.add(sheet.product)
            for in_prod, qty in sheet.inputs.items():
                if qty <= 0:
                    raise ValueError(f"input quantity must be positive, got {qty} for product {in_prod}")
            for cap_type, cap in sheet.require.items():
                if cap <= 0:
                    raise ValueError(f"capacity requirement must be positive, got {cap} for type {cap_type}")
                process_capacities.add(cap_type)

        for sheet in self.process_sheets:
            for in_prod in sheet.inputs:
                if in_prod not in process_products:
                    raise ValueError(
                        f"product {sheet.product} requires input product {in_prod} with no process sheet"
                    )

        producer_ids: set[str] = set()
        producer_cap_types: set[CapacityType] = set()
        for cfg in self.producer_configs:
            if cfg.id in producer_ids:
                raise ValueError(f"duplicate producer id {cfg.id}")
            producer_ids.add(cfg.id)
            if not cfg.capacities:
                raise ValueError(f"producer {cfg.id} owns no capacity type")
            for ct, cap in cfg.capacities.items():
                if cap <= 0:
                    raise ValueError(f"producer {cfg.id} capacity must be positive, got {cap} for type {ct}")
                producer_cap_types.add(ct)
            for inv_type in (cfg.restoration.type, cfg.upgrade.type):
                if inv_type not in cfg.capacities:
                    raise ValueError(f"producer {cfg.id} invests in unowned capacity type {inv_type}")

        for cap_type in process_capacities:
            if cap_type not in producer_cap_types:
                raise ValueError(f"capacity type {cap_type} required by process sheets but no producer provides it")

        # Eligibility (ADR-0025): a producer is eligible for a product only if it
        # owns *every* capacity type the recipe requires. A product whose required
        # bundle is split across producers has no eligible producer and can never
        # be ordered — reject here rather than crash later in env decode on an
        # empty eligible set.
        for sheet in self.process_sheets:
            required = set(sheet.require)
            if not any(required <= cfg.capacities.keys() for cfg in self.producer_configs):
                raise ValueError(
                    f"product {sheet.product} has no eligible producer: no single producer "
                    f"owns every required capacity type {sorted(required)}"
                )

        # Consumers (Phase 1 extension): products must have a process sheet, and
        # consumer ids must be unique and disjoint from producer ids — both share
        # the ordering-agent / env-agent namespace, so a collision would create
        # duplicate PettingZoo agents that double-step a single ordering agent.
        consumer_ids: set[str] = set()
        for c in self.consumers:
            if c.product not in process_products:
                raise ValueError(f"consumer {c.id} demands product {c.product} with no process sheet")
            if c.id in consumer_ids:
                raise ValueError(f"duplicate consumer id {c.id}")
            consumer_ids.add(c.id)
            if c.id in producer_ids:
                raise ValueError(f"consumer id {c.id} collides with producer id {c.id}")

    def process_sheet(self, product: Product) -> ProcessSheet:
        for ps in self.process_sheets:
            if ps.product == product:
                return ps
        raise KeyError(product)

    def inputs_of(self, product: Product) -> dict[Product, int]:
        """Direct input products (and quantities) the recipe consumes; empty for
        primal products."""
        return dict(self.process_sheet(product).inputs)

    def recipe_inputs_closure(self, product: Product) -> set[Product]:
        """All input products transitively reachable from `product` (excluding
        itself). Cycle-safe via a visited set — recipes may be circular across
        producers (ADR-0025)."""
        seen: set[Product] = set()
        stack = list(self.inputs_of(product))
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(self.inputs_of(p))
        return seen


def _load_producer(p: dict) -> ProducingAgentConfig:
    """Build a producer config. Accepts the new multi-type form
    (`"capacities": {type: cap}`) or the legacy single-type form
    (`"type"` + `"capacity"`); legacy investment defaults to the sole type."""
    if "capacities" in p:
        capacities = {k: v for k, v in p["capacities"].items()}
        default_type = None
    else:
        capacities = {p["type"]: p["capacity"]}
        default_type = p["type"]
    r, u = p["restoration"], p["upgrade"]
    exp = p.get("inventoryExpansion")
    return ProducingAgentConfig(
        id=p["id"],
        capacities=capacities,
        degradation=p["degradation"],
        restoration=Restoration(
            require=r["product"], restores=r["restores"], type=r.get("type", default_type), cost=r.get("cost")
        ),
        upgrade=Upgrade(
            require=u["product"], increases=u["capacity"], type=u.get("type", default_type), cost=u.get("cost")
        ),
        inventory_limit=p.get("inventoryLimit", 0),
        inventory_expansion=(
            InventoryExpansion(require=exp["product"], increases=exp["increases"], cost=exp.get("cost"))
            if exp else None
        ),
        replenish_cost_per_unit=p.get("replenishCostPerUnit"),
        max_replenish=p.get("maxReplenish"),
    )


def load_config(path: str | Path) -> Configuration:
    """Parse the JSON config (same schema as emulation/config.json, plus an
    optional top-level `consumers` array)."""
    data = json.loads(Path(path).read_text())
    sheets = [
        ProcessSheet(
            product=ps["product"],
            require={k: v for k, v in ps["require"].items()},
            # JSON object keys are strings; product ids are ints.
            inputs={int(k): v for k, v in ps.get("inputs", {}).items()},
        )
        for ps in data["processSheets"]
    ]
    producers = [_load_producer(p) for p in data["producerConfigs"]]
    consumers = [ConsumerConfig(id=c["id"], product=c["product"]) for c in data.get("consumers", [])]
    cfg = Configuration(
        cycle_emission=data["cycleEmission"],
        process_sheets=sheets,
        producer_configs=producers,
        consumers=consumers,
    )
    cfg.validate()
    return cfg
