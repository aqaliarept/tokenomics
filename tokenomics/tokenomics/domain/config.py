"""Configuration model + loader.

Port of emulation/domain/configuration.go and producing_agent.go config types.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .common import Capacity, CapacityType, Product, Tokens
from .shocks import ProducerFacts, ShockModel, shock_model_from_dict, validate_shock_model


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
    # products this producer is eligible to make (ADR-0028). Authoritative: the
    # ordering side reads this set, not capacity ownership. validate() guarantees a
    # producer only lists products whose required capacity types it owns.
    produces: frozenset[Product] = frozenset()
    inventory_limit: int = 0  # max input-product units held (ADR-0025); 0 = none
    inventory_expansion: InventoryExpansion | None = None
    # fund target per replenished unit (ADR-0027); None = unbounded
    replenish_cost_per_unit: Tokens | None = None
    # per-input per-cycle replenishment ceiling (ADR-0025 B5); None = inventory_limit
    max_replenish: int | None = None


@dataclass(frozen=True)
class ConsumerConfig:
    """Phase 1 scripted demand. Not present in the Go config — added here.

    Each consumer places one order every cycle, funded by its share of the emission
    (cycleEmission/2/num_consumers). By default the order is the fixed `product` at
    quantity 1. B9 seeded randomization: when `products` lists >1 product or
    `quantity_max > quantity_min`, the env samples a per-episode (product, quantity)
    schedule from its seeded RNG (the domain itself stays RNG-free, CLAUDE.md).

    An explicit `schedule` (design D8, add-measurement-gate-harness) takes precedence
    over both: the consumer replays it cyclically, one entry per cycle, in every
    episode regardless of the episode seed, and the env ignores `product` and the B9
    bounds. A calm world sets it so its demand carries no stochastic machinery.
    """

    id: str
    product: Product
    # demand-randomization bounds (B9). Empty `products` -> fixed `product`; default
    # quantity range [1, 1] -> fixed quantity 1 (canonical, deterministic).
    products: tuple[Product, ...] = ()
    quantity_min: int = 1
    quantity_max: int = 1
    # explicit cyclic (product, quantity) demand (design D8). Empty -> B9/fixed path.
    schedule: tuple[tuple[Product, int], ...] = ()

    @property
    def product_pool(self) -> tuple[Product, ...]:
        """Products the consumer may demand: the distinct products of an explicit
        `schedule` (first-appearance order), else the `products` list, else the single
        fixed `product`. Product streams (ADR-0052) and validation read this."""
        if self.schedule:
            return tuple(dict.fromkeys(product for product, _ in self.schedule))
        return self.products if self.products else (self.product,)

    @property
    def is_randomized(self) -> bool:
        """True iff the demand needs the seeded RNG — a multi-product pool or a
        non-degenerate quantity range. A fixed single product at fixed quantity is
        deterministic and keeps the canonical ScriptedConsumer path. An explicit
        `schedule` is fixed demand (D8), never seeded, so it is never randomized."""
        if self.schedule:
            return False
        return len(self.product_pool) > 1 or self.quantity_max > self.quantity_min


@dataclass(frozen=True)
class DemandDrift:
    """Smooth demand drift (ADR-0055, add-shocks-i-drift D4): between consecutive cycles at
    most floor(j * N) of the N B9-randomized consumers change their order. The env draws
    it; this is the parameter only."""

    j: float

    def __post_init__(self) -> None:
        # An exact int j becomes float, so equal configs (j=1 vs the JSON loader's 1.0)
        # serialize and fingerprint identically; bool and other types are left for validate.
        if type(self.j) is int:
            object.__setattr__(self, "j", float(self.j))


@dataclass(frozen=True)
class Configuration:
    cycle_emission: Tokens
    process_sheets: list[ProcessSheet]
    producer_configs: list[ProducingAgentConfig]
    consumers: list[ConsumerConfig] = field(default_factory=list)
    # Supply shocks (ADR-0055, add-shocks-i-drift D1): explicit events and/or seeded
    # generators. None = no shocks; the fingerprint then omits the field (D5).
    shocks: ShockModel | None = None
    # Demand drift (D4) over the B9-randomized consumers. None = per-cycle independent
    # B9 sampling, unchanged; the fingerprint then omits the field (D5).
    demand_drift: DemandDrift | None = None

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
            # Funding targets / ceilings, when set, must be positive (ADR-0027/0030):
            # a zero/negative target funds nothing or corrupts apportionment.
            costs = [("restoration", cfg.restoration.cost), ("upgrade", cfg.upgrade.cost)]
            if cfg.inventory_expansion is not None:
                costs.append(("inventory expansion", cfg.inventory_expansion.cost))
            for label, c in costs:
                if c is not None and c <= 0:
                    raise ValueError(f"producer {cfg.id} {label} cost must be positive, got {c}")
            if cfg.replenish_cost_per_unit is not None and cfg.replenish_cost_per_unit <= 0:
                raise ValueError(
                    f"producer {cfg.id} replenish_cost_per_unit must be positive, "
                    f"got {cfg.replenish_cost_per_unit}"
                )
            if cfg.max_replenish is not None and cfg.max_replenish <= 0:
                raise ValueError(
                    f"producer {cfg.id} max_replenish must be positive, got {cfg.max_replenish}"
                )

        for cap_type in process_capacities:
            if cap_type not in producer_cap_types:
                raise ValueError(f"capacity type {cap_type} required by process sheets but no producer provides it")

        # Eligibility is authoritative (ADR-0028): each producer *declares* the
        # products it makes via `produces`. A producer may only list a product whose
        # recipe it can satisfy — it must own every capacity type the recipe
        # requires — and every process-sheet product must be produced by at least
        # one producer, else it can never be ordered (and env decode would later
        # crash on an empty eligible set).
        produced: set[Product] = set()
        for cfg in self.producer_configs:
            for prod in cfg.produces:
                if prod not in process_products:
                    raise ValueError(f"producer {cfg.id} produces product {prod} with no process sheet")
                missing = set(self.process_sheet(prod).require) - cfg.capacities.keys()
                if missing:
                    raise ValueError(
                        f"producer {cfg.id} produces product {prod} but does not own required "
                        f"capacity type(s) {sorted(missing)}"
                    )
                produced.add(prod)
        for sheet in self.process_sheets:
            if sheet.product not in produced:
                raise ValueError(
                    f"product {sheet.product} has no producer: no producer produces it"
                )

        # Consumers (Phase 1 extension): products must have a process sheet, and
        # consumer ids must be unique and disjoint from producer ids — both share
        # the ordering-agent / env-agent namespace, so a collision would create
        # duplicate PettingZoo agents that double-step a single ordering agent.
        consumer_ids: set[str] = set()
        for c in self.consumers:
            # Every product the consumer may demand (fixed or randomized pool, B9)
            # must have a process sheet, else it can never be produced.
            for prod in c.product_pool:
                if prod not in process_products:
                    raise ValueError(f"consumer {c.id} demands product {prod} with no process sheet")
            for prod, qty in c.schedule:
                if qty < 1:
                    raise ValueError(
                        f"consumer {c.id} schedule quantity must be >= 1, got {qty} for product {prod}"
                    )
            if c.quantity_min < 1:
                raise ValueError(f"consumer {c.id} quantity_min must be >= 1, got {c.quantity_min}")
            if c.quantity_max < c.quantity_min:
                raise ValueError(
                    f"consumer {c.id} quantity_max {c.quantity_max} < quantity_min {c.quantity_min}"
                )
            if c.id in consumer_ids:
                raise ValueError(f"duplicate consumer id {c.id}")
            consumer_ids.add(c.id)
            if c.id in producer_ids:
                raise ValueError(f"consumer id {c.id} collides with producer id {c.id}")

        if self.demand_drift is not None:
            j = self.demand_drift.j
            if isinstance(j, bool) or not isinstance(j, (int, float)) or not math.isfinite(j) \
                    or not 0.0 <= j <= 1.0:
                raise ValueError(f"demand drift j must be a number in [0, 1], got {j!r}")

        # Shocks (design D1): kind table, start cycle >= 2, unbuilt kinds rejected by name.
        if self.shocks is not None:
            validate_shock_model(self.shocks, {
                cfg.id: ProducerFacts(frozenset(cfg.capacities), cfg.produces)
                for cfg in self.producer_configs
            })

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
        # JSON `produces` is an array of int product ids; absent -> makes nothing.
        produces=frozenset(p.get("produces", [])),
        inventory_limit=p.get("inventoryLimit", 0),
        inventory_expansion=(
            InventoryExpansion(require=exp["product"], increases=exp["increases"], cost=exp.get("cost"))
            if exp else None
        ),
        replenish_cost_per_unit=p.get("replenishCostPerUnit"),
        max_replenish=p.get("maxReplenish"),
    )


def _load_drift(d: dict) -> DemandDrift:
    """`DemandDrift` itself turns an integer j into a float (one fingerprint per value);
    anything else is kept for validation to reject."""
    return DemandDrift(j=d["j"])


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
    consumers = [
        ConsumerConfig(
            id=c["id"],
            product=c["product"],
            products=tuple(c.get("products", ())),
            quantity_min=c.get("quantityMin", 1),
            quantity_max=c.get("quantityMax", 1),
            # JSON `schedule` is an array of [product, quantity] pairs (design D8).
            schedule=tuple((product, quantity) for product, quantity in c.get("schedule", ())),
        )
        for c in data.get("consumers", [])
    ]
    cfg = Configuration(
        cycle_emission=data["cycleEmission"],
        process_sheets=sheets,
        producer_configs=producers,
        consumers=consumers,
        # optional top-level `shocks` object (design D1); absent -> no shocks.
        shocks=shock_model_from_dict(data["shocks"]) if "shocks" in data else None,
        # optional top-level `demandDrift: {"j": ...}` (design D4); absent -> no drift.
        demand_drift=_load_drift(data["demandDrift"]) if "demandDrift" in data else None,
    )
    cfg.validate()
    return cfg
