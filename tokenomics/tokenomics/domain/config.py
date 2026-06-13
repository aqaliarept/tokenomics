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
    """Recipe: product -> required capacity per type. Exogenous, fixed."""

    product: Product
    require: dict[CapacityType, Capacity]


@dataclass(frozen=True)
class Restoration:
    require: Product
    restores: Capacity


@dataclass(frozen=True)
class Upgrade:
    require: Product
    increases: Capacity


@dataclass(frozen=True)
class ProducingAgentConfig:
    id: str
    type: CapacityType
    capacity: Capacity
    degradation: int  # DegradationRate, percent of max capacity per cycle
    restoration: Restoration
    upgrade: Upgrade


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
            for cap_type, cap in sheet.require.items():
                if cap <= 0:
                    raise ValueError(f"capacity requirement must be positive, got {cap} for type {cap_type}")
                process_capacities.add(cap_type)

        producer_ids: set[str] = set()
        producer_cap_types: set[CapacityType] = set()
        for cfg in self.producer_configs:
            if cfg.id in producer_ids:
                raise ValueError(f"duplicate producer id {cfg.id}")
            producer_ids.add(cfg.id)
            if cfg.capacity <= 0:
                raise ValueError(f"producer {cfg.id} capacity must be positive, got {cfg.capacity}")
            producer_cap_types.add(cfg.type)

        for cap_type in process_capacities:
            if cap_type not in producer_cap_types:
                raise ValueError(f"capacity type {cap_type} required by process sheets but no producer provides it")

        # Consumers (Phase 1 extension): products must have a process sheet.
        for c in self.consumers:
            if c.product not in process_products:
                raise ValueError(f"consumer {c.id} demands product {c.product} with no process sheet")

    def process_sheet(self, product: Product) -> ProcessSheet:
        for ps in self.process_sheets:
            if ps.product == product:
                return ps
        raise KeyError(product)


def load_config(path: str | Path) -> Configuration:
    """Parse the JSON config (same schema as emulation/config.json, plus an
    optional top-level `consumers` array)."""
    data = json.loads(Path(path).read_text())
    sheets = [
        ProcessSheet(product=ps["product"], require={k: v for k, v in ps["require"].items()})
        for ps in data["processSheets"]
    ]
    producers = [
        ProducingAgentConfig(
            id=p["id"],
            type=p["type"],
            capacity=p["capacity"],
            degradation=p["degradation"],
            restoration=Restoration(require=p["restoration"]["product"], restores=p["restoration"]["restores"]),
            upgrade=Upgrade(require=p["upgrade"]["product"], increases=p["upgrade"]["capacity"]),
        )
        for p in data["producerConfigs"]
    ]
    consumers = [ConsumerConfig(id=c["id"], product=c["product"]) for c in data.get("consumers", [])]
    cfg = Configuration(
        cycle_emission=data["cycleEmission"],
        process_sheets=sheets,
        producer_configs=producers,
        consumers=consumers,
    )
    cfg.validate()
    return cfg
