"""Request and value types shared across orders, producers, consumers.

ConsumerRequest (consumer.go), InvestmentRequest/InvestmentType
(producing_agent.go), Request/CapacityValue (ordering_agent.go). Centralized
here to avoid import cycles in the Python port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .common import (
    Capacity,
    CapacityType,
    CapacityUnitPrice,
    ConsumerId,
    Product,
    ProducerId,
    Tokens,
)


@dataclass(frozen=True)
class ConsumerRequest:
    consumer_id: ConsumerId
    product: Product
    tokens: Tokens
    quantity: int = 1  # units of `product` demanded as a whole (ADR-0025)


class InvestmentType(IntEnum):
    UPGRADE = 0
    RESTORATION = 1
    INVENTORY_EXPANSION = 2


@dataclass
class InvestmentRequest:
    producer_id: ProducerId
    type: InvestmentType
    product: Product
    cut_off_price: CapacityUnitPrice
    # funding target: tokens the fund fills this order to before advancing to the
    # next priority (ADR-0027). None = unbounded (take whole remaining budget).
    cost: Tokens | None = None


@dataclass
class ReplenishmentRequest:
    """A producer buying `quantity` units of an input `product` into its
    inventory (ADR-0025). Funded by the central fund at priority 1 — distinct from
    investment so the survival-first ordering holds. Not an investment."""

    producer_id: ProducerId
    product: Product
    quantity: int
    cut_off_price: CapacityUnitPrice
    cost: Tokens | None = None  # total funding target (per-unit × quantity), ADR-0027


@dataclass(frozen=True)
class CapacityValue:
    type: CapacityType
    power: Capacity


@dataclass(frozen=True)
class Request:
    tokens: Tokens
    capacities: list[CapacityValue] = field(default_factory=list)
