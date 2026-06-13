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


class InvestmentType(IntEnum):
    UPGRADE = 0
    RESTORATION = 1


@dataclass
class InvestmentRequest:
    producer_id: ProducerId
    type: InvestmentType
    product: Product
    cut_off_price: CapacityUnitPrice


@dataclass(frozen=True)
class CapacityValue:
    type: CapacityType
    power: Capacity


@dataclass(frozen=True)
class Request:
    tokens: Tokens
    capacities: list[CapacityValue] = field(default_factory=list)
