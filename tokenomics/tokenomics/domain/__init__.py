"""Pure Python port of the Go tokenomics domain (emulation/domain).

No RL concepts, no I/O. The env/ package adapts this to PettingZoo.
"""

from __future__ import annotations

from .common import ConsumerId, ProducerId
from .config import Configuration, ConsumerConfig, load_config
from .consumer import Consumer, ScheduledConsumer, ScriptedConsumer
from .ordering_agent import OrderingAgentCommand
from .producing_agent import ProducingAgentCommand
from .shocks import ShockEvent
from .system import (
    ConsumerOrderOutcome,
    CapacitySnapshot,
    CycleResult,
    ProducerAuctionRecord,
    SequentialOrderIdGenerator,
    System,
    SystemState,
    product_stream_agents,
)


def build_system(
    config: Configuration,
    consumers: dict | None = None,
    product_stream_ordering: bool = False,
    shock_schedule: tuple[ShockEvent, ...] = (),
) -> System:
    """Construct a System with a fresh id generator. If `consumers` is None,
    ScriptedConsumers are built from config.consumers, or a ScheduledConsumer where
    the config carries an explicit `schedule` (pure data, design D8).
    `product_stream_ordering` keys ordering agents by (source, product) per ADR-0052.
    `shock_schedule` is the env-drawn schedule the domain replays (ADR-0055)."""
    if consumers is None:
        consumers = {
            c.id: ScheduledConsumer(c.id, list(c.schedule)) if c.schedule
            else ScriptedConsumer(c.id, c.product)
            for c in config.consumers
        }
    return System(
        SequentialOrderIdGenerator(), config, consumers,
        product_stream_ordering=product_stream_ordering,
        shock_schedule=shock_schedule,
    )


__all__ = [
    "Configuration",
    "ConsumerConfig",
    "Consumer",
    "ConsumerId",
    "ProducerId",
    "ScriptedConsumer",
    "ScheduledConsumer",
    "OrderingAgentCommand",
    "ProducingAgentCommand",
    "System",
    "SystemState",
    "CycleResult",
    "CapacitySnapshot",
    "ConsumerOrderOutcome",
    "ProducerAuctionRecord",
    "SequentialOrderIdGenerator",
    "load_config",
    "build_system",
    "product_stream_agents",
]
