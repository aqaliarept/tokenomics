"""Pure Python port of the Go tokenomics domain (emulation/domain).

No RL concepts, no I/O. The env/ package adapts this to PettingZoo.
"""

from __future__ import annotations

from .common import ConsumerId, ProducerId
from .config import Configuration, ConsumerConfig, load_config
from .consumer import Consumer, ScheduledConsumer, ScriptedConsumer
from .ordering_agent import OrderingAgentCommand
from .producing_agent import ProducingAgentCommand
from .system import (
    CycleResult,
    SequentialOrderIdGenerator,
    System,
    SystemState,
)


def build_system(config: Configuration, consumers: dict | None = None) -> System:
    """Construct a System with a fresh id generator. If `consumers` is None,
    ScriptedConsumers are built from config.consumers."""
    if consumers is None:
        consumers = {c.id: ScriptedConsumer(c.id, c.product) for c in config.consumers}
    return System(SequentialOrderIdGenerator(), config, consumers)


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
    "SequentialOrderIdGenerator",
    "load_config",
    "build_system",
]
