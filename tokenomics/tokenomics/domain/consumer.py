"""Consumer: exogenous demand source. Port of emulation/domain/consumer.go
(interface) plus a Phase 1 ScriptedConsumer (the Go emulator left consumers
unimplemented — see application/emulator.go empty consumer map).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .common import ConsumerId, Product, Tokens
from .requests import ConsumerRequest

# One cycle's scripted demand: the product wanted and how many units.
ScheduleEntry = tuple[Product, int]


@runtime_checkable
class Consumer(Protocol):
    def id(self) -> ConsumerId: ...
    def order(self) -> list[ConsumerRequest]: ...
    def emit(self, tokens: Tokens) -> None: ...


class ScriptedConsumer:
    """Places one fixed-product order per cycle, spending its whole balance.

    Balance accrues from emission (system.emit) and rejection refunds
    (system.complete_cycle calls emit(remaining) on a rejected consumer order).
    """

    def __init__(self, consumer_id: ConsumerId, product: Product) -> None:
        self._id = consumer_id
        self._product = product
        self._balance: Tokens = 0

    def id(self) -> ConsumerId:
        return self._id

    def emit(self, tokens: Tokens) -> None:
        self._balance += tokens

    def order(self) -> list[ConsumerRequest]:
        if self._balance <= 0:
            return []
        req = ConsumerRequest(self._id, self._product, self._balance)
        self._balance = 0
        return [req]


class ScheduledConsumer:
    """Replays a per-cycle (product, quantity) schedule, spending its whole balance
    on each cycle's scheduled demand (B9 seeded randomization).

    Pure data — the *env* layer generates the seeded schedule and injects it, so the
    domain stays RNG-free (CLAUDE.md). The cycle index advances every `order()` call
    (even when the balance is empty) so the schedule stays aligned to the simulation
    cycle; it wraps when exhausted, so a schedule shorter than the horizon repeats.
    """

    def __init__(self, consumer_id: ConsumerId, schedule: list[ScheduleEntry]) -> None:
        if not schedule:
            raise ValueError("ScheduledConsumer needs a non-empty schedule")
        self._id = consumer_id
        self._schedule = schedule
        self._cycle = 0
        self._balance: Tokens = 0

    def id(self) -> ConsumerId:
        return self._id

    def emit(self, tokens: Tokens) -> None:
        self._balance += tokens

    def order(self) -> list[ConsumerRequest]:
        product, quantity = self._schedule[self._cycle % len(self._schedule)]
        self._cycle += 1
        if self._balance <= 0:
            return []
        req = ConsumerRequest(self._id, product, self._balance, quantity=quantity)
        self._balance = 0
        return [req]
