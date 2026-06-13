"""Consumer: exogenous demand source. Port of emulation/domain/consumer.go
(interface) plus a Phase 1 ScriptedConsumer (the Go emulator left consumers
unimplemented — see application/emulator.go empty consumer map).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .common import ConsumerId, Product, Tokens
from .requests import ConsumerRequest


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
