"""Producer acceptance policies (ADR-0061, add-producer-acceptance-policy D3 / D5).

Each is an `AcceptanceRule`: it reads only the producer's own `AcceptanceView` (its routed bids,
budget, capacity, carried claims, inventory and inbound) and returns one `AcceptanceDecision` per
bid, in the view's (price-descending) order. Set on the env as `env.acceptance_rule`; None keeps
the domain's own auction path (ADR-0025), byte-identical.

- `auction` (D3): ADR-0025's rule as a policy: accept iff the bundle fits the budget left,
  price-first; a decline keeps today's reason (no-fit vs displaced).
- `ttl_feasible` (A1): accept also a bid that does not fit now when the capacity it claims is
  free in the next cycle, after the claims already queued (pending ones and this cycle's
  accepted-later ones); a producer cannot see the order's age, so one future cycle is assumed.
- `ttl_inputs` (A2): A1, and decline a composite whose inputs are neither in stock nor in flight
  beyond what carried claims need (the orderer's makeable check, duplicated on the producer side).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..domain.common import Capacity, CapacityType, Product
from ..domain.producing_agent import (REJECT_DECLINED, REJECT_DISPLACED, REJECT_NO_FIT,
                                      AcceptanceDecision, AcceptanceView, Bid)

FUTURE_CYCLES = 1  # A1 / A2: cycles after this one a claim may still execute in (order age unseen)


def _fits(budget: dict[CapacityType, Capacity], caps: dict[CapacityType, Capacity]) -> bool:
    return all(budget.get(t, 0) >= c for t, c in caps.items())


def _take(budget: dict[CapacityType, Capacity], caps: dict[CapacityType, Capacity]) -> None:
    for t, c in caps.items():
        budget[t] = budget.get(t, 0) - c


def acceptance_auction(view: AcceptanceView) -> list[AcceptanceDecision]:
    budget, start = dict(view.available), dict(view.available)
    out = []
    for bid in view.bids:
        if _fits(budget, bid.capacities):
            _take(budget, bid.capacities)
            out.append(AcceptanceDecision(True))
        else:
            out.append(AcceptanceDecision(
                False, reason=REJECT_NO_FIT if not _fits(start, bid.capacities)
                else REJECT_DISPLACED))
    return out


def _inputs_reachable(view: AcceptanceView, committed: dict[Product, int], bid: Bid) -> bool:
    """A composite's inputs are in stock or in flight, beyond what claims ahead of it need."""
    for g, q in bid.inputs.items():
        have = view.inventory.get(g, 0) + view.outstanding_inbound.get(g, 0) - committed.get(g, 0)
        if have < q:
            return False
    return True


def _ttl_rule(check_inputs: bool) -> Callable[[AcceptanceView], list[AcceptanceDecision]]:
    def rule(view: AcceptanceView) -> list[AcceptanceDecision]:
        now, start = dict(view.available), dict(view.available)
        later = {t: c * FUTURE_CYCLES for t, c in view.capacities.items()}
        for b in view.pending:
            _take(later, b.capacities)
        committed: dict[Product, int] = {}
        for b in (*view.held, *view.pending):
            for g, q in b.inputs.items():
                committed[g] = committed.get(g, 0) + q
        out = []
        for bid in view.bids:
            if check_inputs and bid.is_composite() and not _inputs_reachable(view, committed, bid):
                out.append(AcceptanceDecision(False, reason=REJECT_DECLINED))
                continue
            if _fits(now, bid.capacities):
                _take(now, bid.capacities)
            elif _fits(later, bid.capacities):
                _take(later, bid.capacities)
            else:
                out.append(AcceptanceDecision(
                    False, reason=REJECT_NO_FIT if not _fits(start, bid.capacities)
                    else REJECT_DISPLACED))
                continue
            for g, q in bid.inputs.items():
                committed[g] = committed.get(g, 0) + q
            out.append(AcceptanceDecision(True))
        return out
    return rule


acceptance_ttl_feasible = _ttl_rule(False)
acceptance_ttl_inputs = _ttl_rule(True)

ACCEPTANCE_POLICIES: dict[str, Callable[[AcceptanceView], Sequence[AcceptanceDecision]]] = {
    "auction": acceptance_auction,
    "ttl_feasible": acceptance_ttl_feasible,
    "ttl_inputs": acceptance_ttl_inputs,
}
