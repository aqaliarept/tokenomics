"""Deterministic producer planner over ProducerEntityView (EXP-0010B).

This is the non-learned controller slice: transparent local scores are computed
from the entity-factored view, then converted into the existing
ProducingAgentCommand. The planner owns feasibility gates that are local to one
producer command: inventory headroom, per-cycle replenish ceiling, optional local
funding budget, and already-running investment streams. The domain auction and
fund allocator remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import Product
from .producer_view import ProducerEntityView
from .producing_agent import ProducingAgentCommand

DEFAULT_INPUT_STOCK_TARGET = 3


@dataclass(frozen=True)
class ProducerLocalScores:
    """Transparent scores the deterministic planner uses.

    Scores are keyed by entity ids only where the output command needs an id
    (`replenishments`). Capacity investment scores stay scalar because the current
    command has one configured restoration and one configured upgrade toggle.
    """

    product_urgency: dict[Product, float] = field(default_factory=dict)
    input_value: dict[Product, float] = field(default_factory=dict)
    restoration_roi: float = 0.0
    upgrade_roi: float = 0.0
    inventory_expansion_value: float = 0.0


def score_producer_entity_view(
    view: ProducerEntityView, *, input_stock_target: int = DEFAULT_INPUT_STOCK_TARGET
) -> ProducerLocalScores:
    """Compute local, explainable planner scores from a producer entity view."""
    product_urgency = {
        r.product: float(r.open_demand_qty) * max(1.0, r.bottleneck_ratio)
        for r in view.product_rows
        if r.eligible > 0.0 and r.open_demand_qty > 0
    }
    input_value = {}
    for r in view.input_rows:
        direct_pressure = r.downstream_demand_pressure + r.blocked_reservation_qty
        stock_deficit = max(0, input_stock_target - (r.inventory + r.outstanding_inbound))
        value = direct_pressure + stock_deficit
        if value > 0:
            input_value[r.input_product] = float(value)

    restoration_roi = 0.0
    upgrade_roi = 0.0
    for r in view.capacity_rows:
        pressure = max(r.demand_pressure, r.previous_requested_capacity)
        shortage = max(0, pressure - r.free)
        restore_need = max(0.0, 1.0 - r.ratio) + shortage / max(1, r.current)
        if r.restore_effect > 0:
            restoration_roi = max(
                restoration_roi,
                restore_need * r.restore_effect / max(1, r.restore_cost),
            )
        if r.upgrade_effect > 0 and pressure > r.max:
            upgrade_roi = max(
                upgrade_roi,
                (pressure - r.max) * r.upgrade_effect / max(1, r.upgrade_cost),
            )

    inventory_expansion_value = (
        sum(input_value.values()) if view.globals.inventory_headroom <= 0 else 0.0
    )

    return ProducerLocalScores(
        product_urgency=product_urgency,
        input_value=input_value,
        restoration_roi=restoration_roi,
        upgrade_roi=upgrade_roi,
        inventory_expansion_value=inventory_expansion_value,
    )


def plan_producer_command(
    view: ProducerEntityView,
    *,
    max_replenish: int,
    replenish_cost_per_unit: int | float | None = None,
    funding_budget: int | float | None = None,
    restore_threshold: float = 0.60,
    inventory_expansion_increases: int = 0,
    inventory_expansion_cost: int | float = 0,
    input_stock_target: int = DEFAULT_INPUT_STOCK_TARGET,
) -> ProducingAgentCommand:
    """Plan one producer command from the entity view.

    `funding_budget` is optional because the current system fund is shared across
    producers. When supplied, the planner spends it conservatively in the domain's
    priority order: restoration, replenishment, upgrade, then inventory expansion.
    """
    scores = score_producer_entity_view(view, input_stock_target=input_stock_target)
    remaining_budget = None if funding_budget is None else float(max(0, funding_budget))

    restore = False
    restore_cost = _first_positive((r.restore_cost for r in view.capacity_rows), default=0)
    if view.globals.active_investments[0] == 0.0:
        restore = any(
            r.restore_effect > 0 and (r.ratio < restore_threshold or r.demand_pressure > r.free)
            for r in view.capacity_rows
        )
        if restore and not _spendable(remaining_budget, restore_cost):
            restore = False
        if restore and remaining_budget is not None:
            remaining_budget -= max(0, restore_cost)

    replenishments = _plan_replenishments(
        view,
        scores,
        max_replenish=max_replenish,
        replenish_cost_per_unit=replenish_cost_per_unit,
        funding_budget=remaining_budget,
        input_stock_target=input_stock_target,
    )
    if remaining_budget is not None and replenish_cost_per_unit:
        remaining_budget -= sum(replenishments.values()) * float(replenish_cost_per_unit)

    upgrade = False
    upgrade_cost = _first_positive((r.upgrade_cost for r in view.capacity_rows), default=0)
    if view.globals.active_investments[1] == 0.0 and scores.upgrade_roi > 0.0:
        upgrade = _spendable(remaining_budget, upgrade_cost)
        if upgrade and remaining_budget is not None:
            remaining_budget -= max(0, upgrade_cost)

    expand_inventory = False
    if (
        view.globals.active_investments[2] == 0.0
        and inventory_expansion_increases > 0
        and scores.inventory_expansion_value > 0.0
    ):
        expand_inventory = _spendable(remaining_budget, inventory_expansion_cost)

    return ProducingAgentCommand(
        do_restoration=restore,
        do_upgrade=upgrade,
        do_inventory_expansion=expand_inventory,
        replenishments=replenishments,
    )


def _plan_replenishments(
    view: ProducerEntityView,
    scores: ProducerLocalScores,
    *,
    max_replenish: int,
    replenish_cost_per_unit: int | float | None,
    funding_budget: float | None,
    input_stock_target: int,
) -> dict[Product, int]:
    headroom = max(0, view.globals.inventory_headroom)
    if headroom <= 0 or max_replenish <= 0:
        return {}
    if funding_budget is not None and replenish_cost_per_unit:
        headroom = min(headroom, int(funding_budget // float(replenish_cost_per_unit)))
    if headroom <= 0:
        return {}

    wants: list[tuple[float, Product, int]] = []
    for r in view.input_rows:
        target = max(
            input_stock_target,
            r.downstream_demand_pressure + r.blocked_reservation_qty,
        )
        have = r.inventory + r.outstanding_inbound
        want = min(max_replenish, max(0, target - have))
        if want > 0:
            wants.append((scores.input_value.get(r.input_product, 0.0), r.input_product, want))
    if not wants:
        return {}

    # Spend by local value. Product id is only a final deterministic tie-breaker; ties
    # with scarce headroom are economically identical under this local view.
    wants.sort(key=lambda x: (-x[0], x[1]))
    out: dict[Product, int] = {}
    remaining = headroom
    for _, product, want in wants:
        if remaining <= 0:
            break
        qty = min(want, remaining)
        out[product] = qty
        remaining -= qty
    return out


def _spendable(budget: float | None, cost: int | float) -> bool:
    return budget is None or cost <= 0 or budget >= cost


def _first_positive(values, *, default: int = 0) -> int:
    for value in values:
        if value > 0:
            return int(value)
    return default
