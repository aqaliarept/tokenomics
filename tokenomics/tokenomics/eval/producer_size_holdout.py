"""Producer size-holdout audit helpers (EXP-0010D).

The audit runs the entity-factored producer planner on generated scale topologies
and reports controller quality separately from runtime/memory scaling. It also
checks the local row-order, product-relabel, and capacity-relabel gates for each
size so larger-row behavior is not judged only by one lucky episode return.
"""

from __future__ import annotations

import copy
import csv
import time
import tracemalloc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from ..domain import build_system
from ..domain.config import Configuration
from ..domain.producer_planner import plan_producer_command
from ..domain.producer_view import Relations
from ..env import TokenomicsAECEnv
from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from .runner import run_episode
from .scale import _stride_for_width, build_scale_config, permute_scale_config


@dataclass(frozen=True)
class ProducerSizeGates:
    row_order: bool
    product_relabel: bool
    capacity_relabel: bool


@dataclass(frozen=True)
class ProducerSizeAuditRow:
    products_per_layer: int
    producers_per_layer: int
    depth: int
    seed: int
    horizon: int
    consumer_completed_units: float
    dead: bool
    cycles: int
    elapsed_sec: float
    peak_kib: float
    row_order: bool
    product_relabel: bool
    capacity_relabel: bool


def audit_entity_planner_size_holdout(
    *,
    sizes: Iterable[int],
    seeds: Iterable[int],
    depth: int = 3,
    producers_per_layer: int = 4,
    num_consumers: int = 3,
    horizon: int = 100,
) -> list[ProducerSizeAuditRow]:
    """Run the accepted entity planner over product-count size holdouts.

    `sizes` maps to `products_per_layer`, so total process-sheet count is
    `depth * products_per_layer`. The producer controller is the EXP-0010B
    deterministic planner; EXP-0010C did not promote a learned scorer.
    """
    rows: list[ProducerSizeAuditRow] = []
    for size in sizes:
        for seed in seeds:
            cfg = build_scale_config(
                seed,
                depth=depth,
                products_per_layer=size,
                producers_per_layer=producers_per_layer,
                num_consumers=num_consumers,
            )
            gates = producer_size_gates(
                cfg,
                products_per_layer=size,
                producers_per_layer=producers_per_layer,
            )
            tracemalloc.start()
            start = time.perf_counter()
            stats = run_episode(
                TokenomicsAECEnv(config=cfg, horizon=horizon, entity_producer_obs=True),
                PRODUCING_BASELINES["entity_planner"],
                ORDERING_BASELINES["proportional"],
                seed=seed,
            )
            elapsed = time.perf_counter() - start
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append(
                ProducerSizeAuditRow(
                    products_per_layer=size,
                    producers_per_layer=producers_per_layer,
                    depth=depth,
                    seed=seed,
                    horizon=horizon,
                    consumer_completed_units=float(stats["consumer_completed_units"]),
                    dead=bool(stats["dead"]),
                    cycles=int(stats["cycles"]),
                    elapsed_sec=elapsed,
                    peak_kib=peak / 1024.0,
                    row_order=gates.row_order,
                    product_relabel=gates.product_relabel,
                    capacity_relabel=gates.capacity_relabel,
                )
            )
    return rows


def producer_size_gates(
    config: Configuration,
    *,
    products_per_layer: int,
    producers_per_layer: int,
) -> ProducerSizeGates:
    """Check local permutation/relabel gates for one generated scale topology."""
    pid = _composite_producer_id(config)
    base = _planned_for(config, pid)

    view = build_system(config).producer_entity_view(pid)
    row_order = base == _plan_view(_permute_entity_rows(view))

    product_perm = list(reversed(range(products_per_layer)))
    prod_identity = list(range(producers_per_layer))
    product_cfg = permute_scale_config(
        config,
        product_perm=product_perm,
        producer_perm=prod_identity,
    )
    inv_prod = _inverse_product_map(products_per_layer, product_perm)
    product_cmd = _planned_for(product_cfg, pid)
    product_relabel = (
        _unmap_replenishments(product_cmd.replenishments, inv_prod) == base.replenishments
        and product_cmd.do_restoration == base.do_restoration
        and product_cmd.do_upgrade == base.do_upgrade
        and product_cmd.do_inventory_expansion == base.do_inventory_expansion
    )

    capacity_cfg = _relabel_capacity_types(
        config,
        {f"capacity-{i}": f"renamed-capacity-{i}" for i in range(_depth(config))},
    )
    capacity_relabel = _planned_for(capacity_cfg, pid) == base

    return ProducerSizeGates(
        row_order=row_order,
        product_relabel=product_relabel,
        capacity_relabel=capacity_relabel,
    )


def write_size_audit_csv(rows: list[ProducerSizeAuditRow], path: str | Path) -> None:
    """Write size audit rows with quality and scaling columns side-by-side."""
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ProducerSizeAuditRow.__annotations__)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _planned_for(config: Configuration, producer_id: str):
    return _plan_view(build_system(config).producer_entity_view(producer_id))


def _plan_view(view):
    return plan_producer_command(
        view,
        max_replenish=10_000,
        replenish_cost_per_unit=None,
        funding_budget=None,
        inventory_expansion_increases=100,
        inventory_expansion_cost=0,
    )


def _composite_producer_id(config: Configuration) -> str:
    sheets = {s.product: s for s in config.process_sheets}
    for p in config.producer_configs:
        if any(sheets[x].inputs for x in p.produces):
            return p.id
    return config.producer_configs[0].id


def _permute_entity_rows(view):
    out = copy.deepcopy(view)
    cap_order = list(reversed(range(len(view.capacity_rows))))
    prod_order = list(reversed(range(len(view.product_rows))))
    input_order = list(reversed(range(len(view.input_rows))))
    cap_pos = {old: new for new, old in enumerate(cap_order)}
    prod_pos = {old: new for new, old in enumerate(prod_order)}
    input_pos = {old: new for new, old in enumerate(input_order)}

    out.capacity_rows = [out.capacity_rows[i] for i in cap_order]
    out.product_rows = [out.product_rows[i] for i in prod_order]
    out.input_rows = [out.input_rows[i] for i in input_order]
    out.relations = Relations(
        product_capacity_required={
            (prod_pos[xi], cap_pos[ti]): amount
            for (xi, ti), amount in view.relations.product_capacity_required.items()
        },
        product_input_required={
            (prod_pos[xi], input_pos[gi]): amount
            for (xi, gi), amount in view.relations.product_input_required.items()
        },
    )
    return out


def _inverse_product_map(products_per_layer: int, product_perm: list[int]) -> dict[int, int]:
    stride = _stride_for_width(products_per_layer)
    inv_slot = {new: old for old, new in enumerate(product_perm)}
    return {
        level * stride + new: level * stride + old
        for level in range(100)
        for new, old in inv_slot.items()
    }


def _unmap_replenishments(replenishments: dict[int, int], inverse: dict[int, int]) -> dict[int, int]:
    return {inverse.get(product, product): qty for product, qty in replenishments.items()}


def _depth(config: Configuration) -> int:
    return len({next(iter(s.require)).removeprefix("capacity-") for s in config.process_sheets})


def _relabel_capacity_types(config: Configuration, cap_map: dict[str, str]) -> Configuration:
    sheets = [
        replace(
            s,
            require={cap_map.get(t, t): amount for t, amount in s.require.items()},
        )
        for s in config.process_sheets
    ]
    producers = [
        replace(
            p,
            capacities={cap_map.get(t, t): amount for t, amount in p.capacities.items()},
            restoration=replace(p.restoration, type=cap_map.get(p.restoration.type, p.restoration.type)),
            upgrade=replace(p.upgrade, type=cap_map.get(p.upgrade.type, p.upgrade.type)),
        )
        for p in config.producer_configs
    ]
    out = replace(config, process_sheets=sheets, producer_configs=producers)
    out.validate()
    return out
