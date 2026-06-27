"""Deterministic producer replacement audit (EXP-0011A).

Compare the accepted entity-factored deterministic producer planner against the
current v4 producer under the same stochastic product-stream champion orderer. This
is the gate for replacing only the producer side of the champion stack.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from ..env import TokenomicsAECEnv
from ..train.orderer_improve import _stochastic_closure
from ..env.spaces import build_spec
from ..train.smoke_ppo import checkpoint_obs_dims, load_order_actor, load_policies
from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from .orderer_oracle import ENVELOPE_COMBOS
from .runner import run_episode
from .scale import build_scale_config


@dataclass(frozen=True)
class ProducerChampionAuditRow:
    topology_seed: int
    eval_seeds: int
    horizon: int
    champion_completed_units: float
    champion_ci95: float
    entity_completed_units: float
    entity_ci95: float
    champion_dead_rate: float
    entity_dead_rate: float
    envelope_completed_units: float = float("nan")
    envelope_name: str = ""
    entity_vs_champion_delta: float = 0.0
    entity_vs_champion_ratio: float = float("nan")
    entity_vs_envelope_ratio: float = float("nan")


@dataclass(frozen=True)
class ProducerChampionAuditSummary:
    promotion_recommended: bool
    all_no_death: bool
    all_no_mean_regression: bool
    all_clear_envelope_mean: bool
    mean_delta: float
    rows: int


@dataclass(frozen=True)
class ProducerSizeTransferRow:
    products_per_layer: int
    topology_seed: int
    eval_seeds: int
    horizon: int
    checkpoint_compatible: bool
    checkpoint_completed_units: float
    checkpoint_dead_rate: float
    entity_completed_units: float
    entity_dead_rate: float
    checkpoint_prod_obs: int
    target_prod_obs: int
    checkpoint_prod_act: int
    target_prod_act: int
    checkpoint_order_obs: int
    target_order_obs: int
    checkpoint_order_act: int
    target_order_act: int


def audit_entity_planner_vs_champion(
    checkpoint: str,
    *,
    topology_seeds: Iterable[int],
    eval_seeds: int,
    horizon: int = 100,
    include_envelope: bool = True,
) -> list[ProducerChampionAuditRow]:
    """Score `entity_planner + champion orderer` against the current champion stack.

    The checkpoint is the product-stream stochastic champion artifact. Its producer is
    the current v4 producer; its orderer is deployed stochastically from the learned
    orderer head. Both candidates use product-stream ordering agents.
    """
    champion_prod, _ = load_policies(checkpoint)
    order_actor, bound = load_order_actor(checkpoint)
    champion_order = _stochastic_closure(order_actor, bound)

    rows: list[ProducerChampionAuditRow] = []
    for topology_seed in topology_seeds:
        cfg = build_scale_config(topology_seed, depth=3)
        champion_stats = []
        entity_stats = []
        for eval_seed in range(eval_seeds):
            champion_stats.append(
                run_episode(
                    TokenomicsAECEnv(config=cfg, horizon=horizon, product_stream_ordering=True),
                    champion_prod,
                    champion_order,
                    eval_seed,
                )
            )
            entity_stats.append(
                run_episode(
                    TokenomicsAECEnv(
                        config=cfg,
                        horizon=horizon,
                        product_stream_ordering=True,
                        entity_producer_obs=True,
                    ),
                    PRODUCING_BASELINES["entity_planner"],
                    champion_order,
                    eval_seed,
                )
            )

        champion_cu, champion_ci = _mean_ci(
            [s["consumer_completed_units"] for s in champion_stats]
        )
        entity_cu, entity_ci = _mean_ci(
            [s["consumer_completed_units"] for s in entity_stats]
        )
        envelope_cu = float("nan")
        envelope_name = ""
        if include_envelope:
            envelope_cu, envelope_name = _strict_envelope_units(
                cfg, eval_seeds=eval_seeds, horizon=horizon
            )
        delta = entity_cu - champion_cu
        rows.append(
            ProducerChampionAuditRow(
                topology_seed=topology_seed,
                eval_seeds=eval_seeds,
                horizon=horizon,
                champion_completed_units=round(champion_cu, 3),
                champion_ci95=round(champion_ci, 3),
                entity_completed_units=round(entity_cu, 3),
                entity_ci95=round(entity_ci, 3),
                champion_dead_rate=_dead_rate(champion_stats),
                entity_dead_rate=_dead_rate(entity_stats),
                envelope_completed_units=round(envelope_cu, 3) if not math.isnan(envelope_cu) else envelope_cu,
                envelope_name=envelope_name,
                entity_vs_champion_delta=round(delta, 3),
                entity_vs_champion_ratio=round(entity_cu / champion_cu, 4) if champion_cu else float("nan"),
                entity_vs_envelope_ratio=round(entity_cu / envelope_cu, 4) if envelope_cu and not math.isnan(envelope_cu) else float("nan"),
            )
        )
    return rows


def audit_producer_size_transfer(
    checkpoint: str,
    *,
    sizes: Iterable[int],
    topology_seed: int,
    eval_seeds: int = 1,
    horizon: int = 100,
    depth: int = 3,
    producers_per_layer: int = 4,
    num_consumers: int = 3,
) -> list[ProducerSizeTransferRow]:
    """Measure fixed-checkpoint compatibility and entity-planner runability by size.

    This is the architecture-transfer side of EXP-0011A. A fixed learned checkpoint is
    only runnable when its actor input/output widths match the target size. The entity
    planner is algorithmic over rows, so it is still evaluated on larger product sets
    under the deterministic `proportional` product-stream orderer.
    """
    ck_dims = checkpoint_obs_dims(checkpoint)
    champion_prod = champion_order = None
    rows: list[ProducerSizeTransferRow] = []
    for size in sizes:
        cfg = build_scale_config(
            topology_seed,
            depth=depth,
            products_per_layer=size,
            producers_per_layer=producers_per_layer,
            num_consumers=num_consumers,
        )
        target = build_spec(cfg, orderer_obs_version=3, producer_obs_version=1)
        compatible = (
            ck_dims["prod_obs"] == target.producing_obs_dim
            and ck_dims["prod_act"] == target.producing_action_dim
            and ck_dims["order_obs"] == target.ordering_obs_dim
            and ck_dims["order_act"] == target.ordering_action_dim
        )

        checkpoint_stats: list[dict] = []
        if compatible:
            if champion_prod is None or champion_order is None:
                champion_prod, _ = load_policies(checkpoint)
                order_actor, bound = load_order_actor(checkpoint)
                champion_order = _stochastic_closure(order_actor, bound)
            checkpoint_stats = [
                run_episode(
                    TokenomicsAECEnv(config=cfg, horizon=horizon, product_stream_ordering=True),
                    champion_prod,
                    champion_order,
                    seed,
                )
                for seed in range(eval_seeds)
            ]

        entity_stats = [
            run_episode(
                TokenomicsAECEnv(
                    config=cfg,
                    horizon=horizon,
                    product_stream_ordering=True,
                    entity_producer_obs=True,
                ),
                PRODUCING_BASELINES["entity_planner"],
                ORDERING_BASELINES["proportional"],
                seed,
            )
            for seed in range(eval_seeds)
        ]
        checkpoint_units = (
            mean(s["consumer_completed_units"] for s in checkpoint_stats)
            if checkpoint_stats
            else float("nan")
        )
        checkpoint_dead = _dead_rate(checkpoint_stats) if checkpoint_stats else float("nan")
        rows.append(
            ProducerSizeTransferRow(
                products_per_layer=size,
                topology_seed=topology_seed,
                eval_seeds=eval_seeds,
                horizon=horizon,
                checkpoint_compatible=compatible,
                checkpoint_completed_units=round(checkpoint_units, 3)
                if not math.isnan(checkpoint_units)
                else checkpoint_units,
                checkpoint_dead_rate=checkpoint_dead,
                entity_completed_units=round(
                    mean(s["consumer_completed_units"] for s in entity_stats), 3
                ),
                entity_dead_rate=_dead_rate(entity_stats),
                checkpoint_prod_obs=ck_dims["prod_obs"],
                target_prod_obs=target.producing_obs_dim,
                checkpoint_prod_act=ck_dims["prod_act"],
                target_prod_act=target.producing_action_dim,
                checkpoint_order_obs=ck_dims["order_obs"],
                target_order_obs=target.ordering_obs_dim,
                checkpoint_order_act=ck_dims["order_act"],
                target_order_act=target.ordering_action_dim,
            )
        )
    return rows


def summarize_producer_champion_audit(
    rows: list[ProducerChampionAuditRow],
) -> ProducerChampionAuditSummary:
    if not rows:
        raise ValueError("summarize_producer_champion_audit requires at least one row")
    all_no_death = all(r.champion_dead_rate == 0.0 and r.entity_dead_rate == 0.0 for r in rows)
    all_no_mean_regression = all(
        r.entity_completed_units >= r.champion_completed_units for r in rows
    )
    envelope_rows = [r for r in rows if not math.isnan(r.envelope_completed_units)]
    all_clear_envelope_mean = all(
        r.entity_completed_units >= r.envelope_completed_units for r in envelope_rows
    )
    if not envelope_rows:
        all_clear_envelope_mean = True
    promotion_recommended = all_no_death and all_no_mean_regression and all_clear_envelope_mean
    return ProducerChampionAuditSummary(
        promotion_recommended=promotion_recommended,
        all_no_death=all_no_death,
        all_no_mean_regression=all_no_mean_regression,
        all_clear_envelope_mean=all_clear_envelope_mean,
        mean_delta=mean(r.entity_vs_champion_delta for r in rows),
        rows=len(rows),
    )


def write_producer_champion_audit_csv(
    rows: list[ProducerChampionAuditRow], path: str | Path
) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ProducerChampionAuditRow.__annotations__)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_producer_size_transfer_csv(
    rows: list[ProducerSizeTransferRow], path: str | Path
) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ProducerSizeTransferRow.__annotations__)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _mean_ci(values: list[float]) -> tuple[float, float]:
    avg = mean(values)
    if len(values) <= 1:
        return avg, 0.0
    return avg, 1.96 * pstdev(values) / math.sqrt(len(values))


def _dead_rate(stats: list[dict]) -> float:
    return mean(1.0 if s["dead"] else 0.0 for s in stats)


def _strict_envelope_units(cfg, *, eval_seeds: int, horizon: int) -> tuple[float, str]:
    best_cu = -1.0
    best_name = ""
    for name, (prod_name, order_name) in ENVELOPE_COMBOS.items():
        values = [
            run_episode(
                TokenomicsAECEnv(config=cfg, horizon=horizon, product_stream_ordering=True),
                PRODUCING_BASELINES[prod_name],
                ORDERING_BASELINES[order_name],
                seed,
            )["consumer_completed_units"]
            for seed in range(eval_seeds)
        ]
        cu = mean(values)
        if cu > best_cu:
            best_cu = cu
            best_name = name
    return best_cu, best_name
