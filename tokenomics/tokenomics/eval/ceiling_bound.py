"""Offline achievable-units upper bounds for archived curriculum tasks.

EXP-0033 asks a different question from Stage 4 training: is the current stack already
near the task family's offline ceiling, or did the myopic search bar hide headroom?  This
module computes analytic, dependency-free upper bounds from replayed ``Configuration``
objects:

* B0: scripted consumer demand over the same episode seeds as the observed matrix.
* B1: B0 plus a direct recipe capacity-throughput cap.
* B2: B1 with each demanded product's recipe expanded transitively through inputs.

Each tier is intentionally optimistic.  It aggregates capacity across producers, pins
capacity at an optimistic maximum, ignores TTL/inventory/funding/routing constraints, and
lets randomized consumers demand the cheapest product mix for each capacity type.  That
direction matters: these are upper bounds, so every tier must stay above observed
champion/bar units.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..domain.common import CapacityType, Product
from ..domain.config import Configuration
from .curriculum import TaskRecord, _chain_depth, _dims, _episode_seeds, replay
from .curriculum_replay import _load_archive

_OBSERVED_ROUNDING_TOL = 0.01


@dataclass(frozen=True)
class Bound:
    """One upper-bound tier in consumer-completed units."""

    tier: str
    units: float
    binding_constraint: str


def demanded_products(cfg: Configuration) -> set[Product]:
    """Products that scripted consumers may demand in seed-free fallback mode."""
    out: set[Product] = set()
    for consumer in cfg.consumers:
        out.update(consumer.product_pool)
    return out


def demand_cap_units(cfg: Configuration, horizon: int) -> float:
    """B0 fallback: maximum consumer units that can be requested over ``horizon`` cycles.

    Archive audits should prefer ``episode_demand``/``ceiling_bounds(..., episode_seeds=)``
    so the cap matches the deterministic schedules used for the observed matrix.
    """
    return float(horizon * sum(c.quantity_max for c in cfg.consumers))


def episode_demand(cfg: Configuration, horizon: int, seed: int) -> dict[Product, int]:
    """Demand quantities by product for the exact env consumer schedules at ``seed``.

    Mirrors ``TokenomicsAECEnv._build_consumers``: one RNG is shared across consumers,
    randomized consumers get a horizon-length product/quantity schedule, and deterministic
    consumers repeat their fixed schedule.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    demand: dict[Product, int] = {}
    for consumer in cfg.consumers:
        if not consumer.products and consumer.quantity_min == 1 and consumer.quantity_max == 1:
            demand[consumer.product] = demand.get(consumer.product, 0) + horizon
            continue
        pool = consumer.product_pool
        if consumer.is_randomized:
            for _ in range(horizon):
                product = pool[int(rng.integers(len(pool)))]
                quantity = int(rng.integers(consumer.quantity_min, consumer.quantity_max + 1))
                demand[product] = demand.get(product, 0) + quantity
        else:
            demand[pool[0]] = demand.get(pool[0], 0) + horizon * consumer.quantity_min
    return demand


def capacity_cycles_by_type(
    cfg: Configuration,
    horizon: int,
    *,
    include_upgrade_slack: bool = True,
) -> dict[CapacityType, float]:
    """Optimistic aggregate capacity-cycles available by type.

    Base capacity is pinned at each producer's initial max every cycle.  When
    ``include_upgrade_slack`` is true, upgrades are relaxed to be free and instant, with
    at most one upgrade per producer per cycle.  This is deliberately loose, but it keeps
    the tier an upper bound even for policies that expand max capacity during the run.
    """
    caps: dict[CapacityType, float] = {}
    for producer in cfg.producer_configs:
        for cap_type, cap in producer.capacities.items():
            caps[cap_type] = caps.get(cap_type, 0.0) + float(cap) * horizon
        if include_upgrade_slack and producer.upgrade.type in producer.capacities:
            caps[producer.upgrade.type] = caps.get(producer.upgrade.type, 0.0) + (
                float(producer.upgrade.increases) * horizon * horizon
            )
    return caps


def recipe_capacity_bundle(
    cfg: Configuration,
    product: Product,
    *,
    transitive: bool,
) -> dict[CapacityType, float]:
    """Capacity required to deliver one unit of ``product`` under a relaxed recipe model.

    Direct mode returns only the product's own recipe requirements (B1).  Transitive mode
    also expands input products and charges their capacity bundles (B2).  If a cyclic
    recipe is encountered, the repeated edge is ignored; that is an optimistic relaxation,
    so it stays on the upper-bound side.
    """
    visiting: set[Product] = set()

    def expand(p: Product) -> dict[CapacityType, float]:
        if p in visiting:
            return {}
        visiting.add(p)
        sheet = cfg.process_sheet(p)
        bundle = {t: float(v) for t, v in sheet.require.items()}
        if transitive:
            for input_product, qty in sheet.inputs.items():
                for cap_type, amount in expand(input_product).items():
                    bundle[cap_type] = bundle.get(cap_type, 0.0) + amount * qty
        visiting.discard(p)
        return bundle

    return expand(product)


def _type_unit_bound(
    available: float,
    costs_by_product: dict[Product, float],
    demand: dict[Product, int],
) -> float:
    """Max delivered units allowed by one capacity type under fixed product demand.

    This is a fractional knapsack relaxation: serve zero/cheap-cost demanded products
    first.  It can overstate the true mix-constrained optimum, but not understate it.
    """
    delivered = 0.0
    remaining = available
    for product, cost in sorted(costs_by_product.items(), key=lambda item: item[1]):
        qty = float(demand.get(product, 0))
        if qty <= 0.0:
            continue
        if cost <= 0.0:
            delivered += qty
            continue
        take = min(qty, remaining / cost)
        delivered += take
        remaining -= take * cost
        if remaining <= 1e-12:
            break
    return delivered


def _capacity_bound_for_demand(
    cfg: Configuration,
    horizon: int,
    *,
    transitive: bool,
    demand: dict[Product, int],
    include_upgrade_slack: bool = True,
) -> Bound:
    caps = capacity_cycles_by_type(cfg, horizon, include_upgrade_slack=include_upgrade_slack)
    products = set(demand)
    bundles = {
        product: recipe_capacity_bundle(cfg, product, transitive=transitive)
        for product in products
    }
    demand_cap = float(sum(demand.values()))

    best_units = math.inf
    binding = "capacity:none"
    for cap_type, available in caps.items():
        costs = {product: bundle.get(cap_type, 0.0) for product, bundle in bundles.items()}
        positive = [c for c in costs.values() if c > 0.0]
        if not positive:
            continue
        units = _type_unit_bound(available, costs, demand)
        if units < best_units:
            best_units = units
            binding = f"capacity:{cap_type}"

    if demand_cap <= best_units:
        return Bound("B2" if transitive else "B1", demand_cap, "demand")
    return Bound("B2" if transitive else "B1", best_units, binding)


def _mean_bound(tier: str, bounds: list[Bound]) -> Bound:
    units = statistics.mean(b.units for b in bounds)
    bindings = Counter(b.binding_constraint for b in bounds)
    binding = bindings.most_common(1)[0][0]
    if len(bindings) > 1:
        binding = "mixed:" + ",".join(f"{k}={v}" for k, v in sorted(bindings.items()))
    return Bound(tier, units, binding)


def ceiling_bounds(
    cfg: Configuration,
    horizon: int,
    *,
    episode_seeds: Iterable[int] | None = None,
    include_upgrade_slack: bool = True,
) -> dict[str, Bound]:
    """Return B0/B1/B2 upper bounds for ``cfg`` over ``horizon`` cycles.

    With ``episode_seeds`` the bounds are the mean of the exact deterministic consumer
    schedules for those seeds, matching the curriculum matrix convention.  Without seeds,
    a conservative seed-free max-demand cap is used.

    ``include_upgrade_slack`` (default True, EXP-0033 byte-identical) relaxes upgrades to
    free/instant, so the capacity tiers stay above any policy that expands max capacity.
    The Track A capacity-binding pre-flight sets it False: base capacity only, so a cell
    whose *base* throughput cannot meet demand (investment skill required) reads as
    capacity-bound instead of the structurally-demand-bound slack tier.
    """
    if episode_seeds is None:
        demand = {product: int(demand_cap_units(cfg, horizon)) for product in demanded_products(cfg)}
        b0 = Bound("B0", demand_cap_units(cfg, horizon), "demand")
        b1_raw = _capacity_bound_for_demand(cfg, horizon, transitive=False, demand=demand,
                                            include_upgrade_slack=include_upgrade_slack)
        b2_raw = _capacity_bound_for_demand(cfg, horizon, transitive=True, demand=demand,
                                            include_upgrade_slack=include_upgrade_slack)
        b1 = Bound(
            "B1",
            min(b0.units, b1_raw.units),
            "demand" if b0.units <= b1_raw.units else b1_raw.binding_constraint,
        )
        b2 = Bound(
            "B2",
            min(b0.units, b2_raw.units),
            "demand" if b0.units <= b2_raw.units else b2_raw.binding_constraint,
        )
        return {"B0": b0, "B1": b1, "B2": b2}

    b0s: list[Bound] = []
    b1s: list[Bound] = []
    b2s: list[Bound] = []
    for seed in episode_seeds:
        demand = episode_demand(cfg, horizon, seed)
        b0s.append(Bound("B0", float(sum(demand.values())), "demand"))
        b1s.append(_capacity_bound_for_demand(cfg, horizon, transitive=False, demand=demand,
                                              include_upgrade_slack=include_upgrade_slack))
        b2s.append(_capacity_bound_for_demand(cfg, horizon, transitive=True, demand=demand,
                                              include_upgrade_slack=include_upgrade_slack))
    b0 = _mean_bound("B0", b0s)
    b1 = _mean_bound("B1", b1s)
    b2 = _mean_bound("B2", b2s)
    return {"B0": b0, "B1": b1, "B2": b2}


def load_observed_units(path: str | Path) -> dict[str, dict[str, float]]:
    """Read EXP-0028 ``curriculum_matrix.csv`` champion/bar units by task id."""
    out: dict[str, dict[str, float]] = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            task_id = row["task_id"]
            out[task_id] = {
                "champion_units": float(row["learner_units"]),
                "bar_units": float(row["reference_units"]),
            }
    return out


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    return round(numerator / denominator, 4)


def _bound_rows(
    records: Iterable[TaskRecord],
    observed: dict[str, dict[str, float]],
    *,
    horizon: int,
    eval_seeds: int,
) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        if rec.task_id not in observed:
            raise ValueError(f"observed matrix has no row for archived task {rec.task_id!r}")
        cfg = replay(rec)
        products, producers, caps = _dims(cfg)
        seeds = _episode_seeds(rec.episode_seed, eval_seeds)
        bounds = ceiling_bounds(cfg, horizon, episode_seeds=seeds)
        obs = observed[rec.task_id]
        champion = obs["champion_units"]
        bar = obs["bar_units"]
        best_observed = max(champion, bar)
        params = rec.world_spec.get("params", {})
        for tier in ("B0", "B1", "B2"):
            bound = bounds[tier]
            rows.append({
                "task_id": rec.task_id,
                "tier": tier,
                "bound_units": round(bound.units, 3),
                "binding_constraint": bound.binding_constraint,
                "champion_units": round(champion, 3),
                "bar_units": round(bar, 3),
                "best_observed_units": round(best_observed, 3),
                "champion_bound_ratio": _ratio(champion, bound.units),
                "bar_bound_ratio": _ratio(bar, bound.units),
                "best_observed_bound_ratio": _ratio(best_observed, bound.units),
                "upper_bound_ok": bound.units + _OBSERVED_ROUNDING_TOL >= best_observed,
                "b0_units": round(bounds["B0"].units, 3),
                "b1_units": round(bounds["B1"].units, 3),
                "b2_units": round(bounds["B2"].units, 3),
                "b1_binding": bounds["B1"].binding_constraint,
                "b2_binding": bounds["B2"].binding_constraint,
                "horizon": horizon,
                "eval_seeds": eval_seeds,
                "episode_seeds": ",".join(str(seed) for seed in seeds),
                "world_seed": rec.world_seed,
                "episode_seed": rec.episode_seed,
                "generator": rec.world_spec.get("generator", ""),
                "depth": params.get("depth", _chain_depth(cfg)),
                "products_per_layer": params.get("products_per_layer"),
                "producers_per_layer": params.get("producers_per_layer"),
                "target_products": products,
                "target_producers": producers,
                "target_capacity_types": caps,
                "target_depth": _chain_depth(cfg),
            })
    return rows


def _write_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty ceiling-bound summary")
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tier_stats(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tier in ("B0", "B1", "B2"):
        vals = [float(r["champion_bound_ratio"]) for r in rows if r["tier"] == tier]
        bindings: dict[str, int] = {}
        for r in rows:
            if r["tier"] != tier:
                continue
            key = str(r["binding_constraint"])
            bindings[key] = bindings.get(key, 0) + 1
        out[tier] = {
            "mean_champion_bound_ratio": round(statistics.mean(vals), 4) if vals else None,
            "min_champion_bound_ratio": round(min(vals), 4) if vals else None,
            "max_champion_bound_ratio": round(max(vals), 4) if vals else None,
            "rows_at_or_above_90pct": sum(v >= 0.90 for v in vals),
            "rows_at_or_above_95pct": sum(v >= 0.95 for v in vals),
            "binding_counts": bindings,
        }
    return out


def _verdict(stats: dict[str, dict]) -> str:
    b2 = stats["B2"]
    mean_ratio = b2["mean_champion_bound_ratio"] or 0.0
    min_ratio = b2["min_champion_bound_ratio"] or 0.0
    if mean_ratio >= 0.90 and min_ratio >= 0.80:
        return "near_bound"
    if mean_ratio <= 0.80:
        return "hidden_headroom_or_loose_bound"
    return "ambiguous_consider_b3"


def _emit_result(out: Path, summary: dict, rows: list[dict]) -> None:
    b2 = summary["tier_stats"]["B2"]
    worst = min((r for r in rows if r["tier"] == "B2"),
                key=lambda r: float(r["champion_bound_ratio"]))
    lines = [
        "# EXP-0033 - Offline Achievable-Units Upper Bound", "",
        "Ceiling audit over the EXP-0028 archived tasks.  B0/B1/B2 are analytic",
        "upper bounds, not learned policies: demand cap, direct capacity cap, and",
        "transitive recipe-capacity cap respectively.", "",
        "## Result", "",
        f"- Tasks audited: {summary['tasks']}",
        f"- B2 mean champion/bound ratio: {b2['mean_champion_bound_ratio']}",
        f"- B2 min champion/bound ratio: {b2['min_champion_bound_ratio']}",
        f"- B2 rows >= 90%: {b2['rows_at_or_above_90pct']}/{summary['tasks']}",
        f"- B2 binding counts: {json.dumps(b2['binding_counts'], sort_keys=True)}",
        f"- Upper-bound invariant failures: {summary['upper_bound_failures']}",
        f"- Verdict: `{summary['verdict']}`", "",
        "Worst B2 row:", "",
        "```json",
        json.dumps(worst, indent=2),
        "```", "",
        "## Interpretation", "",
        "If B2 is tight enough to be informative, ratios near 0.90-0.95 mean the",
        "champion is already close to the offline ceiling on this archived distribution.",
        "Low ratios would justify a stronger exact/relaxed oracle before more expert",
        "iteration.  B3 temporal LP was not run here; the B0-B2 arithmetic already",
        "provides the dependency-free ceiling audit requested by the handoff.", "",
    ]
    (out / "RESULT.md").write_text("\n".join(lines))


def run_ceiling_audit(
    archive_path: str | Path,
    curriculum_matrix: str | Path,
    out_dir: str | Path,
    *,
    horizon: int = 100,
    eval_seeds: int = 3,
    check_invariants: bool = True,
) -> dict:
    """Compute EXP-0033 B0/B1/B2 rows and write the experiment layout."""
    records = _load_archive(archive_path)
    if not records:
        raise ValueError(f"archive has no task records: {archive_path}")
    observed = load_observed_units(curriculum_matrix)
    rows = _bound_rows(records, observed, horizon=horizon, eval_seeds=eval_seeds)
    failures = [r for r in rows if not r["upper_bound_ok"]]
    if check_invariants and failures:
        bad = failures[:5]
        raise ValueError(f"{len(failures)} upper-bound invariant failures; examples={bad}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out / "summary.csv")

    stats = _tier_stats(rows)
    summary = {
        "stage": "EXP-0033",
        "title": "offline achievable-units upper bound",
        "archive": str(archive_path),
        "curriculum_matrix": str(curriculum_matrix),
        "horizon": horizon,
        "eval_seeds": eval_seeds,
        "tasks": len(records),
        "rows": len(rows),
        "tiers": ["B0", "B1", "B2"],
        "tier_stats": stats,
        "upper_bound_failures": len(failures),
        "verdict": _verdict(stats),
    }
    (out / "spec.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _emit_result(out, summary, rows)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="EXP-0033 offline achievable-units upper bound")
    ap.add_argument("--archive", required=True, help="mined task_archive.jsonl")
    ap.add_argument("--curriculum-matrix", required=True,
                    help="curriculum_matrix.csv with champion/bar units for the archive")
    ap.add_argument("--out-dir", required=True, help="experiment directory to write")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--eval-seeds", type=int, default=3,
                    help="number of contiguous episode seeds used in the observed matrix")
    ap.add_argument("--no-invariant-check", action="store_true",
                    help="write rows even if a tier undercuts observed units")
    args = ap.parse_args()

    summary = run_ceiling_audit(
        args.archive,
        args.curriculum_matrix,
        args.out_dir,
        horizon=args.horizon,
        eval_seeds=args.eval_seeds,
        check_invariants=not args.no_invariant_check,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
