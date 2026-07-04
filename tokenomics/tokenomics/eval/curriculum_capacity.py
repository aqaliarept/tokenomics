"""Stage 4b Track A: capacity-binding mining (CURRICULUM_GAME_MASTER_APPROACH.md Track A).

Stage 4 hit a real fixed point on a demand-saturated family (EXP-0031/0032b/0033): the
scale grid is over-capacitated, so a near-perfect score is structurally easy and there is
nothing left to distill. Track A reruns the Stage-3 machinery where *capacity*, not
demand, is the ceiling — the only regime with headroom by construction.

The pure, torch-free piece that makes the regime "capacity-binding" is the ceiling
pre-flight: before mining a cell, compute its offline upper bound and keep it only when
capacity binds below demand. Crucially this uses the NO-UPGRADE-SLACK tier
(`include_upgrade_slack=False`). The EXP-0033 slack tier relaxes upgrades to free/instant
(`increases * horizon^2` capacity), which is optimistic enough that demand is *always* the
binding constraint — so under the slack tier every cell reads demand-bound and Track A
would mine nothing. The no-slack tier instead asks whether *base* capacity can meet demand;
a cell where it cannot is exactly one where investment skill (not luck of a saturated grid)
separates a strong policy from a weak one — a mineable policy weakness.

This module only selects/labels cells. The actual mining reuses `curriculum_mining`
(`grid_pool` + `mine_hard_cases`) unchanged, and the recoverability gate reuses
`gap_attribution`. Keeping this layer free of torch means the pre-flight and the shipped
grid are unit-testable without a checkpoint.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .ceiling_bound import ceiling_bounds
from .curriculum import REPRODUCERS, _episode_seeds, write_archive
from .curriculum_replay import _load_archive

# The pre-flight always uses the base-capacity (no free upgrade) tier; see module docstring.
_PREFLIGHT_SLACK = False


def cell_ceiling(generator: str, params: dict, *, world_seeds: Iterable[int],
                 eval_seeds: int, horizon: int) -> dict:
    """No-slack B0 (demand) vs B2 (capacity) bound for a cell, ACROSS every world seed.

    `build_scale_config` draws each producer's capacity per WORLD seed (the strong/weak
    rng split), so total capacity — and therefore whether capacity binds — varies seed to
    seed. Mining then sweeps a whole world-seed range per cell, so a single-seed pre-flight
    would certify a cell capacity-bound at seed 0 while saturated (demand-bound) tasks at
    other seeds leak into the archive — the exact failure this gate exists to stop. So the
    cell is bounded at EVERY `world_seed`, each with its own episode-demand schedule
    (`_episode_seeds(world_seed, eval_seeds)`, mirroring `curriculum_mining._evaluate`).

    Reported: `b2_over_b0_worst` is the LOOSEST seed's capacity/demand ratio (the seed most
    likely to be saturated); a cell binds everywhere iff that worst ratio is below the
    threshold. `binds_frac` is the fraction of world seeds that bind, so a partial-bind cell
    is visible in the report rather than silently dropped."""
    if generator not in REPRODUCERS:
        raise KeyError(f"unknown generator {generator!r} (not a vetted reproducer)")
    world_seeds = list(world_seeds)
    b0s: list[float] = []
    b2s: list[float] = []
    ratios: list[float] = []
    bind_flags: list[bool] = []
    worst_ratio = 0.0
    worst_binding = "demand"
    for ws in world_seeds:
        cfg = REPRODUCERS[generator](ws, **params)
        bounds = ceiling_bounds(cfg, horizon, episode_seeds=_episode_seeds(ws, eval_seeds),
                                include_upgrade_slack=_PREFLIGHT_SLACK)
        b0, b2 = bounds["B0"].units, bounds["B2"].units
        ratio = b2 / b0 if b0 > 0 else 1.0
        b0s.append(b0); b2s.append(b2); ratios.append(ratio)
        bind_flags.append(b2 < b0 - 1e-9)
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_binding = bounds["B2"].binding_constraint
    n = len(world_seeds)
    return {
        "generator": generator,
        "params": dict(params),
        "n_world_seeds": n,
        "b0_units_mean": round(sum(b0s) / n, 2),
        "b2_units_mean": round(sum(b2s) / n, 2),
        "b2_over_b0_worst": round(worst_ratio, 4),
        "b2_over_b0_best": round(min(ratios), 4),
        "capacity_headroom_frac_worst": round(1.0 - worst_ratio, 4),
        "worst_binding_constraint": worst_binding,
        "binds_frac": round(sum(bind_flags) / n, 4),
        "capacity_binds": False,  # decided by preflight_cells against its margin
    }


def preflight_cells(generator: str, cells: list[dict], *, world_seeds: Iterable[int],
                    eval_seeds: int, horizon: int,
                    margin: float = 0.0) -> tuple[list[dict], list[dict]]:
    """Keep only cells where capacity binds below demand at EVERY world seed (Stage-3 amendment).

    A cell survives when the loosest world seed's capacity ceiling still sits a `margin`
    fraction below demand (`b2_over_b0_worst < 1 - margin`) — i.e. capacity binds across the
    whole world-seed range the mine will sweep, so no saturated task leaks in. `margin=0.0`
    keeps any cell that strictly binds everywhere; a positive margin demands that much
    headroom so near-ties are dropped. Returns `(kept_cells, report_rows)` — kept preserves
    input order and identity so survivors feed straight into `curriculum_mining.grid_pool`;
    report_rows carries the bound telemetry for every candidate for the RESULT.md table."""
    kept: list[dict] = []
    rows: list[dict] = []
    for cell in cells:
        row = cell_ceiling(generator, cell, world_seeds=world_seeds,
                           eval_seeds=eval_seeds, horizon=horizon)
        binds = row["b2_over_b0_worst"] < (1.0 - margin) - 1e-9
        row["capacity_binds"] = binds
        rows.append(row)
        if binds:
            kept.append(cell)
    return kept, rows


# --- recoverability gate (Stage-3 amendment) -----------------------------------------
# A mined policy_weakness enters the archive only if a cheap cross-stack probe attributes
# the gap to a TRAINABLE component. This reuses `gap_attribution`'s producer/orderer swap:
# its `dominant_channel` already folds the swap deltas into one label. A gap no single
# component swap closes (joint-bar / interaction mismatch) is unlearnable by component
# distill — exactly the EXP-0031 (5.27% closure) / EXP-0032b dead ends — so it is dropped.

_RECOVERABLE_CHANNELS = frozenset({"producer", "producer_partial", "orderer", "mixed"})


def is_recoverable(attribution_row: dict, *, min_component_gain: float = 0.5) -> bool:
    """True when a gap_attribution row's gap attributes to a trainable component.

    Two conditions: the swap named a component channel (not `no_positive_gap` /
    `interaction_or_noise` / `not_producer`), AND that component actually moved the task by
    at least `min_component_gain` units (the noise floor that keeps a 0.1u attribution
    fluke out of the archive).

    Run gap_attribution WITH the orderer cross (`--orderer-cross`): `not_producer` is only
    emitted when the orderer swap was skipped, so an orderer-recoverable task would then be
    silently dropped as unverified. With the cross on, an orderer-closable gap surfaces as
    `orderer`/`mixed` and is kept."""
    if attribution_row.get("dominant_channel") not in _RECOVERABLE_CHANNELS:
        return False
    prod = attribution_row.get("producer_only_gain") or 0.0
    ordr = attribution_row.get("orderer_only_gain") or 0.0
    return max(prod, ordr) >= min_component_gain


def recoverable_task_ids(attribution_rows: list[dict], *,
                         min_component_gain: float = 0.5) -> list[str]:
    """Task ids of the component-recoverable subset of a gap_attribution matrix (order kept)."""
    return [r["task_id"] for r in attribution_rows
            if is_recoverable(r, min_component_gain=min_component_gain)]


def _read_attribution_csv(path: str | Path) -> list[dict]:
    """Read a `gap_attribution` summary.csv, casting the gain columns to float.

    CSV stores everything as strings and a skipped orderer cross as empty; cast so the
    recoverability gate's numeric floor works on real rows the same as synthetic ones."""
    rows: list[dict] = []
    with Path(path).open(newline="") as fh:
        for r in csv.DictReader(fh):
            for k in ("producer_only_gain", "orderer_only_gain"):
                v = r.get(k)
                r[k] = float(v) if v not in (None, "") else 0.0
            rows.append(r)
    return rows


def filter_archive_recoverable(archive_path: str | Path, attribution_csv: str | Path,
                               out_path: str | Path, *,
                               min_component_gain: float = 0.5) -> tuple[int, int]:
    """Write the component-recoverable subset of a mined archive (Stage-3 amendment).

    Joins the mined `task_archive.jsonl` to a `gap_attribution` matrix by task id and keeps
    only tasks whose gap a producer/orderer swap closes. Returns `(total, kept)`. A mined
    task with no attribution row is conservatively dropped (unverified recoverability)."""
    records = _load_archive(archive_path)
    keep = set(recoverable_task_ids(_read_attribution_csv(attribution_csv),
                                    min_component_gain=min_component_gain))
    kept = [r for r in records if r.task_id in keep]
    write_archive(kept, out_path)
    return len(records), len(kept)


def write_preflight_report(generator: str, cells: list[dict], out_path: str | Path, *,
                           world_seeds: Iterable[int], eval_seeds: int, horizon: int,
                           margin: float = 0.0) -> list[dict]:
    """Run the ceiling pre-flight over `cells`, write the report CSV, return surviving cells.

    The report tables the per-cell bound telemetry (mean B0/B2, worst-seed capacity/demand
    ratio, bind fraction, capacity_binds) for every candidate cell — the RESULT.md evidence
    for which cells were mined and which were dropped as saturated (at any world seed).
    Returns the capacity-bound survivors to hand to `curriculum_mining --grid`."""
    kept, rows = preflight_cells(generator, cells, world_seeds=world_seeds,
                                 eval_seeds=eval_seeds, horizon=horizon, margin=margin)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["generator", "params", "n_world_seeds", "b0_units_mean", "b2_units_mean",
              "b2_over_b0_worst", "b2_over_b0_best", "capacity_headroom_frac_worst",
              "worst_binding_constraint", "binds_frac", "capacity_binds"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "params": json.dumps(r["params"], sort_keys=True)})
    return kept


def capacity_tight_cells() -> list[dict]:
    """The shipped Track A grid: capacity-bound scale cells below the current envelope.

    Axes (doc): producer density below the envelope (`producers_per_layer` 2-3, vs the
    default 4), raised demand (`quantity_max`), tightened capacity (`capacity_scale` < 1),
    and one harsher-decay cell (`degradation`). Every cell is tuned to bind at EVERY world
    seed of the default mining range (`--world-seeds 20-119`) — not just seed 0 — so no
    saturated task leaks into the archive, while staying alive enough that a reference bar
    can out-perform a weak learner rather than everything dying to a wall. Distinct params
    give the mining novelty term distinct descriptor cells to spread across (EXP-0025)."""
    return [
        {"depth": 3, "producers_per_layer": 3, "quantity_max": 14, "capacity_scale": 0.26},
        {"depth": 3, "producers_per_layer": 3, "quantity_max": 16, "capacity_scale": 0.30},
        {"depth": 3, "producers_per_layer": 2, "quantity_max": 14, "capacity_scale": 0.35},
        {"depth": 3, "producers_per_layer": 3, "quantity_max": 20, "capacity_scale": 0.30,
         "degradation": 9},
    ]


def primal_tight_cells() -> list[dict]:
    """Track A round-2 grid: cells whose ceiling binds at the PRIMAL (layer-0) source.

    Round 1 (EXP-0034) mined cells whose bottleneck sat at the mid/late layers, so the
    augment-replay producer learned to spend upgrades there and traded away the champion's
    "upgrade the tight primal source first" skill — the s54 blind-tail regression (−107u).
    These cells tighten ONLY layer-0 (`primal_capacity_scale` well below `capacity_scale`,
    upper layers left loose) so the no-slack B2 ceiling binds at `capacity-0` and the
    cheapest lever is the primal upgrade. Every cell binds at EVERY world seed of the
    default `--world-seeds 20-119` range and ~all seeds bind at layer 0 (verified via the
    pre-flight). Mining these ALONGSIDE `capacity_tight_cells()` gives round-2 replay both
    the mid/late- and primal-bottleneck regimes, so it can learn both skills."""
    return [
        {"depth": 3, "producers_per_layer": 3, "quantity_max": 14,
         "capacity_scale": 0.60, "primal_capacity_scale": 0.22},
        {"depth": 3, "producers_per_layer": 3, "quantity_max": 16,
         "capacity_scale": 0.60, "primal_capacity_scale": 0.25},
        {"depth": 3, "producers_per_layer": 2, "quantity_max": 14,
         "capacity_scale": 0.60, "primal_capacity_scale": 0.30},
        {"depth": 3, "producers_per_layer": 3, "quantity_max": 20,
         "capacity_scale": 0.60, "primal_capacity_scale": 0.25, "degradation": 9},
    ]


def main() -> None:
    """Track A local glue (cheap, torch-free; the mine + gap_attribution steps run on the pod).

    Pipeline: `preflight` selects the capacity-bound grid to mine (emits a `--grid` JSON for
    `curriculum_mining`), then (after mining + `gap_attribution` on the pod) `filter-recoverable`
    keeps only the component-recoverable mined tasks. Both ends are pure CSV/JSON I/O."""
    import argparse

    ap = argparse.ArgumentParser(description="Curriculum Stage 4b Track A: capacity-binding mining")
    sub = ap.add_subparsers(dest="mode", required=True)

    pf = sub.add_parser("preflight", help="ceiling pre-flight: keep capacity-bound cells")
    pf.add_argument("--generator", default="build_scale_config")
    pf.add_argument("--grid", default=None,
                    help="JSON list of param-dict cells; default = capacity_tight_cells()")
    pf.add_argument("--world-seeds", default="20-119",
                    help="inclusive WORLD-seed range LO-HI to certify (match the mine's "
                         "--seeds; a cell is kept only if capacity binds at every one)")
    pf.add_argument("--eval-seeds", type=int, default=3,
                    help="episode-demand seeds per world seed (match the mine's --eval-seeds)")
    pf.add_argument("--horizon", type=int, default=100)
    pf.add_argument("--margin", type=float, default=0.0,
                    help="min capacity headroom below demand to keep a cell (0 = strict bind)")
    pf.add_argument("--report", default=None, help="write the pre-flight table CSV here")
    pf.add_argument("--emit-grid", default=None,
                    help="write the surviving cells as a curriculum_mining --grid JSON here")

    fr = sub.add_parser("filter-recoverable",
                        help="keep only component-recoverable mined tasks (needs gap_attribution)")
    fr.add_argument("--archive", required=True, help="mined task_archive.jsonl")
    fr.add_argument("--attribution", required=True, help="gap_attribution summary.csv")
    fr.add_argument("--out", required=True, help="recoverable subset archive.jsonl to write")
    fr.add_argument("--min-component-gain", type=float, default=0.5)
    args = ap.parse_args()

    if args.mode == "preflight":
        if args.generator not in REPRODUCERS:
            ap.error(f"unknown generator {args.generator!r}; available: {','.join(REPRODUCERS)}")
        lo, hi = (int(x) for x in args.world_seeds.split("-"))
        world_seeds = range(lo, hi + 1)
        cells = json.loads(args.grid) if args.grid else capacity_tight_cells()
        report = args.report or (Path(args.emit_grid).with_name("preflight.csv")
                                 if args.emit_grid else "preflight.csv")
        kept = write_preflight_report(args.generator, cells, report, world_seeds=world_seeds,
                                      eval_seeds=args.eval_seeds, horizon=args.horizon,
                                      margin=args.margin)
        grid_json = json.dumps(kept)
        if args.emit_grid:
            Path(args.emit_grid).write_text(grid_json + "\n")
        print(f"preflight: {len(kept)}/{len(cells)} cells capacity-bound; report -> {report}")
        print(f"--grid {grid_json}")
        return

    total, kept = filter_archive_recoverable(
        args.archive, args.attribution, args.out, min_component_gain=args.min_component_gain)
    print(f"recoverable: kept {kept}/{total} mined tasks -> {args.out}")


if __name__ == "__main__":
    main()
