"""Champion-iteration driver (EXP-0007B).

Automates one full `champion -> candidate -> gate -> record` loop on the stable
product-stream ordering contract (ADR-0052, EXP-0007A), orchestrating the existing
orderer expert-iteration pieces rather than replacing them:

    load champion registry -> preflight champion checkpoint -> train candidate ->
    evaluate candidate + champion on held-out suite -> gate -> write experiment record
    -> emit PROMOTE_RECOMMENDED

Promotion stays human-approved: the driver never overwrites the champion or the registry.
Real training is Runpod-only (CLAUDE.md); the orchestration takes injected train/eval
callables so the loop, gates, and record-writing are testable without cloud work.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Masked decode-inactive orderer dims keep this log_std init (dagger_orderer default);
# a learned stochastic head grows some decode-active dims above it (EXP-0006B).
_MASKED_LOG_STD_INIT = -2.0

# Registry fields without a sensible default — absent ⇒ the registry is unusable.
_REQUIRED = (
    "champion_id",
    "checkpoint",
    "orderer_obs_version",
    "checkpoint_version",
    "held_out_topology_seeds",
)


@dataclass
class ChampionRegistry:
    """The active champion + gate configuration — one source of truth for the driver
    and promotion audit."""

    champion_id: str
    checkpoint: str
    orderer_obs_version: int
    checkpoint_version: int
    held_out_topology_seeds: list[int]
    producer: str = "v4"
    # Checkpoint paths are relative to `repo_root`, which the production Runpod path sets to
    # the synced package dir (`tokenomics/` -> `/workspace/tokenomics`). So the champion +
    # producer are staged *inside* that dir (`m2_inputs/`, like EXP-0007A) — repo-root
    # `experiments/...` paths are NOT in the rsync payload and would fail preflight on the pod.
    producer_checkpoint: str = "m2_inputs/d3_rung_dagger_v4.pt"
    product_stream: bool = True
    depth: int = 3
    hard_cells: list[int] = field(default_factory=lambda: [6, 9])
    eval_episode_seeds: int = 40
    strict_envelope: bool = True
    primary_metric: str = "consumer_completed_units"
    max_death_rate: float = 0.0

    @classmethod
    def load(cls, path: "str | Path") -> "ChampionRegistry":
        """Parse + validate a registry JSON. Missing required field ⇒ `ValueError`
        naming it, so the driver fails before launching any cloud work."""
        data = json.loads(Path(path).read_text())
        for key in _REQUIRED:
            if key not in data:
                raise ValueError(f"champion registry missing required field: {key!r}")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class PreflightResult:
    """Outcome of validating a champion checkpoint against the registry contract.
    `ok` gates the run; `reasons` lists every blocking mismatch (fail fast, before cloud)."""

    ok: bool
    reasons: list[str]


def check_checkpoint(ckpt: dict, registry: ChampionRegistry) -> PreflightResult:
    """Validate a loaded checkpoint against the registry's contract (preflight, EXP-0007B):
    schema version, obs fingerprint, and a *learned* stochastic head — so an eval cannot
    mistake a stale, wrong-obs, or collapsed-deterministic orderer for the champion.

    Pure over the checkpoint dict (no disk IO) so each failure mode is unit-testable;
    `preflight()` is the thin disk-loading wrapper."""
    from ..env.spaces import config_fingerprint
    from ..eval.scale import build_scale_config

    reasons: list[str] = []
    if ckpt.get("version") != registry.checkpoint_version:
        reasons.append(
            f"checkpoint schema version {ckpt.get('version')!r} != "
            f"registry {registry.checkpoint_version}"
        )
    expected_fp = config_fingerprint(
        build_scale_config(0, depth=registry.depth), registry.orderer_obs_version
    )
    if ckpt.get("config_fingerprint") != expected_fp:
        reasons.append(
            f"checkpoint fingerprint {ckpt.get('config_fingerprint')!r} != "
            f"obs-v{registry.orderer_obs_version} contract {expected_fp!r}"
        )
    sigma = np.exp(np.asarray(ckpt["order"]["log_std"], dtype=np.float64))
    if int((sigma > np.exp(_MASKED_LOG_STD_INIT) + 1e-3).sum()) == 0:
        reasons.append(
            "orderer head is collapsed (no log_std dim above the init floor) — "
            "not the learned stochastic champion the registry declares"
        )
    return PreflightResult(not reasons, reasons)


@dataclass
class GateDecision:
    """Promotion recommendation over the held-out suite. `promote` is the
    PROMOTE_RECOMMENDED flag; `reasons` lists every failed gate (empty ⇒ all passed)."""

    promote: bool
    reasons: list[str]


def _mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def evaluate_gates(
    candidate_rows: list[dict],
    champion_rows: list[dict],
    *,
    expected_seeds: list[int],
    hard_cells: list[int],
    max_death_rate: float = 0.0,
) -> GateDecision:
    """The promotion gate, pure
    over eval_m2-shaped rows so it is testable without running an eval. A candidate is
    PROMOTE_RECOMMENDED only if, on the held-out suite: death ≤ `max_death_rate` everywhere,
    the strict envelope ratio > 1.00 on every seed, the candidate mean beats the champion
    mean, and no hard cell (`s6/s9`) regresses below the champion. Promotion stays
    human-approved — this only recommends.

    Coverage is gated first: both the candidate and the champion must score *every*
    `expected_seed`, else an incomplete eval could promote by silently skipping a missing
    (possibly failing) held-out topology and shrinking the mean's denominator."""
    reasons: list[str] = []
    champ_by_seed = {r["seed"]: r for r in champion_rows}
    cand_by_seed = {r["seed"]: r for r in candidate_rows}

    for s in expected_seeds:
        if s not in cand_by_seed:
            reasons.append(f"seed {s}: missing from candidate eval (incomplete coverage)")
        if s not in champ_by_seed:
            reasons.append(f"seed {s}: missing from champion eval (incomplete coverage)")
    if reasons:  # don't score gates on an incomplete suite
        return GateDecision(False, reasons)

    for r in candidate_rows:
        if r["m2_dead"] > max_death_rate:
            reasons.append(f"seed {r['seed']}: death {r['m2_dead']} > {max_death_rate}")
        if r["ratio_env"] <= 1.0:
            reasons.append(f"seed {r['seed']}: envelope ratio {r['ratio_env']:.3f} !> 1.00")

    cand_mean, champ_mean = _mean(candidate_rows, "m2_cu"), _mean(champion_rows, "m2_cu")
    if cand_mean <= champ_mean:  # must *beat* the champion — a tie is not an improvement
        reasons.append(
            f"held-out mean {cand_mean:.1f} does not beat champion {champ_mean:.1f}"
        )

    for r in candidate_rows:
        if r["seed"] in hard_cells and r["seed"] in champ_by_seed:
            if r["m2_cu"] < champ_by_seed[r["seed"]]["m2_cu"]:
                reasons.append(
                    f"hard cell s{r['seed']} regressed: {r['m2_cu']:.1f} < "
                    f"champion {champ_by_seed[r['seed']]['m2_cu']:.1f}"
                )
    return GateDecision(not reasons, reasons)


def _result_md(
    registry: ChampionRegistry,
    candidate_rows: list[dict],
    champion_rows: list[dict],
    decision: GateDecision,
) -> str:
    """RESULT.md skeleton (experiments/README.md contract): the gate verdict, the
    PROMOTE_RECOMMENDED line the driver emits, and a per-seed candidate-vs-champion table."""
    champ = {r["seed"]: r for r in champion_rows}
    flag = "true" if decision.promote else "false"
    lines = [
        f"# EXP-0007B champion iteration — candidate vs {registry.champion_id}",
        "",
        f"PROMOTE_RECOMMENDED={flag}",
        "",
        "Champion-iteration loop (EXP-0007B): rolled out the registry champion, relabeled "
        "with the search teacher, distilled a candidate, scored both on the held-out suite.",
        "",
        "## Gate",
        "",
        ("All gates passed." if decision.promote
         else "Blocked by:\n" + "\n".join(f"- {r}" for r in decision.reasons)),
        "",
        "## Held-out (candidate vs champion)",
        "",
        "| seed | cand cu | champ cu | env cu | cand ratio_env | dead |",
        "|---|---|---|---|---|---|",
    ]
    for r in candidate_rows:
        c = champ.get(r["seed"], {})
        lines.append(
            f"| {r['seed']} | {r['m2_cu']:.1f} | {c.get('m2_cu', float('nan')):.1f} | "
            f"{r['env_cu']:.1f} | {r['ratio_env']:.3f} | {r['m2_dead']} |"
        )
    lines += ["", "## Conclusions", "", "_(fill in before promotion)_", ""]
    return "\n".join(lines)


def write_experiment_record(
    out_dir: "str | Path",
    *,
    registry: ChampionRegistry,
    candidate_ckpt: "str | Path",
    candidate_rows: list[dict],
    champion_rows: list[dict],
    decision: GateDecision,
    spec_extra: "dict | None" = None,
) -> Path:
    """Emit the standard experiments/ dir for one iteration (experiments/README.md
    contract): `spec.json` + `RESULT.md` + `summary.csv`, candidate checkpoint + raw eval
    under `artifacts/`. Writes only under `out_dir` — never the champion registry or its
    checkpoint (promotion is human-approved). Returns the dir."""
    out = Path(out_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)

    spec = {
        "driver": "tokenomics.train.champion_iterate (EXP-0007B)",
        "registry": asdict(registry),
        "promote_recommended": decision.promote,
        "gate_reasons": decision.reasons,
        **(spec_extra or {}),
    }
    (out / "spec.json").write_text(json.dumps(spec, indent=2))
    (out / "RESULT.md").write_text(
        _result_md(registry, candidate_rows, champion_rows, decision)
    )

    if candidate_rows:
        fields = list(candidate_rows[0].keys())
        for name in ("summary.csv", "artifacts/heldout.csv"):
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(candidate_rows)

    shutil.copyfile(candidate_ckpt, out / "artifacts" / "checkpoint.pt")
    return out


def preflight(registry: ChampionRegistry, *, repo_root: "str | Path") -> PreflightResult:
    """Load the registry's champion checkpoint from disk (relative to `repo_root`) and
    `check_checkpoint` it. The registry checkpoint path is repo-root-relative so the same
    JSON works from any cwd / on the pod."""
    import torch

    path = Path(repo_root) / registry.checkpoint
    if not path.exists():
        return PreflightResult(False, [f"champion checkpoint not found: {path}"])
    ckpt = torch.load(path, weights_only=True)
    return check_checkpoint(ckpt, registry)


# Default teacher (the L12/K12 routing search EXP-0006B/0007A distilled from). `elite_k=1`
# = EXP-0006B single-argmax distillation; `--distill elite_set` raises it (EXP-0008).
_DEFAULT_TEACHER = {"samples": 12, "lookahead": 12,
                    "candidates": ["even_split", "proportional", "obs_price"],
                    "elite_k": 1, "elite_eps": 0.05}

# EXP-0008 `--distill elite_set` family defaults: keep the top routings within 10% of the
# best, capped at 6. Applied only to knobs the user did NOT set on `--teacher`, so explicit
# overrides win (`_distill_teacher` + `_explicit_teacher_keys`).
_ELITE_SET_K = 6
_ELITE_SET_EPS = 0.1


def _distill_teacher(distill: str, teacher: dict, *, explicit=frozenset()) -> dict:
    """Apply the `--distill` candidate family on top of a parsed teacher (EXP-0008).
    `stochastic_mle` (EXP-0006B) leaves the single-argmax teacher (`elite_k=1`) untouched;
    `elite_set` turns it into an elite teacher (`elite_k>1`, wider tolerance) so the fit
    distills over the per-state elite set. The family defaults apply only to knobs absent
    from `explicit` (the `--teacher` keys the user actually set), so an explicit
    `--teacher elite_eps=…` / `elite_k=…` is preserved rather than clobbered."""
    t = dict(teacher)
    if distill == "elite_set":
        if "elite_k" not in explicit:
            t["elite_k"] = _ELITE_SET_K
        if "elite_eps" not in explicit:
            t["elite_eps"] = _ELITE_SET_EPS
    return t


def _default_train_fn(
    registry: ChampionRegistry, candidate_ckpt: Path, *,
    repo_root: Path, teacher: dict, eval_seeds: int, log,
) -> None:
    """Production trainer: distill a candidate orderer on the registry's contract via the
    existing expert-iteration driver (`orderer_improve.run_orderer_improve`) — producer
    fixed at v4, search-teacher relabel, stochastic deploy. A real run is Runpod-only
    (CLAUDE.md); this is the thin pass-through the orchestration wires by default.

    Passes `eval_seeds=0` so the trainer only trains+saves: the driver re-evals candidate
    AND champion itself (the gate, under identical conditions, at `eval_seeds`), so a
    train-step candidate eval would just be a wasted second held-out pass."""
    from .orderer_improve import run_orderer_improve

    run_orderer_improve(
        depth=registry.depth,
        prod_ckpt=str(repo_root / registry.producer_checkpoint),
        train_seeds=6, held_out_seeds=4,
        candidates=teacher["candidates"], n_samples=teacher["samples"],
        sample_scale=1.0, lookahead=teacher["lookahead"], dagger_iters=2,
        label_seeds=3, horizon=100, epochs=1500, lr=3e-3, seed=0,
        log_std_save=None, eval_seeds=0, save=str(candidate_ckpt),
        held_out_seed_list=registry.held_out_topology_seeds,
        enriched_orderer_obs=(registry.orderer_obs_version == 2),
        product_stream=registry.product_stream, stochastic=True,
        elite_k=teacher.get("elite_k", 1), elite_eps=teacher.get("elite_eps", 0.05),
        log=log,
    )


def _default_eval_fn(
    ckpt: "str | Path", registry: ChampionRegistry, *,
    eval_seeds: int, repo_root: Path, log,
) -> list[dict]:
    """Production evaluator: score a checkpoint on the held-out suite via `eval_m2`, in the
    registry's obs/stream contract, deploying the orderer stochastically (EXP-0006B)."""
    from .orderer_improve import eval_m2

    return eval_m2(
        str(ckpt), depth=registry.depth, seeds=registry.held_out_topology_seeds,
        eval_seeds=eval_seeds, horizon=100,
        enriched_orderer_obs=(registry.orderer_obs_version == 2),
        product_stream=registry.product_stream, stochastic=True, log=log,
    )


def champion_iterate(
    registry_path: "str | Path",
    out_dir: "str | Path",
    *,
    repo_root: "str | Path",
    eval_seeds: "int | None" = None,
    teacher: "dict | None" = None,
    preflight_fn=preflight,
    train_fn=_default_train_fn,
    eval_fn=_default_eval_fn,
    log=print,
) -> GateDecision:
    """One full champion-iteration loop (EXP-0007B): load registry -> preflight champion ->
    train a candidate -> evaluate candidate + champion on the held-out suite -> gate ->
    write the experiment record -> emit PROMOTE_RECOMMENDED. Returns the `GateDecision`.

    `preflight_fn`/`train_fn`/`eval_fn` are injectable so the orchestration is testable
    cloud-free; the defaults wire `run_orderer_improve`/`eval_m2` (real run on Runpod).
    Promotion stays human-approved — the loop only writes under `out_dir` and never the
    champion registry or its checkpoint."""
    repo_root = Path(repo_root)
    reg = ChampionRegistry.load(registry_path)
    es = eval_seeds if eval_seeds is not None else reg.eval_episode_seeds
    teacher = teacher if teacher is not None else _DEFAULT_TEACHER

    pf = preflight_fn(reg, repo_root=repo_root)
    if not pf.ok:
        raise RuntimeError(f"champion preflight failed (no training launched): {pf.reasons}")

    out = Path(out_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    candidate_ckpt = out / "artifacts" / "candidate.pt"

    log(f"=== champion_iterate: training candidate vs {reg.champion_id} ===")
    train_fn(reg, candidate_ckpt, repo_root=repo_root, teacher=teacher,
             eval_seeds=es, log=log)

    candidate_rows = eval_fn(candidate_ckpt, reg, eval_seeds=es, repo_root=repo_root, log=log)
    champion_rows = eval_fn(repo_root / reg.checkpoint, reg, eval_seeds=es,
                            repo_root=repo_root, log=log)

    decision = evaluate_gates(
        candidate_rows, champion_rows,
        expected_seeds=reg.held_out_topology_seeds,
        hard_cells=reg.hard_cells, max_death_rate=reg.max_death_rate,
    )
    write_experiment_record(
        out, registry=reg, candidate_ckpt=candidate_ckpt,
        candidate_rows=candidate_rows, champion_rows=champion_rows, decision=decision,
        spec_extra={"eval_seeds": es, "teacher": teacher},
    )
    log(f"PROMOTE_RECOMMENDED={'true' if decision.promote else 'false'}")
    if decision.reasons:
        log("  blocked by: " + "; ".join(decision.reasons))
    return decision


def _parse_teacher(spec: str) -> dict:
    """Parse a `--teacher samples=12,lookahead=12` CLI spec, overriding only the named keys
    on top of the default teacher (candidate set stays unless overridden)."""
    t = dict(_DEFAULT_TEACHER)
    for tok in spec.split(","):
        if not tok.strip():
            continue
        k, _, v = tok.partition("=")
        k = k.strip()
        if k in ("samples", "lookahead", "elite_k"):
            t[k] = int(v)
        elif k == "elite_eps":
            t[k] = float(v)
        elif k == "candidates":
            t[k] = v.split("|")
    return t


def _explicit_teacher_keys(spec: str) -> set[str]:
    """The teacher knobs the user actually set on `--teacher` (key names present in `spec`),
    so `_distill_teacher` can distinguish "not provided" (apply the family default) from
    "provided" (preserve), instead of guessing from the value (EXP-0008 P2 fix)."""
    return {tok.partition("=")[0].strip() for tok in spec.split(",") if tok.strip()}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default="champion.json", help="champion registry JSON")
    ap.add_argument("--out-dir", required=True, help="fresh EXP-NNNN dir for the record")
    ap.add_argument("--repo-root", default=None,
                    help="repo root for relative checkpoint paths (default: registry dir)")
    ap.add_argument("--teacher", default="samples=12,lookahead=12",
                    help="search teacher spec, e.g. samples=12,lookahead=12")
    ap.add_argument("--distill", default="stochastic_mle",
                    choices=["stochastic_mle", "elite_set"],
                    help="candidate family: stochastic_mle (EXP-0006B single-argmax) or "
                         "elite_set (EXP-0008: distill over the per-state elite routing set)")
    ap.add_argument("--eval-seeds", type=int, default=None,
                    help="held-out eval episode seeds (default: registry eval_episode_seeds)")
    args = ap.parse_args()

    registry_path = Path(args.registry)
    repo_root = Path(args.repo_root) if args.repo_root else registry_path.resolve().parent
    decision = champion_iterate(
        registry_path, args.out_dir, repo_root=repo_root, eval_seeds=args.eval_seeds,
        teacher=_distill_teacher(args.distill, _parse_teacher(args.teacher),
                                 explicit=_explicit_teacher_keys(args.teacher)),
    )
    raise SystemExit(0 if decision.promote else 1)


if __name__ == "__main__":
    main()
