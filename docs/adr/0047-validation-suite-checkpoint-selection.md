# Validation-topology-suite checkpoint selection

Status: accepted

## Context

`train()` keeps the best-on-eval checkpoint (ADR-0011), but it scored that eval on
`base_cfg` alone — one fixed topology. Under topology domain-randomization (ADR-0046) the
goal is held-out generalization, so selecting on a single topology can keep a checkpoint that
wins `base_cfg` while losing the held-out distribution (the plan's "Current Diagnosis", item
3, and the canonical failure in the status table: trained fixed-only scores 227 on seed0 but
collapses to 114/33% held-out).

## Decision

Add `validation_configs: list[Configuration] | None` to `train()`. When given, checkpoint
selection scores the greedy policy on the **whole suite** via `evaluate_validation_suite`
(the per-topology eval mean averaged across topologies) and ranks survival-first on **mean
death rate, then mean consumer units** — the held-out goal metric, not return. Without a
suite, selection falls back to the prior base-`cfg` behaviour (death, then return), so the
canonical phase1 path is unchanged.

Topology-seed bands are kept disjoint so a checkpoint is never validated on a topology it
trained on: training DR draws `[1, TRAINING_SEED_HI=9000)`, the validation suite uses
`[VALIDATION_SEED_OFFSET=9000, …)`, held-out uses `[HELD_OUT_SEED_OFFSET=10000, …)`. The
per-iter displayed eval still runs on `base_cfg` (preserving the canonical benchmark in the
training curve); the suite eval drives selection only.

CLI: `--validation-seeds N` builds the suite from `build_scale_config` at `--topology-depth`.

## Considered options

- **Select on mean return across the suite.** Rejected: at scale, return is dominated by
  investment-order churn (memory `scale-reward-calibration`); the valid objective is fulfilled
  consumer units, which is what the release bar and held-out comparison use.
- **Replace the displayed base eval with the suite eval.** Rejected: the release bar requires
  preserving the canonical benchmark result visibly; keeping base_cfg in the curve and using
  the suite only for selection serves both.

## Consequences

- `evaluate_validation_suite` is unit-tested for the cross-topology averaging
  (`test_validation_suite_averages_metrics_across_topologies`) and the end-to-end selection
  path runs (`test_train_selects_on_validation_suite`). CLI verified (`best eval … val_units`).
- Checkpoint selection now targets the held-out objective; whether a run clears the held-out
  bar is the A/B question (plan item 5), measured separately.
- Full suite green (199 tests).
