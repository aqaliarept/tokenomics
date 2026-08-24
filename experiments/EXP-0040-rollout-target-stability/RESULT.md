# EXP-0040 - Rollout Target Stability

**Date:** 2026-08-01 | **Implementation:** 2026-08-24 | **Status:** IMPLEMENTED, POD RUN PENDING | **Plan:** [HANDOVER.md](HANDOVER.md)

## Question

Is the single-continuation marginal-value label a stable and useful function of the
deployable pre-action state after future exogenous demand is marginalized out?

## Method

The collector and artifact-only analyzer are implemented in
`tokenomics.eval.rollout_target_stability`.

- The collector SHA-validates and exactly joins the EXP-0039 decision/candidate authority to
  the canonical EXP-0038B actor rows before simulator work. It selects the lowest SHA-256-ranked
  decisions in each registered task/seed/cycle-quartile/label-class stratum, replays the frozen
  pre-action state, checks graph + original-label parity, and persists every raw continuation.
- The analyzer reads only `continuation_labels.csv`, checks exact sample/candidate/continuation
  coverage, computes all task + trajectory-seed convergence/usefulness metrics, and emits the
  mechanical classification plus `convergence_by_k` consumed by EXP-0041.
- The protocol collects K=16 first and appends IDs 16..31 exactly once only when Gate 1 fails.
  It never re-buys the first tranche.

The frozen authority contains 13,940 decisions. Deterministic stratum selection yields **424**:
**325 signal + 99 flat controls**; short strata are recorded and never backfilled.

### Pre-run verification

- Two-decision real-simulator smoke (one signal + one flat, champion producer, horizon 100,
  lookahead 8, K=16): **68 raw rows in 9.7 s**, exact state/candidate/original-label parity.
- Focused collector/fork/GNN regression suite: **96 passed**.
- Full repository suite: **699 passed, 1 deselected** (only the existing PettingZoo warnings).
- Registered archive, producer, EXP-0039 decision, and EXP-0039 candidate hashes match.
- **Pre-registration correction:** the originally drafted canonical-actor hashes
  (`b663...` / `b917...`) did not name any committed artifact. The actual canonical EXP-0038B
  `dump_sys` files at commit `8cba827` hash to `b22af97e...` / `06c5cd55...`; they join exactly
  to all 13,940 EXP-0039 decisions and all eligible candidates. The pins were corrected before
  rollout spend.

## Result

No empirical result yet. Pod collection was authorized by the user on 2026-08-24 and is pending
the provenance-clean commit and launch.

## Conclusions

None. This experiment cannot license a GNN actor; it can only license a subsequent
averaged-label or coverage experiment, or close direct imitation in favor of search.
