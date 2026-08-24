# EXP-0040 - Rollout Target Stability

**Date:** 2026-08-01 | **Run:** 2026-08-24 | **Status:** COMPLETE — **SEARCH_ONLY** | **Plan:** [HANDOVER.md](HANDOVER.md)

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

The registered RunPod collection completed over all **424 decisions** and both continuation
tranches. Gate 1 failed at K=16, so the collector appended continuation IDs 16..31 exactly once.
It persisted **36,828 raw label rows**: 1,116 eligible candidates times the original-suffix
control plus 32 resampled continuations. The artifact-only analyzer then returned the mechanical
classification **`SEARCH_ONLY`**.

| Task | Split-half Spearman | Winner overlap | Expected selector regret | Actor regret | Reduction vs actor | Stable? |
|---|---:|---:|---:|---:|---:|---|
| s45 | 0.500 | 0.599 | 2.310 | 3.782 | 38.9% | no |
| s77 | 0.500 | 0.602 | 2.920 | 5.272 | 44.6% | no |
| s29 | 1.000 | 0.743 | 2.093 | 3.452 | 39.4% | no |

- **Gate 1 — convergence: FAIL.** The registered thresholds are Spearman >= 0.70 and winner
  overlap >= 0.80 for every task aggregate and trajectory seed. No task converged, and all nine
  task/seed cells failed. The K-prefix diagnostic was also false at K=4, K=8, and K=16.
- **Gate 2 — usefulness: PASS.** An oracle choosing from the K=16 continuation means reduced
  regret versus the canonical actor by 38.9–44.6% on all three tasks, with the required >=10%
  reduction holding for every trajectory seed.
- **Diagnostic only:** original-single-suffix to continuation-mean winner overlap was
  0.510 / 0.436 / 0.600 (s45 / s77 / s29), with centered rank correlations
  0.298 / 0.197 / 0.200. The useful action ranking materially changes as future demand changes.

Collection ran from commit `817464d9a08393f6677bce0843a75a21de2fc798` on a 2-vCPU RunPod
CPU instance in about 25 minutes. A post-run artifact-only analyzer correction at commit
`afe6a66644a339b86612a9d17142ac75b36295c1` fixed a non-gating diagnostic that had compared
each split against its own winner; the corrected fixed-selector changes are nonzero, while every
gate value and the classification are unchanged. The raw-label SHA-256 is
`20aa37136dd8933cac96228540dfa60bafc13dcdc8a30847060231c956c30c9e`; the sample SHA-256 is
`3be41019247d037f3f6b30c16633c62013169215d6f05c888fb44997e1c44c06`.

Authoritative artifacts: [selection manifest](artifacts/selection_manifest.json),
[raw continuation labels](artifacts/continuation_labels.csv),
[task metrics](artifacts/metrics_by_task.csv), [seed metrics](artifacts/metrics_by_seed.csv),
and [run specification](run_spec.json).

## Conclusions

The label contains real decision value, but it is not stable enough to be a supervised target
after the full registered continuation budget. Direct GNN imitation is therefore **not licensed**,
and EXP-0041 is cancelled by its own prerequisite contract. The surviving use of this signal is
online rollout/search, where future-demand uncertainty can be sampled at decision time; the next
bounded experiment is a search-depth comparison over {1, 2, 3}.
