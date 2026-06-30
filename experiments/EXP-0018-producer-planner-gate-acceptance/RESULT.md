# EXP-0018 — producer planner-gate acceptance rerun (ordered-plan step 7)

Status: **DONE — contract PASS, joint value PASS, producer-alone gap MATERIALLY REDUCED
(not fully closed).** The EXP-0017 producer-side out-of-family fix — replace only the
learned upgrade / inventory-expansion investment gates with deterministic planner gates
(`portable_scorer_planner_gates`), learned continuous heads unchanged, **same checkpoint
and schema** — is rerun across all four fork/join held-out seeds 6–9 under a
`proportional` orderer. The producer-alone regression that EXP-0017 surfaced is largely
gone: min ratio lifts **0.450 → 0.955**, mean **0.874 → 1.019**, and **3/4 seeds clear
the bar** (s6/s7/s8 ≥ 1.0). **s9 stays at 0.955**, so the strict "no producer-alone seed
below 1.0" gate is not cleared. Contract and death are unchanged from EXP-0017 (zero
failures, zero death on every portable cell), and the joint stack still beats the bar —
the planner-gated joint (mean **2.91×**) edges out the learned-gate joint (2.86×). Local
eval, no retrain.

## Question
EXP-0017 surfaced a producer-side *policy* gap: under a weak (`proportional`) orderer the
portable producer with **learned** investment gates regresses below the
`entity_planner + proportional` bar out-of-family (s8 0.45, s9 0.83). Step 6 landed the
fix as a distinct producer kind. Does it hold across all held-out fork/join seeds, and
which acceptance bar does it clear? See `spec.json`.

## Setup
- **Fixture / runner / bar** — identical to EXP-0017: `build_forkjoin_config(seed,
  depth=3, width=4, inputs_per_product=2)` on held-out seeds 6–9; `transfer_matrix` +
  `out_of_family_targets`; bar = `entity_planner + proportional` on the **same** fork/join
  target; 5 episodes/cell, horizon 100; no retrain (EXP-0012 + EXP-0013 checkpoints).
- **New stacks** — `DEFAULT_STACKS` now carries **8** stacks: the original 6 (the
  learned-gate `portable_producer+proportional` and `portable_producer+portable_orderer`
  kept as **regression bars**) PLUS the two planner-gated rows. No silent flip of
  `portable_scorer`; the planner-gated kind is `portable_scorer_planner_gates` with a
  `producer_policy_mode` (`learned_gates | planner_gates`) and `producer_tie_rel_tol`
  column per row.
- **Tie tolerance** — shipped default `PLANNER_GATE_TIE_REL_TOL = 0.0` (planner-gated
  rows ran at tol 0.0; tie tolerance is a non-monotonic ablation knob, off by default).

## Result (ratio vs `entity_planner + proportional` on the SAME fork/join target)

| stack | mode | s6 | s7 | s8 | s9 | mean | min | dead |
|-------|------|----|----|----|----|------|-----|------|
| noop+proportional (floor) | n/a | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 |
| entity_planner+proportional (bar) | n/a | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| stock_aware+proportional | n/a | 1.064 | 1.190 | 0.205 | 0.243 | 0.675 | 0.205 | 0 |
| portable_producer + proportional | learned_gates | 1.175 | 1.046 | 0.450 | 0.826 | 0.874 | 0.450 | 0 |
| **portable_producer_planner_gates + proportional** | planner_gates | 1.106 | 1.010 | **1.005** | **0.955** | **1.019** | **0.955** | 0 |
| stock_aware + portable_orderer | n/a | 2.200 | 1.798 | 1.933 | 2.024 | 1.989 | 1.798 | 0 |
| portable_producer + portable_orderer (joint) | learned_gates | 2.894 | 2.637 | 2.824 | 3.072 | 2.857 | 2.637 | 0 |
| **portable_producer_planner_gates + portable_orderer** (joint) | planner_gates | 3.133 | 2.669 | 2.908 | 2.944 | **2.913** | 2.669 | 0 |

All 28 non-floor cells: `producer_schema_loads = orderer_schema_loads =
producer_contract_compatible = orderer_contract_compatible = True`, `runs_episode =
True`, `failure_class = none`, `dead_rate = 0`. Only the `noop` floor rows are dead (by
design). Full rows in `summary.csv`.

## Conclusion
**Contract: PASS (unchanged).** Swapping the producer policy adapter does not touch the
id-free contract — the planner-gated kind loads, is contract-compatible, and runs on
every fork/join seed exactly like the learned-gate kind. Zero contract failures, zero
death across all 28 non-floor cells.

**Joint value: PASS.** The planner-gated joint stack beats the bar on every seed (mean
**2.91×**, min 2.67×), marginally **ahead of** the learned-gate joint (2.86×). The fix
does not cost the joint stack; if anything it helps slightly.

**Producer-alone gate: MATERIALLY REDUCED, not strictly closed.** The planner-gated
producer-alone (under `proportional`) recovers the EXP-0017 regression:

- s8 **0.450 → 1.005** (clears the bar) — the seed EXP-0017 flagged worst.
- s9 **0.826 → 0.955** (improved, still below).
- s6 1.175 → 1.106 and s7 1.046 → 1.010 (still ≥ 1.0; the planner gates trade a little
  of the learned-gate upside on the already-passing seeds for the big s8 recovery).
- min **0.450 → 0.955**, mean **0.874 → 1.019**, **3/4 seeds ≥ 1.0**.

**Acceptance bar decision.** Two candidate bars were on the table (spec):
- *"Materially reduced"* — **MET.** Min lifts 0.450→0.955, mean crosses 1.0, 3/4 seeds
  clear the bar, joint unaffected.
- *"No producer-alone seed below 1.0"* — **NOT MET.** s9 sits at 0.955.

So **step 7 lands as the materially-reduced gate**; the strict gate remains open on s9.
This matches the step-6 single-seed prediction exactly — the multi-seed rerun confirms it
at the shipped `tol = 0.0` (the tie-tolerance knob is non-monotonic: it would lift s9 to
~0.992 at tol 0.02 but drop s8 to ~0.946, so it stays off by default).

**Residual / next.** Final P2/C3 producer-side closure (clearing s9 too, without a
seed-specific tie-tolerance tradeoff) needs **training coverage** of the states the
learned gates generalize poorly on — fork/join, shared-capacity, scarcity, and
running-investment — not another deterministic patch. That is a producer-side
training/fidelity question (step 8+ / a future EXP), distinct from this acceptance gate.

## Rerun command
```
python -m tokenomics.eval.transfer_matrix \
  --producer-ckpt experiments/EXP-0013-portable-producer-scorer/artifacts/producer_scorer.pt \
  --orderer-ckpt experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt \
  --depth 3 --seeds 6-9 --out-of-family --inputs-per-product 2 \
  --eval-seeds 5 --horizon 100 \
  --out experiments/EXP-0018-producer-planner-gate-acceptance/summary.csv
```
Local eval (evaluation of existing checkpoints — no training). The `--tie-rel-tol` flag
(default 0.0) selects the planner-gated replenishment tie tolerance for ablation rows.

## Artifacts
- `summary.csv` — the 32-row matrix (4 fork/join seeds × 8 stacks).
- (checkpoints reused in place from `EXP-0012-.../artifacts/` and
  `EXP-0013-.../artifacts/`; this experiment adds no new ckpt.)
