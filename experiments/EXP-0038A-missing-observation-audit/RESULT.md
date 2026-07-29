# EXP-0038A — Missing Observation Audit

**Date:** 2026-07-27 · **Driver:** `tokenomics.eval.observation_ladder` · **Reproduce:** see
[`spec.json`](spec.json) / [`audit/spec.json`](audit/spec.json)

**Status: CLOSED for the deployable blocks the dump already logs (current_row, +order,
+local_queue). Verdict: they do NOT license observation enrichment.** The alias-conflict collapse
that motivated this continuation is a between-decision *level* artifact, not a routing signal. Two
per-producer blocks (full producer state, history) remain untested and need a live re-dump — see
"What is still open".

## Question

Which information is missing from the actor observation, and is that information deployable at
decision time? EXP-0038 proved the current 7-feature producer row aliases the marginal-value teacher
(median label range 10 over rows seen ≥5×) but did not prove every deployable observation fails. A
post-run peek suggested adding logged order context (product/quantity) shrinks the fixed-feature
label conflict. This audit finishes that analysis properly.

## What was measured

`tokenomics.eval.observation_ladder` reads the EXP-0038 `dump/` (13,940 decisions / 37,804 eligible
producer rows over s29/s77/s45; no re-simulation) and walks a **cumulative deployable-observation
ladder** over the blocks the dump already logged:

1. **current_row** — the EXP-0037 7-feature producer row.
2. **+order** — product, quantity, stream index, order age, TTL remaining.
3. **+local_queue** — backlog size, same-product backlog.

For each block prefix it computes two things (tables under [`audit/`](audit/)):

- **Exact-alias label spread** (`alias_by_block.csv`): group producer rows by the block's feature
  tuple; for groups seen ≥5×, the teacher-label *range*. A block that resolves the aliasing drives
  the median range → 0.
- **Held-out routing probe** (`probe_by_block.csv`, `block_summary.csv`): leave-one-seed-out per
  task; fit a probe on two seeds' rows and score the held-out seed's routing choices. The metric is
  computed on the **real-choice decisions only** (≥2 eligible producers *and* a non-flat teacher —
  the ~30% where the teacher actually prefers someone; flat decisions carry no choice and would
  dilute every method equally). Labels are **decision-centered** before fitting (the teacher label
  carries a large between-decision *level* that says nothing about *which* producer to pick;
  centering forces the probe onto the within-decision contrast the routing choice needs). Reported:
  **mean teacher regret** (argmax pick), tie-aware **top-1**, and **centered `r2`** (within-decision
  variance explained vs the zero predictor). Every metric is **decision-weighted** by pooling the
  held-out folds (each decision scored once), so it is directly comparable to the floors. Two probe
  families are run and cross-checked (per the plan): **ridge** (standardized block feats +
  current×order-context interactions — the only way an additive model can re-rank producers within a
  decision) and **kNN** (nonparametric; order context enters the distance). Floors, scored the same
  way: **random** (uniform eligible pick) and **actor** (the *deployed* EXP-0037 residual scorer,
  `actor_score`, argmax — the incumbent that sees current_row only).

> **Metric correction (2026-07-27).** An earlier cut of these tables fit *raw* labels, scored *all*
> contested decisions (including flat ones), averaged *fold-means* (not decision-weighted), and used
> an inconsistent top-1 vs the floor. That inflated a spurious "+order kNN beats actor 1/3". With
> the corrected metric above the negative read is **stronger and cleaner** (see Result).

## Result

Per real-choice decision, decision-weighted mean teacher regret (lower is better):

| task | random floor | **actor floor** | +order ridge | +order kNN |
|---|---|---|---|---|
| s45 | 5.239 | **3.935** | 4.837 | 4.784 |
| s77 | 6.241 | **4.754** | 6.178 | 6.149 |
| s29 | 5.380 | **4.120** | 4.904 | 4.676 |

| block | alias median | alias frac>1 | ridge beats-actor | kNN beats-actor | centered r2 (ridge / kNN) |
|---|---|---|---|---|---|
| current_row | **10.0** | 0.669 | 0/3 | 0/3 | 0.001 / −0.04 |
| +order | **0.0** | 0.224 | **0/3** | **0/3** | 0.001 / −0.02 |
| +local_queue | 0.0 | 0.225 | 0/3 | 0/3 | 0.001 / −0.02 |

**Read against the decision gate (both conditions required; probes must agree ≥2/3):**

1. **Label conflict collapses — PASS for +order.** Median exact-alias range 10→0, frac>1 0.669→0.224
   (coverage 16,266 of 37,804 rows still in ≥5× groups). Adding order context genuinely removes the
   fixed-feature label conflict EXP-0038 flagged.
2. **Held-out routing beats the actor floor on ≥2/3 tasks, both probes — FAIL for every block, on
   every task.** No deployable block, in either probe family, beats the deployed current_row-only
   actor on a single task (**0/3** everywhere; +order kNN, the closest, is 4.78/6.15/4.68 vs the
   actor's 3.93/4.75/4.12). The probes do clear the *random* floor — so they are not noise, just
   strictly worse than the incumbent. Centered within-decision `r2` is **≈ 0** for ridge and
   **slightly negative** for kNN across all blocks: the deployable observations explain essentially
   **none** of the within-decision label contrast (kNN is fractionally worse than predicting no
   contrast at all). +local_queue is indistinguishable from +order.

**Why the alias collapse does not become routing skill — the mechanism.** Order/queue fields are
**constant across a decision's eligible producers**. So they can shift the *level* of a decision's
labels (bigger orders → uniformly bigger marginal values, which is exactly what collapses the
between-decision exact-alias spread) but they carry **zero within-decision contrast** — they cannot
say *which* producer is marginally best. The interaction-enriched ridge and the context-aware kNN
were both given the chance to exploit order context for within-decision re-ranking and neither
converted it into a win over current_row alone. Corroborating evidence in `top_mistakes_by_block.csv`:
of the 60 highest-regret deployed-actor mistakes, **96.7%** have the teacher-best and actor-best
producers *differing on the current row* — the producers are already distinguishable to the actor;
it simply ranks them wrong. High-regret decisions span quantity 1–14 (mean 8.35), so order size does
not even flag *where* the hard decisions are. This is not a missing per-producer observation; it is
that the teacher's within-decision preference is not a learnable function of these features — a
global, path-dependent rollout quantity, consistent with EXP-0009 (value-from-global_state did not
transfer) and EXP-0038 (current row aliases the label).

## Verdict

For the deployable blocks the dump logs, **observation enrichment is not licensed**: `+order` /
`+local_queue` resolve the label *level* but not the *routing choice*, and neither probe beats the
deployed current_row actor on **any** of the 3 tasks (0/3), with centered within-decision `r2` ≈ 0.
`blend_0.5` remains the orderer floor. No new actor was trained (HANDOVER non-goal upheld).

## What is still open (do not over-claim)

Per the HANDOVER non-goal "do not declare the teacher target unlearnable until the deployable ladder
fails," this audit covered ladder blocks 1–3 only. The remaining deployable blocks split by whether
they can re-rank *within* a decision:

- **+system_demand (5), +recipe_bottleneck (6):** mostly **decision-constant** (system/order-level),
  so they share the +order limitation — unlikely to help within-decision routing on their own.
- **+full_producer_state (4), +history (7):** genuinely **per-producer** — the only remaining blocks
  that *could* re-rank within a decision. They are not in the current dump and need a live-env
  re-dump (`value_router_errors` extension) to test.

Given the accumulated prior (global rollout label failed to be a function of global_state in
EXP-0009, of the current per-producer row in EXP-0038, and of order/queue context here), the
residual hope is narrow. The continuation is
[`EXP-0038B`](../EXP-0038B-producer-product-observation-ladder/RESULT.md): start with richer
per-producer/product metrics that can re-rank within a decision, then add system-wide context only
as interaction features. If that ladder fails, retire the marginal-value teacher and keep
`blend_0.5` as the floor.

## Artifacts

- Driver: `tokenomics/eval/observation_ladder.py` (torch-free; reuses EXP-0038 `dump/`).
- Tests: `tokenomics/tests/test_observation_ladder.py` (8, torch-free).
- Tables: `audit/{block_summary,alias_by_block,probe_by_block,top_mistakes_by_block}.csv` + `audit/spec.json`.
- **Not computed:** near-neighbor label spread (HANDOVER metric). Rationale: the decisive gate metric
  is the held-out routing probe (done, both families), and gate-condition-1 (label conflict) is
  already answered by exact-alias with adequate coverage (902–1021 groups ≥5×). Near-neighbor
  measures the same already-answered level-conflict axis and cannot change the verdict.
