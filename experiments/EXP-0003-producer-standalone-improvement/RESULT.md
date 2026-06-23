# EXP-0003 — standalone producer improvement after Phase 2

**Date:** 2026-06-22 · **Drivers:** `tokenomics.eval.swap_eval`, `tokenomics.eval.producer_rungs` (new), `tokenomics.train.producer_improve` (new) · **Reproduce:** see [`spec.json`](spec.json)

**Status:** M2 v1 **REJECTED** — producer-only RL plateaus at the clone (no robust
improvement; generalization tail regresses). A real failure surface remains → motivates
EXP-0004 (adversarial generation). M0 baseline + both new tools land.

## Question

EXP-0002 produced a depth-3 producer **clone** (`d3_perm`) that clears the `d3_p00`
producer gate. The next question is whether that producer can be pushed **beyond**
`stock_aware` by RL **while the orderer stays fixed to `proportional`** — the standalone
producer. `smoke_ppo --init-from` is deliberately **excluded** as the main result: it
updates producer *and* orderer jointly, so a producer "win" could really be the learned
orderer co-adapting. The trainer here (`producer_improve`) freezes the orderer to the
white-box `proportional` baseline in both rollout and eval, updates only the producer
actor, and anchors it (BC) to the clone.

## Method

* **M0 (control):** the EXP-0002 `d3_perm` clone, no training — the current bar.
  * `swap_eval` on `d3_p00` (the 2×2 producer/order swap).
  * `producer_rungs` (new): `train_prod + proportional` vs `stock_aware + proportional`
    on the curriculum quantity rungs `staged_config(build_scale_config(seed), p)` for
    `p ∈ {0.00, 0.10, 0.24, 0.25}` × config seeds `{0,3,4,5}`, 5 eval seeds, horizon 100.
    Verdict per cell: `ratio = clone_units/base_units ≥ 0.80` and clone death `≤ 0.05`.
* **M2 (producer-only improvement v1):** `producer_improve` (new) — producer-only PPO,
  orderer frozen to `proportional`, BC anchor (coef 0.05) to the `d3_perm` clone,
  conservative trust region (`actor_lr 5e-5`, `target_kl 0.003`). Rollout across the
  Phase-2 config set (3 base seeds × {base + 2 slot perms} = 9 topologies) and all four
  rungs; selection ranks death-first then consumer-units ratio on the base train
  topologies; held-out report on 3 disjoint seeds × 3 perms = 12 topologies.
  Ran off-box on a Runpod pod (CPU torch; the workload is single-threaded).

## Results

### M0 — swap gate (`d3_p00`, seed 0, 5 seeds)

| combo | units✓ | dead | cons✓ | return |
|---|---|---|---|---|
| base  + base  | 361.8 | 0.00 | 217.8 | 3562.5 |
| train + train | 248.4 | 0.00 | 148.6 | 2428.5 |
| **train + base** | **361.6** | 0.00 | 216.0 | 3557.5 |
| base  + train | 248.6 | 0.00 | 148.6 | 2433.1 |

`train+base` ≈ `base+base` (ratio ~1.00, 0% death): the cloned **producer** matches
`stock_aware` under the baseline orderer. The learned **orderer** is the weak side
(`*+train` drops to ~248) — consistent with EXP-0001/0002 (producer is the blocker; the
orderer the clone shipped with is not the standalone producer's concern here).

### M0 — quantity rungs (`producer_rungs`, clone vs stock_aware, both + proportional)

| rung | seed 0 | seed 3 | seed 4 | seed 5 |
|---|---|---|---|---|
| 0.00 | 1.00 | **0.17** | 0.95 | **0.74** |
| 0.10 | 0.90 | 1.90 | 0.99 | **0.71** |
| 0.24 | **0.68** | 3.14 | **0.60** | **0.69** |
| 0.25 | 0.93 | 1.35 | 0.98 | 0.95 |

(ratio = clone_units / base_units; **bold** = below the 0.80 bar. Full table:
[`m0_d3_perm_rungs.csv`](m0_d3_perm_rungs.csv).)

**No death anywhere** (clone is survival-safe across all rungs/seeds). But the clone is
**not** uniformly ≥ 0.80: 6 of 16 cells fail. The pattern:
* **Rung 0.24 is the systematic soft spot** — fails on 3 of 4 seeds (0.60–0.69). This is
  the near-full single-product quantity rung, just below the mixture switch (`_MIX_AT
  = 0.25`); the clone under-fulfills there even on its **training** topology (seed 0).
* **Seed 5** is broadly weak at the low rungs (0.74, 0.71, 0.69).
* **Seed 3 p=0.00** is a catastrophic outlier (0.17) yet the clone *beats* base at higher
  rungs on the same seed (1.9×, 3.1×) — a topology-specific low-rung pathology, not a
  global failure.

So the standalone clone clears the headline `d3_p00` gate but has a real **rung-0.24
quantity weakness** and **per-topology variance** — the surface M2 should improve.

### M2 — producer-only improvement v1

200 iters, `actor_lr 5e-5`, `target_kl 0.003`, `bc_anchor 0.05`, `explore_log_std -1.5`,
rollout over 9 train topologies × 4 rungs. Ran off-box on a **CPU Runpod pod** (see
"Compute" — `torch 2.12.1+cpu`; the run is single-threaded so the GPU is irrelevant).
Throughout, the central-critic explained variance sits **≈ 0** (the documented depth-3
untrustworthy-critic regime), so PPO has no usable advantage signal — the BC anchor +
KL-stop are all that drive the producer, and they pull toward the clone.

**Headline (selection on train topologies; held-out on 12 disjoint topologies):**

| metric | clone (M0) | prod_improve_v1 (M2) |
|---|---|---|
| best-select mean ratio (3 train topos × 4 rungs) | ~0.96 | **1.030** |
| select min ratio (worst cell) | 0.645 | 0.692 |
| held-out mean ratio (12 topos × 4 rungs) | — | 1.121 |
| **held-out min ratio (worst cell)** | — | **0.299** |
| death (everywhere) | 0.00 | 0.00 |

(M2 = the **selected** checkpoint, iter 90 — highest train-topo mean ratio, saved as
`prod_improve_v1.pt`; its in-training select min is 0.692. The final iter-199 checkpoint is
slightly lower mean / higher min (1.014 / 0.732) — neither beats the clone. Held-out min
0.299 is the worst of the 12 held-out topos × 4 rungs, from the per-iter held-out aggregate
in [`logs/`](logs/); it is not a train-topo selection cell.)

**Per-cell rung ratios, clone (M0) → improved (M2)** (vs `stock_aware`; both + `proportional`;
[`m2_prod_improve_v1_rungs.csv`](m2_prod_improve_v1_rungs.csv)):

| rung | seed 0 | seed 3 | seed 4 | seed 5 |
|---|---|---|---|---|
| 0.00 | 1.00 → 1.00 | 0.17 → 0.26 | 0.95 → **0.58** ↓ | 0.74 → 0.85 ↑ |
| 0.10 | 0.90 → 0.91 | 1.90 → 1.51 | 0.99 → 0.82 | 0.71 → 0.70 |
| 0.24 | 0.68 → 0.69 | 3.14 → 2.90 | 0.60 → 0.58 | 0.69 → 0.66 |
| 0.25 | 0.93 → 0.95 | 1.35 → 1.37 | 0.98 → 0.99 | 0.95 → 0.99 |

Both clone and improved have **6/16 cells below 0.80 — but different cells**. M2 leaves the
**systematic rung-0.24 weakness untouched** (0.69 / 2.90 / 0.58 / 0.66 ≈ unchanged),
helps two low-rung cells (seed-5 p=0.00 0.74→0.85, seed-3 p=0.00 0.17→0.26), and
**regresses a previously-fine held-out cell** (seed-4 p=0.00 0.95→0.58). Net: the weakness
is *redistributed*, not removed. L1 trace on the improved producer is healthy — 100/100
cycles, `dead=False`, no L1 capacity decay ([`logs/m2_prod_improve_v1_trace.txt`](logs/m2_prod_improve_v1_trace.txt)).

## Conclusions

1. **Producer-only RL did not beat the clone.** Best-select mean ratio 1.030 ≈ the clone;
   on the *meaningful* axis (the systematic rung-0.24 weakness) it does not improve at all.
   With the critic uninformative (EV ≈ 0), PPO has no gradient to exploit, so the
   conservative anchor + KL-stop just hold the policy in the clone's basin.

2. **Acceptance: REJECT.**
   * death ≤ 0.05 everywhere — **PASS** (0.00 throughout).
   * `d3_p00` (seed 0) ratio ≥ clone − 0.02 — **PASS** (1.00 vs 1.00).
   * L1 trace healthy — **PASS**.
   * held-out seed/perm ratio ≥ 0.80 — **FAIL** (held-out min 0.299; seed-4 p=0.00 0.58,
     seed-5 p=0.10 0.70).
   * beats `base+base` by ≥ 5% on a meaningful rung, **or** matches units with fewer
     rejects/waste — **FAIL** (no gain on rung-0.24; replenishment rejects not reduced).
   * explicit reject trigger "held-out ratios drop" — **TRIPPED** (seed-4 p=0.00 0.95→0.58).

3. **EXP-0004 (adversarial generation) is warranted.** The producer-only path plateaus and
   a real failure surface remains (rung-0.24 systematic under-fulfilment + a weak held-out
   tail), exactly the precondition the spec set for moving to adversarial generation.

## Follow-up

* The binding constraint is the **value function**, not the policy update: at depth-3 scale
  the central critic never earns trust (EV ≈ 0), so any PPO-style improvement is starved of
  signal. Before more producer-only RL (M3) or EXP-0004, a critic that fits the depth-3
  return (richer global state / longer warmup / return normalization per rung) is the
  higher-leverage fix.
* `M1` (stronger clone via more-seed/-perm decode-aware DAgger) stays optional; the
  rung-0.24 weakness is shared by clone *and* improved, so it is a property of the
  `obs_stock_aware` expert being cloned, not of slot memorization — M1 would not fix it.
