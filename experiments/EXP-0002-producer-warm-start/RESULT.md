# EXP-0002 — producer-first warm-start (ADR-0049 Phase 1 + Phase 2)

**Date:** 2026-06-22 · **Driver:** `tokenomics.train.warm_start` + `tokenomics.train.clone_fidelity` · **Reproduce:** see [`spec.json`](spec.json)

**Status:** Phase 1 **PASSED**. Phase 2 **PASSED** — both the multi-topology slice
(seed sampling) *and* obs/action **slot permutation** (steps 1–4) are implemented and
passing. The permutation-augmented clone clears `d3_p00`, holds on unseen seeds, and —
the sharpest probe — stays above bar on unseen *permutations* of the trained economy
(ratio ≥ 0.85), confirming it learned the local rule, not slot positions.

## Question

EXP-0001 isolated the depth-3 wall to the **producer**. ADR-0049 makes the gate the
producer-isolated swap column on `d3_p00`:

```text
clone+base >= 0.8 * base+base, near-zero death     (bar ≈ 290 cu✓, base+base ≈ 362)
```

Can a clone of `obs_stock_aware` clear that gate, why does naive cloning fail, and does
the fix hold across same-depth topologies and as a depth-1/2 regression?

## Method

Clone the `obs_stock_aware` producer, save a PPO-loadable checkpoint, score the 2×2 swap
gate (`swap_gate`) on `build_scale_config(0, depth=3)` with `stock_aware`+`proportional`
as baseline, 5 swap seeds, horizon 100. Fidelity is measured on the **decoded command**
(`clone_fidelity`): decode is gated at 0 (toggles fire raw>0, replenish raw>0, masked
dims no-op), so raw MSE can look fine while behavior is wrong. Held-out topologies are
each normalized by **their own base+base** (absolute units are not comparable across
seeds — demand bounds/capacities vary), so the **ratio** is the judgment.

**Slot permutation (Phase 2):** `permute_scale_config` relabels a scale config's slots
without changing the economy — product index π applied across every level (DAG-isomorphic)
and producer slot σ within each level. Because `build_spec` assigns obs/action slots by
**sorted id**, relabeling moves the per-producer obs blocks, the input/open-demand product
slots, and the replenish action dims (and, run through the real env, the expert labels) off
fixed positions. It is a true relabeling symmetry (per-level capacity multiset preserved);
`base+base` shifts only ≤5% because `proportional` routing tie-breaks on sorted id — the
very slot dependence the augmentation hardens against. `scale_topology_set` feeds DAgger
each seed plus `perms` permuted relabelings; `d3_perm` trains on 3 seeds × {base + 2 perms}
= 9 topologies.

## Result

(machine-readable: [`summary.csv`](summary.csv); checkpoints: [`artifacts/`](artifacts/))

**d3_p00 gate (the producer wall):**

| checkpoint | clone path | clone+base | base+base | ratio | dead | decoded match | restore fp/fn | replen exact | verdict |
|---|---|---|---|---|---|---|---|---|---|
| d3_plain_mse | plain full-vector MSE | 217.2 | 361.8 | 0.60 | 0.00 | 0.122 | 485 / 0 | 0.800 | **FAIL** |
| d3_decode_aware | masked-MSE + BCE-toggle DAgger | **373.6** | 361.8 | **1.03** | 0.00 | 0.999 | 0 / 0 | 0.999 | **PASS** |
| d3_multi | + multi-topology (3 topos) | 361.8 | 361.8 | 1.00 | 0.00 | 0.995 | 0 / 1 | 0.993 | **PASS** |
| d3_perm | + slot permutation (3 seeds × {base+2 perms} = 9) | 361.6 | 361.8 | 1.00 | 0.00 | — | — | — | **PASS** |

**Depth-1 / depth-2 regressions** (each depth = its own net; dims differ by depth):

| depth | clone+base | base+base | ratio | dead | verdict |
|---|---|---|---|---|---|
| 1 | 562.0 | 562.0 | 1.00 | 0.00 | PASS |
| 2 | 560.6 | 560.6 | 1.00 | 0.00 | PASS |

**Held-out depth-3 SEEDS** (trained on seeds 0–2; unseen seeds 3–5):

| seed | d3_multi ratio | d3_perm ratio | dead |
|---|---|---|---|
| 3 | 1.22 | 1.19 | 0.00 |
| 4 | 0.94 | 0.97 | 0.00 |
| 5 | 1.00 | 0.92 | 0.00 |

**Held-out PERMUTATIONS of seed 0** (`d3_perm`; same economy, slot arrangements the net
never trained on; training used `perm_seed=1`, these are fresh): the slot-memorization probe.

| perm | clone+base | base+base | ratio | dead |
|---|---|---|---|---|
| p_a | 326.2 | 383.8 | 0.85 | 0.00 |
| p_b | 323.8 | 376.4 | 0.86 | 0.00 |
| p_c | 365.2 | 359.8 | 1.02 | 0.00 |

**L1 trace** (seed 0, `clone+base`) is healthy for both `d3_decode_aware` and `d3_perm`
and matches `stock_aware+base` (cu_total 377, repl_done 618, repl_rej 495); L1 capacity
never approaches 0 (~197–294 through horizon 100), no L0/L2 ballooning.

> A legacy plain-MSE **DAgger** clone (the original motivating failure, pre-fix) scored
> `clone+base = 82.4` — worse than the plain-MSE BC baseline (217.2) because the
> multi-round relabel compounded the gated-decode error. Both plain-MSE paths FAIL.

## Conclusions

1. **The clone failure was loss/labeling, not the expert and not data coverage.** The
   `obs_stock_aware` expert scores `361.8` (== `stock_aware`); the plain-MSE clone scored
   217.2 (ratio 0.60) with **decoded match 0.12** and restoration **false positives 485**
   — it crossed decode's `>0` boundary, restoring far too often. An overfit test (one
   config/seed) confirmed the procedure could not fit the decoded command.

2. **Decode-aware cloning clears the gate (Phase 1 PASS).** Masked MSE on decode-active
   replenish dims + **BCE on the toggle sign** + L1/positive upweight, with DAgger
   relabeling under the base orderer, lifts `clone+base` 217→**373.6** (ratio 1.03),
   decoded match → 0.999, restore FP 485→0, replen exact 0.80→1.00, death 0.00.

3. **It holds across same-depth topologies and as a depth-1/2 regression.** Trained on
   seeds 0–2 it keeps decoded match ≥ 0.99 on each, passes `d3_p00`, and on unseen seeds
   3–5 the **ratio is 0.94–1.22** (clone beats base on the harder seed 3), 0% death.
   Depth-1/2 clones reach ratio 1.00.

4. **Slot permutation completes Phase 2 — the clone learned the local rule, not slot
   positions.** Adding product/producer slot permutations to the training set (`d3_perm`,
   9 topologies) keeps `d3_p00` at ratio 1.00 and held-out seeds at 0.92–1.19, with no
   regression vs `d3_multi`. The decisive probe is held-out *permutations* of the trained
   seed-0 economy (slots the net never saw in that arrangement): ratio **0.85–1.02**, all
   above the 0.80 bar, 0% death. A position-memorizing net would collapse here; instead
   it generalizes. The 0.85/0.86 (below the identity gate's 1.00) is honest *residual*
   slot sensitivity — augmentation mitigates but does not fully erase it.

## Follow-up

- **Phase 2 complete (steps 1–6).** Both multi-topology sampling and obs/action slot
  permutation are implemented and passing; held-out permutations stay above bar. Residual
  slot sensitivity (held-out perm ratio 0.85–0.86 vs identity 1.00) is small and above the
  bar — increase `--perms` or add more train seeds only if a downstream phase needs it.
- **Unseen DEPTH is not yet testable with this policy, by construction.** The producing
  obs/action dims are functions of depth (`num_inputs`, `num_producers`,
  `num_order_products`), so a depth-3 net cannot be loaded on depth 1/2/4. Per-depth
  clones pass, but genuine depth-agnostic behavior needs a set/graph policy — **this is
  the real GNN trigger (ADR-0049 Phase 7), and it is premature until same-depth gaps are
  exhausted.**
- **Quantity rungs (d3_p10/p24/p25) not run.** `warm_start` has no quantity-variation eval
  knob yet; EXP-0001 showed quantity is a *separate survival* axis. Deferred to Phase 7.
- **Phase 3 hard-case mining:** the failure-tag classifier is scaffolded
  (`producer_hard_cases.py`), but given the healthy held-out ratios its necessity for
  same-depth gaps is not yet evidenced — gate Phase 3 on a demonstrated same-depth ratio
  collapse. Tracked in memory `warm-start-gate-result`; builds on EXP-0001.
