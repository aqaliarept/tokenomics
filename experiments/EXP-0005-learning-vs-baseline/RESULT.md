# EXP-0005 — learned agent vs hardcoded baseline (improvement, not imitation)

**Date:** 2026-06-23 · **Drivers:** `tokenomics.eval.signal_check` (headroom audit); `tokenomics.eval.orderer_oracle` (search oracle); `tokenomics.train.orderer_improve` (M2 expert iteration) · **Reproduce:** see [`spec.json`](spec.json)

**Status: CLOSED (accepted M2-v1).** Stage 0 + Stage 1 + Stage 2 (M2) + Stage 2b (teacher audit) +
Stage 2c (M2-v2 coverage probe) DONE. **Improvement claim PROVEN against the named hardcoded
baseline** (M2-v1 beats M0 +6→+35%, mean +15%, CIs, 0 death); **PARTIAL vs the strict random-inclusive
envelope** (ties s6/s8, loses s9). The residual is a clone-CAPACITY gap — Stage 2c showed more training
topologies do **not** close it (they regress it). All cheap levers exhausted (teacher sufficient, more
data hurts, oracle deprioritized); the remaining levers (higher-capacity/topo-conditioned net,
stochastic routing distillation) are larger changes deemed **not worth the spend** at this scope.
**Decision (2026-06-23): accept M2-v1 as the EXP-0005 deliverable and close** — improvement over the
deployed baseline is the proven claim; the envelope-tie is the documented standing limit. EXP-0001–0004
proved the
trained producer can *imitate / generalize* `stock_aware` (a clone is bounded by its teacher — it
cannot prove *improvement*). EXP-0005 targets **policy improvement**: can a learned agent *beat* the
hardcoded baseline? The headroom gate passed decisively (Stage 1): holding `stock_aware` fixed, a
**random orderer beats `proportional` by 14–76%** — general, reachable, orderer-side headroom. A
**directed per-cycle routing search** turns it into clean *deterministic* labels (+32–52% over
`proportional`). **Stage 2 trained M2 (learned orderer + v4 producer) on those labels (Runpod, DAgger
×2) and it beats the named baseline M0 (`stock_aware`+`proportional`) on all 4 held-out topologies by
+6→+35% (mean +15%), 0 death, CIs clearing M0** — a *genuine learned improvement* (M1 = v4+prop ≡ M0
by construction, M2 exceeds it). The deterministic clone **faithfully captured the teacher** (s8:
oracle +37%, M2 +35%) → the **imitability risk is resolved; no stochastic fallback needed.** It does
*not* clear the strict best-of-baseline envelope (ties `random` on s6/s8, loses on s9) — but **Stage 2b
shows the teacher does** (the same L12/K12 search beats the envelope on **all 4** held-out seeds, s9
+8%), so the residual is a **clone/generalization gap (s9 fidelity 84%), not a teacher ceiling.**
**Stage 2c then tested the obvious fix — more training topologies — and REFUTED it:** doubling the
train set (6 → 12 topologies, seeds 0–5 + 10–15) *regressed* every held-out seed against the envelope
(s9 0.91 → 0.87) and raised the fit MSE — a single shared orderer net regresses-to-mean across more
diverse operating points. So the gap is **clone capacity/expressivity, not data coverage**; the next
lever (if pursued) is a higher-capacity / topology-conditioned or stochastic distillation — **not**
more data, **not** a stronger oracle, no PPO. The honest current result: **M2-v1 is the best clone;
improvement over the named baseline is proven, the envelope-tie is the standing limit.**

## Question

EXP-0004 closed the imitation arc: v4 is the best producer clone, it *matches* `stock_aware`
(ratio ≈ 1.00) but by construction cannot exceed it. The unproven claim is the real one — **a
learned agent greatly beats the hardcoded baseline `stock_aware` + `proportional`.** A clone can't
show this. We need an improvement target, and first a proof the benchmark *has* reachable headroom.

## Stage 0 — headroom audit (DONE)

`signal_check` baseline matrix on the depth-3 scale topologies (seeds 0,3,4,5, 5 episode seeds,
horizon 100). Key rows (native, p=1.0):

| seed | stock_aware+proportional cons✓ / cons✗ / cu✓ / capmin | random+random cons✓ / cu✓ | envelope |
|---|---|---|---|
| s0 | 217.8 / **80.6** / 361.8 / 52 | 173.8 / 279.4 | stock_aware+proportional |
| **s3** | **117.2 / 179.8 / 200.8 / 38** | **184.6 / 332.0** | **random+random — baseline BEATEN** |
| s4 | 277.6 / 21.8 / 511.6 / 68 | 177.6 / 318.6 | stock_aware+proportional (near-saturated) |
| s5 | 219.8 / **78.4** / 440.4 / 67 | 196.4 / 391.6 | stock_aware+proportional |

**Findings:**

1. **Demand is NOT saturated.** The baseline rejects 78–180 consumer orders on s0/s3/s5 (s4 is
   near-saturated at 21.8) while `capmin` holds 38–68 — **not a death/capacity-collapse wall**.
   Orders are left on the table → there is *apparent* units headroom.
2. **Reachable headroom is PROVEN (existence proof).** On s3, *random* fulfils **184.6 vs 117.2**
   consumer orders (332 vs 201 units) — a dumb policy strictly beats `stock_aware+proportional`. So
   a learnable policy can do better too. **Gate passed: the benchmark is suitable.**
3. **The lever is the ORDERER.** Holding the `stock_aware` producer fixed, `proportional` is the
   best *deterministic* orderer on every seed (beats `even_split` everywhere), yet it still rejects
   heavily; the s3 random win + EXP-0004's orderer-coupling (`proportional` min-quantity cliff) both
   point at `proportional` as the weak link, with the `stock_aware` producer strong. Prioritize a
   learned orderer (M2) before a learned producer.
4. **M0 should be the best-of-baseline ENVELOPE, not a fixed combo.** `stock_aware+proportional` is
   the best baseline on s0/s4/s5 but is *beaten by random* on s3. Beating a bar that random already
   beats is unimpressive — the honest, strict M0 is the per-topology upper envelope of the baseline
   grid (named reference stays `stock_aware+proportional`).
5. **Headroom is topology-dependent.** Large on s3 (cons✗ 180), moderate on s0/s5 (~80), small on
   s4 (~22). The success criterion must therefore be **blended**: units where headroom exists;
   efficiency (rejects / waste / restoration / shock survival) where units saturate.

## Stage 1 — orderer ceiling probe (DONE)

The domain is deterministic and `deepcopy`-able (~3 ms), so a search/MPC oracle is feasible. Two
probes:

**1a. Producer-vs-orderer isolation (the decisive result).** Holding the `stock_aware` producer
fixed and swapping the orderer (cu✓ mean ± stdev over 5 episode seeds,
`logs/orderer_headroom_random_vs_proportional.txt`):

| seed | proportional | even_split | obs_price | **random** | random/prop |
|---|---|---|---|---|---|
| 0 | 361.8 ± 12 | 160.4 | 187.6 | **448.2 ± 10** | **1.24** |
| 3 | 200.8 ± 6 | 84.4 | 118.4 | **352.8 ± 14** | **1.76** |
| 4 | 511.6 ± 21 | 292.0 | 321.0 | **513.6 ± 18** | 1.00 (saturated) |
| 5 | 440.4 ± 12 | 286.2 | 299.8 | **536.2 ± 11** | **1.22** |
| 6 | 424.8 ± 30 | 354.0 | 301.4 | **483.4 ± 18** | **1.14** |
| 7 | 438.6 ± 14 | 354.0 | 392.0 | 434.0 ± 7 | 0.99 (tie) |
| 8 | 403.4 ± 31 | 434.0 | 317.8 | **531.4 ± 11** | **1.32** |
| 9 | 405.4 ± 27 | 306.0 | 220.8 | **504.0 ± 11** | **1.24** |

With a *random producer* instead, everything collapses (s3: 22–73). So:

- **The headroom is entirely orderer-side and general** — a *random* orderer beats `proportional`
  by **14–76% on 6/8 topologies**, CIs non-overlapping (s3: 200.8±6 vs 352.8±14). s4 is saturated,
  s7 a tie. The `stock_aware` producer is near-optimal; **the orderer is the whole opportunity.**
- **`proportional` is systematically suboptimal** — it concentrates an order's tokens/quantity onto
  few producers, creating bottlenecks / TTL-2 failures; random *spreads load* and avoids the trap.
  A learned orderer that load-balances on state should beat both.
- **The success bar is already shown achievable** — random clears the ≥10–20% target. M2 (learned
  orderer) is now strongly motivated and the producer can stay fixed at `stock_aware`/v4.

**1b. Per-cycle MPC over deterministic orderers — too narrow (negative result).** A receding-horizon
oracle (`tokenomics.eval.orderer_oracle`) that picks, per cycle, the best of
{even_split, proportional, obs_price} by `lookahead`-cycle rollout scored **0.95 on s3** — it does
*not* beat the baseline. The winning behavior (random ≈ 1.76×) is **outside the deterministic
candidate set**, and short greedy lookahead is myopic. ⇒ switching among fixed orderers is the wrong
oracle; search the routing itself.

**1c. Mechanism — it's per-cycle re-routing, not a better fixed shape.** A *fixed* random split
reused every cycle (open-loop) is **bad** (s0/3/5/8 = 169/125/110/250, ≤ proportional), while
*re-sampled* random is great (448/353/536/531) — `logs/orderer_mechanism_probe.txt`. So no single
fixed routing (uniform, throughput-weighted, or arbitrary) wins; the gain is **temporal
diversification** — re-routing each cycle escapes the bottleneck ruts a fixed policy falls into.
⇒ the headroom is capturable only by a **state-reactive** orderer, not a static heuristic; and the
test of *learnability* is whether a *directed* per-cycle search beats `proportional` (random proves
it's reachable but isn't a clean, reproducible label source).

**1d. Directed routing search — captures it, deterministically (the label source).**
`orderer_oracle --samples K`: each cycle, score the baselines **plus K freshly-sampled fixed routing
vectors** by `lookahead`-cycle rollout, commit the argmax (reproducible — sample RNG seeded by
(seed, cycle)). With lookahead 12, K=12 (`logs/search_oracle_l12_s12.csv`):

| seed | search oracle cu✓ | baseline | uplift | sample_wins/ep |
|---|---|---|---|---|
| 0 | 499 | 377 | **+32%** | 68 |
| 3 | 317 | 208 | **+52%** | 94 |
| 5 | 619 | 438 | **+41%** | 67 |
| 8 | 615 | 448 | **+37%** | 58 |

Beats `proportional` by **+32–52% on all four, 0 death**, and `sample_wins/ep` 58–94 means a *searched*
routing (not a baseline) wins the majority of cycles — the improvement is genuinely found by
per-cycle search. It is **deterministic and reproducible**, so unlike random it is a clean **DAgger
label source** for M2. (lookahead-12 doesn't fully reach random's 1.76× on s3 — still mildly myopic;
more samples / longer rollout would close it, but +52% already clears the bar.) **M2 is de-risked:**
a state-conditioned routing teacher that beats the baseline by ~40% exists and is labelable. Tool:
`eval.orderer_oracle --samples`.

## Stage 2 — M2: learned orderer beats the baseline (DONE)

`tokenomics.train.orderer_improve` (Runpod CPU, ~22 min). Producer **fixed at the EXP-0004 v4 clone**;
the orderer is expert-iterated toward the search oracle (lookahead 12, K 12, candidates
{even_split, proportional, obs_price}) — **DAgger ×2** (round 0 on-oracle BC, round 1 relabel of the
clone's visited states), masked-MSE over the decode-active dims, train topologies seeds 0–5 × 3
episode seeds (27k → 54k labels). Saved M2 = v4 producer + learned orderer. Held-out eval on **seeds
6,7,8,9** (5 episode seeds), M2 vs M0 (`stock_aware+proportional`) and the best-of-baseline envelope
(`stock_aware`×{prop, even, random} + `random+random`); `artifacts/m2_v1.pt`,
`artifacts/m2_v1_heldout.csv`:

| seed | M2 cu✓ ±95%CI | M0 cu✓ | M2/M0 | envelope cu✓ (best) | M2/env | dead | rej✗ |
|---|---|---|---|---|---|---|---|
| 6 | 478.6 ± 15.0 | 424.8 | **1.13** | 483.4 (`sa+random`) | 0.99 | 0 | 29 |
| 7 | 465.4 ± 19.4 | 438.6 | **1.06** | 438.6 (`sa+proportional`) | 1.06 | 0 | 41 |
| 8 | 542.8 ± 18.7 | 403.4 | **1.35** | 531.4 (`sa+random`) | 1.02 | 0 | 68 |
| 9 | 456.0 ± 14.7 | 405.4 | **1.13** | 504.0 (`sa+random`) | 0.91 | 0 | 100 |

<sub>Regenerated locally from `m2_v1.pt` (canonical/reproducible); the pod-run CSV differed ≲2%
(cross-platform float drift in the actor outputs → routing) with the same verdict.</sub>

**Findings:**

1. **Improvement over the named baseline is PROVEN.** M2 beats M0 (`stock_aware`+`proportional`, the
   deployed hardcoded baseline) on **all 4 held-out topologies by +6→+35% (mean +15%)**, 0 death. M0
   is deterministic (seed-invariant), so M2's 95% CI clears it on every seed (s7 the tightest:
   446 > 439). Since **M1 = v4 producer + `proportional` ≡ M0** by construction, this is a *genuine
   learned improvement*, not the imitation ceiling — the EXP-0005 question is answered **yes** for the
   named baseline.
2. **The deterministic clone faithfully captured the teacher → imitability risk RESOLVED.** On s8 the
   Stage-1d oracle was +37% over `proportional` and M2 lands **+35%**; M2 retains most of the directed
   search's gain across seeds. So the orderer observation **is** sufficient to learn the search-improved
   routing *deterministically* — the documented open risk (temporal diversification needing a stochastic
   policy) did **not** bite; no Gaussian-sampling fallback required.
3. **Against the strict random-inclusive envelope, M2 is PARTIAL (0/4 at ≥10%).** It *ties / edges*
   the envelope where `random` is the winner (s6 → 0.99, s8 → 1.02), *beats* it where the envelope is
   M0 itself (s7 → 1.06), and **loses on s9 (0.91** — `random` 504 > M2's CI upper 471). M2 reached
   `random`'s level but did not clearly surpass it.
4. **The residual gap is a CLONE / GENERALIZATION issue, NOT a teacher ceiling** (Stage 2b audit
   below). The search teacher itself, at the *same* L12/K12 setting on the *same* v4 producer, **beats
   the strict envelope on all 4 held-out seeds — including s9 (+8%)**, the one M2 lost. So the teacher
   is already strong enough; M2 falls short because the deterministic clone retains only **84% of the
   teacher on s9** (92–96% on s6–8). The earlier "teacher-myopia" reading was wrong — it extrapolated
   from Stage 1d, which measured vs `proportional` (not the envelope) on different seeds. The lever is
   the **clone's held-out generalization**, not a stronger oracle.
5. **Efficiency:** 0 death on all seeds; rejects track topology hardness (s6 29 → s9 100); investment
   churn ~130–145 (the v4 producer, unchanged). M2 buys more fulfilled units at no survival cost.

**Verdict: M2-v1 = PARTIAL PASS.** Learned orderer beats the deployed hardcoded baseline (+15% mean,
CIs, 0 death) — improvement is real and deterministic-clonable. It does not yet beat the strict
random-inclusive envelope, but **the teacher does** (Stage 2b) — so the limiter is **clone
generalization to unseen topologies** (s9 fidelity 84%), not the teacher. Next move is a **better
clone (M2-v2): more training topologies / label coverage + multi-seed**, re-using the current L12/K12
teacher — no stronger oracle, no stochastic orderer, no PPO.

## Stage 2b — teacher-strength audit (DONE): the teacher beats the envelope, the clone is the gap

`orderer_oracle --prod-ckpt <v4> --envelope` on held-out seeds 6,7,8,9 (5 episode seeds, L12/K12) —
does the *search teacher* (what M2 imitates) beat the envelope on the held-out suite, especially s9?

| seed | oracle cu✓ | vs M0 (v4+prop) | envelope cu✓ (best) | **oracle/env** | M2/env (clone) | clone retains |
|---|---|---|---|---|---|---|
| 6 | 499.0 | 1.09 | 483.4 (`sa+random`) | **1.03** | 0.99 | 96% |
| 7 | 490.6 | 1.12 | 438.6 (`sa+proportional`) | **1.12** | 1.06 | 95% |
| 8 | 590.8 | 1.32 | 531.4 (`sa+random`) | **1.11** | 1.02 | 92% |
| 9 | 545.0 | 1.44 | 504.0 (`sa+random`) | **1.08** | 0.91 | **84%** |

`oracle ≥ envelope on 4/4`, 0 death. **Decisive:** the current L12/K12 teacher already clears the
strict bar on every held-out topology, including s9 (+8% over `random`). So **EXP-0005's residual is a
cloning/generalization gap, not a teacher-strength gap** — the clone reproduces ≥92% of the teacher on
s6–8 but only 84% on s9, and that sag is exactly where M2 ties/loses the envelope. Per the
pre-registered decision rule (*"oracle beats envelope but the clone fails → clone/generalization
issue"*), the next lever is the **clone**, not the oracle: the lookahead/K grid is **deprioritized**
(teacher already sufficient). Artifact: `artifacts/oracle_audit_L12_K12.csv`.

## Stage 2c — M2-v2 coverage probe (DONE): more topologies REGRESS the held-out clone

The Stage 2b decision rule pointed at the clone, and the pre-registered first lever was *more
training topologies* (the coverage fix that worked for the EXP-0004 producer). M2-v2a tests it
directly: identical recipe to M2-v1 (same v4 producer, same L12/K12 teacher, DAgger ×2) but **train
topologies doubled 6 → 12** (seeds 0–5 **+ 10–15**), label-seeds 3, held-out 6,7,8,9 × **20** episode
seeds (Runpod CPU). Canonical local re-eval of `artifacts/m2_v2a.pt`
(`artifacts/m2_v2a_heldout.csv`):

| seed | M2-v2a cu✓ ±95%CI | M2/M0 | M2/env | env (best) | **vs M2-v1 M2/env** | dead |
|---|---|---|---|---|---|---|
| 6 | 456 ± 10 | 1.10 | **0.96** | `sa+random` | 0.99 → **0.96** | 0 |
| 7 | 459 ± 6 | 1.05 | 1.05 | `sa+proportional` | 1.06 → 1.05 | 0 |
| 8 | 537 ± 8 | 1.35 | **1.00** | `sa+random` | 1.02 → **1.00** | 0 |
| 9 | 435 ± 28 | 1.08 | **0.87** | `sa+random` | 0.91 → **0.87** | ~0 |

**Findings:**

1. **Coverage is REFUTED as the lever.** Doubling the training topologies *regressed* the held-out
   envelope ratio on **all 4 seeds** (s6 0.99→0.96, s8 1.02→1.00, s9 0.91→0.87) and M2/M0 on 3 — the
   opposite of the hypothesis. The round-1 fit MSE also *rose* (M2-v1 → M2-v2a) rather than fell.
2. **Mechanism — single-net regression-to-mean.** One shared `GaussianActor` distilling the search
   labels of 12 heterogeneous topologies fits a blander average policy than one distilling 6; the
   extra operating points dilute per-topology fidelity instead of teaching transfer. (Contrast
   EXP-0004's *producer* clone, where coverage helped — the producer rule is more topology-invariant;
   the orderer's winning routing is more topology-specific, so it does not pool.)
3. **⇒ the s9 gap is clone CAPACITY/EXPRESSIVITY, not data.** With the teacher already clearing the
   bar (Stage 2b) and *more data hurting*, the only remaining clone levers are **higher capacity /
   topology-conditioning** or **stochastic routing distillation** — both larger changes, deferred. The
   pre-registered "if it still sags on s9 → genuine clone capacity (revisit architecture/stochastic
   distillation *then*, not now)" branch is the one we landed on.
4. **M2-v1 stays the best clone / the EXP-0005 deliverable.** M2-v2a does not dominate it anywhere;
   M2-v1's held-out numbers are at least as good on every seed. Improvement over the *named* baseline
   remains proven (M2-v2a still beats M0 on all 4); the envelope-tie is the honest standing limit.

<sub>Run-to-run note: `--eval-only` does not pin BLAS threads, so float-reduction nondeterminism over
100 cycles gives ~3% run-to-run drift in cu✓ (within the CI bands; verdict unchanged). The table uses
the saved-artifact CSV.</sub>

## Models (planned)

- **M0** — `stock_aware` + `proportional` (named baseline); bar = best-of-baseline envelope/topology.
- **M1** — EXP-0004 **v4 producer** + `proportional` (best clone; *matches*, does not beat — the
  imitation ceiling, the control).
- **M2** — **learned orderer** (search-improved labels) + v4 producer — tests the orderer lever.
- **M3** — learned producer + learned orderer, both from search-improved labels — the full claim.

## Success criterion (strict, with CIs)

- **Primary:** M3 (or M2) beats the M0 envelope by **≥10–20%** on consumer units/orders fulfilled,
  on a **held-out** suite (seeds 6,7,8,9), mean ± CI across episode seeds.
- **Secondary (where units saturate, e.g. s4):** *same* completed units with materially fewer
  replenishment rejections / less wasted inventory / less restoration cost / better survival under
  harder shocks.

## Follow-up (no PPO-first, no adversarial generation)

* **Stage 1 + 2 DONE.** Headroom is orderer-side and reachable (random +14–76%); a directed search
  turns it into deterministic labels (+32–52%); **M2 (learned orderer + v4 producer) beats the named
  baseline M0 by +6→+35% (mean +15%) on held-out seeds 6,7,8,9, 0 death, CIs clearing M0** — improvement
  proven, and the deterministic clone faithfully reproduced the teacher (open state-decodability risk
  RESOLVED, no stochastic orderer needed). Tools: `eval.orderer_oracle --samples`,
  `train.orderer_improve`. Artifacts: `m2_v1.pt`, `m2_v1_heldout.csv`.
* **Stage 2b DONE — the teacher already clears the bar; the gap is the clone.** The L12/K12 search,
  on the same v4 producer, beats the strict envelope on **all 4 held-out seeds** (s6 1.03 … s9 1.08),
  0 death — while M2 retains only 84% of it on s9. So this is a **clone/generalization issue, not a
  teacher one** (`artifacts/oracle_audit_L12_K12.csv`).
* **Stage 2c DONE — coverage REFUTED; the gap is clone capacity, not data.** M2-v2a (train topologies
  6 → 12, same teacher/recipe, held-out 6–9 × 20 seeds) *regressed* the envelope ratio on all 4
  held-out seeds (s9 0.91 → 0.87) with a higher fit MSE — a single shared orderer net regresses-to-mean
  across more topologies. Artifacts: `m2_v2a.pt`, `m2_v2a_heldout.csv`.
* **DECISION (2026-06-23): accepted M2-v1; EXP-0005 CLOSED.** With the teacher sufficient and both
  *more data* (Stage 2c) and *stronger oracle* (Stage 2b, deprioritized) ruled out, the s9 envelope gap
  is genuine clone capacity/expressivity. The remaining levers — **(B) higher-capacity/topology-
  conditioned orderer**, **(C) stochastic routing distillation** (distill the teacher as a *distribution*
  over routings, capturing the temporal-diversification gain a deterministic mean can't) — are larger
  changes judged **not worth the spend** at this scope. **Outcome: M2-v1 is the deliverable** —
  improvement over the deployed named baseline is the proven EXP-0005 claim; the random-envelope tie
  (clone capacity, not teacher) is the documented standing limit, reopenable via (C) if the envelope
  bar later matters. *No* PPO, *no* stronger oracle, *no* more data was pursued.
* **M3 (joint producer+orderer) only if M2-v2 leaves producer-side headroom** — Stage 1a says it
  likely does not (the `stock_aware`/v4 producer is near-optimal). Links: [[exp0005-headroom]],
  [[exp0004-rung-fidelity]].
