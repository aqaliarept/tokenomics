# Design: producer-side learning on top of buy-ahead (M5)

## Context

- **What the producer evaluation found** (M4 7.12–7.22b):
  - The gap to the atlas bounds is on the supply side.
  - The dominant loss is inputs ordered but not delivered in time (EXP-0074: b2 ≈ 75% of lost
    consumer units).
  - Buying ahead moves it: +46% at k* = 4 (EXP-0076 / 0077).
  - Inventory room is the binding resource. Re-allocating it does not help, and expanding it is
    the only extra lever found (EXP-0078).
  - The residual late loss is the one-cycle delay between accepting an order and ordering its
    input, which the replenishment command cannot remove.
- **What it means for learning:** the hand-written producer has three blunt knobs that a learned
  correction can set per input and per state:
  - one k for every input and producer;
  - inventory expansion by a fixed trigger;
  - a fixed priority between buffer stock and waiting orders.
- **M4 step 2's lessons for the learner** (DIAGNOSTICS 25 / 27 / 28):
  - ES on a per-order residual needed minibatch 24 to see a direction (25);
  - it needed ≥ 40 generations to clear its training floor (27);
  - on the MWW base it hit real side-effect damage, not noise (28).
  These shape D3, D5 and D6.

## Goals / Non-Goals

**Goals.**
- Beat `entity_planner_buyahead_k4` + MWW on held-out worlds, at 12 producers (deciding) and at
  42 (no transfer failure), without extra deaths or run-down beyond the measured margin.
- A policy whose parameter count does not depend on the number of producers, inputs or the
  topology (train at 12, read at 42).

**Non-Goals.**
- Learning the orderer, or both roles jointly, in this phase (D1).
- A GNN in this phase (D4).
- Changing `domain/`, the gate harness, the roster or the guards.
- A TTL change (a world-design decision, EXP-0075).

## Decisions

### D1. Single-role producer learning first; joint (potential game) later, gated
- **Why the producer first.**
  - Every simple orderer ties on the loaded worlds (M4 steps 1–2, notes/producer-eval.md step 1).
  - Learned corrections to MWW found no headroom without guard damage (DIAGNOSTICS 28).
  - The supply side carries the measured gap (EXP-0074–0078).
- **Why alone.** Learning one role against a frozen orderer gives the learner a stationary
  environment and halves the variance sources. Joint learning from scratch failed on
  coordination at depth 3 (EXP-0001).
- **The joint phase** (the potential game with a topology-agnostic GNN, this change's earlier
  version) is phase 2. It starts only after a `confirmed` held-out producer read (L6) and a
  diagnostic showing orderer-side headroom given the learned producer, e.g. a supply trace whose
  `c_outbid` / routing classes then dominate.

### D2. The policy: a residual on the base producer over shared per-row features

**Base (EXP-0079, D8 applied): `entity_planner_buyahead_k4_rf3`** (buy-ahead k = 4 + early inventory expansion; +11% over k = 4 at 12 producers). Below, "the base" and "k = 4" in the residual's definition mean rf3: its expansion trigger is the residual's starting trigger, and θ = 0 equals rf3.

- **The base:** `entity_planner_buyahead_k4`. The residual changes exactly three things, with
  everything else the base's (restoration, upgrade, the planner's input priority):
  - **Buffer per input:** k_g = 4 · exp(clip(θ_k · φ_g)), applied through `buyahead_view`
    (fractional k allowed).
  - **Buying beyond the target:** extra_g = ⌊max(0, θ_b · φ_g)⌉ units on top of the planner's want,
    within headroom and the per-cycle ceiling.
  - **Expansion timing:** expand when θ_e · ψ > 0. The base trigger (headroom 0) stays a
    sufficient condition.
- **Features:** φ_g per input row, ψ per producer. All agent-local and past-only (ADR-0058; the
  producer's own `ProducerEntityView` plus a per-episode agent-local history like 7.22's EMA,
  weak-keyed by the system):
  - inventory, inbound and `blocked_reservation_qty`, normalised by recent use;
  - recent use (the completed-units EMA × recipe);
  - headroom / limit;
  - the input's recent late count;
  - own recent completion latency;
  - recipe depth.
- **Initialisation:** zero, so θ = 0 equals the base exactly (tested byte for byte, as M4's
  `MWWBasePolicy`).
- **Size:** about 30 parameters. The same θ serves every producer and every input row, so the
  count is independent of size and topology.

### D3. The learner: ES with paired episodes, not policy gradient
- **Pre-registration (task 2.3, notes/2.3.md):** `es_prereg_m5_v1.json` (`67721b38…`).
  - Features are standardised by M4's median / IQR normaliser (clip ±5), fitted on rf3's rows
    from the bar and pinned by SHA.
  - σ = 0.05; the L2 probe also runs 2σ = 0.1, and M4's step reading decides the pilot's σ.
  - The monitor follows M4: the centre and rf3 at replicates 0 and 1, every 10 generations and at
    generation 0.
  - Adam lr / weight decay and L5's 8 episodes per world are carried from M4 (flagged).
- **Fitness (decided 2026-11-05, supervisor; notes/2.2.md):** consumer units only, relative to
  the bar (rf3 + MWWk on the same world-episode and key). There is no holding term, unlike M4's
  `order_es`. The D5 guard (run-down / starvation, relative to rf3) is unchanged. Holding units
  per cycle and the investment rejection rate are reported in every generation and read, never
  deciding. The pre-registration's sanity check `stock_heavy` flags a selected θ whose held stock
  exceeds rf3's by > 50%. It is reported only, and goes to the user.
- **Why ES:**
  - The decisions act through multi-cycle delays (an input order resolves 1–4 cycles later, losses
    land at TTL), so per-decision credit for policy gradient is noisy and needs a critic. M4 never
    validated one.
  - ES scores whole-episode consumer units directly.
  - The domain is deterministic given the draw key, so paired episodes with common keys (M4's bar
    precompute) cancel world noise.
  - θ is small (D2), the regime where ES is efficient.
- **What M4 taught, applied up front:**
  - minibatch 24 world-episodes (DIAGNOSTICS 25);
  - ≥ 40 generations in the plan (27);
  - antithetic pairs, rank-normalised fitness, Adam on the ES gradient (`order_es`).
- **The fallback is policy gradient** with paired-episode baselines. It is considered only if the
  probe (L2) reads `no_signal` and its diagnosis points at the estimator, not the policy class.

### D4. Per-row features, not a GNN, in this phase
- **Why not a GNN now.** Topology-agnostic is already met by shared per-row parameters (D2): one θ
  over input and producer rows, any number of producers. A GNN adds supplier-state messages, but
  a producer's information set (ADR-0058) holds no supplier state, and the diagnosed losses act
  through its own buying.
- **When a GNN would come in:** only if L5 reads `plateau` and a diagnostic shows supplier state
  would separate the late losses that the per-row features cannot. That would be a new task, with
  the supplier information set reviewed against ADR-0058.

### D5. The guard: margins measured on this base, before training
- **Why measure first.** Buy-ahead k = 4 already uses MWWk's run-down allowance (+5 of 86 against
  `entity_planner`, EXP-0077), and M4's re-based step hit real guard damage (DIAGNOSTICS 28).
- **How.**
  - L1 replays the base at fresh draw keys and sets the relative disqualification margin
    (run-down and starvation; 2 × √2 × pooled within-world SD, the M4 v4 formula) on this base.
  - The training guard is relative to the base (k = 4), not to `entity_planner`.
  - Target: ≤ 5% self-flags.

### D6. The ladder and pre-registered readings (frozen in `es_prereg_m5_v1.json` before L2)
- **L1, base and noise floor:** bar precompute for rf3 + MWWk on M4's pinned training set (D7);
  r0 reproduces the bar's rf3 rows, and rf3 replayed on 2 EXP-0057 world-episodes equals EXP-0079's
  rows; margin per D5.
- **L2, gradient probe:** cosine of the ES gradient across two disjoint 24-episode minibatches.
  - `signal` ≥ 0.3, `weak` ≥ 0.1, else `no_signal` (DIAGNOSTICS entry, stop).
- **L3, pilot:** 20 generations, seed 0. `feasible` if g_tr > F_tr; else `unresolved` (M4's
  DIAGNOSTICS 27 check first).
- **L4, main:** to 40 generations (resume), seed 0; then seed 1 to 40.
- **L5, validation at 12 (deciding) and 42 (reported):**
  - `success` if θ − base exceeds the floor on the validation split for both seeds, with no extra
    deaths and guard-margin side effects;
  - `plateau` if it is within the floor;
  - `kill` if it is below the floor or deaths increase;
  - `transfer_failure` if 12 succeeds and 42 is below its floor.
- **L6, held-out read** (atlas held-out worlds, fresh seeds; ledger key `held-out/m5-producer`,
  pending entry committed before launch, M4 design D9's C0 / C1 rule):
  - `confirmed` if θ beats rf3 + MWW beyond the floor at 12 and is not below the floor at
    42, with no extra deaths and side effects within the margin;
  - else `not_confirmed` (DIAGNOSTICS, user).

### D7. Data, splits and pairing
- **Training (supervisor decision 2026-11-05, notes/2.2.md):** M4's pinned ES training set and
  schedule (EXP-0058 `bar/`). That is 64 worlds at 12 producers (the 23 loaded atlas v2 train
  worlds plus 41 fresh world-space draws), 128 world-episodes with fresh episode seeds, the
  16-world-episode monitor set, the per-seed minibatches and the probe's two minibatches.
- **Why:**
  - the pool size minibatch 24 and a 16-episode monitor need (DIAGNOSTICS 25 / 27);
  - the validation / held-out splits are disjoint by construction;
  - none of the EXP-0057 world-episodes that selected rf3 (EXP-0079) are reused.
- **L1** builds a new rf3 + MWWk bar on those world-episodes.
- **L5 / L6:** the atlas v2 validation split for L5, held-out for L6, at 12 and 42.
- **Pairing:** MWWk at DrawKey(world, episode, replicate) (replicate 0 for training members;
  re-keyed replicates for the noise floor). The observer is off in training (speed) and on in the
  L5 / L6 reads (loss classes reported).
- **TTL 2 throughout.**

### D8. A cheap pre-step: "expand earlier" alone on k = 4
- **Why.** EXP-0078 left it untested and it may raise the bar the learner must beat.
- **How.** It is a hand-written arm, k4 + part 3 only: one `EXTRA_PRODUCING` entry and one
  supply-trace run (≈ 1 vCPU-h). If it reads `helps` against k4 (the 7.22b rule), it becomes the
  base instead of k4. D2's residual then sits on it, and its θ = 0 equivalence is to it.

### D9. Budget: tranche M5-P, ≈ 80 vCPU-h requested
- **Priced from M4's measured runs:** EXP-0059 pilot 7.8, EXP-0060 main ≈ 11–17 per seed to 40
  generations, EXP-0069 probe 3.0. Producer episodes run ≈ 1.3× an `entity_planner` episode
  (EXP-0076–0078).

  | run | estimate | stop |
  |---|---|---|
  | D8 expand-only arm | 1 | 3 |
  | L1 bar + noise floor | 3 | 6 |
  | L2 probe | 4 | 8 |
  | L3 pilot (20 generations) | 10 | 20 |
  | L4 main, seed 0 (to 40 generations) | 10 | 20 |
  | L4 main, seed 1 (40 generations) | 20 | 40 |
  | L5 validation | 6 | 12 |
  | L6 held-out | 6 | 12 |
  | **total on the success path** | **≈ 60** | |

- **The request is 80**, which leaves ≈ 20 for one failed pod per stage (EXP-0075 / 0077) and a
  re-run.
- **Pod practice:** per run, hard stop = 2 × the estimate, ≤ 40; launch-fit on billed spend; pods
  pinned to a known-good data center (never unpinned).
- **A timing probe** (task 3.1) re-sizes the plan before L1.

## Risks / Trade-offs

- **The guard binds** (as DIAGNOSTICS 28). Mitigation: the D5 margin on this base, and the residual
  starts at the base.
- **k* sits at the sweep's edge,** so a larger uniform k might beat both the base and the learner.
  Mitigation: a per-input k_g lets the learner find it. L5 reports the learned mean k.
- **Over-fitting to 12-producer worlds.** Mitigation: shared row parameters, a 42-producer
  validation read, and `transfer_failure` registered.
- **The late loss the command cannot reach** (the one-cycle delay) caps the gain. Mitigation:
  none in this phase. It is reported as the remaining class.

## Migration Plan

The change is new code under `train/` and `eval/`, and touches no existing behaviour. It is
archived after the ladder's last task writes HISTORY §15.

## Open Questions (for the user)

1. **Phase order:** approve single-role producer learning first, with joint potential-game / GNN
   learning as a gated phase 2 (D1)?
2. **The learner:** ES as the primary (D3), with policy gradient only as a diagnosed fallback?
3. **Features:** shared per-row features now (D2 / D4), with a GNN only if a later diagnostic
   calls for it?
4. **The base:** run D8 ("expand earlier" alone) first, and re-base on it if it helps?
5. **Budget:** release tranche M5-P ≈ 80 vCPU-h (D9)?
6. **Guard:** the training guard is relative to buy-ahead k = 4 (the base), with the margin
   re-measured, not relative to `entity_planner` (D5)?
7. **A buffer sweep beyond k = 4** (k = 5–6, ≈ 1 vCPU-h) before learning, since k* sat at the
   edge?
