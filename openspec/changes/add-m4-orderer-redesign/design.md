# Design: M4 orderer redesign (gated ladder)

Revision 3 (2026-10-30): addresses review round 2 (7 concerns), on top of revision 2 (review
round 1, 16 concerns, and the user's information-set decision, ADR-0058).

## Context

See proposal.md (Why). The facts this design rests on:

- **Deployed orderer.** `train/orderer_scorer.py` `OrdererScoreModel` scores each producer row of
  a (source, product) stream from 9 product features and 31 producer-row features, once per
  cycle. `scorer_ordering_policy` tiles the row over `k_stream` slots. `env/decode.py`
  `decode_ordering` then calls `env/allocator.py` `allocate_suborders` per order: a softmax over
  eligible producers, then a largest-remainder split of quantity and tokens. Orders beyond
  `k_stream` even-split (ADR-0054). Training was regression onto `teacher_logits` = log
  `makeable_qty_now` (EXP-0012). softmax(log m) = m / Σ m, so a faithful copy of the target,
  sampled per order, is `makeable_weighted_whole` (MWW). The EXP-0051 addendum shows the scorer
  reproduces its target's behaviour (split degree 1.90 vs 2.03; median −21.0% vs −25.7%).
- **Diagnosed cause.**
  - EXP-0056 (`joint`, 3 of 3 worlds): the gap to `round_robin` closes only with whole orders and
    `round_robin`-like receivers together.
  - `lead_concurrent_top_share` (0.98–0.99) groups the concurrent orders of a product across all
    ordering agents (`eval/teacher_audit.py` 1359–1362).
  - Each stream agent serves one (source, product). A consumer places at most one order per cycle
    (`domain/consumer.py`), and `incoming` resets every cycle. So a consumer-stream agent rarely or
    never holds two orders in one turn. The concentration is therefore mostly **across agents**:
    every agent of a product reads the same shared rows and picks the same top producer. The
    pre-pod trace (task 1.3) confirms this composition.
- **Information visible to an orderer (ADR-0058).**
  - Every input was written before cycle t's `start_ordering`: previous cycles' outcomes plus
    cycle t's consumer orders, capacity hits and producing phase (`domain/system.py` 458–468,
    518–525). Nothing written by any ordering agent in cycle t is an input.
    `ProducerStats.requested_load`
    reads `requested_capacity`, written after the previous cycle's auction (`domain/metrics.py`
    `producer_requested_load`; `domain/producing_agent.py` line ≈ 503). It is last cycle's value.
  - `StreamViewCache` (`domain/orderer_view.py`) freezes every producer row for the whole
    ordering phase. Routing only appends to `p.bids`, which no row reads.
  - There is **no within-cycle cross-agent signal**, and ADR-0058 makes that intended: orderers
    decide from what was written before `start_ordering`, plus their own orders, own history and
    own sampling stream.
  - The shared producer-product telemetry EMAs are pooled over all agents, so they are identical
    for every agent of a product. No input today differs between two agents of the same product.
- **Atlas v2** (EXP-0052 `reclassified/atlas_v2.json`, content hash `220886a7…`). 96 selected
  worlds: train 60 / validation 16 / held-out 16 / held-out-region 4. Each world carries its
  record, ρ, choice level, `bar`, `se` and `noise_floor`. floor_w = max(3·SE_w, 0.01·bar), with
  SE_w = sqrt(σ_ep² + Σ_stoch σ_rep²/R) / sqrt(n_w) from pooled-MAD spreads per size × choice
  cell (`eval/atlas.py` `standard_error`, `noise_floor`).
  - Loaded (ρ > 0.28), non-collapse counts, read from `reclassified/worlds.csv`:

    | Split | at 12 producers | at 42 producers |
    |---|---|---|
    | train | 23 (median bar 1,696) | 14 |
    | validation | 9 (median bar 1,727, mean bar 1,831) | 3 |
    | held-out | 8 (median bar 1,289) | 2 |

- **Cost** (sizing.md §EXP-0051, local s per horizon-100 episode; pod ≈ 0.646 × local).
  - 12 producers: `round_robin` 0.8–3.1 s, trained 1.7–7.3 s (4.5 s mean on loaded worlds).
  - 42 producers: `round_robin` 4.5–29 s, trained 8.5–61 s.
  - The deployed path is not superlinear up to 42 (EXP-0055).
- **Archive.** `archive-1` (`7aeef1f`) holds `domain/graph_view.py`, the `graph_primitives_v1`
  encoder (`eval/graph_value_router_dump.py`; relations include `pending_bid`, current-cycle
  bids, and `reservation`) and the MPNN (`train/graph_router_probe.py`). The graph branch closed
  on label instability (EXP-0040 `SEARCH_ONLY`, winner overlap 0.60–0.74 < 0.80), not on
  architecture.
- **DIAGNOSTICS entries 6, 9, 11, 12** (memorising, pod stalls, learnable teacher signal,
  target-is-a-function) were removed by `18c717d`; task 1.1 restored them verbatim into
  `/srv/research/tokenomics/docs/DIAGNOSTICS.md`, and this change cites them there as
  DIAGNOSTICS 6, 9, 11 and 12.

## Goals / Non-Goals

**Goals:**
- A ladder in which each step has a reading rule fixed before its run, and no step starts until
  the previous step's reading licenses it. Each licence names the diagnosed cause the step
  addresses (root-cause rule).
- A per-order orderer, under ADR-0058's information set, whose agents diverge through agent-local
  history and sampling. Trained on whole-episode outcomes and judged on loaded atlas worlds
  against paired references and floors.
- Spend sized from probes, inside a user-approved budget, under a launch-fit rule.

**Non-Goals:**
- Any same-cycle cross-agent signal, coordination protocol or learned agent identity
  (ADR-0058).
- A sealed gate read, roster v3, registry changes, a 42-producer teacher read (user decisions).
- Training the producer role. M5–M7. Worlds above 42 producers. Explaining what limits `d2`.
- **Visible outage status (user decision 2026-10-30).** Orderers do not see producer outage
  status in M4. ADR-0055's "consequences only" stands: agents demote failing producers from
  their own failure history (D5). An explicit, visible producer outage status for all policies,
  the hand-written ones included, is deferred to its own ADR (M6 shocks II or a separate small
  change).

## Decisions

### D1. Ladder structure and gates

```
pre-pod local checks (group 1) ──▶ step 1 decoder check (EXP-0057)
   decoder | decoder_partial ──▶ gradient-signal probe (EXP-0058) ──signal──▶ ES pilot (EXP-0059)
   not_decoder | no_gap ──▶ diagnose, stop for user
pilot feasible ──▶ seed-0 main ES (EXP-0060: A1 resumed, A2) ──▶ seed-0 validation reading
   unresolved | kill | other ──▶ diagnose, stop
   budget_limited ──▶ user
   success | plateau ──▶ seed 1 of the selected arm ──▶ final validation reading
        success ──▶ held-out confirmation (EXP-0061) + cost (EXP-0062) ──confirmed──▶ user: gate exit
        plateau ──▶ step-3 gate: relational check (EXP-0063, env-side, no restore)
               relational ──▶ GNN (EXP-0064…, design update + restore) ; not_relational ──▶ diagnose, stop
```

Every arrow is a pre-registered reading, recorded in its EXP `RESULT.md` before the next task
starts. A "diagnose" branch makes the supervisor insert a diagnosis task; strategies are not
pre-planned. EXP numbers are planned; an inserted diagnosis takes the next free one.

### D2. Producer: `entity_planner` by default; the training producer is a user decision before the pilot

- **Default: `entity_planner`.** Every atlas reference row, the EXP-0056 2 × 2 and the
  PoC-A/C rows were played with it, so arms pair exactly with recorded rows. It also isolates the
  trained role (EXP-0054 c_P = 0.12 against c_O = 1.02).
- **Why it must be settled early.** ES co-adapts the orderer to whichever producer it trains
  with, and the gate's portable subject carries the EXP-0035 producer. So "which producer the
  orderer trains with (`entity_planner` / EXP-0035 / both)" is an open user decision that must be
  answered (AskUserQuestion) before group 6, the gradient-signal probe.
- **If the answer is EXP-0035 or both:**
  - The training bar and fitness are played with that producer (both: each minibatch
    world-episode is played with both producers, with fitness averaged over them).
  - The validation and held-out reads add an EXP-0035-producer column, and `success` /
    `confirmed` must hold in it too.
  - The cost of every ES run doubles under "both". The task sizing and the launch-fit check are
    redone before the pilot.
- **Under the default,** the held-out read still reports the EXP-0035-producer cross-check and
  `portable`. They never select or gate.

### D3. Step 1 — decoder check (EXP-0057), pre-registered

**Arms** (all with `entity_planner`, tracer on):

| Arm | What it is |
|---|---|
| TR | trained orderer as deployed |
| TS `trained_sampled_whole` | EXP-0012 scores (SHA-256 `d25388bf…`); per slot order one eligible producer drawn by inverse CDF of softmax(score) at T = 1, the order routed whole; R = 2 |
| MWWk `mww_keyed` | MWW's weights (∝ `makeable_qty_now`) drawn with the same keyed uniform as TS, so TS − MWWk is paired; R = 2 |
| MWW | the atlas arm, with `action_space.np_random`, for reproduction; R = 2 |
| TW | `trained_whole`, argmax whole |
| RR | `round_robin` |

- The draw key for TS and MWWk is SHA-256(world, episode, replicate, agent, cycle, slot). The
  agent id is a key only, never an input (ADR-0058). Task 2.1 tests that permuting or altering
  the actions of the ordering agents before agent a leaves TS's and MWWk's inputs and sub-orders
  for a byte-identical.
- TS and MWWk live in `routing_diagnostics.TRAINED_WRAPPERS` / `DIAGNOSTIC_ROUTERS`, outside
  the roster.
- **Worlds.** Every atlas v2 train-split world with ρ > 0.28 whose v2 class is not `collapse`:
  23 at 12 producers (including EXP-0056's 12-0060 / 12-0111 / 12-0344) and 14 at 42. Episodes
  use the confirm's seeds: 2 per world, or 1 for cyclic shock-free worlds (12-0005, 0155 and
  0176 have one: 43 world-episodes at 12 producers, 28 at 42). **Only the 12-producer worlds
  enter the readings.** The 42-producer worlds are reported.
- **Floors (n = 1 rule).** For each compared pair of arms on a world:
  - floor_w = max(atlas `noise_floor` at the run's counts, 3·SE of the run's own paired
    per-world differences);
  - the own SE is used only when the pair has ≥ 2 paired observations (episodes × replicates
    where the two arms differ), otherwise the atlas floor alone;
  - each comparison has its own floor (DIAGNOSTICS 21: a stage's own spread decides).
- **Primary reading** (`decoder_check.csv`, 12 producers). G_w = RR_w − TR_w; a gap world has
  G_w > floor_w; rec_w = (TS_w − TR_w) / G_w. First match wins:

  | # | Reading | Rule | Licence |
  |---|---|---|---|
  | 1 | `no_gap` | fewer than 5 gap worlds | none: diagnose |
  | 2 | `decoder` | rec_w ≥ 2/3 on ≥ 2/3 of gap worlds | step 2 |
  | 3 | `decoder_partial` | median rec_w ≥ 1/3 | step 2 (the unexplained share is recorded) |
  | 4 | `not_decoder` | otherwise | none: diagnose |

- **Fidelity reading** (12 producers, all 23 worlds). TS and MWWk are *equivalent* on a world
  when |TS_w − MWWk_w| ≤ floor_w. First match wins:

  | # | Reading | Rule |
  |---|---|---|
  | 1 | `faithful` | equivalent on ≥ 2/3 of worlds |
  | 2 | `mixed` | both of the next two rules hold, or neither does |
  | 3 | `fidelity_loss` | TS < MWWk beyond the floor on ≥ 1/3 of worlds |
  | 4 | `beyond_target` | TS > MWWk beyond the floor on ≥ 1/3 of worlds |

  Score fidelity on visited states is reported as medians: Spearman correlation against log
  `makeable_qty_now`, top-1 agreement, and KL(softmax(score) ‖ makeable share).
- **The licence comes from the primary reading alone.** The fidelity reading never grants or
  withholds one. It is recorded with an interpretation, and it decides where a diagnosis starts:

  | Primary \ fidelity | `faithful` | `fidelity_loss` | `beyond_target` | `mixed` |
  |---|---|---|---|---|
  | `decoder` / `decoder_partial` (licensed) | decode is the bottleneck; scores carry their target | decode is the bottleneck; MWWk is the stronger reference (noted) | decode is the bottleneck; scores add to their target (noted) | decode is the bottleneck; fidelity varies by world (noted) |
  | `not_decoder` (diagnose) | whole input-aware sampling does not close the gap: start from rotation and per-agent offsets vs sampling | start from score fidelity (a design update would change the base) | start from what TS's extra signal costs against RR | start world by world |
  | `no_gap` (diagnose) | start from why the addendum's loss did not reproduce | same | same | same |

- **Dispersion reading (pre-registered; tests D5's premise).** On the 12-producer worlds under
  TS:
  - `agent_history_dispersion` is the median, over products with ≥ 2 agents and producers that
    ≥ 2 agents routed ≥ 3 units to, of the across-agent SD of own completion rate;
  - `routed_share_dispersion` is the median across-agent total-variation distance between
    agents' routed-share vectors over the product's eligible producers.

  | Reading | Rule | Next |
  |---|---|---|
  | `history_differs` | both measures reach their thresholds (≥ 0.05 and ≥ 0.10) on ≥ 2/3 of the worlds | none |
  | `history_flat` | otherwise | recorded; the supervisor asks the user before group 4, since agent-local history would carry little contrast |

  MWW and RR are reported alongside.
- **Concentration and herding columns** (tracer, every arm):
  - `lead_concurrent_top_share` (EXP-0056's metric);
  - `share_swing_consumer` (D9);
  - `concurrent_cross_agent_share`;
  - `orders_per_turn_{consumer,producer}`.
- **Secondary reads** (never gating): TS − TW, per size × choice cell and per ρ quartile.
- **Prediction recorded before the pod run.** Task 1.2 computes the primary reading with MWW in
  place of TS, offline from EXP-0052's confirm rows. It is the prediction for TS under
  `faithful`.
- **Validity** (exit 2 if any fails):
  - TR / TW / RR on the three EXP-0056 worlds, first episode, reproduce EXP-0056's units and
    digests;
  - TR / RR / MWW reproduce EXP-0052's confirm rows;
  - digests are equal across arms.
- **Driver.** `eval/decoder_check.py` uses `routing_audit`'s job path and pins the atlas and
  checkpoint SHA-256. Real runs need the registered hashes, a 40-hex commit and no filters.

### D4. Choice decode and validity layer (ADR-0059)

- **Env option** `ordering_decode = "choice-v1"`; `"logits-v1"` (legacy) stays the default and
  byte-identical. The option is hashed into the env config fingerprint and pinned in every driver
  `spec.json` and resume check.
- **Action encoding: shares, not quantities**, so the bound is fixed at [0, 1] under any burst
  multiplier. Row k of an ordering agent's (k_stream, P) action is slot k's per-producer shares.
  - `allocator.validate_choice` requires support ⊆ eligible, shares ≥ 0, and |Σ − 1| ≤ 1e-6.
    Quantities come from the existing `_allocate_int` (largest remainder) of shares × q, and
    tokens are apportioned ∝ quantity.
  - A whole order is the one-hot share 1.0 (exact). The validity layer accepts any valid share
    vector, but the step-2 and step-3 policies emit one-hot shares only (D5: whole orders only;
    tested).
  - A violation raises and is a `job` contract failure (`eval/contract.py`). It is never
    rerouted.
- **Overflow orders** (beyond `k_stream`) keep the legacy even split and are counted
  (`overflow_orders`). Changing that is M7 (ADR-0054).
- **Checkpoint versioning.** The composite per-order checkpoint carries `decoder_version =
  "per-order-choice-v1"`, `architecture_version = "orderer-per-order-residual-v1"`, the base
  checkpoint's SHA-256 and the normaliser hash.
  - The refusal is scoped to the composite checkpoint: loading one into a `logits-v1` env, or a
    legacy scorer as a deployed `choice-v1` policy, is refused.
  - The EXP-0012 base (`allocate_suborders-v1`) loads as a component of the composite, and a test
    covers that.
- `pettingzoo` `api_test` runs on both decodes.
- No `domain/` change. If agent-local history (D5) needs a per-sub-order outcome the domain does
  not yet expose, it is added as additive pure data (as M3 task 1.1 did).

### D5. Step 2 policy — whole-order, per-order residual scorer with agent-local history

**Whole orders only.** Every step-2 choice sends the whole order to one eligible producer.
- **Why:** EXP-0056 reads `joint`. The gap to `round_robin` closes with whole orders plus varied
  receivers, while splitting under the scorer's rule wasted capacity on orders that do not
  complete.
- **Revision 3:** the earlier split gate is dropped because ES could not reach it (review round 2,
  concern 3).
- Splitting is out of scope for M4 step 2. It could come back only with a separate diagnosis
  showing that a split recovers units that whole routing cannot.
- **Evidence (EXP-0066, 2026-10-30, `whole_suffices`):** on all 71 EXP-0057 world-episodes, units
  that only a split could have saved are 0.55% (12 producers) / 1.07% (42) of filled units under
  MWWk (completion-corrected joint count; uncorrected 1.95% / 3.42%), below the pre-registered 5%.
  Most `no_fit` loss is unfillable in its cycle, and a better single choice recovers more than a
  split would. Two worlds exceed 5% on their own (12-0230, 42-0076), reported, never deciding.

**Order of decisions.** Within agent a's turn, orders are decided in ascending order id (the
same order as `_slots`).

**Agent-local history** h_{a,i} (built in `env/agent_history.py`, not `domain/`). It is built
**only from bid-outcome kinds** of a's own past sub-orders to producer i:
- `accepted` (the bid cleared the auction);
- `completed`;
- `failed`, with the domain's dropped and TTL failures merged;
- `rejected`, all rejection kinds merged.

No diagnostic `reason` field is read. The domain marks those fields as diagnostic only
(`domain/producing_agent.py` 32–40, 121–122), and both TTL failures during an outage and outage
rejections encode hidden shock state (ADR-0056 / ADR-0058). Membership in the accepted /
completed / failed / rejected lists is the only input.
- **Features**, for each eligible producer i of a's stream product:
  - EMAs of routed, accepted, completed, failed and rejected units;
  - completion latency (cycles from routing to completion);
  - completion rate = completed / routed, with a 1-unit prior;
  - recency = log(1 + cycles since a last routed to i);
  - a never-routed flag.

  That makes 9 features.
- **Folding.** The EMA α equals the domain telemetry's (`domain/telemetry.py`) and is recorded
  in the pre-registration. The history is folded after each `complete_cycle` from a's own
  sub-orders only, and reset at episode start.
- The history is distinct from the shared producer-product telemetry, which is pooled over all
  agents and stays unchanged in the rows.

**Score for order o and producer i:**

```
s_i(o) = b_i + Δ_θ(i, o)
b_i     = frozen EXP-0012 score (one forward per stream per cycle; shared rows only)
φ_i     = normalised shared row (31) ⊕ normalised [h_{a,i} (9) ⊕ has_completed (1)] ⊕ normalised own-turn [assigned_units, assigned_orders] (2)   (43)
ψ_o     = normalised product (9) ⊕ [quantity, tokens/quantity, TTL remaining, kind]
Δ_θ     = B·tanh( (w·φ_i + Σ_{r≤R} (u_r·φ_i)(v_r·ψ_o)) / B ),  B = 2.0 logits
τ       = exp(θ_τ)            (θ_τ = 0 → τ = 1)
choice  = inverse-CDF draw of softmax(s/τ) with the D3 key; the whole order goes to the drawn producer
```

- **Has-completed indicator (task 4.3).** The history's latency reads 0.0 before a pair's first
  completion, so a linear term would read a pair with only failures or rejections as the
  fastest. φ therefore adds `has_completed` (1 once a's sub-orders to i have completed any
  units this episode), which takes φ from 42 to 43. An imputation (e.g. the median latency of
  completed pairs) was rejected on data: on the 4.3 fixture world 83% of completed latencies
  are 0, so the fitted median is the 0.0 default itself.
- Normalisation is a fixed median / IQR per feature (all 43 + 13 inputs), fitted once on the
  pilot's bar-precompute rows (A0's rows, task 5.1), then hashed and frozen. A zero IQR falls
  back to the feature's standard deviation, and a zero one to 1. Normalised values are clipped
  to ±5.
- **Arms:**
  - **A0, zero residual:** θ = 0, which is exactly TS (spec: "Zero residual equals the decoder
    check"). Ties need no rule, because sampling is proportional.
  - **A1, row-only:** w (43) and θ_τ; 44 parameters.
  - **A2, low-rank bilinear:** A1 plus u_r, v_r with R = 4 (or R = 2 if the gradient probe finds
    low signal, D7): 44 + 4·(43 + 13) = 268 parameters (R = 2: 156). u_r is initialised from
    a fixed-seed N(0, 0.1) and v_r = 0, so Δ starts at zero.
  - A1 and A2 at θ = 0 equal A0 exactly.
- **Per-parameter scales** (ES perturbation σ_j): σ = 0.05 for w, u, v, in units of the
  normalised features; σ_τ = 0.05 on θ_τ (log τ).
- **Reachable range.** With Adam lr 0.02, 40 generations move θ_τ by at most ≈ 0.8, so τ stays in
  ≈ [0.45, 2.2]. No claim is made that ES reaches a near-deterministic τ.
- **One network.** One shared, portable model with no agent ids. Agents diverge only through
  h_{a,i}, the own-turn features and their keyed draws (ADR-0058). Agents with empty histories at
  episode start still differ through sampling.
- **ADR-0058 test (task 4.3).** Permuting or altering the actions of the ordering agents before
  agent a in a cycle leaves a's φ, ψ, b_i, own-turn features and sub-orders byte-identical.

### D6. Step 2 training — bounded ES on outcomes

- **Algorithm.** Antithetic OpenAI-ES with centered-rank shaping. n = 12 pairs, plus the center
  every generation. Adam lr 0.02, weight decay 0.005. Optimizer seeds {0, 1}; seed 0 runs first
  (D10). Pilot: generations 1–20. Main: the pilot's checkpoint is **resumed** to generation 40.
  All values live in the frozen pre-registration file; a real run refuses any mismatch.
- **Common random numbers.** In a generation every member plays the same 12 training
  world-episodes: the same world, episode seed, shock / demand draws and sampling keys. Keys
  exclude the member. The minibatch schedule comes from a recorded seed stream disjoint from
  every atlas seed.
- **Scheduling.** Within a generation, jobs are submitted longest first. The predicted cost per
  (world, producer) comes from the bar precompute's timings. This keeps the 32 workers busy
  across 1.7–7.3 s episodes.
- **Fitness (ES-internal only).** f = mean over the minibatch of U / B_we − h · H / D_we, with
  h = 0.1.
  - U is consumer units.
  - B_we, the training bar, is the max over {RR, MWW (R = 2 mean), TR, roster v2 deterministic
    members} on that world-episode, precomputed once.
  - H is `holding_units_per_cycle`; D_we is mean scheduled demand per cycle.
  - Fitness is used only to rank members; every read is in consumer units (D7).
- **Relative disqualification.** A0 plays every training world-episode in the bar precompute.
  - A member's world-episode is disqualifying when the member dies there and A0 does not, or when
    the member violates run-down (r_end < 0.7516 · r_start) or starvation (a consumer's fill
    < 0.7302) and A0 on the same world-episode and keys does not.
  - A disqualified member takes the worst rank.
  - Absolute X / Y would flag A0 and the bar arms on worlds where the bar fills ≈ 46% (task 1.2
    measures the rates), which would flatten the gradient.
- **"A0 infeasible" branch (D5 of the baseline plan).**
  - A training world on which A0 dies in the bar precompute is replaced by the next draw in the
    fixed order.
  - If more than 1/4 of the candidate training worlds need replacement, or fewer than 50 feasible
    worlds can be drawn, the run stops as `no_feasible_start` and goes to the user.
- **Training worlds (≥ 50).** The 23 loaded non-collapse 12-producer atlas train worlds plus 41
  fresh `world_space` draws at 12 producers: 64 worlds, all loaded.
  - Fresh draws use the atlas lever distribution with shocks and patterns on and a seed stream
    disjoint from the atlas sample.
  - A draw is dropped if measured ρ ≤ 0.28, its bar fill is ≥ 0.95, or a bar arm dies. Each world
    is pinned.
- **Training-monitor set.** 16 fixed training world-episodes (fixed seeds and keys). Every 10
  generations the center and A0 play them at R = 2. This set gives the plateau / budget clauses
  and g_tr. It is never used for selection. The center is the run's candidate at its last
  generation.
- **Selection.** Among seed-0 candidates of A1 / A2, the one with the highest validation
  equal-weight world-mean units (12-producer loaded worlds) is selected. A tie within F_val goes
  to A1.
- **Driver.** `train/order_es.py`, split into three modules: worlds + bar, ES loop + resume, and
  validate + readings. Process pool with one job per member × world-episode, BLAS and torch at 1
  thread (DIAGNOSTICS 9). Checkpoints every generation, resume only on matching pins.

### D7. Step 2 readings, pre-registered

**Units, gains and floors.** Every read is in consumer units per world-episode, as an
equal-weight mean over worlds.
- **g_tr**: center − A0 on the training-monitor set (16 fixed world-episodes, R = 2).
- **g_val**: the selected candidate − A0 on the 12-producer loaded validation worlds.
- **Floors** are computed per comparison: candidate vs A0, RR, MWW, TR, and A2 vs A1. Each is
  max(atlas-spread floor at the read's counts, 3·SE of the read's own paired per-world
  differences, episodes and replicates included). The own SE is used only for a pair with ≥ 2
  paired observations on a world, otherwise the atlas floor alone (the n = 1 rule, as in D3).
  F_tr is the floor on the monitor set; F_val (and F_val(ref) per reference) on validation.

**Gradient-signal probe (EXP-0058; one generation per arm, its own cpu8 pod before the
pilot).**
- A1 and A2 at their registered initial θ (Δ = 0; A2's u is its seeded draw, v = 0;
  pre-registration `probe.theta`). Each of the 12 pairs is scored on two disjoint CRN
  minibatches of 6 world-episodes.
- A2's cosine is structurally lower than A1's: at v = 0 its u block (172 of 268 parameters) has
  no first-order effect, so its estimate carries noise there. The thresholds read on A1 and are
  not moved for A2.
- Reads: the cosine between the two minibatches' ES gradient estimates, and the variance ratio of
  paired (CRN) against unpaired antithetic differences.

| Reading | Rule | Next |
|---|---|---|
| `signal` | A1 cosine ≥ 0.3 | pilot as registered |
| `weak` | 0.1 ≤ A1 cosine < 0.3 | minibatch 24; pilot and main runs re-sized and launch-fit-checked |
| `no_signal` | A1 cosine < 0.1 | diagnose, stop |

- A2 cosine < 0.1 while A1 ≥ 0.1 → A2 uses R = 2.

**Pilot (EXP-0059, A1 seed 0, generations 1–20).**
- `feasible`: g_tr > F_tr. The main runs proceed.
- `unresolved`: otherwise. Diagnose and stop.

**Detectable effect (task 5.4, before the pilot).**
- The minimum detectable effect of a candidate-vs-reference comparison on the 9 loaded
  validation worlds is 3 × the SE of the world mean of **paired per-episode differences**. It is
  derived from the EXP-0052 confirm rows (for example MWW − RR on the same world-episodes) and
  scaled to the read's episode and replicate counts.
- It is expressed as a share of the **mean** validation bar.
- Pre-registered (task 5.3): the deciding difference is the larger of MWW − RR and TR − RR,
  stochastic arms at R = 2 (a stochastic arm's per-episode value is its replicate mean), from the
  pooled within-world SD. `success` needs every comparison to clear, so the read is as weak as
  its noisiest one. Task 5.4 computes the values and records the outcome; the rule, not the
  outcome, is frozen.
- Review round 2's estimate from those rows: ≈ 35 units at 4 episodes ≈ 1.9% of the mean bar
  1,831. The single-arm `se` column gives 44 / 31 and the atlas formula 48 at the confirm's
  counts; neither is used.
- Rule:
  - ≤ 2% at 4 episodes → the validation and held-out reads play 4 episodes;
  - otherwise 8 episodes;
  - > 3% at 8 episodes → to the user before the pilot.
- At ≈ 1.9% the 4-episode case is the expected one but close to the trigger. D10 sizes both
  cases.
- For scale: `d2`'s gains were 0.2–6.6% of the bar (EXP-0053).

**Readings of one seed's candidates.** First match wins:

| # | Reading | Rule | Next |
|---|---|---|---|
| 1 | `unresolved` | g_tr ≤ F_tr for every candidate read | diagnose, stop |
| 2 | `kill` | the selected candidate has g_tr > F_tr but g_val < g_tr / 3 (D5's kill rule) | diagnose the cross-world cause, stop |
| 3 | `success` | see below | seed 1, then held-out |
| 4 | `plateau` | see below | seed 1, then the step-3 gate |
| 5 | `budget_limited` | no success, and a run still improving: for the seed-0 read, **any** seed-0 arm's monitor-set improvement over its last 10 generations is ≥ F_tr; for the seed-1 read, the seed-1 run's own last-10-generation improvement is ≥ F_tr | to the user (more generations need approval) |
| 6 | `other` | anything else | diagnose, stop |

- **`success`** (12-producer loaded validation worlds), all of:
  - the selected candidate − each of A0, RR, MWW and TR > F_val(ref) on the world mean;
  - no more than 1/6 of the worlds are lost to RR by more than floor_w;
  - zero deaths;
  - run-down and starvation flags no worse than RR's;
  - under D2's EXP-0035 / both answer, the same in that producer's column.
- **`plateau`**, all of:
  - (i) for both seed-0 arms, monitor-set improvement over the last 10 generations < F_tr (for
    the seed-1 read: this AND the seed-1 run's own last-10-generation improvement < F_tr);
  - (ii) A2 − A1 (seed 0) on validation is within F_val(A2 vs A1);
  - (iii) the candidate is within or below F_val(ref) of MWW or of RR.

**Seed combination (final reading).** Seed 1 runs only for the selected arm, and only when the
seed-0 reading is `success` or `plateau`. The seed-1 candidate is read with the same table, and
the selection is not redone:
- clause (ii) is taken from seed 0, because only the selected arm has a seed 1;
- clause (i) is seed 0's clause (i) AND the seed-1 run's own last-10-generation monitor-set
  improvement < F_tr;
- `budget_limited` fires when the seed-1 run is still improving (its own last-10-generation
  improvement ≥ F_tr).

So a seed-1 run still climbing at its last generation never reads `plateau`. Then:

| Seed 0 | Seed 1 | Final reading |
|---|---|---|
| `success` | `success` | `success` (the seed-0 selected candidate goes on) |
| `plateau` | `plateau` | `plateau` |
| `success` | `plateau` | `seed_disagreement`: diagnose, stop |
| `plateau` | `success` | `seed_disagreement`: diagnose, stop |
| `success` or `plateau` | `unresolved` / `kill` / `budget_limited` / `other` | the seed-1 reading carries through, with its Next |

**Labels recorded with a `success` or `plateau`** (pre-registered; not licence-changing except
as stated). They are computed at the seed-0 read and again at the final read:

| Label | `success` | `plateau` | Consequence |
|---|---|---|---|
| `not_via_deconcentration` | yes | — (no gain to explain) | diagnosed before the held-out read; group 8 waits |
| `history_inert` | yes | yes | recorded; must be cited before history becomes GNN edge attributes (D8, task 12.1) |
| `herding` | yes | yes | under `success`: diagnosed before the held-out read (group 8 waits); under `plateau`: recorded and carried into the relational check's record |

- **Concentration.** `lead_concurrent_top_share` and `routed_share_dispersion` (D3) are reported
  for the candidate, A0, MWW, RR and the shared-only ablation. Under `success`, if the
  candidate's `lead_concurrent_top_share` is not below A0's by more than 3·SE (episode spread),
  the label is `not_via_deconcentration`: the gain did not come by spreading concurrent orders.
- **History effect.** `history_inert` when |candidate − its shared-only ablation (D9)| ≤
  F_val(ablation) on the 12-producer validation world mean: agent-local history adds nothing
  measurable. It is recorded, and the licence is unchanged.
- **Herding.** The `herding` flag (D9).

**Separation.** Validation selects (D6) and also judges `success`, so `success` is only a licence
to the held-out read, which is the confirmation (D9).

### D8. Step 3 — gated GNN

- **Relational check (EXP-0063), built env-side, no archive restore.**
  - Arm A2+R is A2 with φ_i extended by per-producer aggregates of state written before the
    cycle's `start_ordering`: `reservation` units and cycles waited for the product (formed by
    earlier cycles' auctions), inbound units and inventory cover of the recipe's inputs. No
    pending-bid or same-cycle routing aggregate is used (ADR-0058).
  - The aggregates are computed in `env/` from domain state.
  - Trained by the same ES: seed 0, then seed 1 under the D7 seed rule.
  - **`relational`:** A2+R − A2 on validation > F_val for both seeds. Missing relational
    observation is diagnosed, and a GNN is licensed. If A2+R also meets `success`, the user
    decides between it and a GNN.
  - **`not_relational`:** otherwise. Diagnose and stop.
- **GNN (EXP-0064 onward, only after `relational`).**
  - Restore `domain/graph_view.py`, the `graph_primitives_v1` encoder and the MPNN from
    `archive-1` `7aeef1f`.
  - **The restored schema drops the `pending_bid` relations and their reverses** (current-cycle
    bids by other agents, ADR-0058). It keeps `reservation` relations, which are state written
    before `start_ordering`. The schema version becomes `graph_primitives_v1-adr0058` with a new
    hash.
  - Agent-local history may enter as attributes of the (orderer, producer) candidate edges only
    after the design update (task 12.1) cites step 2's `history_inert` label. If history was
    inert, the update states why it is still included, or it drops it.
  - The candidate head is per (order, eligible producer) and feeds D5's whole-order choice rule
    and the D4 validity layer. No split.
  - Training is outcome-only, with no critic. The optimiser is fixed by a design update at
    licensing (Open Questions).
  - The success rule is D7's, plus beating the step-2 best by more than F_val.

### D9. Evaluation sets, reads, herding and concentration checks

| Use | Split (atlas v2) | Loaded worlds (ρ > 0.28, non-collapse) | Read by |
|---|---|---|---|
| Step 1, method choices | train | 23 at 12 (gating), 14 at 42 (reported) | EXP-0057 |
| ES training | train + fresh draws | 23 + 41 at 12 | EXP-0058–0060 |
| Selection and validation readings | validation | 9 at 12 (gating), 3 at 42 (reported) | EXP-0060, EXP-0063 |
| Confirmation, once | held-out | 8 at 12 (gating), 2 at 42 (reported) | EXP-0061 |
| Reported apart | held-out-region | 4 at 42 | EXP-0061 |

- Every read plays every world of its split: the held-out read plays 12 worlds at 12 and 8 at 42
  including held-out-region. Light (ρ ≤ 0.28) and `collapse` worlds are reported in their own
  tables and never gate.
- Results are also tabulated by atlas ρ quartile, so the 0.28 cut carries no decision by itself.
- The 42-producer reads run the 12-producer-trained policy unretrained.
- **Held-out confirmation (EXP-0061).** Fresh seeds, the D7 episode count (4, or 8), stochastic
  arms R = 2.
  - `confirmed`: the candidate − each of A0, RR, MWW and TR > the per-comparison floor on the
    12-producer loaded held-out world mean, with zero deaths. Under D2's EXP-0035 / both answer,
    the same in that column.
  - `not_confirmed`: otherwise. Diagnose (generalisation) and go to the user.
  - Only `confirmed` leads to the gate-exit question.
  - The concentration, history and herding measures of D7 are reported again.
- **Held-out ledger.** "Read once" is enforced by the tracked ledger
  `experiments/m4_heldout_ledger.json`, keyed by atlas hash + change name + split. It lives
  outside the package, so experiment tasks never edit `tokenomics/tokenomics/`.
  - **The commit problem.** A commit cannot name itself. So the pending entry records the code
    commit C0 that the read will run, plus the *package source hash* at C0 and the θ hash. The
    package source hash is a SHA-256 over the sorted relative paths and contents of every file
    under `tokenomics/tokenomics/` and `tokenomics/tests/` plus `tokenomics/pyproject.toml`,
    excluding `__pycache__`. The same function computes it locally and on the pod, because the pod
    has no `.git`. Committing the entry then produces the launch commit C1 = C0 + ledger, and
    `{SOURCE_COMMIT}` at launch is C1.
  - **Before launch, locally:** the task writes the entry with `status: pending` (C0, the
    package source hash at C0, θ hash, date), commits it (C1), and launches at C1. The
    pending entry is in git before any pod exists.
  - **On the pod:** the driver requires `--ledger` (the staged copy) to hold exactly one entry
    for (atlas, change, split). The entry must be `pending` and match the θ hash.
    - The package source hash of the synced tree must also equal the entry's. The launcher's
      `--verify-source-commit` guarantees the synced tree is C1's, clean.
    - C1 is accepted because only the ledger, outside the package, differs from C0. Otherwise the driver refuses. A `done` or `invalid` entry always refuses.
    The driver never writes the ledger.
  - **When the entry is closed,** C1 is recorded next to C0.
  - **After the run:** the task sets the entry to `done` (with the EXP path) if the run exits 0
    with valid rows. On exit 2, a crash, a lost pod or a missing pull it sets `invalid`.
  - **Pre-registered:** an `invalid` read counts as used. It goes to the user and is never re-run
    automatically. A re-read needs the user's decision and a new ledger key.
- **Herding check (pre-registered; risk named in ADR-0058).** If every agent reacts to the same
  last-cycle shared signal, they may all switch together and oscillate.
  - Metric `share_swing`: per product, the total-variation distance between consecutive cycles'
    producer shares of the product's consumer order units, (1/2) Σ_i |s_{i,t} − s_{i,t−1}|. It is
    averaged over cycles and products (unit-weighted), per episode.
  - Contrast at every validation and held-out read:
    - the candidate;
    - its *shared-only* ablation: h_{a,i} replaced by the mean over the product's agents, so
      all agents of a product see identical inputs, without retraining;
    - A0, MWW and RR.
  - Flag `herding` when the candidate's share_swing exceeds MWW's by more than 3·SE (its own
    episode spread) on ≥ 1/3 of the loaded validation worlds. The flag is recorded and inserts a
    diagnosis before the held-out read. It does not change the units reading.
- **Concentration check (staying together, the diagnosed failure).** At every validation and
  held-out read, report `lead_concurrent_top_share` and `routed_share_dispersion` for the
  candidate, A0, MWW, RR and the shared-only ablation. The `not_via_deconcentration` label is
  D7's.
- The shared-only ablation is out of distribution for a model trained with history. Its swing and
  concentration are descriptive. Its units enter only the `history_inert` label (D7), never a
  licence.

### D10. Compute and the proposed M4 budget

Rates and rules:
- pod s ≈ 0.646 × local s; the per-order policy costs ≈ 1.3 × the trained orderer (probes 2.3 /
  5.4 confirm);
- each run includes ≈ 10 min of pod setup at its shape: 1.33 vCPU-h on cpu8, 2.7 on cpu16, 5.3
  on cpu32;
- hard stop = 2 × estimate, capped at 40 per run;
- secure pods only, the largest shape whose workers stay busy, longest-first job submission.

**Gradient probe placement (review round 2, concern 6): its own cpu8 pod, estimate 2.0, stop 4.**
Running it at the start of the pilot's cpu32 pod would save one cpu8 setup (≈ 1.3) only on the
`signal` branch. On `weak` (minibatch 24, pilot re-sized and launch-fit re-checked) or
`no_signal` (stop), the cpu32 pod would have paid ≈ 5.3 of setup for ≈ 0.7 of compute, and the
pilot would need a second pod anyway. A separate pod also keeps the supervisor's check between
the probe's reading and the pilot's launch. The separate pod is cheaper unless `signal` is more
than ≈ 75% likely, which nothing measured supports.

**Launch-fit rule:** a run launches only if billed spend so far in its tranche plus its hard
stop is ≤ the tranche. The supervisor records the check in the task. A resize by the gradient
probe (`weak`), by D2's "both" answer or by D7's 8-episode rule reruns the check, and a misfit
goes to the user.

The table assumes 4-episode reads; the 8-episode case follows it. "Spend before" uses the planned
estimates, with EXP-0057 at its re-sized 3.52 (task 2.3; rounded 3.5), so every later row is 0.5
below the proposal's values and the success path's largest spend + stop is **91.4**. EXP-0057
billed ≈ 1.41 (task 3.1); the launch-fit rule uses billed spend, so the real checks have more room.

| Run | Step | Shape | Estimate (billed vCPU-h) | Hard stop | Grounding | Launch-fit (spend before + stop) |
|---|---|---|---|---|---|---|
| EXP-0057 decoder check | 1 | cpu8 | 3.52 (re-sized by task 2.3, sizing.md §EXP-0057); actual ≈ 1.41 billed | 7.0 | 71 world-episodes, 639 jobs ≈ 11,376 local s (sizing.md §EXP-0057) | 0 + 7.0 |
| EXP-0058 gradient-signal probe | 2 | cpu8 | 2.0 (0.7 + 1.3) | 4 | 2 arms × 12 pairs × 2 × 2 × 6 world-ep × 5.9 s ≈ 3,400 local s | 3.5 + 4 = 7.5 |
| EXP-0059 pilot, A1 s0, gen 1–20 | 2 | cpu32 | 13 (7.4 + 5.3) | 26 | 25 × 12 × 5.9 s × 20 + bar precompute 64 × 2 × 25 s + monitor | 5.5 + 26 = 31.5 |
| EXP-0060a A1 s0 resumed, gen 21–40 | 2 | cpu32 | 12 (6.7 + 5.3) | 24 | 20 gen × 0.32 + monitor | 18.5 + 24 = 42.5 |
| EXP-0060b A2 s0, gen 1–40 | 2 | cpu32 | 19 (13.3 + 5.3) | 38 | 40 gen × 0.32 + monitor | 30.5 + 38 = 68.5 (80.5 if launched with 0060a) |
| EXP-0060c seed-0 validation read | 2 | cpu16 | 3.9 (1.2 + 2.7) | 8 | 13 worlds at 12 × 4 × 46 s + 3 at 42 × 4 × 350 s ≈ 6,600 local s | 49.5 + 8 = 57.5 |
| EXP-0060d seed 1, selected arm (conditional) | 2 | cpu32 | 19 | 38 | as 0060b | 53.4 + 38 = **91.4** |
| EXP-0060e final validation read (conditional) | 2 | cpu16 | 3.0 | 6 | the seed-1 candidate only; references reused | 72.4 + 6 = 78.4 |
| EXP-0061 held-out confirmation | 2 | cpu16 | 5.4 (2.7 + 2.7) | 11 | 12 worlds at 12 × 4 × 51 s + 8 at 42 × 4 × 390 s ≈ 14,900 local s | 75.4 + 11 = 86.4 |
| EXP-0062 per-order cost | 2 | cpu8 | 1.6 | 3 | EXP-0055: 0.23 vCPU-h compute on 4 vCPU | 80.8 + 3 = 83.8 |
| **Tranche A (steps 1–2)** | | | **53.4 seed-0 path; 82.4 full success path** | | | **max 91.4; authorised 100 (user, 2026-10-30); margin 8.6 (3.6 under 95)** |
| EXP-0063a A2+R s0 | 3 gate | cpu32 | 21 | 40 | as 0060b + ≈ 10% feature cost | 0 + 40 |
| EXP-0063b validation read | 3 gate | cpu16 | 3.9 | 8 | as 0060c | 21 + 8 = 29 |
| EXP-0063c A2+R s1 (conditional) + read | 3 gate | cpu32 | 24 (21 + 3.0) | 40 | as above | 24.9 + 40 = **64.9** |
| **Tranche B1 (relational check)** | | | **24.9–48.9** | | | **proposed authorisation 65, released only after `plateau`** |
| Tranche B2 (GNN, EXP-0064+) | 3 | sized at licensing | placeholder ≈ 46 (2 × 20 training + 5.6 read) | ≤ 40 per run | design update + probe | approved by the user at licensing |

**8-episode case** (if task 5.4's paired MDE is > 2% at 4 episodes):

| Run | Estimate | Hard stop |
|---|---|---|
| EXP-0060c | 5.1 | 11 |
| EXP-0060e | 3.3 | 7 |
| EXP-0061 | 8.1 | 17 |
| EXP-0063b | 5.1 | 11 |

- Tranche A: success path 86.6, max spend + stop 93.9 (at EXP-0061), so the margin under the
  approved 100 is 6.1 (1.1 under 95).
- Tranche B1: max 66.1 (at EXP-0063c, 26.1 + 40), which needs 67.

**Margin, stated in the tranche-A approval question** (values at proposal time, before task 2.3
re-sized EXP-0057; the user approved 100 on 2026-10-30, and the current margin is 8.6, 6.1 under
8-episode reads, per the table above). The success path's largest "spend so far
+ hard stop" is 91.9, so the margin under 95 is **3.1 vCPU-h** (0.6 under 8-episode reads). Review
round 2's 3.6 was computed before the probe's re-estimate (+0.5). An overrun of more than the
margin across the earlier runs blocks seed 1 (EXP-0060d) or the held-out read until the user
approves more. The question offers:
- 95 (margin 3.1 / 0.6);
- 100 (margin 8.1 / 5.6).

Against revision 2 (tranche A max 91.4): the maximum rises 0.5 from the probe estimate; the
success-path total moves 82.6 → 82.9 (+0.5 probe, −0.2 from the corrected held-out world counts
12 + 8). A gate read is in no tranche.

### D11. Pre-pod local checks (group 1, before any pod run)

1. **Restore DIAGNOSTICS** entries 6, 9, 11, 12 from `18c717d^` into `docs/DIAGNOSTICS.md` (docs
   task).
2. **Offline from the EXP-0052 confirm rows** (no episodes):
   - (c) the X / Y violation rates (run-down, per-consumer starvation, death) of RR / MWW / TR on
     loaded world-episodes;
   - (d) the MWW-substituted step-1 reading, written down as the prediction;
   - (e) the paired per-episode MWW − RR differences on the 9 loaded validation worlds, a first
     look at D7's detectable effect (task 5.4 recomputes it).
3. **Minimal local trace.** The recorded rows are per-episode aggregates, with no per-agent,
   per-turn or per-producer-by-agent fields, so (a) and (b) need one. A probe replays EXP-0056's
   three worlds (first episode, horizon 100, TR / RR / MWW, R = 1) and one loaded 42-producer
   f14 train world (TR only). Each command runs in under 2 min.
   - (a) orders per agent turn by source class, and the within-agent vs across-agent composition
     of concurrent same-product groups;
   - (b) the D3 dispersion measures, under MWW, the A0-like sampling regime. Under TR every agent
     routes to the top producer, so the other producers' rates are just the prior; TR and RR are
     reported for context only. The measures:
     - `agent_history_dispersion`: the median, over products with ≥ 2 agents and producers that
       ≥ 2 agents routed ≥ 3 units to, of the across-agent SD of own completion rate;
     - `routed_share_dispersion`: the median across-agent total-variation distance of routed
       shares.
   - This is descriptive, not a go / no-go: EXP-0057's dispersion reading (D3) repeats it at scale
     under TS.
4. Outputs go to `openspec/changes/add-m4-orderer-redesign/notes/pre_pod_checks.md`. Numeric
   triggers ask the user before group 3's pod run:
   - (a) the across-agent share of concurrent same-product groups is < 0.5 (the concentration is
     not mostly across agents);
   - (b) under MWW, `agent_history_dispersion` < 0.05 or `routed_share_dispersion` < 0.10 on 2 or
     more of the 3 12-producer worlds (agent-local history would carry little contrast).

   If (c) shows A0-like arms flagged on most loaded world-episodes, D6's relative rule is
   confirmed as necessary.

### D12. Records

- ADR-0058 (orderer information set) is written with this revision.
- ADR-0059 (choice decode + validity layer, D4) is written in task 4.1.
- Each EXP writes:
  - `RESULT.md`, plain English first;
  - a README row;
  - a HISTORY §14 "M4: the ordering ladder" paragraph (§14 opens with EXP-0057).
- A diagnosed cause gets a DIAGNOSTICS entry, as does every terminal "diagnose" reading.
- Sizing goes in `openspec/changes/add-m4-orderer-redesign/sizing.md`.

## Risks / Trade-offs

- **[Herding on last cycle's shared signal (ADR-0058)]** → D9's `share_swing` contrast; sampling
  and agent-local history are the dampers; the flag inserts a diagnosis before the held-out read.
- **[Agent-local histories are too similar to diversify agents]** → pre-pod check (b) with
  numeric triggers, EXP-0057's pre-registered dispersion reading (`history_flat` → the user), and
  the `history_inert` label after training; sampling still diversifies.
- **[Agents stay together on one producer (the diagnosed failure) rather than switching
  together]** → `lead_concurrent_top_share` and `routed_share_dispersion` at every read;
  `not_via_deconcentration` is diagnosed before the held-out read.
- **[Agent-local history leaks a hidden shock through failure / rejection reasons]** → the
  history is built from outcome kinds only; a reason-scramble test (task 4.2).
- **[No splits in step 2]** → justified by EXP-0056 `joint` (whole + varied receivers);
  splitting could come back only with a separate diagnosis.
- **[TS only reproduces MWW (review concern 3)]** → the paired MWWk arm, the fidelity reading and
  the offline prediction. Step 1's question is the decoder, given the fidelity read.
- **[Stochastic deployed orderer complicates the gate]** → open user decision 4. Reads report the
  R = 2 spread.
- **[Validation is small and both selects and judges]** → a paired-difference detectable effect
  against the mean bar (4 or 8 episodes), per-comparison floors with the n = 1 rule, `success`
  only as a licence, held-out confirmation and the ledger.
- **[A held-out read is lost or silently repeated]** → the ledger entry is committed as `pending`
  before launch; a crashed or invalid read counts as used and goes to the user.
- **[ES fits training worlds, fails across worlds (EXP-0009)]** → the D7 kill rule.
- **[Relative disqualification lets a member match A0's run-down]** → `success` still requires
  flags no worse than RR's.
- **[Per-order decode and history slow episodes]** → probes size every run; EXP-0062 checks the
  slope rule.
- **[Legacy behaviour drifts]** → goldens unedited, `api_test` on both decodes, decode option in
  the fingerprint.
- **[Step-3 aggregates or graph leak same-cycle or hidden state]** → ADR-0058 permutation tests
  (tasks 2.1, 4.2, 4.3, 10.1, 12.2), the shock sentinel, `pending_bid` relations dropped.
- **[Tranche-A margin (8.6 under the approved 100; 6.1 under 8-episode reads)]** → the user
  approved 100 vCPU-h (2026-10-30; 95 would have left 3.6 / 1.1); the launch-fit rule stops
  before overrun.

## Migration Plan

Additive. `logits-v1` stays the default everywhere, so a rollback means not selecting
`choice-v1`. No frozen gate file changes. The champion is unchanged (human promotion).

## Open Questions

- Step-3 optimiser: ES over a reduced GNN (hidden ≤ 16), or a score-function estimator on the
  sampled head with paired-episode baselines. It is fixed by a design update only after
  `relational`, and does not change steps 1–2.
- cpu32 vs cpu16 for the main runs, set by EXP-0059's measured worker utilisation (the largest
  busy shape).
