# EXP-0038B - Producer/Product Observation Ladder

**Date:** 2026-07-27 · **Driver:** `tokenomics.eval.value_router_errors --feature-set
producer_product_v1` (dump) + `tokenomics.eval.observation_ladder` (audit) · **Reproduce:** see
[`spec.json`](spec.json)

**Status: BOTH PHASES CLOSED — gate FAILED; marginal-value actor target RETIRED.** Phase 1
(per-producer/product) and Phase 2 (system-wide interaction) both ran the s29/s77/s45 re-dump on a
fresh CPU pod (13,940 decisions, identical to EXP-0038) and were scored by the local audit with the
corrected EXP-0038A gate. **Every** tested deployable block — the six Phase-1 blocks AND the five
Phase-2 `sys_*` blocks (`+sys_demand`, `+sys_capacity`, `+sys_recipe`, `+sys_cutoff`, `+sys_actor`) —
**beats the deployed actor floor 0/3 tasks for BOTH ridge+interactions and kNN**, with centered
within-decision r² ≤ 0 (Phase-2 ridge drifts *more* negative as blocks accumulate: −0.02 → −0.18 →
−0.22 → −0.21 → −0.25; kNN ≈ −0.05 throughout). No enriched actor was trained (gate never passed —
HANDOVER non-goal). **Final verdict: neither deployable class — per-producer/product NOR
system-wide interaction context — makes the EXP-0037 marginal-value teacher label a learnable
within-decision routing target. The teacher's producer preference is not a function of any tested
decision-time observation; it lives in the 8-cycle rollout. Retiring the marginal-value target as an
actor objective is now LICENSED (per the HANDOVER planned read). `blend_0.5` remains the orderer
floor. Do not train an actor on this target.**

## Phase 2 Result (2026-07-28 — pod re-dump `5gi98ku7gfnf2e`, audit `audit_sys/`)

**The system-wide interaction ladder fails the licensing gate on every block, for both probes, on
all three tasks.** The `producer_product_system_v1` re-dump added 32 `sys_*` columns (S1 demand /
S2 capacity / S3 recipe / S4 cutoff-urgency / S5 other-actor) on top of the 60 `pp_*` + 8 `h_*`
Phase-1 columns. Most `sys_*` fields are decision-constant, so the audit scored them as ridge
interaction *context* for the producer-varying columns (`observation_ladder._split_cols`), with kNN
(which reads every column as a distance dim) as the additive cross-check. Deployed actor floor
regret: s45 **3.93**, s77 **4.75**, s29 **4.12**. No `sys_*` block gets near it:

| block | ridge regret | ridge beats-actor | ridge r² | knn regret | knn beats-actor | knn r² |
|---|---|---|---|---|---|---|
| +history (Phase-1 end) | 5.66 | 0/3 | −0.009 | 5.52 | 0/3 | −0.048 |
| +sys_demand | 5.93 | 0/3 | −0.020 | 5.55 | 0/3 | −0.047 |
| +sys_capacity | 5.81 | 0/3 | −0.183 | 5.52 | 0/3 | −0.049 |
| +sys_recipe | 5.62 | 0/3 | −0.216 | 5.66 | 0/3 | −0.049 |
| +sys_cutoff | 5.67 | 0/3 | −0.208 | 5.67 | 0/3 | −0.049 |
| +sys_actor | 5.77 | 0/3 | −0.252 | 5.60 | 0/3 | −0.050 |

Per task, every `sys_*` block loses to the actor floor for both probes (e.g. +sys_actor ridge:
s45 5.18 vs 3.93, s77 6.46 vs 4.75, s29 5.68 vs 4.12 — all L). The regret sits at or above the
random floor (s45 5.24 / s77 6.24 / s29 5.38); adding `sys_*` blocks makes the ridge probe *worse*
(the interaction terms give it more dimensions to overfit — ridge r² drifts to −0.25). Alias
collapse is not a pass: exact-repeated-row coverage falls to 543 rows / 93 groups at +sys_demand
(from 28,383 rows / 902 groups at current_row), so median label range 0 is measured on almost no
evidence — the held-out routing gate is decisive, and it fails.

**Interpretation.** After decision-centering, neither the per-producer/product facts (Phase 1) nor
the economy-wide demand/capacity/recipe/urgency/other-actor context (Phase 2), nor their
interactions with the producer-varying columns, explain which eligible producer the teacher prefers
inside a decision. The teacher's within-decision contrast is not a function of the deployable
observation space at all — it is produced by the 8-cycle marginal-value rollout. This confirms and
extends the EXP-0038 finding (the label is not a function of the obs) to the full deployable feature
set. The marginal-value teacher is therefore not a viable **actor training target**; a learned
orderer must either emulate the rollout (a compute lever, not an observation lever) or optimise a
different, obs-measurable objective.

## Phase 1 Result (2026-07-28 — pod re-dump `s8spjt6z0xs5s6`, audit `audit/`)

## Result (2026-07-28 — pod re-dump `s8spjt6z0xs5s6`, audit)

**The Phase-1 enrichment reduced repeated-row alias conflict — and still did not help routing.** The
rich per-producer/product observation collapsed the exact-alias label conflict on repeated rows:
median label range at fixed features 10.0 → 0.0, and the fraction of repeated feature-rows with
label range > 1 unit fell 0.669 (current_row) → 0.034 (+pp_throughput) → 0.012 (+pp_base). This is
not proof that the teacher label is deterministic over the full observation space: repeated-row
coverage falls sharply as continuous `pp_*`/`h_*` columns are added (75.1% at `current_row`, 26.2% at
`+pp_throughput`, 8.7% at `+pp_base`, 4.6% at `+history`). The decisive metric is therefore the
held-out routing probe below, not alias collapse alone.

**But the licensing metric — held-out within-decision routing regret vs the deployed actor floor —
fails on every block.** Deployed actor floor regret: s45 **3.93**, s77 **4.75**, s29 **4.12**. No
probe on any block gets near it; the best (base blocks) sit at ~4.8–6.2, and adding the pp_ blocks
makes the probe *worse*, not better:

| block | ridge regret | ridge beats-actor | ridge r² | knn regret | knn beats-actor | knn r² |
|---|---|---|---|---|---|---|
| current_row | 5.53 | 0/3 | 0.0006 | 5.46 | 0/3 | −0.042 |
| +order | 5.31 | 0/3 | 0.0008 | 5.20 | 0/3 | −0.016 |
| +local_queue | 5.30 | 0/3 | 0.0006 | 5.26 | 0/3 | −0.017 |
| +pp_throughput | 5.38 | 0/3 | 0.0001 | 5.49 | 0/3 | −0.043 |
| +pp_opportunity | 5.52 | 0/3 | −0.003 | 5.43 | 0/3 | −0.045 |
| +pp_dependency | 5.53 | 0/3 | −0.004 | 5.59 | 0/3 | −0.041 |
| +pp_base | 5.61 | 0/3 | −0.005 | 5.49 | 0/3 | −0.051 |
| +pp_fulfilment | 5.68 | 0/3 | −0.008 | 5.59 | 0/3 | −0.046 |
| +history | 5.66 | 0/3 | −0.009 | 5.52 | 0/3 | −0.048 |

**Interpretation — Phase 1 did not find within-decision contrast.** The tested per-producer/product
metrics resolve some between-decision level ambiguity but do not explain which eligible producer the
teacher prefers inside a decision. After decision-centering, the 60 `pp_*` columns carry essentially
no held-out signal about the teacher's producer choice: r² ≈ 0, and the extra dimensions give the
probes noise to overfit (r² drifts more negative and regret rises as blocks accumulate). The probes
do not reliably clear the random floor (s45 5.24 / s77 6.24 / s29 5.38).

**Decision.** Do not train an enriched actor from the Phase-1 per-producer/product ladder. Keep
`blend_0.5` as the orderer floor while continuing the diagnostic. The marginal-value teacher is not
retired by this result because Phase 2 system-wide demand/capacity/recipe/other-actor metrics were
not dumped or scored. Those fields may still encode global context that the teacher sees through its
8-cycle rollout, but they must be tested as interaction context because most system facts are
decision-constant and cannot re-rank producers by themselves.

## Question

Does a richer per-producer/product observation make the EXP-0037 marginal-value teacher label
learnable as a *within-decision* routing target, or does the useful teacher signal still require
rollout emulation? EXP-0038A closed the base blocks (`current_row`/`+order`/`+local_queue`): they
resolve the between-decision label *level* but carry zero within-decision contrast (they are constant
across a decision's eligible producers). EXP-0038B tests the missing class — metrics that *vary*
across candidate producers inside one decision and can therefore re-rank them.

## What was built (2026-07-27)

New torch-free block builders in `tokenomics/eval/producer_product_features.py`, each a pure read of a
`ProducingAgent` snapshot + its process sheets (white-box, deployable at decision time — no teacher
rollout, no future/oracle facts). Each block owns one stable column prefix so the ladder can split it:

- **P1 `pp_throughput_`** — dedicated throughput of the producer for the *target product*, per
  required capacity type (current/free/reserved/max/headroom units), summarised as min/mean/max/std.
  The dispersion two producers tied on the binding min still differ on (the whole re-ranking bet),
  plus qty-over-throughput pressure and binding slack/zero.
- **P2 `pp_opp_`** — opportunity cost of spending capacity on the target vs the other products the
  producer declares that COMPETE for the same capacity types: alternate dedicated throughput,
  same-capacity overlap, demand-weighted (queued-quantity) alternate throughput, and the
  overlap-scaled opportunity cost. A highly-demanded alternate on a *different* capacity type costs
  nothing; one on the *same* type does.
- **P3 `pp_dep_`** — dependency (input) cover converted into units of the target product
  ((inventory+inbound)/need per input), bottleneck cover, missing input units for the order, and an
  explicit primal sentinel (`pp_dep_is_primal` + high cover) so a primal producer never collapses
  into a starved composite. Raw inventory count is deliberately not promoted.
- **P4 `pp_base_`** — base-routing context from the base policy (`blend_0.5`) per-producer logits:
  logit, softmax weight, rank, margin-to-best, margin-to-next. Separates "teacher disagrees because
  the base already nearly chose this producer" from "teacher wants a large rerank".
- **P6 `pp_fulfill_`** — TTL-feasibility estimate folding P1 capacity throughput + P3 cover +
  telemetry: est cycles to produce the quantity / to inputs-ready / to complete, ttl-minus-estimate,
  feasible-before-ttl, recent completion rate.
- **P5 history** — reuses the existing `h_*` telemetry-EMA block (EXP-0038A block 7).

Orchestrated per decision in `value_router_errors.producer_product_rows` (adds P4 cross-producer
margins from the base logits; P2 demand-weighting uses the orderer's queued quantity by product), and
scored as cumulative ladder blocks by `observation_ladder.ladder_for` /`run_audit` (`+pp_throughput`,
`+pp_opportunity`, `+pp_dependency`, `+pp_base`, `+pp_fulfilment`, `+history`).

**Local schema smoke (green):** a 2-task depth-3 scale dump emits 60 `pp_*` columns; the ladder
discovers all six blocks and scores each against the actor floor with the corrected EXP-0038A metric
(decision-centered, real-choice-only, decision-weighted pooled folds, tie-aware, ridge+interactions &
kNN). This proves the pipeline; it is NOT a gate read (toy horizon-14 economies, not s29/s77/s45).

### Deviations from the HANDOVER ladder (deliberate, deployability-driven)

- **P3 transitive / upstream-speed columns omitted** (`pp_transitive_input_cover_min`,
  `pp_upstream_replenish_speed_min`, `pp_est_cycles_to_cover_missing_inputs`,
  `pp_dependency_completion_speed`). A producer only holds inventory of its *direct* inputs; transitive
  cover and upstream replenish speed depend on the recipe graph + *other* producers' state/telemetry —
  that is a system-wide (Phase-2) concern, not a per-producer deployable fact. Direct cover, bottleneck
  cover, missing units, and the primal sentinel are implemented.
- **P4 price columns omitted** (`pp_cutoff_price`, `pp_price_or_score_margin_to_cutoff`,
  `pp_expected_wait_cost_proxy`). This env has no cut-off/market price the ordering side sees at
  decision time — cut-off price is a producer-internal auction scalar computed post-auction. The base
  logit is the routing proxy (HANDOVER fallback). Documented absence, not a faked price column.
- **P5 uses folded EMAs, not an exact 5-cycle window.** Telemetry keeps folded per-cycle EMAs only;
  no exact `K=5` window exists. Per the HANDOVER fallback, the `h_*` EMA block stands in for P5. EMAs
  are strictly past-cycle (folded at `end_cycle`), so deployable at decision time (unit-tested: a
  current-cycle event does not leak into the block).

## Phase 2 — System-Wide Interaction Ladder (DONE, 2026-07-28)

Implemented as specified in the HANDOVER and run to a gate read (see the Phase 2 Result above):

- New feature set `producer_product_system_v1` (preserves `producer_product_v1`), adding 32 `sys_*`
  columns per eligible producer row: S1 `sys_demand_*`, S2 `sys_capacity_*`, S3 `sys_recipe_*`,
  S4 `sys_cutoff_*` (urgency proxies — env has no decision-time price, documented absence, no faked
  price), S5 `sys_actor_*`. Builders are torch-free pure reads of live economy state in
  `tokenomics/eval/producer_product_features.py`; orchestrated per decision by
  `value_router_errors.producer_product_rows(system=True)` (system context precomputed once/decision).
- `observation_ladder`: `SYS_BLOCKS` appended to the ladder after `+history`; `_split_cols` routes
  `sys_*` columns into ridge interaction *context* (they are mostly decision-constant, so they can
  only re-rank through interactions with producer-varying `feat_/pp_/h_` columns); kNN reads them as
  distance dims (additive cross-check). The few producer-varying `sys_*` fields
  (`*_competing_for_p_capacity_qty`) re-rank through those same interactions.
- Gate unchanged: decision-centered labels, real-choice-only, decision-weighted pooled held-out
  folds, tie-aware top1, ridge+interactions & kNN, vs the same deployed actor floor. Alias spread
  reported as secondary (coverage collapses — not a pass).

**Result: FAILED (0/3 both probes, every block).** See the Phase 2 Result section above.

### Deviations (deployability-driven, same spirit as Phase 1)

- **S4 has no real price column.** The env exposes no market/cut-off price the ordering side sees for
  a new order at decision time (the producer `cut_off_price` is a post-auction internal scalar,
  NaN-initialised). S4 logs deployable urgency proxies only (`sys_cutoff_order_urgency`,
  `_ttl_pressure_bucket`, `_expected_lost_units_if_wait`, `_oldest_age`) with an explicit
  `sys_cutoff_price_available=0` absence flag — no faked price.
- **S6 `oracle_*` not run.** The deployable ladder already fails on all of Phase 1 + Phase 2, so an
  oracle upper bound is unnecessary to make the retirement call (the HANDOVER planned read only needs
  the deployable blocks to fail, which they do).

## Planned Read (from HANDOVER gate) — OUTCOME

- A deployable block **licenses** enrichment only if it beats the deployed actor floor on held-out
  real-choice regret on **≥2/3** gate tasks for **both** ridge+interactions and kNN, with **positive
  centered within-decision r2 on ≥2/3** tasks and a material regret reduction, first appearing in a
  **named deployable** block.
- **Phase 1 AND Phase 2 both failed this gate on every block (0/3, both probes, r² ≤ 0).** Per the
  HANDOVER planned read, retiring the marginal-value teacher as an actor target is now **licensed**.
  `blend_0.5` remains the orderer floor; no enriched actor is trained on this target.

## Artifacts

- Blocks: `tokenomics/eval/producer_product_features.py` (torch-free, pure; Phase-1 `pp_*` +
  Phase-2 `sys_*` builders).
- Dump orchestration: `tokenomics/eval/value_router_errors.py` (`--feature-set producer_product_v1`
  and `producer_product_system_v1`, `producer_product_rows(system=…)`).
- Ladder scoring: `tokenomics/eval/observation_ladder.py` (`PP_BLOCKS`, `SYS_BLOCKS`, `ladder_for`,
  `_split_cols`).
- Tests (all green, full suite 708 passed): `tokenomics/tests/test_producer_product_features.py`
  (12, torch-free), `tokenomics/tests/test_value_router_errors_run.py` (+2 sys integration),
  `tokenomics/tests/test_observation_ladder.py` (+3: split-cols, interaction-recovery, sys ladder
  discovery).
- **Phase 1 (2026-07-28):** `dump/` (60 `pp_*` + 8 `h_*` cols, 37,804 rows / 13,940 decisions) and
  `audit/`. Pod `s8spjt6z0xs5s6` (create-pod, `runpod/base:0.7.0-ubuntu2004`, `--cpu` python3.12,
  16 vCPU) — stopped + terminated after pull.
- **Phase 2 (2026-07-28):** `dump_sys/` (60 `pp_*` + 8 `h_*` + 32 `sys_*` cols, same 37,804 rows /
  13,940 decisions; top1 0.49 / mean regret 1.322 — byte-identical trajectory to EXP-0038) and
  `audit_sys/` (`block_summary.csv`, `probe_by_block.csv`, `alias_by_block.csv`,
  `top_mistakes_by_block.csv`, `spec.json`). Pod `5gi98ku7gfnf2e` (create-pod,
  `runpod/base:0.7.0-ubuntu2004`, `--cpu` python3.12, 16 vCPU) — stopped + terminated after pull.
