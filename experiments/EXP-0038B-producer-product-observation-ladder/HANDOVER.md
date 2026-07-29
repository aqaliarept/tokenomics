# EXP-0038B Handover - Producer/Product Observation Ladder

## Current Status - 2026-07-28

Phase 1 is complete and failed the gate. The fresh s29/s77/s45 pod re-dump with
`--feature-set producer_product_v1` produced 13,940 decisions / 37,804 candidate rows with 60
`pp_*` columns plus folded `h_*` history. The corrected `observation_ladder` audit reproduced the
same trajectory as EXP-0038 and every tested Phase-1 per-producer/product block was 0/3 against the
deployed actor floor for both ridge+interactions and kNN.

Narrowed conclusion after review: do not train an enriched actor from Phase 1, but do not retire the
marginal-value teacher yet. Phase 2 system-wide demand/capacity/recipe/other-actor metrics were not
dumped or scored. Continue with a `sys_*` interaction ladder before making a target-level retirement
call. Alias collapse is secondary evidence only; repeated-row coverage falls sharply as continuous
features are added, so the held-out routing gate remains decisive.

## Plain Goal

EXP-0038B continues EXP-0038A. EXP-0038A tested only observations already present in the old dump:
the current 7-feature producer row, order fields, and local queue fields. Order/queue context
collapsed label level conflict but did not help within-decision routing.

This experiment tests the missing class that can actually re-rank producers inside one decision:

```text
per-producer, per-target-product metrics
```

Start with producer/product metrics. Add system-wide metrics only after the per-producer ladder is
measured, and score system-wide fields only through interactions with producer features because
system fields are usually constant across all candidate producers in a decision.

No actor training is licensed until the corrected held-out routing gate passes.

## Question

Can a richer deployable observation make the EXP-0037 marginal-value teacher label a learnable
within-decision routing target?

The current row is probably too compressed:

- A product can require several capacity types, but the current row collapses them to one binding
  min.
- A producer can use the same capacity types for multiple other products, so current capacity is not
  enough; opportunity cost matters.
- Raw inventory is ambiguous because the same input inventory can serve multiple products.
- History may reveal fulfilment speed, reliability, and dependency latency that instantaneous state
  hides.

EXP-0038B must discover whether these deployable facts explain the teacher's producer preference,
or whether the useful teacher information still requires rollout emulation.

## Inputs

- Same gate tasks as EXP-0038/0038A: `s29`, `s77`, `s45`.
- Same archive and producer checkpoint as EXP-0038.
- Same floor and teacher: `blend_0.5`, `marginal_value_labels(delta=1.0, lookahead=8)`.
- Same seeded actor rerun or loaded actor path only as the incumbent floor; do not train a new
  actor.
- Existing code hooks: `tokenomics.eval.value_router_errors --full` and
  `tokenomics.eval.observation_ladder`.

Current `--full` support logs a useful first sketch (`b4_*`, `h_*`), but it is not enough for this
experiment. Extend it rather than treating the existing `b4_total_inv` / min-ratio summaries as the
final observation contract.

## Observation Ladder

Each row is one `(decision, eligible producer)` candidate. Prefix names below are suggested stable
column prefixes for `producers.csv`.

### Phase 1 - Per-Producer/Product Blocks

These blocks can vary across candidate producers inside one decision and can therefore re-rank.
Run these first.

#### P0 `current_row`

The EXP-0037 baseline row:

- `feat_0`: eligible
- `feat_1`: makeable quantity now
- `feat_2`: free capacity binding ratio
- `feat_3`: reserved capacity binding ratio
- `feat_4`: capacity headroom binding ratio
- `feat_5`: input inventory plus inbound binding ratio
- `feat_6`: output product inventory

#### P1 `pp_throughput_*` - dedicated throughput for target product

For producer `p`, order product `A`, and capacity requirements `req_A[t]`:

```text
units_current_if_dedicated = min_t capacity[p,t] / req_A[t]
units_free_if_dedicated    = min_t (capacity[p,t] - reserved[p,t]) / req_A[t]
units_max_if_dedicated     = min_t max_capacity[p,t] / req_A[t]
units_headroom_for_A       = min_t (max_capacity[p,t] - capacity[p,t]) / req_A[t]
```

Log:

- `pp_req_type_count`
- `pp_current_units_min`, `pp_current_units_mean`, `pp_current_units_max`, `pp_current_units_std`
- `pp_free_units_min`, `pp_free_units_mean`, `pp_free_units_max`, `pp_free_units_std`
- `pp_reserved_units_min`, `pp_reserved_units_mean`, `pp_reserved_units_max`, `pp_reserved_units_std`
- `pp_max_units_min`, `pp_max_units_mean`, `pp_max_units_max`, `pp_max_units_std`
- `pp_headroom_units_min`, `pp_headroom_units_mean`, `pp_headroom_units_max`, `pp_headroom_units_std`
- `pp_qty_over_free_units`, `pp_qty_over_current_units`, `pp_qty_over_max_units`
- `pp_binding_cap_slack_units`
- `pp_binding_cap_is_zero`

Do not log only a single min. Dispersion matters because two producers can tie on the binding min
while differing in non-binding capacities that affect downstream recovery.

#### P2 `pp_opp_*` - opportunity cost and capacity competition

For every other product `B` that producer `p` can make, compute its dedicated throughput and how
much it competes with `A` for the same capacity types.

Log summarized alternatives:

- `pp_alt_product_count`
- `pp_alt_best_free_units`
- `pp_alt_best_current_units`
- `pp_alt_best_max_units`
- `pp_alt_mean_free_units`
- `pp_alt_demand_weighted_free_units`
- `pp_alt_same_cap_overlap_max`
- `pp_alt_same_cap_overlap_mean`
- `pp_alt_target_vs_best_margin`
- `pp_alt_target_vs_demand_weighted_margin`
- `pp_capacity_opportunity_cost`
- `pp_capacity_scarcity_weighted_cost`

Demand weighting should use only decision-time demand state. If demand is not yet available in the
dump, start with queued quantity by product and add better demand signals in Phase 2.

#### P3 `pp_dep_*` - dependency inventory and fulfilment speed

Raw inventory count is weak because inventory can serve multiple products. Convert it to target
product cover and dependency speed.

For each input `i` of target product `A`, log cover in units of `A`:

```text
input_cover_units_i = (inventory[p,i] + inbound[p,i]) / input_need_A[i]
```

Log:

- `pp_input_count`
- `pp_input_cover_min`, `pp_input_cover_mean`, `pp_input_cover_max`, `pp_input_cover_std`
- `pp_input_inventory_cover_min`
- `pp_input_inbound_cover_min`
- `pp_missing_input_units_for_order`
- `pp_bottleneck_input_cover`
- `pp_transitive_input_cover_min`
- `pp_upstream_replenish_speed_min`
- `pp_est_cycles_to_cover_missing_inputs`
- `pp_dependency_completion_speed`

For primal products with no inputs, use explicit sentinel columns, not accidental zeros. A primal
producer with no input dependency should be distinguishable from a non-primal producer with no
input cover.

#### P4 `pp_base_*` - base routing and cutoff/margin context

If the environment has a true cutoff price, bid, or market threshold, log it. If not, use routing
proxy margins from the base policy.

Log per producer:

- `pp_base_logit`
- `pp_base_softmax_weight`
- `pp_base_rank`
- `pp_base_margin_to_best`
- `pp_base_margin_to_next`
- `pp_cutoff_price` if present
- `pp_price_or_score_margin_to_cutoff` if present
- `pp_expected_wait_cost_proxy`

The purpose is to distinguish "teacher disagrees because the base already nearly chose this
producer" from "teacher wants a large rerank against the base."

#### P5 `pp_hist5_*` - recent producer/product history

Use the previous `K=5` completed cycles only. No current-cycle or future leakage.

For producer `p` and target product `A`:

- `pp_hist5_routed_A`
- `pp_hist5_completed_A`
- `pp_hist5_rejected_A`
- `pp_hist5_failed_A`
- `pp_hist5_expired_A`
- `pp_hist5_completion_rate_A`
- `pp_hist5_rejection_rate_A`
- `pp_hist5_mean_completion_latency_A`
- `pp_hist5_units_per_cycle_A`

Also log producer-total history:

- `pp_hist5_routed_all`
- `pp_hist5_completed_all`
- `pp_hist5_rejected_all`
- `pp_hist5_completion_rate_all`
- `pp_hist5_capacity_utilization_mean`
- `pp_hist5_capacity_utilization_max`
- `pp_hist5_input_received_units`
- `pp_hist5_output_inventory_delta_A`

If only telemetry EMAs exist, log the EMAs but record that the exact 5-cycle window is missing.

#### P6 `pp_fulfill_*` - fulfilment speed estimate

This block turns dependency and history into a producer-specific "can this order finish soon?"
estimate.

Log:

- `pp_order_qty_over_hist_rate_A`
- `pp_est_cycles_to_produce_qty_A`
- `pp_est_cycles_to_inputs_ready`
- `pp_est_cycles_to_complete_order`
- `pp_ttl_minus_est_completion`
- `pp_completion_feasible_before_ttl`
- `pp_recent_dependency_chain_success_rate`

Keep this estimate causal and deployable. It can use current state and past telemetry, not teacher
rollout outcomes.

### Phase 2 - System-Wide Blocks

Add these only after Phase 1 has been scored. Most are decision-constant, so they must be evaluated
through interactions with producer/product features. They should not be allowed to pass on alias
spread alone.

#### S1 `sys_demand_*` - target and competing demand

- `sys_target_queued_qty`
- `sys_target_order_count`
- `sys_target_consumer_pressure`
- `sys_target_oldest_age`
- `sys_target_min_ttl`
- `sys_same_product_demand_other_orderers`
- `sys_competing_products_queued_qty`
- `sys_products_competing_for_p_capacity_qty` as a producer-specific interaction-ready summary

#### S2 `sys_capacity_*` - global scarcity for target product

- `sys_target_total_free_units`
- `sys_target_total_current_units`
- `sys_target_total_max_units`
- `sys_target_total_reserved_units`
- `sys_target_total_headroom_units`
- `sys_target_eligible_producer_count`
- `sys_target_capacity_hhi`
- `sys_target_bottleneck_capacity_slack`
- `sys_required_type_global_scarcity_min`
- `sys_required_type_global_scarcity_mean`

#### S3 `sys_recipe_*` - recipe and downstream bottlenecks

- `sys_product_depth`
- `sys_downstream_demand_unlocked_by_A`
- `sys_downstream_inventory_pressure`
- `sys_transitive_required_input_count`
- `sys_transitive_input_cover_min`
- `sys_upstream_capacity_cover_min`
- `sys_dependency_chain_completion_speed`
- `sys_recipe_bottleneck_depth`

#### S4 `sys_cutoff_*` - market/cutoff/urgency context

- `sys_cutoff_price_A` if present
- `sys_market_price_A` if present
- `sys_price_margin_A` if present
- `sys_order_urgency_score`
- `sys_expected_lost_units_if_wait`
- `sys_ttl_pressure_bucket`

If the env has no price concept, document the absence and use explicit proxy names. Do not invent a
fake price column.

#### S5 `sys_actor_*` - other actor presence and demand

- `sys_other_orderer_count_with_A`
- `sys_other_orderer_queued_qty_A`
- `sys_other_orderer_competing_qty_for_same_producers`
- `sys_other_orderer_min_ttl_A`
- `sys_other_orderer_recent_routed_A`
- `sys_other_orderer_recent_completed_A`

#### S6 `oracle_*` - non-deployable upper bound

Only after deployable blocks are measured, add an oracle block to bound what teacher emulation
could do:

- `oracle_base_rollout_units`
- `oracle_bumped_rollout_units_for_candidate`
- `oracle_future_demand_A`
- `oracle_realized_downstream_fulfilment_after_decision`

These columns must never be used as deployable actor input. If only oracle fixes the target, stop
the marginal-value actor target.

## Metrics And Gate

Use the corrected EXP-0038A gate, not the first-cut metric.

For every cumulative block:

- Fit labels after decision-centering.
- Score only real-choice decisions: at least two eligible producers and non-flat teacher labels.
- Pool held-out folds decision-weighted; do not average fold means.
- Use tie-aware top-1.
- Report mean teacher regret, top-1, centered within-decision `r2`, and centered MSE.
- Compare against random and the deployed actor floor scored with the same metric.
- Run both ridge with producer/context interactions and kNN.

Observation enrichment is licensed only if a deployable block passes all of:

- beats the deployed actor floor on at least 2 of 3 gate tasks for both probe families;
- has positive centered within-decision `r2` on at least 2 of 3 gate tasks;
- reduces real-choice regret by a material margin, suggested floor: at least 5 percent or 0.25
  units/task versus the actor floor;
- improvement first appears in a named deployable block, not only in `oracle_*`.

Alias spread remains supplementary. It can identify level conflict but cannot license a routing
feature by itself.

## Implementation Plan

1. Extend `value_router_errors --full` into a richer `--feature-set producer_product_v1` or similar
   flag. Preserve old `--full` behavior if tests depend on it, but do not call the old columns the
   final EXP-0038B feature set.
2. Add unit tests for the feature builders:
   - multi-capacity product where binding min and non-binding dispersion differ;
   - capacity shared with an alternate demanded product;
   - inventory that can satisfy multiple products, proving raw inventory is not used as the key
     signal;
   - no-input primal product sentinel behavior;
   - history uses only past cycles.
3. Run a tiny local smoke for schema only.
4. Run the real s29/s77/s45 re-dump on a CPU pod, same parameters as EXP-0038.
5. Run `observation_ladder` on the new dump.
6. Only if the gate passes, open a separate actor-training experiment with the winning observation
   contract.

## Expected Outputs

```text
dump/decisions.csv
dump/producers.csv
dump/top_mistakes.csv
dump/alias_groups.csv
dump/summary.csv
dump/spec.json
audit/block_summary.csv
audit/alias_by_block.csv
audit/probe_by_block.csv
audit/top_mistakes_by_block.csv
audit/spec.json
```

`producers.csv` should be git-lfs if it becomes large. The small audit summaries can be tracked
directly.

## Non-Goals

- Do not train a new actor.
- Do not run the 81-row archive.
- Do not use oracle/future/rollout facts as deployable features.
- Do not let decision-constant system fields pass the gate through alias collapse alone.
- Do not promote raw inventory count as a meaningful feature without converting it to target-product
  cover and opportunity cost.
