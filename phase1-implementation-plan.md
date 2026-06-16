
## Progress on NEXT STEPS

**Stage 1 — observation-compatible baselines: DONE (ADR-0036).** Added obs-only
baselines (`obs_stock_aware`, `obs_price`) that decide from `env.observe(agent)` alone,
never `env.system` — the fair bar the release ladder is measured against (oracle
baselines remain the upper-bound reference). `eval/baselines.py` +
`tests/test_eval.py` (5 new tests) + `signal_check` COMBO rows.

- **The release bar is `obs_stock_aware + even_split`** — phase1.json: return ≈ 1255,
  100 % consumer fulfilment, 0 % death (within ~1 % of oracle `stock_aware` 1267; gap =
  obs blind spot to `outstanding_inbound`).
- Finding: a naive cut-off-price ordering router collapses on competition topologies
  (price 0 = "unproven", not "cheapest", and capacity is hidden by ADR-0029) — hardened
  to park unproven producers at the logit floor. See ADR-0036.

**Stage 2 — training robustness: infra DONE (ADR-0037); return bar FAILS worst-case.**
Added `summarize_seeds` (pure: median + direction-aware worst-case across model seeds)
+ `train_seeds`/`SeedSweep` + CLI `--num-seeds N`. `eval/runner.py` + `smoke_ppo.py` +
4 new tests.

- 5-seed sweep (phase1.json, 100 iters, `--gate`, 10 held-out seeds): **all 5 seeds
  0% death, 100% consumer fulfillment** — survival/fulfillment bar met across seeds.
- **But return: worst 1124, median 1216, best 1506 — below the obs-compatible bar
  ~1255** (ADR-0036). Only 2/5 seeds clear it. The earlier single-checkpoint ~1330
  was a lucky seed; the worst-case lens (the whole point of stage 2) exposes it.
- Gap = inventory holding penalty (over-stocking), NOT unmet demand or death. A
  reward/curriculum tuning follow-up, not an infra gap.

**Stage 3 — controlled generalization: infra DONE (ADR-0038); degradation axis FAILS hard.**
Added pure (torch-free) `eval/generalization.py`: single-axis, dims-preserving
perturbations (`perturb_degradation`/`perturb_quantity`/`perturb_producer_capacity` +
`generalization_suite`). 4 new tests (each asserts one axis changes + obs/action dims
fixed). Mixture / multiple-viable-producers deferred to a multi-declarer base (phase1
is single-declarer, ADR-0028).

Trained seed-0 vs obs baseline, 10 seeds/axis (one dim at a time):
- **`higher_degradation` (decay 5→10): trained COLLAPSES — 100% death, 54% fulfil,
  ret 733; baseline unaffected (0% death, 100%, 1291).** Trained restores open-loop at
  the training cadence → overfits decay rate; fails the "degrade gracefully" bar.
- `unseen_quantities` (1→4): trained generalizes well — 0% death, MORE units+return
  than baseline (119 vs 102 units, 543 vs 255).
- `leaner_producers` (cap ×0.7): non-discriminating — phase1 capacity headroom too large.

Fix = train across a degradation *range* (reward/curriculum follow-up).

**Stage 4 — combined held-out suite: infra DONE (ADR-0039); trained FAILS the bar.**
Added `combined_config` + new `perturb_costs` axis (`eval/generalization.py`), and
`eval/held_out.py`: `SeedManifest` (fixed, versioned, JSON round-trip, base resolved
relative to the manifest), `run_manifest`, `beats_baseline`→`SuiteVerdict` (pure bar:
death no-worse AND units no-fewer AND return strictly-higher). Shipped versioned
suite `eval/manifests/heldout_v1.{json,_base.json}` (competition base → competition +
costs + inventory; deg×1.2, qty≤3, cap×0.8, cost×1.3, seeds 10000–10009). 9 new tests.

Trained seed 0 (trained on competition base) vs strongest obs baseline, 10 held-out seeds:
- **Unperturbed base: trained BEATS baseline (1225 vs 852 return, equal units, 0%
  death)** — it learned the multi-hop topology; the failure below is generalization.
- **Full heldout_v1: trained COLLAPSES — 100% death, 16 units, ret 60; baseline
  unaffected (0%, 100 units, 375). VERDICT FAIL.**
- Isolation (reuse checkpoint): **each single axis alone — qty≤3, cap×0.8, OR cost×1.3
  — already drives 100% trained death** on this tight base; degradation 1.2× adds
  nothing. Broader than stage 3 (single axes survivable on loose phase1): the policy
  memorized the exact operating point; the closed-loop baseline tracks any shift.

Fix (now clearly load-bearing): **domain-randomized training** across ranges of
demand/capacity/cost/degradation on the competition topology — same family as the
stage 2/3 follow-ups, consolidated.

**Stage 5 — scale test: generator DONE + verified (ADR-0040); scale RL DEFERRED (2 blockers).**
Added pure `eval/scale.py::build_scale_config(seed, depth, products_per_layer,
producers_per_layer, num_consumers)` — a layered economy generator. Fixed structural
skeleton (recipe DAG, capacity-type set, per-producer `produces`) → **dims
seed-invariant** (all 8 config seeds → identical (prod_obs, ord_act)=(70,180), so one
net spans the suite); seed randomizes capacities/degradation/costs/demand and splits
each level's declarers into strong vs marginal (feasible-topology variation without
touching dims; `produces` held fixed because `k` depends on `max_rel`). 7 new tests.
Defaults: 12 products, 12 producers, 3 consumers, depth 3.

Scale bar (obs baseline vs noop vs oracle, depth 3, 5 seeds/config, horizon 60):
- **Env runs at scale (~27-35 agents/cycle) and poses a clean FULFILLMENT problem**:
  obs survives (0% death), fulfils 45-93 orders vs noop's 0 (100% death).
- **Finding A — depth ≥4 infeasible**: consumer fulfilment → 0 (pipeline can't prime
  vs order TTL=2). Only the shallow end of "3-5 deep" is fulfillable as-is.
- **Finding B — `return` is not a valid scale objective**: dominated by investment-order
  churn, so do-nothing has the LEAST-negative return (noop -544 > oracle -3341 > obs
  -6487). A return-maximizing policy at scale degenerates to inaction. Valid signal =
  fulfilled-unit rate. **Scale PPO deliberately NOT run** — the objective rewards doing
  nothing until the reward is recalibrated.

Blockers join the consolidated reward/curriculum follow-up (stages 2-4-5). Generator
is the reusable substrate for stage 6 (stress) once reward is fixed.

Remaining ladder stage (6 stress) is below — needs the reward recalibration first.

NEXT STEPS:
Use a staged verification ladder.

  1. Correctness gate
      - Full tests and deterministic replay.
      - Correct economy fingerprint and checkpoint compatibility.
      - Unit-based reward, curriculum, and metrics.
      - Observation-only baselines alongside oracle baselines.

  2. Training robustness
      - Train at least 5 independent model seeds.
      - Separate training, validation, and held-out environment seeds.
      - Require median and worst-case results, not one successful checkpoint.
      - Compare learning curves, survival, fulfilled units, return, and critic explained variance.

  3. Controlled generalization
     Evaluate separately on:
      - Unseen demand schedules.
      - Unseen quantities and product mixtures.
      - Numeric producer variations.
      - Multiple viable producers.
      - Higher degradation.

     Change one dimension at a time to identify failures.

  4. Combined held-out suite
     Combine randomized demand, producer competition, capacities, costs, and degradation. Use fixed versioned seed manifests. The model
     should reliably outperform the strongest observation-compatible baseline.

  5. Scale test
     Then create a larger economy:
      - 8–16 products.
      - 10–20 producers.
      - 3–5 consumers.
      - Supply chains 3–5 levels deep.
      - Several eligible producers per important product.
      - Episode-seeded demand and bounded configuration variation.

  Keep action and observation dimensions fixed across this suite. Randomize values and feasible topology within that envelope.

  6. Stress and transfer tests
      - Demand spikes and regime shifts.
      - Producer outage or rapid degradation.
      - Distribution shifts outside training bounds.
      - Zero-shot evaluation, followed separately by fine-tuning.
      - Compare against heuristics, oracle routing, and ablated policies.

  A useful release bar would be:

  - 0% death on validation and less than 5% on held-out stress seeds.
  - Better median return and fulfilled-unit rate than the strongest observation-compatible baseline.
  - No material regression on the canonical benchmark.
  - Similar results across at least 5 training seeds.
  - Performance degrades gradually outside the training distribution rather than collapsing.

  Only after passing that ladder should the result be treated as evidence that MAPPO learns general economic coordination rather than
  memorizing the Phase 1 scenario.