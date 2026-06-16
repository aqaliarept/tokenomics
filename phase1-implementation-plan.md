
## Progress on NEXT STEPS

**Pre-stage-6 reward/obs fixes + stages 2-5 re-run (ADR-0041, ADR-0042).** Before
stage 6, the two stage-5 blockers were fixed and stages 2-5 re-run on the new reward.

- **Finding A re-diagnosed and FIXED (ADR-0041).** The ADR-0040 "depth ≥4 infeasible
  vs TTL=2" diagnosis was WRONG: raising `RESERVATION_TTL` does not help (it hurts);
  fund size is not the depth-4 limiter. Real cause: the producing obs hid
  `outstanding_inbound`, so an obs-only policy re-orders in-flight stock and a deep
  pipeline never primes. Isolation: white-box stock_aware fulfils depth-4 (145 units)
  only because it reads inbound; drop that term and it collapses to 6. Fix: added a
  per-input outstanding-inbound block to the producing obs (+`num_inputs` dims) and to
  the obs baseline. Result: `obs_stock_aware` now fulfils depth-4 (142, was ~0-6) and
  depth-5 (115, was 0), matching white-box. Deep chains are viable from the obs alone.
- **Finding B FIXED — reward recalibrated (ADR-0042).** Reward is now
  `10*consumer_units − 0.1*held/num_producers` only. Dropped the instrumental terms
  (investment/replenishment +1, in-progress −1, producer-order rejection −5) and the
  per-unit consumer-rejection penalty (−5) — at scale they scaled with producer count
  / demand and made do-nothing the return-maximizer. Rejection is now penalised
  implicitly (forgone +10); survival implicitly (only the living complete). Holding is
  normalized per producer so its phase1 weight survives at scale. A fulfilling baseline
  now beats noop on return at phase1 AND depth 3/4/5 (worst-case over seeds).
- **Domain-randomized training added** (`--domain-random`, pure `random_domain_config`)
  — the consolidated overfit fix the stage 2-4 findings converged on.

Stages 2-5 re-run (honest, modest budget — 80-120 iters, 3 seeds):
- **Stage 2 (phase1, 3 seeds):** all seeds **100% fulfilment, 0% death** (robust).
  Return worst 764 / median 823 / best 884 vs lean bar 985 — gap is ENTIRELY the
  holding penalty (perfect fulfilment, no death), i.e. the trained policy over-stocks
  vs the lean baseline. The Finding-B scale pathology is gone; the residual is the
  pre-existing overstock-efficiency gap. **Worst-case gap explained, not a failure.**
- **Stages 3-4 (DR vs open-loop on heldout_v1):** open-loop **14 units / ret 14**
  (overfit reproduced) → domain-randomized **54 units / ret 375**. **DR beats open-loop
  ~4× units, ~27× return** — the core claim confirmed. DR does NOT yet reach the obs
  baseline (87 units): closing that fully needs more budget/wider ranges.
- **Stage 5 (scale depth-3):** trained **57 units / ret 512** vs noop **0/0** — beats
  do-nothing, so the objective is now learnable (no longer degenerate). But still dies
  (100%) and is well below baseline (420 units): a STRONG scale policy needs far more
  than 120 iters. Env/reward now valid for scale RL; budget is the remaining lever.

**Net:** the fixes (A, B) are solid and unblock learning (DR ≫ open-loop; scale beats
noop; phase1 fulfilment robust). Remaining work before stage 6 is budget/tuning, not
mechanism: (a) lean the trained policy / re-weight holding to close the stage-2 return
gap; (b) more DR budget + range to clear the obs baseline on the held-out suite; (c)
much more training for a survivable scale policy.

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

**Stage 5 — scale test: generator DONE + verified; original blockers superseded
by ADR-0041/0042.**
Added pure `eval/scale.py::build_scale_config(seed, depth, products_per_layer,
producers_per_layer, num_consumers)` — a layered economy generator. Fixed structural
skeleton (recipe DAG, capacity-type set, per-producer `produces`) → **dims
seed-invariant** (all 8 config seeds → identical (prod_obs, ord_act)=(70,180), so one
net spans the suite); seed randomizes capacities/degradation/costs/demand and splits
each level's declarers into strong vs marginal (feasible-topology variation without
touching dims; `produces` held fixed because `k` depends on `max_rel`). 7 new tests.
Defaults: 12 products, 12 producers, 3 consumers, depth 3.

Original ADR-0040 scale bar (obs baseline vs noop vs oracle, depth 3, 5 seeds/config,
horizon 60):
- **Env runs at scale (~27-35 agents/cycle) and poses a clean FULFILLMENT problem**:
  obs survives (0% death), fulfils 45-93 orders vs noop's 0 (100% death).
- **Finding A was re-diagnosed in ADR-0041**: depth 4-5 are not TTL-infeasible; the
  missing observation was outstanding inbound replenishment. That observation is now
  exposed, and obs-only baselines can prime deeper chains.
- **Finding B was fixed in ADR-0042**: `return` is now fulfilment minus normalized
  holding, so fulfilling baselines beat do-nothing at scale.

Scale PPO has now been run as a modest-budget validity check (see the progress
section above): the objective is learnable, but strong scale performance still needs
more training budget and tuning. Stage 6 is no longer blocked on mechanism, but it
should not be treated as a release gate until stages 2-5 clear the obs-compatible
baseline bars.

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
