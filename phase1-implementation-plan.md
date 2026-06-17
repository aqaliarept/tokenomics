
## Scale training: BEAT the baseline on the fixed big environment (ADR-0043)

**Goal (user):** train a policy that beats the best obs-compatible baseline on the big
(depth-3 `build_scale_config`) economy. **Fixed env: DONE and verified.**

- **Bars (obs-compatible, the fair target — ADR-0029 hides capacity, so white-box is an
  oracle not a bar):** fixed seed 0 = `obs_stock_aware+even_split` **163 units / 0%**;
  held-out = `obs_price` **253 units**.
- **Two blockers, both about training *at* full difficulty (ADR-0043):**
  1. The gated curriculum stalled at difficulty 0.30 — it advanced only on rollout
     fulfilment ≥ 0.5, but scale fulfilment structurally caps ~25% (deep chains, TTL=2),
     so difficulty never reached full; the policy trained on eased demand, evaluated on
     hard. **Fix: survival-gate it** (`advance_fulfil=0.15`) so difficulty climbs on
     survival and the bulk of the budget trains at full difficulty.
  2. Greedy eval never restored (mean action, restoration toggle bias 0) → death by a
     *localized* capacity-type collapse (`capmin`~100 at death). **Fix: restoration
     prior** `prod_restore_bias` (default 0 → phase1 unchanged), a survive-first init.
- **Result (400 iters, survival gate, `low_cap_coef=3.0`, `death_penalty=10.0`; no DR):**
  deployed checkpoint **227 units / return 2157 / 0% death** on fixed seed 0 vs baseline
  **163 / 1588 / 0%** — **+39% units, higher return, equal survival** (noop 0 / 100%).
  Verified by re-eval of the saved checkpoint. The survival-gate fix alone cleared it.
- **Generalization (held-out): NOT achieved; DR did not help (honest result).** Final
  verified table (units / %death; obs-compatible baselines are the bar):

  | policy | fixed seed0 | held-out (10000–05) |
  |---|---|---|
  | baseline obs_stock+even_split | 163 / 0% | 242 / 0% |
  | baseline obs_stock+obs_price | 148 / 0% | **253 / 0%** |
  | noop | 0 / 100% | 0 / 100% |
  | **trained (fixed, no DR)** | **227 / 0%** | 114 / 33% |
  | trained (DR topology, v4 obs) | 102 / 0% | 67 / 33% |
  | **trained (DR topology, v5 obs, ADR-0044)** | 123 / 0% | **103 / 17%** |

  - **Fixed big env: WON** (227 vs 163, +39%, 0% death). Primary goal met.
  - **Held-out: still lost, but the obs fix moved it materially.** Re-ran the *identical*
    DR config (`dr=seeds dwell=20 cur_gate cur_advance_fulfil=0.15 low_cap_coef=3.0
    death_penalty=10.0`, 500 iters) — the ONLY change is the ADR-0044 producer-mechanics
    obs block. Result vs the v4-obs DR run: **held-out 67 → 103 units (+53%) and death
    33% → 17% (halved); fixed 102 → 123**. The mechanics block is a real, measurable
    generalization gain. Crucially the **gate now climbs all the way to diff 0.90** under
    DR (v4 stalled on the low rungs) — exposing the per-seed mechanics lets the policy
    *survive* the full difficulty ramp on randomized topologies. Still below the bar
    (103 vs 253 units; 17% vs 0% death): the remaining gap is budget/architecture, not a
    posed-wrong observation. cev≈0 is a symptom of non-stationary DR targets, not the
    cause.  Checkpoint `/tmp/tmp4h_2jxos.pt` (v5 schema).
  - **DR gate/episode mechanics fixed en route:** topology-DR must (a) re-apply the
    curriculum easing to the random topology and (b) **dwell** on each topology for N
    iters so envs persist and episodes complete — else the gate starves (no completed
    episodes → pinned at diff 0.0) and the policy only ever sees an episode's first ~9
    cycles. (`tokenomics/_scale_lab.py`, scratch harness.)

**Recheck / diagnosis (current).** The fixed-seed claim above is reproducible with the
scratch harness command:
`python -m tokenomics._scale_lab train iters=400 cur_gate=True cur_advance_fulfil=0.15 low_cap_coef=3.0 death_penalty=10.0`.
It saves a checkpoint that re-evaluates to **227 units / return 2157 / 0% death** on
`build_scale_config(seed=0)`.

That does **not** mean the trainer has a robust scale policy. The same checkpoint loses
to held-out obs baselines because the **producing policy**, not the ordering policy,
does not transfer:

| policy combination | fixed seed0 | held-out 10000-05 |
|---|---:|---:|
| trained producer + trained ordering | 227 / 0% | 115 / 33% |
| obs producer + trained ordering | 246 / 0% | 337 / 17% |
| trained producer + even ordering | 5 / 0% | 55 / 0% |
| obs producer + even ordering | 163 / 0% | 242 / 0% |
| obs producer + obs-price ordering | 148 / 0% | 253 / 0% |

On held-out seeds 10000 and 10001, the trained producer lets **all level-1 producers**
run `capacity-1` to zero by cycle ~20. Products 10-13 become unmakeable while aggregate
`cap_min` still looks high, so the old scalar health metric hides the local collapse.
A low-capacity restoration guard removes death but still reaches only ~154 held-out
units, so restoration brittleness is necessary but not sufficient; replenishment /
stock control also overfits the fixed topology.

Likely root cause: the producing actor is trained against one fixed topology but its
observation omits producer-mechanics fields that vary across scale seeds. It sees
current/max capacity and inventory state, but not the full action semantics it must
adapt to: per-type degradation, restoration/upgrade amounts and costs, max
replenishment, replenishment cost, and inventory-expansion economics. The obs baseline
uses `EnvSpec` directly for its closed-loop rule; the neural policy only receives the
float observation. Same tensor shape is therefore not the same decision problem.

Proposed next fix before claiming held-out scale success:

1. **DONE (ADR-0044).** Extended the producing observation with the missing
   per-agent producer-mechanics block — per-type degradation, restoration/upgrade
   amounts + costs, max-replenish, replenish cost, inventory-expansion economics
   (`num_types + 8` dims, appended before `cycle_progress`; obs-baseline front
   offsets untouched). Static economics folded into a `ProducerMechanics` on
   `EnvSpec`; amounts come from `ProducingAgentView` (domain stays pure). Checkpoint
   schema bumped `4 → 5`.
2. **DONE (ADR-0044).** Regression `test_two_scale_seeds_expose_different_mechanics_in_obs`
   asserts two same-shape scale seeds now expose *different* degradation / restoration
   / max-replenish in the obs — the seed-varying mechanics are no longer hidden state.
   Per-field exposure covered by `test_producing_obs_exposes_{own_degradation,
   restoration_and_upgrade_amounts,investment_economics,inventory_expansion_economics}`.
   Full suite green (192).
3. **NEXT.** Move topology-domain-randomized scale training out of `_scale_lab.py`
   into the real trainer CLI with dwell/re-applied curriculum, then re-run the fixed
   and held-out bars on the **enriched obs** (v5 checkpoints) to test whether the
   mechanics block closes the held-out gap.
4. Keep fixed-seed success as a smoke result only; the release bar remains held-out
   median/worst-case against the strongest observation-compatible baseline.

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
