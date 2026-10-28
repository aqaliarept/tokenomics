# Design

Motivation: `proposal.md`. Requirements: `specs/development-audit/spec.md`,
`specs/gate-harness/spec.md`. Roadmap: `establish-project-baseline` design D1 (M3 row) and D4.

## Context

Facts from the code and the record that shape the plan:

- **How `random` routes.** `random_action` (`eval/baselines.py`) returns
  `env.action_space(agent).sample()`: one uniform logit in [−10, 10] per (slot, producer)
  (`ACTION_LOGIT_BOUND = 10`, `env/spaces.py`). `allocate_suborders` (`env/allocator.py`)
  restricts the logits to the order's eligible producers, applies a softmax, splits the quantity
  by largest remainder, and splits the tokens along the integer quantities.
  - Sparse-wide worlds have fanout 2, and consumer quantities are 1–2 or 1–3 (`eval/scale.py`,
    `quantity_min = 1`, `qd = randint(2, 3)`). With logit difference d over 2 eligible
    producers, the largest-remainder split sends the order whole when |d| exceeds a threshold:
    - a quantity-1 order always goes whole;
    - a quantity-2 order goes whole when the larger share is above 0.75, i.e. |d| > ln 3, in
      (1 − ln 3 / 20)² ≈ 89% of decisions;
    - a quantity-3 order goes whole when the smaller share is below 1/6, i.e. |d| > ln 5, in
      ≈ 85% of decisions.

    (The larger share is ≥ 0.99 in only ≈ 59% of decisions, but rounding makes whole routing
    far more frequent.) So in practice `random` sends almost every order whole to one producer
    picked at random, and draws again every cycle, every order and every replicate.
- **How the deterministic routers round.**
  - `even_split` emits zero logits. With equal fractions, `_allocate_int` gives the leftover unit
    to `np.argsort(-frac)`'s first index, and for equal keys that is the lowest index (checked:
    `argsort(-[0.5, 0.5]) = [0, 1]`). So a quantity-1 order always goes to the lowest-indexed
    eligible producer, and the odd unit of a quantity-3 order goes there too.
  - `proportional` (shares ∝ throughput, net of reservations) and `blend_0.5`
    (shares ∝ √throughput) give a quantity-1 order to the larger share every time.
  - So `even_split` splits every quantity-2 order 1/1 and every quantity-3 order 2/1, and
    `proportional` / `blend_0.5` split whenever the eligible producers' throughputs are similar.
  - None of the three sees what other orderers route in the same cycle: every orderer reads the
    same pre-auction snapshot.
- **Whole-order completion.** An order completes only when its whole quantity completes
  (`domain/order.py`, module docstring and `complete_cycle`). A rejected sub-order's quantity
  is re-bid in a later cycle within the TTL. A failed portion sinks the whole order, and the
  accepted part's tokens stay burnt. The env counts fulfilled units only for completed orders
  (`env/aec_env.py` `_accumulate_consumer_units`). An order split over two producers therefore
  completes only if both sub-orders complete in time (≈ p₁·p₂ against p for a whole order).
- **Auction.** The auction (`domain/producing_agent.py` `produce`) is atomic and greedy by price.
  A bid is rejected because of an outage, or because it does not fit what is left after
  higher-priced bids. The domain does not record which. Failed reservations (dropped on
  re-validation, or TTL expired while waiting for inputs) also carry no reason.
- **Who `random` routes for.** `random` routes every ordering agent: consumer orders, and also
  producers' input purchases and investment orders (each producing agent has its own ordering
  agent). So it also changes the producers' cut-off prices, and with them the
  investment-fund distribution.
- **Where `random` wins (EXP-0047, recounted from `episodes.csv`, `entity_planner` members).**
  `random` sets the per-episode best, ties included, on:
  - 6/24 at 12-calm, 28/40 at 12-shocked, 21/24 at 42-calm and 40/40 at 42-shocked;
  - ties between deterministic members explain why the 12-shocked count differs slightly from
    `RESULT.md`.

  The median of random's mean over the best deterministic member is:

  | Cell | Median | Range |
  |---|---|---|
  | 12-calm | 0.967 | — |
  | 12-shocked | 1.024 | — |
  | 42-calm | 1.017 | — |
  | 42-shocked | 1.050 | 1.010–1.100 |

  - *Random's worst replicate* is its lowest-scoring run among the 5 replicates of an episode.
    It still beats the best deterministic member on 38/40 episodes at 42-shocked, 13/24 at
    42-calm and 18/40 at 12-shocked.
  - The median relative spread between replicates is 2.9%.
  - This is a preliminary read on explanation (e) below. EXP-0048 confirms it on its own rows.
- **EXP-0006B.** A learned orderer with a small per-dimension σ ∈ [0.14, 0.60], deployed by
  sampling, closed the seeds that `stock_aware + random` won. So stochasticity helped with noise
  far below `random`'s scale.
- **Teacher code.** The EXP-0042 teacher is still on `new-model`:
  - `eval/bounded_rollout_orderer.py` (`BoundedScenarioOrderer`, a `blend_0.5` floor, a +1-logit
    bump, confidence checkpoints at 4/8/16 scenarios);
  - `env/future_fork.py` (`fork_future_demand`: a deep copy that resamples the unconsumed demand
    tail independently from the legal pool).

  Under shocks a deep copy carries the true shock schedule and the domain's outage state, and
  the independent tail resampling breaks drift continuity. Both violate the D4 fork rule.
  Nothing has to be restored from `archive-1`. The EXP-0033 analytic bound code was removed
  earlier (18c717d) and is not needed: the analytic demand bound is the ordered units already
  in every row (fill rate 1.0).
- **Teacher cost.** EXP-0042 spent 267–758 s per episode at depth 2 and 532–1461 s at depth 3,
  on 6–9-producer worlds with 16 workers. Cost on the 12/42 gate worlds is unknown, so a probe
  is required.
- **Baseline episode cost.** EXP-0047: 12,800 roster episodes in about 3.7 vCPU-h of execution,
  ≈ 1.04 vCPU-s per horizon-100 episode, averaged over 12 and 42 producers.
- **Deployed-path refactor not done.** The D2 refactor (eligible-only rows, indexed per-producer
  telemetry, O(E) remainder) has not been done. Only the observation-cost fix of M1 task 8.4
  landed (DIAGNOSTICS 17). PoC-B measures the path as it is.
- **Scale and budget (2026-10-27).** The acceptance target is 12 → 42 producers, 120 is dropped
  for now, and there is no budget for runs of 1000+ vCPU-h.

## Goals / Non-Goals

**Goals:**
- A diagnosed, recorded cause for `random` setting the hindsight baseline before PoC-A is read.
- PoC-A, PoC-B and PoC-C recorded as EXP results, with honest teacher information sets.
- An M4 input: is there headroom, which role trains first, what does the deployed path cost.

**Non-Goals:**
- No training, no M4 arm selection beyond "which role first", and no deployed search.
- No change to frozen v1/v2 files, to roster members' behaviour, or to the acceptance margins.
- No producer search teacher: PoC-C uses the best hand-written producer as the producer-role
  reference (D6).
- No economy above 42 producers in any probe, unless the user approves a larger PoC-B timing point
  (Open questions); such a point would be non-gating.

## Decisions

### D1. Order of work and gates

```
routing diagnosis (1-3, EXP-0048) --> [user: roster v3? only if a rule defect, D3]
        --> PoC-A (4-5, EXP-0049) --> PoC-C (6-7, EXP-0050)
PoC-B (8-9, EXP-0051): depends only on its own code; runs in any order after its probe
M3 exit record (10)
```

PoC-A waits for the EXP-0048 record because its ratios are read against a bar that `random`
mostly sets. If EXP-0048 diagnoses a defect in a hand-written rule, PoC-A also waits for the
user's roster decision (D3), since that decides which bar it is read against. PoC-B shares
nothing with the other probes. The EXP numbers are planned; an inserted diagnosis takes the next
free number and shifts them.

### D2. Random-routing diagnosis

**Competing explanations, checks and consequences.** "Edge" is random's mean units minus the
best deterministic roster member's units on the same world-episode. A variant "recovers the
edge" in a size × regime cell when its equal-weight world mean recovers at least 0.8 of
random's edge there. The cells where random dominates are 12-shocked, 42-calm and
42-shocked; "most cells" means at least 2 of these 3. The thresholds are pre-registered here.

| | Explanation | Discriminating check | If confirmed, what changes |
|---|---|---|---|
| (a) | **Packing / atomic acceptance**: deterministic routers pile sub-orders onto the same producers, and whole bundles do not fit or are displaced | Per-bid rejection reason (outage / no-fit / displaced) and failure reason (dropped / TTL), units lost per reason, per router, on identical episodes; `proportional_load_aware` (below) | Herding is a rule defect, so D3 applies (a load-aware rule as roster v3 candidate). M4's residual context must see same-cycle load. |
| (b) | **Temporal variation, not randomness**: spreading orders over time is what helps | `round_robin` (deterministic rotation, whole orders) against `proportional_whole` (deterministic, whole orders, no rotation); `random_fixed_episode` (random weights drawn once per episode); the scale sweep `random_scaled_s`, s ∈ {0.5, 1, 2, 5}, against `random` (s = 10); `blend_noise_σ`, σ ∈ {0.15, 0.6} (EXP-0006B's σ range). Every comparison is read at matched split degree (tracer), because the sweep and `round_robin` also change how often orders are split, which is (f). | Read only after the (f) split control. `round_robin` recovers the edge and `proportional_whole` does not, both whole: rotation is the lever, a deterministic rotating rule is a roster v3 candidate (D3), and M4 does not need stochastic deployment. `random_whole` recovers it and neither deterministic whole router does: genuine stochastic benefit (symmetry breaking among concurrent orderers), so M4 keeps sampling at deployment (EXP-0006B) and the roster is unchanged. Fixed weights also recover it: any decorrelated allocation beats the rules, which points back to (a)/(c)/(f). An edge that grows with s but vanishes at matched split degree is (f), not stochasticity. |
| (c) | **Rule defects** in `even_split` / `proportional` / `blend_0.5`: the lowest-index tie-break, rounding of 1–3-unit orders, routing to producers that cannot fulfil, slot overflow (`k_stream`) | Unit-level single-episode trace (task 2.1). Per order: quantity, eligible set, shares, integer split, receiving producer, outcome and reason. Plus aggregates: share of quantity-1 units to the lowest-index eligible producer, routes to zero-throughput producers, overflow orders. Each defect then gets its fixed variant (`even_split_rotating`, `proportional_load_aware`) in EXP-0048. | A defect whose fixed variant recovers the edge: D3 (new baseline name, roster v3 draft, re-screen, user gate). |
| (d) | **Investment-fund channel**: random routing of producers' purchases and investments changes cut-off prices, and with them fund concentration and capacity run-down | `random_consumer_only` and `random_producer_only` (random on one order class, `blend_0.5` on the other); per cycle, each producer's fund share and cut-off price, the capacity ratio, and the investment rejection rate, per router | The producer-only variant recovers the edge: the lever is the producer-side purchasing route, so PoC-C is read with this in mind, and a producer-orderer rule fix goes through D3. The consumer-only variant recovers it: the lever is consumer routing, (a)–(c). |
| (e) | **Scoring artifact**: random is the mean of 5 replicates and the deterministic members are single runs | Per-replicate distribution; count of episodes where random's worst replicate beats the best deterministic member; R = 1 versus R = 5 | Random wins only through averaging or ties: no rule change, and the gate documentation notes it. Preliminary (Context): not an artifact at 42-shocked (38/40). |
| (f) | **Split penalty under whole-order completion**: the rules split 2–3-unit orders over both eligible producers, `random` almost never does, and a split order completes only if both sub-orders complete | The 2×2 of whole/split × deterministic/random, at identical episodes: `proportional_whole` and `round_robin` (deterministic, whole), `random_whole` (random, whole), `even_split` / `even_split_rotating` (deterministic, split) and `even_split_random_leftover` (random, split). Tracer: split degree per order (sub-orders per order, share of split orders) and orders rejected with completed units > 0 (completed but wasted units), consumer and producer orders separately | The whole routers recover the edge and the split routers do not, deterministic or random alike: the hand-written rules' splitting is a rule defect for this order lifecycle, so D3 applies (a whole-order rule as roster v3 candidate). M4's residual must be able to route whole, and the reading of (b) uses the whole routers only. If wasted completed units are small for every router, splitting is not the lever. |

The explanations are not exclusive. EXP-0048 reports each one's share of the edge per cell, and
the recorded cause names every explanation that recovers the edge.

**Diagnostic routers** live in a new `eval/routing_diagnostics.py` as `DIAGNOSTIC_ROUTERS`. They
are not in `ORDERING_BASELINES`, so the roster, `signal_check` and the gate never see them
(spec "Diagnostic routers stay outside the roster"). Each is a `Policy` with the roster
contract. The producer is `entity_planner`, the producer of every roster v2 member. Stochastic variants draw from the env action-space RNG,
seeded by the gate's replicate seed (`eval/seeding.py`), with R = 5.

| Router | Kind | Definition |
|---|---|---|
| `round_robin` | deterministic | Whole order to eligible producer number (cycle + stable hash(agent, slot)) mod E |
| `even_split_rotating` | deterministic | Even shares; leftover units go to producers by a cycle-indexed rotation, not the lowest index |
| `proportional_load_aware` | deterministic | `proportional`, with throughput also net of the capacity bids already placed on the producer this cycle |
| `proportional_whole` | deterministic | Whole order to the producer with the largest `proportional` share; ties to the lowest index; no rotation |
| `random_scaled_s` | stochastic | Uniform logits in [−s, s] |
| `random_fixed_episode` | stochastic | Uniform [−10, 10] logits drawn once per (agent, slot, producer) at reset, then held |
| `blend_noise_σ` | stochastic | `blend_0.5` logits + N(0, σ²) |
| `random_consumer_only` / `random_producer_only` | stochastic | Random on one order class, `blend_0.5` on the other |
| `random_whole` | stochastic | Whole order to one eligible producer drawn uniformly |
| `even_split_random_leftover` | stochastic | Even shares; leftover units go to eligible producers drawn uniformly (the split counterpart of `random_whole`) |

The six roster v2 members are rerun with the trace as references, each under the contract its
roster entry records (`roster_v2.json`: `entity_planner/obs_price` runs `legacy`, the other five
`joint_stack`), so their rows must reproduce EXP-0047. The diagnostic routers run `joint_stack`:
they are white-box like `proportional` / `blend_0.5`, whose roster contract is `joint_stack`
(`eval/roster.py` `contract_for`), so they share the mechanics of the routers they are compared
with.

**Telemetry.** Rejection and failure reasons are additive pure data on the domain's per-cycle
production result (`domain/producing_agent.py` `ProductionResult`). For each rejected bid the
domain records outage, no-fit (it does not fit the budget available at the start of the auction
step) or displaced (it fits that budget but not the remainder). Failed reservations get dropped
or TTL. The classification is integer arithmetic on state the auction already holds: no RNG, IO
or logging, and goldens stay byte-identical (spec "Tracing changes no statistic"). An eval-side
tracer (`eval/routing_trace.py`) collects per-episode aggregates:
- units lost per reason, consumer orders and producer orders separately;
- split degree per order (sub-orders per order, share of split orders) and orders rejected with
  completed units > 0 (completed but wasted units), consumer and producer orders separately;
- a Herfindahl concentration of routed units per product and cycle;
- the quantity-1 lowest-index share;
- routes to zero-throughput producers;
- overflow orders;
- per-producer fund share and cut-off price, and two concentrations of the granted fund (task
  2.2; reference 1/N for N producers; shares of the whole fund square to about 0, since only
  about 3% of it is granted per cycle): the episode Herfindahl of each producer's total grants,
  which (d) reads for investment starvation, and the per-cycle Herfindahl of each cycle's
  grants averaged over cycles with grants, reported but confounded with how many producers are
  granted per cycle;
- the capacity ratio trace;
- the investment rejection rate.

Alternative considered: reconstructing reasons in eval by replaying the auction. Rejected
because it duplicates domain logic and can drift from it.

**EXP-0048 grid.**
- All 128 v2 development world-episodes.
- 9 deterministic runs: the 5 deterministic roster members and 4 variants.
- 12 stochastic routers × R = 5 (`random` among them, the sixth roster member).
- 69 runs per world-episode, 8,832 horizon-100 episodes.
- At the EXP-0047 rate times the tracer overhead measured by the probe (assumed ≤ 1.5), that is
  about 2.5–4 vCPU-h. Hard stop 8 vCPU-h.
- Driver: `eval/routing_audit.py`. It reuses the gate's registry loading, replicate seeds, digest
  checks, contract checks and EXP layout, and writes `episodes.csv` (gate row format plus trace
  columns), `routing_metrics.csv`, `edge.csv`, `spec.json` and a `RESULT.md` skeleton.

### D3. What a found rule defect does to the gate

- A defective hand-written rule is never edited in place. Frozen roster v1/v2 name it, and their
  content hashes and "Legacy measurement unchanged" require that it replay identically.
- The fix is a new `ORDERING_BASELINES` entry with a new name, for example
  `even_split_rotating`. It is promoted from `DIAGNOSTIC_ROUTERS` only by a code task after the
  user's decision.
- The draft roster v3 is built the way v1 and v2 were (gate-harness "Frozen baseline roster",
  as modified here): every hand-written producing baseline paired with every hand-written
  ordering baseline that exists when v3 is drafted (the fixed rule and `shock_reactive`
  included), plus `blend_0.5`. The recorded never-sets-the-best pruning rule is then applied
  before freezing ("Roster pruning uses development evidence only"). It is not "v2 members plus
  the fix": v2's pruning was evidence for v2's member set only. The re-screen costs about what
  EXP-0047 did (≈ 4 vCPU-h).
- A baseline-only `gate --mode screen` of a draft registry v3 then runs. It has the same worlds
  and seeds as v2, so it is a re-screen. It writes frozen registry, roster and parameter v3
  sharing one `screened_on`. X and Y may move.
- The frozen v2 files are untouched, and EXP-0045 and EXP-0047 keep naming v2.

This raises the bar every candidate is judged against, so it is a **human gate**: the supervisor
stops after EXP-0048 and asks the user whether to open roster v3. Making v3 the default is a
second approval after the screen (spec "Raising the bar is a user decision"). The supervisor
inserts the tasks for this path when the user says yes. They are not pre-planned here.

### D4. Teacher fork rule (PoC-A, PoC-C)

New `env/shock_posterior.py`, in the env layer because it draws random numbers.

- **Observable history.** The extractor may read exactly these public sources:
  - every agent's encoded observation (`env.observe(agent)`), which includes each producer's own
    capacities;
  - the domain-owned telemetry the ordering views expose (per producer and product: routed,
    accepted, completed, rejected and failed EMAs and counts), which the orderers already see;
  - realised consumer orders and the agents' own actions.

  It may not read the env's shock schedule, the domain's shock or outage state, or any
  `ProductionResult` field that no observation encodes. In particular it may not read the
  rejection and failure reasons task 1.1 adds, since `outage` there is computed from hidden
  state. Tests replace the schedule and shock state with a sentinel that raises on access, as
  the `shock_reactive` test does, and show labels byte-identical when the 1.1 reason fields are
  scrambled or removed.
- **Posterior sampling.**
  - The `shocks-i` generator draws a fixed number of events per kind, each i.i.d.: a uniform start
    in the window, then duration, magnitude and target.
  - **Constructive conditional sampler, not prior rejection.** Rejection from the prior is
    infeasible. Under `shocks-i` a prior particle reproduces one observed hit with probability
    ≈ 3e-4 at 12 producers (start 1/70 × target 1/12 × type × magnitude interval × slots), so
    almost every shocked decision after the first hit would be unresolved. The events of a kind
    are i.i.d. and exchangeable, so the posterior factorises. A particle is built in three steps:
    1. **Observed consequences → assigned slots.** Each unambiguous observed consequence is
       assigned to one event slot of its kind. Its parameters are drawn from the generator
       conditioned on the observation:
       - a capacity hit: start, target and capacity type fixed; m uniform on {m in the
         profile range : floor(m·c) equals the observed drop};
       - an outage: target fixed; start uniform over the cycles consistent with the first
         observed blocked acceptance and the last observed normal one.
    2. **Ambiguous observations → a mixture.** A producer that rejected every bid may have been
       in an outage or simply full. Each consistent explanation is chosen with weight
       proportional to its generator probability, and its parameters are drawn as in step 1.
       Rejection is used only within this mixture.
    3. **Remaining slots → the truncated generator.** The rest of the fixed per-kind count is
       drawn from the generator truncated to "not observable by cycle t": an onset after t, or
       an event on a producer whose past cycles could not have revealed it (for example an
       outage on a producer that received no bids in its active cycles).
  - **Validation, two-sided.** Every constructed particle is replayed with the agents' recorded
    actions, and it is kept only if it **reproduces every observed consequence** and **implies
    none that was not observed**. The acceptance rate therefore measures real residual
    ambiguity, not prior improbability. Consequence-level predicates may replace the replay where
    they are exact; the replay from reset is the reference the tests compare against.
  - An active outage's remaining duration is drawn from the duration range conditioned on at
    least the elapsed cycles.
  - Sampling runs once per decision cycle and is cached for every orderer in that cycle.
- **Pre-registered numbers.**
  - Draw budget: 1,024 per decision cycle.
  - Minimum accepted: 16, the EXP-0042 maximum scenario count. The acceptance floor is 16/1,024.
  - Below the floor the cycle's decisions are "unresolved" and act as `blend_0.5`.
  - The acceptance rate is logged per cycle.
- **Stripping.** A fork's domain state loses the true pending shock events and outage end
  cycles before the sampled ones are injected (spec "Hidden fields stripped").
- **Drift continuity.** `env/future_fork.py` gains a drift-continuous resampler. From the current
  cycle's orders it applies the drift process (at most floor(j·N) changes per transition) on a
  fork-seeded stream. B9 worlds without drift keep the independent tail; calm cyclic schedules
  are not resampled.
- **Deterministic futures.** On calm worlds nothing is hidden, so every scenario is identical and
  the teacher uses one scenario. With zero variance, the confidence rule accepts a strictly
  positive gain. Pre-registered.
- **ADR.** The information-set rule is a design decision, so task 4.1 writes ADR-0056 "Teacher
  information set and posterior forks". It covers the public sources, the two-sided acceptance,
  the floor, stripping, drift continuity and the deterministic-future rule, and cites D4 of
  `establish-project-baseline` and ADR-0055.
- **The byte-identical-labels pytest.** Two envs share the observable history up to cycle t, and
  differ in a true remaining outage duration and a future capacity hit. For the same fork seed
  they yield byte-identical teacher labels: every stream's bump, unresolved flag and acceptance
  rate.

Alternatives considered:
- Sampling hidden state from the prior and ignoring history. Rejected: it contradicts observed
  consequences, so it produces labels no observation can support, which is the EXP-0038B cause.
- Prior rejection against the history. Rejected: it is correct but infeasible under the
  `shocks-i` profile (≈ 3e-4 acceptance per observed hit), so the unresolved fallback would
  swallow the shocked episodes.

### D5. PoC-A, the teacher against the hindsight baseline (EXP-0049)

- **Arms.** All arms use the `entity_planner` producer and the joint-stack contract.
  - floor: `blend_0.5`, a roster member;
  - teacher at depth 2;
  - teacher at depth 3;
  - UB: a clairvoyant depth-3 search over the true future, with one scenario.

  The teacher configuration is otherwise EXP-0042's frozen one. On calm worlds UB equals depth 3
  by construction and is not run; a test asserts the identity on a fixture.
- **Why `entity_planner` and not the portable producer.** Every roster v2 member uses it, so the
  ratio isolates routing headroom. The portable producer enters in PoC-C.
- **World-episodes.** A pre-registered subset of v2 development.
  - Calm worlds: 1 episode each, the first episode seed, 16 world-episodes. Calm demand is
    identical across episode seeds, and the deterministic teacher replays identically.
  - Shocked worlds: the first 3 episode seeds in registry order, 24 world-episodes.

  The hindsight baseline is the frozen roster v2 (or v3 if the user adopted it, D3), run on the
  same world-episodes with demand and shock digests checked.
- **Reading** (spec "Teacher audit reading").
  - Per-world ratio, regime means per size, worlds below 1.0, deaths, and the unresolved fraction.
  - Necessary checks: regime margin (≥ 1.05 calm, ≥ 1.10 shocked) and per-world no-loss.
  - UB / hindsight baseline ≥ 1.15 is the optional slack target.
  - It is not an acceptance verdict.
  - Each arm's per-world ratio is also reported against the **deterministic-only hindsight
    baseline**: the same roster without `entity_planner/random`. This is a diagnostic, not a
    margin. It separates "the teacher cannot beat random's mechanism" from "the teacher cannot
    beat the rules".
  - A shortfall is diagnosed by world family, roster, fork (unresolved rate) or floor, and
    recorded (spec "A failed probe ends in a diagnosis"). It never moves a margin.
- **What EXP-0048 changes for PoC-A** (pre-registered reading per diagnosed cause):

  | EXP-0048 cause | Bar PoC-A is read against | How a shortfall is read |
  |---|---|---|
  | Rule defect ((a), (c) or (f)), roster v3 adopted by the user | v3 | as above; random's share of the v3 bar is reported |
  | Rule defect, v3 declined, or rotation (b) | v2, plus the deterministic-only bar | Shortfall on the full bar and pass on the deterministic-only bar = "the teacher beats the rules but not the mechanism EXP-0048 named". The `blend_0.5` floor lacks that mechanism (whole-order routing, rotation, or load awareness). Recorded as a floor limitation, not as "no headroom". |
  | Genuine stochastic benefit (b) | v2, plus the deterministic-only bar | Same split. Also recorded: the deterministic teacher's labels cannot express sampling, so M4 must keep stochastic deployment whatever PoC-A shows. |
  | Investment-fund channel (d) | v2, plus the deterministic-only bar | Same split, read together with PoC-C. The lever is producer-side purchasing routing, which the teacher searches only through producer orderers. |
  | Scoring artifact (e) | v2 | The full-bar reading stands as is. |

  Several causes at once: every matching row applies.
- **Driver.** A new `eval/teacher_audit.py`. `eval/bounded_rollout_orderer.py` gets a pluggable
  scenario source (posterior fork / deterministic / clairvoyant), a baseline producer option,
  and registry-world input, besides its archive tasks. Row format, digests, contract checks and
  hindsight-baseline arithmetic are reused from `eval/gate.py`.
- **Sizing.** Task 4.3 probes each arm × size × regime cell. A 42-producer cycle may not finish
  in the probe's wall cap, so the probe records the costs that compose a cycle:
  - streams searched per cycle and seconds per searched stream (scenarios × rollout depth);
  - posterior-sampler seconds per cycle, at a horizon-100 history length (task 4.3 builds that
    history by playing a cheap baseline episode to cycle 90 first);
  - a local-to-pod speed factor (one baseline horizon-100 episode per size timed locally against
    EXP-0047's ≈ 1.04 vCPU-s per episode on the pod).

  Hard stop 40 vCPU-h. If the extrapolation exceeds it, the grid is reduced in this
  pre-registered order:
  1. drop depth 3 at 42;
  2. use 2 shocked episodes per world at 42;
  3. drop UB at 42;
  4. run the 12-producer grid alone. The 42-producer part waits for a profile-and-optimize code
     task (compute-budget rule), which the supervisor inserts, and is run after it as its own
     launch under the same stop.

  If the 12-producer grid alone exceeds the stop, nothing launches and the implementer reports
  to the supervisor.
- **A-priori cost** (to be replaced by the probe; the arithmetic, not a measurement):
  - EXP-0042 depth 2: 267–758 s per horizon-100 episode with 1,002–1,598 streams searched
    (6–9 producers), i.e. 2.7–7.6 s per cycle and ≈ 0.5 s per searched stream. Depth 3 costs
    about twice that.
  - At 12 producers, assume ×1–2 per stream. PoC-A's 12-producer grid has 20 world-episodes
    (8 calm at 1 scenario, 12 shocked). d2 + d3 + UB ≈ 4× a d2 episode, so the grid costs about
    20 × 4 × (0.07–0.4) vCPU-h ≈ 6–30 vCPU-h. The calm cells cost less (one scenario).
  - At 42 producers, more streams per cycle (consumers and producers co-scale) and dearer
    rollouts (DIAGNOSTICS 17: a cycle costs several times more at larger P) plausibly put a
    search cycle 1–2 orders of magnitude above the 6–9-producer one, about 3–30 vCPU-h per
    shocked d2 episode. The 42 part is therefore **likely to exceed the stop**: the expected
    outcome of task 4.3 at 42 is step 4 above (12 first; 42 after a profile-and-optimize task).

### D6. PoC-C, role attribution under shocks (EXP-0050)

- **Arms**, on PoC-A's 24 shocked world-episodes at both sizes:
  - S: the portable gate subject, EXP-0035 `producer_candidate.pt` + EXP-0012
    `orderer_scorer.pt`;
  - A: portable producer + teacher orderer at depth 2, the EXP-0042 promoted depth, which is
    cheaper than depth 3;
  - B: `entity_planner` + portable orderer;
  - T: `entity_planner` + teacher at depth 2.
- **T rows.** T is reused from EXP-0049 only when the run commit's source fingerprint over the
  teacher path (`eval/bounded_rollout_orderer.py`, `eval/teacher_audit.py`,
  `env/shock_posterior.py`, `env/future_fork.py`, `domain/`) equals EXP-0049's. Otherwise T is
  rerun.
- **Arithmetic, per size.**
  - Gap: G = Σ_w (T_w − S_w), over world mean units.
  - Orderer closure: c_O = Σ_w (A_w − S_w) / G.
  - Producer closure: c_P = Σ_w (B_w − S_w) / G.
  - Interaction: T − A − B + S, reported.
  - Pre-registered band: |c_O − c_P| < 0.15 is a tie and goes to the user.
  - G ≤ 0 means the attribution is undefined; it is recorded and diagnosed.
- **Limitation, recorded.** The producer-role reference is a hand-written producer, not a
  teacher, because no producer search teacher exists. c_P therefore measures the portable
  producer's distance from the best hand-written producer.
- **Sizing.** Task 6.2 probe, with the same composed-cost method as task 4.3. Hard stop 20
  vCPU-h. Reduction order: 2 shocked episodes per world at 42, then the 12-producer grid alone,
  with the 42 part after the same profile-and-optimize task as PoC-A. A-priori cost: arm A (and
  T when it is not reused) at 12 producers ≈ 12 shocked world-episodes × (0.07–0.4) vCPU-h
  ≈ 1–5 vCPU-h. The 42 part is likely above the stop, as for PoC-A.

### D7. PoC-B, deployed-path cost (EXP-0051)

- **Path.** The portable stack under `joint_stack_env`, calm `sparse_wide` family worlds.
- **Per-decision wall time per stage**, from `perf_counter` wrappers in the driver
  (`eval/deployed_cost.py`):
  1. observation build;
  2. policy forward;
  3. decode / allocate;
  4. domain cycle (`complete_cycle`);
  5. harness remainder.
- **Axes.**
  - P ∈ {12, 24, 42} at fanout E = 2;
  - E ∈ {2, 4, 7, 14} at P = 42 (14 = the whole layer);
  - no point above 42. `eval/deployed_cost.py` has a `--allow-above-42` flag so that a larger
    timing point can be run if the user asks for one in answer to the open question. Without
    the user's yes, tasks 8.2 and 9.1 do not use it. Such a point would be non-gating.
- **Protocol.** Horizon 20, 3 world seeds per cell, BLAS pinned to 1 thread, one process per
  core with at least 2 cores left idle, median of 3 repeats.
- **Output.** The log-log OLS slope per stage and axis. A slope > 1.1 on the E axis is flagged
  (D2 threshold; a warning, not proof).
- **Sizing.** DIAGNOSTICS 17 puts a sparse-120 h = 12 episode at 3.4 s after task 8.4, so a
  42-producer horizon-20 episode is a few seconds, and 7 cells × 3 seeds × 3 repeats stay well
  under 1 vCPU-h. Hard stop 2.
- **Not here.** The D2 refactor stays a separate change. PoC-B only measures.

### D8. Compute budget

| EXP | Probe | Profile | Expected | Hard stop | Source |
|---|---|---|---|---|---|
| 0048 | 2.1 | cpu16 | 2.5–4 vCPU-h | 8 | EXP-0047 rate × 8,832 episodes × tracer overhead |
| 0049 | 4.3 | cpu16 (cpu32 if > 4 h wall) | 12-producer part ≈ 6–30 (a priori, D5); 42 part unknown until the probe and likely above the stop | 40 | EXP-0042 per-stream cost × D5 arithmetic; replaced by the task 4.3 measurement |
| 0050 | 6.2 | cpu16 | 12-producer part ≈ 1–5 (a priori, D6); 42 part as for 0049 | 20 | the same per-stream cost |
| 0051 | 8.2 | cpu8 | ≤ 0.5 | 2 | DIAGNOSTICS 17 timings |

Expected spend for the parts with a grounded estimate (EXP-0048, EXP-0051, and the 12-producer
parts of EXP-0049/0050) is about 10–40 vCPU-h. The 42-producer teacher parts are unknown until
their probes. Their likely outcome is a stop, then a profile-and-optimize code task before they
launch, each under its own stop. The hard ceiling for the whole change is 70 vCPU-h. Every run
uses a fresh pod (`--create-pod`) and `--verify-source-commit`. A run whose probe extrapolates
above its stop does not launch.

### D9. Records

Each experiment task writes, in its own task:
- its `RESULT.md`, plain English first;
- the `experiments/README.md` row;
- its paragraph in a new HISTORY §13, "The development audit (M3)".

Task 4.1 writes ADR-0056 (D4). EXP-0048 also adds DIAGNOSTICS entry 19 (random routing: problem, the six checks, outcome
branches). New domain terms go to the CLAUDE.md Domain Language: "Unresolved label",
"Clairvoyant bound", "Closure (role attribution)". The probes write `sizing.md` and
`routing_trace.md` in this change directory, as M2's `sizing.md` did. Task 10.1 writes the M3
exit summary.

## Risks / Trade-offs

- [The diagnosis finds several partial causes and none recovers the edge alone] → EXP-0048
  reports each explanation's share per cell. The recorded cause may be a combination. PoC-A
  proceeds once it is recorded.
- [The teacher's `blend_0.5` floor sits below a bar that `random` mostly sets, so the teacher
  needs large bumps to win] → PoC-A reports the floor's ratio and the unresolved fraction, so a
  shortfall can be attributed to the floor or to the search. The floor is frozen
  (EXP-0042); changing it would be a new strategy that needs a diagnosed cause.
- [Posterior acceptance is low at 42 producers (many outages unobserved because nothing was
  routed to them)] → the unresolved fallback keeps labels honest. The rate is reported per size.
  Many unresolved decisions read as "fork-limited", not as "no headroom".
- [Teacher cost at 42 exceeds the stop, the a-priori expectation] → the pre-registered reduction
  order: the 12-producer part runs; the 42 part waits for a profile-and-optimize task
  (compute-budget rule). M3's exit needs the 42 part, so the exit can come later than the
  12-producer readings.
- [Tracing perturbs the episode] → additive pure data. Goldens and a with/without-trace equality
  test.
- [A subset of shocked episodes gives a noisy shocked mean] → 3 episodes × 4 worlds per size is
  what the budget allows. Per-world spreads are reported, and the reading is diagnostic, not
  acceptance.

## Migration Plan

Nothing is deployed. Frozen v1/v2 stay the gate defaults. A roster v3 exists only after the D3
human gate, a re-screen and a second approval.

## Open Questions

- **PoC-B scale, for the user.** D4 says "up to 1200 producers". The 2026-10-27 decisions are
  12 → 42 as the target, 120 dropped until agents beat the baselines at 42, and no budget for
  runs of 1000+ vCPU-h. This change measures 12/24/42 only. Two questions: should D4 be amended
  to "up to 42 (larger when scale is reopened)", and does the user want one larger, non-gating
  timing point (for example P = 84, which the driver's `--allow-above-42` flag supports)? The
  default is no. A yes adds the point to task 8.2's probe and to 9.1. The rest of the plan does
  not depend on the answer.
- **Roster v3, after EXP-0048.** Whether a found rule defect becomes roster v3 is the user's
  decision at the D3 gate. It cannot be answered before the diagnosis exists.
