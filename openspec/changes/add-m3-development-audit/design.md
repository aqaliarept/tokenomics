# Design

Motivation: `proposal.md`. Requirements: `specs/development-audit/spec.md`,
`specs/world-space/spec.md`, `specs/stochastic-environment/spec.md`,
`specs/gate-harness/spec.md`. Roadmap: `establish-project-baseline` design D1 (M3 row) and D4.

Revised 2026-10-28 after EXP-0048–0050, on the user's decision (option a): the rest of M3 replaces
"PoC-A teacher on the v2 gate worlds" with a multi-dimension headroom atlas (D10–D12); PoC-A, PoC-C
and PoC-B are re-pointed (D5–D7); D2 and D3 are the record of the done diagnosis and stay as run.

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
  The teacher needs nothing from `archive-1`.
- **EXP-0033 bound code.** `eval/ceiling_bound.py` (analytic achievable-units bounds B0 demand,
  B1 direct recipe capacity, B2 transitive recipe capacity; optimistic: capacity aggregated over
  producers and pinned at its maximum, free upgrade slack, TTL, inventory, funding and routing
  ignored) and `tests/test_ceiling_bound.py` were removed by cleanup commit 18c717d. They exist at
  its parent b96ecc0, an ancestor of branch `archive-1`; the `archive-1` head 7aeef1f no longer has
  them. They import the removed `eval/curriculum.py` (archive replay only), and `episode_demand`
  mirrors the pre-M2 `_build_consumers` (no explicit schedules, no drift). Its capacity term adds
  free, instant upgrades (`upgrade.increases` × horizon² per producer) to base capacity × horizon.
  On family worlds that slack is 33–74× the base capacity per capacity type (world seed 0 of
  sparse 12 f2, sparse 42 f7, dense 12, horizon 100; the review of this revision measured 37–127×
  on EXP-0050's worlds), so B2's capacity term essentially never binds and **B2 equals the
  scheduled demand**: EXP-0033's "98.3% of the bound" was 98.3% of exact demand. B2 is sound but
  cannot see capacity limits; D12 therefore adds a base-capacity ceiling.
- **The easy corner, measured.** Every gate family puts demand at about 6% of initial top-layer
  throughput (`families.family_demand_ratio`, shocked: sparse_wide-12 f2 0.056, sparse_wide-42 f2
  0.057, f14 0.066, dense-12 0.063, dense-42 0.065). Consumer quantities are 1–2 or 1–3. Registry
  v2 has fanout 2 everywhere. Demand is a 7-entry cyclic schedule (calm) or B9 draws with drift
  j = 0.34 (shocked). EXP-0048–0050: units lost to bids that did not fit are ≤ 0.15 per episode;
  the losses are reservations expiring while they wait for inputs; blind whole-order rotation
  (`round_robin`) sets the bar; routing differences grow sharply with fanout (EXP-0050); 12-f4
  sparse and dense-12 are saturated (fill 0.97–0.99).
- **Generator levers that exist.** `families.build_family_config` takes family, size, fanout,
  `capacity_scale` (multiplies capacity, restoration and upgrade amounts after the draw, neutral
  1.0 byte-identical) and a shock profile. Consumer count (3·P/12), quantity range (1..2–3),
  the strong/marginal capacity mix (strong with probability 0.6, 80–140 vs 8–20), degradation
  (3–7%) and costs (20–50) are fixed draws.
- **Per-episode cost** (local, horizon 100, `--workers 1`, sizing.md §EXP-0048–§EXP-0050): a
  router episode 0.83 s (sparse 12 f2), 1.16 (12 f4), 1.27 (dense 12), 4.5 (sparse 42 f2), 6.7
  (f4), 9.8 (f7), 18.5 (f14), 7.1 (dense 42); the trained orderer about 2× a router, the full
  portable stack 2.4–3.1×; pod vCPU-s = 0.646 × local s. Calm (cyclic) demand is fixed per world,
  so one calm episode per world suffices for deterministic arms.
- **Teacher cost.** EXP-0042 spent 267–758 s per episode at depth 2 and 532–1461 s at depth 3,
  on 6–9-producer worlds with 16 workers. Cost on the 12/42 gate worlds is unknown, so a probe
  is required.
- **Baseline episode cost.** EXP-0047: 12,800 roster episodes in about 3.7 vCPU-h of execution,
  ≈ 1.04 vCPU-s per horizon-100 episode, averaged over 12 and 42 producers.
- **Deployed-path refactor not done.** The D2 refactor (eligible-only rows, indexed per-producer
  telemetry, O(E) remainder) has not been done. Only the observation-cost fix of M1 task 8.4
  landed (DIAGNOSTICS 17). PoC-B measures the path as it is.
- **Scale and budget.** The acceptance target is 12 → 42 producers, 120 is dropped for now, and
  there is no budget for runs of 1000+ vCPU-h (2026-10-27). M3's pod budget is 40 vCPU-h; EXP-0048,
  EXP-0049 and EXP-0050 billed ≈ 10.8 (4.4 + 2.5 + 3.9). PoC-B is capped at 42 producers
  (2026-10-28).

## Goals / Non-Goals

**Goals:**
- A diagnosed, recorded cause for `random` setting the hindsight baseline (done: EXP-0048–0050).
- A **headroom atlas**: a world space with many axes, a cheap per-world headroom measure, and a
  mined, recorded set of worlds where a better policy than the best blind or hand-written one is
  reachable, with a train / validation / held-out split. This is the input the self-evolution
  loop needs (worlds that can teach).
- PoC-A, PoC-B and PoC-C recorded as EXP results, with honest teacher information sets, the
  expensive teacher only on a mined sample.
- An M4 input: where the headroom is, which role trains first, what the deployed path costs.

**Non-Goals:**
- No training, no M4 arm selection beyond "which role first", and no deployed search.
- No change to frozen v1/v2 files, to roster members' behaviour, to existing worlds or to the
  acceptance margins. New gate world families or a roster v3 are user decisions (D3, D12); this
  change only produces their evidence.
- No producer search teacher: PoC-C uses the best hand-written producer as the producer-role
  reference (D6).
- No economy above 42 producers anywhere: atlas, probes, PoC-A/C or PoC-B (user decision for
  PoC-B, 2026-10-28; compute-budget rule for the rest).
- No 42-producer teacher run in this budget (D5, D8).
- Demand patterns "basket / co-ordered products" and "heavy-tailed buyers", and the M6 shock
  kinds (supply-link cut, demand surge, fund shock), are out of scope.

## Decisions

### D1. Order of work and gates

```
routing diagnosis (1-3, EXP-0048) --> informed routing + trained trace (3.2-3.3, EXP-0049)
        --> routing versus producers per product (3.4-3.5, EXP-0050)          [done]
        --> atlas instruments (4.1-4.6: docstring fix + deciding-check arms, bound,
            demand patterns, world space, atlas driver, calibration + timing probe)
        --> atlas screen (5.1, EXP-0051) --> atlas confirm (5.2, EXP-0052; atlas v1 frozen)
        --> robust floor + dead-arm rule, offline reclassification (5.3; atlas v2)
        --> PoC-A code (6.1-6.3) --> PoC-A (7.1, EXP-0053)
        --> PoC-C code (8.1-8.2) --> PoC-C (9.1, EXP-0054)
PoC-B (10.1-10.2 code, 11.1 EXP-0055): depends only on its own code
M3 exit record (12.1)
[user, any time: roster v3 (D3); atlas families into a gate registry (D12)]
```

The atlas comes before PoC-A because PoC-A now runs on mined worlds. Its bar is measured per
world from blind and hand-written arms, `round_robin` included, so it does not wait for the user's
roster v3 decision (D3): that decision changes the gate's bar, not the atlas's. PoC-B shares
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

**Informed spreading routers and trained-agent trace (EXP-0049, tasks 3.2–3.3).** Inserted
after EXP-0048 on the user's request. EXP-0048 found that `random` wins because it varies the
receiving producer over time and almost never splits an order, and that `round_robin` does both
deterministically. `round_robin` is blind, though. Two questions follow:
1. Can a rule that reads producer data spread whole orders better than `round_robin` /
   `random_whole`?
2. Did the trained portable orderer learn random's trick? EXP-0006B credited "stochasticity".

*Spare capacity.* `spare_units(p, x)` = min over the capacity types t of x's recipe of
floor((C[p,t] − reserved[p,t]) / R[x,t]), i.e. `domain.metrics.accept_qty_now`. It is the
`accept_qty_now` field of the ordering agent's own product-stream view (`ProducerProductStats`,
orderer obs v3/v4), so a joint_stack orderer observes it. It holds no hidden shock state: an
outage does not lower it, and a capacity hit lowers it only as a visible consequence. It does not
include this cycle's bids from other orderers, which no orderer observation shows (below). The
legacy orderer obs carries prices only (ADR-0029), so the three routers are valid under
joint_stack only, like the trained orderer.

| Router | Kind | Definition | Blind counterpart |
|---|---|---|---|
| `two_choice_spare` | stochastic | Per order, two distinct eligible producers drawn uniformly; the whole order goes to the one with more spare (the first drawn on a tie) | `random_whole` |
| `spare_weighted_whole` | stochastic | Whole order to one eligible producer drawn with probability ∝ spare (uniform if all are 0) | `random_whole` |
| `spare_rotating_whole` | deterministic | `round_robin`'s rotation and per-orderer offset; the first producer in rotation order whose spare takes the whole order gets it, else the one with the most spare | `round_robin` |

Draws use the env action-space RNG (per orderer and call, seeded with the gate's replicate seed),
R = 5, as for the other stochastic routers. The risk is herding: every orderer reads the same
spare values at the same time, so informed choice can send many orderers to the same producer,
the mechanism that sank `proportional_whole`. The herding metric below measures it.

`sequential_fill` is not built. The ordering agents do act one after another within a cycle (AEC
order: every producer, then every ordering agent, each placing its bids at once), but no
orderer observation shows the bids placed earlier in the cycle. The legacy obs is prices; the
stream view's rows read capacities, reservations, last cycle's requested load and telemetry EMAs
that fold only in `complete_cycle` (`StreamViewCache`: routing only appends to `p.bids`, which no
row reads). A later orderer therefore cannot legitimately see earlier sub-orders.
`proportional_load_aware` reads `p.bids` white-box as a diagnostic of (a) only, and it did not
help in EXP-0048.

*Trained subjects.* The portable stack is EXP-0035 `producer_candidate.pt` + EXP-0012
`orderer_scorer.pt`. It runs as `entity_planner/trained_orderer` (the routers' producer, so its
routing compares with theirs) and as `portable` (its own producer). The orderer is a scorer
distilled from log(makeable units now) (`train/orderer_scorer.py` `teacher_logits`:
`makeable_qty_now`, capacity and inputs in stock; corrected after EXP-0050's review, an earlier
version called this `proportional`'s target, but `proportional` weighs capacity only). Its policy emits deterministic logits and the decode does not sample, so both
subjects are deterministic and run once (R = 1). They are not EXP-0006B's sampled policy.

*Tracer additions.* Receiver variation: a stream is one ordering agent's orders of one product;
each order routed in a cycle is one decision; `receiver_change_share` is the share of
consecutive decisions of a stream, in (cycle, order id) order with re-bids included, whose sets
of receiving producers differ (consumer and producer orders separately). Herding: per product and
cycle with at least two routed orders and two declaring producers, the Herfindahl of
order-weighted producer shares (each order counts 1, spread over its receivers by quantity),
averaged over those groups (`herding_hhi`). 1 means every order went to one producer, 1/E means
the orders were spread evenly over E producers.

*Grid (`routing_audit --plan exp0049`).* All 128 world-episodes. The six roster v2 members (for
the edge), the references `round_robin`, `even_split_rotating`, `proportional_whole` and
`random_whole`, the three informed routers and the two trained subjects: 31 runs per
world-episode, 3,968 horizon-100 episodes. The references are rerun, not joined from EXP-0048's
`episodes.csv`. Joined rows would lack the receiver-variation and herding columns that both
questions read, and the rerun costs about 1.2 vCPU-h. The overlapping rows then serve as the
reproduction check: every reference row must equal EXP-0048's units for the same world-episode
and replicate. That checks all of them, not 2 cells. The trained path has a reference too:
EXP-0046 ran the same portable stack (its `portable_producer` / `portable_orderer` pins are the
sha256 of the two checkpoints) through the gate job path on the 80 shocked development
world-episodes, so every `portable` row there must equal EXP-0046's `candidate` units. That
checks the checkpoint wiring, the contract and tracer neutrality on the trained path.
`entity_planner/trained_orderer` has no prior reference.

*Reading (pre-registered).* The D2 recovered fraction and the 0.8 / 2-of-3-cells rule apply to
the new arms. An informed router *beats its blind counterpart* when its equal-weight world mean
units exceed the counterpart's in at least 2 of the 3 random-dominated cells.
- An informed router beats its counterpart: producer data helps beyond rotation. It is a roster
  v3 candidate beside `round_robin` (D3), and M4's residual context needs spare capacity.
- None does: blind spreading of whole orders is enough at these sizes, and spare capacity is
  not the lever. If the informed routers herd more than their counterparts (`herding_hhi`), the
  shared signal is the reason.
- The trained orderer's split degree, receiver variation and herding near `proportional`'s and
  far from `random`'s: it learned `proportional`'s mechanics, not random's trick, and its
  distillation target is the thing to change for M4. Near `random`'s / `round_robin`'s: it
  learned to spread whole orders.

**Routing versus producers per product (EXP-0050, tasks 3.4–3.5).** Inserted after EXP-0049 on
the user's request ("we have only 2 factories to choose from — increase the number"). Registry
v2 is sparse_wide with fanout 2 everywhere (`families.DEFAULT_SPARSE_FANOUT`): every product
has exactly two eligible producers, so a routing choice is nearly binary, `two_choice_spare`
reduces to an argmax and EXP-0048/0049's findings may not hold when the choice widens.

*Grid (`routing_audit --plan exp0050`, `fanout_worlds`).* Family worlds built with
`families.build_family_config(fanout=...)`, never added to the gate registry, roster or
parameter files: sparse_wide at 12 producers with fanout 2 and 4 (4 = the whole layer) and at
42 with 2, 4, 7 and 14 (14 = the whole layer); dense at 12 and 42 (4 products per layer, each
declared by the whole layer) as a reference. EXP-0044 found dense-12 saturated (no headroom),
so every cell reports consumer fill and a saturated cell is read as such. Both regimes, shocked
with `shocks-i`. Per cell × regime 4 worlds; 1 calm episode per world, because calm demand is
fixed per world and a deterministic arm then replays the same episode for every episode seed
(EXP-0049: 176 of 240 calm subject-world pairs identical, every differing one stochastic), and
2 shocked episodes per world. Seeds are sha256 of (plan, family, size, fanout, world index), so a
cell's calm and shocked worlds share their topology, and episode seeds hash the world id.
`spec.json` pins every world with its measured declarers per product and its config sha256.
96 world-episodes.

*Arms.* As EXP-0049: the six roster v2 members (`random` R = 5, the deterministic five set the
best deterministic member), `round_robin`, `random_whole` (R = 5), `proportional_whole`,
`even_split_rotating`, the three informed routers (stochastic R = 5) and the two trained
subjects. 31 runs per world-episode, 2,976 episodes, tracer on.

*Outputs and reading.* `routing_metrics.csv` per (family, size, fanout, regime) with fill and
every tracer scalar, and `fanout_summary.csv` with each arm's gain over the best deterministic
member, `random`, `round_robin` and `random_whole`, and each informed arm's gain over its blind
counterpart. No `edge.csv`: the D2 edge rule is defined on the v2 development cells. Reported,
with no new pre-registered threshold: whether `round_robin`'s gain over `random` and over the
best deterministic member holds as fanout grows, and in which cells an informed arm beats its
blind counterpart (same "more units in the cell" reading as EXP-0049, with the paired
world-clustered spread). New worlds, so no row reproduces an earlier run; the fanout-2
sparse_wide cells link the run to EXP-0049.

*Power and limitations.*
- Detectable differences at 4 worlds per cell × regime (80% power, two-sided 5%, paired
  world-clustered standard error with 3 degrees of freedom, so the minimum detectable difference
  is 4.16 × SE; SE from EXP-0049's episodes subsampled to this design: 4 worlds, 1 calm / 2
  shocked episodes per world), in units per episode for 12-calm / 12-shocked / 42-calm /
  42-shocked: `spare_rotating_whole` − `round_robin` 14 / 3 / 37 / 25 (0.6–2.6% of units);
  `two_choice_spare` − `random_whole` 67 / 13 / 100 / 51; `spare_weighted_whole` −
  `random_whole` 19 / 7 / 38 / 29; `round_robin` − `random` 45 / 18 / 38 / 30;
  `round_robin` − best deterministic member 82 / 42 / 108 / 71; trained orderer −
  `round_robin` 39 / 21 / 62 / 32. Calm cells are underpowered for effects of EXP-0049's size:
  its `round_robin` − `random` gains on calm worlds (+26 at 12, +31 at 42) are below the calm
  detectable differences (45, 38), so a calm fanout cell may not even reproduce that edge; the
  shocked cells resolve it. Every "more units" reading is reported with its paired
  world-clustered SE; the spread among fanout worlds may differ from v2's.
- Cross-fanout comparisons are not paired. The world seed hashes the fanout, and
  `build_family_config` uses `strong_every = min(fanout, 4)`, which changes the forced-strong
  producers and shifts the random stream. The share of strong producers (capacity ≥ 80) in the
  grid's worlds differs by cell: sparse_wide fanout 2 0.74–1.00, 4 0.67–0.92, 7 0.64–0.76, 14
  0.60–0.83, dense-12 0.42–0.67. Only within-cell arm contrasts are paired; a trend of any gain
  with fanout mixes the fanout with the capacity mix and the topology draws, and EXP-0050's
  RESULT reads fanout trends with that caveat.
- One calm episode per world: sound for deterministic arms (EXP-0049: no calm world's demand
  digest differs across its episodes, no deterministic arm's units differ), but calm cells rest
  on 4 world draws only, and stochastic arms there get their 5 replicates on one demand
  schedule.
- The fanout worlds are unscreened: no zero-baseline replacement and no headroom screen as for
  registry v2; consumer fill is the only saturation read.

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

Status (2026-10-28): EXP-0048 named a rule defect and the supervisor held the gate; EXP-0049 and
EXP-0050 followed on the user's requests, and the user then chose the atlas (option a). The roster
v3 question stays open and is not a task here. The atlas does not depend on it: its per-world bar
is measured from blind and hand-written arms, `round_robin` included (D12).

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
  - Below the floor the cycle's decisions are "unresolved" and act as the teacher's floor
    policy (`round_robin` in PoC-A, D5; it was `blend_0.5` in EXP-0042).
  - The acceptance rate is logged per cycle.
- **Stripping.** A fork's domain state loses the true pending shock events and outage end
  cycles before the sampled ones are injected (spec "Hidden fields stripped").
- **Drift continuity.** `env/future_fork.py` gains a drift-continuous resampler. From the current
  cycle's orders it applies the drift process (at most floor(j·N) changes per transition) on a
  fork-seeded stream. B9 worlds without drift keep the independent tail; calm cyclic schedules
  are not resampled.
- **Demand-pattern continuity** (D11). A pattern's parameters are part of the world config, i.e.
  the known generator. Patterns whose future draws do not depend on hidden past draws are forked
  by resampling the future from the generator on a fork-seeded stream: bursts (each future order
  is a burst with probability p, independently) and seasonal waves (the envelope is a function of
  the cycle and the config; only the stochastic rounding is redrawn). The moving hit and the
  regime switch carry hidden state (where the hit is, when the next switch comes) that this rule
  does not define a posterior for. Worlds with those two patterns are therefore not in the
  teacher's sample (D5); extending the rule to them is an ADR-0056 amendment.
- **Deterministic futures.** On worlds with cyclic demand and no shocks nothing is hidden, so
  every scenario is identical and the teacher uses one scenario. With zero variance, the confidence rule accepts a strictly
  positive gain. Pre-registered.
- **ADR.** The information-set rule is a design decision, so task 6.1 writes ADR-0056 "Teacher
  information set and posterior forks". It covers the public sources, the two-sided acceptance,
  the floor, stripping, drift and demand-pattern continuity and the deterministic-future rule,
  and cites D4 of `establish-project-baseline`, ADR-0055 and ADR-0057.
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

### D5. PoC-A, the teacher on a mined sample (EXP-0053)

Re-pointed 2026-10-28 (user decision, option a): the teacher no longer runs on the registry v2
development worlds but on a sample of the headroom atlas (D12). Question: on worlds where the
bound leaves room, does a lookahead search reach more than the best cheap arm?

- **Search.** The EXP-0042 bounded-rollout search (`eval/bounded_rollout_orderer.py`): per
  stream decision, candidates are evaluated by depth-2 rollouts over scenarios with confidence
  checkpoints at 4/8/16 scenarios, and a candidate replaces the floor's action only when its gain
  is confidently positive. Scenarios come from the D4 fork rule (posterior forks; one scenario on
  deterministic-future worlds). Depth 3 is not run: EXP-0042 promoted depth 2, and depth 3 costs
  about twice as much.
- **Floor and candidates, re-pointed for a diagnosed cause.** EXP-0042's floor was `blend_0.5`
  with a +1-logit bump per eligible producer as candidates. EXP-0048–0050 diagnosed that
  `blend_0.5` splits orders and keeps its receivers, and loses 100–400 units per episode to
  `round_robin` at 42 producers (EXP-0050: best roster rule 1,335–1,662 against `round_robin`'s
  1,686–1,882). A search that starts that far below the bar would most likely read
  "floor-limited", which the previous version of this plan already expected on v2. So:
  - floor = `round_robin` (whole order, rotating receiver), which is also the unresolved fallback;
  - candidates per stream decision = the whole order to each eligible producer (E candidates, the
    floor's receiver among them; a +1-logit bump does nothing on a whole-order floor), plus the
    trained orderer's action (EXP-0012 `orderer_scorer.pt`; EXP-0050: input-aware splitting is
    the strongest cheap arm at fanout ≥ 4). At most E + 1 candidates.
  The frozen EXP-0042 configuration is not rerun; its result stands for its worlds.
- **Arms**, all with the `entity_planner` producer and the joint-stack contract: `floor`
  (`round_robin`), `d2` (the teacher) and `ub` (clairvoyant: the same search over the true future,
  one scenario; run only on worlds with shocks or stochastic demand, since on deterministic-future
  worlds it equals `d2` by construction, which a test asserts on a fixture). The world's blind arms
  and reach arms (D12) are rerun in the same driver on the same world-episodes, so the bar and the
  digests are checked in one run.
- **Sample** (pre-registered, from atlas v2's **train** split (D12 "Reclassification"), 12
  producers only):
  - 6 hard and 6 open worlds (D12 classes), in descending order of their confirm-stage gain (hard)
    or room (open); if a class has fewer, the other fills (12 worlds in total);
  - half of each class with shocks on, when available: of a class's n worlds, the best n // 2
    shocked and the best n − n // 2 unshocked, a shortfall on either side filled by the best
    remaining worlds of the class;
  - demand pattern cyclic, iid, drift, burst or seasonal only (D4: the moving hit and the regime
    switch carry hidden state the fork rule does not cover; recorded as a limitation);
  - 1 episode per world: the confirm stage's first episode seed.
- **Bar and reading** (spec "Teacher audit reading"). Per world: `d2` − bar and `d2` / bar, with
  the bar = the best blind arm (D12) on the same world-episode; `d2` against the reach arms and the
  ceiling C and B2; `ub` − bar; the unresolved fraction and acceptance rates; deaths. Pre-registered
  readings, per class:
  - open worlds: `d2` beats the bar by more than the world's noise floor on at least half of them →
    search finds headroom the cheap arms miss there; such worlds are teacher-label candidates for M4
    (the self-evolution loop has a teacher). `ub` beats the bar but `d2` does not → fork-limited
    (read with the unresolved rate). Neither beats it → the room under the ceiling is not
    reachable at depth 2 even with the future known; recorded as "ceiling-limited" (the ceiling
    ignores TTL, routing, funding and degradation);
  - hard worlds: `d2` ≥ the best reach arm → the teacher labels beyond the trained orderer;
    below it → the search's candidate set or depth is the limit, recorded with evidence.
  The gate margins (1.05 calm, 1.10 shocked) are reported as reference ratios only. No acceptance
  verdict, no margin change (spec "A failed probe ends in a diagnosis").
- **42 producers: not run.** At 42 the a-priori cost is 1–2 orders of magnitude above 12 (more
  streams per cycle, dearer rollouts, DIAGNOSTICS 17: about 3–30 vCPU-h per shocked d2 episode),
  far above this budget. A 42-producer teacher read needs a probe, a profile-and-optimize code task
  and the user's approval of the spend; the supervisor inserts them only then.
- **Driver.** A new `eval/teacher_audit.py`. `eval/bounded_rollout_orderer.py` gets a pluggable
  scenario source (posterior fork / deterministic / clairvoyant), a pluggable floor and candidate
  set (the EXP-0042 defaults unchanged), a baseline producer option, and atlas-world input from an
  atlas manifest besides its archive tasks. Row format, digests and contract checks are reused from
  `eval/gate.py` / `eval/atlas.py`.
- **Sizing.** Task 6.3 probes by composed cost: streams searched per cycle, seconds per searched
  stream (timed per stream, so a cell yields numbers even if no cycle finishes in the wall cap),
  candidates per stream, the posterior sampler's seconds at a horizon-100 history length (play
  `round_robin` to cycle 90 on a shocked atlas world first), and one `round_robin` horizon-100
  episode against the 0.646 local-to-pod factor. A priori (the arithmetic, not a measurement):
  EXP-0042 spent ≈ 0.5 s per searched stream at depth 2 with about 3 candidates on 6–9-producer
  worlds; D5's earlier arithmetic put a 12-producer d2 world-episode at 0.07–0.4 vCPU-h; with up
  to E + 1 = 5 candidates at fanout 4, ×1.7 → 0.12–0.7 vCPU-h. 12 world-episodes → 1.4–8.4; `ub`
  uses one scenario, ≈ 0.1–0.5 more. **Hard stop 8 vCPU-h.** Reduction order if the extrapolation
  (× 2 safety factor) exceeds it: 1. drop `ub`; 2. 8 world-episodes (4 hard + 4 open); 3. nothing
  launches and the implementer reports to the supervisor.

### D6. PoC-C, role attribution on mined shocked worlds (EXP-0054)

- **Re-pointed** from v2's shocked subset to the shocked worlds of PoC-A's sample (about 6
  world-episodes at 12 producers). Reason: attribution needs a gap G between the portable stack
  and the teacher reference. On v2's easy corner the portable orderer already beats every roster
  member (EXP-0049), so G would be small or ≤ 0 and the attribution undefined; on mined worlds a
  gap exists by selection.
- **Arms**:
  - S: the portable gate subject, EXP-0035 `producer_candidate.pt` + EXP-0012
    `orderer_scorer.pt`;
  - A: portable producer + the PoC-A teacher orderer (`d2`);
  - B: `entity_planner` + portable orderer;
  - T: `entity_planner` + the teacher (`d2`).
- **T rows.** T is reused from EXP-0053 only when the run commit's source fingerprint over the
  teacher path (`eval/bounded_rollout_orderer.py`, `eval/teacher_audit.py`,
  `env/shock_posterior.py`, `env/future_fork.py`, `env/demand_patterns.py`, `domain/`) equals
  EXP-0053's. Otherwise T is rerun.
- **Arithmetic, per size.**
  - Gap: G = Σ_w (T_w − S_w), over world mean units.
  - Orderer closure: c_O = Σ_w (A_w − S_w) / G.
  - Producer closure: c_P = Σ_w (B_w − S_w) / G.
  - Interaction: T − A − B + S, reported.
  - Pre-registered band: |c_O − c_P| < 0.15 is a tie and goes to the user.
  - G ≤ 0 means the attribution is undefined; it is recorded and diagnosed.
- **Limitation, recorded.** The producer-role reference is a hand-written producer, not a
  teacher, because no producer search teacher exists. c_P therefore measures the portable
  producer's distance from the best hand-written producer. Only 12-producer worlds (D5).
- **Sizing.** Task 8.2 probe, with the same composed-cost method as task 6.3. Only A (and T when
  not reusable) costs search: 6 × 0.12–0.7 ≈ 0.7–4.2 vCPU-h a priori. **Hard stop 4 vCPU-h.**
  Reduction: 4 shocked world-episodes; then nothing launches and the implementer reports.

### D7. PoC-B, deployed-path cost (EXP-0055)

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
  - load: consumers per producer × {1, 2, 4} (the world-space consumer lever, D10) at P = 42,
    E = 4. Added because the atlas varies the number of consumers, and with it the number of
    ordering streams per cycle, which the two other axes hold fixed;
  - no point above 42 producers (user decision, 2026-10-28); the driver refuses one and has no
    flag to allow it.
- **Protocol.** Horizon 20, 3 world seeds per cell, BLAS pinned to 1 thread, one process per
  core with at least 2 cores left idle, median of 3 repeats.
- **Output.** The log-log OLS slope per stage and axis. A slope > 1.1 on the E axis or the load
  axis is flagged (D2 threshold; a warning, not proof).
- **Sizing.** DIAGNOSTICS 17 puts a sparse-120 h = 12 episode at 3.4 s after task 8.4, so a
  42-producer horizon-20 episode is a few seconds; 8 cells × 3 seeds × 3 repeats stay well under
  1 vCPU-h. **Hard stop 1 vCPU-h.**
- **Not here.** The D2 refactor stays a separate change. PoC-B only measures.

### D8. Compute budget

M3's pod budget is **40 vCPU-h**; EXP-0048–0050 billed ≈ **10.8**, so ≈ **29.2** remain. Every
run uses a fresh `cpu16` pod (`--create-pod`; `cpu8` for PoC-B) and `--verify-source-commit`, is
sized by a local probe that extrapolates with the measured local-to-pod factor (0.646) and a × 2
safety factor where the D-section says so, and has a hard stop. A run whose probe extrapolates
above its stop does not launch until its pre-registered reduction brings it under; failing that,
the implementer reports to the supervisor.

| EXP | What | Probe | Expected billed vCPU-h | Hard stop | Source of the estimate |
|---|---|---|---|---|---|
| 0048 | random-routing diagnosis | 2.1 | done: 4.4 | 8 | billed |
| 0049 | informed routing + trained trace | 3.2 | done: 2.5 | 6 | billed |
| 0050 | routing vs fanout | 3.4 | done: 3.9 | 6 | billed |
| 0051 | atlas screen, 1,024 + 128 worlds (512 + 128 after reduction step 1, task 4.6: 8.26 pod vCPU-h) | 4.6 | 6–12 (central ≈ 8) | 10 | per-cell router times (Context) × 13 router-equivalents × load factor 1–2 (D12) × 0.646 |
| 0052 | atlas confirm, 64 + 32 worlds | 4.6 | 2.6–5.3 (central ≈ 3.5) | 6 | the same, 18 router-equivalents, 2 episodes |
| 0053 | PoC-A, 12 world-episodes at 12 | 6.3 | 1.5–9 (central ≈ 4, a priori) | 8 | EXP-0042 per-stream cost × D5 arithmetic |
| 0054 | PoC-C, 6 world-episodes at 12 | 8.2 | 0.7–4.2 (central ≈ 2, a priori) | 4 | the same |
| 0055 | PoC-B, 8 cells | 10.2 | ≤ 0.5 | 1 | DIAGNOSTICS 17 timings |

- **Planned remainder:** expected ≈ 11–31 vCPU-h, central ≈ 18; the hard stops sum to **29**,
  inside the remaining 29.2. M3 total: central ≈ 29 of 40, worst case (every stop hit) 39.8.
- **Needs the user's approval (not planned):** any run above its stop after its reduction order;
  a 42-producer teacher read for PoC-A or PoC-C (a priori 3–30 vCPU-h per shocked d2 episode, plus
  a profile-and-optimize task); a second batch of 42-producer atlas worlds; the roster v3
  re-screen (≈ 4 vCPU-h, D3); a gate re-screen with atlas families (D12). Nothing above 42
  producers and no run near 1000 vCPU-h is planned.
- The remote tasks' `Resources:` lines carry these numbers; the probes replace the a-priori ones.

### D9. Records

Each experiment task writes, in its own task:
- its `RESULT.md`, plain English first;
- the `experiments/README.md` row;
- its paragraph in HISTORY §13, "The development audit (M3)". EXP-0051 also adds a HISTORY §13
  sub-heading "The headroom atlas" opening the new arc, with the reason (the easy corner,
  EXP-0048–0050) and how it differs from the closed mining arc of §7 (many axes, a bound per world,
  headroom against a noise floor, a recorded split).

Task 4.1 records the docstring fix. Task 4.4 writes ADR-0057 (D10–D12). Task 4.5 adds
DIAGNOSTICS entry 20 ("A world set has no headroom": problem, the atlas class check, outcome
branches); EXP-0051/0052 add its "Seen in". EXP-0052 adds the deciding-check result to entry 19.
Task 6.1 writes ADR-0056 (D4). New domain terms go to the CLAUDE.md Domain Language: "World
space" and "Demand pattern" (task 4.3/4.4), "Headroom", "Atlas bar" and "Headroom atlas" (task
4.5), "Unresolved label" (6.1), "Clairvoyant bound" (6.2), "Closure (role attribution)" (8.1).
Probes write `sizing.md` sections in this change directory. Task 12.1 writes the M3 exit summary.

### D10. World space

A world is a point on named axes. The generator (`eval/world_space.py`, on top of
`eval/families.py`) turns a point's **levers** into a configuration and records the world's
**measured coordinates**, which are what coverage and every reading use. Levers are sampled;
coordinates are measured, because several levers move one coordinate (tightness depends on
consumers, order size and capacity) and the realized value depends on the world's draws.

| Axis | Measured coordinate | Levers (initial range; task 4.6 calibrates) | Neutral value (today) |
|---|---|---|---|
| Size | producers P | P ∈ {12, 42}, a separate sample per size | — |
| Choice | eligible producers per product E, family | 12: sparse E ∈ {2, 4}, dense (E = 4); 42: sparse E ∈ {2, 4, 7, 14}, dense (E = 14); equal shares | sparse E = 2 |
| Capacity tightness | ρ = expected consumer demand units per cycle ÷ initial top-layer throughput (`demand_units_per_cycle` / `top_layer_throughput`) | consumer multiplier k_c ∈ [0.5, 8] (consumers = max(1, round(k_c · 3P/12))), with m_q and c below; log-uniform | k_c = 1 (ρ ≈ 0.06) |
| Order size vs capacity | σ = mean over orders of (quantity × capacity units per unit) ÷ the median capacity of the product's declarers | quantity multiplier m_q ∈ [1, 8] (quantity range [m_q, m_q · q_max], rounded half up), capacity scale c ∈ [0.25, 1] (`capacity_scale`; widened to [0.25, 2] by task 4.6 so σ reaches 0.05, sizing.md §EXP-0051, ADR-0057); log-uniform | m_q = 1, c = 1 |
| Producer heterogeneity | coefficient of variation among producers of capacity, of degradation and of restoration/upgrade cost | h_cap, h_deg, h_cost ∈ [0, 2]: each drawn value v becomes med + h · (v − med) (med = the median of the drawn values in the world's layer), rounded half up and clamped to the valid range, after every draw | h = 1 (no transform) |
| Shocks | expected shock events per producer per 100 cycles, mean hit magnitude, mean outage duration | off / on (equal shares); on: `shocks-i` with counts × r, r ∈ [0.5, 3], hit magnitude range centred at m ∈ [0.2, 0.8] (width 0.3), outage duration range [d, d + 3], d ∈ [2, 10] | off, or `shocks-i` (r = 1, m = 0.45, d = 3) |
| Demand pattern | pattern kind and parameters; realized demand concentration (Herfindahl of demanded units over products) and volatility (coefficient of variation of per-cycle demanded units) | kind ∈ {cyclic, iid, drift, burst, hit, seasonal, switch} (equal shares) and its parameters (D11) | cyclic (calm) or drift j = 0.34 (shocked) |

- **Neutral levers are byte-identical.** Every lever at its neutral value skips its transform, so
  the configuration, its fingerprint and every episode statistic equal today's
  `build_family_config` output (spec world-space "Neutral levers reproduce existing worlds").
  Lever transforms run after every existing draw, on values already drawn, so the random stream,
  the topology (recipes, `produces` sets) and every other value do not move when a lever moves.
  The consumer multiplier adds or removes consumers after the existing consumers' draws, from a
  stream derived from the world seed; the extra consumers' pools follow the family's pool rule.
- **Shocks and drift.** A world with shocks on gets a shock profile object built in code from its
  levers (the `shocks-i` window 21–90 and kinds, counts × r, the shifted magnitude and duration
  ranges); `shock_profiles_v1.json` stays frozen. The `shocks-i` profile's drift (j = 0.34) is not
  applied with it: in the atlas, drift is one value of the demand-pattern axis, so shocks and the
  demand process vary independently.
- **Sizes.** 12 and 42 only; the generator refuses any other size in the atlas.
- **Seeds.** World seed = sha256(atlas name, size, index). The sample itself: a seeded Latin
  hypercube per size (numpy only, no new dependency), one column per continuous lever plus one per
  categorical axis mapped to equal-width strata, so categorical levels are balanced. Episode seeds
  hash the world id, as in `routing_audit`'s `fanout_worlds`.
- **Contract.** Larger quantities (m_q, bursts) must stay inside every observation space. Task 4.4
  runs the contract monitor on the lever corners (every lever at both ends, horizon ≤ 12); a corner
  that breaks the contract narrows that lever's range and is recorded in ADR-0057.
- **Record.** Each world's record holds its levers, seed, generator version, measured coordinates,
  measured declarers per product and config sha256, as `routing_audit.world_record` does.

### D11. Demand patterns (env layer)

Patterns are demand, not shocks: they shape what consumers order, are part of the demand schedule,
and are kept in the shock-free twin. They are not the M6 "demand surge" shock (an unannounced,
targeted, time-limited multiplication measured by resilience against the twin), which stays out of
scope. Each pattern is a record in the configuration (data only, like `DemandDrift`; the domain
draws nothing) and is generated by a new `env/demand_patterns.py` at reset:

- **One demand process.** Configuration validation rejects a world with both drift and a
  pattern (task 4.3).
- **Own streams.** Each pattern kind draws from `derived_rng(seed, "pattern-<kind>")`
  (ADR-0055 style). The legacy B9 stream, the drift stream and every shock stream are never read
  by a pattern, and a pattern never moves them. Generation is cycle-major and its windows live in
  the config, so a shorter horizon's schedule is a prefix of a longer one.
- **Scope.** A pattern applies to the world's randomized (B9) consumers and replaces drift on them
  (a world has at most one demand process). Consumers with an explicit cyclic schedule are never
  patterned.
- **Kinds built** (parameters sampled by the world space within these ranges):
  1. **Burst**: each order is independently a burst with probability p ∈ [0.02, 0.10]; a burst
     multiplies its quantity by an integer factor uniform in [5, 10]. The product and base
     quantity are drawn from the B9 distribution on the burst stream itself (as drift does): the
     legacy B9 stream is consumer-major, so drawing the base there would break the prefix
     property across horizons (task 4.3).
  2. **Moving hit**: a seeded permutation of the top-layer products sets a popularity rank, with
     weight ∝ rank^(−α), α ∈ [0.8, 2]; every T_h ∈ [5, 30] cycles the ranking shifts by one, so the
     hit moves to the next product. Each consumer draws its product from its pool with these
     weights; quantity as B9.
  3. **Seasonal**: quantity × s(t) = 1 + A · sin(2π(t + φ)/T), A ∈ [0.3, 0.8], T ∈ [10, 40],
     phase φ drawn from the world seed (fixed per world, so the waves are predictable from
     history); the product and base quantity are drawn from the B9 distribution on the seasonal
     stream (as for bursts, for the prefix property), then the scaled quantity is rounded
     stochastically on the same stream, never below the consumer's `quantity_min`.
  4. **Regime switch**: a seeded half of the top-layer products is favoured (weight 1, the rest
     w_low ∈ [0.05, 0.3]); K ∈ [1, 4] switches at cycles drawn without replacement from the config
     window [11, 90] with gaps ≥ 10; each switch swaps the favoured half and its complement
     (abrupt, persistent until the next switch); quantity as B9.
- **Pools.** Product patterns (hit, switch) need choice inside a pool. In sparse worlds a pool has
  ceil(W/C) products, which is 1–2 (and 1 once consumers outnumber products). A world with a
  product pattern therefore widens each consumer's pool to min(4, W) consecutive top-layer
  products (the family's pool rule, longer window). This is a lever of the pattern, recorded in
  the world record; pools are unchanged otherwise.
- **Tokens.** A consumer's tokens per cycle stay its emission share; a burst or a seasonal peak
  is backed by the same tokens (a lower price per unit), as the M2 spec defines for surges.
- **Unannounced, digest-covered, contract-preserving.** No observation feature exposes a pattern
  kind, parameter, phase or switch; agents see only the orders. The demand digest hashes the
  schedule as built, so it covers every pattern draw; the configuration fingerprint covers the
  pattern record when present and omits it when absent (ADR-0055 decision 5). Spaces depend on
  topology only, so checkpoints load unchanged.
- **Byte-identical without a pattern.** Goldens, fingerprints and every episode statistic of
  every configuration without a pattern are unchanged (spec stochastic-environment "No pattern,
  no change").
- **Deferred.** Basket / co-ordered products and heavy-tailed buyers are not built: both need a
  consumer that places several orders per cycle or per-consumer volume weights, which changes
  the consumer model, not just its draws. A later change can add them as further kinds on their
  own streams.

### D12. Headroom and the atlas (EXP-0051, EXP-0052)

**Arms per world.** All with the `entity_planner` producer, through `routing_audit`'s job path
(tracer and contract monitor on, the gate's replicate seeds).
- **Blind and hand-written arms** (the bar): the five deterministic roster v2 orderers
  (`even_split`, `proportional`, `blend_0.5`, `shock_reactive` under joint_stack, `obs_price`
  under legacy), `random`, `round_robin`, `random_whole`.
- **Reach arms** (better policies that are cheap to run): EXP-0050's deciding check,
  `makeable_split` (each order split in proportion to the eligible producers' `makeable_qty_now`,
  the scorer's distillation target as a router; uniform if all are 0) and
  `makeable_weighted_whole` (the whole order to one producer drawn ∝ `makeable_qty_now`; the same
  expected allocation, never split); `entity_planner/trained_orderer`; and, reported in the confirm
  stage only, `portable`. Both deciding-check routers read `makeable_qty_now` from the orderer's own stream
  view (it is in the encoded observation), so they are joint_stack only and white-box like
  `proportional`. They stay in `DIAGNOSTIC_ROUTERS`, outside the roster.
- **Bound and ceiling**, per world-episode, from the demand schedule the env built for that
  episode seed (so drift and patterns are covered), never policies:
  - B2, the restored analytic bound (`eval/ceiling_bound.py`, upgrade slack on). Sound: invariant
    B2 ≥ every played arm's units (tolerance 0.01); a violation makes the run invalid. On this lever
    range it is a demand cap (Context), so it is not used to judge room;
  - B_base, the same bound with the upgrade slack off: base capacity pinned at each producer's
    initial maximum every cycle (degradation ignored, so optimistic there), no upgrades.
    **Not guaranteed**: a policy that upgrades can exceed it. It is the informative
    "base-capacity ceiling" that sees capacity: at a high-tightness corner (demand far above
    throughput) it falls below demand where B2 does not;
  - **B_lp**, the input-output bound (task 4.7, `ceiling_bound.lp_bound`, user request
    2026-10-29): one linear program over every producer, capacity type and product, a dynamic
    Leontief model relaxed only where that keeps it an upper bound. Per-producer `produces`
    eligibility and capacity (no pooling across non-declarers); one joint product mix (not a
    per-type minimum); a product balance per product, so recipe inputs, restoration and upgrade
    products are paid for and recipe cycles are exact; capacity per producer × type × cycle that
    decays geometrically at the degradation rate (the domain's ceil(d·M/100) ≥ d/100 × current
    capacity, so geometric decay is never faster) and rises only through restorations and
    upgrades, at most one completion of each per producer per cycle (they run one at a time),
    each costing one unit of its product through that product's recipe, so upgrades are a
    costed growth bound instead of free slack; consumer service in the TTL window (an order
    placed in cycle t is served only by the auctions of t..t + `ORDER_TTL`, per-cycle capacity).
    Product flows are aggregated over the horizon (a product may be used before it is made), and
    funding, inventory limits, routing and shocks are ignored. **Sound**: every played
    trajectory is a feasible point (argument in `lp_bound`), so the invariant B_lp ≥ every
    played arm's units holds like B2's (tolerance 0.01; a violation makes the run invalid). It is
    solved with HiGHS through scipy, the optional extra `lp` (not a core dependency: only the
    atlas bound needs an LP solver, and an in-repo simplex at 5k–45k columns would be the larger
    risk); a real atlas run refuses to start without it and the pod launcher installs it. It is
    solved once per world-episode, on the first job of that world-episode;
  - **B_lp_base**, the same program without upgrades: the base-capacity ceiling, replacing
    B_base wherever the LP ran. Every B_lp_base solution is a B_base relaxation point, so
    B_lp_base ≤ B_base. Not guaranteed, as B_base was not;
  - the world's **ceiling** C = max(min(B2, B_lp, B_lp_base), the best played arm's units) where
    the LP ran, else max(min(B2, B_base), best arm) (EXP-0051's screen rows have no LP cells), so
    C never sits below what was delivered. Every world-episode where an arm exceeds the
    base-capacity ceiling (B_lp_base, else B_base) is counted and reported (evidence of upgrades
    paying off), per arm and region;
  - **measured on EXP-0051** (task 4.7's offline re-bound of all 648 world-episodes, EXP-0051
    `RESULT.md` addendum): B_lp ≥ every recorded arm (0 of 7,776 rows above it); B_lp = B2 on
    601 and ≥ 0.91 × demand on all, because one costed upgrade per cycle can grow capacity about
    linearly on this lever range, so B_lp is still a demand cap and, like B2, is not used alone
    to judge room; B_lp_base < B_base on 202 (median ratio 1.000), never above it. With the LP
    ceiling one world changes class (open → saturated, not selected);
  - **selection before confirm**: the LP ceiling can only lower C (B_lp_base ≤ B_base), so it can
    only move a world from open or hard toward flat or saturated, never create a candidate.
    EXP-0052 therefore reads EXP-0051's recorded selection and split unchanged (the split is
    recorded before confirm) and classifies every selected world with the LP ceiling on its own
    fresh episodes; a selected world that the LP ceiling demotes shows up as a class change. On
    EXP-0051 none of the 96 selected worlds is demoted; rerunning the selection rule with the LP
    ceiling keeps all 32 at 42 and 60 of 64 at 12, the other 4 a re-ranking among worlds that stay
    open (room ≥ 0.59). No derived screen directory is written and EXP-0051's files are never
    edited.

**Measures per world** (equal-weight mean over the world's episodes; paired by world-episode with
equal demand and shock digests):
- `bar` = the largest mean units among the blind arms; `reach` = the largest among the reach arms;
- `gain` = reach − bar, from the paired per-episode differences of the two argmax arms;
- `room` = (C − bar) / C;
- `noise floor` (pre-registered; a per-world standard error from 1–2 episodes has 0–1 degrees of
  freedom and cannot serve, so the spread is pooled):
  - d_{w,e} = the paired per-episode gain (reach argmax arm − bar argmax arm, replicate-averaged
    for a stochastic arm); σ_ep(s) = the pooled within-world spread of d_{w,e} over all confirm
    worlds of stratum s = size × choice level that have ≥ 2 differing episodes (stochastic demand
    or shocks), with df = Σ_w (n_w − 1) (≥ 30 expected: about 6–20 such worlds × 1 df per
    stratum, pooled over the two regimes; strata below 30 df are merged within size, recorded);
  - σ_rep(s) = the pooled within-episode spread between the R = 2 replicates of the stochastic
    argmax arms, when either argmax arm is stochastic (0 otherwise);
  - **spread estimator (revised by task 5.3, after EXP-0052)**: each pool is a set of residuals
    r = (x − x̄_g) · √(n_g / (n_g − 1)) within each group g (a world's episode gains for σ_ep, an
    arm's replicates on one episode for σ_rep; n_g = 2 gives r = ±(x₁ − x₂) / √2, each with the
    variance of one value), and σ = median |r| / Φ⁻¹(0.75) (Φ⁻¹(0.75) = 0.6745), the
    normal-consistent median absolute residual (MAD). Pooling, strata, the 30-df merge, SE_w and
    the floor are unchanged. Atlas v1 (EXP-0052 as run) used the pooled standard deviation
    sqrt(Σ r² / df) of the same residuals; `worlds.csv` keeps both (`sigma_*` used, `sd_*`
    reported). The screen keeps the pooled SD of EXP-0050's spreads as recorded in EXP-0051;
    only confirm classes decide. Reasons, on statistical properties: (1) the pooled SD weights
    each world by its squared residual, so its breakdown point is 0: one world can set every
    floor of its stratum, which is what EXP-0052 showed (5 worlds carry 89% / 72% of the σ_ep sum
    of squares, 69% / 87% of σ_rep's; floors 397 / 577 units against median per-world swings of
    52 / 134); the MAD's breakdown point is 50%, so fewer than half the worlds, however wild, move
    it only to the edge of the others; (2) its only assumption is the one the floor already
    makes (roughly normal errors of a common scale within the stratum), with no trimming count,
    tuning constant or model of how spread scales with volume (the alternatives considered:
    dropping the top-k worlds needs a k; spread relative to the bar adds a scaling model and stays
    a sum of squares; a Huber or Qn scale adds efficiency at the cost of a tuning constant or a
    centre-free construction the paired residuals do not need; excluding worlds with a death was
    checked and does not move σ_ep, EXP-0052 RESULT addendum); (3) the price is efficiency: about
    37% of the pooled SD's at the normal, so 58 / 31 df act like about 21 / 11, and the
    false-hard rate per world at 3 × SE rises from ≈ 0.2% to ≈ 0.3–0.6% (≈ 0.4 of 96 worlds,
    before sign consistency and the 1% term). The residual risk is the opposite assumption
    failing: when noise is genuinely heteroscedastic, a world noisier than its stratum's median
    gets too low a floor; sign consistency in every episode and replicate is the per-world guard,
    and the reclassification reports each hard world's own episode spread against σ_ep.
    **Honesty note:** this revision was chosen after EXP-0052's sensitivity counts for eight
    estimators were known (2–12 hard at 12 producers, 5–17 at 42; median-based 12 / 13). The
    choice rests on the properties above, not on the count, and EXP-0052's addendum shows the
    counts under the pre-registered rule and this one side by side;
  - SE_w = sqrt(σ_ep(s)² · [world has differing episodes] + σ_rep(s)² / R) / sqrt(n_w);
  - floor_w = max(3 × SE_w, 0.01 × bar). Worlds with cyclic demand, no shocks and two
    deterministic argmax arms replay identically, so their gain is exact and only the 1% term
    applies;
  - **hard additionally needs sign consistency**: the gain is positive in every episode (and, for a
    stochastic argmax arm, in each replicate) of the world.
  - Expected false-hard rate (as pre-registered, for the pooled SD; the revised estimator's is
    above): under a true gain of 0 and roughly normal errors with ≥ 30 pooled df,
    P(gain > 3 · SE) ≈ 0.2% per world (one-sided t), before the sign-consistency and 1% terms; over
    the 96 confirmed worlds that is ≈ 0.2 expected false-hard worlds. EXP-0050's per-episode SD of
    paired arm differences (2.6–17.5 units, review of this revision) puts 3 × SE at 6–37 units for
    2 episodes, below the 25–420-unit gains EXP-0050 measured between router families.
  - **Screen** (1 episode, no own spread): provisional classes use σ_ep and σ_rep per size ×
    fanout estimated from EXP-0050's `episodes.csv` (task 4.6 records them; dense uses its own
    cells) with a 2 × SE multiplier and the 1% term. Screen classes only drive the selection;
    confirm decides.

**Classes** (pre-registered, evaluated in this order):
1. *collapse*: every arm reaches system death before cycle 50 in every episode; or (confirm,
   **dead-arm rule**, task 5.3) the fixed bar or reach arm reaches system death, at any cycle, in
   any episode or replicate. A death cuts that episode's units short, so a gain over a dying bar
   arm measures the death, not routing, and the room under a dying bar is inflated the same way
   (EXP-0052's 12-0435: `random_whole` dies at cycle 4 in all 4 runs, bar 0, gain +874, room 1.00,
   while other blind arms deliver 470–790). Such a world is reported as collapse with the dying
   arm named (`argmax_died`), never hard or open; the screen, whose argmax arms are not fixed,
   keeps rule 1's first clause only. The rule is deliberately wider than the bar-death case that
   motivated it (user decision 2026-10-29: "a world whose fixed bar or reach arm dies … is
   collapse, never hard"), and the trade-off is accepted: (a) a reach-arm death does not bias
   room (C − bar) / C, so it costs worlds that would read open (EXP-0052: 12-0312, 12-0423,
   42-0051); it is still excluded because the open class feeds the teacher sample (D5), whose
   readings compare the teacher with the reach arms on the same world-episodes, and a world where
   a cheap input-aware policy can drive the economy to system death mixes routing headroom with
   death risk that neither the gain nor the ceiling measures; (b) a late or single-replicate
   death biases the bar by little (12-0140: one replicate at cycle 84), and the alternatives,
   dropping the dying run or counting only deaths before cycle 50, are not taken: the first
   changes the paired design and the replicate count per world, the second needs a cut-off
   with no stated basis for a fixed arm, and both keep worlds whose reference arm is fragile in
   the teacher sample. Collapse therefore has two meanings, every-arm failure (a world-design
   problem) and a dying fixed arm (a fragile world, other arms may deliver normally); readers of
   atlas v2 tell them apart by `argmax_died`;
2. *saturated*: bar fill ≥ 0.95 (the registry's saturation ceiling) or bar ≥ 0.97 × C. This is
   the saturation filter: EXP-0050's 12-f4 sparse and dense-12 cells (fill 0.97–0.99) fall here by
   fill; a capacity-infeasible world (demand far above what base capacity can make) falls here by
   the ceiling instead of reading as room, so it is not sent to the teacher;
3. *hard*: gain > noise floor: a better policy is reachable and measured. "Hard" never means that
   everyone fails;
4. *open*: room ≥ 0.10 and not hard: the ceiling leaves room no cheap arm reached; candidates for
   the expensive teacher (D5). Since B_base is not guaranteed, a world just outside "open" because
   an upgrading policy could exceed B_base is a possible miss; the count of arms exceeding B_base
   says how often that happens;
5. *flat*: the rest.

**Screen (EXP-0051).** Per size, a Latin hypercube sample (D10): **1,024 worlds at 12 producers
(512 after reduction step 1, applied by task 4.6: sizing.md §EXP-0051),
128 at 42**, 1 episode per world, the blind arms (`random` R = 1, `random_whole` R = 2) and the
reach arms without `portable`: 13 router-equivalent episodes per world (the trained orderer counts
2). Plus 8 **anchor** world-episodes: world 0 of each EXP-0050 cell type, its first shocked
episode, rebuilt by `fanout_worlds`; their rows for the arms both runs share must reproduce
EXP-0050's units (same world-episode, subject and replicate), linking the two drivers. The screen
classifies every world with the screen's provisional noise floor (above) and then
selects:
- **Selection** (deterministic, pre-registered): K = 64 worlds at 12 and 32 at 42 from the hard and
  open worlds. Coverage bins are the levels of every categorical axis and the quartiles of every
  measured continuous coordinate (ρ, σ, the three heterogeneity CVs, shock intensity, demand
  concentration, volatility), per size. Pass 1: for each bin in a fixed order, if no selected world
  covers it, add the candidate in that bin with the best score. Pass 2: fill up to K by score.
  Score: hard worlds by gain / bar, then open worlds by room. At least K/4 open worlds when that
  many exist.
- **Split** (recorded before confirm, by world): an out-of-region held-out part = every selected
  42-producer world with the regime-switch pattern; then, within strata of size × choice × pattern,
  20% held-out, 20% validation, 60% train by sha256(world id) order. Held-out worlds are run and
  classified in confirm, but no M3 choice reads their results (PoC-A samples from train only); they
  are reported in aggregate.
- **Load factor.** Larger k_c and m_q mean more orders and bids per cycle; task 4.6 measures the
  cost multiplier L over the lever range. The estimate uses L ∈ [1, 2] until then.
- **Reduction order** if task 4.6 extrapolates the screen above its stop: 1. 512 worlds at 12;
  2. 64 worlds at 42; 3. nothing launches and the implementer reports.

**Confirm (EXP-0052).** The K selected worlds on **fresh** episode and replicate seeds (selection
used the screen's draws, so confirming on new draws removes the winner's-curse bias): 2 episodes
per world (1 for worlds with cyclic demand and no shocks), the blind arms with `random` and
`random_whole` at R = 2, the reach arms with `makeable_weighted_whole` at R = 2, plus `portable`:
18 router-equivalents per world-episode. The argmax arms of `bar` and `reach` are fixed from the
screen; confirm re-estimates gain, room and the full noise floor and reclassifies. `portable` is
reported beside the reach arms and does not change the fixed argmax, so the confirmed gain stays an
estimate on fresh draws. A world whose
class changes is reported as such. EXP-0052 freezes **atlas v1**: `atlas_v1.json` in its EXP
directory (content hash; per world the record of D10, screen and confirm classes, gain, room,
noise floor, split; the run commits of EXP-0051/0052). It is never written into the package's
registry directory.

**Reclassification (task 5.3, after EXP-0052).** EXP-0052's confirm floor was dominated by a few
worlds and one hard world had a dying fixed bar arm (DIAGNOSTICS 20, 21). The revised estimator
and the dead-arm rule above are applied offline to EXP-0052's recorded rows (`python -m
tokenomics.eval.atlas --mode reclassify --confirm-dir <EXP-0052> --out-dir <EXP-0052>/reclassified`;
nothing is played, the confirm's own files and `atlas_v1.json` are not edited). It writes
`reclassified/worlds.csv` (with the recorded `class_v1`), `deciding_check.csv`,
`reclassify.json` (inputs' hashes, rules, counts, flips) and **atlas v2** (`atlas_v2.json`,
format `atlas-v2`, content hash, `reclassified_from` = the hashes of `episodes.csv`,
`worlds.csv` and atlas v1); the split, the fixed arms and the world records are atlas v1's. Every
later reader of the atlas (PoC-A, PoC-C) reads atlas v2. A future confirm run writes
`atlas_v2.json` directly.

**Deciding check read in confirm** (EXP-0050's, pre-registered there): on hard worlds with E ≥ 4,
`makeable_split` within the noise floor of the trained orderer on at least 2/3 of them →
input-aware weighting is the lever and splitting is not the cost; `makeable_weighted_whole` above
`makeable_split` by more than the noise floor on at least 2/3 → splitting still costs even with
input-aware weights; `makeable_split` below the trained orderer by more than the floor on at least
2/3 → the scorer learned more than its target. Plus the new tracer field: the share of units lost
to expired reservations whose receiving producer's input stock at routing time could not cover the
order (`ttl_lost_units_inputs_short` against `_inputs_ok`), per arm. The result goes to
DIAGNOSTICS entry 19 and EXP-0052's `RESULT.md`.

**What the atlas does not do.** It never edits the gate registry, roster or parameter files, and
atlas worlds never enter a gate report. Adding atlas families or worlds to a gate registry (a v3
re-screen) or opening roster v3 is the user's decision; EXP-0052's `RESULT.md` states the evidence
for it (which regions are hard, which are saturated) and stops there.

**Relation to the closed mining arc (HISTORY §7).** EXP-0023–0036 mined topology and capacity
tightness against a search bar and found no headroom (EXP-0033: 98.3% of the bound). The atlas
differs in three ways that answer that failure's cause: it measures room per world against an
analytic bound before any training (a saturated world is filtered, not mined); it varies seven
axes including demand patterns and choice, where EXP-0050 showed routing differences grow; and it
calls a world hard only when a cheaper-than-search policy already demonstrates a gain above the
paired noise floor (EXP-0025's lesson).

## Risks / Trade-offs

- [The atlas finds few hard worlds] → the classes still map where headroom is not (saturated or
  flat regions), which is itself the input M4 needs; open worlds go to PoC-A. Recorded with
  DIAGNOSTICS entry 20's branches.
- [The ceiling is loose (it ignores TTL, routing, funding and degradation), so "open" over-counts;
  B2 alone would be the demand cap] → room is read against the base-capacity ceiling C, not B2;
  open is only a teacher-sample class, never "hard"; PoC-A's `ub` arm separates ceiling-limited
  from fork-limited.
- [B_base is not a guaranteed bound] → it only shapes the saturated / open classes (not "hard",
  which rests on played arms), C is never below a played arm, and arms exceeding B_base are
  counted.
- [Latin hypercube coverage at 42 is thin (128 points over about 13 dimensions)] → the 12-producer
  atlas maps the space densely; the 42 sample checks where the map transfers. A second 42 batch
  needs the user's approval (D8).
- [Load-heavy corners cost more than the probe's cells] → task 4.6 measures the load factor at
  the lever corners; the reduction order caps the screen.
- [Demand patterns or larger quantities break an observation bound] → the contract monitor runs on
  every lever corner in task 4.4 and on every atlas episode; a failure narrows the lever.
- [The new teacher floor (`round_robin`) changes the EXP-0042 teacher] → justified by the diagnosed
  cause (D5); the frozen configuration's result stands for its own worlds; the reviewer or user may
  revert (Open Questions).
- [Posterior acceptance is low on some shocked worlds] → the unresolved fallback keeps labels
  honest; the rate is reported; many unresolved decisions read as "fork-limited".
- [Tracing perturbs the episode] → additive pure data. Goldens and a with/without-trace equality
  test.

## Migration Plan

Nothing is deployed. Frozen v1/v2 stay the gate defaults. Every existing world and statistic is
byte-identical. Atlas v1 lives in its EXP directory. A roster v3 or an atlas-based registry exists
only after a user decision, a re-screen and a second approval.

## Open Questions

- **Teacher floor.** PoC-A uses `round_robin` as the floor and whole-order receivers plus the
  trained orderer's action as candidates (D5). The reviewer or the user may prefer the frozen
  EXP-0042 configuration (`blend_0.5` floor, +1-logit bumps) for comparability; it starts the
  search 100–400 units below the bar on the worlds EXP-0050 measured.
- **42-producer teacher.** Not in the budget (D5, D8); needs the user's approval when the 12-producer
  reads exist.
- **Roster v3 and atlas families in the gate.** User decisions (D3, D12). EXP-0052 records the
  evidence.
- **PoC-B scale (answered, 2026-10-28):** capped at 42 producers; no larger point.
