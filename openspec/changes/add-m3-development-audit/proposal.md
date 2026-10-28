# Proposal

## Why

Roadmap milestone M3 "Development audit" (`establish-project-baseline` design D1, D4) must show,
before any training spend in M4, whether a better policy than the hand-written ones is reachable
(PoC-A), which role M4 should train first (PoC-C), and what the deployed path costs as the
economy grows (PoC-B). M3's exit is: PoC-A/B/C recorded, and every failure recorded with a
diagnosed cause.

The project's main goal is to show that agents can **evolve themselves**: find worlds they handle
badly, learn from them, and improve. That needs worlds where a better policy exists. M3's first
three experiments showed that the gate worlds are not such worlds:

- **EXP-0048** diagnosed why the `random` ordering baseline set the hindsight baseline on 95 of
  the 128 registry v2 development world-episodes: routing each order **whole** to a producer that
  **changes over time** wins. A blind deterministic rotation (`round_robin`) does it better than
  `random`. A hand-written rule has a defect (`even_split`'s lowest-index tie-break).
- **EXP-0049** found that routers which read each producer's spare capacity do not beat blind
  rotation: spare capacity almost never binds (units lost to bids that did not fit ≤ 0.15 per
  episode). The losses are reservations that expire waiting for inputs.
- **EXP-0050** gave each product 2 to 14 possible producers instead of registry v2's 2. Routing
  differences grow sharply with the choice (the gap between `round_robin` and the best roster
  rule grows from about 150 to 300–400 units per episode at 42 producers), and the trained
  portable orderer becomes the best router at wider fanout. Why its split orders complete is not
  diagnosed. Two cells are saturated (12-producer fanout 4 and dense-12, fill 0.97–0.99) and
  separate nothing.

So the registry v2 worlds sit in an easy corner: capacity rarely binds (demand is about 6% of
top-layer throughput in every family), each product has 2 possible producers, and demand is
steady. There, blind spreading sets the bar. Hard-world mining so far (EXP-0023–0036, HISTORY §7)
varied about two dimensions, topology and capacity tightness, and closed when it found that the
mined distribution had no headroom (EXP-0033: the champion delivered 98.3% of an offline upper
bound).

**User decision (2026-10-28), option (a):** replace "PoC-A teacher on the v2 gate worlds" with a
**multi-dimension headroom atlas**: sample a world space with many axes, measure on each world how
far the best blind or hand-written policy sits below what better policies reach, and mine the
worlds where a better policy is reachable. PoC-A, PoC-C and PoC-B stay. The expensive teacher runs
only on a mined sample.

## What Changes

- **Random-routing root cause (done).** Instrumentation (per-bid rejection and failure reasons, a
  routing tracer, diagnostic routers outside the roster) and EXP-0048, EXP-0049 and EXP-0050, as
  recorded. Unchanged.
- **Atlas instruments (new, first).**
  - Fix the `train/orderer_scorer.py` `teacher_logits` docstring, which wrongly says
    `proportional` routes by the same signal (it weighs capacity only; the target is input-aware).
  - EXP-0050's deciding check as cheap arms: the scorer's teacher as a router (split each order in
    proportion to `makeable_qty_now`), its whole-order version (the whole order to one producer
    drawn in proportion to `makeable_qty_now`), and a tracer field that classifies every expired
    reservation by the receiving producer's input stock at routing time.
  - An offline achievable-units upper bound per world: EXP-0033's analytic bound
    (`eval/ceiling_bound.py`), restored from commit b96ecc0 (the parent of cleanup commit
    18c717d, reachable from branch `archive-1`; the `archive-1` head 7aeef1f no longer holds the
    file). Stated reason for the restore: the atlas needs a cheap per-world ceiling to tell a
    saturated world from a world with room; EXP-0033 is the record that this bound is sound
    (bound ≥ every observed arm, 0 violations). It is ported to read the demand schedule the env
    actually built, so it covers drift and the new demand patterns. Its free-upgrade slack makes
    it equal to the scheduled demand on these worlds, so it is kept as the sound invariant and a
    second tier without upgrade slack (base capacity only, not guaranteed) is the ceiling that
    room and saturation are read against.
  - **Demand patterns** in the env layer: bursty spikes, a skewed popularity with a moving hit
    product, seasonal waves and abrupt regime switches. Each draws on its own seeded stream,
    is visible to agents only through the orders, is covered by the demand digest, and leaves
    every world without a pattern byte-identical. Basket (co-ordered products) and
    heavy-tailed buyers are deferred.
  - A **world-space generator**: every world is a point on named axes (capacity tightness, order
    size against capacity, choice, producer heterogeneity, shocks, size 12/42, demand pattern),
    built from levers with neutral defaults that reproduce today's families byte for byte, with
    each world's measured coordinates recorded.
  - An **atlas driver** with a screen mode and a confirm mode: space-filling (Latin hypercube)
    sampling, per-world headroom against the paired noise floor, a saturation filter, selection
    by headroom and coverage, and a recorded train / validation / held-out split.
  - ADR-0057 "Multi-dimension world space and headroom atlas".
- **Atlas screen (EXP-0051)** on about 1,150 worlds (1,024 at 12 producers, 128 at 42), cheap arms
  only, and **atlas confirm (EXP-0052)** on the selected worlds with more episodes, the full
  portable stack and the deciding check read. Together they freeze atlas v1.
- **PoC-A, teacher on a mined sample (EXP-0053).** The EXP-0042 confidence-bounded rollout search
  with the design D4 fork rule (ADR-0056) runs only on a mined 12-producer sample (hard worlds,
  where cheap arms already show headroom, and open worlds, where the ceiling leaves room no cheap
  arm reached), sized by a probe. Its floor and candidate set are re-pointed at the diagnosed
  cause: the EXP-0042 floor `blend_0.5` splits orders and does not rotate, which EXP-0048–0050
  show loses 100–400 units to `round_robin`, so the floor becomes `round_robin` and the
  candidates are whole-order receivers plus the trained orderer's action.
- **PoC-C, role attribution (EXP-0054)** on the shocked worlds of PoC-A's mined sample instead of
  v2's shocked subset, because attribution needs a gap to attribute, and v2's easy corner gives
  little.
- **PoC-B, deployed-path cost (EXP-0055)**, capped at 42 producers (user decision, 2026-10-28),
  plus a load axis (consumers per producer), because the atlas varies it.
- **Compute.** M3 budget 40 vCPU-h, ≈ 10.8 used. The planned remainder expects about 11–31
  vCPU-h (central ≈ 18), with hard stops summing to 29, inside the remaining ≈ 29.2 (design D8).
  Anything beyond a stop, and every 42-producer teacher run, needs the user's approval and is not
  planned here.

No **BREAKING** changes. The frozen v1/v2 registry, roster and parameter files, every existing
episode statistic, `ORDERING_BASELINES` / `PRODUCING_BASELINES`, every existing world
configuration and every existing fingerprint stay byte-identical. New world families or a roster
v3 for the gate stay user decisions; this change may only produce their evidence.

## Answered and open questions

- **PoC-B scale (answered by the user, 2026-10-28):** capped at 42 producers. No larger point is
  timed, and the driver has no flag for one.
- **Teacher floor (for the reviewer / user):** PoC-A re-points the EXP-0042 floor from
  `blend_0.5` to `round_robin` (design D5). The alternative, the frozen EXP-0042 configuration,
  would start the search 100–400 units below the bar on the mined worlds.
- **42-producer teacher (for the user, later):** not in the budget. It needs a probe, a
  profile-and-optimize code task and the user's approval of its spend.

## Capabilities

### New Capabilities

- `development-audit`: how the M3 audit diagnoses a baseline anomaly, measures per-world
  headroom and mines a headroom atlas, what information a teacher label may depend on (the fork
  rule), how PoC-A, PoC-B and PoC-C are read, and that a failed probe ends in a recorded
  diagnosis.
- `world-space`: how a world is described as a point on named axes, how the generator realizes
  and records it, and that neutral axis values reproduce today's worlds.

### Modified Capabilities

- `gate-harness`: modifies "Frozen baseline roster", so that a roster version's pairings range
  over the baselines that exist when it is drafted and a later baseline enters only a later
  version. Adds "Baseline rule corrections create a new roster version" and "Raising the bar is a
  user decision".
- `stochastic-environment`: adds demand patterns (bursts, moving hit, seasonal waves, regime
  switches), each on its own stream, observable only through orders, digest-covered, and
  byte-identical when absent.

## Impact

- Domain: `domain/producing_agent.py` gained per-rejected-bid and per-failed-reservation reasons
  (done). `domain/config.py` gains an optional demand-pattern record (data only, like
  `DemandDrift`). No RNG, IO or logging in `domain/`; every existing statistic is unchanged.
- Env: new `env/demand_patterns.py`; `env/aec_env.py` builds pattern schedules on their own
  streams; new `env/shock_posterior.py`; `env/future_fork.py` gains drift- and pattern-continuous
  demand forks.
- Eval: `eval/routing_diagnostics.py` / `eval/routing_trace.py` / `eval/routing_audit.py` (done,
  plus the deciding-check routers and the input-stock expiry field); restored
  `eval/ceiling_bound.py`; new `eval/world_space.py` (axes, levers, sampler, measured
  coordinates); `eval/families.py` gains neutral-default levers; new `eval/atlas.py` (screen,
  confirm, headroom, selection, split, manifest); new `eval/teacher_audit.py`;
  `eval/bounded_rollout_orderer.py` gains a pluggable scenario source, floor and candidate set;
  new `eval/deployed_cost.py`.
- Train: `train/orderer_scorer.py` docstring only.
- Records: EXP-0048 to EXP-0055, HISTORY §13, DIAGNOSTICS entries 19 (done) and 20 (no headroom
  in a world set), ADR-0056 and ADR-0057, Domain Language terms ("World space", "Demand pattern",
  "Headroom", "Atlas bar", "Headroom atlas", "Unresolved label", "Clairvoyant bound", "Closure
  (role attribution)").
- Compute: five CPU pod runs after EXP-0050, each sized by a local probe, with hard stops summing
  to 29 vCPU-h inside the M3 budget of 40 (design D8).
