# World space and the headroom atlas

Status: accepted (add-m3-development-audit, milestone M3 development audit; design D10–D12)

## Context

The gate's registry v2 worlds leave little room for a better policy. EXP-0044 and EXP-0050 found
several cells saturated (dense-12 and sparse 12-f4 fill 0.97–0.99). The routing audits
(EXP-0048 → 0050) showed where routing differences come from: rotation and whole orders, not
randomness (EXP-0048). Spare-capacity rules do not beat blind choice (EXP-0049). Routing matters
more as the number of producers per product grows, and there the trained orderer leads
(EXP-0050). All of this was measured on one narrow slice of possible worlds: sparse fanout 2,
scripted B9 demand and the `shocks-i` profile.

The closed mining arc (`experiments/HISTORY.md` §7, EXP-0023 → 0036) varied topology and
capacity tightness against a search bar and found no headroom. EXP-0033 put the champion at
"98.3% of the bound". Task 4.2 showed that this bound is a demand cap on family worlds: upgrade
slack is 33–74× base capacity, so B2 equals the scheduled demand.

Mining again on more topologies would repeat that failure. What is missing is a way to:
- describe a world along more axes;
- measure each world's room against an analytic ceiling before any training;
- call a world "hard" only when a cheap policy already shows a gain above the noise.

## Decision

### World space (D10; `eval/world_space.py`, levers in `eval/families.py`)

A world is a size, a world seed and one value per **lever**. Levers are sampled, and
**measured coordinates** are recorded, because several levers move one coordinate and the
realized value depends on the world's draws.

| Axis | Lever (atlas v1 range) | Neutral | Measured coordinate |
|---|---|---|---|
| Size | P ∈ {12, 42}; any other size is refused | — | producers |
| Choice | 12: sparse_f2, sparse_f4, dense; 42: sparse_f2, f4, f7, f14, dense (equal shares) | sparse_f2 | declarers per product |
| Capacity tightness | k_c ∈ [0.5, 8], log-uniform; consumers = max(1, round_half_up(k_c · 3P/12)) | 1 | ρ = `demand_units_per_cycle` / `top_layer_throughput` |
| Order size | m_q ∈ [1, 8], log-uniform; capacity scale c ∈ [0.25, 2], log-uniform (D10's initial [0.25, 1] widened by the task 4.6 calibration, below) | 1, 1 | σ = mean over orders of quantity × capacity units per unit ÷ median declarer capacity |
| Heterogeneity | h_cap, h_deg, h_cost ∈ [0, 2], uniform; v → med + h·(v − med) per layer | 1 | within-layer spread: mean over layers of the population CV of capacity, of degradation, of restoration and upgrade cost (0 at h = 0) |
| Shocks | off / on; r ∈ [0.5, 3], m ∈ [0.2, 0.8], d ∈ {2..10} | off, or (1, 0.45, 3) | expected events per producer per 100 cycles, mean hit magnitude, mean outage duration |
| Demand pattern | cyclic, iid, drift (j = 0.34), burst, hit, seasonal, switch (equal shares); D11 parameters drawn per world | cyclic (calm), drift (shocked) | Herfindahl concentration and CV volatility of the realized demand |

**Calibration (task 4.6).** The screen sample (atlas-v1: 1,024 worlds at 12, 128 at 42) must
span ρ and σ each over at least [0.05, 1.5] per size. With D10's initial c ∈ [0.25, 1], ρ spanned
[0.038, 15.8] / [0.047, 13.0] but σ only [0.058, 11.6] / [0.071, 7.4] (12 / 42): no world with
orders small against capacity. c's upper end was raised to 2, the limit `families` already
accepts (c > 1 scales capacity, restoration and upgrade amounts up); σ then spans
[0.032, 10.1] / [0.038, 7.1] and ρ [0.024, 14.6] / [0.028, 13.0]. c = 1.5 was not enough (σ
minimum 0.049 at 42). The lever-corner contract test covers c = 2 (sizing.md §EXP-0051).

Implementation choices that the design leaves open, recorded here:
- **Drift j is fixed at 0.34.** This is the registered `shocks-i` value. D11 gives no range for
  drift, so the atlas does not sample it.
- **Pattern parameters are not Latin-hypercube columns.** They are drawn uniformly in their D11
  ranges, on a stream derived from the world seed. Burst factors stay at 5–10 and the switch
  window at [11, 90] with gap 10. The seasonal phase is uniform in [0, period).
- **Extra consumers and their pools.** Consumer i ≥ C (C = the base count) takes the pool of
  consumer i mod C, under the family's pool rule at the base count. Its quantity bound (and,
  when calm, its schedule) is drawn on a stream derived from (world seed, i). Two consequences:
  - per-product load stays balanced as k_c grows;
  - a smaller k_c's consumers are a prefix of a larger one's.

  With k_c < 1 the world keeps the first consumers as drawn.
- **Shocks.** The shock profile is an object built in code
  (`families.shock_profile_from_levers`):
  - it keeps the `shocks-i` kinds and window;
  - counts are × r, kept as an exact fraction;
  - hit magnitudes are [m − 0.15, m + 0.15];
  - outage durations are [d, d + 3].

  It applies in either regime, so cyclic demand can carry shocks, and its drift is never
  applied. `shock_profiles_v1.json` is neither read nor changed.
- **What ρ measures.** ρ uses the B9 range midpoint (or a calm schedule's mean). It does not
  include the burst or seasonal factor; the realized volatility carries those. σ, concentration
  and volatility are measured on the schedule the env builds for the measurement seed (the
  world seed) over 100 cycles, so drift and patterns are covered.
- **Latin hypercube.** One numpy permutation of n strata per column.
  - A continuous column puts one uniform point in each stratum.
  - A categorical column assigns level (stratum · L) // n, so every level gets ⌊n/L⌋ or ⌈n/L⌉
    worlds. For 1,024 worlds and 7 patterns that is 146 or 147.
- **Seeds.** World seed = sha256(atlas, size, index). Episode seeds hash the world id, and a
  stage salt gives the confirm stage a disjoint set.
- **Record.** Each world's record holds:
  - its levers, seed and generator version (`world-space-v1`);
  - the shock window, which a horizon-12 smoke moves to cycles 2–8, and the pool width;
  - the measurement seed and horizon, and the coordinates;
  - the declarers per product and the config sha256.

  Rebuilding from the record must reproduce the hash.

**Neutral byte identity.** Every lever transform runs after every existing draw, on values that
are already drawn, and each neutral value skips its transform. With all levers neutral:
- the configuration, its fingerprint and a horizon-12 episode's statistics equal
  `build_family_config`'s, for sparse f2/f4 and dense × calm/shocked × 12/42;
- two registry v2 development worlds keep their config hashes;
- all 64 worlds pinned in EXP-0050's `spec.json` reproduce their archived config sha256.

Recipes and `produces` sets never move when a lever moves.

**Stable config hash.** The record's hash is `routing_audit.config_digest`, the sha256 of the
config's dataclass tree. Adding `Configuration.demand_pattern` (task 4.3) had moved every digest,
because the unset field was hashed as null. The digest now drops an unset optional field that was
added after the first pins (`DIGEST_OMIT_WHEN_UNSET`, currently `demand_pattern`). `shocks` and
`demand_drift` predate the pins and stay hashed as null, so the calm pins hold too. A later
optional field must join that tuple, and a test refuses one that does not.

**Lever corners.** Task 4.4 ran the following at horizon 12, with the shocks inside the
horizon:
- each lever at both ends, with the others neutral;
- the all-low and all-high corners (shocks on, burst pattern), at 12 producers (sparse_f2) and,
  for the two all-corners, at 42 (dense);
- every pattern kind at its parameter ends.

Each was played with `entity_planner/round_robin` and `entity_planner/trained_orderer`. The
contract monitor recorded **no failure**, so no lever range was narrowed. Task 4.6 widened c to
[0.25, 2] (Calibration, above); the same corner test, now with c = 2 at its upper end and in the
all-high corner, again recorded no failure.

### Demand patterns are demand (D11)

Burst, moving hit, seasonal and regime switch are part of the demand schedule. Each draws on its
own stream, is covered by the demand digest and is kept in the shock-free twin. None is a shock,
and none is the M6 demand surge. Product patterns (hit, switch) widen each pool to min(4, W)
consecutive top products, which is recorded as the world's pool width. Patterns act on B9
consumers only, so a cyclic world is never patterned.

### Headroom atlas (D12; `eval/atlas.py`, task 4.5)

- **Arms.** Each world is played by:
  - blind arms, which set the **bar**: the roster v2 rules, `random`, `round_robin` and
    `random_whole`;
  - reach arms: `makeable_split`, `makeable_weighted_whole`, `entity_planner/trained_orderer`,
    and `portable` in confirm only.
- **Bound and ceiling.** Two bounds are computed from the episode's built demand:
  - B2, which is sound: no arm may exceed it, and a violation makes the run invalid;
  - B_base, the base-capacity ceiling, which is not guaranteed.

  The ceiling is C = max(min(B2, B_base), best arm), and room = (C − bar) / C.

  *Note (task 4.7, 2026-10-29).* Two more bounds come from one input-output linear program
  (`ceiling_bound.lp_bound`, design D12):
  - B_lp, which is sound like B2. It accounts for producer eligibility, the joint product mix,
    exact recipes, degradation with costed restorations, costed upgrades (at most one per
    producer per cycle) and the TTL window;
  - B_lp_base, the same program without upgrades. It replaces B_base as the base-capacity
    ceiling and is never above it.

  Where the LP ran, the ceiling is C = max(min(B2, B_lp, B_lp_base), best arm). The solver is
  scipy's HiGHS, in the optional `lp` extra. On EXP-0051's lever range a costed upgrade per cycle
  still lets capacity reach demand, so B_lp is mostly a demand cap too.
- **Classes.** Evaluated in order:
  1. collapse;
  2. saturated (bar fill ≥ 0.95 or bar ≥ 0.97 · C);
  3. hard (gain > the noise floor, with the sign consistent in every episode);
  4. open (room ≥ 0.10);
  5. flat.
- **Noise floor.** Per-world errors come from spreads pooled per size × choice: σ_ep over
  episodes and σ_rep over replicates, with strata below 30 df merged within size. The floor is
  max(3 · SE_w, 0.01 · bar). The screen uses EXP-0050's spreads at 2 · SE.
- **Selection.** Deterministic: a coverage pass over the categorical levels and the coordinate
  quartiles, then filling by score. K = 64 worlds at 12 and 32 at 42, with at least K/4 open.
- **Split.** The selected 42-producer regime-switch worlds form an out-of-region held-out part.
  The rest is split 20/20/60 by sha256(world id) within size × choice × pattern. No M3 choice
  reads the held-out results.
- **Confirm.** Confirm runs on fresh episode and replicate seeds and freezes `atlas_v1.json` in
  its EXP directory.
- **Outside the gate.** The atlas never writes the gate registry, roster or parameters, and
  atlas worlds never enter a gate report. A v3 re-screen is the user's decision.

The atlas answers the mining arc's failure in three ways:
- it measures room against a bound before any training;
- it varies seven axes, including choice and demand patterns, where EXP-0050 saw routing
  differences grow;
- it calls a world hard only when a cheaper-than-search policy already shows a gain above the
  paired noise floor (EXP-0025's lesson).

## Considered Options

- **Mine more topologies against a search bar (HISTORY §7).** Rejected: it closed without
  headroom, and the bound it used was a demand cap.
- **Sample coordinates directly.** Rejected: coordinates such as ρ depend on several levers and
  on the world's draws, so they can only be measured after the world is built.
- **Apply the `shocks-i` drift with shocks.** Rejected: shocks and the demand process must vary
  independently, so drift is one value of the pattern axis.
- **Draw extra consumers' pools at the new consumer count.** Rejected: it shrinks pools to one
  product and skews load toward the last products as k_c grows. The base-count pool of i mod C
  keeps load balanced.
- **Use a third-party Latin-hypercube library.** Rejected: it adds a dependency for about ten
  lines of numpy.

## Consequences

- Gate worlds, fingerprints and goldens are unchanged: every lever is neutral by default.
- An atlas world is reproducible from its record alone, and the config hash is checked on
  rebuild.
- The B_base ceiling is optimistic about degradation and blind to upgrades, so the count of arms
  above B_base is reported. From task 4.7 on, B_lp_base takes its place: it costs degradation
  and restorations, but it stays blind to upgrades, so the count is kept.
- Sizes are capped at 12 and 42 until agents beat baselines at 42 (the scale target).

Cites: ADR-0055 (seeded streams, fingerprint omission, shock-free twin), EXP-0033
(`experiments/EXP-0033-offline-ceiling-bound`), EXP-0048–0050 (routing audits),
`experiments/HISTORY.md` §7.
