# Design: gate v3

## Context

See proposal.md (Why). The facts this design rests on:

- **Frozen v2 set** (EXP-0047, packaged in `eval/registry/`): 24 sparse-wide development worlds
  (per size 8 calm × 3 episodes + 4 shocked × 10 episodes = 64 world-episodes; 128 in all) and the
  sealed split (per size 16 calm + 8 shocked), fanout 2, light load. Roster v2 = 6 members, all
  `entity_planner`. X 0.7516 / Y 0.7302 from one global nearest-rank q = 0.05 rule. All three files
  share `screened_on` = the v2 draft hash, and `gate_cli.version_set_problem` refuses mixed sets.
  The v2 screen cost ≈ 4.0 billed vCPU-h for 12,800 episodes (48 members, R = 5).
- **Baselines.** `eval/baselines.py` has 8 producing and 6 ordering baselines. Roster drafts are
  `roster._cross(<ordering list>)` over every producing baseline. A member is stochastic only when
  either policy is `random` (`RANDOM_POLICY in (p, o)`). Stochastic routers draw from the env
  action-space RNG, which the gate seeds with `seeding.replicate_seed(world, episode, r)`.
- **Diagnostic routers** (`eval/routing_diagnostics.py` `DIAGNOSTIC_ROUTERS`) are white-box
  `Policy`s outside `ORDERING_BASELINES` by rule (development-audit "Diagnostic routers stay
  outside the roster"; M3 D3: a fixed rule reaches the gate only under a new name, through a roster
  v3 draft, after the user's decision, which was given on 2026-10-30).
- **Evidence on the candidate routers** (all with `entity_planner`):
  - `round_robin`: beats `random` on all 104 random-dominated v2 episodes (EXP-0048); the atlas bar
    arm on most confirmed hard worlds (EXP-0052).
  - `even_split_rotating`: recovers random's edge in 2 of 3 cells (EXP-0048).
  - `makeable_weighted_whole` (MWW): beats `round_robin` on 62% of 639 screen worlds, median
    +1.8% (EXP-0051 addendum); the reach argmax on 226 of 281 screen-hard worlds.
  - `random_whole`: the atlas bar arm on 7 of 24 confirmed hard worlds; beats `round_robin` on 40%
    of screen worlds.
  - `makeable_split`: the reach argmax on 7 of 24 confirmed hard worlds, although it beats
    `round_robin` on only 16% of screen worlds.
- **Atlas v2** (`experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json`, content
  hash `220886a7…`): 96 selected worlds split train 60 / validation 16 / held-out 16 /
  held-out-region 4. Every hard world of the two target regions is in one of these splits: at 42
  f14, hard worlds are train 6, validation 1, held-out 3, held-out-region 1; at 12 dense / f4,
  train 4, validation 5, held-out 2.
- **M4 claims every split** (add-m4-orderer-redesign D6, D9): train = step 1 and ES training (plus
  41 fresh 12-producer draws on the stream `m4-es-train-v1`), validation = selection and the
  success reading, held-out + held-out-region = the one confirmation read, tracked by
  `experiments/m4_heldout_ledger.json` (not yet written). No atlas split is free.
- **World space** (`eval/world_space.py`, ADR-0057): `sample_levers(atlas, size, n)` is a Latin
  hypercube over 9 continuous levers and 3 categorical axes; world seed = sha256(atlas, size,
  index). `build_world` with shocks on at the neutral shock levers (1, 0.45, 3) gives exactly the
  `shocks-i` kinds (`families.shock_profile_from_levers`), and pattern `drift` uses `shocks-i`'s
  j = 0.34. The `world_space` generator is already a vetted reproducer (`task_archive.REPRODUCERS`).
- **Cost references** (local s per horizon-100 episode, pod ≈ 0.646 × local; sizing.md §EXP-0051):
  42 f14 at load `round_robin` 22.6–29.0 s, `proportional` 27.8–53.0 s, trained 43–61 s; 12 dense
  / f4 at load 1.5–3.1 s for whole routers. EXP-0057: 639 atlas jobs at 12 and 42 in ≈ 1.4 billed
  vCPU-h.
- **Teacher cost** (sizing.md §EXP-0053 / §EXP-0054, 12 producers): posterior `d2` 8–33 s of
  search per cycle at 38–62 orderer turns, 21–47 streams per cycle and 2–4 candidates per stream;
  12-0408 (108 turns, 4 candidates) 75.8 s per cycle. Cost grows with turns per cycle (each floor
  rollout plays them all), streams per cycle, candidates per stream and the share of 16-scenario
  decisions. At 42 f14 every product has 14 makers, so 14 candidates per stream.
- **Teacher findings** (EXP-0053): `d2` is safe but weak at 12 (never below the bar, 2 of 4 open
  worlds above the floor, 1.6–6.3% of streams overridden). `ub` lands 1–99 units *below* `d2` on all
  5 worlds where it ran: with a two-cycle window on the true future most candidates tie at exactly
  0 and keep the floor (DIAGNOSTICS 23). So `ub` is no upper bound for this search, and no
  clairvoyant arm can serve as the 42-producer "ceiling".
- **M4 candidate format** (add-m4-orderer-redesign D4 / D5, ADR-0059): env option
  `ordering_decode = "choice-v1"`; composite checkpoint `per-order-composite-v1` with
  `decoder_version = "per-order-choice-v1"`, `architecture_version =
  "orderer-per-order-residual-v1"`, the EXP-0012 base SHA-256, the normaliser hash and θ; whole
  orders drawn by inverse CDF from a key (world, episode, replicate, agent, cycle, slot) set with
  `routing_diagnostics.set_draw_key`. The gate already averages candidate replicates for a random
  baseline subject and checks deaths, run-down and starvation per replicate (`gate.py`
  `Failure.replicate`).

## Goals / Non-Goals

**Goals:**
- A v3 set (registry, roster, parameters) that holds the strongest known hand-written routers and
  worlds where routing matters, frozen by one baseline-only screen with its own `screened_on`.
- Gate worlds that no M4 step trains, selects or confirms on.
- A 42-producer lookahead reference whose cost is measured before it is spent.
- A gate harness that can judge the M4 per-order orderer as the user specified, so the exit read
  needs no further harness work after M4's `confirmed`.

**Non-Goals:**
- Changing the acceptance margins (1.05 / 1.10), the hindsight-baseline arithmetic or the
  verdict's clauses.
- Trained policies or the `d2` teacher as roster members (D1).
- Editing v1 / v2 files or making v3 the default without the user (gate-harness "Raising the bar
  is a user decision").
- A sealed screen at 42 producers (M5) and a gate read of any candidate before M4's `confirmed`.
- Using the teacher read as a label source, a training target or an M4 role decision.
- Worlds above 42 producers.

## Decisions

### D1. Roster v3 draft: promote five routers, keep the cross, let pruning decide

- **Promotion.** `round_robin`, `even_split_rotating`, `random_whole`, `makeable_weighted_whole`
  and `makeable_split` enter `ORDERING_BASELINES` under their existing names, bound to the same
  functions as in `DIAGNOSTIC_ROUTERS` (they stay there too, for the audits). Their names never
  belonged to a baseline, so M3 D3's "new name" condition holds and no existing baseline's
  behaviour changes. A test asserts byte-identical actions between the two entries on a horizon-12
  episode, and unchanged v1 / v2 draft and frozen roster hashes (the drafts use explicit
  ordering lists, `ROSTER_V1_ORDERING` / `ROSTER_V2_ORDERING`).
- **Stochastic members.** A member is stochastic when either policy is in
  `STOCHASTIC_BASELINES = {random, random_whole, makeable_weighted_whole}`. Stochastic members run
  R = 5 replicates with the existing replicate seeds, valued at the replicate mean (gate-harness
  "Random-baseline replicates"). The stored `random` flag of v1 / v2 members is unchanged.
- **Draft.** `build_roster_v3()` = every producing baseline (8) × the 6 v2 ordering baselines plus
  the 5 promoted routers = 88 members; 32 stochastic (11 with the `random` producer, 21 = 7
  producers × 3 stochastic routers), so 216 runs per world-episode.
- **"If the evidence says so"** is the never-argmax pruning rule on the v3 development screen
  (D3): a promoted router stays only if, with some producer, it sets the per-episode best on at
  least one development episode. No hand rule picks MWW in or out.
- **Not members, and why:**
  - `mww_keyed` / `trained_sampled_whole`: keyed duplicates of MWW / the trained scorer for paired
    M4 comparisons, not separate rules.
  - The EXP-0012 trained orderer and the portable stack: checkpoints, not hand-written rules; the
    roster is defined as hand-written stacks (gate-harness "Frozen baseline roster"), and M4
    already compares its candidate to TR at validation and held-out. Whether the "strongest
    policy" rule should reach trained incumbents is listed for the user (Open Questions).
  - `d2`: a search teacher (minutes to hours per episode at 42), not a baseline.
- **Legacy contract.** `obs_stock_aware` × a promoted router runs under `LEGACY`
  (`roster.contract_for`). The promoted routers read `env.system` and emit the raw ordering action
  both contracts decode, so a horizon-12 contract-monitor smoke per promoted router under both
  contracts is part of the code task.

### D2. Registry v3: v2 unchanged plus two region families of fresh draws

**Why fresh draws, not atlas worlds.** Every hard world of both target regions is in one of M4's
four splits (Context), and M4 uses all four. Reusing train or validation worlds would let the gate
read worlds M4 trained or selected on. Reusing held-out or held-out-region would burn M4's
confirmation. The unselected EXP-0051 screen worlds were also rejected:
- the atlas selection read them and left them as the lower-scoring remainder of each bin, so they
  are a biased sample of the region;
- the EXP-0051 addendum regressions that shaped M4's design were fitted on all 639 of them;
- only one screen episode and screen labels exist for them.

Fresh draws from the same lever distribution, conditioned on the region's categorical levers, are
independent of every M3 / M4 choice. They are screened by the gate's own baseline-only screen,
like any registry world.

**Composition.**

| Split | Family | Size | Calm worlds × episodes | Shocked worlds × episodes | Reserves per regime |
|---|---|---|---|---|---|
| development | `sparse_wide` (v2, unchanged) | 12, 42 | 8 × 3 | 4 × 10 | 4 |
| development | `ws_loaded` (new) | 12 | 4 × 3 | 2 × 10 | 4 |
| development | `ws_wide` (new) | 42 | 4 × 3 | 2 × 10 | 4 |
| sealed | `sparse_wide` (v2, unchanged) | 12, 42 | 16 × 3 | 8 × 10 | 4 |
| sealed | `ws_loaded` (new) | 12 | 8 × 3 | 4 × 10 | 4 |
| sealed | `ws_wide` (new) | 42 | 8 × 3 | 4 × 10 | 4 |

- At each size and split, calm worlds stay twice the shocked ones (12 + 6 development, 24 + 12
  sealed), so gate-harness "Regimes" holds without change.
- Region worlds are one third of each (size, regime) group. The regime ratio stays the
  equal-weight mean over all its worlds; reports add a per-family table (reported, never a
  clause).
- The counts mirror v2's per-family shape at half its weight. Cost is the limit: a 42 f14 world
  costs about 5–10 × a v2 42-producer world per episode.

**Region definition.**
- `ws_wide`: 42 producers, choice `sparse_f14`, where 11 of the 13 hard 42-producer worlds sit
  under atlas v2 (9 of 10 under atlas v1).
- `ws_loaded`: 12 producers, choice alternating `dense` / `sparse_f4` by slot index within each
  (split, regime) group, gate worlds first, then reserves. These are the two choice levels of the
  12-producer hard worlds under atlas v2: 6 dense, 5 f4, none at f2.
- **Continuous levers** (`k_c`, `m_q`, `capacity_scale`, `h_cap`, `h_deg`, `h_cost`) come from
  `world_space.sample_levers`' Latin hypercube with n = 128 per (split, size). Its categorical
  columns are ignored.
- **Regime levers fixed by the slot.** Calm: shocks off, pattern `cyclic` (deterministic demand, no
  shocks). Shocked: shocks on at the neutral shock levers (1, 0.45, 3) and pattern `drift`, which
  is `shocks-i` with its drift. The generator reference names `shock_profile: "shocks-i"`, and the
  registry validates that the built world's shock kinds equal the registered profile's.
  - Why not the atlas's patterns (burst, iid, hit, seasonal) or its shock levers? Gate-harness
    "Regimes" requires every v2+ shocked gate world to carry a named, versioned shock profile
    with drift, and one shocked definition keeps the shocked regime one population across families.
  - The atlas's hard worlds carry several patterns, so this narrows the region. The narrowing is
    recorded as a limit.
- **Acceptance filter, in index order.** A draw is accepted for a slot when the world built for
  that slot has measured ρ in (0.28, 3.1].
  - 0.28 is M4's "loaded" cut (D9).
  - 3.1 covers every confirmed-hard world (max ρ 3.06) and excludes the over-demanded corner the
    atlas classed saturated (ρ up to 14).
  - Rejected draws are recorded with their ρ. If the pool runs out, the draft refuses to build;
    it never silently shrinks.
- **Seeds and ids.**
  - World seed = `world_space.world_seed("gate-v3-<split>", size, index)`.
  - Ids follow v2: `dev-ws_wide-42-calm-00`, `sea-ws_loaded-12-shocked-r1`.
  - Episode seeds = `registry.episode_seed(id, j)`.
- **Disjointness, tested.** Region world seeds must not collide with:
  - every v2 world and episode seed;
  - the atlas-v1 sample (512 + 128 screen worlds and the 8 EXP-0050 anchors);
  - the first 4,096 indices of M4's `m4-es-train-v1` stream at both sizes, which covers the
    pre-registered fresh pool;
  - each other across splits.
- **World record in the entry.** Each region entry stores its world record (levers, seed, generator
  version, measured coordinates, config SHA-256). Loading checks that the replayed config hash
  equals the recorded one (world-space "Same record, same world"). The record is inside the
  registry's content hash.
- **Families.** `GATE_FAMILIES` becomes version-aware: v1 / v2 admit `sparse_wide` only, and v3
  admits `sparse_wide`, `ws_loaded` and `ws_wide`. The legacy `dense` family stays refused: it
  failed headroom because it is over-capacitated (EXP-0044). `ws_loaded`'s dense half is loaded by
  construction (ρ > 0.28).
- **Constant economy ratios** apply to families built at both sizes (`sparse_wide`). Region
  families exist at one size each and are exempt; their size is part of the family definition.

### D3. The v3 development screen

Order (`gate_cli --mode screen`, extended):

1. **Wave 1.** The full 88-member draft on every development gate world except the 42-producer
   region worlds: the 24 v2 worlds plus the 6 `ws_loaded` worlds, 160 world-episodes, reserves on
   demand.
2. **Wave 2.** `ws_wide` development worlds (32 world-episodes, reserves on demand) with
   W2 = {members that set the per-episode best on ≥ 1 wave-1 episode} ∪ {`entity_planner` × every
   ordering baseline}.
3. **Replacement** within each wave, before anything else is decided:
   - zero-baseline replacement (existing);
   - bar-death replacement, region families only: a region world on which the per-episode
     argmax member reaches system death in any episode is replaced by the next reserve of its
     group. This is the gate's analogue of the atlas dead-arm rule. Such a world's X / Y would come
     from a dying stack, and the atlas classes it collapse, not hard. v2 worlds keep their v2
     rule, so they replay unchanged.
4. **Headroom** per family × size present (`sparse_wide` 12 and 42, `ws_loaded` 12, `ws_wide`
   42): median argmax fill < `saturation_ceiling` 0.95. A failure stops the screen (exit 3, no
   `frozen/`), as today.
5. **Pruning, rule `never-argmax-two-wave-v1`.**
   - A member is kept iff it is in W2 and sets the per-episode best on ≥ 1 episode of either wave.
     Each episode's argmax is taken over the members run on it.
   - Members outside W2 are dropped and recorded as "not carried to wave 2".
   - The pruning record holds both waves' member sets, episode lists and row digests.
6. **X / Y per family** (D4) over the final worlds with the pruned roster.
7. **Write** candidate `frozen/{worlds,roster,side_effect_params}_v3.json`, all with `screened_on`
   = the v3 draft registry's hash, plus `screening.json` and `headroom.csv`.

**Why two waves.**
- **Cost.** The full draft on the 42 f14 worlds alone is ≈ 32 world-episodes × 216 runs ×
  ≈ 19 pod-s ≈ 37 vCPU-h, more than everything else together.
- **Evidence for the restriction.** In EXP-0044 attempt 2 and EXP-0047 no stack whose producer is
  not `entity_planner` ever set the per-episode best: 35 of 40 and 42 of 48 members were dropped,
  and every kept member is `entity_planner`. That covers 256 development world-episodes at both
  sizes (the calm ones are the same worlds in both screens).
- **The safeguard.** W2 still carries every wave-1 winner, whatever its producer, plus every
  router with `entity_planner`. So the only members the restriction can wrongly drop are non-
  `entity_planner` stacks that never win on 160 cheaper world-episodes yet would win on f14.
- **The rule is development evidence only.** It reads baseline-only development rows, as
  gate-harness "Roster pruning uses development evidence only" requires. The requirement is
  amended to allow a pre-registered wave plan.

**Estimate.**

| Part | World-episodes × runs | Pod-s per run | Pod vCPU-h |
|---|---|---|---|
| Wave 1, v2 worlds | 128 × 216 | 1.125 (EXP-0047's average) | 8.6 |
| Wave 1, `ws_loaded` | 32 × 216 | ≈ 1.5 (sizing.md §EXP-0051, 12 dense / f4 at load) | 2.9 |
| Wave 2, `ws_wide` | 32 × ≈ 29 (≈ 13 members, 4 stochastic) | ≈ 19 (42 f14 at load, whole and splitting routers) | 4.9 |
| Total | | | ≈ 16.4 |

- On `cpu32` that is ≈ 35 min of driver wall. With pod setup, ≈ 18 billed vCPU-h; stop 36.
- Task 5.1's local probe replaces the per-run costs with timed ones on the drafted worlds before
  launch.
- **Pre-registered reductions,** applied in order only if the probe's × 2 billed estimate exceeds
  40:
  1. shocked region worlds at 5 episodes instead of 10, in both splits, recorded in the
     registry;
  2. wave 1's v2-world rows for the 48 v2-draft members reused from EXP-0047, after a reproduction
     check of one world-episode per (size, regime) × all 48 members matches exactly (saves
     ≈ 4 vCPU-h);
  3. stop and report to the supervisor.

### D4. Side-effect parameters per family (from v3)

- **Rule.** `side_effect_params_v3` records X and Y per gate family: `sparse_wide` (both sizes
  pooled, exactly v2's rule), `ws_loaded` and `ws_wide`. Each is the nearest-rank q = 0.05
  quantile of the per-episode argmax member's run-down ratio, and of its per-consumer fill rates,
  over that family's development episodes.
- **Use.** The run-down and starvation checks use the values of the world's family. v1 / v2 files
  keep their scalar X / Y and are read unchanged.
- **Why per family.** On loaded atlas worlds the best simple rule fills ≈ 46% of orders (EXP-0052).
  M4 D6 already found that v2's absolute X / Y would flag the bar arms themselves on such worlds.
  A pooled quantile would either flag region baselines or loosen the sparse-wide checks, depending
  on the mix. Per family keeps each check at "no worse than the best baseline's own 5% tail" on
  worlds of its kind.
- **Sample size.** `ws_wide` has 32 run-down values (4 × 3 + 2 × 10 development episodes), so
  nearest-rank q = 0.05 is its 2nd smallest. The evidence block records the count per family.
- **Alternative rejected:** one X / Y over all v3 worlds. It is simpler, but its value would be set
  by the arbitrary 2 : 1 family weighting.

### D5. Sealed split: drafted now, screened at 12 producers in this change

- Sealed region worlds are drafted with the same rules (D2). Sealed seeds come from their own
  stream, disjoint from development.
- **A new screen mode** (`--mode screen-sealed --sizes 12`) runs the frozen v3 roster alone on the
  sealed gate worlds of the named sizes, with zero-baseline and bar-death replacement only.
  - No pruning and no calibration: those use development evidence only.
  - It writes a new screened registry version that keeps `screened_on`, and records which
    (split, size) groups are screened.
  - A sealed report refuses a size whose sealed group is unscreened.
- **Scope.** M4's exit is the sealed 12-producer gate (`establish-project-baseline` D1). So only
  the 12-producer sealed groups are screened here; 42 waits for M5. Roster-only screening is not a
  candidate evaluation and needs no pre-registration (gate-harness "Sealed worlds need a
  pre-registration"); it is still appended to the access log.
- **Order.** It runs only after the user approves v3, because it screens with v3's frozen roster.

### D6. The per-order gate subject (user decision 4)

- **Subject.** `--subject per-order` with:
  - `--orderer-ckpt` (a `per-order-composite-v1` checkpoint);
  - `--deciding-producer <white-box producing baseline>` (default `entity_planner`);
  - `--reported-prod-ckpt <producer checkpoint>` (the EXP-0035 producer).
- **Deciding rows.** Subject `candidate`, producing = the named baseline, ordering = the composite
  policy. The env is `joint_stack` with `ordering_decode = "choice-v1"`. These rows alone enter
  the verdict.
- **Reported rows.** Subject `candidate:reported`: the same orderer, keys and seeds, with the
  checkpoint producer. Like twin rows, they are split off before the verdict. `reported.csv`
  gives, per world and regime, the reported ratio against the same hindsight baseline, deaths and
  the hard side-effect checks. They are labelled "reported, not gating" in `RESULT.md` and
  `verdict.json`.
- **Replicates.** The candidate runs R = the frozen roster's R (5) replicates per episode, in both
  columns.
  - Replicate r sets the env action seed `replicate_seed(world, episode, r)` and the draw key
    `DrawKey(world, episode, r)`.
  - The world ratio uses the replicate mean (existing).
  - Deaths, run-down and starvation are judged on every replicate (existing `Failure.replicate`):
    a deployed stochastic policy must be safe in each realised episode, and a mean would hide a
    dying replicate.
  - Shock-free twins run for deciding replicates only (resilience is reported, never gating); the
    reported column has no twins.
- **Pins** (in `spec.json`, the resume check and `--mode recompute`):
  - `ordering_decode`, `decoder_version`, `architecture_version` and the checkpoint format;
  - SHA-256 of the composite, the EXP-0012 base and the normaliser; the θ hash; the arm;
  - the M4 pre-registration file hash;
  - the deciding producer's name and the reported producer's SHA-256;
  - R.
- **Refusals.**
  - Every M4 D4 refusal: a composite under `logits-v1`, or a legacy scorer as `choice-v1`.
  - A keyed policy without a draw key is a contract failure (existing D3 rule).
  - A real report with R ≠ the roster's R, or with the deciding producer not a white-box baseline,
    is refused.
- **Sealed manifest.** For a sealed read, the pre-registration manifest binds every pin above, as
  well as the registry, roster and parameter versions (gate-harness "Sealed worlds need a
  pre-registration").
- **Why a white-box deciding producer is admissible.** The gate judges the ordering role M4 trains
  (EXP-0054 c_O = 1.02 vs c_P = 0.12). The roster's members carry the same white-box producer, so
  the deciding comparison isolates the orderer, as every atlas and ladder read did. The reported
  column keeps the deployable stack visible.

### D7. The 42-producer teacher read (user decision 3)

**Sample (pre-registered).** Atlas v2 42-producer train worlds with choice `sparse_f14`, class
hard or open, not collapse, pattern not hit or switch (spec: teacher worlds are train-only, and
hit / switch carry product-pattern state the posterior cannot sample). Each world uses the
confirm's first episode seed, so its rows pair with EXP-0052's and EXP-0057's by digest.

| World | Class | Shocks / pattern | ρ |
|---|---|---|---|
| 42-0054 | hard | shocks, cyclic | 0.25 |
| 42-0076 | hard | shocks, burst | 1.24 |
| 42-0073 | hard | –, iid | 0.10 |
| 42-0126 | hard | –, iid | 0.80 |
| 42-0113 | open | shocks, burst | 0.29 |
| 42-0122 | open | –, seasonal | 0.04 |

**Arms.**
- `floor` (`round_robin`) and `d2`: the EXP-0053 search over the `round_robin` floor, with
  whole-order candidates plus the trained orderer's row and ADR-0056 posterior forks.
- The atlas bar and reach arms, rerun on the same world-episodes.
- **PoC-C** on the shocked worlds (0054, 0076, 0113): S (portable), A (portable producer + `d2`),
  B (`entity_planner` + portable orderer), with T = this run's `d2` row.
- **`ub` is not run.** DIAGNOSTICS 23: it is no upper bound for this search, and it costs a full
  search episode. The read's upper references are the atlas's analytic ceilings (B2, B_lp, C)
  from atlas v2's rows, reported beside `d2`.

**Readings** (pre-registered; never an acceptance read, a label source or a training target):
- Per world: `above_bar` / `within_floor` / `below_bar` for `d2` − bar against the one-episode
  floor 3·σ_ep of the 42 f14 cell. The same three-way reading for `d2` against MWW, `round_robin`
  and TR.
- The share of room `d2` closes, (d2 − bar) / (C − bar).
- **Summary.**
  - `teacher_headroom_42` if `d2` is `above_bar` on ≥ 2 of the hard worlds played, else
    `no_teacher_headroom_42`.
  - PoC-C closures c_O and c_P with the 0.15 tie band.
  - c_P ≥ c_O at 42, or a tie, is recorded and goes to the user, because it would question M4's
    role choice at 42. It never changes M4 automatically.
- **Wording.** `d2` is a lower bound on what lookahead reaches, not a ceiling. The read is a
  *reference* for the orderer at 42, named as such in the record.

**Cost and the reduction ladder.**
- **Extrapolation.** Scaling §EXP-0053's posterior 12-producer cycle (≈ 25 s at 4 candidates,
  ≈ 25 streams, ≈ 55 turns) by 42 f14's ≈ 3.5 × streams, 3.5 × candidates (14 vs 4) and ≈ 17 ×
  per-cycle simulation cost (EXP-0051: 0.23–0.29 s vs 0.011–0.016 s per cycle) gives ≈ 5,000 s
  per searched cycle. That is ≈ 150 h for one full `d2` episode.
- So the full design cannot run, and the read is sized by task 5.2's probe. The probe measures s
  per searched stream on 0054 and 0126 with all candidates and with K = 4, on three cycles each,
  under a per-command time budget, as `timing_63.py` did.
- **Ladder** (pre-registered; each step is applied, in order, only while the probe's × 2 billed
  estimate exceeds the stop or the longest job's × 2 wall exceeds the shape's stop wall):
  - **L1:** sample → the 4 hard worlds; PoC-C on 0054 and 0076.
  - **L2:** candidate cap K = 4: the trained orderer's row plus the whole order to each of the 3
    eligible producers with the largest `makeable_qty_now` (ties by producer id), the floor's
    receiver excluded. This is the candidate count of the 12-producer f4 / dense worlds.
  - **L3:** stream cap S = 8 searched streams per cycle, largest consumer order quantity first
    (ties by order id); the floor routes the rest.
  - **L4:** PoC-C dropped (S and B alone attribute nothing).
  - **L5:** sample → 0054 and 0126 (one shocked, one stochastic-demand).
  - **Beyond L5:** no launch; report to the supervisor and the user.
- **Prior estimate at L4.** With K = 4 and S = 8, a per-stream cost of ≈ 1.2 s at 12 scaled by
  17 gives ≈ 160 s per cycle, ≈ 4.4 h local, ≈ 2.9 h pod per `d2` episode. Four episodes run in
  parallel on a 4 vCPU pod, so ≈ 3 h wall and ≈ 12 billed vCPU-h; stop 24. This figure is an
  extrapolation and is replaced by the probe.
- Every step that applies changes what is measured (L2 / L3 weaken the teacher). The record names
  the applied steps, and the readings carry them ("`d2`@K4S8").

**Code.**
- `teacher_audit --size 42 --reduction 42-L<n>` registers the sample and the ladder.
- The real-run options are `--candidate-cap` and `--stream-cap`, accepted only as the registered
  values of the chosen step.
- The existing refusals (above 42 producers, non-train, hit / switch) are unchanged.

### D8. Records and human gates

- **Durable records.**
  - One ADR (next free number, "Gate v3: promoted routers, region families, per-family side-effect
    parameters, per-order gate subject").
  - `ARCHITECTURE.md` §7.2.
  - The CLAUDE.md gate bullets, when v3 is packaged.
  - An EXP record per run.
  - `experiments/HISTORY.md` §15 "Gate v3", opened by the screen's record.
- **Human gates.**
  - The screen writes candidate frozen files into its EXP dir only.
  - Copying them to `eval/registry/*_v3.json`, and making v3 the packaged default, is a separate
    code task that waits for the user's approval (it raises the bar).
  - The sealed-12 screen and both gated reads wait for that approval.
  - The gated reads also wait for M4's `confirmed`.

### D9. The gated reads of the M4 candidate

These are planned here so the harness work is not repeated. They do not launch until M4's held-out
read is `confirmed` (EXP-0061) and the user has approved v3.

1. **Development report on v3**, both sizes. The deciding column (`entity_planner`) gives the
   verdict lines at 12 and 42; 42 is reported for M4. The reported column carries the EXP-0035
   producer. R = 5.
2. **Sealed 12-producer exit read** under a pre-registration manifest (D6 pins), after D5's sealed
   screen.

Whether the exit read is the sealed one (as `establish-project-baseline` D1 states) or the
development one is confirmed with the user when M4 reaches `confirmed`.

### D10. Spend

Estimates come from probes or prior EXP wall times. Every stop is 2 × the estimate and ≤ 40.

| Run | Shape | Estimate (billed vCPU-h) | Stop | Ground |
|---|---|---|---|---|
| v3 development screen | `cpu32` | ≈ 18 | 36 | EXP-0047 per-run cost, sizing.md §EXP-0051; task 5.1 probe |
| 42-producer teacher read | `cpu4`–`cpu8` by probe | ≈ 12 (prior, at L4) | 24 | §EXP-0053 × §EXP-0051 extrapolation; task 5.2 probe |
| Sealed-12 roster screen | `cpu16` | ≈ 2.5 | 5 | 192 world-episodes × ≈ 25 runs × 1.3 pod-s |
| Gated: development report (D9.1) | `cpu32` | ≈ 14 | 28 | EXP-0057 per-job cost, sizing.md §EXP-0051 |
| Gated: sealed-12 exit read (D9.2) | `cpu16` | ≈ 4.5 | 9 | as above, 192 world-episodes |
| Total | | ≈ 51 (≈ 33 ungated) | 102 | |

The supervisor holds M4's budget. Which budget gate v3 draws from is listed for the user.

## Risks / Trade-offs

- **[Two-wave pruning drops a stack that would win on f14]** → W2 keeps every wave-1 winner and
  every router with `entity_planner`. No non-`entity_planner` stack has won on 256 v1 / v2
  development world-episodes. The record lists the members not carried to wave 2, so a later
  screen can test them.
- **[Region worlds narrower than the atlas region]** (shocked = `shocks-i` + drift only; calm =
  cyclic only) → recorded as a limit. Burst, iid and seasonal demand stay atlas and M4 evidence.
  Widening the shocked regime would need a new versioned profile, a later change.
- **[The roster now holds MWW, a stochastic, input-aware router]** → the bar rises most on exactly
  the worlds where the deployed scorer trails (EXP-0051 addendum). That is the user's intent. The
  record reports the v2-roster bar beside the v3 bar on v2 worlds, so the size of the rise is
  visible.
- **[Per-family X / Y makes safety relative to each family's own baselines]** → on loaded worlds
  the starvation floor is lower than on sparse-wide. That matches what the baselines achieve
  there, and the evidence block records each family's values and counts.
- **[Region families are small]** (4 + 2 development worlds per size) → headroom and X / Y rest on
  32 world-episodes per region family. The counts are recorded; more worlds are a cost decision
  for the user.
- **[The teacher read may land at L5 or not launch]** → the probe decides before any spend, and
  every applied step is named in the readings. A read that cannot launch is a recorded result
  (the 42 teacher is unaffordable in this form), not a failure to diagnose.
- **[The teacher read's d2 is weaker at 42 than at 12]** (K / S caps) → it is a reference, read as
  a lower bound. The atlas ceilings stay the upper references.
- **[v3 seeds collide with an unlisted stream]** → the disjointness test covers every recorded
  world source (v2, atlas-v1, M4 fresh pool). A new stream added later must extend the test.

## Migration Plan

1. Code tasks land promotion, drafts, screen modes, per-order subject and teacher options. v1 / v2
   files and the default stay untouched, and tests assert their hashes.
2. Development screen → candidate frozen v3 files in its EXP dir → record → user approval.
3. On approval: package v3 (copy, set the default, update CLAUDE.md and ARCHITECTURE.md), then the
   sealed-12 screen.
4. **Rollback.** Before packaging, discard the EXP dir's `frozen/`. After packaging, the v2 files
   stay selectable by path (as v1 is today), and every earlier report keeps naming its versions.

## Open Questions

These are for the user. Defaults are applied in the specs and tasks.

1. **Region weight.** Region worlds are 1/3 of each (size, regime) group in the regime ratio
   (default). Alternatives: their own margin clause, or more worlds (cost ≈ +1.2 vCPU-h per extra
   42 f14 development world).
2. **Trained incumbents in the roster.** Should the "strongest policy" rule also cover the
   EXP-0012 trained orderer, which beats `round_robin` at 42 f14 under light load (EXP-0050)?
   Default: no, the roster stays hand-written.
3. **Per-family X / Y.** Default: per family (D4). The alternative is one pooled X / Y.
4. **Exit read.** Sealed 12-producer (default, per `establish-project-baseline` D1) or development
   v3 at 12.
5. **Budget line.** Which budget gate v3's ≈ 33 ungated vCPU-h draws from: M4's (supervisor
   authority) or a separate gate-change budget.
