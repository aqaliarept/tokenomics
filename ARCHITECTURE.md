# Training Architecture Guide

This document explains how the tokenomics project trains, measures, and
improves its decision-making models: what each component is, how it works
internally, and how the whole improvement pipeline fits together. It is
written for readers who understand ordinary software systems but may be new
to machine learning; every specialised term is defined the first time it is
used and again in [GLOSSARY.md](GLOSSARY.md). Terms about the simulated
economy itself (token, cycle, capacity, order, process sheet, system death)
are defined in [CONTEXT.md](CONTEXT.md).

The narrative follows the pipeline in execution order:

1. What is being simulated and what "good" means.
2. The two deployed decision-makers and how a decision is actually made.
3. The reference bar: the expensive search that defines "better is possible".
4. How models are trained (imitation, not trial-and-error).
5. How everything is measured (seeds, failure classes, transfer, ceilings).
6. The game-master pipeline: mining, attribution, probing, replay, closure,
   promotion — with one real round (EXP-0036) as the running example.
7. Infrastructure, records, and legacy components.
8. How the approach evolved (the experiment history).

Deeper references:

- [docs/rl-training-architecture.md](docs/rl-training-architecture.md) — the
  legacy reinforcement-learning layer in full detail (observation encoding,
  actor/critic networks, PPO). Still accurate for the components marked
  *legacy* below.
- [CURRICULUM_GAME_MASTER_APPROACH.md](CURRICULUM_GAME_MASTER_APPROACH.md) —
  the staged plan the current pipeline follows.
- [experiments/README.md](experiments/README.md) — the indexed record of
  every experiment.
- [docs/adr/](docs/adr/) — design decision records.

## 1. The Short Version

The project simulates a cooperative production economy. Consumers demand
products; producing agents own manufacturing capacity and assemble products
(possibly from other products, forming supply chains); ordering agents split
each order — and the tokens backing it — across the producers able to fulfil
it. Capacity wears down every cycle, so producers must keep investing in
restoration and upgrades or the economy collapses.

Two decisions are learned, one model per role:

- the **producer score model** scores a producer's options each cycle (which
  inputs to restock, whether to invest);
- the **orderer score model** scores each eligible producer as a destination
  for an order.

Neither model issues commands directly. Deterministic, hand-written code — a
*planner* on the producer side, an *allocator* on the orderer side — turns
scores into legal commands and enforces every hard rule. A bad model can make
poor choices, never illegal ones.

Training is **imitation** (called *distillation*): an expensive
search-based strategy demonstrates good decisions on specific worlds, and the
fast models are fitted to reproduce those decisions. The current best pair of
trained models is the **champion**. An automated loop — nicknamed the **game
master** — generates thousands of candidate worlds, finds the ones where the
champion verifiably underperforms the search, retrains on them, and
re-verifies everything. Replacing the champion is always a human decision.

Two facts about the simulation make all of this workable:

1. **The simulation is deterministic.** Same world + same starting seed +
   same policy = exactly the same episode, every time. Comparisons are
   therefore exact: two strategies can be run on literally the same sequence
   of events, and any difference in outcome is caused by their decisions.
2. **Worlds are generated.** Generator functions build economies of chosen
   size and shape from a seed. Training and evaluation deliberately use
   different worlds, which is how memorisation is told apart from skill.

## 2. What Is Being Optimised

Success is one number: **fulfilled consumer units** — how many units of
consumer demand were actually delivered over an episode (by default 100
simulated cycles). The experiment records abbreviate this to "units" or
`cons✓`. A secondary hard constraint is avoiding **system death**, the
irrecoverable state where some product can no longer be made by anyone;
death ends the episode immediately, so it usually shows up as a catastrophic
units number anyway.

A worked scale: on a typical generated depth-3 economy, doing nothing
delivers 0 units, the strongest hand-written strategies deliver roughly
450–550 units over 100 cycles, and the differences that experiments argue
about range from ~2 units (noise floor) to ~973 units (a world where the
champion dies and better routing keeps the economy alive).

### 2.1 Seeds: the only source of variation

Because the simulation itself is deterministic, seeds are the entire
vocabulary of variation. Three kinds appear everywhere in the records:

| Seed | What it controls | Example |
|---|---|---|
| World seed | Which economy the generator builds (sizes, capacities, recipes) | `build_scale_config(seed=77, depth=3, ...)` |
| Episode seed | Which demand schedule the consumers play out on that world | episode seed 77 = one exact 100-cycle demand trace |
| Eval seeds (a count) | How many consecutive episode seeds a measurement averages over | `eval_seeds=3` on episode seed 77 = average of episodes 77, 78, 79 |

The convention that evaluation seeds are **anchored** at the task's recorded
episode seed (77 → 77, 78, 79 — never 0, 1, 2) is load-bearing: evaluating
at different episode seeds means a different demand trace, which is a
*different task*. A bug of exactly this kind invalidated a probe run in
EXP-0036 and is now diagnostics-registry entry 10.

## 3. The Three Layers

The codebase keeps three layers strictly separated (the legacy guide covers
the first two in depth):

| Layer | Directory | Responsibility | Never contains |
|---|---|---|---|
| Domain | `tokenomics/tokenomics/domain/` | The economic rules: production, auctions, degradation, investment, death | Neural networks, reward logic, I/O |
| Environment | `tokenomics/tokenomics/env/` | Translating domain state to numbers for models and model outputs back to valid commands | Economic rules |
| Training & evaluation | `tokenomics/tokenomics/train/`, `eval/` | Models, teachers, distillation, baselines, the mining pipeline, all measurement | Domain rule changes |

One consequence worth spelling out: the domain layer is also what makes the
expensive search teacher possible. A live economy object can be **deep-copied
in about 3 milliseconds**, and a copy can be played forward without
disturbing the original. The search in Section 5 does exactly that hundreds
of times per cycle.

## 4. The Deployed Decision-Makers

### 4.1 The design split: learned scores, deterministic decode

Both roles use the same pattern:

```text
domain state
  -> entity view        (domain builds a structured, id-free summary)
  -> score model        (neural network: one score per option)
  -> planner/allocator  (deterministic code: scores -> legal command)
  -> domain command
```

The neural network expresses *preference*; fixed code owns *feasibility* —
inventory headroom, budgets, eligibility, integer quantities, conservation.
Why not let the model output commands directly?

| Alternative | Problem |
|---|---|
| Model outputs integer quantities | Networks output continuous values; independent rounding loses or invents quantity |
| Model outputs a single producer choice | Cannot split an order across producers |
| Model outputs percentages | May be negative, may not sum to 1 |
| Model outputs raw commands, validated after | Invalid commands must be rejected or repaired, creating silent behaviour differences |
| Scores + deterministic decode (chosen) | Fixed-size, learnable preference; decode is exact, legal, and testable on its own |

### 4.2 The producer side

Each cycle, each producer must decide: which input products to restock (and
how many), and whether to start restoration, a capacity upgrade, or an
inventory expansion.

**The entity view.** The domain builds a `ProducerEntityView`: a set of
*rows*, one per entity the producer cares about, each row a short list of
named facts.

| Row type | One row per | Example facts (of 15–18 per row) |
|---|---|---|
| capacity | capacity type owned | current, max, degradation, reserved, restore cost/effect, upgrade cost/effect, demand pressure |
| product | product it can make | open demand, recent completions/rejections, bottleneck ratio |
| input | input product its recipes need | inventory, outstanding inbound, blocked reservations, downstream demand pressure |
| global | (one row) | cut-off price, inventory units/limit/headroom, which investments already run |

Crucially, rows carry **no identifiers** — no "product 30" label, just the
facts. Rows arrive in a stable sorted order, and the model treats them
symmetrically, which is why the same trained model runs on an economy with 6
products or 24 (Section 4.4).

**The planner.** `plan_producer_command` (in `domain/producer_planner.py`)
converts scores into one command. Its hand-written scoring rules
(`score_producer_entity_view`) are transparent enough to state exactly:

```text
product urgency      = open demand x bottleneck pressure   (per makeable product)
input value          = downstream demand pressure + blocked reservations
                       + max(0, stock target 3 - (inventory + inbound))
restoration ROI      = capacity shortfall x restore effect / restore cost
upgrade ROI          = (demand pressure - max capacity) x upgrade effect / upgrade cost
                       (only when pressure exceeds the ceiling)
inventory expansion  = sum of input values, but only when headroom is 0
```

Decoding those scores, the planner:

1. requests restoration when any capacity type is below 60% of max (or its
   pressure exceeds free capacity) and no restoration already runs;
2. plans replenishment: each input wants `min(per-cycle ceiling, target −
   have)` units; spends limited inventory headroom on inputs in descending
   value; inputs with exactly tied value and need get equal integer amounts —
   if headroom cannot cover a tie group evenly, the remainder stays unspent
   rather than picking an arbitrary winner (ties must not be broken by row
   order, or the policy would depend on labels);
3. fires the upgrade/expansion toggles when their score is positive and the
   budget allows.

This planner is itself a strong baseline, called `entity_planner` in every
result table — and it is the producer half of the reference bar (Section 5).

**The learned producer model.** `ProducerScoreModel`
(`train/producer_scorer.py`) reproduces the planner's score *interface* but
computes the scores with neural networks: shared row encoders (one per row
type) plus *recipe edges* — sparse connections that let a product row see the
capacity and input rows its recipe actually uses. Its outputs are exactly the
score fields above, so it plugs into the same planner.

Two output kinds are trained differently, a distinction discovered the hard
way (EXP-0013):

- **continuous scores** (urgency, input value, replenishment) are fitted by
  regression — predict the teacher's number;
- **investment toggles** (upgrade, inventory expansion) are **learned binary
  gates** — trained to predict the teacher's yes/no *decision*, then fired by
  calibrated probability thresholds (upgrade 0.60, expansion 0.50) stored in
  the checkpoint.

Why gates: in the first attempt the toggles were regressed like everything
else, and the planner read tiny positive prediction noise (a predicted return
of 0.0001) as a hard "spend now". On a 24 000-decision audit the model fired
11 442 false upgrade commands. Regressing "how much is it worth" and
thresholding "should I act" are different questions; the gate heads answer
the second directly.

### 4.3 The orderer side

Each ordering agent serves one **product stream** — all orders from one
source for one product (ADR-0052). Each cycle it must split every order it
holds (quantity and tokens) across the producers eligible for that product.

**The view.** A `ProductStreamView` contains 9 product-level facts (open
demand, recent rejections, recipe complexity, eligible producer count, …) and
31 facts per candidate producer — capacity totals and ratios, current load,
cut-off price, whether investments are running, and observed track record
(recent routed/completed/rejected quantities, success rate, throughput,
completion latency). Again: facts only, no identifiers.

**The model.** `OrdererScoreModel` (`train/orderer_scorer.py`) is small and
symmetric:

```text
product facts (9)     -> 2-layer encoder (64 units) -> context vector
each producer row (31)-> shared 2-layer encoder (64) -> row vector
score(producer)       = linear head(row vector + context)
```

Because the same row encoder scores every producer independently, reordering
the producers reorders the scores identically (order-equivariance), and any
producer count works — there is no fixed-width output layer to outgrow.

**The allocator.** The scores are treated as **logits** — unconstrained
preference numbers where only differences matter. For each order the decoder:

1. masks out ineligible producers;
2. applies **softmax** (turns preference differences into proportions);
3. splits the integer quantity by those proportions using largest-remainder
   apportionment;
4. splits the tokens according to the resulting integer quantities;
5. drops zero-quantity pieces.

Worked example — an order for 10 units backed by 100 tokens, two eligible
producers, model scores `[2.0, 0.0]`:

```text
softmax([2.0, 0.0])   = [0.88, 0.12]
quantity split        = [9, 1]         (largest remainder keeps the total exact)
token split           = [90, 10]
guaranteed: quantities sum to 10, tokens sum to 100
```

The allocator (`env/allocator.py`) owns those conservation guarantees; the
model never can violate them.

### 4.4 Portability, checkpoints, and the champion

Both models are **entity-factored**: input is a variable-length table of
rows, not a fixed snapshot of one economy. Proven consequences (arc 4 of the
history): the same checkpoints run unchanged on economies twice as wide, on
supply-chain *shapes* never trained on (fork/join recipes), and are exactly
invariant to relabelling or reordering entities — a property verified by
byte-identical-episode tests, not assumed.

A **checkpoint** is a saved file of learned parameters plus a **schema
block**: the named list of features, entity types, and decoder version it was
trained against. Loading verifies the schema by name, so a silent feature
drift refuses to load instead of quietly misreading inputs.

The **champion stack** is one producer checkpoint + one orderer checkpoint.
As of 2026-07-04: the EXP-0035 producer + the EXP-0012 orderer.

## 5. The Reference Bar: How the Search Teacher Acts

Everything in the improvement pipeline is measured against a **reference
bar**: a strong, checkpoint-free, deterministic-given-seed strategy. The
standard bar is called `entity_planner+mpc_search`: the producer side is
fixed to the rule-based planner of Section 4.2, and the orderer side is
chosen fresh every cycle by *search*.

### 5.1 The per-cycle search, step by step

At the start of each cycle, the search builds a candidate set of routing
policies (`_candidate_list` in `eval/orderer_oracle.py`):

- 2 named baselines: `even_split` (equal shares) and `proportional`
  (shares proportional to each producer's current makeable throughput);
- 8 sampled routings: random preference vectors drawn from a Gaussian
  (spread 1.0), each held fixed for the cycle. The random generator is
  seeded by `(episode seed, cycle number)`, so the "random" candidates are
  exactly reproducible.

Then, for each of the 10 candidates:

1. **deep-copy the live economy** (the ~3 ms clone from Section 3);
2. play the copy forward **6 cycles** (`lookahead=6`) with the planner
   producer and the candidate orderer;
3. score the candidate by **consumer units delivered inside that window**.

The winning candidate is committed for the real cycle; next cycle the whole
search repeats from the new state. One episode of the bar therefore looks
like:

```text
cycle 12 begins
  clone economy, try even_split      -> 31 units over cycles 12-17
  clone economy, try proportional    -> 29 units
  clone economy, try sample s0..s7   -> 27, 34, 30, 22, 33, 28, 31, 26
  commit sample s1 (34 units) for cycle 12
cycle 13 begins: search again from the actual cycle-13 state
```

### 5.2 What this costs, and why it is not the product

Per real cycle the search simulates up to `10 candidates × 6 cycles = 60`
extra cycles. A 100-cycle episode costs ~6 000 simulated cycles instead of
100 — roughly **60× the compute of a deployed model**, and the gap widens
with economy size. That is fine for a measuring stick and a teacher run
offline on rented machines; it is not a deployable policy.

### 5.3 Why the bar is trusted

- **Legal**: it is checkpoint-free and deterministic given the seed — a rule,
  not a trained artifact, so it cannot "overfit" to the archive.
- **Strong where it matters**: re-choosing the routing every cycle from live
  state reaches combinations no single fixed heuristic can (this was proven
  before the bar was adopted: on some worlds even a *random* orderer beat the
  standard heuristic, showing routing headroom existed).
- **Honest**: it is also *bounded* — it is a myopic 6-cycle argmax, not an
  oracle. When the champion got close to it, the project computed analytic
  ceilings (Section 7.5) to check whether the residual gap was real headroom
  or bar weakness. Answer, on that world family: neither — demand itself was
  nearly exhausted.

Two refinements were tried and settled empirically:

| Variant | Finding |
|---|---|
| Longer lookahead (12 cycles) | A 3-cycle window already ranks candidates almost identically at depth 3 — short rollouts suffice, 4× cheaper (EXP-0009) |
| Replace the rollout tail with a learned value estimate | Rejected: the value model memorised its 6 training economies and transferred to new ones with negative accuracy (EXP-0009) |
| Distill over the whole set of near-best candidates, not just the argmax | Rejected: over-spread the student's routing and lost the high-value decisive choices (EXP-0008) |

## 6. How Models Are Trained: Distillation

The project does **not** train its skill models by trial-and-error
reinforcement learning (see Section 10 for what remains of that path and the
history for why it was abandoned). Training is supervised imitation: collect
situations, ask a teacher what it would do, fit the model to answer the same.

### 6.1 Producer distillation

Teacher: the deterministic planner's own transparent scores
(`score_producer_entity_view`). Collection drives episodes, snapshots every
`ProducerEntityView` the producer faces, and records the planner's scores for
it. Fitting minimises the prediction error on the continuous scores and the
classification error on the two gate heads (Section 4.2). The result is a
model that *matches* the planner in-family — the value of the learned version
is that its continuous heads generalise, and that it can then be moved
*beyond* the planner by replay training on mined worlds (Section 8.7).

### 6.2 Orderer distillation, plain form

Teacher: the `proportional` heuristic's signal. For each product-stream view
the target score for producer *j* is:

```text
target_j = log(units producer j could make right now), clipped to [-10, 10]
         = -10 for ineligible or zero-throughput producers
```

The logarithm matters: the allocator softmaxes scores, and softmax of
log-throughput reproduces proportional-to-throughput routing exactly. The fit
is **masked mean-squared error** — only eligible producers' scores
participate in the loss, so the model is never punished on rows the allocator
will mask anyway.

### 6.3 Orderer distillation, oracle-relabel form

Used when the headroom being chased is the *search's* advantage (Section 5),
which the proportional teacher by definition cannot demonstrate. Collection
(`collect_oracle_orderer_samples`) replays a specific archived world with the
champion producer while the **search chooses the routing each cycle**, and
records the search's committed routing as the target. Two normalisations make
a committed routing usable as a training target:

- routings that differ across an agent's backlog slots are averaged over the
  occupied slots (the deployed model emits one routing per stream);
- the eligible scores are shifted to zero mean and clipped — softmax only
  reads differences, so centring removes meaningless scale without changing
  the induced routing.

The same masked-MSE fitting then applies unchanged. This is the machinery the
EXP-0036 representability probe runs per-task (Section 8.8).

### 6.4 DAgger

Plain imitation has a compounding-error failure mode: the student is trained
only on states the *teacher* visits, drifts slightly at deployment, and finds
itself in states it was never taught. **DAgger** fixes this by letting the
student drive the episodes and asking the teacher to label the states the
student actually reaches. The early producer-clone work (arcs 1–2) depended
on it; the current replay pipeline composes the same relabel/train machinery.

### 6.5 Two training rules written in scar tissue

- **Augment, don't replace.** Retraining a model *only* on newly mined worlds
  erases skills it already had (demonstrated twice: EXP-0026/0027 lost
  four transfer families that way). The rule: training data = the original
  broad set *plus* the mined additions.
- **Train to convergence before judging the architecture.** EXP-0034's
  candidate captured ~16% of the mined headroom and looked like a
  representational wall; the diagnosed cause was under-training (one model ×
  121 worlds × 400 passes averages away per-task behaviour). Raising to
  2 000 passes took the capture to ~100% with no architecture change
  (EXP-0035).

## 7. Evaluation: How Anything Gets Believed

### 7.1 Paired comparison on identical episodes

Every comparison runs both strategies on the *same* world and the *same*
anchored episode seeds (Section 2.1). With `eval_seeds=3`, a task also
reports the **win fraction**: on how many of the 3 paired episodes the bar
beat the learner. A mean gap with win fraction 1/3 is a coin flip; 3/3 is a
reproduced weakness. Both floors are enforced before anything is mined
(Section 8.4).

### 7.2 The baseline ladder and the envelope

Hand-written strategies (`eval/baselines.py`) serve as floors and bars:

| Role | Baselines (weakest → strongest) |
|---|---|
| Producer | `noop`, `random`, `always_restore`, `restore_when_low`, `stock_aware`, `entity_planner` |
| Orderer | `random`, `even_split`, `proportional` |

The **envelope** is the per-world maximum over baseline combinations
*including random ones*. Clearing "the strict envelope" means beating
whichever baseline happens to be best on each held-out world separately — a
much harder claim than beating any single named baseline, and the standard
the champion orderer line had to meet (EXP-0006B was the first to do it).

### 7.3 Failure classification

Every below-bar result in the curriculum is assigned exactly one class, by
precedence (`classify_failure` in `eval/curriculum.py`):

| Precedence | Class | Meaning | Consequence |
|---|---|---|---|
| 1 | `contract_failure` | Checkpoint could not load/run on this world (an exception) | Fix the contract; never counts as skill |
| 2 | `implementation_failure` | The episode crashed or produced NaN | Fix the bug |
| 3 | `feasibility_wall` | A world labelled impossible, and both learner *and* bar fail on it | Excluded from skill scores (e.g. depth-5 chains vs the 2-cycle order TTL) |
| 4 | `distribution_mismatch` | The world lies outside what the champion is claimed to handle | Scored, tracked, but not counted as an in-claim weakness |
| 5 | `policy_weakness` | In-claim world, ran fine, bar beat the learner | **The only class that feeds training** |
| 6 | `none` | Nothing wrong | — |

This is the mechanism that keeps "it crashed", "it can't be done", and "it
played badly" from ever being averaged into one misleading number.

### 7.4 The transfer and stress battery

`eval/transfer_matrix.py` and the stress suite run a stack across held-out
scale worlds, wider economies (more products / more producers), out-of-family
recipe shapes (fork/join, shared capacity, uneven strength), five scarcity
axes, and degenerate edge cases — ~18–30 cells per run — reporting contract
flags separately from value in every cell. Any candidate must come through
this battery with zero contract failures and no regressions before promotion
is even discussed.

### 7.5 Ceiling bounds: is there anything left to win?

`eval/ceiling_bound.py` computes analytic upper limits on the units *any*
policy could deliver on a world, from arithmetic alone:

| Tier | Bound |
|---|---|
| B0 | The exact scripted demand on those episode seeds — you cannot deliver more than was ordered |
| B1 | B0 capped by raw capacity throughput per recipe |
| B2 | B1 with recipes expanded transitively through their input products |

Each tier is deliberately optimistic (capacity pinned at max, no TTL or
funding constraints), so every tier must sit *above* every observed result —
a built-in sanity invariant. Its decisive use (EXP-0033): on the mined
archive of the day, demand itself was the binding tier on 26 of 26 tasks and
the champion was already at 98.3% of it. Translation: the remaining gap was
not trainable headroom, and mining had to move to a different world family —
which is exactly what Track A did.

## 8. The Game-Master Pipeline, End to End

This section walks one real round — EXP-0036, "Track A round 2" — through
every stage, with the actual numbers. The pipeline lives in
`eval/curriculum*.py`, `eval/gap_attribution.py`,
`eval/orderer_representability.py`, and `eval/train_closure.py`.

### 8.1 The task record: a world in one line

Everything downstream operates on `TaskRecord`s — one JSON line each in a
**task archive**. A record stores a *recipe for rebuilding the world*, never
a dump of it:

```text
task_id:        mine-build_scale_config-capacity_scale0.6-depth3-
                primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77
world_spec:     {"generator": "build_scale_config",
                 "params": {"depth": 3, "capacity_scale": 0.6,
                            "primal_capacity_scale": 0.22,
                            "producers_per_layer": 3, "quantity_max": 14}}
world_seed:     77          episode_seed: 77
expected_claim_boundary: in_claim
notes:          "mined: score=1.03 gap=973.3u win=1.0"
```

`replay(record)` calls the named generator — only vetted, pure,
seed-deterministic builders are allowed, by whitelist — and reproduces the
exact economy. Champion, candidate, and bar all replay the same world; the
record is the single source of truth for "which task was this".

### 8.2 Choosing the regime, and preflight

A mining round starts from a diagnosis, not a dice roll. Round 2's diagnosis:
round 1 had shown the champion trades away the skill "upgrade the tight
raw-material source first", so round 2 targets worlds whose capacity ceiling
binds at the **primal layer** (layer 0 — the raw-material producers), with
upper layers left loose.

Before any compute is spent mining, **preflight**
(`eval/curriculum_capacity.py`) checks the candidate grid actually has the
property: for each grid cell and every world seed, a no-slack arithmetic
check confirms which constraint binds. EXP-0036's grid — 4 cells varying
primal capacity scale 0.22–0.30, producers per layer 2–3, max order quantity
14–20, one high-degradation cell — passed 4/4 with the binding constraint at
`capacity-0` on every seed. Worlds that would silently test the wrong regime
never reach the miner.

### 8.3 Mining: finding verified weaknesses

`eval/curriculum_mining.py` samples the pool — here 4 cells × world seeds
20–119 = **400 tasks** — and for each task, replays the world and runs both
sides on the same anchored episode seeds (3 per task):

- the **learner**: the champion stack;
- the **bar**: `entity_planner+mpc_search` (Section 5).

Each task becomes a row: mean units for both sides, `gap_units` (bar −
learner), the paired-seed `win_frac`, and a failure class (Section 7.3).

### 8.4 Gates: separating weakness from luck

A row survives only if **all** of these hold (`passes_gates`):

```text
failure_class == policy_weakness      (in-claim, ran clean, bar won)
gap_units     >= 2.0                  (absolute floor)
win_frac      >= 0.6                  (bar won >= 2 of 3 paired episodes)
```

The two floors exist because the bar itself samples candidate routings — a
~2-unit gap on one seed can be search luck. When mining first ran at scale
(EXP-0025), the floors visibly flipped the ranking: the largest raw gap in
the pool (4.67 units, won 1 of 3) was *dropped*, while a smaller but
3-of-3-reproduced gap topped the archive. In EXP-0036, 400 tasks → 96
survivors.

### 8.5 Ranking: big gaps, but spread across world types

Survivors are archived greedily by a two-term score:

```text
score = 1.0 x normalized_gap + 0.5 x novelty

normalized_gap = gap / what the bar could plausibly deliver (its completed
                 + rejected units) - so a 50-unit gap on a small world
                 outranks a 50-unit gap on a huge one
novelty        = d / (1 + d), d = distance from the task's shape descriptor
                 (depth, products, producers, capacity types) to the nearest
                 already-archived one - first pick of a new shape scores 1.0,
                 a duplicate shape scores 0
```

After each pick the chosen shape counts as archived, so the next pick prefers
a fresh cell. This is what keeps the archive from collapsing onto one world
type — the exact failure that poisoned the first replay attempt. EXP-0036
archived **81 tasks** across all 4 cells (26/35/2/18), with gaps median 53,
mean 99.9, maximum 973.3 units — an order of magnitude beyond round 1's mean
of 3.67.

Mining fans out across processor cores as independent single-threaded shards
(one process per core, one cell × seed-range each); the parent merges rows
and ranks globally, so a parallel mine ranks identically to a sequential one.
If any shard dies, the whole run aborts rather than silently ranking a
biased partial pool.

### 8.6 Gap attribution: which role owns the headroom

For each archived task, `eval/gap_attribution.py` replays **four stacks** on
the same seeds:

| Stack | Producer | Orderer |
|---|---|---|
| champion | champion model | champion model |
| full bar | planner | per-cycle search |
| producer cross | planner | champion model |
| orderer cross | champion model | per-cycle search |

Substituting one side at a time measures how much of the gap each role can
recover alone:

```text
producer_only_gain = producer-cross units - champion units
orderer_only_gain  = orderer-cross units  - champion units
```

and each task gets a **dominant channel** label: producer, orderer, mixed,
or interaction/noise. Worked example, the hardest EXP-0036 task (world seed
77 above):

```text
champion:        0 units    (the economy dies - routing starves the primal layer)
full bar:        973.3
champion producer + search orderer: 997.3
=> orderer_only_gain = 997.3: better routing ALONE prevents the collapse;
   the channel is orderer.
```

Across the round-2 archive: 45 orderer / 15 mixed / 11 producer / 10 noise;
of 8 088 total gap units, the orderer cross recovers 7 172 (89%) — the exact
inverse of round 1, where 70% was producer-owned. This one table decides
where the retraining budget goes. The attribution also re-runs the *previous*
round's archive as **retention**: 54 of round 1's 113 tasks showed no
positive gap left, confirming the newly promoted champion had genuinely
absorbed round 1 rather than dodged it.

### 8.7 Split, priorities, replay training

Before any training, the archive is split — train / validation / blind
(50/25/25 by default) — by a seeded shuffle *independent of the ranking*, so
a spectacular task is as likely to land in blind as in train. The blind split
is never touched by any training decision; it exists to answer "did this
generalise?" at the end.

Replay training (`eval/curriculum_replay.py`) then rebuilds the indicated
model:

- training data = the champion's **original broad training worlds plus** the
  mined train split (the augment rule, Section 6.5);
- mined tasks are drawn with replacement, weighted by
  `(0.01 + gap)^eta` — bigger verified gaps get replayed more often, the
  floor keeps small-gap tasks from starving, and a per-task cap stops one
  monster from monopolising the batch. (An earlier version weighted by the
  mining score, which includes novelty; one high-novelty task dominated a
  whole round. Replay now weights by gap only.)
- the candidate is trained in the champion's exact checkpoint format, so
  every downstream comparison is apples-to-apples.

### 8.8 The representability probe: check before you spend

When a round's channel calls for distilling *search behaviour* (Section 6.3),
a cheap question comes first: **can this model architecture express that
behaviour at all?** `eval/orderer_representability.py` answers it per task:

1. run the search on the task's own world and collect its per-cycle routing
   labels (thousands of samples — 5 127 on the world-seed-77 task);
2. distill a *fresh* orderer model on exactly those labels;
3. deploy it on the same task and compute
   `recovered_vs_teacher = (oracle - champion) / (search - champion)`.

A model that cannot reproduce its teacher *on the training task itself* will
not be rescued by more data — that outcome ends the lever and forces an
architecture discussion instead of a doomed full-scale run. (As of this
writing, EXP-0036's probe is running; its first completed task returned
`recovered_vs_teacher = 0.0` — the distilled model also loses the economy —
an early signal in exactly the direction this probe exists to catch cheaply.)

### 8.9 Closure: how much of the promise was collected

After replay training, `eval/train_closure.py` scores the candidate on each
split:

```text
closure = (candidate - champion) / (bar - champion), summed over tasks
```

1.0 means the candidate absorbed everything the bar demonstrated; 0 means
nothing. Per-task ratios are noise when the mined gap is small, so the
headline number is aggregate (sums), with per-task rows for digging. The
EXP-0034 → EXP-0035 pair shows the metric doing its job: train closure 0.159
diagnosed a problem, the epochs fix moved it to 1.096 train / 0.860
validation / 1.000 blind, and the worst blind-task regression (−107 units)
healed to +61.7.

### 8.10 Gates and promotion

A candidate must clear, at minimum:

- closure gains on mined validation **and** blind splits;
- the full transfer/stress battery (Section 7.4) with zero contract failures,
  zero new deaths, no regressions — including byte-identical behaviour on
  world families its training never touched;
- a tail gate: no single blind task regressing beyond a conservative floor.

The pipeline emits an advisory recommendation; **a human makes the promotion
call**, recorded with date and rationale in the experiment index. The
mechanism is deliberately conservative — EXP-0030 was promoted by explicit
human override *after* follow-up experiments showed its advisory gate miss
was measured against a saturated archive, and EXP-0035 passed every gate and
was promoted the ordinary way.

### 8.11 One round, in one table

| Stage | Tool | EXP-0036 outcome |
|---|---|---|
| Regime + preflight | `curriculum_capacity` | 4/4 cells bind at primal capacity, all seeds |
| Mine 400 tasks | `curriculum_mining` | 96 gated survivors → 81 archived, gaps 2.7–973 u |
| Attribute | `gap_attribution` | 89% of headroom orderer-owned; 6 champion-death tasks |
| Probe | `orderer_representability` | running; task 1: recovered 0.0 of a 997-u teacher gap |
| Replay + closure + gates | `curriculum_replay`, `train_closure` | pending the probe verdict |
| Promotion | human | — |

## 9. Infrastructure and Records

### 9.1 Cloud execution

All real training and every experiment run on rented cloud machines (Runpod);
only seconds-long smoke checks and one-shot diagnostics run locally. The
driver `run_pod_training.py` owns the full lifecycle: create/start machine →
sync code → install → launch **detached** (the run survives disconnects and
laptop sleep; a client can re-attach) → mirror live logs → pull result
directories back → **always stop the machine**, even on error or interrupt,
so idle time is never billed. Classification of what counts as "real" is by
compute profile and purpose, never by what a run is named — a rule with its
own failure story recorded in CLAUDE.md.

### 9.2 The records system

The project treats its own history as part of the architecture:

| Record | Purpose |
|---|---|
| `experiments/EXP-NNNN-*/` | One directory per experiment: question, exact config (`spec.json`), results (`summary.csv`), conclusion (`RESULT.md`), artifacts via git-lfs. The index in `experiments/README.md` is the timeline. |
| `docs/adr/` | 54 numbered design decisions with reasoning |
| `docs/DIAGNOSTICS.md` | Symptom → discriminating check registry; recurring problems get recognised, not re-debugged |
| `PROGRESS.md` | Running work log; every completed task appends an entry |
| `VERIFICATION.md` | Open questions, linked to the experiment that answers each |

One standing rule binds them: **root cause before next strategy** — a failed
or underperforming run must have its cause diagnosed with evidence (a trace,
an ablation, a split — not a hypothesis) and written down before any next
strategy is proposed. The EXP-0034→0035 sequence is this rule working: the
tempting response to 16% closure was "mine more"; the diagnosed cause was
under-training; the one-knob fix recovered ~everything.

## 10. Legacy Components

Kept in the tree, not on the active skill-training path. The legacy guide
([docs/rl-training-architecture.md](docs/rl-training-architecture.md))
documents the first three in full depth.

| Component | What it was | Status |
|---|---|---|
| `train/smoke_ppo.py`, `train/parallel.py` | The original trial-and-error PPO trainer: shared producer/orderer actor networks, centralised critic, GAE, reward shaping | Environment smoke checks only |
| `train/boundary_grid.py` | The capability-mapping driver (which depths × demand levels PPO could learn) | Historical; produced EXP-0001 |
| `train/warm_start.py`, `train/clone_fidelity.py` | The original behaviour-cloning + DAgger tooling for the flat (non-portable) producer | Superseded by scorer distillation; DAgger machinery still composed by replay |
| `train/value_model.py` | Learned units-to-go estimator meant to shorten the search rollout | Rejected (no cross-world generalisation, EXP-0009) |
| `train/producer_improve.py`, `train/orderer_improve.py`, `train/champion_iterate.py` | Improvement and automation drivers for the pre-portable champions | Superseded by the Section 8 pipeline |

## 11. How We Got Here — the Experiment History

Fifty-plus experiments, readable as six arcs; per-experiment detail is in
[experiments/README.md](experiments/README.md).

### Arc 1 — Finding the wall (EXP-0001 … 0004)

Classic PPO was mapped across economy sizes. Supply-chain **depth**, not
order quantity, is where it failed — depth-2 trains, depth-3 collapses — and
the failing role was the **producer** (restoration misallocation starves the
middle of the chain). Cloning a decent hand-written producer into a network
worked only after discovering the naive loss was wrong for gated actions;
staged DAgger produced a faithful clone. Producer-only PPO on top of the
clone went nowhere — the critic's predictions explained ~none of the outcome
variance, i.e. the learning signal was empty — and the project pivoted to
imitation of strong teachers.

### Arc 2 — Beating the baseline via the orderer (EXP-0005 … 0009)

A headroom audit showed the weakness was orderer-side: even random routing
sometimes beat the standard heuristic. The per-cycle search (Section 5)
turned that into teacher labels; distilled students beat the named baseline
by double digits. Making the distilled orderer **stochastic** (sampling its
routing) cleared the strict envelope on every held-out world — the first
model in the project to do so. The learned value shortcut for the search was
rejected (memorised its training economies); the surviving byproduct was that
short rollouts suffice.

### Arc 3 — The portable rebuild (EXP-0010 … 0014)

Early models were welded to one economy's exact size. This arc rebuilt both
roles entity-factored behind deterministic safety layers: the transparent
planner (`entity_planner`), then learned score models distilled from strong
teachers. Portable orderer ≈1.28× the routing heuristic on held-out worlds;
portable producer ties the planner; the joint stack ≈1.11× the reference
stack, zero deaths. The EXP-0012 orderer checkpoint from this arc is still
the active champion orderer.

### Arc 4 — Proving portability (EXP-0015 … 0022)

A ladder of evaluation-only experiments: the transfer matrix (contract vs
policy separated per cell); size transfer (24-product / 24-producer worlds,
zero failures, value kept); out-of-family recipe shapes (fork/join — the
contract held, one producer policy gap found); audits against every
deterministic strategy; a training-coverage broadening that closed the found
gap; and ablations that quantified which architectural pieces earn their keep
(the recipe edges buy value only under scarcity; a flat fixed-size network is
not portable at all, by construction). Remaining weaknesses were named, not
hidden.

### Arc 5 — The game master, and a real ceiling (EXP-0023 … 0033)

The Section 8 pipeline was built stage by stage: replayable task records and
failure classification first; then the discovery that the existing reference
was too weak to expose any weakness — fixed by adopting the search bar.
Mining at scale worked (the noise floors proved load-bearing); the replay
experiments produced the augment-don't-replace and archive-diversity lessons.
The arc ended with a high-quality null: after absorbing everything mineable,
the ceiling bound showed the champion at ~98% of what is mathematically
achievable on that world family. The family was saturated; the mining
distribution itself had to change.

### Arc 6 — Capacity-binding regimes, the current front (EXP-0034 … 0036)

Mining moved to worlds where capacity, not demand, binds — verified by
preflight. Round 1: large producer headroom found; first retrain captured
only ~16% of it; root-caused to under-training; 5× more epochs captured
~100% (EXP-0035, promoted 2026-07-04 — the current champion producer).
Round 2 (EXP-0036): worlds bottlenecked at the raw-material layer; headroom
is large and 89% orderer-owned, including six worlds where the champion's
routing collapses the economy and search routing alone prevents it. The
representability probe (Section 8.8) is in flight; its verdict decides
between a full orderer oracle-relabel round and an architecture discussion.

## 12. Current State (2026-07-04)

- **Champion stack**: EXP-0035 producer score model + EXP-0012 orderer score
  model (human-approved promotion, 2026-07-04).
- **Active work**: EXP-0036 probe verdict pending; next decision gated on it.
- **Test suite**: ~608 tests, kept green; environment and pipeline changes
  land with tests first.
