# Glossary — training, evaluation, and experiment terms

Plain-English definitions for the terms used in [ARCHITECTURE.md](ARCHITECTURE.md)
and throughout the experiment records. **Domain terms** — the vocabulary of the
simulated economy itself (token, cycle, capacity, order, process sheet, system
death, …) — are defined in [CLAUDE.md](CLAUDE.md) under *Domain Language*.

---

## Models and strategies

**Policy** — any decision-maker that can play a role in the simulation, whether
hand-written rules or a trained neural network. "The producer policy" means
whatever is currently making producer decisions.

**Baseline** — a hand-written, non-learned policy used as a comparison point.
They range from deliberately dumb (`noop`, do nothing) to strong
(`entity_planner`). A learned model earns nothing until it beats the relevant
baseline.

**Planner** (producer side) — the deterministic, rule-based producer decision
layer (`entity_planner`). Used both as a strong baseline in its own right and
as the safety layer that turns a learned model's scores into legal commands.

**Allocator** (orderer side) — the deterministic layer that turns an orderer
model's per-producer scores into a legal split of an order and its tokens,
enforcing eligibility and conservation.

**Score model / scorer** — the kind of neural network this project trains. It
does not output commands; it outputs numeric *scores* ("how valuable is this
option right now"). The planner or allocator reads the scores and issues the
actual commands. There are two: the producer score model and the orderer score
model.

**Entity view** — the structured summary the domain hands a model: a table of
*rows* (one per capacity type, product, or input the agent cares about), each
row a fixed list of named numeric facts, with no identifiers. The producer
side uses `ProducerEntityView`; the orderer side uses `ProductStreamView`.

**Product stream** — the unit an ordering agent serves: all orders from one
source for one product (ADR-0052). Every order in a stream shares the same
eligible producers, so one set of per-producer scores routes the whole
backlog.

**Logit** — an unconstrained preference score; only differences between
logits matter. **Softmax** turns a set of logits into proportions (e.g.
`[2.0, 0.0]` → `[0.88, 0.12]`); adding the same constant to all logits
changes nothing. The orderer's scores are logits, softmaxed per order over
the eligible producers, then converted to exact integer splits by
largest-remainder apportionment.

**Gate head** — a model output trained to predict a yes/no decision
(classification) rather than a value (regression), fired by a calibrated
probability threshold stored in the checkpoint. Used for the producer's
upgrade and inventory-expansion toggles after regression noise caused
thousands of false "invest now" commands (EXP-0013).

**Masked loss** — a training error that only counts the outputs that matter
in that situation (e.g. only eligible producers' scores); the model is never
punished on rows the decoder will discard anyway.

**Entity-factored / portable** — the property that a model reads the economy
as a variable-length table of entities (products, capacity types, producers)
rather than as one fixed-size snapshot. A portable model loads and runs on
economies of different sizes and shapes than it was trained on, without
retraining. "Id-free" means the same thing seen from another angle: the model
attaches no meaning to *which* slot an entity sits in, so renaming or
reordering entities cannot change its behaviour.

**Checkpoint** — a file containing a trained model's learned parameters. "The
champion producer checkpoint" is literally a file that can be loaded and run.

**Stack** — a complete pair of policies, one producer + one orderer, evaluated
together. "The champion stack" = the current best pair.

**Champion** — the incumbent best stack. Never replaced automatically; see
*promotion*.

**Candidate** — the stack (or single model) a training run produces, which
must prove itself against the champion.

**Teacher / student** — imitation-learning roles. The teacher is the strong
(usually expensive or hand-written) policy whose decisions are recorded; the
student is the neural network trained to reproduce them.

**Search / `mpc_search`** — the expensive teacher-and-measuring-stick: at every
cycle it scores 10 candidate routings (2 named heuristics + 8 reproducibly
sampled preference vectors) by cloning the live economy and playing each copy
forward, then commits the winner for that one cycle. Too slow to deploy (~60×
the cost of a deployed model); excellent for defining what better play is
possible. See ARCHITECTURE.md §5.

**Lookahead / rollout** — playing a cloned copy of the economy forward a fixed
number of cycles under fixed policies to score a candidate decision. The
standard bar uses a 6-cycle lookahead; a 3-cycle window ranks candidates
almost as well at depth 3 (EXP-0009), which is why the search stays
affordable.

**Bar / reference bar** — the fixed strong policy a champion is measured
against on every mined world. The standard bar is `entity_planner+mpc_search`
(planner producer + search orderer). "Beating the bar" or "the gap to the bar"
always refers to this reference.

---

## Training methods

**Distillation** — training a model to reproduce a teacher's decisions on
recorded situations (a form of imitation learning). The project's primary
training method.

**DAgger** — an imitation-learning refinement: instead of only showing the
student the teacher's own play, let the *student* drive episodes and have the
teacher label the situations the student actually reaches. Fixes the
compounding-error problem where a student drifts into states it was never
taught about.

**Oracle relabeling** — running the expensive search teacher on specific worlds
and using its per-cycle choices as fresh training labels, rather than labels
from a cheap heuristic. Used when the headroom being chased *is* the search's
advantage.

**Augment (vs replace)** — the rule that retraining data must be the model's
original broad training set *plus* the new material, never the new material
alone. Training on mined weak spots only ("replace") erases previously learned
skills; this was demonstrated empirically twice before becoming a rule.

**Epochs / iterations ("iters")** — how many passes the training procedure
makes over its data. Under-training (too few) makes a fixable model look
architecturally broken; the arc-6 fidelity-wall fix was simply 5× more epochs.

**Warm start / cloning** — the older (arc 1–2) form of distillation: initialise
a network by copying a hand-written policy's behaviour before any further
training. Kept in tree as legacy tooling.

**PPO** — the classic trial-and-error reinforcement-learning algorithm the
project started with (learn by rewarding good episodes). Abandoned as the
skill-training path after its learning signal proved too weak on this problem;
still used for environment smoke checks.

**Critic** — in PPO, the helper network estimating expected future reward; its
"explained variance ≈ 0" (it predicted nothing) was the concrete evidence that
PPO had no learning signal here.

---

## Evaluation and measurement

**Fulfilled consumer units ("units", `cons✓`)** — the headline success metric:
how much consumer demand was actually delivered over an episode. Almost every
comparison in the records is in units.

**Seed** — a number that fixes a random generator's output, making runs
reproducible. Three distinct kinds appear in the records: a **world seed**
(which economy the generator builds), an **episode seed** (which demand trace
plays out), and **eval seeds** (how many consecutive episode seeds a
measurement averages over). The simulation itself is deterministic, so seeds
are the *only* source of variation — which is why seed conventions are
load-bearing (see `docs/DIAGNOSTICS.md` entry 10).

**Topology** — the size and shape of a generated economy: how many products,
how many producers, how deep the supply chains are, how recipes connect.

**Held-out** — worlds (or seeds) deliberately excluded from training and used
only for evaluation, so results measure learning rather than memorisation.

**Blind split** — the strictest tier of held-out: a subset of mined worlds set
aside before training that no training decision ever touched, used for the
final honesty check. (Mined archives are split train / validation / blind.)

**Transfer** — how well a model trained on one set of worlds performs on
different ones: bigger ("size transfer"), differently shaped ("out-of-family"),
or stressed (scarcity, degenerate cases).

**Transfer matrix** — the standing regression battery that runs a stack across
all those world families and reports, per cell, contract flags separately from
value.

**Contract failure (vs policy failure)** — the crucial reporting distinction:
a *contract* failure means the model crashed, emitted an illegal command, or
could not load on a world; a *policy* failure means it ran fine and played
badly. The two are never mixed in a result table.

**Hindsight baseline** (formerly *envelope*; older EXP records use that word) — the per-world best of *all* baselines combined (including the
random one). "Clearing the strict hindsight baseline" means beating, on every held-out
world, whichever baseline happens to be best there — a much harder test than
beating any single baseline.

**Gap** — how many units the reference bar beats the champion by on a given
world. The raw material of mining: a positive, reproducible gap = trainable
headroom.

**Headroom** — improvement that is provably available: some existing policy
(usually the bar) already achieves it, so a learner could in principle absorb
it. Distinguished from improvement that would first require inventing stronger
play than any reference exhibits.

**Win fraction** — on how many of a task's paired evaluation episodes the bar
beat the learner (e.g. 2 of 3 = 0.667). A large mean gap with a low win
fraction is a coin flip, not a weakness.

**Noise floor / paired-seed floor** — the mining admission rules that separate
real weakness from luck: the gap must exceed a minimum size (`min_gap`, 2.0
units) *and* the win fraction must reach a minimum (`min_win_frac`, 0.6).
Added after early mining showed coin-flip gaps polluting the archive.

**Normalized gap** — the gap divided by what the bar could plausibly deliver
on that world (its completed + rejected units), so gaps on small and large
worlds rank fairly.

**Novelty** — a bounded 0–1 term in the mining rank measuring how different a
task's shape descriptor (depth, products, producers, capacity types) is from
everything already archived; it spreads the archive across world types
instead of letting one cell dominate. Used for archiving only — replay
priorities deliberately exclude it (one high-novelty task once dominated a
whole training round).

**Closure** — the fraction of a mined gap a retrained candidate actually
recovered: (candidate − champion) / (bar − champion). Closure 1.0 = the
candidate fully absorbed the demonstrated headroom; the arc-6 fix moved
closure from ~0.16 to ~1.0. Reported per split (train / validation / blind).

**Harvest (rate)** — informal synonym for closure viewed over a whole round:
how much of everything mined the retraining actually converted.

**Ceiling bound** — an analytic upper limit on the units *any* policy could
deliver on a world, computed from exact demand and capacity arithmetic rather
than from running any policy. A champion near the ceiling means the remaining
gap is not trainable headroom.

**Failure classification** — the per-world label the curriculum assigns to
every below-bar result, exactly one of a fixed taxonomy (e.g.
`policy_weakness`, `distribution_mismatch` — the world lies outside what the
model is claimed to handle, `feasibility_wall` — no policy can do better,
contract/implementation errors, or `none`). Only `policy_weakness` feeds
training.

**Gate / promotion gate** — a pass/fail check a candidate must clear:
improvement on mined validation and blind splits, no transfer or stress
regressions, zero contract failures, no new economy collapses. The pipeline
outputs a recommendation; **promotion** — actually replacing the champion — is
always a recorded human decision.

**Probe** — a small, targeted measurement run to answer one question cheaply
before committing to an expensive one. The **representability probe** asks:
distilled directly on a specific world's teacher labels, can this model
architecture reproduce the teacher's result *on that same world*? If not, no
amount of full-scale training will help.

**Smoke check** — a seconds-long run whose only purpose is "does it execute at
all"; the only category of run allowed on the development machine. Anything
that produces a decision-relevant number runs on the cloud.

---

## The curriculum pipeline

**Game master** — the collective name for the automated loop that chooses the
champion's next training challenges: generate worlds, find verified
weaknesses, retrain, re-verify. Named for the tabletop-gaming role; a process,
not a person. The arc that used it is closed — see `experiments/HISTORY.md` section 7.

**Mining (hard-case mining)** — running champion vs bar across hundreds of
generated worlds and keeping the ones where the champion verifiably loses
(subject to the noise floors).

**Preflight** — the cheap check run *before* mining that the generated worlds
actually have the property being targeted (e.g. that capacity, not demand, is
the binding limit), so compute is not spent mining the wrong regime.

**Task / task archive** — a mined world saved as a compact replayable record
(the generator recipe, its parameters, and the seeds). The archive is the
curriculum's durable training material; any pipeline step can reconstruct the
exact world and demand trace from a record.

**Replay (curriculum replay)** — the retraining step: distill a candidate on
the broad original training set plus archived tasks, weighted toward the
largest verified gaps ("prioritized replay").

**Gap attribution / channel** — the diagnostic that decides *which role* owns
a mined gap: substitute the bar's producer (or orderer) into the champion
stack one side at a time and measure how much of the gap each substitution
recovers. A task's gap is then labelled with a dominant channel: producer,
orderer, mixed, or interaction/noise.

**Retention** — previously mined tasks carried into a later round's evaluation
to confirm the new champion still handles them (guarding against forgetting).

**Track A / Track B / Track C** — the planned Stage-4b mining directions:
capacity-binding worlds (A, current), scripted demand shocks (B), per-order
TTL variation (C).

---

## Infrastructure and records

**Pod** — a rented cloud machine (Runpod). All real training and experiments
run on pods via `run_pod_training.py`, which handles the full lifecycle and
always stops the machine afterwards.

**EXP-NNNN / experiment record** — one directory per experiment under
`experiments/`, holding the question, exact configuration, results, and
conclusion. The index in `experiments/README.md` is the project's timeline.

**ADR** — Architecture Decision Record (`docs/adr/`): a short numbered
document explaining a design decision and its reasoning.

**Diagnostics registry** — `docs/DIAGNOSTICS.md`: known failure symptoms
mapped to the specific check that identifies each, so recurring problems are
recognised instead of re-debugged.

**Root-cause rule** — the standing project rule that a failed or
underperforming run must have its cause diagnosed *with evidence* and written
down before any next strategy is proposed.
