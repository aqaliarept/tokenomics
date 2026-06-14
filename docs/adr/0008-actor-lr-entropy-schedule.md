# Actor LR annealing and entropy decay; longer runs

Track A step 2 (Phase 1 plan). With the critic fixed (ADR-0007), the next
bottleneck is the actor schedule. The 20-iter smoke scale can't converge, and
the observed regression once the curriculum (ADR-0006) pins difficulty at 1.0
has the signature of a too-high actor LR late in training: the policy keeps
taking large steps at full difficulty and wanders off a survivable region.

## Decision

**Linear schedules over the run** (`anneal=True`, on by default):

- **Actor LR**: `actor_lr · (1 − 0.9·frac)` — decays to 10 % of the initial LR
  by the final iter. Cools the actor steps as difficulty reaches 1.0.
- **Entropy coef**: `ent_coef · (1 − frac)` — decays to 0. Early exploration,
  late exploitation; the deployed greedy policy wants low entropy.

`frac = it / (iters − 1)` is run progress. The critic LR is left fixed (1e-3) —
ADR-0007 made it steady; only the actors need cooling. `ent_coef` was previously
hard-coded at 0.01 inside `actor_update`; it is now a passed argument the train
loop decays.

**Longer runs.** Schedules are meaningless at 20 iters — Track A runs are
200–500 iters. The schedule is parameterized by `iters`, so it stretches to the
run length automatically.

New `train()` args: `actor_lr`, `ent_coef`, `anneal`; CLI `--actor-lr`,
`--ent-coef`, `--no-anneal`.

## Trade-offs

- Linear, not cosine. Simpler and the decay shape is not the sensitive knob here
  — the late-run magnitude is. Revisit if a run shows otherwise.
- A fixed-difficulty run (`--no-curriculum`) still benefits from the schedule;
  the schedule is independent of the curriculum (ADR-0006) and composes with the
  eval-gating it may grow (Track A step 3).
- Decaying entropy to exactly 0 risks premature collapse if the run is too short
  to have found a survivable policy first — coupled to run length, watch the
  eval curve.
