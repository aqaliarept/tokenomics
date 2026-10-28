# Proposal

## Why

Roadmap milestone M3 "Development audit" (`establish-project-baseline` design D1, D4) must show,
before any training spend in M4, whether a teacher can beat the hindsight baseline on the frozen
v2 development worlds (PoC-A), which role M4 should train first (PoC-C), and what the deployed
path costs as the economy grows (PoC-B). M3's exit is: PoC-A/B/C recorded, and every failure
recorded with a diagnosed cause.

One finding has to be explained first. In the v2 calibration screen (EXP-0047) the `random`
ordering baseline (`entity_planner/random`, uniform random routing logits, scored as the mean of
5 replicates) sets the hindsight baseline on 95 of the 128 development world-episodes: 6/24 at
12-calm, 28/40 at 12-shocked, 21/24 at 42-calm and 40/40 at 42-shocked. EXP-0006B found the same
pattern earlier: the seeds a deterministic orderer missed were the ones a random orderer won, and
sampling at deployment closed them. Nobody knows why. Possible causes include packing under
atomic acceptance, the value of spreading orders over time, defects in the hand-written rules, the
investment-fund channel, the penalty of splitting an order when only whole orders count as
fulfilled, and a scoring artifact. These call for different responses: the bar may
be too low (a rule defect), M4 may need a load-aware feature, or M4 may need stochastic
deployment. PoC-A measures the teacher against this bar, so its reading cannot be interpreted
until the cause is known (CLAUDE.md "Root cause before next strategy").

## What Changes

- **Random-routing root cause (first).** Two pieces of diagnostic instrumentation:
  - per-bid rejection and failure reasons, recorded as additive pure data on the domain's
    production result (no RNG, IO or logging in `domain/`) and collected by an eval-side
    tracer;
  - diagnostic routers that separate the competing explanations: deterministic rotation, random
    weights held fixed per episode, a noise-scale sweep of `random`, rules with their suspected
    defects fixed, `random` on one order class only, and a whole-order / split-order ×
    deterministic / random control.

  Then a local single-episode unit trace, and one cloud experiment, **EXP-0048**, that runs the
  diagnostic routers on all v2 development world-episodes. The result goes into its `RESULT.md`
  in plain English, a `docs/DIAGNOSTICS.md` entry and a HISTORY paragraph. If a hand-written rule
  is found defective, the fixed rule is a new ordering baseline under a new name. It reaches the
  gate only through a new roster version and a re-screen (registry, roster and parameters v3).
  The frozen v2 files are never edited. Adopting the new bar is a user decision.
  A follow-up, **EXP-0049** (inserted after EXP-0048 on the user's request), runs three
  whole-order routers that read each producer's spare capacity against the blind
  `round_robin` / `random_whole`, and traces the trained portable orderer with the same
  routing metrics, to see whether it learned random's spreading or `proportional`'s splitting.
  A second follow-up, **EXP-0050**, reruns those arms on family worlds with 2 to 14 producers
  per product (registry v2 has 2 everywhere), to see whether the routing findings hold when
  the choice widens.
- **PoC-A, teacher against the hindsight baseline (EXP-0051).** The EXP-0042 confidence-bounded
  rollout orderer runs at depths 2 and 3, plus a clairvoyant upper bound, over the
  `entity_planner` producer. It plays a pre-registered subset of the v2 development
  world-episodes, calm and shocked, and is compared with the frozen roster v2 on identical
  episodes. The teacher follows the D4 **teacher fork rule**:
  - forks keep the observable history;
  - hidden shock state is resampled from its posterior given that history and the known
    generator;
  - the posterior acceptance rate is logged, and below a pre-registered floor the decision is
    labelled "unresolved" and falls back to `blend_0.5`;
  - hidden domain fields are stripped before a rollout;
  - future demand keeps drift continuity;
  - a pytest shows byte-identical labels for two envs that differ only in hidden state.

  The EXP-0042 teacher code is still on `new-model` (`eval/bounded_rollout_orderer.py`,
  `env/future_fork.py`), so nothing is restored from `archive-1`.
- **PoC-C, role attribution under shocks (EXP-0052).** In the style of EXP-0001 and EXP-0031,
  the portable stack is run with one role swapped at a time: the teacher orderer in the orderer
  role, the best hand-written producer in the producer role. This happens on the PoC-A shocked
  subset, and the per-role closure of the gap decides which role M4 trains first.
- **PoC-B, deployed-path cost (EXP-0053).** Per-decision cost of the portable stack is measured
  stage by stage on two independent axes:
  - total producers P ∈ {12, 24, 42} at fixed fanout;
  - eligible producers E (fanout) ∈ {2, 4, 7, 14} at P = 42.

  No point above 42 producers is timed unless the user asks for one. **Capped at the
  12 → 42 target**, not 1200 producers (see Open question).
- **Compute.** Every experiment runs on a CPU pod, is sized by a local timing probe, and has a
  hard stop in vCPU-h. The parts with a grounded estimate (EXP-0048, EXP-0049, EXP-0050, EXP-0053 and the
  12-producer parts of PoC-A/C) are expected to cost about 10–40 vCPU-h. The 42-producer teacher
  parts are unknown until their probes and will likely hit their stop first, which inserts a
  profile-and-optimize code task before they launch. The hard ceiling for the change is
  70 vCPU-h, and no run comes near 1000 vCPU-h.

No **BREAKING** changes. The frozen v1/v2 registry, roster and parameter files, every existing
episode statistic, and `ORDERING_BASELINES` / `PRODUCING_BASELINES` stay byte-identical.

## Open question for the user

**PoC-B scale.** D4 says PoC-B measures cost "up to 1200 producers". That conflicts with the
2026-10-27 decisions: the acceptance target is 12 → 42 producers, 120 is dropped until agents
beat the hand-written baselines at 42, and there is no budget for runs of 1000+ vCPU-h. This
change plans PoC-B at 12/24/42 producers only. Should D4's "up to 1200" be amended to "up to 42
(larger sizes when scale is reopened)"? And does the user want one larger, non-gating timing
point despite the scale decision, for example P = 84? The default is no. The driver supports
it behind a flag, and a yes adds it to the PoC-B probe and run. The rest of the plan does not
depend on the answer.

## Capabilities

### New Capabilities

- `development-audit`: how the M3 audit diagnoses a baseline anomaly before reading a teacher
  against the hindsight baseline, what information a teacher label may depend on (the fork
  rule), how PoC-A, PoC-B and PoC-C are read, and that a failed PoC ends in a recorded diagnosis.

### Modified Capabilities

- `gate-harness`: modifies "Frozen baseline roster", so that a roster version's pairings
  range over the baselines that exist when it is drafted and a later baseline enters only a
  later version. Adds two requirements.
  - "Baseline rule corrections create a new roster version": a fixed hand-written rule enters
    the gate only as a new member of a new roster version with a re-screen, never by editing a
    frozen member.
  - "Raising the bar is a user decision": a new roster version becomes the gate default only
    after the user approves it.

## Impact

- Domain: `tokenomics/tokenomics/domain/producing_agent.py`. The auction's per-cycle result gains
  per-rejected-bid and per-failed-reservation reasons as additive pure data. No RNG, IO or
  logging, and every episode statistic stays byte-identical.
- Env: a new `env/shock_posterior.py`, which resamples hidden shock state from the posterior and
  strips hidden fields from a fork, and `env/future_fork.py`, which gets a drift-continuous
  demand fork.
- Eval:
  - new modules: `eval/routing_diagnostics.py` (diagnostic routers, kept out of the roster
    dicts), `eval/routing_trace.py` (per-bid tracer), `eval/routing_audit.py` (EXP-0048 / 0049 / 0050 driver),
    `eval/teacher_audit.py` (PoC-A/C driver) and `eval/deployed_cost.py` (PoC-B driver);
  - `eval/bounded_rollout_orderer.py` gets a pluggable scenario source, a baseline producer and
    a single-scenario deterministic future.
- Records:
  - EXP-0048 to EXP-0053, each with a `RESULT.md` and an `experiments/README.md` row;
  - a new HISTORY §13 "The development audit (M3)";
  - a `docs/DIAGNOSTICS.md` entry 19, for the random-routing cause;
  - Domain Language entries in CLAUDE.md for any new term, for example "Unresolved label".
- Compute: six CPU pod runs, each sized by a local probe, with a hard ceiling of 70 vCPU-h
  (design D8). The 42-producer teacher parts likely need a profile-and-optimize task first.
- Records: ADR-0056, "Teacher information set and posterior forks".
