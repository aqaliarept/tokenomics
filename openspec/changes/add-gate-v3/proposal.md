# Proposal: gate v3 (roster v3, registry v3, a 42-producer teacher read, the per-order gate subject)

## Why

The user's decisions of 2026-10-30 (`add-m4-orderer-redesign/notes/user-decisions.md`, open
decisions 1–4) change frozen gate inputs, so they are planned here as their own change rather than
inside the M4 ladder:

- **The bar is easier than our own evidence.** Roster v2's six members are all `entity_planner`
  with splitting or blind routers. M3 found stronger hand-written routers: `round_robin` beats
  `random` on all 104 random-dominated episodes (EXP-0048), and `makeable_weighted_whole` beats
  `round_robin` on 62% of the 639 atlas screen worlds (EXP-0051 addendum). The user's standing
  rule is that the roster always holds the strongest policy found.
- **The registry sits where routing matters least.** Registry v2 gives every product two makers
  at light load. Confirmed headroom is a 42-producer wide-choice (f14) phenomenon, with a smaller
  12-producer dense / f4 region at load (EXP-0052, task 5.3 addendum).
- **No lookahead reference exists at 42 producers.** PoC-A / PoC-C (EXP-0053 / EXP-0054) ran at 12
  only.
- **The gate cannot judge the redesigned orderer.** Its candidate is a sampled per-order policy
  under the `choice-v1` decode (ADR-0059), and the user chose a white-box producer
  (`entity_planner`) to decide the exit, the EXP-0035 producer reported only, and R = 5 candidate
  replicates judged on the mean. The gate today accepts only `portable` (EXP-0035 producer +
  `logits-v1` scorer) or a baseline pair.

## What Changes

- **Roster v3 draft.** `round_robin`, `even_split_rotating`, `random_whole`,
  `makeable_weighted_whole` and `makeable_split` are promoted from the diagnostic routers into the
  gate's ordering baselines (same functions, same names, byte-identical actions). The draft pairs
  all 8 producing baselines with all 11 ordering baselines (88 members, R = 5 for stochastic
  members). Pruning, not a hand-picked list, decides which of the new routers stay ("if the
  evidence says so").
- **Registry v3 draft.** Every v2 entry is kept unchanged. Two world-space region families are
  added: `ws_wide` (42 producers, 14 makers per product) and `ws_loaded` (12 producers, dense or 4
  makers per product). They are **fresh draws** from the atlas lever distribution on seed streams
  of their own, never atlas worlds. Each split, size and regime gets a fixed count (development 4
  calm + 2 shocked per size, sealed 8 + 4), plus reserves. Calm region worlds are cyclic and
  shock-free; shocked ones carry `shocks-i` and its demand drift exactly.
- **Screen v3.** A baseline-only development screen with a pre-registered two-wave member plan
  (all 88 members everywhere except on the costly 42-producer region worlds, where only wave-1
  winners plus `entity_planner` × every router run), never-argmax pruning, zero-baseline and
  bar-death replacement, headroom per family × size, and side-effect parameters X / Y calibrated
  **per family**. It writes candidate frozen `worlds_v3` / `roster_v3` / `side_effect_params_v3`
  sharing one `screened_on`. A later roster-only screen of the sealed split at 12 producers
  prepares the M4 exit read.
- **42-producer teacher read.** The PoC-A teacher (`d2`) and the PoC-C role swap on the 42-producer
  f14 atlas **train** worlds. Sized by a local timing probe, with a pre-registered reduction
  ladder. It is a reference for the orderer at 42, never a label source, a training target or a
  gate input. `ub` is not run (DIAGNOSTICS 23: it is no upper bound for this search).
- **Per-order gate subject.** The gate report accepts a `choice-v1` per-order orderer checkpoint
  paired with a white-box deciding producer, plus a reported column with a checkpoint producer.
  The candidate runs R replicates (R = the roster's R) with keyed draws. Decode and component pins
  go into `spec.json`, the resume check and the sealed pre-registration. `--mode recompute` rebuilds
  both the verdict and the reported table.
- **Not changed:** frozen v1 / v2 registry, roster and parameter files; the hindsight-baseline
  arithmetic; the acceptance margins; `domain/`; the gate default (v2 stays the default until the
  user approves v3, gate-harness "Raising the bar is a user decision").

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `gate-harness`: roster v3 contents and promoted routers; stochastic roster members beyond
  `random`; two-wave pruning; registry v3 (region families, fresh draws disjoint from atlas and M4
  worlds, bar-death replacement); regimes and families from v3; per-family side-effect parameters;
  sealed-split screening by size; the per-order candidate subject with its reported column,
  replicates and decode pins.
- `development-audit`: promoted routers leave the "diagnostic routers stay outside the roster"
  rule; atlas evidence may shape gate regions by user decision while no atlas world enters a
  registry; the 42-producer teacher read and its reduction ladder.
- `world-space`: region samples (the atlas lever distribution with the categorical levers fixed by
  the caller) and their acceptance filter.

## Impact

- **Code (outside `domain/`):** `eval/baselines.py` (promoted routers, stochastic baseline set),
  `eval/roster.py` (`build_roster_v3`, two-wave pruning record), `eval/registry.py`
  (`build_worlds_v3`, region entries with world records, validation), `eval/world_space.py`
  (region sampling), `eval/gate.py` (per-family X / Y, reported rows), `eval/gate_cli.py`
  (two-wave screen, bar-death replacement, sealed screen by size, per-order subject, pins,
  manifest), `eval/teacher_audit.py` (42-producer sample and reduction flags), tests per layer.
- **New files after the screen (human gate):** `eval/registry/worlds_v3.json`, `roster_v3.json`,
  `side_effect_params_v3.json`, copied from the screen's `frozen/` only after the user approves.
- **Docs:** a new ADR (gate v3), `ARCHITECTURE.md` §7.2, the CLAUDE.md gate bullets (when v3 is
  packaged), `experiments/README.md`, `experiments/HISTORY.md` (a §15 "gate v3" arc).
- **Records:** three EXP records in this change (screen, 42-producer teacher read, sealed-12
  screen) and two gated reads (the M4 candidate's development and sealed exit reads), numbered at
  launch.
- **Compute:** Runpod secure CPU pods only, each run probe-sized with a hard stop of 2 × the
  estimate, ≤ 40 vCPU-h. Central ≈ 33 billed vCPU-h for the three ungated runs, ≈ 18 more for the
  two gated reads (design D10).
- **Interaction with M4:** none of M4's training, validation or held-out worlds is played by the
  gate. The gated reads run only after M4's `confirmed` held-out read (EXP-0061) and the user's
  approval of v3 as the gate default.

## User decisions (2026-10-30)

- Registry v3 uses fresh draws from the atlas hard regions on disjoint seeds, not the atlas worlds (all of which M4 uses for training, validation or held-out reads).
- Run-down / starvation limits (X / Y) are calibrated per family.
- The roster holds hand-written policies only; the "strongest policy" rule never adds a trained orderer.
- The M4 exit read is the sealed 12-producer split (both regimes); the development read is reported first.
- Region weight: the draft default (region worlds count as one third of each regime ratio) stands unless the plan review finds a reason to change it.
- Budget: counted against M4's tranche under the supervisor's budget authority.
