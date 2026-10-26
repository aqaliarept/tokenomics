# Design

Agreed plan (PLAN v3.1) for reaching the baseline specs. Converged in agent-bus team
`plan-debate` (task 4) by Opus 5.5, codex1 and codex2; Fable was unavailable (rate limit).
Motivation: see `proposal.md`. Requirements: see `specs/`.

## Context

Findings from the code and experiment record that shape the approach:

- **No evidence the margin is reachable yet.** EXP-0042's +15.3% is measured against
  `blend_0.5` (absent from `ORDERING_BASELINES`, `eval/baselines.py:346`) on mined,
  non-held-out worlds s29/s45/s77 (`TASK_S29`/`TASK_S77`/`TASK_S45` in
  `eval/bounded_rollout_orderer.py`); s45 depth-2 death rate was 0.2. It is not an
  envelope/held-out acceptance result.
- **EXP-0033's 0.9902 is search-bar/upper-bound**, not envelope/bound (champion 0.983). It
  does not prove +5% impossible, nor that only capacity-binding worlds have headroom.
- **EXP-0043 is not deployable**: its surrogate only short-circuits predicted-zero streams;
  all others still run real depth-2 rollouts (`eval/scenario_rollout_surrogate.py:1-15`, now
  on branch `archive-1`).
- **Deployed decision cost is ~O(P^2), not linear**: all-producer rows
  (`domain/orderer_view.py:278-284`), full EMA-key scans per producer aggregate
  (`domain/telemetry.py:36-39`), full-P decode masks (`env/decode.py:49-50`), remainder
  `argsort` (`env/allocator.py:38`).
- **The scale generator cannot express sparse economies**: every layer producer declares
  every layer product (`eval/scale.py:138-161`); growing producers with fixed consumers
  adds capacity without demand (EXP-0033-style saturation risk).
- **Demand RNG is one sequential stream** (`env/aec_env.py:169-195`); inserting shock draws
  would perturb existing demand schedules.
- **Accounting boundary trap**: `start_cycle` runs at construction (`domain/system.py:254`)
  and again inside `complete_cycle` (`:460`); naive counting includes an unplayed cycle.
- **Context cannot rerank producers today**: the orderer score is `a(row) + b(context)`
  (`train/orderer_scorer.py:132-151`); `b` cancels in pairwise differences/softmax; clipping
  (`:174-175`) is monotone and cannot reverse ranking.
- **Random baselines reuse the event seed** for action sampling (`eval/runner.py:59-64`).
- Shock consequences are partly observable (`domain/orderer_view.py:91-97` track-record
  fields), but history is folded after production (`domain/system.py:457-458`), so the
  first affected decision can be blind. Observability is an opportunity, not evidence the
  shocked margin is easier.

## Goals / Non-Goals

**Goals:**
- An honest, frozen measurement gate before any training spend.
- Shocks I + drift implemented to the `stochastic-environment` contract.
- A diagnostic go/no-go on headroom, cost and role attribution.
- A bounded, pre-registered learning attempt whose failure produces a diagnosis.

**Non-Goals:**
- No restart of closed arcs (learned value function, curriculum mining, direct imitation of
  marginal-value/depth-2 targets, PPO) without addressing their recorded causes.
- No deployed decision-time search (search is a teacher only).
- No margin renegotiation as a reaction to a failed PoC.

## Decisions

### D1. Milestone order

```
M1 harness --> M2 shocks I + drift --> M3 dev audit --> M4 bounded ES --> M5 sealed 120 gate
                       |
                       +--> M6 shocks II (parallel to M3-M5) --> M7 lifecycle (TTL>2)
```

| # | Milestone | Exit |
|---|---|---|
| M1 | Measurement + gate harness (D2) | Champion full gate report at 12 and 120 producers on development families |
| M2 | Shocks I + drift (D3) | Every delivered spec scenario is a pytest; champion runs shocked unretrained, zero contract failures |
| M3 | Development audit (D4) | PoC-A/B/C recorded; failure → diagnosed cause in RESULT.md |
| M4 | Bounded ES feasibility → candidate trained on ≤12 producers (D5) | Sealed 12-producer gate pass, both regimes |
| M5 | Sealed 120-producer gate, unretrained, sparse-wide + dense | All margins, no-losing-world, 0% death, side-effect guards |
| M6 | Shocks II: link cut, demand surge, fund shock | Remaining shock scenarios as pytests |
| M7 | Order lifecycle: TTL > 2, multi-stage | Backlog/TTL compatibility; meaningful full episodes without uniform-overflow routing (`env/decode.py:55-58`, ADR-0054); checkpoints load + run |

M7 follows all supply shocks (M6) per `project-mission`. M6/M7 never wait for agent margins.

### D2. Measurement + gate harness (M1)

- **Per-consumer accounting** on the played-cycle cohort; unresolved-at-horizon demand stays
  in the denominator; terminal capacity pinned before any unplayed next-cycle event;
  fixtures for horizon = 1 and early death.
- **Side-effect metrics** per `side-effect-guards`, with exact aggregation (incl.
  zero-ordered consumers). X and Y frozen from baseline-only calibration on development
  worlds before any gate.
- **Frozen envelope roster**: all hand-written producer × orderer stacks + `blend_0.5` + a
  shock-reactive baseline (frozen from development evidence before any gate). Random-policy
  replicate count R frozen with the roster; replicate seeds = SHA-256(world, episode,
  replicate), never the event seed.
- **Envelope arithmetic**: per-episode max over roster → mean over episodes →
  candidate/envelope per world → equal-weight world mean per regime.
- **World registry**: development worlds (method selection, teacher audits, calibration) vs
  sealed gate worlds (never inspected for algorithm selection); zero-envelope worlds
  replaced; candidate-independent filters.
- **Regimes**: calm = no stochasticity, using explicitly configured deterministic cyclic
  schedules (never a silent replacement of legacy fixed demand); shocked = any stochastic
  demand (incl. B9) or shocks. Calm gate uses ≥2× the shocked world count (calm episode
  seeds replay identical demand). Schedule families and counts pre-registered.
- **Scale families**: new sparse-wide and dense generators with explicit `fanout`; demand /
  capacity and fund / producer ratios held constant from 12 to 120 producers by co-scaling
  `num_consumers` and `cycle_emission`; headroom verified on development worlds at both sizes.
- **Deployed-path refactor** (hard prerequisite of M4): eligible-only rows, indexed
  incremental per-producer telemetry totals, O(E) remainder selection. Per-stage complexity
  derivation + timings on independent total-P and eligible-E axes (slope ≤ 1.1 is a warning
  threshold, not proof). Invariant: on the existing fixture set, legacy checkpoints produce
  byte-identical actions and `episode_stats`; any feature that cannot stay identical becomes
  a versioned schema change whose value effect is measured on development worlds.

### D3. Shocks I + drift (M2)

- Schedules generated before episode start solely from world config + episode seed; each
  shock kind has its own derived stream, so adding kinds later leaves earlier schedules
  byte-identical; existing demand draws preserved.
- Outage blocks both acceptance of new bids and completion of production during its active
  cycles; capacity still degrades; fixture for in-flight production.
- Drift: at most floor(j·N) changed consumer orders per cycle transition.
- Unbuilt shock kinds rejected at config validation. Resilience report (dip, recovery time
  vs matched calm episode).

### D4. Development audit (M3)

- **PoC-A — teacher vs envelope** (CPU pod): EXP-0042 depth 2/3 + upper bound on
  development worlds, calm and shocked. UB/envelope ≥ 1.15 is an optional slack target;
  necessary checks are the regime margin and per-world no-loss rule.
- **Teacher fork rule** (prevents the EXP-0038B "target not a function of observation"
  cause): preserve realised observable history and current observable state; resample all
  forecast-relevant hidden shock state (active, remaining duration, magnitude, target,
  onset) from the posterior given observable history and the known generator, via
  rejection/particle sampling that keeps only draws reproducing observed consequences;
  log acceptance rate; below a pre-registered floor → label "unresolved" and fall back to
  `blend_0.5`; strip hidden domain fields (outage end cycles, pending event queues) before
  rollout; preserve drift continuity. Pytest: two envs differing only in true hidden
  remainder yield byte-identical teacher labels for the same fork seed.
- **PoC-B — cost** (pod): full deployed path, independent P and E axes, up to 1200 producers.
- **PoC-C — role attribution under shocks** (pod): teacher swapped in per role
  (EXP-0001/0031 style) to choose which role M4 trains first.
- Failure of any PoC → diagnose and record; not automatic abandonment.

### D5. Primary learning approach: bounded closed-loop ES on a residual (M4)

Deployed score = champion score + small residual; arms: zero residual (control), row-only,
and low-rank bilinear row × context `Σ_r (u_r·φ_row)(v_r·ψ_ctx)`, r ≤ 4 — needed because the
current context term cannot rerank. Initialise u_r ≠ 0, v_r = 0 (residual starts at zero
without a degenerate zero-gradient point under antithetic sampling). Explicit producer-toggle
parameterisation (EXP-0002). Bounded residual magnitude.

- **Fitness**: paired units/envelope with common random numbers + pre-registered holding
  penalty term; death, run-down or starvation disqualifies a population member.
- **Data**: ≥50 generated worlds with ≤12 producers, fanout varied, shocks + drift on;
  train / validation / untouched final worlds.
- **Pre-registered before PoC-E**: finite pod compute budget, optimizer seeds, arms,
  handling of infeasible perturbations / no feasible initial candidate, and a validation
  success rule vs zero-residual and frozen hand-written controls.
- **Kill rule**: gains vs the same frozen comparator and regime weights; if training gain is
  positive and above the noise floor but validation gain < ⅓ of it → stop and diagnose
  (EXP-0009 cross-world cause); if training gain is within noise → feasibility unresolved,
  diagnose optimisation / feasibility / headroom first.
- **Teacher warm-start**: only after a fresh audit shows the target is useful and stable
  (EXP-0040), not correlation alone.

Why ES on a residual: optimises deployed trajectories directly, so no future-sensitive
per-decision labels (EXP-0038B/0039/0040) and no critic (EXP-0003/0009); low dimension and
linear cost by construction.

Alternatives rejected: EXP-0043 hybrid (rollouts at decision time); distilling the depth-2
argmax (unstable winner, EXP-0040); named-policy selection (anti-transferred, EXP-0036);
PPO (rejected for this stage); depth-1 search deployed (O(N·eligible) cost, −14.4% safety
failure in EXP-0042).

### D6. Versioning and compute

- Every acceptance report binds immutable rule/config, policy, baseline-roster and
  world-registry versions. A pass certifies that ruleset only; after a rule change, contract
  compatibility is reported separately and acceptance needs the full gate again (all
  supported shock kinds, all lifecycle-feasible depths).
- All decision-gating runs (PoC-A–E, calibration, scale/headroom, timing) are CPU pod runs.
  Local = pytest fixtures and single-episode timing at horizon ≤ 12, never decision-gating.

## Risks / Trade-offs

- [No envelope headroom: depth-3 teacher below 1.10 shocked on development worlds] → diagnose
  (world family, roster, fork) before any training; margins are not renegotiated reactively.
- [Shocked margin is a benchmark artifact: a trivial reactive baseline closes most of it] →
  that baseline joins the frozen roster before the gate; the margin must be re-earned.
- [ES fails cross-world (validation ≪ training)] → kill rule; diagnose before changing arms.
- [Dense-fanout transfer fails at 120] → features are wrong; fix before retraining. Transfer
  drop ≤ 3 points is diagnostic only.
- [Refactor silently changes champion behaviour] → byte-identical fixture invariant.
- [Teacher peeks at hidden schedule] → posterior fork + byte-identical-labels pytest.

## Open Questions

- Numeric values to pre-register before first use: X, Y, j, shock magnitudes / durations /
  frequencies, recovery tolerance, posterior acceptance floor, ES budget and success rule,
  R (random replicates), noise floor. None changes the approach or milestone order.
