# EXP-0020 — Broad C3/P3 fixture matrix + failure classification (ordered-plan step 10)

Status: **DONE — ZERO schema/load and ZERO decoder/allocator contract failures across all
22 out-of-family / scarcity / degenerate fixtures, in producer-only, orderer-only, and
joint-stack modes.** The id-free contract (ADR-0053/0054) loads and runs unchanged on every
new fixture family — shared capacity types, deeper+wider joins, dedicated uneven strengths,
five scarcity stress axes, and five degenerate edge economies. Every miss is a *policy* or
*feasibility* effect, separable per row, never a contract one:

- **Producer (orderer fixed `proportional`):** portable producer ties the `entity_planner`
  bar (mean **0.998**, min **0.949**, no contract failure); the only sub-1.0 cells are small
  policy dips on **uneven-strength** (0.949–0.977) and a few **scarcity** axes (low_inventory,
  high_backlog, running_investment ≈ 0.95–0.98). Planner-gate mirror ≈ learned (mean 0.999).
- **Orderer (producer fixed `stock_aware`):** portable orderer **ties-or-beats every
  deterministic orderer on every fixture** (mean **1.055**, no death), with large wins exactly
  where routing matters — **uneven-strength 1.005–1.307**, **scarcity 0.999–1.228**. The two
  sub-1.0 cells (`degen-minimal` 0.419, `single_producer` 0.971) are degenerate
  **no-routing-choice** economies where *all four orderers tie at the same value* → a fixed
  `stock_aware`-producer floor, **not an orderer miss** (the portable *producer* scores 1.0 on
  the same `minimal` target).
- **Joint (full portable stack):** mean **1.231**, max **2.002**; biggest lift on
  **uneven-strength (1.556–2.002×)** — the learned scorer/allocator pair concentrates routing
  and investment on the one dominant producer per level, the regime the proportional baseline
  cannot exploit. Only genuine joint dip: `degen-no_backlog` 0.95 (trivial-demand edge).
- **Deeper-join wall:** `deepjoin-d5-w4-in3` completes **0 units for every policy including
  `entity_planner`** — the depth×TTL feasibility wall (chains ≥4 deep are unmakeable at TTL=2,
  see `scale-reward-calibration`). This is a **scaling/feasibility** result, *not* a contract
  or policy failure (the fork/join recipe DAG itself loads fine — see the feasible depth-3
  fork/join in EXP-0017/0019). Reported as undefined ratio (baseline 0).

## Setup
- **Runner** — `tokenomics.eval.transfer_matrix --fixtures` (new step-10 target source). Six
  reusable generators wrap the step-9 fixture families into `TargetTopology` lists —
  `shared_capacity_targets`, `deep_join_targets`, `uneven_strength_targets` (per-seed builder
  families), `scarcity_targets`, `degenerate_targets` (dims-preserving suites off the base
  seed; `primal_only` carries depth=1), and the `fixture_matrix_targets` umbrella. `--fixtures`
  composes with `--audit {producer,orderer}` (audit stack sets) or the joint `DEFAULT_STACKS`.
  No policy code changed; EXP-0012/0013 checkpoints run unchanged. TDD'd in
  `tests/test_transfer_matrix.py` (6 new cases).
- **Targets (22)** — shared-capacity ×4, deeper-join (d5, 3-input) ×4, uneven-strength ×4
  (seeds 6–9); scarcity ×5 + degenerate ×5 (off seed 6). 5 episodes/cell, horizon 100.
- **Bar** — every ratio vs `entity_planner + proportional` on the SAME target. `noop` floor
  mean 0.013 (22 deaths) confirms the targets are nontrivial.

## Per-family mean ratio (deeper-join excluded — baseline 0 / infeasible for all)

| family | producer (portable, learned) | orderer (portable) | joint (full portable) |
|--------|------|------|------|
| shared-capacity | 1.009 (1.000–1.022) | 1.095 (0.990–1.158) | 1.109 (1.042–1.173) |
| uneven-strength | 0.986 (0.949–1.048) | 1.163 (1.005–1.307) | **1.731 (1.556–2.002)** |
| scarcity        | 0.996 (0.981–1.016) | 1.101 (0.999–1.228) | 1.155 (1.078–1.251) |
| degenerate      | 1.000 (1.000–1.000) | 0.891\* (0.419–1.075) | 1.005 (0.950–1.073) |

\* degenerate orderer mean is dragged only by `minimal`/`single_producer`, where all orderers
tie (no routing choice) — a fixed-`stock_aware`-producer floor, not an orderer regression.

## Failure classification (the step-10 deliverable)

| class | count | where |
|-------|-------|-------|
| schema / load | **0** | every fixture family loads (id-free contract holds) |
| decoder / allocator contract | **0** | no episode-time raise on any of 22×8 cells |
| producer policy quality | small | uneven-strength 0.949–0.977; scarcity ≈0.95–0.98 |
| orderer policy quality | **0** | portable ties-or-beats every deterministic orderer everywhere |
| scarcity / stress behavior | clean | all 5 axes run, no death; orderer/joint gain on them |
| scaling / feasibility | deeper-join | `deepjoin-d5` infeasible for ALL policies (depth>TTL=2) |

## Conclusions / next
1. **Contract transfer is total** on the broadened surface: shared capacity, wider joins,
   uneven strengths, scarcity, and degenerate edges all load and run with the unchanged
   EXP-0012/0013 checkpoints — no schema/decoder/allocator work owed. This is the strongest
   confirmation yet that the id-free entity contract generalizes off the scale family.
2. **The remaining producer gap is narrow and concentrated** on **uneven-strength** (one
   dominant producer/level, where investment/restoration must concentrate) and a few
   **scarcity** axes — exactly the coverage step 11 should add to producer training. No new
   mechanism; the planner-gate and learned modes behave alike here.
3. **The orderer needs no further work on these axes** — it ties-or-beats every deterministic
   orderer on every fixture and the joint lift is largest precisely on the off-family
   uneven-strength regime.
4. **Fixture note for step 11:** `deepjoin-d5` is a feasibility floor (depth>TTL), so it does
   not discriminate policies — a *feasible* wider-join probe (depth 3, 3-input) would isolate
   join-WIDTH policy quality from the depth wall. Carry as an open item; degenerate
   no-routing economies correctly tie across orderers and need no fix.

Local eval, no retrain (EXP-0012 + EXP-0013 checkpoints). Data: `producer_fixtures.csv`,
`orderer_fixtures.csv`, `joint_fixtures.csv`.
