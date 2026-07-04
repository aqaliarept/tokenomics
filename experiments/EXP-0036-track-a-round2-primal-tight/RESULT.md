# EXP-0036 — Track A round 2: primal-tight mining (IN PROGRESS)

Round 1 (EXP-0034) tightened capacity uniformly and the bottleneck landed mid/late-chain;
the s54 diagnosis showed the replay traded away the "upgrade the tight primal source
first" skill. Round 2 mines worlds where the ceiling binds at layer 0
(`primal_capacity_scale` ≪ `capacity_scale`, upper layers loose), against the freshly
promoted EXP-0035 champion.

## Question

Does the EXP-0035 champion have real headroom on primal-bottleneck worlds, which channel
owns it, and can that channel's learner absorb it?

## Pipeline + results so far

1. **Preflight (local)**: `primal_tight_cells()` 4/4 cells capacity-bound at every world
   seed 20-119, worst binding constraint `capacity:capacity-0` (layer 0, as designed).
2. **Mine (pod)**: 400 tasks (4 cells × seeds 20-119) vs `entity_planner+mpc_search`,
   paired-seed floors (min_gap 2.0, min_win 0.6). **96 policy_weakness → 81 archived**
   (per cell 26/35/2/18). Gaps are an order of magnitude beyond round 1: min 2.7 /
   median 53 / mean 99.9 / **max 973** units (round-1 mean was 3.67u). 28/81 win all 3
   seeds.
3. **Gap attribution (pod)** over the combined archive (113 round-1 recoverable as
   retention + 81 new):
   - **Round-2 headroom is ORDERER-owned**: channels 45 orderer / 15 mixed / 11 producer
     / 10 noise; of 8088u total gap, the search-orderer cross recovers **7172u (89%)**,
     the producer cross only 1123u (14%). `orderer_closure_ratio` 0.928 vs producer
     0.310. Round 1 was the inverse (70% producer).
   - The monster gaps are **champion system-death**: 6/81 tasks (5 full-death, all in
     the tightest cells) where the champion's routing starves the primal layer and the
     economy collapses; the bar's search routing keeps the same producer alive.
   - Round-1 sanity: **54/113 old tasks have no positive gap left** — the promoted
     champion genuinely absorbed round 1.
4. **Orderer representability probe (pod, running)**: user picked orderer-probe-first
   (freeze-orderer replay would strand ~86% of headroom). EXP-0032 oracle-relabel
   machinery on 10 tasks (top-2 orderer-gain per cell incl. the 4 death monsters + 2
   mid-gap), `--iters 1000`, verdict = `recovered_vs_teacher`.

## Found + fixed: probe/archive seed-convention mismatch (DIAGNOSTICS entry 10)

First probe launch produced numbers contradicting the attribution on the same tasks
(champ alive at 957u where the archive has it full-death, gap 973). Cause:
`orderer_representability` evaluated at episode seeds `range(eval_seeds)` = 0-2 while
mining/attribution anchor at `_episode_seeds(record.episode_seed, ...)` = 77-79 — a
different demand trace, i.e. a different task. Collection had the archived seed but
deploy didn't (double mismatch). Pod killed mid-run; fixed (evals + collector anchored,
collector now covers all eval seeds); regression test
`test_headroom_row_anchors_at_record_episode_seed`; suite 608 passed.
**Caveat propagated**: EXP-0032b's headroom scan shared the bug — its "0/26 orderer
headroom" conclusion stands only via EXP-0033's exact-seed analytic ceiling, not its own
measurement.

## Pending

Probe verdict → lever decision (orderer oracle-relabel replay vs architecture
discussion). This RESULT will be updated.
