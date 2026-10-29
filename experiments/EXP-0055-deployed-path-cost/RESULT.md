# EXP-0055 — PoC-B: what a decision of the deployed portable stack costs, up to 42 producers

**Status: complete, valid. Not an acceptance read; no stage flagged.**
`python -m tokenomics.eval.deployed_cost`, run commit `5520746476622f6aa5079a198219201b9d5ed906`
(task 11.1 of `add-m3-development-audit`, design D7; tree SHA-256 `bd5b0c5cb1f4…`), portable
producer EXP-0035 `producer_candidate.pt` (SHA-256 `8b51b77cb5ac…`) and orderer EXP-0012
`orderer_scorer.pt` (SHA-256 `d25388bfa33e…`), staged copies hashed equal to the `spec.json`
pins. 72 timed episodes = 8 cells × 3 world seeds × 3 repeats, horizon 20, calm `sparse_wide`
worlds, 0 deaths. Pod: fresh **4 vCPU** secure `cpu5c` (the 8 vCPU shape had no secure
capacity, two create attempts), CPU **AMD EPYC 9575F 64-Core Processor**, `host.vcpu` = 4 =
the allocation (affinity mask 4, no cgroup quota), `--workers 2`. ≈ 0.23 vCPU-h billed
([Details](#details)).

## In plain English

**The question.** The deployed agent is the *portable stack*: one trained producer policy
(EXP-0035) shared by every producing agent and one trained orderer policy (EXP-0012) shared by
every ordering stream. Each cycle every agent makes one *decision*: it builds its observation,
runs its network, and its output is decoded into an action; then the simulation plays the cycle.
This run asks how the time of one decision grows as the economy grows along three separate
directions, all within the 12 → 42 producer target:
- **producers** P = 12, 24, 42 (at 2 eligible producers per product);
- **choice** (*fanout* E, how many producers can make each product) E = 2, 4, 7, 14 at P = 42
  (14 = every producer of the layer);
- **consumers** per producer × 1, 2, 4 at P = 42, E = 4 (11, 21, 42 consumers; more consumers
  mean more ordering streams per cycle).

No economy above 42 producers is timed: that cap is the user's decision (2026-10-28), and the
driver refuses a larger size with no override.

**What is measured.** The wall time of each episode is split into five exclusive *stages*:
`obs` (observation build), `forward` (the networks), `decode` (turning the output into
sub-orders / production plans, including allocation), `domain` (the simulation's cycle) and
`harness` (the rest of the episode loop). The stages sum to the episode's wall time (worst
0.0015% gap). Each stage is reported in four *measures*, which this record always names:
- **per pooled decision**: stage seconds over all decisions (producers' and orderers' together);
- **per producer decision** / **per orderer decision**: the policy-side stages (`obs`,
  `forward`, `decode`) charged to the role whose call it is, over that role's decisions only;
- **per cycle**: stage seconds over cycles.

The per-role measures matter because the decision mix changes: at P = 42 there are always 42
producer decisions per cycle, while orderer decisions go from 106 (E 2) to 441 (E 14) per cycle.
Pooled numbers would hide a cost that grows on one role only.

Then for each axis, measure and stage the run fits a straight line through log(time) against
log(axis value); its *slope* says how cost scales (1 = proportional, 2 = quadratic, 0 = flat).
The pre-registered rule (D7, from the D2 *bounded per-decision cost* rule): a per-decision slope
(pooled or per role) above **1.1** on the fanout or the consumer axis flags the stage as
"Superlinear stage", i.e. a decision would get more expensive faster than the choice or the
load it handles. Per-cycle slopes are reported but never flagged, because a cycle holds more
decisions when the world grows.

**The result: no stage is superlinear; per decision the cost barely moves.** Times below are
pod microseconds (µs) per decision, recomputed from `timings.csv` (median over repeats, mean
over the 3 worlds):

| cell | pooled decision, all stages | producer decision, policy stages | orderer decision, policy stages | per cycle, all stages |
|---|---|---|---|---|
| P12-E2-k1 | 248 µs | 357 µs | 173 µs | 10.3 ms |
| P24-E2-k1 | 278 µs | 364 µs | 211 µs | 23.3 ms |
| P42-E2-k1 | 333 µs | 399 µs | 268 µs | 49.3 ms |
| P42-E4-k1 | 355 µs | 512 µs | 276 µs | 72.2 ms |
| P42-E7-k1 | 379 µs | 680 µs | 286 µs | 109.0 ms |
| P42-E14-k1 | 395 µs | 1,016 µs | 285 µs | 190.9 ms |
| P42-E4-k2 | 356 µs | 518 µs | 279 µs | 79.5 ms |
| P42-E4-k4 | 374 µs | 547 µs | 295 µs | 98.8 ms |

- **Fanout (choice), the flagged axis.** Per pooled decision the slope is 0.09 (333 → 395 µs
  from E 2 to E 14). Per **orderer** decision it is flat, 0.03 (268 → 285 µs): an ordering
  stream that can choose among 14 producers costs the same as one choosing among 2. Per
  **producer** decision it grows with slope **0.48** (policy stages 399 → 1,016 µs, 2.5× for
  7× the fanout); its stages are `obs` 0.44, `forward` **0.54** (the largest per-decision stage
  slope on a flagged axis in the run) and `decode` 0.32. Sublinear, far under 1.1. Why the
  producer's observation and forward grow with fanout (e.g. more trading partners in its view) is
  not profiled; only the stage timers are measured.
- **Consumers (load), the flagged axis.** Every per-decision slope is ≤ 0.25: pooled total
  0.04, producer 0.05, orderer 0.05; the largest are `domain` per pooled decision 0.25 and
  orderer `decode` 0.11. Doubling the consumers adds streams, not cost per stream.
- **Producers (not a flagged axis).** Per pooled decision 0.23 (248 → 333 µs); per orderer
  decision 0.34, driven by its `obs` (0.52: 63 → 120 µs) and `forward` (0.30); per producer
  decision 0.09. **Per cycle** the slope is **1.24** (10.3 → 49.3 ms for 3.5× the producers),
  with `obs` 1.48 and `harness` 1.38: a cycle grows a little faster than the economy. Measured
  next to it: decisions per cycle grow in proportion (41 → 148), and `obs` per orderer decision
  with slope 0.52 (≈ √P); reading these two as the cause is from stage timers, not a profile.
  Reported, not flagged (D7 flags only per-decision slopes on fanout and
  consumers).
- **Per cycle on the other axes**: fanout 0.70 (49 → 191 ms), consumers 0.24 (72 → 99 ms).

**What dominates.** The observation build and the network forward take ≈ 80% of the time in
every cell (`obs` 27.9–41.0%, `forward` 38.0–48.6% of the wall), `decode` 8.5–15.1%, the
simulation itself (`domain`) only 1.5–3.5%, the harness 4.8–10.0% (largest at E 14). The cost of the deployed path is the
policy side, not the economy model.

**Against DIAGNOSTICS 17 (the earlier timing).** Before M1 task 8.4, the cost was every orderer
stream rebuilding every producer's rows. If that were back, the cost of one orderer decision
would grow about in proportion to the producers at fixed fanout (slope near 1). Here, at E = 2,
the orderer decision's slope on producers is 0.34 (policy stages 173 → 268 µs), its `obs` 0.52:
no per-stream × per-producer blow-up on the deployed path up to 42 producers. What growth is left
sits in that place, the orderer's observation, which is what the pending D2 refactor's
*eligible-only rows* would remove; that attribution is a reading of the stage timers, not a
profiled diagnosis (nothing failed, so none was run). For context only, not like-for-like
(different subject, range and host): DIAGNOSTICS 17's pre-fix cycle grew ≈ 150× from 12 to 120
producers (slope ≈ 2.2), timed locally with `entity_planner/proportional`; this run's
portable-stack cycle grows 4.8× from 12 to 42 on the pod (slope 1.24), where `forward` is 38–49%
of the wall. A pre-fix path would also show a lower slope over 12 → 42, so 1.24 against 2.2 is not
by itself evidence that the fix holds.

**What changes because of it.**
- **Bounded per-decision cost rule (D2): holds at ≤ 42 producers.** No stage violates it on
  either flagged axis, in any measure; the closest is the producer decision's `forward` on
  fanout at 0.54 (its total 0.48), less than half the threshold. Timing noise cannot change that.
  Repeats of a world differ in wall time by 0.5–15.2%, apart from two first-job warm-ups (one per
  worker: P12 world 1000 and world 1001, repeat 0, 68% and 67% above the other repeats; the median
  removes them); single stages differ by up to 24% (`forward`, P42-E4-k1 world 1000) and 23%
  (`domain`, P24-E2 world 1001). Refitting the flagged-axis per-decision slopes on each repeat
  alone, or on every cell's minimum or maximum episode, keeps the maximum at 0.52–0.54 (the same
  producer `forward` on fanout; 0.56 with the minimum at the low-fanout and the maximum at the
  high-fanout cells), against 1.1.
- **The D2 refactor is not needed for cost at the 12 → 42 target.** It stays a separate change
  (D7); its eligible-only rows would address the orderer observation's growth with P, which
  matters only if economies above 42 return (dropped by the user for now).
- **M4/M5 cost.** A horizon-100 portable episode at 42 producers costs ≈ 5 s of pod time at
  E 2 and ≈ 19 s at E 14 (per-cycle × 100); 12 producers ≈ 1 s. Training and evaluating the
  deployed stack at the target is cheap; M4's cost will be set by the teacher (EXP-0053: hundreds
  of seconds per episode), not by the deployed path.
- **Sizing.** The pod ran these batch-1 torch episodes at ≈ 0.37 of the local wall (10.2's local
  h12 × 1.69), not D8's 0.646 measured on a torch-free mix, so the run took 0.53–0.59 of its
  sizing per cell. Future torch-heavy sizing on `cpu5c` can start from ≈ 0.4, a factor measured
  once, on this CPU model (EPYC 9575F) under host contention (see Limits).

**Limits.** Calm worlds only (no shocks, fixed demand), horizon 20, one family (`sparse_wide`),
3 points per axis (4 on fanout). The container's `loadavg` (290–334) is the 128-CPU host's,
not this pod's. With more runnable work than the host's cores, other processes may have shared
the 4 CPUs of the affinity mask (a mask of 4 does not reserve them); the noise bounds above are
what the run shows of that.
Run on 4 vCPU with 2 workers (the fallback shape), still leaving D7's 2 idle cores.

## Slopes (driver output, `slopes.csv`)

| axis | measure | stage | slope | flag |
|---|---|---|---|---|
| producers | decision | obs | 0.4654 |  |
| producers | decision | forward | 0.1377 |  |
| producers | decision | decode | -0.01 |  |
| producers | decision | domain | -0.1138 |  |
| producers | decision | harness | 0.3689 |  |
| producers | decision | total | 0.2328 |  |
| producers | prod_decision | obs | 0.3783 |  |
| producers | prod_decision | forward | -0.0405 |  |
| producers | prod_decision | decode | -0.0699 |  |
| producers | prod_decision | total | 0.0874 |  |
| producers | order_decision | obs | 0.515 |  |
| producers | order_decision | forward | 0.3016 |  |
| producers | order_decision | decode | 0.0253 |  |
| producers | order_decision | total | 0.3442 |  |
| producers | cycle | obs | 1.4772 |  |
| producers | cycle | forward | 1.1496 |  |
| producers | cycle | decode | 1.0019 |  |
| producers | cycle | domain | 0.8981 |  |
| producers | cycle | harness | 1.3807 |  |
| producers | cycle | total | 1.2447 |  |
| fanout | decision | obs | 0.1062 |  |
| fanout | decision | forward | 0.0525 |  |
| fanout | decision | decode | -0.0386 |  |
| fanout | decision | domain | -0.1524 |  |
| fanout | decision | harness | 0.3646 |  |
| fanout | decision | total | 0.0903 |  |
| fanout | prod_decision | obs | 0.4406 |  |
| fanout | prod_decision | forward | 0.5371 |  |
| fanout | prod_decision | decode | 0.3232 |  |
| fanout | prod_decision | total | 0.4821 |  |
| fanout | order_decision | obs | 0.0763 |  |
| fanout | order_decision | forward | -0.0038 |  |
| fanout | order_decision | decode | -0.0723 |  |
| fanout | order_decision | total | 0.0345 |  |
| fanout | cycle | obs | 0.7151 |  |
| fanout | cycle | forward | 0.6614 |  |
| fanout | cycle | decode | 0.5703 |  |
| fanout | cycle | domain | 0.4566 |  |
| fanout | cycle | harness | 0.9735 |  |
| fanout | cycle | total | 0.6992 |  |
| consumers_per_producer | decision | obs | 0.0756 |  |
| consumers_per_producer | decision | forward | -0.033 |  |
| consumers_per_producer | decision | decode | 0.0778 |  |
| consumers_per_producer | decision | domain | 0.2537 |  |
| consumers_per_producer | decision | harness | 0.1458 |  |
| consumers_per_producer | decision | total | 0.0383 |  |
| consumers_per_producer | prod_decision | obs | 0.1204 |  |
| consumers_per_producer | prod_decision | forward | 0.0035 |  |
| consumers_per_producer | prod_decision | decode | 0.0383 |  |
| consumers_per_producer | prod_decision | total | 0.0493 |  |
| consumers_per_producer | order_decision | obs | 0.0825 |  |
| consumers_per_producer | order_decision | forward | 0.0062 |  |
| consumers_per_producer | order_decision | decode | 0.1093 |  |
| consumers_per_producer | order_decision | total | 0.0505 |  |
| consumers_per_producer | cycle | obs | 0.2729 |  |
| consumers_per_producer | cycle | forward | 0.1643 |  |
| consumers_per_producer | cycle | decode | 0.2751 |  |
| consumers_per_producer | cycle | domain | 0.451 |  |
| consumers_per_producer | cycle | harness | 0.3431 |  |
| consumers_per_producer | cycle | total | 0.2356 |  |

## Details

- **Launch.** `run_pod_training.py --create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu
  --python python3.12 --vcpu 4 --mem 8 --disk 30 --verify-source-commit`, train
  `-u -m tokenomics.eval.deployed_cost --prod-ckpt … --orderer-ckpt … --workers 2
  --source-commit 5520746… --out-dir /workspace/tokenomics/runs/exp0055`. Two `--vcpu 8 --mem 16`
  creates (18:54:02Z, 18:54:15Z) returned HTTP 500 "no longer any instances available" → the
  4 vCPU fallback of sizing.md §EXP-0055 (hard stop 1 billed vCPU-h = 15 min of pod life).
- **Pod life.** Pod `6vodd7hzik4ert` created 18:54:20Z, driver started 18:56:25Z (2.1 min),
  driver 60.3 s, terminated 18:57:44Z → 3.4 min × 4 vCPU = **0.23 billed vCPU-h** (sized 0.32
  central on 4 vCPU). `runpodctl pod list` = [] after.
- **Host** (`spec.json`): AMD EPYC 9575F, `vcpu` 4, `affinity` 4, `cgroup_cpu_raw` `max 100000`
  (no quota), `os_cpu_count` 128, `loadavg_start` 290.8, `loadavg_end` 334.4 (host-wide).
- **Pod vs sizing** (mean of world medians, horizon 20): P12-E2 0.21 s (sized 0.43), P24-E2 0.47
  (0.81), P42-E2 0.99 (1.67), P42-E4 1.44 (2.52), P42-E7 2.18 (3.71), P42-E14 3.82 (6.59),
  P42-E4-k2 1.59 (2.72), P42-E4-k4 1.98 (3.70). Sum of episode walls 114.5 s.
- **Stage shares of the wall** (all 9 episodes of a cell): P12-E2 obs 28 / forward 49 / decode 15
  / domain 4 / harness 5%; P42-E2 38 / 42 / 11 / 2 / 6; P42-E14 40 / 40 / 9 / 2 / 10; P42-E4-k4
  41 / 38 / 11 / 3 / 7.
- **Decisions** per episode are identical across repeats (calm worlds): e.g. P12-E2 820 (240
  producer + 580 orderer), P42-E14 9,660 (840 + 8,820).
- **Independent check**: `/srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/exp0055_check.py`
  (stdlib, no driver import) → ALL PASS: 72 rows = every cell × seed × repeat once; horizon 20,
  no death; identical decisions / units across repeats; stages sum to the wall (worst 0.0015%);
  producer + orderer = the pooled policy stage; `cells.csv` and all 60 `slopes.csv` rows (slopes to
  4 dp, flags) recompute from `timings.csv`; both pins in `spec.json` and equal to the staged files.
- **Record precision.** `cells.csv` writes seconds with 6 decimals, so its per-decision columns
  (≈ 10⁻⁵ s) keep only 1–3 significant digits (`domain_per_decision_s` 0.000008). `slopes.csv` is
  computed before rounding and is exact; per-decision values quoted above come from `timings.csv`
  (seconds per episode stage, 4–6 significant digits). A formatting change for the driver, not a
  measurement problem.

## Conclusions

No stage is superlinear: the largest per-decision slope on a flagged axis is the producer
decision's `forward` on fanout, 0.54 (its total 0.48; 0.52–0.56 under the per-repeat and min/max
refits), against 1.1. Orderer decisions are flat on fanout (0.03) and every per-decision slope on
the consumer axis is ≤ 0.25. The bounded per-decision cost rule holds at the 12 → 42 target;
obs + forward are ≈ 80% of the wall, the simulation 1.5–3.5%. On producers at fixed fanout the
orderer decision grows with slope 0.34 (`obs` 0.52), not the ≈ 1 a per-stream × per-producer
rebuild would give; the per-cycle slope is 1.24 (context only: DIAGNOSTICS 17's pre-fix ≈ 2.2 is
not like-for-like, different subject, range and host). The D2 refactor is not needed for cost
at ≤ 42; a horizon-100 episode at 42 producers costs 5–19 pod s.
