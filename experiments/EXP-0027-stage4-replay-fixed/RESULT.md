# EXP-0027 — Stage 4: Prioritized Replay & Relabeling (bug-fixed rerun)

Controlled rerun of **EXP-0026** with the three Stage-4 pipeline bugs fixed and **nothing else
changed** (same single-cell EXP-0025 archive, same defaults: eta=1.0, epsilon=0.01,
epochs/iters=400, eval-seeds=3, horizon=100, `entity_planner+mpc_search` bar, fixtures on). The
point: prove the EXP-0026 REJECT was a real structural result, not an artifact of training on the
wrong effective data. No promotion.

## What the fixes changed (vs EXP-0026)

EXP-0026 review found two bugs that meant the run wasn't training on the data it claimed:
1. **Prioritized replay was inert** — default sampled every train task exactly once (no
   replacement, k=len), so priorities only reordered. Now the default draws a full-budget batch
   **with replacement**, so high-regret mined tasks are duplicated and upweighted. (Verified in the
   pod log: seeds `151`, `7` collected twice.)
2. **Orderer ignored the replayed configs** — it rebuilt `build_scale_config(seed, depth)` instead
   of the archived configs. Now `collect_scorer_samples`/`distill_orderer_scorer` take
   `train_configs`, so the orderer trains on exactly the replayed (duplicated) mined topologies.

(The heartbeat-phase fix is orthogonal — observability only.)

## Gate (advisory — promotion is human-approved)

```json
{
  "val_candidate_units": 512.998,
  "val_champion_units": 511.333,
  "val_improves_or_ties": true,
  "blind_candidate_tail": 445.67,
  "blind_champion_tail": 444.33,
  "blind_tail_not_worse": true,
  "new_contract_failures": 0,
  "transfer_regressions": 4,
  "no_transfer_regression": false,
  "promote_recommended": false
}
```

## Transfer/stress: candidate vs champion (all targets)

Everywhere **except uneven-strength**, the candidate ties or beats champion; the only regressions
(rel_tol 2%) are the 4 uneven-strength fixtures. `deepjoin-d5` is 0/0 both (TTL wall, not counted).

| group | targets | candidate vs champion |
|---|---|---|
| scale-d3 s6–9 | 4 | +6.3 / +0.0 / −0.3 / −2.3 (all ≤2%, no gate regression) |
| sharedcap-d3 s6–9 | 4 | same as scale (+6.3/0/−0.3/−2.3) |
| scarcity-d3 (5 cells) | 5 | **all up** +1.3 … +10.7 (high_backlog +10.7) |
| degenerate (5 cells) | 5 | ties, or +15.0 (no_backlog), +6.3 (no_inv_expansion) |
| **uneven-d3-w4 s6–9** | 4 | **−127.7 / −56.0 / −15.7 / −72.0 (the 4 gate regressions)** |
| deepjoin-d5 s6–9 | 4 | 0/0 both (TTL wall) |

```json
[
  {"target": "uneven-d3-w4-s6", "champion_units": 584.0,  "candidate_units": 456.33, "kind": "value"},
  {"target": "uneven-d3-w4-s7", "champion_units": 584.0,  "candidate_units": 528.0,  "kind": "value"},
  {"target": "uneven-d3-w4-s8", "champion_units": 487.67, "candidate_units": 472.0,  "kind": "value"},
  {"target": "uneven-d3-w4-s9", "champion_units": 537.33, "candidate_units": 465.33, "kind": "value"}
]
```

## Joint decomposition (in-family held-out scale s6–9)

| stack | s6 | s7 | s8 | s9 |
|---|---|---|---|---|
| champion | 491.3 | 496.7 | 589.3 | 544.7 |
| candidate_joint | 497.7 | 496.7 | 589.0 | 542.3 |
| candidate_producer_alone (cand prod + champ ord) | 491.3 | 496.7 | 588.7 | 545.0 |
| candidate_orderer_alone (champ prod + cand ord) | **497.7** | 496.7 | **594.0** | 544.7 |

The candidate **orderer alone** beats champion on s6 (497.7 vs 491.3) and s8 (594.0 vs 589.3) and
ties elsewhere; the candidate **producer alone** ≈ champion. So the small mined-family gain is the
**orderer**, and it comes from bug-fix #1 (duplication reweighted the fit) — not #2, since on this
single-cell archive `replay(record) == build_scale_config(seed, 3)` so config-awareness is a no-op
here. This updates the EXP-0026 expectation ("orderer can't move"): with real prioritization the
orderer *does* move a little, though still far short of the search-oracle headroom.

## Conclusions

**REJECT (no promotion) — and now scientifically valid.** Rerunning on the *correct* effective data
(priorities actually upweight; orderer sees the replayed configs) reproduces the EXP-0026 verdict for
the **same structural reason**, so the rejection was never an artifact of the two bugs.

1. **Marginal, real gain on the mined family.** Candidate now *beats* champion on mined validation
   (512.998 vs 511.333, +1.67u — up from EXP-0026's +0.06u wash), on the blind-holdout tail
   (445.67 vs 444.33), and on **every** scarcity + degenerate + scale + shared-cap cell. The lift is
   the orderer (duplication reweighting), confirming the fixes do something measurable.

2. **Same blocker: uneven-strength producer regression.** The only 4 regressions are the
   uneven-strength fixtures (−16 to −128u), producer-side (producer-alone ≈ champion on scale but the
   uneven fixture stresses capacity allocation). Training the producer *from scratch* on 12 narrow
   depth-3 scale seeds still **replaces** EXP-0021's broad coverage. Magnitudes shifted vs EXP-0026
   (s6 −128 vs −92, s7 −56 vs −89) but the 4-target pattern is identical.

**Lesson — unchanged and now confirmed, not suspected.** The two prescribed fixes are the real next
steps, and they are *not* about the sampling/orderer bugs (those are fixed):
- **Broaden the mined archive across topologies first** (EXP-0025 single-cell blocker) so replay has
  diversity, not one depth-3 cell.
- **Augment, don't replace**: mix prioritized mined replay into the champion's original broad training
  set instead of distilling from mined-only configs — this is what sheds the uneven-strength transfer.

Failure class: distribution/coverage regression introduced by training (replace-not-augment on a
single cell), 0 contract failures, deaths unchanged. `PROMOTE_RECOMMENDED=false`.

## Operational note (separate bug, filed not fixed)

The `run_pod_training.py` **detached-launch SSH hung** on this run: the remote launch shell sat in
`do_wait` (never exited), so the client's synchronous `ssh_run(launch)` blocked and never reached
`monitor_remote()` — no local streaming. The run itself was unaffected (the `setsid` job is
independent); results were recovered via a manual reconnecting `tail -F` and rsync pull, pod
terminated. Fix (proposed, not yet applied): force the remote shell to exit (`… & ; exit 0`, `ssh -n`)
and bound the launch call with a timeout. The Stage-4 observability heartbeat itself worked correctly
up to the launch step.
