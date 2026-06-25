# EXP-0008 elite-set distillation — candidate vs m2_stream_stochastic

PROMOTE_RECOMMENDED=false

Champion-iteration loop (`tokenomics.train.champion_iterate --distill elite_set`): rolled
out the registry champion, relabeled with the search teacher keeping the **per-state elite
routing set** (within `elite_eps=0.1` of the best, capped at `elite_k=6`) instead of the
single argmax, distilled a stochastic candidate, scored both on the held-out suite at 40
episode seeds.

(The driver's auto-generated header reads "EXP-0007B"; that string is hardcoded in
`champion_iterate._result_md` and is corrected here — this is the EXP-0008 run.)

## Gate

Blocked by:
- held-out mean 511.7 does not beat champion 513.4
- hard cell s9 regressed: 516.0 < champion 524.5

## Held-out (candidate vs champion, 40 episode seeds)

| seed | cand cu | champ cu | Δ | env cu | cand ratio_env | dead |
|---|---|---|---|---|---|---|
| 6 | 492.9 | 490.9 | +2.0 | 473.4 | 1.041 | 0.0 |
| 7 | 478.6 | 472.2 | +6.4 | 436.3 | 1.097 | 0.0 |
| 8 | 559.3 | 566.0 | −6.7 | 533.5 | 1.048 | 0.0 |
| 9 | 516.0 | 524.5 | −8.5 | 498.7 | 1.035 | 0.0 |
| **mean** | **511.7** | **513.4** | **−1.7** | | | |

## Conclusions

**Reject promotion. Elite-set distillation (K=6, ε=0.1) is slightly net-negative vs the
M2-v3-stochastic champion, and notably *regresses the random-dominated cells it was meant
to help*.** The candidate is a valid policy — zero death, clears the strict envelope on all
four held-out topologies (1.041 / 1.097 / 1.048 / 1.035) — it simply does not beat the
champion.

The per-seed split is the finding. Elite-set helps where the envelope-best is
`proportional`-like (s6 +2.0, s7 +6.4) but loses where the envelope-best is
`stock_aware+random` (s8 −6.7, s9 −8.5). s9 is a hard cell and regressed below the
champion, tripping the no-hard-cell-regression gate on its own. This is the **opposite** of
the EXP-0008 hypothesis: preserving multiple near-optimal routings was supposed to lift the
random-dominated cells (the EXP-0006B residual that the *stochastic* fix already addressed).
Instead, widening the fitted Gaussian over a 6-member elite set pulls the mean toward the
elite *centroid* and inflates the within-state std, which on the random-dominated cells
appears to wash out the specific high-value routings the single-argmax-plus-learned-std
champion already captures — a small but consistent loss exactly there.

Mechanistically: EXP-0006B already distills routing *diversity* via the learned `log_std`
around the argmax. Adding an elite **centroid** target on top does not add useful diversity
on these topologies; it shifts the mean off the argmax and over-spreads, so the net effect
is a wash-to-loss. Elite-set is therefore **not the margin lever** at these settings.

Not ruled out (left to a cheaper follow-up, not pursued here): a much **tighter** elite
(small ε, e.g. 0.02, or `elite_k=2`) that keeps only true near-ties might avoid the s8/s9
regression while retaining the s6/s7 gains. If pursued, run at `--eval-seeds 10` first and
only promote-audit at 40. Absent that, the champion stays **M2-v3-stochastic**.

## Reproduce

```
python run_pod_training.py --pod-id <id> --cpu --python python3.12 \
  --train "-u -m tokenomics.train.champion_iterate --registry champion.json \
           --out-dir /workspace/tokenomics/boundary_runs/exp0008 \
           --distill elite_set --eval-seeds 40" \
  --pull "/workspace/tokenomics/boundary_runs/exp0008:./experiments/EXP-0008-elite-set-distillation"
```

- Compute: Runpod CPU pod `knm304p919l6zb` (runpod/base:0.7.0-ubuntu2004, python3.12,
  torch 2.12.1+cpu) via `run_pod_training.py --cpu --python python3.12` (DETACHED
  client-server). Wall time ~1h45m (eval-seeds 40; label-gen + the redundant candidate
  eval dominate — the latter fixed post-run, see below).
- Teacher: L12/K12 routing search, candidates `even_split,proportional,obs_price`, 2 DAgger
  rounds, 3 label-seeds, elite `K=6 ε=0.1`. Producer fixed at EXP-0004 v4.
- Artifacts: `artifacts/candidate.pt` (the rejected elite-set candidate),
  `artifacts/checkpoint.pt` (driver copy of the same), `artifacts/heldout.csv`,
  `summary.csv` (candidate per-seed rows).

## Note (driver cleanup applied after this run)

This run double-evaluated the candidate (train-step `run_orderer_improve` self-eval +
driver gate eval), wasting one 40-seed held-out pass. Fixed afterward: `run_orderer_improve`
gained an `eval_seeds=0` skip path and `_default_train_fn` now passes it, so the gate eval
is the single candidate eval. Does not affect this run's numbers.
