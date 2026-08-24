# EXP-0042 - Bounded Online Rollout

**Date:** 2026-08-24 | **Status:** COMPLETE — **PASS_BOUNDED_SEARCH**, depth 2 promoted

## Question

Can confidence-bounded online scenario search improve closed-loop delivery over
`blend_0.5`, and what is the shallowest sufficient rollout depth in `{1,2,3}`?

## Method

The search starts from and falls back to `blend_0.5`. It searches only ambiguous
top-two base routes, samples common future-demand scenarios at checkpoints 4/8/16,
and applies a producer bump only after a paired confidence lower bound clears every
alternative. The shallowest depth that clears all value and safety gates is promoted.

The closed-loop comparison used 10 fresh paired episode seeds per task and four arms:
the floor plus rollout depths 1, 2, and 3. The producer, topology set, horizon, candidate
action, confidence rule, fallback, scenario budget, task seeds, and gates were frozen
before pod launch. Every completed episode was checkpointed immediately.

The gate required an arm to beat the floor on at least two tasks, average at least 5%
relative improvement, avoid a task regression beyond 5%, avoid any paired episode
regression beyond 10%, and not increase deaths on any task.

## Result

**Depth 2 passes every gate and is promoted.** It beat `blend_0.5` on all three tasks,
improved the task-average relative delivery by **15.3%**, kept the worst paired episode
to **-7.3%**, and did not increase task death rates.

| Task | Floor units | Depth 1 | Depth 2 | Depth 3 | Depth-2 gain | Depth-2 worst episode | Death rate floor / d2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s45 | 463.1 | 503.4 | **550.0** | 605.2 | **+18.8%** | -7.3% | 0.2 / 0.2 |
| s77 | 716.3 | 763.4 | **813.8** | 927.5 | **+13.6%** | -0.4% | 0.0 / 0.0 |
| s29 | 848.0 | 883.3 | **963.0** | 1029.7 | **+13.6%** | -3.4% | 0.0 / 0.0 |

| Depth | Tasks beating floor | Mean relative gain | Worst task | Worst paired episode | Death-safe | Gate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3/3 | +6.5% | +4.2% | **-14.4%** | yes | **FAIL** |
| 2 | 3/3 | **+15.3%** | +13.6% | -7.3% | yes | **PASS** |
| 3 | 3/3 | +27.2% | +21.4% | 0.0% | yes | **PASS** |

Depth 1 improves every task mean but is disqualified by one s77 episode that regressed
14.4%. Depth 3 is the strongest value arm, with no paired regression, but the registered
selection rule promotes the **shallowest** passing arm. It also costs materially more:
mean per-episode wall time was 532–1461 seconds for depth 3 versus 267–758 seconds for
depth 2 on the 16-worker host.

The confidence machinery is conservative in action count. At depth 2, a typical episode
searched 1,002–1,598 ambiguous streams but overrode only 22–45. Between 84% and 90% of
searched streams were exact-zero decisions resolved at the first checkpoint, and roughly
8%–10% remained unresolved and fell back. The resulting policy is useful, but still far
too simulator-heavy for a low-latency production orderer without pruning or a rollout
surrogate.

## Validity and compute

- **120/120 exact paired rows**: 3 tasks x 10 seeds x 4 arms; local artifact-only
  recomputation exactly matches `result.json`.
- Source commit: `ed3ae95d5fe69547a8811adc3821e48eb3e17331`; tree SHA-256
  `6e27dd193aa7817b5089168dc7e74420fc31d94de7e722c12735eadae006045c`.
- Pinned archive and producer hashes match the registered EXP-0040 inputs.
- Authoritative run: 16 workers on a 16-vCPU RTX 4000 Ada host; the GPU was unused and
  the host was selected for CPU availability. End-to-end experiment wall time was about
  65 minutes. A partial 2-vCPU run was retired only after overlapping floor/search rows
  matched exactly across hosts.
- All three pods created during execution were deleted after the local artifacts and
  independent recomputation passed. Total compute spend was approximately **$0.42**.

Authoritative artifacts: [episodes](artifacts/episodes.csv),
[task metrics](artifacts/metrics_by_task.csv), [arm metrics](artifacts/metrics_by_arm.csv),
[result](artifacts/result.json), [run spec](artifacts/spec.json), and
[run log](artifacts/run.log). Raw episode SHA-256:
`7c847148e3771cdce42056783e831edf29d18d00787e3f45e01f215d9584a883`.

## Conclusion

EXP-0040's surviving direction works: future-sensitive marginal value is useful when
sampled online. Promote **depth-2 confidence-bounded rollout** as the research orderer for
these fixed topologies. Keep direct GNN imitation cancelled. The next useful model role is
cost reduction—candidate pruning or a scenario-conditioned rollout surrogate—with depth 2
and `blend_0.5` fallback as the value and safety authorities.

Pre-run engineering note: s29 seed 3300 was used for a complete depth-1 timing and
policy smoke (753 floor to 821 search units, 64.7 seconds locally). It is excluded
from the primary gate; the untouched registered s29 block is 3301–3310, and the CLI's
bounded smoke uses non-gating seed 3299.
