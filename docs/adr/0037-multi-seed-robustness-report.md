# Multi-seed robustness report (median + worst-case across model seeds)

Status: accepted

## Context

The Phase 1 NEXT STEPS ladder, stage 2 ("training robustness"), requires:

- at least 5 independent model seeds;
- separate training / validation / held-out environment seeds;
- **median and worst-case results, not one successful checkpoint**;
- a comparison across learning curves, survival, fulfilled units, return, and
  critic explained variance.

The trainer already separates seeds correctly: rollout env seeds come from the
training RNG, validation seeds are `range(eval_seeds)` (checkpoint selection,
ADR-0011), and the held-out report uses a disjoint range starting at
`HELD_OUT_SEED_OFFSET` (#7). The per-iteration `IterStats` already record the
learning-curve quantities (survival, fulfilled units, return, critic explained
variance).

What was missing is the **cross-model-seed** view. `eval.runner.evaluate`
aggregates over *environment* seeds for one policy as `(mean, std)`. There was no
way to train several model seeds and report the **worst** seed — the figure the
release bar is actually stated against ("better median … than the strongest
observation-compatible baseline", "similar results across at least 5 training
seeds"). A mean (or a single lucky checkpoint) hides a seed that collapsed.

## Decision

Add two pieces, split by the pure/IO boundary so the robustness math is testable
without torch.

1. **`eval.runner.summarize_seeds(per_seed, directions) -> {metric: SeedSummary}`**
   — a pure function over the per-seed metric dicts. `directions[m]` is `"higher"`
   (return, fulfilled units, cap_min) or `"lower"` (dead_rate); the **worst** case
   is the bad-direction extreme (min for higher, max for lower), `best` the other.
   `SeedSummary` carries `median, worst, best, mean, n`. The headline is `worst`
   and `median` — a single outlier cannot lift them the way it lifts a mean.

2. **`train.smoke_ppo.train_seeds(seeds, *, held_out_seeds, report, ...)`** — trains
   each model seed through the full `train` loop (each selects its own
   best-on-validation checkpoint on validation seeds disjoint from the held-out
   range), then scores the selected policy on the **same** held-out seed range so
   the per-seed numbers are comparable and never the figure a checkpoint was picked
   on. `summarize_seeds` rolls them up. Exposed on the CLI as `--num-seeds N`
   (`seed..seed+N-1`); `N=1` keeps the single-run behavior unchanged.

`DEFAULT_SEED_REPORT` summarizes return, dead_rate, consumer_completed,
consumer_completed_units, cap_min — the survival + fulfillment axes the bar names.

## Considered options

- **Report mean/std across model seeds (reuse `evaluate`'s shape).** Rejected: the
  ladder explicitly wants *worst-case* and *median*; a mean±std blurs exactly the
  collapsed-seed failure the report exists to catch.
- **Have `train` return the held-out dict and aggregate in the caller.** Rejected:
  `train` returns the learning curve (`list[IterStats]`); changing its return type
  churns every caller/test. `train_seeds` instead reuses the existing
  `save`→`load_policies`→`evaluate` path on the held-out range — the same path
  `train` already uses for its held-out log line.
- **Parallelize the seeds.** Out of scope; the domain is single-process and the
  sweep is a background run. Recorded as a speedup if sweeps grow.

## Consequences

- The stage-2 bar is now measurable: train `--num-seeds 5 --gate` and read the
  `worst` row. A seed that collapses shows up as a worst-case death rate / return
  far from the median, which a mean would have hidden.
- `summarize_seeds` is reusable for stages 3–6 (generalization / held-out suite /
  scale / stress) — point it at any per-seed metric dicts with a direction map.

### Result (phase1.json, 5 seeds 0–4, 100 iters, `--gate`, 10 held-out seeds)

| metric                   | worst  | median | best   |
|--------------------------|-------:|-------:|-------:|
| return                   | 1124.4 | 1215.9 | 1506.2 |
| dead_rate                |   0.00 |   0.00 |   0.00 |
| consumer_completed       |  100.0 |  100.0 |  100.0 |
| consumer_completed_units |  100.0 |  100.0 |  100.0 |
| cap_min                  |   93.0 |  120.0 |  150.0 |

Per-seed return: 1252.7 / 1124.4 / 1215.9 / 1506.2 / 1138.5 (seeds 0–4).

**The robustness lens changes the verdict.** On survival and fulfillment the policy
is rock-solid: every seed reaches 0% death and 100% consumer fulfillment on the
held-out range. But the stage-2 release bar is also *"better median return than the
strongest observation-compatible baseline"* (ADR-0036: `obs_stock_aware +
even_split` ≈ **1255**), held *"across at least 5 training seeds"*. Against that:

- **worst-case return 1124 < 1255**, and even the **median 1216 < 1255** — only
  seeds 0 (1253) and 3 (1506) clear the obs-compatible bar; 3 of 5 fall short.
- A single previously-reported checkpoint (~1330, one seed) **overstated**
  robustness — exactly the "one successful checkpoint" trap this report exists to
  expose.

The gap is *not* survival or unmet demand (both perfect) — it is the inventory
**holding penalty** (Theory of Constraints, ADR-0025): the lower-return seeds
fulfil everything but over-replenish/over-invest, and the env docks the carried
units. So stage 2 **passes on survival + fulfillment, fails the return bar on a
worst-case (and median) basis.** Closing it is a reward/curriculum-tuning task
(discourage over-stocking) for a follow-up, not an infrastructure gap — the
report now makes the shortfall visible instead of hiding it behind a lucky seed.
