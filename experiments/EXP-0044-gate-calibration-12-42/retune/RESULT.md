# EXP-0044 retune probe (task 10.3): dense capacity scale

> **PROBE — calibration evidence, not an acceptance read.** Baseline-only headroom at several
> capacity scales (design D10); no frozen files.

**Status: NO SCALE IN THE GRID WORKS.** Cutting dense capacity to 0.8× does not bring dense-12
below the 0.95 ceiling (probe median 0.955–0.962 at every scale). A follow-up diagnostic finds
why: the best baseline's shortfall is mostly a warm-up effect, and capacity is far from binding.
No scale is recommended; the next step is a decision for the supervisor and user (§ What
changes). Run 2026-10-27 at commit `f5b87c63269524fbc2bcfc6b348641646174c744`.

## In plain English

**The question.** EXP-0044's screen failed because the dense 12-producer worlds leave the
hand-written baselines too little room: the best baseline already delivers a median 95.4% of
ordered units, above the 0.95 ceiling. The approved fix (tasks 10.2–10.5) shrinks every dense
producer's capacity by a fixed factor at both sizes, then screens again. This probe asks which
factor is the mildest that brings both dense sizes safely under the ceiling.

**What was measured.** For each scale (1.0, 0.95, 0.9, 0.85, 0.8), every dense development world
at 12 and 42 producers was rebuilt with that capacity factor and played by the seven baselines
that set the best result on dense in EXP-0044. Then the same headroom statistic was taken: per
world, the mean over episodes of the best baseline's fill rate (delivered / ordered units); per
size, the median over the 12 worlds.

**The approximation.** To fit about 2 vCPU-hours, the probe is narrower than the gate:
- only those 7 members (not all 40);
- random members ran 1 replicate (not 5);
- at most 3 episodes per world (shocked worlds have 10).
Its medians estimate the screen's; the full rescreen (10.5) would be binding. The scale-1.0 row
checks the estimate: 0.9548 vs the screen's 0.9539 at dense-12, and 0.9259 vs 0.9247 at
dense-42. The probe reads slightly high, as expected from fewer replicates.

**The numbers.**

| capacity scale | dense-12 median fill | dense-42 median fill |
|---|---|---|
| 1.0 | 0.9548 | 0.9259 |
| 0.95 | 0.9549 | 0.9212 |
| 0.9 | 0.9618 | 0.9211 |
| 0.85 | 0.9583 | 0.9162 |
| 0.8 | 0.9583 | 0.9116 |

Dense-12 does not move: it stays at 0.955–0.962 at every scale, with no downward trend.
Dense-42 falls slowly, by 1.4 points over the whole grid. The target (both medians ≤ about
0.93) is not met by any scale.

**Why: what limits the fill (diagnosed).** A follow-up diagnostic checked where the undelivered
units come from. It replayed the strongest baseline (`entity_planner/random`, 51 of 64 dense-12
episode wins in EXP-0044) on all 24 dense development worlds at horizon 20 and horizon 100. The
difference between the two runs gives the fill rate of cycles 21–100 alone. It used one episode
per world, replicate 0, at scale 1.0.

| size | median fill (h = 100) | share of shortfall in cycles 1–20 | median fill rate, cycles 21–100 | lowest, cycles 21–100 |
|---|---|---|---|---|
| 12 | 0.952 | 62% (180 of 292 units) | 0.977 | 0.928 |
| 42 | 0.904 | 86% (1,951 of 2,261 units) | 0.994 | 0.893 |

A single-world horizon sweep shows the same shape: on dense-12 calm-00 the best baseline
delivers every ordered unit in cycles 21–50, and on dense-42 calm-00 it delivers 99.4–99.7%
after cycle 20 (`horizon_sweep.out`).

So, in this sample:
- Most of the shortfall that the headroom check sees is **warm-up**: the depth-3 supply chain
  needs its first cycles to fill, and orders placed then expire (TTL 2). The wider 42-producer
  economy warms up more slowly. That is why dense-42 shows more "headroom" than dense-12, not
  because it is harder to serve later.
- After warm-up, the best baseline serves nearly all demand. Capacity is far from binding: the
  family's demand is about 6% of its initial top-layer throughput (task 8.7 ratio 0.063). A
  0.8× cut leaves the top layer roughly 12× over-provisioned, which matches the flat dense-12
  column.

This diagnosis rests on one episode per world for the best member, at scale 1.0. Whether a much
deeper cut (toward the demand/throughput ratio) would bind is expected from the ratio, but not
measured.

**What changes.** A dense capacity scale in the approved range cannot fix the dense-12 headroom
failure. It would have to be cut by an order of magnitude before capacity binds, and that is
untested. No scale is set as the default (10.4 is blocked). The levers that address the
diagnosed cause are different, and choosing among them is the supervisor's and user's call:
- change what the headroom check measures, for example fill after a warm-up window;
- raise demand pressure instead of cutting capacity;
- probe much deeper capacity cuts (≈ 0.05–0.3), with a probe of the same shape.

## Details

**Command** (repo root; pod created by curl as in EXP-0044, `--pod-id`, PATH shim
`/tmp/rpshim`, launcher verified the clean tree and injected the commit):

```
python run_pod_training.py --pod-id 4e49w7lmdwj727 --ready-timeout 900 --cpu --python python3.12 \
  --verify-source-commit --train "-u -m tokenomics.eval.gate --mode probe \
  --scales 1.0,0.95,0.9,0.85,0.8 --families dense --sizes 12,42 \
  --members entity_planner/blend_0.5,entity_planner/obs_price,entity_planner/proportional,entity_planner/random,obs_stock_aware/blend_0.5,obs_stock_aware/random,stock_aware/random \
  --replicates 1 --episodes-per-world 3 --workers 8 --source-commit {SOURCE_COMMIT} \
  --out-dir /workspace/tokenomics/runs/exp0044_retune" \
  --pull "/workspace/tokenomics/runs/exp0044_retune/:./experiments/EXP-0044-gate-calibration-12-42/retune/"
```

The members are the per-episode argmax members on dense in EXP-0044's `episodes.csv`
(`gate_cli.baseline_setters`). Inputs: registry 310673c3…, roster 8791e24b…, horizon 100.

**Outputs.**
- `probe.json`: bound configuration.
- `probe_headroom.csv`: one row per scale × size.
- `scale-<s>/`: `spec.json`, `episodes.csv` (504 rows each: 24 worlds × ≤ 3 episodes × 7
  members), `headroom.csv`.

**Diagnostics** (local, one-shot, at the same commit):
- `horizon_sweep.py` → `horizon_sweep.out`: three worlds, horizons 3–100.
- `warmup_share.py` → `warmup_share_{12,42}.jsonl`: 24 worlds, horizons 20 and 100. No episode
  died.

**Compute.** Runpod secure cloud, cpu5c, 8 vCPU / 16 GB, `runpod/base:0.7.0-ubuntu2004`, pod
`4e49w7lmdwj727`.
- Pod created 15:58:57 and deleted ≈ 16:11 UTC: ≈ 12.5 min ≈ 1.7 vCPU-h billed, ≈ $0.06.
- Probe execution 15:59:34 → 16:10:14 (10.7 min ≈ 1.4 vCPU-h), against an estimate of ≈ 2.0.
- Setup took about 30 s; the launcher's SSH wait passed immediately with the shim.
- `runpodctl pod list --all` returned `[]` afterwards.

## Conclusions

- No capacity scale in 0.8–1.0 brings dense-12 under the 0.95 ceiling; the approved retune
  lever, as sized, does not work.
- Diagnosed cause, in this sample: the headroom statistic is dominated by warm-up loss (62% at
  12, 86% at 42), and capacity is about 12× above demand even at 0.8×.
- 10.4/10.5 are blocked pending a decision on the lever: what the headroom check measures,
  demand pressure, or a much deeper capacity cut.
