# EXP-0044 — Gate calibration screen at 12 and 42 producers

**Status: HEADROOM FAILED. The dense family at 12 producers is saturated (median best-baseline
fill 0.9539, above the 0.95 ceiling). No frozen files; no X/Y calibration; no pruning.**
Run 2026-10-27 at commit `016695b64a88c14b9bd21e7cea482d9ad89b3349` (OpenSpec change
`add-measurement-gate-harness`, task 10.1).

## In plain English

**The question.** Before any agent can be judged by the measurement gate, the gate needs a
fixed set of development worlds where the hand-written baselines leave room to improve. This
run is that screen. It played every one of the 40 roster baselines (the 12 random ones 5 times
each) on all 48 development gate worlds: two families (dense and sparse-wide), two economy
sizes (12 and 42 producers), calm and shocked demand. That is 256 world-episodes and 22,528
episodes in total, each run at horizon 100 cycles. An episode ends earlier on system death:
8,666 of the 22,528 rows have `dead = True`.

**What was measured.** For each episode, the best baseline is the one that delivers the most
units. Its *fill rate* is the share of the consumers' ordered units it actually delivered.
The *headroom check* (design D10) takes, per family × size, the median over worlds of that
best fill rate. A family passes only if the median is below 0.95. The reason: if the best
hand-written policy already fills 95% or more of demand, a learned agent has almost nothing
left to win, and a 5–10% gate margin could not be shown.

**The numbers.**

| family | producers | median best fill | ceiling | result |
|---|---|---|---|---|
| dense | 12 | **0.9539** | 0.95 | **fail** |
| dense | 42 | 0.9247 | 0.95 | pass |
| sparse-wide | 12 | 0.9342 | 0.95 | pass |
| sparse-wide | 42 | 0.8856 | 0.95 | pass |

The dense 12-producer worlds fail by a small margin (0.0039 above the ceiling). This run
gives one screen of fixed worlds and seeds. Its spread across worlds and episodes is below,
but no repeat screen or confidence interval was computed, so how far the median would move on
other draws is not measured. Within this run the saturation is broad, not a few outlier
worlds:
- in 39 of their 64 episodes the best baseline filled at least 95% of demand;
- per-world means run from 0.911 to 0.990, with calm worlds at a median of 0.971.

For comparison, at dense-42 only 2 of 64 episodes reach 0.95, and at sparse-wide-42 none do.

**Why: what the rows show, and what they do not.** The dense 12-producer world is the old
`build_scale_config` economy with only its maintenance spread out (test
`test_dense_12_shocked_keeps_build_scale_config_draws`). EXP-0033 already found that economy
over-capacitated: the champion reached 98.3% of exact demand. Here the winning baseline is
almost always `entity_planner/random`, the planner producer with a random orderer (51 of 64
episodes): simple planning already saturates it. Why dense-42 keeps headroom is not
isolated. One possible explanation: the 42-producer dense worlds carry slightly more demand
per unit of initial top-layer throughput (+1.6% calm, +2.8% shocked; task 8.7). These rows do
not separate that small difference from other size effects, such as the forced-strong
fraction (4/14 vs 1/4), the consumer count rounding, or the wider layers. Only a scale probe
(task 10.3) would discriminate.

**What else happened.**
- *Replacements: none.* No gate world had a zero best-baseline mean (the lowest world mean
  is 433 units), so the on-demand screen ran no reserve wave (`spec.json` `"waves": []`).
- *Pruning, X and Y: not reached.* The screen stops at a headroom failure by design, before
  pruning and calibration, and writes no `frozen/` files.

**What changes.** M1 cannot proceed to the champion report (task 12.1) with this world set.
The observed failure is dense at 12 producers only. The user-approved response (2026-10-27,
tasks 10.2–10.5; design D10) keeps the 0.95 ceiling and retunes the **dense family at both
sizes** with one family-level scale, so the 12-vs-42 ratio contract (D9) still holds:
- 10.2 adds the scale and a probe path (code);
- 10.3 probes candidate scales on dense development worlds at 12 and 42, on baseline evidence
  only, and picks the mildest one that puts both medians below the ceiling with margin;
- 10.4 makes it the default and regenerates the registry;
- 10.5 reruns this screen (attempt 2).
Sparse-wide passed at both sizes and is not retuned.

## Details

**Command** (from the repo root; the launcher verified a clean tree and injected the commit):

```
python run_pod_training.py --pod-id ir88g7hs4nr4r2 --ready-timeout 900 --cpu --python python3.12 \
  --verify-source-commit \
  --train "-u -m tokenomics.eval.gate --mode screen --reserves on-demand --prune --q 0.05 \
    --workers 16 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0044" \
  --pull "/workspace/tokenomics/runs/exp0044:./experiments/EXP-0044-gate-calibration-12-42"
```

**Inputs.**
- Registry `eval/registry/worlds_v1.json` 310673c3… (unscreened draft, 12/42).
- Roster `eval/registry/roster_v1.json` 8791e24b… (40 members, R = 5).
- Horizon 100, q 0.05, `--prune`, `--reserves on-demand`.
- Source tree sha256 `bc149e23…` over 52 files; the local tree at the commit gives the same
  fingerprint.

**Outputs** (driver layout):
- `spec.json`: pins, provenance, environment, status `headroom-failed`.
- `episodes.csv`: 22,528 rows.
- `screening.json`: baseline means, headroom, the would-be registry version b9edc235…, no
  replacements.
- `headroom.csv`.
- No `frozen/`, by design.
`episodes.csv` (8.8 MB) stays at the EXP top level in normal git, as the driver writes it and
as `--mode recompute` / resume expect (supervisor decision); it is not under `artifacts/`.
`spec.json` records `"experiment": "exp0044"`, the pod-side out-dir name. The launcher pulled the
directory one level deep (`exp0044/`); it was flattened here unchanged.

**Local verification.**
- `check_rows(complete=True)` over the pinned grid: all 22,528 expected rows are present, no
  extras.
- `screen_outcome` recomputed from `episodes.csv` gives a report equal to `screening.json` (dict
  equality) (headroom medians 0.953909 / 0.924736 / 0.934178 / 0.885571).

**Best-baseline detail per cell** (world-mean fills; winners = argmax member counts over 64
episodes):

| cell | median | min | max | calm median | shocked median | episodes ≥ 0.95 | top winners |
|---|---|---|---|---|---|---|---|
| dense-12 | 0.9539 | 0.9108 | 0.9900 | 0.9707 | 0.9539 | 39/64 | entity_planner/random 51, entity_planner/proportional 6 |
| dense-42 | 0.9247 | 0.8086 | 0.9516 | 0.9267 | 0.9210 | 2/64 | entity_planner/random 22, obs_stock_aware/random 21, stock_aware/random 17 |
| sparse-wide-12 | 0.9342 | 0.7738 | 0.9891 | 0.9775 | 0.8969 | 17/64 | entity_planner/blend_0.5 27, entity_planner/random 19 |
| sparse-wide-42 | 0.8856 | 0.8548 | 0.9253 | 0.8903 | 0.8856 | 0/64 | entity_planner/random 55, entity_planner/obs_price 9 |

**Compute** (Runpod, secure cloud, cpu5c, 16 vCPU / 32 GB, image
`runpod/base:0.7.0-ubuntu2004`, python 3.12.11, torch 2.14.1+cpu; `spec.json` records
`image: null` because the launcher's image lookup hit the same 403 described below):
- Job execution: 14:33:54 → about 15:13:00 UTC, 39 min, ≈ 10.4 vCPU-h. Sizing (`sizing.md`
  §A4) expected 13.6 (low) to 16.9 (reference) for the same work.
- Billed pod time: two pods, ≈ 62 min in total ≈ 16.6 vCPU-h, about $0.58 at $0.56/h.
  - Pod `n2cbku634xe06l`: about 11 min, no work done.
  - Pod `ir88g7hs4nr4r2`: 14:22:12 → about 15:13:30, including about 11 min waiting for SSH.
- Both pods were deleted; `runpodctl pod list --all` returned `[]` afterwards.

**Launch deviations (tooling, not the experiment).** Two defects in the launch path were
worked around without code changes; both are queued as a launcher code task.
1. *Pod creation.* `run_pod_training.create_pod`'s REST POST was rejected with 403
   "error code: 1010": Cloudflare blocks Python-urllib's default User-Agent (reproduced: the
   same request with User-Agent `curl/8.5.0` returns 200). The pod was therefore created by
   `curl` with the launcher's own request body (name changed):
   `{"name": "exp0044-gate-screen-2", "imageName": "runpod/base:0.7.0-ubuntu2004",
   "computeType": "CPU", "containerDiskInGb": 30, "volumeMountPath": "/workspace",
   "cloudType": "SECURE", "ports": ["22/tcp"], "vcpuCount": 16, "cpuFlavorIds": ["cpu5c"]}`.
   The launcher then ran with `--pod-id` instead of `--create-pod`, and the pod was deleted by
   hand after the pull.
2. *SSH discovery.* runpodctl 3.0.1 `ssh info` kept reporting "pod not ready: no container
   reported yet" for the running CPU pod, while `runpodctl pod get` already listed its SSH
   endpoint and a direct `ssh` worked. The launcher reads `ssh info`, so the first pod timed
   out after 600 s, with no work done, and was deleted. The run used a PATH shim
   (`/tmp/rpshim/runpodctl`) that answers `ssh info <id>` from `pod get <id>`'s `ssh` block and
   passes every other call through.

Neither deviation touches the measured code path: the source commit, the tree fingerprint and
the gate CLI arguments are exactly the reviewed ones.

## Conclusions

- Three of four family × size cells have headroom. Dense-12 is saturated, consistent with the
  over-capacitation EXP-0033 found in the `build_scale_config` economy, now measured under the
  full 40-member roster.
- The gate's world set is not admissible yet, so M1's champion report (12.1) is blocked.
- Next (D10, user decision 2026-10-27): keep the 0.95 ceiling; retune the dense family at both
  12 and 42 with one family-level scale (10.2 code, 10.3 probe at ≤ 3 vCPU-h, 10.4 default and
  registry), then rescreen (10.5, attempt 2 in this directory). Separately, a launcher fix: set a
  User-Agent in `create_pod`, and fall back to `pod get` for SSH discovery.

## Retune probe (task 10.3)

See [`retune/RESULT.md`](retune/RESULT.md). In short: dense capacity scales 0.95–0.8 leave
dense-12's probe median at 0.955–0.962, so no scale in the approved grid passes. A follow-up
diagnostic finds the best baseline's shortfall is mostly warm-up loss (62% of it in cycles 1–20
at 12 producers, 86% at 42), with capacity about 12× above demand even at 0.8×. 10.4/10.5 wait
for a decision on the lever.
