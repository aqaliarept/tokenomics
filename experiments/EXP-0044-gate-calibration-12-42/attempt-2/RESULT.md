# EXP-0044 attempt 2 — sparse-wide calibration screen (task 10.5)

**Status: HEADROOM PASSED. Frozen files written.** Run 2026-10-27 at commit
`9f29d7307b2c000899703f8f8e59e1bd4b3e2f52` (task 10.4: M1 gates on sparse-wide only).

## In plain English

**The question.** Same as attempt 1: before the gate can judge an agent, it needs a fixed set of
development worlds where the hand-written baselines leave room to improve, a fixed roster of
baselines that set the bar, and fixed bounds X and Y for the side-effect checks. After attempt 1
failed on the dense family, the user decided M1 gates on the sparse-wide family only (dense-12 failed the
headroom statistic and capacity cuts did not move it; on calm worlds its shortfall is mostly
warm-up, see `../retune/RESULT.md`). This run screens the sparse-wide
development worlds.

**What was measured.** All 40 roster baselines (the 12 random ones 5 times each) played every
sparse-wide development gate world:
- 24 worlds, at 12 and 42 producers, calm and shocked;
- 128 world-episodes and 11,264 episodes, at horizon 100 cycles (an episode ends earlier on
  system death).
These are the same worlds, seeds and code path as attempt 1's sparse-wide cells. The screen
then ran its four steps:
1. Replace any world where no baseline delivers anything.
2. Check headroom: per size, the median over worlds of the best baseline's fill rate (delivered
   over ordered units) must be below 0.95.
3. Prune: drop every baseline that never set the best result on any episode.
4. Calibrate the side-effect bounds X and Y from the remaining best baselines.

**Results.**
- *Admitted.* Sparse-wide passes at both sizes: median best fill 0.9342 at 12 producers and
  0.8856 at 42. These equal attempt 1's sparse-wide cells to the last digit, as expected for the
  same worlds and code.
- *Replacements: none.* Every world had a non-zero best baseline, so no reserve ran
  (`"waves": []`).
- *Pruned members: 35 of 40.* Only the five `entity_planner/*` members ever set the per-episode
  best (blend_0.5, even_split, obs_price, proportional, random). Every other producing baseline
  never won an episode at either size and is dropped. The frozen roster that judges candidates
  is therefore those five, with R = 5 for the random one. Winners per size:
  - 12 producers: blend_0.5 27, random 19, proportional 12, even_split 3, obs_price 3 (of 64);
  - 42 producers: random 55, obs_price 9.
- *X and Y (q = 0.05).*
  - X = 0.7297: a candidate fails the run-down check if it ends an episode with less than 73%
    of the capacity ratio it started with. 95% of the best baselines' episodes stay above it.
  - Y = 0.7681: a candidate starves a consumer if it fills less than 77% of that consumer's
    orders. 95% of the best baselines' consumer fill rates stay above it.
  Both are nearest-rank 5% quantiles of the best baseline's values, frozen with their evidence
  digests.

**What changes.**
- The gate now has its frozen inputs, `frozen/{worlds,roster,side_effect_params}_v1.json`, for
  task 11.1 to install.
- The champion report (12.1) runs the 5-member roster on 24 sparse-wide worlds. That is 9 roster
  jobs per world-episode instead of 88, so it costs far less than the sizing's full-roster
  scenario.
- Because the bar is now set by `entity_planner` alone, a candidate has to beat the strongest
  hand-written producer on every world.

## Details

**Command** (repo root; pod created by `curl` because of the launcher's 403/1010 bug, `--pod-id`,
PATH shim `/tmp/rpshim` for the `runpodctl ssh info` bug; see the EXP-0044 top RESULT):

```
python run_pod_training.py --pod-id 7na9iqttdggx2a --ready-timeout 900 --cpu --python python3.12 \
  --verify-source-commit --train "-u -m tokenomics.eval.gate --mode screen --reserves on-demand \
  --prune --q 0.05 --workers 16 --source-commit {SOURCE_COMMIT} \
  --out-dir /workspace/tokenomics/runs/exp0044_a2" \
  --pull "/workspace/tokenomics/runs/exp0044_a2/:./experiments/EXP-0044-gate-calibration-12-42/attempt-2/"
```

**Inputs.**
- Registry `worlds_v1.json` 0f8b09b0… (sparse-wide only, unscreened).
- Roster `roster_v1.json` 8791e24b… (40 members, R = 5).
- Horizon 100, q 0.05.
- Source tree sha256 `8e5e959a…`, 52 files.

**Frozen outputs** (`spec.json` "frozen"; each reloads locally with the recorded hash):

| file | content hash |
|---|---|
| `frozen/worlds_v1.json` | 566946fe0c0363ec888f9a58bfe7bd480bcc948e049c93e9923088e2cab911e3 (screened development split, headroom recorded, admission problems: none) |
| `frozen/roster_v1.json` | c2832dbb44d26601f279fc04c0c483c543c5a5fcdc8cb2231090fc03f76a3731 (frozen, 5 members, R = 5) |
| `frozen/side_effect_params_v1.json` | e6651b2ebe541b18ade0a8904f27b0a08844e2a4f8ee0e231ca062514472f2c1 (X 0.729674051435041, Y 0.768075117370892, q 0.05, nearest-rank) |

**Other files.**
- `spec.json`: pins, provenance, status `complete`.
- `episodes.csv`: 11,264 rows, 4.5 MB, kept at the top level in normal git like attempt 1's.
- `screening.json`, `headroom.csv`.

**Local verification.**
- `check_rows(complete=True)`: all 11,264 expected rows are present.
- `screen_outcome` recomputed from `episodes.csv` equals `screening.json`.
- The recomputed frozen registry and params JSON are byte-identical to `frozen/`.
- The local source fingerprint at 9f29d73 equals the pod's.

**Compute** (Runpod secure cloud, cpu5c, 16 vCPU / 32 GB, `runpod/base:0.7.0-ubuntu2004`):
- Pod `7na9iqttdggx2a` was created 16:40:36 and deleted 16:56:00 UTC: 15.4 min ≈ 4.1 vCPU-h
  billed, ≈ $0.14.
- Screen execution 16:41:40 → ≈ 16:53:58 (12.3 min ≈ 3.3 vCPU-h), against the sparse-only
  sizing of 5.0–7.9 (reference 7.0).
- `runpodctl pod list --all` returned `[]` afterwards.

## Conclusions

- Sparse-wide is admitted at both sizes.
- The roster prunes to the five `entity_planner` members.
- X = 0.7297 and Y = 0.7681 are frozen.
- Next: 11.1 installs the frozen files; then the champion development report (12.1).
