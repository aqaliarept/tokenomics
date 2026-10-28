# EXP-0047 — registry v2 calibration screen (baseline-only)

**Screening: HEADROOM PASSED** (see `screening.json`, `headroom.csv`, `frozen/`). Run 2026-10-28
at commit `ce6b3671c9398e2f617ed343b4b267c647e2ff37` (OpenSpec change `add-shocks-i-drift`, task
9.1). Produced by `python -m tokenomics.eval.gate --mode screen` (see `spec.json`).

## In plain English

**The question.** The measurement gate judges an agent against the best of a roster of
hand-written baselines, on a fixed set of development worlds, with two fixed side-effect bounds
X and Y. M1 fixed these from registry v1 (EXP-0044 attempt 2). M2 replaced the shocks on the
shocked worlds with the `shocks-i` profile (capacity hits and producer outages) and added a
shock-aware ordering baseline, `shock_reactive`, to the roster. Registry v2 and roster v2 are the
drafts that carry those changes. Before a v2 gate can judge anything, the draft has to be screened
the same way as v1: do the worlds still leave room above the baselines, which baselines still
matter, and what are X and Y now?

**What was measured.** All 48 roster v2 baselines played every sparse-wide development gate
world of registry v2, at horizon 100 cycles:
- 24 worlds: 12 and 42 producers, 8 calm and 4 shocked each;
- 128 world-episodes (3 seeds per calm world, 10 per shocked world);
- 35 deterministic baselines once and 13 random baselines 5 times each, i.e. 100 runs per
  world-episode and 12,800 episodes in total.

The screen then ran its four steps:
1. **Replacement.** A world where no baseline delivers anything is swapped for a **reserve**
   world (a spare world held back in the registry for this purpose), in a fixed order. Reserves
   run only when a swap is needed ("on demand").
2. **Headroom.** The **hindsight baseline** of an episode is the best result any roster member
   reached on it. Headroom asks whether that best still leaves room to improve. Per world, take
   the best baseline's **fill rate** (delivered over ordered consumer units), averaged over the
   world's episodes. Per family × size, the median over worlds must be below 0.95. A failure stops
   the screen with no frozen files.
3. **Pruning.** Drop every baseline that never set the hindsight baseline on any episode. Since
   the hindsight baseline is a maximum, dropping members that never attain it changes nothing.
4. **Calibration.** X and Y are set from the best baseline's own episodes. X is the 5% quantile
   of its **run-down ratio**: capacity ratio at the end of the episode over the ratio at the start.
   Y is the 5% quantile of its per-consumer fill rates. A candidate fails the run-down check below
   X·start and starves a consumer below Y.

**Results.**
- *Replacements: none.* Every world has a best baseline above zero, so no reserve ran
  (`spec.json` `"waves": []`, `screening.json` `"replacements": []`).
- *Headroom: passed at both sizes.* The median best fill is **0.9037 at 12 producers** (v1
  0.9342) and **0.8749 at 42** (v1 0.8856), against the 0.95 ceiling. The calm worlds are
  unchanged from v1: same seeds, and identical best fills and baseline means on all 16. The whole
  change comes from the 8 shocked worlds. Their best fill dropped by 0.03–0.07 under `shocks-i`
  (12 producers: 0.81–0.88, was 0.84–0.94; 42 producers: 0.83–0.87, was 0.87–0.91), and their
  best-baseline mean units by 4.0–5.5%. So the shocked worlds now leave more room than in v1, not less.

  | size | calm worlds: best fill (8 worlds, v1 = v2) | shocked worlds v2 | shocked worlds v1 | median v2 | median v1 |
  |---|---|---|---|---|---|
  | 12 | 0.774–0.989 | 0.812, 0.814, 0.879, 0.867 | 0.840, 0.882, 0.940, 0.912 | 0.9037 | 0.9342 |
  | 42 | 0.855–0.925 | 0.871, 0.848, 0.841, 0.834 | 0.908, 0.891, 0.880, 0.872 | 0.8749 | 0.8856 |

- *Pruning: 42 of 48 dropped; `shock_reactive` survives, only with `entity_planner`.* The
  frozen roster v2 has 6 members: the five `entity_planner` members of roster v1 (`blend_0.5`,
  `even_split`, `obs_price`, `proportional`, `random`) plus `entity_planner/shock_reactive`. No
  other producing baseline set a single episode's best at either size, which matches v1. Winners
  per size and regime:
  - 12 calm (24 episodes): blend_0.5 6, proportional 6, random 6, obs_price 3, even_split 3;
  - 12 shocked (40): random 28, blend_0.5 6, **shock_reactive 4**, proportional 2;
  - 42 calm (24): random 21, obs_price 3;
  - 42 shocked (40): random 40.

  `shock_reactive` earns its place on 4 shocked 12-producer episodes and nowhere else. It is
  still useful on average: with `entity_planner` it delivers more than `proportional` on every
  size × regime cell, e.g. 1,542 vs 1,438 units per shocked 42-producer episode. But on
  every other episode another `entity_planner` member sets the best, most often `random`
  (scored as the mean of its 5 seeded replicates), which wins 95 of the 128 episodes.
- *X and Y, v2 vs v1 (q = 0.05, nearest rank).*

  | | v1 (EXP-0044 attempt 2) | v2 (this run) |
  |---|---|---|
  | X (run-down bound) | 0.7297 | **0.7516** |
  | Y (starvation bound) | 0.7681 | **0.7302** |

  X rises by 0.022: the run-down check gets slightly stricter, so a candidate must end with at
  least 75% (was 73%) of the capacity ratio it started with. Y falls by 0.038: a consumer counts as
  starved only below 73% fill (was 77%), so the starvation check is looser. A diagnostic split by
  regime (not part of the screen; same quantile rule on each half) puts the change on the
  shocked worlds. The calm-only X/Y are identical in v1 and v2 (0.7755 / 0.7327). The
  shocked-only values move from 0.7228 / 0.7821 to 0.7430 / 0.7278: `shocks-i` lowers the best
  baseline's consumer fills on shocked worlds, and the low tail of its run-down ratio is higher.
  Why the run-down tail rises is not diagnosed here.

**What changes.**
- Registry v2 is admissible: `frozen/{worlds,roster,side_effect_params}_v2.json` are written,
  each with its content hash and the same `screened_on` (the draft registry hash
  `3032147f1f17…`). Installing them into the package is task 10.1, not done here.
- The v2 hindsight baseline is 6 members (5 deterministic + `entity_planner/random` × 5), 10
  runs per world-episode in a report.
- A v2 gate judges shocked worlds with more headroom than v1 and a looser starvation bound Y.
  EXP-0045's v1 reading cannot be compared number for number with a v2 report.

## Details

**Command** (repo root):

```
python run_pod_training.py --create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12 \
  --vcpu 16 --mem 32 --disk 30 --pod-name exp0047-registry-v2-screen --verify-source-commit \
  --train "-u -m tokenomics.eval.gate --mode screen \
    --registry /workspace/tokenomics/tokenomics/eval/registry/worlds_v2.json \
    --roster /workspace/tokenomics/tokenomics/eval/registry/roster_v2.json \
    --split development --reserves on-demand --prune --q 0.05 --workers 16 \
    --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0047" \
  --pull "/workspace/tokenomics/runs/exp0047:./experiments/EXP-0047-registry-v2-calibration"
```

The pull landed in `EXP-0047-registry-v2-calibration/exp0047/` (source path without a trailing
slash); its files were moved up one level, unchanged. Nothing was staged (baselines only).

**Inputs** (`spec.json` pins).
- Registry `worlds-v2` draft (unscreened), `3032147f1f17233703f486c66e4c56e006c026399364dd1cbaead5b46bcfceb5`;
  shock profiles `c92d066085c0…`.
- Roster `roster-v2` draft (48 members, R = 5, not frozen),
  `a050c99e4960eb422b9eda34fd4a0de56d259629c934ccf28e4ffcb33da816fe`.
- Horizon 100, q 0.05, `prune: true`, `reserves: on-demand`, `dry_run: false`.
- Source commit `ce6b3671c9398e2f617ed343b4b267c647e2ff37`, tree sha256 `2e9473eb25a0…` (59 files).
- `resumes: []`, `waves: []`.

**Frozen outputs** (`spec.json` "frozen"; each reloads locally with the recorded hash, and all
three record `screened_on` `3032147f1f17233703f486c66e4c56e006c026399364dd1cbaead5b46bcfceb5`):

| file | content hash |
|---|---|
| `frozen/worlds_v2.json` | 7b09f536e9880bc31370694f80eaca828587621eb5daa16fdf64e80ed05ee7ff (screened development split, headroom recorded) |
| `frozen/roster_v2.json` | 5050508ecb25f0eda0ae0a61b0dde4ba948a6e9b0d901cd14cde28e0db161f38 (`roster-v2`, frozen, 6 members, R = 5) |
| `frozen/side_effect_params_v2.json` | e2aee97ac38524d709862bc956c43940413ac0cb44c846304a4b8c3d324c3a99 (`side-effect-params-v2`, X 0.7515841609654592, Y 0.7301724137931035, q 0.05, nearest-rank) |

**Other files.**
- `spec.json`: pins, provenance, status `complete`, result.
- `episodes.csv`: 12,800 rows, 6.2 MB, at the top level in normal git like EXP-0044's.
- `screening.json`, `headroom.csv`.

**Local verification** (read-only, at ce6b367, package code unchanged):
- `gate_cli.check_rows(complete=True)` against the `spec.json` grid and the draft roster: all
  12,800 expected rows (24 worlds × their episodes × 48 members × replicates) are present once.
- The draft registry and roster hashes equal the `spec.json` pins and `screened_on`.
- `gate_cli.screen_outcome(…, q=0.05, prune=True, tag=2, screened_on=<draft hash>)`
  recomputed from `episodes.csv` gives a report equal to `screening.json`.
- Its `worlds_v2.json`, `roster_v2.json` and parameter JSON are byte-identical to `frozen/`.
- `load_registry` / `load_roster` / `load_side_effect_params` reload the frozen files with the
  `spec.json` hashes; `gate_cli.version_set_problem` admits the three as one set (returns None).
- v1 comparison: calm world grids and baseline means equal EXP-0044 attempt 2 on all 16 calm
  worlds; shocked worlds have the same ids and seeds and different results.

**Compute.** Runpod secure cloud, cpu5c, 16 vCPU / 32 GB, `runpod/base:0.7.0-ubuntu2004`,
python 3.12.11, torch 2.14.1+cpu, pod `rdatomtfo9awgi` (created and terminated by the launcher).
- Pod life: created after 07:10:21 UTC and terminated before 07:25:25 UTC (local clock, stamps
  taken around the launcher): ≤ 15.1 min ≈ **4.0 vCPU-h billed**. sizing.md expected 4.5–5.
- Screen execution (pod clock): `spec.json` `started` 07:11:18 → last output 07:25:18 ≈ 14.0 min
  ≈ 3.7 vCPU-h. sizing.md expected 14–15 min.
- Goal pod budget: 0.41 (EXP-0046) + 4.0 ≈ 4.4 of 25 vCPU-h used.
- `runpodctl pod list --all` returned `[]` afterwards.

## Conclusions

- Registry v2 (sparse-wide, 12/42 producers) passes the headroom check: median best fill 0.9037
  / 0.8749, both lower than v1 because `shocks-i` lowers the best baseline's fill on shocked
  worlds. No reserve was needed.
- The roster prunes to 6 members: v1's five `entity_planner` members plus
  `entity_planner/shock_reactive`, which sets the best on 4 shocked 12-producer episodes.
- X 0.7516 (v1 0.7297) and Y 0.7302 (v1 0.7681) are frozen in
  `frozen/side_effect_params_v2.json`.
- Next: task 10.1 installs the frozen v2 files.
