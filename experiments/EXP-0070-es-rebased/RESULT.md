# EXP-0070: Re-based step 2 on MWW (pre-registration v3): bar, A1 seed 0 (40 generations)

**Status: stopped before the validation read.**
- **Training gain below its floor.** On the training-monitor set the trained A1 beats its new
  starting point A0 (= MWWk) by only **+25.8** units at generation 40, against F_tr **92.3**.
- **The read is predetermined.** Under D7, any read of this candidate is `unresolved` (rule 1:
  g_tr ≤ F_tr for every candidate read), which means "diagnose, stop". So step 3 (the v3
  validation read) was not launched. The supervisor decides.
- **Task:** 7.7 of `add-m4-orderer-redesign`, user decision "Re-base on MWW". Pre-registration v3
  `edd6efc2c325…`, commit `90f24ad71d95…` for both steps, from a clean worktree.

## In plain English

**The question.** EXP-0060's learned correction started from a weak routing rule (TS) and could
only climb to the level of the simple rules. The user chose to restart it from the strongest
simple rule, MWW (keyed as MWWk), and asked whether a learned correction on top of MWW does
better still.

**What ran.**
1. **A new bar** for that starting point. The registered training-world selection was re-run
   with A0 playing MWWk: 64 training worlds, 128 world-episodes, 1 world replaced because A0
   died there.
2. **A1, seed 0, trained for 40 generations** on it, under the same training rule as before.

**What happened.**
- **The correction barely moved MWWk.** On the 16 monitor world-episodes (each played twice),
  the trained router's gain over A0 went +20.0 → +12.3 → +4.4 → **+25.8** at generations
  10 / 20 / 30 / 40. That is about 1.2% of A0's 2,113 units, and it is not a steady climb.
- **Run-to-run variation is larger than the gain.** The floor rose from 52.8 (the atlas-spread
  floor) to 92.3. That 92.3 is 3 × the standard error of the run's own paired per-world
  differences, which grew. At generation 40, 10 of 16 world-episodes are up and 6 down (gains
  +844, losses −432).
- **Far more members are disqualified than before.** A median of 11.5 of 24 per generation, up
  to 23, against 8–10 in EXP-0060's runs.
  - The perturbed members now tend to run producers down or starve consumers more often than
    MWWk itself.
  - MWWk is the reference, and MWWk's own run-down / starvation shares on the training worlds
    are already 0.80 / 0.97.
- **By D7's first rule the read would be `unresolved`.** That reading means "diagnose and
  stop", so the validation read was not run.

**What this does and does not say.** It does not show that MWW cannot be improved. It shows
that this training (40 generations, σ 0.05, 12 pairs, minibatch 24 and the TS-calibrated noise
margin) does not find a measurable improvement on MWWk.

Three explanations compete, each with a different fix. DIAGNOSTICS 28 records them:
- MWW is near what the per-order residual can reach;
- the disqualification rule, calibrated on the old base, now ties most members;
- the gain is real but too small for this monitor set.

## Details

### Step 1: bar (`bar/`; `order_es_worlds --mode bar --base mww`)
- **Status:** complete, 1,400 rows, 0 contract failures, A0's pins carry base `mww`.
- **Selection:** 70 candidates considered: 64 training, 14 dropped (ρ), 5 dropped (bar arm dead),
  1 replaced (A0 dead).
- **Values:** mean bar 2,069.1. A0 (MWWk) run-down share 0.80, starved share 0.97.
- **Hashes:** normaliser `0691beffca1b…` (flags: `static_psi_clip_42` empty, no heavy clip),
  training worlds `39e18771…`, schedule `dcf6a31f…`.
- **Between-steps check** (after the pull, no pod up): complete, header-only failures, flag not
  raised.
- **Pod:** `rj2vxkzdgplljk`, secure 32 vCPU, 08:51:59 → terminated by 08:54:07 (≤ 2.1 min),
  **≈ 1.1 vCPU-h**. Stop 13.4, launch-fit 74.0 + 13.4 = 87.4 ≤ 125.
- **Layout:** `bar/episodes.csv` → `artifacts/bar_episodes.csv` and `bar/fit_rows/` →
  `artifacts/bar_fit_rows/` (79 MB), following EXP-0058's layout. Staging for a read copies them
  back.

### Step 2: A1 seed 0, 40 generations (`artifacts/a1-s0/`)

Monitor set, read offline with `order_es_validate.load_candidate` / `run_stats`:

| generation | center | A0 (MWWk) | g_tr | F_tr (atlas / own 3·SE) | up / down | summed gains / losses |
|---|---|---|---|---|---|---|
| 10 | 2133.09 | 2113.09 | +20.00 | 63.75 (52.76 / 63.75) | 7 / 9 | +698.0 / −378.0 |
| 20 | 2125.38 | 2113.09 | +12.28 | 52.76 (52.76 / 45.94) | 8 / 8 | +509.0 / −312.5 |
| 30 | 2117.53 | 2113.09 | +4.44 | 81.03 (52.76 / 81.03) | 10 / 6 | +428.5 / −357.5 |
| **40** | **2138.88** | 2113.09 | **+25.78** | **92.35** (52.76 / 92.35) | **10 / 6** | **+844.0 / −431.5** |

- Improvement over the last 10 generations: +21.3.
- Disqualified members per generation (of 24): median 11.5, range 0–23. EXP-0060b / d: medians
  10 / 8.
- **Pins:** v3 (`disqualification.prereg_version` 3), base `mww`, minibatch 24, commit
  `90f24ad…`.
- **Final θ:** `ed4ae8ee59e7…`. 0 contract failures.
- **Pod:** `94va8th6qpr058`, secure 32 vCPU, 08:54:49 → terminated by 09:19:29 (≤ 24.7 min),
  **≈ 13.2 vCPU-h**. Stop 36, launch-fit 75.1 + 36 = 111.1 ≤ 125.

### Step 3: not run
D7 rule 1 (`unresolved`: g_tr ≤ F_tr for every candidate read) would decide any read of this
single candidate.
- **What a read would add:** validation numbers (the candidate against MWW, RR, TR and the
  reported EXP-0060 arm).
- **Estimated cost:** ≈ 3.5 vCPU-h at val-seed0's measured rate (stop 12.4).
- **Launch-fit:** 88.3 + 12.4 = 100.7 ≤ 125.
- **Note on clause (ii):** it would rest on EXP-0060's old-base A2-R2 − A1 comparison.

### Cost
Tranche A billed ≈ 74.0 + 1.1 + 13.2 = **≈ 88.3 of 125**.

## Conclusions

- **The re-based A1 does not clear the training floor.** g_tr is +25.8 against F_tr 92.3 at
  generation 40, and the gain is not monotone (+20.0 / +12.3 / +4.4 / +25.8).
- **The decision is D7's:** `unresolved`, diagnose and stop. The validation read was not
  launched, and seed 1 does not arise.
- **DIAGNOSTICS 28 opened.** The next move is the supervisor's and the user's.
