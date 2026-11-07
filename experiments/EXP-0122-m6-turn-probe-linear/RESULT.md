# EXP-0122: the EXP-0121 probe with v15's linear own-state fractions

**Setup.**
- Pre-registration note `notes/exp0122-prereg.md` (SHA `e3e770f9…`, committed 8189555 before data). Driver
  `train/m6_turn_probe.py --recollect`.
- Pod wn3a4ttnj5oey1 (secure 8 vCPU + 20 GB volume, EUR-IS-1), 15:19–15:38 UTC ≈ **2.6 vCPU-h** (stop 3, M6-A).
- **Validity: valid.** EXP-0120's 32 held-out jobs were replayed with the fractions appended: 62,854 of 62,854 turns,
  every old column equal, trunk embeddings identical (max difference 0.0). So these are the same halves as
  EXP-0121.

## Reading: **no winner**
Each pair is accept agreement / accept-turn rank. The slack columns give the share of bids that fit alone but get
the wrong accept bit, by slack in own capacities.

| setting | train | probe held-out | slack [−0.5, 0) | slack [0, 0.5) | slack [0.5, 2) | slack ≥ 2 |
|---|---|---|---|---|---|---|
| w32 L2 e30 | 0.959 / 0.988 | 0.943 / 0.985 | 28.1% | 24.9% | 4.3% | 0.1% |
| w32 L2 e90 | 0.974 / 0.991 | 0.945 / 0.986 | 26.1% | 23.0% | 4.9% | 0.1% |
| w32 L3 e30 | 0.966 / 0.989 | 0.947 / 0.986 | 26.6% | 23.1% | 4.3% | 0.1% |
| w32 L3 e90 | 0.987 / 0.991 | 0.949 / 0.986 | 24.0% | 21.9% | 4.8% | 0.1% |
| w64 L2 e30 | 0.972 / 0.990 | 0.945 / 0.986 | 27.4% | 22.8% | 4.2% | 0.1% |
| w64 L2 e90 | 0.996 / 0.994 | 0.942 / 0.984 | 28.5% | 21.3% | 4.6% | 0.2% |
| w64 L3 e30 | 0.976 / 0.990 | **0.950** / 0.986 | 23.8% | 21.6% | 4.2% | 0.1% |
| w64 L3 e90 | 0.997 / 0.995 | 0.948 / 0.984 | 24.7% | 20.0% | 4.5% | 0.1% |

Fund-turn rank 1.000 (0.9998 in one setting). Cost as in EXP-0121.

## Conclusions
- **The fractions do not help:** held-out 0.942–0.950 against EXP-0121's 0.943–0.949, with the same boundary
  errors.
- **The input-scaling hypothesis (EXP-0121's reading) is refuted.** The heads now see linear, scale-free
  quantities and still miss the threshold. They memorise the training half (0.997), so representation and fit
  capacity are not the limit either.
- **The remaining cause (DIAGNOSTICS 38 addendum): learning the algorithm from little data.**
  - The rule is: order the turn's items by price, take a running sum of capacity, accept while it fits.
  - A 2–3 layer set encoder learns it only approximately from ≈ 31k turns. The approximation generalizes poorly
    near the threshold.
  - More data helps slowly: EXP-0120's 427k turns gave 0.962, with train ≈ held-out.
- **No v15** (supervisor 2026-11-07: report, no option (b) yet).
