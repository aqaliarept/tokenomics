# EXP-0106: why the Stage I v3 producer over-fires its upgrade gate (offline probe)

add-m6-joint-mpg-gnn, supervisor decision 2026-11-06 (root-cause rule after EXP-0105 `short`). Code `train/m6_gate_probe.py`
(commit `m6 [code]: EXP-0106 …`); pod ts6iux5ztcn8ek (secure 8 vCPU / 16 GB, EUR-IS-1), 15:20–15:24 UTC ≈ **0.6 vCPU-h** (stop 3).

## Question
On EXP-0105's recorded k5_rf5 decisions, is the v3 upgrade over-firing caused by the margin feature (ignored or
misused), by the 42-producer data (interference), or by underfitting? Three producers on the same decisions:
- **v2:** EXP-0100's producer, without the margin column;
- **v3:** EXP-0105's producer;
- **v3 ablated:** v3 with the margin set to its normaliser median, 0.758.

The decision sets: the 42-producer training block (every 3rd: 53,200 decisions), the 12-producer training block
(44,800) and the 12-producer monitor set (19,200).

## Pre-check
At 12 producers every decision has **one** capacity row (44,800 / 44,800; 19,200 / 19,200), and no row has
restore_effect = 0. At 42 also one row per decision. So the mean-pool "dilution" explanation is excluded at both
sizes. The margin is < 0 on 5.0% (42), 5.6% (12 train) and 8.4% (12 monitor) of the rows.

## Results
**Upgrade gate at margin < 0.** k5_rf5's label there is always 0, so any firing is an error.

| set | decisions at margin < 0 | v2 fires | v3 fires | v3 ablated fires |
|---|---|---|---|---|
| 42 train | 2,669 | 0.304 | **0.095** | 0.699 |
| 12 train | 2,524 | **0.045** | 0.395 | 0.684 |
| 12 monitor | 1,611 | **0.024** | 0.356 | 0.712 |

**At margin ≥ 0** (monitor): label 0.133; v2 fires 0.164, v3 0.169, v3 ablated 0.168. Nearly the same.

**Per-head predicted rate, 12 monitor** (label / v2 / v3):
| head | label | v2 | v3 |
|---|---|---|---|
| restore | 0.373 | 0.548 | **0.422** |
| upgrade | 0.122 | **0.153** | 0.184 |
| expand | 0.027 | 0.111 | **0.080** |

- v3's extra upgrade firing at 12 is almost all at margin < 0: (0.356 − 0.024) × 1,611 / 19,200 = 2.8 points, of
  the 3.1-point gap.
- v3 is better than v2 on restore and expand.

**Label balance.**
- Upgrade label rate: 0.096 (42), 0.103 (12 train), 0.122 (monitor).
- Restore: 0.356 / 0.335 / 0.373. Expand: 0.028 / 0.021 / 0.027.
- The fit's producer decisions: 54% at 42, 46% at 12; each decision weighs the same.

**Producer BC loss** (v2 / v3 / v3 ablated): 42 train 1.55 / **1.06** / 1.44; 12 train 1.80 / **1.01** / 1.37;
12 monitor 2.26 / **1.16** / 1.87.

## Diagnosis
- **Not underfit.** v3 has the lower BC loss at both sizes, and its end-of-fit loss is about the same at 42 and
  at 12.
- **Not label imbalance:** the label rates per head are similar at both sizes.
- **Not dilution:** one capacity row per decision.
- **The feature is used, not ignored.** Ablating it raises v3's firing at margin < 0 from 0.36–0.40 to about 0.70
  at 12, and from 0.095 to 0.70 at 42.
- **The feature is misused under joint-scale training.**
  - v3 leans on the scale-free margin, but responds to it smoothly rather than as a step at 0. The upgrade
    logit falls with the margin at a slope of −16 to −21, with no sharp change at 0.
  - At 12 the margins below 0 are small (−0.11 … −0.05), so v3 still fires on 36–40% of those decisions.
  - v2, trained at 12 only and without the feature, had learned that boundary from scale-specific cues
    (2–4%).
  - At 42, v3 is better than v2 on recorded states (0.095 vs 0.304), but not good enough.
  - The probe cannot separate "the 42 data" from "the feature" as the reason v3 lost v2's sharp 12-producer
    boundary. That needs a v3-feature, 12-only fit.
- **Underlying mechanism, shared by v2 and v3:** the sustainable ceiling is a hard, irreversible constraint (a max
  only grows). An imitated gate with any non-zero fire rate at margin < 0 compounds over a 100–200-cycle closed
  loop. 9.5% per decision at 42 still produced 381 producers past the ceiling in EXP-0105, where k5_rf5's rate
  is exactly 0. Lower BC loss does not mean fewer of these rare, irreversible errors.

DIAGNOSTICS 33.
