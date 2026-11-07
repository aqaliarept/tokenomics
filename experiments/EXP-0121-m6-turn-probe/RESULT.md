# EXP-0121: capacity / fit-budget probe for the v14 turn heads

**Setup.**
- Pre-registration note `notes/exp0121-prereg.md` (SHA `03490fa2…`, committed 83de1fd before data). Driver
  `train/m6_turn_probe.py`.
- EXP-0120's held-out records (62,854 turns, F40 trunk) split into contiguous halves: fit on 31.4k turns, read on
  31.4k.
- Grid: width {32, 64} × layers {2, 3} × epochs {30, 90}, cosine LR from 3e-3.
- Pod s26tp6kup137bk (secure 8 vCPU + 20 GB volume, EUR-IS-1), 14:38–14:55 UTC ≈ **2.3 vCPU-h** (stop 5, M6-A).

## Reading: **no winner** (no setting reaches 0.99)

| setting | params | train accept / rank | probe held-out accept / rank | held-out displaced wrong | s / 1k turn-epochs |
|---|---|---|---|---|---|
| w32 L2 e30 | 63k | 0.959 / 0.988 | 0.943 / 0.985 | 17.8% | 0.23 |
| w32 L2 e90 | 63k | 0.975 / 0.991 | 0.943 / 0.985 | 18.4% | 0.17 |
| w32 L3 e30 | 80k | 0.963 / 0.989 | 0.946 / 0.986 | 16.8% | 0.30 |
| w32 L3 e90 | 80k | 0.985 / 0.991 | 0.945 / 0.985 | 18.6% | 0.22 |
| w64 L2 e30 | 211k | 0.969 / 0.990 | 0.947 / 0.985 | 15.5% | 0.35 |
| w64 L2 e90 | 211k | 0.996 / 0.994 | 0.944 / 0.983 | 18.0% | 0.27 |
| w64 L3 e30 | 278k | 0.974 / 0.990 | 0.948 / 0.985 | 16.9% | 0.49 |
| w64 L3 e90 | 278k | **0.997 / 0.995** | 0.949 / 0.984 | 17.1% | 0.32 |

- Fund-turn rank is 1.000 everywhere, with no ties and no saturation.
- Head cost: **0.17–0.49 s per 1,000 turn-epochs** on one core, larger heads near the top.
- **The bigger heads memorise the half** (train 0.997) and held-out stays at 0.943–0.949: a generalization gap,
  not a capacity limit.
- With 13.6× more data (EXP-0120: 427k turns, w32 L2, 20 epochs), held-out was 0.962 with train ≈ held-out.

## Diagnosis (local, from EXP-0120's held-out records and model)
**The information is all in the bid rows.** Every producer owns one capacity type. Replaying the greedy auction on
each turn's own rows (log1p inverted: price, capacity, the new-work budget) reproduces the labels on **100%** of
128,738 bids.

**The errors sit at the boundary.** Bids that fit alone, by slack = remaining capacity after the higher-priced
bids minus own capacity, in units of own capacity:

| slack / own capacity | bids | wrong |
|---|---|---|
| ≥ 2 | 72,818 | 0.1% |
| [0.5, 2) | 16,962 | 4.2% |
| [0, 0.5) | 7,498 | **23.0%** |
| [−0.5, 0) | 10,107 | **14.7%** |
| [−2, −0.5) | 15,870 | 5.9% |

## Conclusions
- **Cause (DIAGNOSTICS 38 addendum): arithmetic precision at a sharp threshold.**
  - The rule is a running sum of linear capacities compared with a budget.
  - The heads see log1p-compressed, median / IQR-normalised quantities, so attention must undo the log, sum and
    compare exactly.
  - It learns the bulk, but generalizes poorly near the boundary. More capacity memorises, and more data helps
    slowly.
- **Fix candidates for v15, a design decision; none hand-builds the rule:**
  - **(a)** add linear-scale, scale-free copies of each item's quantities: capacity and price as fractions of the
    producer's own budget or capacity (own state, a feature normalisation);
  - **(b)** much more data (several million turns) with the w64 heads;
  - **(c)** both.
- **The gate is not relaxed.** No v15 until the design is chosen.
