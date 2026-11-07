# EXP-0125: the arrival-position feature makes the hidden tie order learnable

**Setup.**
- Pre-registration note `notes/exp0125-prereg.md` (SHA `4f0ceb41…`, committed a9cd94e before data). Driver
  `train/m6_turn_probe.py --recollect --arrival --grid exp0125 --ordered`.
- EXP-0120's 32 held-out jobs replayed with v15's linear fractions + the arrival column.
- Pod j8n79qomsxkbrt (secure 4 vCPU + 20 GB volume, EUR-IS-1), 17:35–17:59 UTC ≈ **1.6 vCPU-h** (stop 40 min =
  1.5 × the real-data estimate, M6-A).
- **Validity: valid.**
  - 62,854 of 62,854 turns: every old column equal to EXP-0120's, trunk embeddings identical.
  - The no-arrival arms reproduce EXP-0124's numbers exactly (w32 L3 e90: 0.9686, displaced 7.96%).

## Reading: **winner w32 L3 e90 + arrival** → v15 goes ahead
Probe held-out half. "Tie-dependent" = the default bit flips when equal-priced bids are taken in reverse order. The
slack column gives the wrong-bit share at slack [−0.5, 0) / [0, 0.5).

| setting | tie-dependent | determinable | overall accept | accept-turn rank | fund rank | train accept / rank | slack wrong | displaced wrong | s / 1k turn-epochs |
|---|---|---|---|---|---|---|---|---|---|
| w32 L3 e90 + arrival | 0.968 | 0.9940 | **0.9924** | 0.9963 | 1.000 | 0.9980 / 0.9989 | 0.029 / 0.060 | 0.015 | 0.37 |
| w32 L3 e90 no arrival | 0.657 | 0.9883 | **0.9686** | 0.9856 | 1.000 | 0.9886 / 0.9900 | 0.120 / 0.144 | 0.080 | 0.38 |
| w64 L2 e90 + arrival | 0.963 | 0.9941 | **0.9923** | 0.9962 | 1.000 | 0.9993 / 0.9992 | 0.033 / 0.053 | 0.018 | 0.42 |
| w64 L2 e90 no arrival | 0.659 | 0.9865 | **0.9671** | 0.9857 | 1.000 | 0.9939 / 0.9914 | 0.138 / 0.129 | 0.091 | 0.42 |
Tie-dependent bits: 3,727 of 62,806.

## Conclusions
- **The hidden tie order was the missing input.**
  - With the bid's arrival position (own state, the domain's tie-break order), tie-dependent bits go from 0.66 to
    0.97, and determinable bits from 0.988 to 0.994.
  - Overall accept is **0.9924** and accept-turn rank **0.9963**: the gate (0.99 / 0.99) is cleared.
  - Fund rank stays 1.000.
- **Displaced errors:** 1.5% (from 8% without arrival, 34% in EXP-0118).
- **Head cost:** 0.37 s per 1k turn-epochs (pod, one core).
- **Next (pre-registered):** v15 = v14 + linear fractions + the ordered pass + arrival + w32 L3 + the cosine
  schedule + the fresh held-out set, run on F40 (4.3), then the closed-loop check.
