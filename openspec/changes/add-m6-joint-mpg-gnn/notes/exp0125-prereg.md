# EXP-0125 pre-registration: the arrival-position feature for the hidden tie order

Frozen before any data (supervisor 2026-11-07, after EXP-0124). Driver `tokenomics/train/m6_turn_probe.py
--recollect --arrival --grid exp0125 --ordered`. Never an acceptance read.

## Question
In EXP-0124, 5.9% of the default accept bits depended on the arrival order of equal-priced bids, which the heads
did not see (64% right). Determinable bits reached 0.985. Does the arrival-position feature (own state: the order the
producer received each bid / claim within the turn, which the domain's stable tie-break uses) make those labels
learnable, and does a larger setting bring the whole to the gate?

## Data
- EXP-0120's 32 held-out jobs (F40 + the start's replenish, default turns), replayed with v15's linear fractions and
  the arrival column appended.
- **Validity:** every existing column and the trunk embeddings must equal EXP-0120's records turn for turn (62,854
  turns); any mismatch invalidates the run.
- The same contiguous halves as EXP-0121–0124: fit on the first, read on the second.

## Settings (2 × 2)
- Heads: ordered pass + linear fractions; width 32 / 3 layers / 90 epochs and width 64 / 2 layers / 90 epochs, each
  **with and without** the arrival feature (without = the same records with that column dropped).
- Fit as in EXP-0124: cosine LR from 3e-3, batch 512, ≤ 400 pairs, seed 0, 3 ranking warm-up epochs, declined
  bids ranked, the student's own order.

## Metrics (per setting, probe held-out half)
- **Accept agreement split three ways:**
  - tie-dependent bits (the default bit flips when equal-priced bids are taken in reverse order; the auction
    replayed from the rows);
  - determinable bits;
  - overall.
- **Accept-turn rank and fund rank** (v13 / v14's `agreement`), the slack split, and the cost per 1k turn-epochs.

## Reading
- **v15 goes ahead** only if some setting **with** the feature reaches 0.99 on **overall** accept agreement **and**
  accept-turn rank. The gate is unchanged.
- **Otherwise:** report; the gate question goes to the user with these numbers.

## Sizing
- From EXP-0124's pod timings on the same halves: slowest ≈ 12.1 s per epoch × 90 ≈ 18 min, + replay, scoring and
  setup ≈ 27 min; stop = 1.5 × that = **40 min**.
- 4 settings in parallel on a secure **4 vCPU** pod with a persistent volume, EUR-IS-1: stop ≈ 2.7 vCPU-h, M6-A.
- Each setting's result is appended as it finishes (`probe_partial.jsonl`).
