# EXP-0007B champion iteration — candidate vs m2_stream_stochastic

PROMOTE_RECOMMENDED=false

Champion-iteration loop (EXP-0007B): rolled out the registry champion, relabeled with the search teacher, distilled a candidate, scored both on the held-out suite.

## Gate

Blocked by:
- held-out mean 515.7 does not beat champion 515.7

## Held-out (candidate vs champion)

| seed | cand cu | champ cu | env cu | cand ratio_env | dead |
|---|---|---|---|---|---|
| 6 | 494.9 | 494.9 | 474.1 | 1.044 | 0.0 |
| 7 | 475.3 | 475.3 | 437.7 | 1.086 | 0.0 |
| 8 | 568.9 | 568.9 | 537.3 | 1.059 | 0.0 |
| 9 | 523.8 | 523.8 | 504.8 | 1.038 | 0.0 |

## Conclusions

The EXP-0007B champion-iteration driver completed the full registry -> preflight ->
train -> evaluate -> gate -> record loop on the stable product-stream ordering
contract. It wrote the standard experiment payload (`spec.json`, `RESULT.md`,
`summary.csv`, `artifacts/checkpoint.pt`, raw held-out CSV) and did not overwrite the
champion registry or checkpoint.

The candidate exactly reproduced the registry champion on every held-out topology:
candidate and champion scored 494.9 / 475.3 / 568.9 / 523.8 consumer units on seeds
6 / 7 / 8 / 9. This is the expected lateral result because the run re-used the
EXP-0007A recipe: same v3 product-stream contract, same L12/K12 teacher, same
topology split, and deterministic training seed.

The strict envelope remained cleared on all four held-out seeds with zero death
(ratios 1.044 / 1.086 / 1.059 / 1.038), but the candidate mean tied the champion mean
at 515.7. The gate therefore correctly emitted `PROMOTE_RECOMMENDED=false` and
returned the no-promote exit code rather than promoting a lateral checkpoint.

Verdict: **ACCEPT the automation, reject promotion.** EXP-0007B proves the driver can
reproduce a valid champion-style candidate run, record it without manual stitching, and
enforce the human-approved promotion boundary.
