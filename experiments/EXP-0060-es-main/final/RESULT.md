# exp0060-val1

Driver: `python -m tokenomics.train.order_es_validate` (add-m4-orderer-redesign D6 / D7 / D9). Status: complete.

## Question

Does outcome-trained per-order routing beat A0 (its zero residual), `round_robin`,
`makeable_weighted_whole` and the trained orderer on the 12-producer loaded validation worlds,
by more than each comparison's floor, without deaths, run-down or starvation worse than RR's?

## Result

- Rows: 2304 over 16 worlds x 8 episodes; arms: A0, A1-s0, A1-s0/shared, A1-s1, A1-s1/shared, A2-R2-s0, A2-R2-s0/shared, MWW, RR, TR.
- Reading: plateau (next: record history_inert and herding; ask the user to release tranche B1); labels: herding, history_inert.
- Tables: `validation.csv`, `concentration.csv`, `herding.csv`; `reading.json` holds every
  clause and floor (the rules: the pre-registration's `readings` section).

## Conclusions

TODO.
