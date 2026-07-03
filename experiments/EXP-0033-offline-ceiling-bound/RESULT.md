# EXP-0033 - Offline Achievable-Units Upper Bound

Question: is the Stage-4 loop stuck because the champion is already near the archived
tasks' offline ceiling, or because the `mpc_search` teacher is too weak to expose real
headroom?

## Method

Audit the 26 EXP-0028 archived tasks with dependency-free analytic upper bounds:

- B0: exact scripted consumer demand over the same three episode seeds used by the
  EXP-0028 matrix.
- B1: B0 plus an optimistic direct recipe capacity-throughput cap.
- B2: B1 plus transitive input-product recipe expansion.

The capacity tiers aggregate capacity across producers, pin capacity optimistically, add
free/instant upgrade slack, and ignore TTL, inventory, funding, and routing. The invariant
is `bound >= max(champion_units, bar_units)` for every task/tier, with 0.01 tolerance for
the two-decimal observed matrix.

## Result

All tiers are identical on this archive: **B0/B1/B2 are demand-bound for 26/26 tasks**.
Capacity is not the binding ceiling, even after transitive recipe expansion.

| Metric | B2 value |
|---|---:|
| Tasks | 26 |
| Mean champion / bound | 0.9830 |
| Min champion / bound | 0.9755 |
| Rows >= 95% | 26/26 |
| Mean bar / bound | 0.9902 |
| Min bar / bound | 0.9819 |
| Upper-bound invariant failures | 0 |

Worst champion row: `mine-build_scale_config-depth3-producers_per_layer4-products_per_layer4-s28`
at 492.0 / 504.33 = **0.9755**. The bar is 495.67 / 504.33 = **0.9828** there.

## Conclusion

**Verdict: NEAR-BOUND.** On the mined EXP-0028 distribution, the champion already delivers
at least 97.55% of the exact available demand in every archived task, and averages 98.3%.
The search bar averages 99.0%, so the mined "gap" is only the last 1-2% of demanded units,
not evidence of a large hidden policy ceiling.

B3 temporal LP is not worth adding for this archive. A temporal LP could in principle
tighten the upper bound below total demand if TTL or timing were binding, but B1/B2 found
no capacity pressure and the observed stacks already reach 98-99% of exact demand. The
decision is therefore unchanged: the Stage-4 fixed point is real for this task family.
Further improvement should move to a different distribution/claim axis or a promotion
hygiene decision, not a stronger `mpc_search` teacher for these archived tasks.
