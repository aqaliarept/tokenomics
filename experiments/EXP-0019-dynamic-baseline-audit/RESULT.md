# EXP-0019 — P-1 / O-1 dynamic baseline audits (ordered-plan step 8)

Status: **DONE — both audits clean (zero contract failures, contract vs policy stays
separable across the whole step-8 sweep).** Holding one side of the stack fixed at a
deterministic baseline and varying only the other, swept over the step-8 target axes
(same-shape capacity relabel, same-shape numeric perturbations, larger product count,
larger producer count, plus a producer-row permutation for the orderer audit). Headlines:
the **learned-gate portable producer ties/beats the planner bar on every in-family target**
(min 1.000) — the EXP-0017 producer-alone regression is **out-of-family-specific, not a
general weakness**; and the **portable orderer beats every deterministic orderer on all 13
targets** (mean 1.088×) **and is the only orderer that is exactly producer-row invariant**.
Local eval, no retrain (EXP-0012 + EXP-0013 checkpoints).

## Question
PRODUCT_AGNOSTIC ordered-plan step 8: run the P-1 (producer) and O-1 (orderer) dynamic
baseline audits through `transfer_matrix`, keeping contract failures distinct from policy
failures. Producer side — compare `entity_planner`, `stock_aware`, learned-gate portable,
and planner-gated portable producers under a fixed `proportional` orderer. Orderer side —
compare `even_split`, `proportional`, `obs_price`, and the portable scorer/allocator under
a fixed `stock_aware` producer. See `spec.json`.

## Setup
- **Runner** — `tokenomics.eval.transfer_matrix --audit {producer,orderer}` (new step-8
  mode). Adds two reusable same-shape target generators — `numeric_perturbation_targets`
  (reuses `eval.generalization`: degr2x, unseen quantity, lean 0.7× capacity, cost 1.5×)
  and `producer_permutation_targets` (reuses `permute_scale_config`) — and two named audit
  stack sets (`PRODUCER_AUDIT_STACKS`, `ORDERER_AUDIT_STACKS`). No policy code changed; the
  EXP-0012/0013 checkpoints run unchanged. TDD'd in `tests/test_transfer_matrix.py`.
- **Targets** (12 producer / 13 orderer) — held-out depth-3 seeds 6–9 + one same-shape
  capacity relabel (`default_targets`); numeric perturbations off s6; larger product
  (`prodW8`=24 products), larger producer (`agentW8`=24 producers), larger both
  (`size_transfer_targets` off s6). Orderer audit adds one producer-row permutation
  (`permP0`) off s6. Every numeric/relabel/permutation target is dims-preserving.
- **Bar** — every ratio is vs `entity_planner + proportional` on the SAME target. 5
  episodes/cell, horizon 100.

## Producer audit (orderer fixed = `proportional`; ratio vs `entity_planner+proportional`)

| target | entity_planner | stock_aware | portable (learned) | portable (planner-gates) |
|--------|------|------|------|------|
| scale-d3-s6 | 1.000 | 0.911 | 1.000 | 1.000 |
| scale-d3-s7 | 1.000 | 0.910 | 1.000 | 1.000 |
| scale-d3-s8 | 1.000 | 0.788 | **1.022** | 1.019 |
| scale-d3-s9 | 1.000 | 0.840 | **1.012** | 1.000 |
| s6-relabel | 1.000 | 0.911 | 1.000 | 1.000 |
| s6-degr2x | 1.000 | 1.012 | 1.000 | 1.000 |
| s6-qty6 | 1.000 | 0.744 | 1.002 | 1.002 |
| s6-lean0.7 | 1.000 | 0.995 | 1.000 | 1.000 |
| s6-cost1.5x | 1.000 | 0.933 | 1.000 | 1.000 |
| s6-prodW8 (24 prod) | 1.000 | 1.120 | **1.085** | 1.003 |
| s6-agentW8 (24 agents) | 1.000 | 0.965 | **1.013** | 1.017 |
| s6-prodW8-agentW8 | 1.000 | 1.037 | 1.008 | 0.975 |
| **mean / min** | 1.000 / 1.000 | 0.930 / 0.744 | **1.012 / 1.000** | 1.001 / **0.975** |

All 60 cells `runs_episode=True`, `failure_class=none`, `dead_rate=0`.

## Orderer audit (producer fixed = `stock_aware`; ratio vs `entity_planner+proportional`)

| target | even_split | proportional | obs_price | portable scorer/allocator |
|--------|------|------|------|------|
| scale-d3-s6 | 0.759 | 0.911 | 0.647 | **1.075** |
| scale-d3-s7 | 0.734 | 0.910 | 0.813 | 0.990 |
| scale-d3-s8 | 0.847 | 0.788 | 0.621 | **1.157** |
| scale-d3-s9 | 0.634 | 0.840 | 0.448 | **1.158** |
| s6-relabel | 0.759 | 0.911 | 0.647 | **1.075** |
| s6-degr2x | 0.710 | 1.012 | 0.624† | **1.081** |
| s6-qty6 | 0.344 | 0.744 | 0.218† | **1.038** |
| s6-lean0.7 | 0.728 | 0.995 | 0.608† | **1.041** |
| s6-cost1.5x | 0.836 | 0.933 | 0.749 | **1.071** |
| s6-prodW8 | 0.488 | 1.120 | 0.425† | **1.256** |
| s6-agentW8 | 0.726 | 0.965 | 0.630† | **1.016** |
| s6-prodW8-agentW8 | 1.075 | 1.037 | 0.586† | **1.123** |
| s6-permP0 (producer perm) | 0.583 | 0.912 | 0.539 | **1.067** |
| **mean / min** | 0.710 / 0.344 | 0.929 / 0.744 | 0.581 / 0.218 | **1.088 / 0.990** |

All 65 cells `runs_episode=True`, `failure_class=none`. †`obs_price` is the weak orderer:
6 targets show partial system death (`dead_rate` 0.2–0.4); every other stack is
`dead_rate=0`. The floor `noop+proportional` is 0 units on every target (nontrivial).

## Conclusion

**Contract vs policy stays separable (the audit's job).** Zero contract failures on every
dynamic/portable cell across both sweeps — every miss is a value number, never a non-run.
The floor stays 0 (targets nontrivial); the only deaths are `obs_price`'s, attributable to
that weak orderer's policy, not a contract or schema fault.

**Producer side: the EXP-0017 regression is out-of-family-specific.** On the in-family
step-8 axes (relabel, numeric perturbations, size transfer) the **learned-gate** portable
producer ties or beats the planner bar everywhere (mean **1.012**, **min 1.000**) — it
does *not* regress here. The producer-alone regression EXP-0017/0018 surfaced (s8 0.450,
s9 0.826) only appears on the **fork/join out-of-family** recipe, confirming it is a
recipe-graph generalization gap in the learned gates, not a general producer weakness. The
**planner-gated** mode is the mirror image: mean **1.001**, **min 0.975** (on the doubly
widened `prodW8-agentW8`) — it trades a little in-family upside for the out-of-family
robustness it was built for (EXP-0018). Both portable modes are byte-identical under the
capacity relabel (466.2 CU base == relabel for all four producer kinds).

**Orderer side: the portable orderer is the clear winner AND the only row-equivariant
one.** Under the fixed `stock_aware` producer it beats every deterministic orderer on all
13 targets (mean **1.088**, min **0.990** — only s7 a hair under the *stronger*-producer
`entity_planner+proportional` bar, while beating `stock_aware+proportional`'s 0.91
decisively), zero death, the lever strengthening on the wider product target (`prodW8`
1.256). `proportional` is the best deterministic orderer (mean 0.93); `even_split` (0.71)
and `obs_price` (0.58, + deaths) are well below. Equivariance: under the **producer-row
permutation** the portable orderer is **exactly invariant** (501.2 CU → 501.2), whereas
`even_split` (354.0→273.6), `obs_price` (301.4→253.2), and even `proportional`
(424.8→428.4) all shift — the domain auction breaks ties by producer id, so split-based
baselines are not producer-row invariant, but the scorer/allocator (which concentrates
routing on the best producer by its stats, not its slot) is. That is the O-1 producer-row
equivariance gate met by the portable orderer and missed by the deterministic baselines.

**What this tells the next step.** No schema/contract work is owed on these axes. The only
producer-side gap left is the out-of-family one (fork/join, already at materially-reduced
via EXP-0018 planner gates; strict closure = producer training coverage, steps 9–12). The
orderer side has no open gap on the step-8 axes. Legacy flat producers/orderers stay
**omitted** (no checkpoint wired; contract-incompatible off-size) — recorded, not run.

## Rerun commands
```
# producer audit (P-1)
python -m tokenomics.eval.transfer_matrix \
  --producer-ckpt experiments/EXP-0013-portable-producer-scorer/artifacts/producer_scorer.pt \
  --orderer-ckpt experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt \
  --audit producer --depth 3 --seeds 6-9 --eval-seeds 5 --horizon 100 \
  --out experiments/EXP-0019-dynamic-baseline-audit/producer_audit.csv
# orderer audit (O-1): same, --audit orderer
```
Local eval (existing checkpoints, no training).

## Artifacts
- `producer_audit.csv` (60 rows), `orderer_audit.csv` (65 rows), `summary.csv` (both,
  prefixed with an `audit` column).
- `artifacts/*.log` — per-audit run logs.
- checkpoints reused in place from `EXP-0012-.../artifacts/` and `EXP-0013-.../artifacts/`;
  this experiment adds no new ckpt.
