# EXP-0022 — Ablation #2: flat-MLP vs row-scorer (does the row/edge factoring earn its keep?)

**Plan:** PRODUCT_AGNOSTIC.md step 14, ablation #2 (after #1 fingerprint-vs-schema gate, before #3 learned-direct-vs-allocator).
**ADR:** 0053 (portable producer entity contract).
**Question:** Does the row/edge factoring of `ProducerScoreModel` actually buy anything? Split into **(A) portability** — is a fixed-pad flat MLP over the same features non-portable? — and **(B) value** — how much do the recipe **edges** buy over an otherwise-identical **per-row** scorer with edges zeroed?

## Setup

Two baselines against the EXP-0021 row+edge champion (`producer_scorer_broad.pt`):

- **(A) `ProducerFlatModel`** — same per-entity features, concatenated into a **fixed-pad flat vector** (cap/prod/input pads from the build economy), one MLP, sliced back to the actual entity ids. Same `forward_view` interface, so it deploys through `score_with_model`. **Non-portability is a construction property** — proven by unit test, no training/pod needed.
- **(B) `ProducerScoreModel(use_edges=False)`** — identical row heads, but the 4 recipe relation contexts are **zeroed**. Head dims unchanged (edge slots → zeros) so it still **loads on variable topologies** = portable. Distilled with the **same broadened recipe** as EXP-0021 plus `--no-edges` (54 topologies, depth 3, seeds 0–5, width 4, 300 epochs). Stamps `producer-row-noedge-scorer-v2`.

Edge value measured by running producer-alone `transfer_matrix --audit producer --fixtures` (stack `portable_producer_planner_gates+proportional`, `proportional` orderer, eval_seeds 5) for **both** checkpoints and diffing per-cell `consumer_completed_units`. `transfer_matrix` now stamps `producer_arch` per row, so no-edge rows are self-labeled (step-14 reviewer caveat).

## Result

**(A) The flat MLP is non-portable by construction.** `forward_view` raises `ValueError` the instant any actual entity count exceeds the pad it was built for (unit test `test_flat_model_scores_within_pad_but_is_non_portable_beyond_it`). It scores correctly *within* its pad but cannot ingest a wider product/producer count — where the row/edge scorer would just add rows. Portability of the portable policy is real, not incidental.

**(B) Edges are a VALUE lever, not a portability lever.**

The no-edge per-row variant is **fully portable**: it distilled cleanly (fit MSE 0.0048, held-out scale seeds 6–9 depths 1–3 ratio-vs-planner **1.0000** mean, min 0.9996, 0 death), and on the acceptance fixtures it ran **every one of the 22 cells with 0 contract failures and 0 death** — including the wider-topology fixtures. So the **per-row structure alone** gives id-free portability; edges are not what makes the policy transferable.

But edges buy a **small, strictly-non-negative** value, concentrated exactly where recipe structure is load-bearing:

| fixture cell | no-edge | champion (edge) | Δ units | Δ % |
|---|---:|---:|---:|---:|
| scarcity-d3-s6-low_inventory | 394.2 | 404.4 | −10.2 | −2.5% |
| sharedcap-d3-w4-s8 | 512.2 | 521.8 | −9.6 | −1.8% |
| scarcity-d3-s6-high_backlog | 837.0 | 843.4 | −6.4 | −0.8% |
| *other 19 cells* | *=* | *=* | 0.0 | 0.0% |

- No-edge **equals** the edge champion on **19/22** cells, is **never better**, loses value on exactly **3** cells.
- The 3 losses are all **scarcity** or **shared-capacity** fixtures — contention states where "which product/capacity is the bottleneck" depends on the recipe graph the edges carry. Everywhere else (degenerate, uneven-strength, in-family scale) the row features alone suffice.
- `deepjoin-d5-*` = 0 units both sides (depth × TTL feasibility wall, EXP-0020), excluded from the value read.

## Conclusions

- **Row/edge factoring earns its keep, and we now know which half does what.** Portability comes from the **per-row** entity factoring (no-edge is fully portable). Value under contention comes from the **edges** (no-edge is worse only on scarcity/shared-cap, at most −2.5%).
- **A flat MLP is not a viable portable policy** — non-portable by construction. This closes the "is the entity factoring over-engineered" challenge: the flat alternative cannot even load on a wider economy.
- **No new champion.** The row+edge scorer (EXP-0021) stays the producer policy. This experiment quantifies the *cost of dropping edges* (≤2.5% on 3 stressed cells, 0 elsewhere), it does not propose a change.
- **Residuals, named:** the edge advantage is real but small; it does not touch the still-open EXP-0021 strict residuals (uneven-strength s6=0.975, in-family scale s9d3=0.9921) — those are unaffected by edges (uneven-strength cells are identical no-edge vs edge here).

## Artifacts

- `artifacts/producer_scorer_noedge.pt` — no-edge distilled checkpoint (`producer-row-noedge-scorer-v2`).
- `artifacts/noedge_fixtures.csv`, `artifacts/champion_fixtures.csv` — full producer audit rows (110 each), `producer_arch`-labeled.
- `summary.csv` — the 22 producer-alone fixture cells, no-edge vs champion units delta.
- Flat-model non-portability + no-edge edge-invariance + arch-label checks: `tokenomics/tests/test_producer_ablation_arch.py`.
