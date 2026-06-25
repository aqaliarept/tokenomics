# EXP-0010A — Entity-Factored Producer Entity View

**Question.** Can the flat producer observation (`encode_producing`) be replaced by a
domain-owned, **entity-factored** producer view — capacity-type rows, product rows,
input-product rows, and recipe-edge relations — behind a **versioned, opt-in** env
encoder, passing the ADR-0053 permutation gates and staying checkpoint-incompatible with
`flat_v1`, **without touching the producer command or the production auction**?

**Verdict: ACCEPT (observation-contract slice).** The view + versioned encoder land,
all ADR-0053 EXP-0010A gates pass, and the flat producer contract is byte-for-byte
unchanged with the flag off. No learned/planning controller, no training (that is
EXP-0010B/C/D). See `spec.json` and `PLAN.md` for the full scope and deviations.

## What landed

- **Domain (pure).**
  - `ProducingAgent.requested_capacity_by_type` — per-type `ReqPrev[p,t]`, accumulated
    over the cycle's bids and committed post-auction next to the scalar `requested_capacity`
    (back-compat scalar retained).
  - `domain/metrics.py` — five new pure snapshot helpers: `capacity_demand_pressure`,
    `restore_effect`/`restore_cost`, `upgrade_effect`/`upgrade_cost`,
    `downstream_demand_pressure`, `active_reservation_qty`, `blocked_reservation_qty`. All
    keep the eligibility guard (0 for products not in `produces`).
  - `domain/producer_view.py` — `ProducerEntityView` + `build_producer_entity_view`. Rows
    are **ordered lists** (default sorted-id); relations are keyed by **row position**
    (`product_capacity_required[(xi,ti)]`, `product_input_required[(xi,gi)]`). The view
    carries **no cycle progress** (env-time scalar). Mirrors `orderer_view.py` exactly.
  - `System.producer_entity_view(pid)` at the `ORDERS_PLACEMENT` decision point.

- **Env (adapter only).**
  - `EnvSpec.producer_obs_version` (default `1`=flat_v1, opt-in `2`=entity),
    `producer_entity_obs_dim`, and `PRODUCER_GLOBAL/CAP/PRODUCT/INPUT` block-width
    constants. `producing_obs_dim` dispatches to the entity dim at v2.
  - `encode_producing_entity(spec, view, cycle_progress)` — emits
    `[globals, capacity rows, product rows, input rows, relation planes, cycle_progress]`.
    It **consumes the view's row order verbatim** (does not re-sort), zero-pads rows to the
    spec's fixed max counts with a leading active-mask column (mask, not slot, defines a
    real row), and lays the relation planes by row position — which is what makes the
    encoder **order-equivariant**.
  - `config_fingerprint` / `build_spec` threaded with `producer_obs_version`. **v1
    fingerprints are byte-identical to history** (the version is folded into the hash only
    for v2+, and `producing_obs_dim` already differs) — so every checked-in flat checkpoint
    still passes preflight; v2 is a distinct, incompatible contract.
  - `TokenomicsAECEnv(entity_producer_obs=False)` flag + `observe()` dispatch. Default off.

## Gates (all green)

`tokenomics/tests/test_producer_view.py` — 24 tests:

| Gate | Result |
|---|---|
| Per-type `ReqPrev` faithful to bids (accepted+rejected) | pass |
| New metrics helpers faithful (pressure / restore / upgrade / downstream / reservation) | pass |
| View row coverage = owned types / `produces` / recipe inputs; capacity/product/input rows faithful to metrics + raw state | pass |
| Product-row eligibility guard (`eligible=1` real rows; metrics 0 for non-`produces`) | pass |
| Relations keyed by row position | pass |
| Spec dim ↔ `encode_producing_entity` length tie | pass |
| **Row-order equivariance** (permute rows → row slots + relation planes permute; globals invariant) | pass |
| **Product-relabeling** invariance (row-set + globals unchanged) | pass |
| **Capacity-relabeling** invariance | pass |
| Versioning: `fingerprint(v2) != (v1)`, `producing_obs_dim` differs, flat≠entity dim | pass |
| Info-completeness (producer-local scope; two flat globals documented as dropped) | pass |
| `reserved_ratio` = Res/**current** capacity (degraded producer) | pass |
| Entity globals carry the 4 producer action-economics; obs differs when only those change | pass |
| Padded rows all-zero with mask 0 | pass |
| Env flag switches contract; `observe` shapes differ | pass |
| **White-box `stock_aware` episode byte-identical** with flag off vs on | pass |
| PettingZoo `api_test` green with the flag on | pass |

Full suite: **370 passed** (`../.venv/bin/python -m pytest -q`). `signal_check` unchanged
(flag off by default).

## Review fixes (post-acceptance)

Two correctness gaps caught in code review, both fixed TDD-style (failing test → fix):

- **`reserved_ratio` divided by max, not current.** ADR-0053 defines it as `Res[p,t] / C[p,t]`
  (current usable capacity). The encoder divided by **max**, so a degraded producer
  (current < max) under-reported its blocked fraction (e.g. current 50 / max 100 with 25
  reserved read 0.25 instead of 0.50). Fixed in `producer_view.py`.
- **Producer action mechanics were dropped from the entity obs.** The flat tail
  (`_producer_mechanics`, ADR-0044) exposed `max_replenish`, `replenish_cost_per_unit`, and
  inventory-expansion `increases`/`cost`. The cap rows already carry restore/upgrade
  effect+cost **per type**, but those four producer-global economics had no home — so two
  configs differing only in replenish/expansion cost encoded identically while
  `decode_producing`/funding interpret the same action differently. Added them as the
  trailing four slots of the global block (`PRODUCER_GLOBAL_FEATURES` 8→12). v2-only, so no
  checkpoint exists to invalidate.

## Notable

- **Backward-compat catch.** The first fingerprint change appended `producer_obs_version`
  unconditionally and silently re-hashed **every** config, failing the checked-in v3
  champion preflight (`6e0096b9… != d4c3d174…`). Fixed by folding the version into the hash
  **only for v2+** (v1 stays on its historical fingerprint); separation is still guaranteed
  because `producing_obs_dim` already differs.
- **Deviations from PLAN** (see `spec.json`): `requested_capacity_by_type` is read off the
  agent directly (no change to the positional `ProducingAgentView`); InputRow replenishment
  EMAs deferred (no telemetry source); the two flat cross-producer globals deliberately not
  carried.

## Next

EXP-0010B (deterministic producer planner over this view), then 0010C (learned local
scorer), 0010D (size-holdout). ADR-0053 stays **proposed** until the controller slices
land — this slice delivers only the observation contract it depends on.
