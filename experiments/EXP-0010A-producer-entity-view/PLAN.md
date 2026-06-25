# EXP-0010A: Entity-Factored Producer Entity View

## Context

The producer controller (the single orchestrator per producer) still sees a **flat**
observation: global capacity slots, input slots, product-demand slots, producer summary,
investment toggles (`encode_producing` in `tokenomics/env/encode.py`). That tensor is
sensitive to arbitrary product IDs, capacity-type IDs, and slot positions, so producer
training memorizes one topology's vocabulary and cannot extrapolate to larger
product/capacity sets. ADR-0053 (`docs/adr/0053-entity-factored-producer-orchestrator.md`)
decided to replace the flat tensor with a **domain-owned, entity-factored** producer
view: capacity-type rows, product rows, input-product rows, and relation masks — so the
learned part later depends on row features + relations, not slot identity.

EXP-0010A is the **observation-contract slice only**: add the view + a versioned, opt-in
env encoding + parity/permutation gates. It does **not** build a learned/planning
controller, does **not** train, and does **not** change the producer command or the
production auction. Per ADR, reservation rows are **deferred** — the equivalent aggregates
(reserved capacity per type, reserved load, active-reservation qty per product) live in the
capacity/product rows instead. This is entirely local work (domain + env + tests); no cloud
training (planner/clone of a *controller* = 0010B/C).

**On ADR-0053's "replay/clone stock_aware/v4 through the new contract"** (ADR line 408):
this slice discharges it with a **no-learning parity harness**, not training, and only for
the **white-box `stock_aware`** path. `stock_aware` reads `env.system` (not the obs), so
flipping the entity-obs flag must leave its episode **byte-identical** — proving the new
contract is non-invasive. **v4 is a learned flat-v1 checkpoint**, whose actor reads
`env.observe(agent)` and would break on the new obs dim; replaying *that controller* through
the entity contract needs a flat-from-entity adapter and belongs to **EXP-0010B**, not here.
A scoped **information-completeness check** proves the entity view carries every
*producer-local / action-relevant* flat fact (see §6 for what is in scope and the two
globals intentionally dropped). Acceptance is thereby narrowed to **view/schema/parity**.
(Confirm the ADR EXP-0010A accept wording matches "view/schema only" when filling RESULT.md.)

This mirrors the established orderer pattern exactly: `OrdererEconomyView` /
`build_orderer_economy_view` (`domain/orderer_view.py`) own the facts; `encode_ordering_*`
(`env/encode.py`) only normalize + clip; `EnvSpec.orderer_obs_version` makes it opt-in and
fingerprinted. We do the producer equivalent.

## Acceptance (ADR-0053 EXP-0010A)

- baseline producer behavior preserved (command/auction untouched; existing tests green)
- row-order / product-relabeling / capacity-relabeling gates pass on the view
- observation/action contract is **versioned** and checkpoint-incompatible with `flat_v1`

## Work

### 1. Domain: one missing fact — per-type requested capacity (`ReqPrev[p,t]`)

`ProducingAgent.requested_capacity` is a **scalar** (set in `produce()`,
`producing_agent.py:468`). ADR requires per-type `ReqPrev[p,t]` = most-recent
post-auction requested capacity per type (current-cycle bids are not yet routed at the
producer decision point, so it must be the previous cycle's pressure).

- Add `self.requested_capacity_by_type: dict[CapacityType, Capacity]` (init `{}`).
- In `produce()`, accumulate per type alongside the existing scalar `requested` while
  iterating `self.bids` (add `bid.capacities` into a local per-type dict), then commit it
  next to `self.requested_capacity = requested`. Scalar stays for back-compat.
- Surface it on `ProducingAgentView`. `ProducingAgentView` is a positional `@dataclass`
  with many non-default fields followed by one trailing default (`outstanding_inbound`,
  `producing_agent.py:136`). Append the new field **after** `outstanding_inbound` as
  `requested_capacity_by_type: dict[CapacityType, Capacity] = field(default_factory=dict)`
  (default factory, never a bare `{}`; keyword/trailing so existing positional constructors
  don't break), and populate it in `view()`.

### 2. Domain: new metrics helpers (`domain/metrics.py`)

Reuse what exists (`capacity_makeable_qty`, `input_makeable_qty`, `complete_qty_now`,
`capacity_bottleneck_ratio`, `input_bottleneck_ratio`, `product_depth`, recipe-shape fns,
`_available_capacity`, `producer_reserved_load`, `producer_active_investments`). Add the
small per-row helpers ADR-0053 names that have no equivalent yet, all pure snapshot reads:

- `capacity_demand_pressure(p, t, open_qty)` = `sum_x eligible[p,x]*open_qty[x]*R[x,t]`
- `restore_effect(p, t)` / `restore_cost(p, t)` — restoration amount/cost iff it targets `t`
- `upgrade_effect(p, t)` / `upgrade_cost(p, t)` — upgrade amount/cost iff it targets `t`
- `downstream_demand_pressure(p, g, open_qty)` = `sum_x eligible[p,x]*open_qty[x]*I[x,g]`
- `active_reservation_qty(p, x)`, `blocked_reservation_qty(p, g)` from `p.reservations`

### 3. Domain: the view (`domain/producer_view.py`, new — mirrors `orderer_view.py`)

Dataclasses + a builder, pure domain (no RL/numpy/IO). **cycle_progress stays OUT of the
domain view** — it is an env-time scalar; the encoder appends it last, exactly like
`OrdererEconomyView` (which carries no cycle progress). The view holds **ordered row lists**
(not id-keyed dicts) so the encoder can be order-equivariant (see §4).

- `ProducerGlobalStats` — cut-off price, inventory units/limit/headroom, outstanding inbound
  units, investment running flags (reuse `producer_active_investments`). **No cycle progress.**
- `CapacityRow` (per owned type `t`): owned, current/max/headroom/ratio, degradation,
  reserved, `previous_requested_capacity` (ReqPrev), free, reserved_ratio, restore/upgrade
  effect+cost, demand_pressure.
- `ProductRow` — **one row per product in `p.produces`** (the products this producer can act
  on). The universe is fixed: `produces` only. `eligible` is therefore `1.0` on every real
  row; it is kept as an explicit field (a) so the makeability metrics' eligible-guard stays
  visible and regression-tested, and (b) so padded rows read `eligible=0`. Fields: eligible,
  open demand qty/tokens, recipe shape (depth/input count/input units/capacity-type count),
  recent telemetry EMAs (routed/completed/rejected/latency via `EconomyTelemetry`),
  `active_reservation_qty`, derived makeability (`capacity_makeable_qty`,
  `input_makeable_qty`, `complete_qty_now`, bottleneck ratios). Padding to a fixed width is
  an **env concern** (§4), not a domain one — the view returns only real `produces` rows.
- `InputRow` (per input product relevant to `p` = `EnvSpec.producer_inputs[p]` semantics):
  inventory, outstanding inbound, `blocked_reservation_qty`, `downstream_demand_pressure` —
  all computable from current `ProducingAgent`/`Order` state. **Recent replenishment EMAs
  (requested/completed/rejected/latency) are DEFERRED, not included in this slice.**
  `EconomyTelemetry` tracks production/routing/demand EMAs only (`telemetry.py:79`); the
  requestor-side replenishment history those fields need has no source — `System.
  complete_cycle()` resolves replenishment completion/rejection (`system.py:404`) but records
  no telemetry there. Adding that plumbing (+ its tests) is its own change; pulling it into
  the contract slice is scope creep. The InputRow fields are simply omitted now (a later
  slice can add the telemetry and the fields together). No zero-placeholder columns — keeping
  them out of the schema avoids baking dead dims into the versioned obs.
- `Relations` — sparse edges **indexed by row position** (not raw ids), built against the
  same ordered row lists: `product_capacity_required[xi,ti]=R[x,t]`,
  `product_input_required[xi,gi]=I[x,g]`, `producer_owns_capacity[ti]`,
  `producer_can_make_product[xi]`, `product_uses_input[xi,gi]`. Storing by `(row_index,
  row_index)` is what keeps the relation planes aligned when the encoder permutes rows.
- `ProducerEntityView` bundles the above as ordered lists; `build_producer_entity_view(
  producer, telemetry, sheets, all_orders)` assembles it. Default row order is **sorted id**
  for determinism, but the order is a property of the returned lists — callers/tests may
  reorder, and the encoder will honor whatever order the lists carry.
- `System.producer_entity_view(pid)` wrapper (next to `orderer_economy_view`,
  `system.py:295`), passing `telemetry`, `sheets`, `self.orders`.

### 4. Env: versioned, opt-in encoder (`env/spaces.py` + `env/encode.py`)

- `EnvSpec`: add `producer_obs_version: int = 1` (default = legacy `flat_v1`). Add
  `producer_entity_obs_dim` + per-block width constants (capacity-row, product-row,
  input-row widths) mirroring the `ORDERER_*` constants. Fixed-width via padding to max
  rows: capacity rows → `num_types`, **product rows → `num_order_products`** (the safe upper
  bound on any producer's `produces` count in a topology), input rows → `num_inputs`. Each
  row carries a leading **active-mask** column (mask, not slot index, defines a real row —
  ADR-0053); padded rows are all-zero with mask 0. Relations → fixed `num_order_products ×
  num_types` and `num_order_products × num_inputs` edge planes, zero outside real rows.
- `build_spec(...)` gains `producer_obs_version` param (threaded like
  `orderer_obs_version`); `config_fingerprint(...)` hashes `producer_obs_version` and the
  new `producer_entity_obs_dim` so an entity-view checkpoint can't collide with a flat_v1
  one (the checkpoint-incompatibility gate).
- `encode_producing_entity(spec, view, cycle_progress) -> np.ndarray` in `encode.py`:
  emits blocks in the order `[global, capacity rows, product rows, input rows, relation
  planes, cycle_progress]`. It **consumes the view's row-list order verbatim** (does not
  re-sort by `EnvSpec`) — that is what makes it order-equivariant: a permuted row list
  yields correspondingly permuted row slots, and the relation planes (indexed by row
  position) permute with them. `cycle_progress` is appended **here**, last, as the single
  trailing scalar (matching the flat encoder and the orderer encoders). Normalize with the
  existing refs (`cap_ref`/`token_ref`/`price_ref`/`inv_ref`), clip to `OBS_CLIP`. The flat
  `encode_producing` stays untouched.

### 5. Env: opt-in wiring (`env/aec_env.py`)

- Constructor flag `entity_producer_obs: bool = False` → `producer_obs_version = 2`,
  threaded into `build_spec`. Default off, so every existing producer checkpoint keeps its
  exact flat contract.
- `observe()` (`aec_env.py:238`): when `producer_obs_version >= 2`, return
  `encode_producing_entity(spec, system.producer_entity_view(did), cp)`; else the current
  `encode_producing(...)` path. `producing_obs_space()` returns the entity Box when the
  flag is on.

### 6. Tests (`tests/test_producer_view.py`, new — mirror `test_orderer_view.py`)

Domain-fact pins + the ADR gates, all local:

- **Faithfulness**: build a `System`, advance to the producer decision point, assert
  capacity/product/input row values equal the metric functions and raw agent state. Product
  rows cover exactly `p.produces`; every real row has `eligible=1`; padded rows are all-zero
  with mask 0. Keep a unit pin that the makeability metrics still return 0 for a
  non-`produces` product (the eligible-guard regression).
- **No-learning parity harness** (covers ADR's "replay existing behavior through the new
  contract"): (a) **baseline byte-identical** — run a **`stock_aware` white-box** producer
  episode with `entity_producer_obs=False` vs `True`; episode stats + capacity trace
  identical (white-box baselines read `env.system`, not the obs, so the new contract must be
  non-invasive). **v4 is excluded here** — it is a *learned flat-v1 checkpoint* whose actor
  reads `env.observe(agent)` (`train/smoke_ppo.py:646`) and would get the wrong obs dim under
  the entity flag; a flat-from-entity actor adapter is **0010B's** problem, not this slice's.
  (b) **information-completeness (producer-local scope)** — assert every **producer-local /
  action-relevant** field the flat `encode_producing` emits is recoverable from the entity
  view: own per-type capacity/max/requested/degradation, investment-running flags, own input
  inventory + outstanding inbound, inventory units/limit/headroom, own producer-mechanics,
  and own-product open demand. The two flat **globals are deliberately NOT carried as
  flat ID-indexed vectors** (that is the anti-goal): all-order-product open demand
  (`_open_demand_features`, `encode.py:89`) and the all-producer summary (`_producer_summary`,
  2P). The view keeps only *this* producer's products/demand; if cross-producer context
  proves necessary it gets a masked **peer-producer row block** in a later slice, not a flat
  global. So the completeness test asserts the **local subset**, and explicitly documents the
  two intentionally-dropped globals — no "superset of flat" claim.
- **ReqPrev semantics**: after a `produce()` cycle, `previous_requested_capacity[t]` equals
  the prior cycle's per-type requested capacity (sum of every bid's `capacities[t]`,
  accepted or rejected), and is the *previous* cycle's pressure — not the current,
  not-yet-routed bids.
- **Row-order gate**: pass the encoder a view whose capacity/product/input row lists are
  shuffled → the encoded row slots permute identically and the relation planes permute with
  them (equivariance), while `ProducerGlobalStats` and any pooled field stay invariant.
  Pins that the encoder honors view order rather than re-sorting.
- **Product-relabeling gate**: remap product IDs in a config, rebuild; the view's product
  rows (un-relabeled back) are equivalent — behavior depends on row features, not IDs.
- **Capacity-relabeling gate**: same for capacity-type IDs.
- **Versioning gate**: `config_fingerprint(cfg, producer_obs_version=2) != ...version=1`;
  entity `producing_obs_dim` differs; a flat_v1 checkpoint dim does not match entity dim.
- **Encoder/spec dim tie** (mirror orderer "test D2"): `producer_entity_obs_dim` equals the
  length `encode_producing_entity` actually emits, across a couple of topologies.
- Keep `pettingzoo.test.api_test` green with the flag on (extend `test_env_api`): spaces
  are finite, fixed-width, and observations stay within the Box.

## Out of scope (later slices)

Reservation rows (deferred); deterministic producer planner (0010B); learned local scorer
(0010C); size-holdout audit (0010D); any training/cloud run.

## Verification

```
cd tokenomics
../.venv/bin/python -m pytest -q                      # all green incl. new test_producer_view
../.venv/bin/python -m pytest -q tests/test_producer_view.py tests/test_env_api.py
../.venv/bin/python -m tokenomics.eval.signal_check   # baselines unchanged (flag off by default)
```

Manual sanity: build `TokenomicsAECEnv(entity_producer_obs=True)`, `reset(seed=0)`,
`observe(prod_agent)` → finite, fixed-width, within `[0, OBS_CLIP]`; confirm the flat-obs
env (`entity_producer_obs=False`) is byte-identical to before.

After acceptance: flip ADR-0053 status `proposed → accepted`, add the
`experiments/EXP-0010A-producer-entity-view/` record (spec.json + RESULT.md + index row in
`experiments/README.md`), and update the memory status pointer.
