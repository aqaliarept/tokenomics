# Huge Economy Strategy Visualizer Draft

## Purpose

Build a strategy-game style replay viewer for huge tokenomics emulations with
thousands of producers, products, and consumer cohorts. The visualizer should make
large-scale behavior legible: supply chains, bottlenecks, outages, substitutions,
recovery, and policy effects over time.

The visualizer is replay-first. The simulator emits snapshots and events; the UI
loads those artifacts, lets the user scrub time, inspect state, and compare runs.
Live control can come later once offline replay is useful.

## Design Principles

- Keep simulation and visualization separate. The UI must not drive domain logic.
- Render aggregates first, then allow drill-down into individual producers/products.
- Make shocks visible as first-class events, not as unexplained metric jumps.
- Prefer sparse data contracts over full environment observations.
- Keep the first screen useful without explanation text: map, timeline, overlays,
  and inspector panels.
- Treat the game layer as an operational command map, not a marketing dashboard.

## Core Experience

The first screen is a 2D strategy map of the economy.

- Regions are large map areas.
- Producer clusters are nodes inside regions.
- Product categories are colored flow layers.
- Consumer cohorts are demand centers.
- Supply routes are animated edges.
- Shocks appear as outage markers, route breaks, demand spikes, and scarcity heat.
- A timeline at the bottom shows cycles, shock windows, fulfillment drops, and
  recovery points.

Users can:

- Play, pause, step, and scrub cycles.
- Change replay speed.
- Rewind to a prior checkpoint, edit the scenario, and replay consequences.
- Toggle overlays for fulfillment, backlog, outage, capacity health, inventory
  scarcity, and price pressure.
- Click a region to inspect aggregate state.
- Click a producer cluster to drill into top producers and affected products.
- Click a product to inspect demand, supply, substitutes, and bottleneck chain.
- Compare baseline and intervention replays.

## Time Travel and Past Correction

The visualizer should support strategy-game "what if" editing: move back to a
checkpoint, add/remove/change an event, and replay all later cycles from that edited
history.

This is not just UI scrubbing. Scrubbing reads an existing replay. Past correction
creates a new branch and requires the simulator to recompute the consequences.

### Core Semantics

- Checkpoints capture complete simulator state at selected cycles.
- Scenario events after the checkpoint are editable inputs, not immutable facts.
- Editing history creates a new branch id.
- The simulator restores the checkpoint, applies the edited event stream, and reruns
  all later cycles.
- Derived events such as reroutes, rejections, substitutions, shortages, and recovery
  times are regenerated, not manually patched.
- The original replay remains available for comparison.

Example:

1. User rewinds to cycle 40.
2. User removes a producer outage at cycle 42.
3. UI creates branch `run-a/no-outage-p42`.
4. Simulator restores checkpoint cycle 40.
5. Simulator reruns cycles 40-200 with the edited scenario.
6. UI shows delta overlays between original and corrected timelines.

### Event Edit Types

Supported first:

- Add a shock.
- Remove a shock.
- Change shock start cycle.
- Change shock duration.
- Change shock severity.
- Change affected producer, product, route, region, or cohort.

Later:

- Change routing policy from a cycle onward.
- Add subsidy or emergency replenishment.
- Lock a producer online/offline.
- Change consumer cohort preferences.
- Clone a branch and compare multiple interventions.

### Branching Replay Model

The run directory should allow immutable base runs plus derived branches.

```text
runs/huge-demo-0001/
  manifest.json
  topology.json
  cycles.ndjson
  events.ndjson
  checkpoints/
    cycle_000000.bin
    cycle_000025.bin
    cycle_000050.bin
  branches/
    no-outage-p42/
      branch.json
      cycles.ndjson
      events.ndjson
      summary.json
    longer-input-shock/
      branch.json
      cycles.ndjson
      events.ndjson
      summary.json
```

Branch metadata:

```json
{
  "schema_version": 1,
  "branch_id": "no-outage-p42",
  "parent_run_id": "huge-demo-0001",
  "base_checkpoint_cycle": 40,
  "edit_description": "Removed producer outage p_182 at cycle 42",
  "scenario_patch": [
    {
      "op": "remove",
      "event_id": "evt_outage_p182_42"
    }
  ]
}
```

### Checkpoint Requirements

A checkpoint must include enough state to rerun deterministically:

- System/domain state.
- Producers, inventories, reservations, active investments, cut-off prices.
- Orders, balances, in-flight replenishment, and telemetry.
- Consumer cohort state and random generator state.
- Shock scheduler state.
- Policy state if policies are adaptive.
- Cycle counter and id generators.

The checkpoint format can start as Python pickle for internal prototypes. Before
shipping or long-lived artifacts, prefer a versioned schema or explicit binary format
so old replays remain readable.

### UI Behavior

The timeline should distinguish:

- Current replay branch.
- Parent branch.
- Checkpoint markers.
- Editable scenario events.
- Derived consequence events.

When the user edits past history:

- Show a branch creation confirmation.
- Keep the original timeline visible as a ghost/delta layer.
- Mark cycles after the edit point as stale until recomputation finishes.
- Stream recomputed cycles into the viewer if live branch generation is available.
- Show impact deltas: fulfilled units, rejected units, backlog, outage exposure,
  substitution volume, and recovery time.

### Simulator Contract

The visualizer should not mutate replay files directly. It should send a branch
request to the simulator or offline branch builder:

```json
{
  "parent_run_id": "huge-demo-0001",
  "base_checkpoint_cycle": 40,
  "scenario_patch": [
    {"op": "remove", "event_id": "evt_outage_p182_42"},
    {
      "op": "add",
      "event": {
        "cycle": 45,
        "type": "demand_spike",
        "product_id": 1204,
        "region_id": "north",
        "multiplier": 1.8,
        "duration": 10
      }
    }
  ]
}
```

The branch builder returns:

```json
{
  "branch_id": "branch-0003",
  "status": "running",
  "output_path": "runs/huge-demo-0001/branches/branch-0003"
}
```

### Determinism Rule

Past correction only works if replay is deterministic. Any random behavior must be
seeded and checkpointed. If two branch runs start from the same checkpoint with the
same scenario patch, they must produce the same replay artifacts.

## Game Objects

### Region

A region groups producers, products, routes, and consumer cohorts for level-of-detail
rendering.

Fields:

- `id`
- `name`
- `position`
- `producer_count`
- `consumer_cohort_count`
- `product_count`
- aggregate fulfillment, backlog, capacity, outage, and inventory metrics

### Producer

An individual productive entity. At high zoom, producer nodes can be rendered
directly; at low zoom they are aggregated into clusters.

Fields:

- `id`
- `region_id`
- `products`
- `capacity_current`
- `capacity_max`
- `inventory_units`
- `inbound_units`
- `cut_off_price`
- `status`: `online`, `offline`, `degraded`, `recovering`
- recent completed, rejected, routed, and failed quantities

### Product

A product is a node in the recipe/supply graph.

Fields:

- `id`
- `name`
- `category`
- `recipe_inputs`
- `eligible_producer_count`
- `demand_units`
- `fulfilled_units`
- `rejected_units`
- `backlog_units`
- `substitute_product_ids`

### Consumer Cohort

An aggregate demand actor representing many similar consumers.

Fields:

- `id`
- `region_id`
- `population`
- `budget`
- `preference_weights`
- `substitution_weights`
- `price_sensitivity`
- `availability_sensitivity`
- current demand, rejected demand, and substitutions

### Route

A flow edge between regions, products, producers, or cohorts.

Fields:

- `source`
- `target`
- `product_id`
- `quantity`
- `latency`
- `status`: `normal`, `delayed`, `blocked`, `rerouted`

## Shocks

Shocks are deterministic, seeded scenario events. They should be shown both on the
map and on the timeline.

### Producer Outage

A producer or producer cluster is unavailable for a time window.

Semantics for the demo runner:

- Exclude unavailable producers from new routing candidate sets.
- Force unavailable producers to no-op for producer actions.
- Defensively reject new sub-orders routed to unavailable producers.
- Let existing reservations age normally unless the shock mode says to pause or
  fail them.
- Let inbound replenishment arrive unless a logistics shock also affects it.

Event example:

```json
{
  "cycle": 40,
  "type": "producer_outage_start",
  "producer_id": "p_182",
  "region_id": "north",
  "duration": 12,
  "severity": "offline"
}
```

### Capacity Shock

Temporary reduction in capacity, degradation spike, or recovery delay.

### Input Shock

Scarcity, price increase, or delayed replenishment for a product/input.

### Demand Shock

Preference shift, demand spike, demand collapse, panic buying, or substitution wave
for one or more consumer cohorts.

### Logistics Shock

Routes are delayed, blocked, or failure-prone for a window.

## Replay Data Contract

The visualizer should load a run directory with metadata, static topology, per-cycle
snapshots, and event rows.

Proposed files:

```text
runs/huge-demo-0001/
  manifest.json
  topology.json
  cycles.ndjson
  events.ndjson
  summary.json
```

### `manifest.json`

```json
{
  "schema_version": 1,
  "run_id": "huge-demo-0001",
  "seed": 7,
  "cycles": 200,
  "producer_count": 2000,
  "product_count": 5000,
  "consumer_cohort_count": 10000,
  "region_count": 24,
  "created_at": "2026-06-25T00:00:00Z"
}
```

### `topology.json`

Static map and graph data:

- regions
- producer clusters
- products and recipe edges
- consumer cohorts
- route graph
- product categories and color palette

### `cycles.ndjson`

One JSON object per cycle. Keep this aggregated enough for fast loading.

```json
{
  "cycle": 42,
  "fulfilled_units": 18320,
  "rejected_units": 940,
  "backlog_units": 2110,
  "offline_producers": 37,
  "degraded_producers": 92,
  "active_routes": 14088,
  "blocked_routes": 17,
  "top_bottleneck_products": [1204, 918, 77],
  "regions": [
    {
      "id": "north",
      "fulfilled_units": 3011,
      "rejected_units": 220,
      "backlog_units": 408,
      "offline_producers": 11,
      "capacity_health": 0.74,
      "inventory_pressure": 0.63
    }
  ]
}
```

### `events.ndjson`

Sparse event stream.

```json
{"cycle": 40, "type": "producer_outage_start", "producer_id": "p_182", "duration": 12}
{"cycle": 43, "type": "reroute", "product_id": 918, "from": "p_182", "to": "p_441", "quantity": 31}
{"cycle": 45, "type": "demand_spike", "product_id": 1204, "region_id": "north", "multiplier": 2.1}
{"cycle": 52, "type": "producer_outage_end", "producer_id": "p_182"}
```

## UI Layout

### Main Map

Canvas/WebGL-rendered 2D map.

- Region polygons or tiles.
- Producer clusters as compact glyphs.
- Consumer cohorts as demand hubs.
- Animated flow edges.
- Shock icons and outage bands.
- Heat overlays for backlog, rejection, inventory pressure, and capacity health.

### Timeline

Bottom panel.

- Cycle scrubber.
- Play/pause/step/speed controls.
- Shock event bands.
- Fulfillment/rejection/backlog mini chart.
- Selected event markers.

### Inspector

Right panel. Content depends on selected object.

Region inspector:

- fulfillment
- backlog
- offline/degraded producers
- top bottleneck products
- recovery time estimate

Producer inspector:

- status
- products made
- capacity current/max
- inventory and inbound
- accepted/completed/rejected recent flow
- active shocks

Product inspector:

- demand, fulfilled, rejected, backlog
- eligible producer count
- current bottleneck stage
- substitute flow
- affected regions

Scenario inspector:

- active shocks
- upcoming shocks
- recovery metrics
- baseline vs intervention difference

## Visual Encoding

Suggested encodings:

- Green/blue flow: fulfilled supply.
- Amber flow: delayed or rerouted supply.
- Red markers: outage or hard rejection.
- Purple markers: demand shock.
- Gray producers: offline.
- Hollow producers: degraded.
- Thicker routes: higher quantity.
- Pulsing edge: active reroute.

Keep labels sparse. Show aggregate labels at low zoom and detailed labels only for
selected objects or high zoom.

## Technical Direction

Initial frontend stack:

- React + TypeScript.
- PixiJS or Canvas/WebGL for the main map.
- Zustand for replay state.
- D3 or lightweight SVG only for side charts.
- Local static file loading for the first version.

Avoid rendering the huge graph as DOM nodes. Thousands of producers/products should
go through Canvas/WebGL with level-of-detail aggregation.

## Implementation Phases

### Phase 0: Static Draft

- Create this document.
- Define the initial replay schema.
- Decide whether replay artifacts live under `runs/` or `experiments/`.

### Phase 1: Fake Replay Viewer

- Build a frontend with generated fake topology/events.
- Implement map, timeline, overlays, and inspectors.
- No simulator integration yet.

### Phase 2: Demo Runner Export

- Add huge direct runner export for `manifest.json`, `topology.json`,
  `cycles.ndjson`, `events.ndjson`, and `summary.json`.
- Emit producer outage, reroute, demand shock, and backlog events.

### Phase 3: Real Huge Economy

- Integrate the post-EXP-10 producer implementation.
- Add behavioral consumer cohorts.
- Run deterministic huge scenarios.
- Validate replay size, loading speed, and visual legibility.

### Phase 4: Interactive Strategy Mode

- Trigger shocks from the UI.
- Run intervention scenarios.
- Compare baseline vs intervention.
- Add policy toggles such as routing policy, subsidy, reserve inventory, or
  emergency replenishment.

## Open Questions

- Should consumer cohorts be visualized as regions only, or as explicit hubs?
- Should routes represent product flow, money/token flow, or both as selectable
  overlays?
- What is the smallest meaningful huge demo target: 1k producers/2k products or
  5k producers/10k products?
- Should producer outages pause reservations, age them, or fail them immediately by
  severity?
- Do we want a map-like geographic layout, a supply-chain graph layout, or both?

## Near-Term Recommendation

Start with a fake replay generator and viewer before wiring to the simulator. That
lets the visual design mature while EXP-10 producer semantics are still changing.
The stable contract should be replay artifacts, not live producer internals.
