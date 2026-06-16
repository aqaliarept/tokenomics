# Scaled-economy generator + scale-test findings

Status: accepted

## Context

NEXT STEPS stage 5 ("scale test") asks for a larger economy — 8-16 products, 10-20
producers, 3-5 consumers, supply chains 3-5 levels deep, several eligible producers
per important product, episode-seeded demand and *bounded configuration variation* —
while **keeping obs/action dimensions fixed** across the suite (randomize values and
feasible topology within that envelope).

The dims envelope (`env/spaces.py`) is determined by `num_types`, `num_inputs`,
`num_order_products`, `num_producers`, and `k = max(ORDER_LIFETIME, INVESTMENT_STREAMS
+ ORDER_LIFETIME * max_rel)`. Crucially `k` depends on `max_rel` (the max distinct
inputs a single producer can replenish), so even the per-producer `produces`
assignment must be held fixed for ordering dims to be seed-invariant.

## Decision

Add **`eval/scale.py::build_scale_config(seed, *, depth, products_per_layer,
producers_per_layer, num_consumers)`** — a pure, deterministic generator of a layered
economy:

- **Fixed structural skeleton (→ fixed dims):** a `depth`-level recipe DAG (product id
  = level*10+idx; capacity type = `capacity-{level}`), `products_per_layer` products
  per level, each level-i product consuming the same-index level-(i-1) product (so the
  input-product set is constant); `producers_per_layer` producers per level, each
  declaring *all* of its level's products (so `max_rel` and thus `k` are
  seed-invariant); `num_consumers` consumers demanding the top layer.
- **Seed-driven bounded variation (dims-preserving):** capacities, degradation,
  investment amounts/costs, inventory limits, and demand ceilings are randomized; and
  within a level capacities split declarers into *strong* and *marginal* — feasible-
  topology variation (who can actually fulfil), the competition.json mechanism, without
  touching dims.
- Consumers carry a randomized product pool + quantity range, so demand is
  episode-seeded by the env RNG (B9), not scripted.

Defaults (depth 3, 4 products/level, 4 producers/level, 3 consumers) → 12 products, 12
producers, 3 consumers, inside the envelope.

## Considered options

- **Vary `produces` per seed for topology variation.** Rejected: it changes `max_rel`
  → `k` → ordering dims, breaking the fixed-dims requirement. Feasible-topology
  variation is expressed through capacity values (strong/marginal) instead.
- **Default to depth 4-5.** Rejected as the default (kept as a parameter): see
  Finding A — deeper chains are currently unfulfillable within an episode.

## Consequences / scale-test findings (obs baseline vs do-nothing vs oracle, depth 3, 5 seeds/config, horizon 60)

**Dims are fixed across config variation.** All 8 config seeds produce identical
`(producing_obs_dim, ordering_action_dim) = (70, 180)` — one trained net spans the
whole bounded-variation suite, as stage 5 requires.

| config seed | policy          | dead% | cons✓ | units✓ | return |
|------------:|-----------------|------:|------:|-------:|-------:|
| 0           | noop            |   100 |   0.0 |    0.0 |  -544 |
| 0           | obs_stock_aware |     0 |  45.6 |   64.4 | -6487 |
| 0           | oracle          |     0 |  62.4 |  103.4 | -3341 |
| 50          | obs_stock_aware |     0 |  93.6 |  170.0 | -3045 |
| 51          | obs_stock_aware |     0 |  84.8 |  143.8 | -4827 |

The env **runs correctly at scale** (≈27-35 PettingZoo agents per cycle) and, at depth
3, **poses a clean learning problem on consumer fulfilment**: the obs-compatible
baseline survives (0% death) and fulfils 45-93 orders where do-nothing fulfils 0 and
dies. Two blockers, though, must be fixed before scale RL training is meaningful:

- **Finding A — depth ≥4 is infeasible.** At depth 4 consumer fulfilment collapses to
  **0** (174 orders rejected, 0 completed) even for the obs/oracle baselines: the
  multi-hop pipeline cannot prime fast enough against order TTL=2. Depth 3 fulfils
  (58 orders / 112 units in the same single-episode probe). So "3-5 levels deep" is
  only fulfillable at the shallow end without raising TTL or extending priming.
- **Finding B — `return` is not a valid scale objective.** At producer scale the
  return is dominated by *investment-order churn* (the baseline issues
  restore/replenish/upgrade orders across many producers; rejected −5 / in-progress −1
  per order swamp the +10 per consumer order). Consequently **do-nothing has the
  least-negative return** (noop −544 > oracle −3341 > obs −6487): a return-maximising
  policy at scale would degenerate to inaction. The valid scale signal is
  **fulfilled-unit rate**, not return. The score/reward must be recalibrated for
  many-producer economies (scale down or cap investment-order penalties / weight
  consumer fulfilment) before scale PPO is run.

**Verdict: stage-5 infra DELIVERED and verified (valid economies, fixed dims, env runs
at scale, clean fulfilment signal at depth 3); scale RL training DEFERRED** pending
the reward recalibration (Finding B) and the depth/TTL feasibility fix (Finding A).
These join the consolidated reward/curriculum follow-up from stages 2-4. Training on
the current return at scale was deliberately not run — the objective rewards inaction.
