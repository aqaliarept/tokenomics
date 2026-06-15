# Consumer reward and metrics are weighted by fulfilled units, not order counts

Status: accepted

## Context

A consumer order demands a *quantity* of a product as a whole (ADR-0025): a qty-3
order consumes ~3x the capacity (and input products) of a qty-1 order, and completes
or rejects atomically. Until now the training reward and the eval metrics counted
**orders**, not units: `cycle_reward` paid `+10` per completed consumer order and
`-5` per rejected order regardless of quantity, and `episode_stats`/`evaluate`
tracked order counts (`consumer_completed`, `consumer_rejected`).

While demand was fixed at quantity 1 (canonical `phase1.json`) this was exact. B9
introduced seeded/randomized demand with quantities 1–N. Under order-count scoring a
policy that fulfils many small orders and drops the large ones scores the same as
one that fulfils the large ones — the metric hides where the real demand (and the
real capacity cost) went, and the reward gives no gradient toward the expensive-but-
valuable large orders. The signal_check table likewise could not show this.

## Decision

**Weight the consumer reward terms and consumer metrics by fulfilled / unmet
quantity (units).**

- `domain/`: `CycleResult` gains `consumer_completed_units` and
  `consumer_rejected_units`. `System.complete_cycle` accumulates `order.quantity`
  onto the completed/rejected unit counters as each consumer order resolves (whole-
  order semantics, ADR-0025). These are pure counts — no RL, no IO; domain stays
  pure.
- `env/`: `cycle_reward` pays `R_CONSUMER_PER_UNIT (= 10)` per delivered unit and
  `R_CONSUMER_REJECTED_PER_UNIT (= −5)` per unmet unit. Investment/replenishment
  completions stay per-order (`+1`), in-progress stays per-order (`−1`), and
  producer-order (investment/replenishment) rejections stay per-order (`−5`,
  `R_REJECTED`). The inventory holding penalty is unchanged.
- `eval/`: `episode_stats`, `evaluate`, and the `signal_check` table expose
  `consumer_completed_units` (new `cu✓` column) alongside the order counts.

At quantity 1 every unit weight equals the old per-order weight, so phase1 behavior
and the baseline bar are unchanged (`stock_aware` = 100 cycles, 0% death, 100/100
consumer, return 1267, and `cu✓ == cons✓`).

## Considered options

- **Replace order counts with unit counts in the headline `cons✓` metric.**
  Rejected: `cons✓` (orders fulfilled) is the established objective and the
  `test_eval` guardrail; changing its meaning would silently move the bar. Units are
  added as a parallel column/metric instead.
- **Weight investment/replenishment by units too.** Deferred: those are
  instrumental and mostly quantity-1 (investments) or already capacity-priced; the
  objective is consumer fulfilment, so only consumer terms are unit-weighted now.
- **Compute units in the env from order views instead of the domain.** Rejected:
  the per-order quantity is domain state resolved exactly where the complet/reject
  event fires; surfacing it on `CycleResult` keeps the env a thin consumer of domain
  accounting (no re-derivation, no drift).

## Consequences

- Reward magnitudes on randomized-demand configs rise/fall with order size, giving a
  gradient toward fulfilling large orders rather than gaming order count.
- No checkpoint-shape change (reward/metrics only), so `CHECKPOINT_VERSION` is
  unaffected by this ADR (it was bumped to 4 by ADR-0032 in the same batch).
- The plan's target numbers were set on order-count return; randomized-demand return
  must be re-read against the new unit-weighted reward when the suite is retrained.
- **Curriculum gate reads units too (review P3, 2026-06-15).** The evidence-gated
  curriculum (ADR-0009) measured fulfillment by *order counts*: under variable
  quantity, four completed qty-1 orders + one rejected qty-3 order clear an 80% order
  gate while delivering only 4/7 = 57% of demanded units — advancing difficulty against
  this ADR's own quantity-weighted objective. `RolloutStats` now carries
  `completed_units`/`rejected_units` (defaulting to order counts for the qty=1
  identity), `collect()` tallies them from the episode stats this ADR added, and
  `Curriculum.observe` gates on unit-weighted fulfillment.
