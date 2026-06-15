# Death detection per declared product + non-positive cost validation

Status: accepted

## Context

Two gaps surfaced after `produces` became authoritative (ADR-0028):

1. **Death detection ignored `produces`.** `System.is_dead()` declared the system
   dead only when a capacity *type* hit zero across **every** owner, treating
   capacity as interchangeable across producers. But eligibility is now declared:
   when several producers own a type yet only one *declares* a given product,
   exhausting that one producer makes the product permanently unmakeable — while
   the type is still "alive" in a producer that does not declare it. The old check
   returned `False`, so a dead supply chain ran to the horizon.

2. **`validate()` accepted non-positive funding fields.** Zero/negative
   `restoration.cost` / `upgrade.cost` / `inventoryExpansion.cost`,
   `replenish_cost_per_unit`, and `max_replenish` passed validation. A zero cost
   funds nothing (the order never clears its target, ADR-0027); a non-positive
   `max_replenish` corrupts the env's replenishment apportionment.

## Decision

**Death is judged per declared product.** A product is unmakeable when no producer
that declares it (`product in produces`) still holds **every** capacity type its
recipe requires (`capacity[t] > 0`). The system is dead the moment any
process-sheet product is unmakeable:

```
is_dead = any(
    not any(all(cap[t] > 0 for t in sheet.require)
            for producer in producers if product in producer.produces)
    for product, sheet in process_sheets
)
```

This reduces to the old type-level rule for the Phase 1 economy (one declaring
producer per product, one type per producer) but correctly detects death when
ownership and declaration diverge. `validate()` guarantees every product has at
least one declaring producer, so the inner `any` is never vacuously dead at start.

**`validate()` rejects non-positive funding fields.** Each of `restoration.cost`,
`upgrade.cost`, `inventoryExpansion.cost`, `replenish_cost_per_unit`,
`max_replenish` must be positive when present (`None` stays "unbounded /
default").

## Considered options

- **Keep type-level death.** Rejected: contradicts authoritative `produces` — a
  broken chain is reported alive.
- **Death only on the consumer product's recipe closure** (ignore products no one
  needs). Rejected as premature: `validate()` already requires every sheet to be
  produced, so every product is part of the economy; per-product over the full
  sheet set is simpler and strictly safer.
- **Use per-unit capacity thresholds** (`capacity[t] >= require[t]`) for death.
  Rejected: that is transient under-capacity (recoverable by restoration), not the
  terminal zero condition death models; would terminate prematurely.

## Consequences

- Broken supply chains terminate at the cycle they break instead of running to the
  horizon — cleaner episode signal and credit assignment for training.
- `signal_check` is unchanged (Phase 1 economy: `stock_aware` + `even_split` = 100
  cycles, 0% death, 100/100 consumer, return 1267); `noop`/`random` still collapse
  exactly as before.
- Misconfigured costs are caught at load time rather than silently funding nothing.
