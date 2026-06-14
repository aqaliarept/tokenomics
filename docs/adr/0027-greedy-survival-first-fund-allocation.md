# Greedy survival-first fund allocation with per-order cost targets

Status: accepted

## Context

ADR-0025 funds producer purchasing in two stages: stage 1 splits the central
fund across producers by cut-off price; stage 2 spends each producer's budget
across its own orders by the fixed survival-first priority **restoration →
replenishment → upgrade → inventory expansion**, "greedily until the budget is
exhausted, unfunded requests expiring."

B4.5 shipped stage 2 as an interim **even split**: each order got `budget // n`,
the remainder going to the highest priority. Review found this is not greedy —
with budget 10 over four equal orders it pays 3/3/2/2, advancing to lower
priorities before higher ones are satisfied, and (when `budget < n`) marks
zero-token orders funded, letting low-priority work execute for free. Defining
"greedily until exhausted" requires a **per-order funding target**: how much an
order needs before allocation advances to the next priority. The domain had none.

## Decision

**Each producer order carries a funding target = a configured token cost.**

- Restoration / upgrade / inventory expansion: a `cost` field on each investment
  config (`Restoration.cost`, `Upgrade.cost`, `InventoryExpansion.cost`).
- Replenishment: `ProducingAgentConfig.replenish_cost_per_unit × quantity`.

The cost rides on the emitted request (`InvestmentRequest.cost`,
`ReplenishmentRequest.cost`) and is exposed as `Order.funding_target()`.

**Stage 2 is greedy.** Walk a producer's orders by fixed priority; fill each up
to its target, advancing to the next only while budget remains. The boundary
order takes whatever is left (partial funding); lower-priority orders get nothing
and **expire unfunded** — they are skipped in `start_ordering` and rejected (−5)
in `complete_cycle` via `Order.expire()`, clearing the producer's running
flag / outstanding-inbound so it can retry next cycle. No order is ever funded 0
tokens, so no producer work executes for free.

**`cost = None` means unbounded** — the order takes the whole remaining budget.
This is the default when a config omits costs (legacy configs, inline test
fixtures): the highest-priority order absorbs the budget and same-tier siblings
expire. Real configs (`phase1.json`, `composite.json` once re-derived) set costs
so each priority is filled to a finite target.

## Considered options

- **Cut-off-price target** (`cost = ceil(cut_off_price × total_capacity)`).
  Rejected: cut-off is NaN before any production (cycle 1), leaving the target
  undefined exactly when the fund first matters, and needs a fallback price anyway.
- **Whole-budget-to-top-tier** (fund only the lowest-priority-number tier, expire
  the rest each cycle). Rejected: a producer could never restore *and* replenish
  in the same cycle even with ample budget — too harsh for normal operation.
- **Keep even split.** Rejected: not greedy; pays lower priorities before higher
  ones are satisfied and can fund 0-token orders.

## Consequences

- Config gains optional cost fields: per-investment `cost`, producer-level
  `replenishCostPerUnit` (and `maxReplenish` for the B5 env replenish ceiling).
  Loader parses them; all default to `None` (unbounded / backward compatible).
- Under a tight budget, lower-priority orders deterministically expire (rejected,
  −5) — the intended survival-first pressure, now with real teeth instead of an
  even split. Under ample budget every order is funded to its cost and leftover
  budget is simply unspent (tokens carry no value).
- `phase1.json` / `composite.json` must set costs when B7 re-derives them, else
  the unbounded default makes same-tier orders starve each other.
