# Design: add-producer-acceptance-policy

## Context: the cycle today (`domain/system.py`)
1. `start_cycle`: emission, consumer orders, cycle counter, capacity hits.
2. Producers act (`producing_agent_action`): investments and replenishment.
3. `start_ordering`: the fund is distributed; open orders go to the ordering agents.
4. Ordering agents route capacity-free `SubOrder`s (ADR-0029). Each producer expands them into `Bid`s from its own
   sheets.
5. `complete_cycle` → `ProducingAgent.produce()`, in order:
   - re-validate reservations against post-degradation capacity (drop → fail);
   - reserved capacity blocks the budget;
   - assemble reservations whose inputs arrived, or age them toward TTL;
   - the **atomic price-greedy auction**: accept iff the whole bundle fits now (else `rejected`, reason no-fit /
     displaced); tokens are burnt at acceptance;
   - the cut-off price is the lowest accepted bid price;
   - degrade.

   Then orders resolve: a rejected sub-order's quantity is re-routed by the orderer next cycle within the TTL.

Eligibility is declared (`produces`, ADR-0028), and the producer owns the recipe and capacity (ADR-0029).

## Decisions (each with a default; every one is a decision for the user)

**D1. Domain = feasibility + execution. Acceptance is a producer decision once per cycle, after routing.**
- **When:** after step 4, every producer's acceptance policy sees its routed sub-orders (with each bundle and
  price) and its own state, and returns accept / decline per sub-order. One decision per cycle, with no same-cycle
  re-routing of declined work; the orderer acts on it next cycle. That keeps the cycle non-recursive.
- **An accepted sub-order** becomes a reservation that executes as soon as capacity and inputs allow, possibly in
  a later cycle, until the order's TTL. Free capacity at acceptance is not a veto. If it is still unfinished at
  TTL, it **fails** (reason: TTL).
- **Execution order:** the domain fills accepted reservations by producer-chosen priority (default: price, then
  age).
- **Tokens:** burnt at acceptance, as the glossary says, so a declined sub-order burns nothing.
- **Cut-off price:** the lowest accepted price this cycle. Unchanged in meaning; fund distribution still uses it.
- **Re-validation:** reservations are capacity claims against post-degradation capacity per cycle, without
  ADR-0025's "reserved at acceptance, no overcommit". Overcommit becomes allowed, which is the user's point, and
  an over-claimed reservation just waits.
- **Shocks (ADR-0055):** an outage still blocks completion, unchanged.

**D2. Explicit per-sub-order outcomes.**
- **States:** `accepted` / `declined` / `completed` / `failed` (reasons: TTL, dropped, outage), recorded per
  sub-order with its cycle.
- **Visibility:** the ordering agent that routed it sees it in its view next cycle (`OrdererEconomyView`
  additive: last cycle's outcomes per product × producer).
- **Additive only:** no existing encoder reads the new fields (as `CapacityRow.degradation_rate`). Today's EMA
  telemetry stays.

**D3. Today's auction = the default acceptance policy, byte-identical.**
- **Default:** `acceptance_auction` = sort by price, accept iff the bundle fits the budget left now, else
  decline (reason no-fit / displaced). Execution follows today's path.
- **Placement:** acceptance policies live in `eval/` (like producing policies), passed through the env.
- **Gate:** goldens (`test_legacy_invariance`) plus a replay of EXP-0088's k5_rf5 rows; every recorded experiment
  stays valid.

**D4. Env / AEC.**
- **The new phase:** after the last ordering agent, every producer agent acts again with an acceptance action,
  before `complete_cycle`.
- **Action shape:** a fixed max K sub-order slots per producer (the env's existing slot convention for orders).
  The action is accept bits plus a priority per slot; slots beyond K are declined by default.
- **Existing policies:** they keep their `Policy` signature for the investment phase. The acceptance phase calls
  a separate acceptance policy, the default D3's, so every current policy, checkpoint and gate run is unchanged.
- **Learned acceptance:** not in this change.

**D5. First experiments after the build**
All with k5_rf5 and MWWk R = 2 / RR, TTL 2, EXP-0057's worlds, supply observer on; reading as EXP-0084's, 12
deciding.

| arm | rule | size |
|---|---|---|
| A0 | default (today) | a 2-episode replay check only, no full run (the rows are EXP-0088's) |
| A1 | "accept if it can finish before TTL": fits now, or the claimed capacity frees within TTL given the accepted queue | ≈ 0.9 vCPU-h |
| A2 | A1 + producer-side input check: decline a composite whose inputs are neither in stock nor in flight to arrive within TTL (duplicating the orderer's makeable check) | ≈ 0.9 vCPU-h |

Total ≈ 2 vCPU-h, stop 4.

**Pre-registered readings, before data:**
- `helps` if mean_w(diff) > mean_w(floor) for both orderers, with no extra deaths and run-down within +5;
- `harmful` if below −floor, or extra deaths;
- else `no_gain`.
- **Reported:** late inputs, outbid / no-fit, declines, TTL failures, held stock.

## Risks
- Overcommit can lengthen queues and raise TTL failures, which A1's "can finish" rule is meant to bound.
- A larger action space for the producer.
- Orderer behaviour trained on today's auction (EXP-0012) sees a different producer; reads compare against
  baselines only.
