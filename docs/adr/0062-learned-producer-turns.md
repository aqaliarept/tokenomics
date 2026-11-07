# ADR-0062: Acceptance, fund spend order and claim scheduling are producer turns

Status: accepted (supervisor review 2026-11-07) (add-m6-joint-mpg-gnn tasks 4.2 / 4.6 / 4.7; user decisions 2026-11-07: learned acceptance, and
the fund spend order and reservation scheduling as learned producer decisions; audit notes/hardcoded-audit.md)

## Context
ADR-0061 made acceptance a producer policy, but only as an env hook (`env.acceptance_rule`) that the domain calls
inside `complete_cycle`. A learned producer cannot act through it as an agent. Two more producer decisions are
still hand-written rules on the trained stack's path (the audit's items 2 and 4):
- **Fund stage 2:** a producer's budget is spent restoration → replenishment (by order id) → upgrade → expansion,
  each order filled to its target and the boundary order partly funded (`distribute_investment_fund`).
- **Claim scheduling:**
  - carried reservations are re-checked after degradation price-descending, so the cheapest are dropped first;
  - they take input stock before this cycle's bids;
  - under a policy, pending claims run before new bids (ADR-0061 step 3b).

The user's goal (2026-11-06): no hand-written rule decides inside the trained team.

## Decision
1. **Two new producer phases in the AEC env, opt-in** (`TokenomicsAECEnv(producer_turns=True)`). Off: the agents,
   spaces, step order and every row are unchanged.
   - **Fund turn** (`fund_<pid>`): after the last investment action, before `start_ordering`. One priority score
     per funding request (the producer's own unfunded orders of this cycle).
   - **Accept turn** (`acc_<pid>`): after the last ordering agent, before `complete_cycle`.
     - One accept logit and one priority per routed bid.
     - One priority per carried reservation and per pending claim.
   - Each phase has its own agents because each has its own observation and action space. **One shared producer
     policy acts in all three phases** (CTDE, two shared policies): its GNN has a head per phase.
2. **Timing.**
   - The fund must be spent before routing, because only funded orders are routed. So the spend order is decided
     in the fund turn, not in the accept turn.
   - Nothing changes a producer's state between its accept turn and its `produce()` in the same `complete_cycle`.
     So the turn sees exactly the state its decisions apply to.
3. **The domain keeps physics:**
   - feasibility (capacity fits, inputs in stock), execution, the TTL, degradation, token accounting;
   - fund stage 1 (the split across producers by cut-off).
   - The turns give scores only; the domain applies them in score order:
     - **Funding:** orders are filled in descending score, each to its target, the boundary order partly funded
       (stage 2's mechanics).
     - **Scheduling (one priority order over every claim the producer holds):**
       - Re-validation keeps reservations by descending score; the lowest is dropped first.
       - Then all claims are processed in descending score:
         - a carried reservation assembles if its inputs are in stock, otherwise it ages, and at TTL it fails
           and frees its capacity;
         - a pending or accepted new claim takes capacity if it fits; it assembles, or holds the capacity as a
           reservation when it lacks inputs; otherwise it waits as a pending claim.
     - **Declines:** a declined bid burns nothing. Its recorded reason is `no_fit` when it does not fit the
       capacity the producer had for new work, `displaced` when it would have fit after the higher-priced
       accepted bids, and `declined` otherwise.
   - Outage: decisions are ignored (no bid accepted, no assembly), as ADR-0055. Re-validation and the claim
     order then follow today's order (carried first, price-descending), whatever the scores.
4. **Defaults reproduce today byte for byte.** `env.producer_turns.default_fund_policy` and
   `default_accept_policy` emit:
   - the ADR-0025 funding ranks;
   - the auction's accept bits (computed on the producer's own state at the turn);
   - priorities that put carried reservations first (price-descending) and new bids price-descending.

   The goldens and an EXP-0088 k5_rf5 replay pin it. Without the flag, the existing code path runs unchanged.
5. **Slots.**
   - Fund slots: 3 investment kinds + the producer's input products, exact per config.
   - Accept turn: K_BID = 64 bids, K_RES = 48 reservations, K_PEND = 48 pending claims.
   - Bids beyond K_BID are declined (ADR-0061 D4).
   - Reservations beyond K_RES keep today's carried-first rule (review of e6cacf0). In default order, they
     rank just below the lowest listed reservation when that one outranks every bid and pending claim (always
     so under the default policy). Otherwise they rank just above the highest bid or pending claim. So the
     default decision stays exact: a reservation that fails at TTL frees its capacity before the new bids
     run, and re-validation still drops the cheapest first.
   - Pending claims beyond K_PEND run after everything else, in default order. The default policy never
     leaves a pending claim.
   - Everything past a slot count is counted in `turn_beyond_k`: bids, reservations, pending claims, and
     funding requests past the fund slots (funded after the listed ones, in today's order).
   - The largest counts seen at 12 producers are 26 bids and 21 reservations per producer-cycle (EXP-0115 build
     check).
6. **Information set (ADR-0058).** A turn's observation and the GNN's turn features read only the producer's own
   state:
   - its routed bids, with each bid's own-order remaining lifetime (ADR-0061 addendum) and its parent
     order's kind (consumer / replenishment / investment, carried on `OrderInfo` → `SubOrder` → `Bid` the same
     way; EXP-0115 / DIAGNOSTICS 36: whether accepting for later delays someone else's input);
   - its reservations, pending claims, capacity, inventory and inbound;
   - its own funding requests, and its budget, the stage-1 result.

   **The one cross-producer signal: the fund-turn budget.** Stage 1 splits the fund across the producers that
   requested funding this cycle, by their cut-offs, so a producer's budget depends on whether others requested.
   It is allowed:
   - It is a physics output and the producer's own allocation.
   - It is fixed before any fund turn, because every investment action precedes the fund turns.
   - Other producers' fund scores cannot move it (stage 2 runs at `start_ordering`). A leak test pins this.

   Otherwise turns never read another producer's bids or decisions of the same cycle. Each producer's stage 2 and
   production are independent of the others', so AEC order cannot matter. Leak tests pin it.

## Considered options
- **Fold the spend order into the investment turn** (more action dims on `prod_<pid>`). Rejected: it changes every
  existing producer action space and checkpoint, and the requests and their costs are not concrete yet.
- **Keep scheduling step-ordered** (carried first, pending first) and learn only priorities within each step.
  Rejected: the step order itself is one of the audit's hand-written rules. EXP-0115's A1-new-first arm measures
  what the order costs.
- **A domain preview of the acceptance view.** Kept implicit: no producer state changes between the turn and
  `produce()`, so the turn reads live state.

## Consequences
- **Learned acceptance, funding order and scheduling become trainable** with the existing RL loop (tasks 4.3–4.5).
- **New code paths** in the domain (scored stage 2, scheduled production) are entered only through the turns.
  The default path (no flag) is untouched.
- **The fund-gaming risk of ADR-0061 stands** (the cut-off is the lowest accepted price), and it is reported.

## Addendum (2026-11-07): the accept turn's network is order-aware
Task 4.3 (EXP-0118 → EXP-0126; DIAGNOSTICS 38; design `notes/4.3-ordered.md`).

**Why it is needed.** A per-item head, even with set context, could not imitate the auction's cumulative rule (accept
while the running sum of capacity, in priority order, still fits).
- Its near-threshold errors cost units in closed loop: EXP-0123 lost −56.7 per episode to late pending claims
  (DIAGNOSTICS 36).

**The accept turn's heads now run four stages:**
1. A set encoder (3 self-attention layers) over the turn's items.
2. Bounded priorities, 10·tanh(z/10) with a pre-tanh penalty, so they never reach the action clip.
3. A gated additive recurrent pass over [reservations, pending, bids] in the heads' **own** detached priority order:
   s ← s + σ(a)·U·h. Ties follow input order, which is the domain's kind order.
4. A bid's accept logit read from the pass.

**Inputs:** v14's own-state rows, linear scale-free fractions (capacity / free capacity, capacity / current
capacity, price / the turn's top price) and each item's **arrival position** in the producer's own lists (the order
the domain's stable tie-break uses).

**What is not hand-written.** Nothing cumulative is handed to the heads. The order is the student's own learned
priority, and training and deployment use the same order (DIAGNOSTICS 37).

**Result (EXP-0126, F40 trunk):**
- Fresh held-out accept agreement 0.9983, accept-turn rank 0.9995, fund rank 1.000.
- Closed-loop agreement 0.998, with the pending claims back near zero.
- Accepted by the user as the 4.4 / 4.5 start.

**RL start (4.4 / 4.5, 2026-11-07).** The learned producer trains as one composite model:
- F40's trunk, gates and these turn heads;
- the EXP-0110 start's replenish head on its own frozen trunk, the only replenish weights that train (exactly frozen
  if 4.5 plays replenish at σ_r = 0).

All heads train on-policy, as deployed: the turn heads sample their orders by Plackett–Luce and their accept bits
tempered, with the ordered pass run in the sampled order. Deployment plays every head greedily (`notes/4.4-4.5-draft.md`).

