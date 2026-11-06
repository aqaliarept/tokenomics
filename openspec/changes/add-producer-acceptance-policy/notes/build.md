# Build notes (2026-11-06)

- **D1 (`domain/producing_agent.py`):** `ProducingAgent.acceptance` (None = the ADR-0025 auction code, untouched)
  is called at the auction's exact point in `produce()`, after carried reservations are re-validated and aged.
  So it sees the same budget the auction did. It receives an `AcceptanceView` with the producer's own state only.
  - An accepted bid that does not fit becomes a **pending claim** (`ProducingAgent.pending`): it holds no
    capacity, takes capacity in later cycles by priority, then price, and fails at TTL.
  - Tokens burn at acceptance; the cut-off is the lowest accepted price, carried when nothing is accepted.
  - A declined bid is `rejected` with the policy's reason (`declined`, or today's no-fit / displaced).
  - Pending claims of a removed order fail with it (`domain/system.py`).
- **D2 (`domain/system.py`, `orderer_view.py`):** a `SubOrderOutcome` (accepted / declined / completed / failed,
  with reason) per sub-order per cycle, keyed by its routing agent. It is exposed as additive `last_outcomes` on
  `OrdererEconomyView` and `ProductStreamView`; no encoder reads it. An `outage` reason is withheld (None), since
  it is hidden shock state (ADR-0055).
- **D3 (`eval/acceptance.py`):** `auction` is today's rule as a policy, plus `ttl_feasible` (A1) and `ttl_inputs`
  (A2). A1 / A2 assume one future cycle, because the producer cannot see the order's age.
- **D4 deviation, flagged.** The acceptance phase is an env-level hook (`TokenomicsAECEnv.acceptance_rule`, applied
  to every producer at reset), called by the domain at the decision point. It is **not** an extra AEC agent turn
  with a K-slot action.
  - Why: no learned acceptance is in scope, and an AEC turn would change agent order, spaces and every policy's
    signature. This keeps them unchanged.
  - The K-slot action belongs with the learned-acceptance change.
- **Traces:** an additive `declined` reject reason (`routing_trace`), loss class `e_declined` and an
  `acceptance_stats` column (`supply_trace`, "" without a policy); `supply_trace --acceptance NAME` (sweep only,
  an EXTRA_PRODUCING producer).
