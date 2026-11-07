# Audit: hand-written decision rules on the M6 trained stack's path (2026-11-07)

User request (2026-11-07): "Revalidate orderer and producers for such hardcoded parts". Read-only code audit;
`file:line` references are as of commit `89d9381`.

**Setup.** M6 rollouts build the env with `joint_stack_env` (`eval/joint_stack.py:48-60`). It uses the default
`ordering_decode=logits-v1` and leaves `acceptance_rule=None` (`env/aec_env.py:163,251`), so the default
auction always runs. Every world builder gives each producer one capacity type (`eval/families.py:404-407`,
`eval/scale.py:164-167`), so "which type to restore / upgrade" is not a live choice. No planner code runs at
execution time; `recent_input_use` / `blocked_shortfall` appear only as GNN input features.

## Hardcoded agent decisions, ranked by likely effect on consumer units
1. **Acceptance:** the default price-greedy auction, including "accept a composite without inputs and reserve
   it" (`domain/producing_agent.py:631-653`). Known; tasks.md §4.
2. **Fund stage 2, the spend order inside a producer:** restoration → replenishment → upgrade → expansion, each
   filled to its target, with the boundary order partly funded (`domain/system.py:221-225, 290-300`; ADR-0025 /
   0027). Among the replenishment orders the lowest input product id wins (`system.py:292`, stable sort). A
   restoration can leave the input purchases unfunded; they expire that cycle (`system.py:557-562`).
3. **No agent sets a bid price:**
   - Orderers must allocate all of an order's available tokens (`domain/ordering_agent.py:124-130`,
     `allocator.py:63-64`).
   - Producer requests bid their config cost (`producing_agent.py:341-366`).

   With price-ordered acceptance, config costs alone set priority between orders upstream. The inference that
   investment orders always outbid input purchases is not measured.
4. **Reservation scheduling:**
   - Carried reservations take stock first (`producing_agent.py:604-608`).
   - The re-check after degradation drops the cheapest first (`:549-564, 591-594`).
   - A reservation waits at most the order TTL (`:610-619`).
5. **Whole-order routing with no "wait":** an order goes whole to one producer (ADR-0059,
   `eval/rotation_routers.py:73`). The unaccepted remainder must be re-routed every cycle with all its tokens
   (`order.py:211-216`, `ordering_agent.py:64-69`).
6. **Pilot freezes (EXP-0113 / 0114 only):** input buying uses a frozen imitation copy, and the orderer is frozen
   at EXP-0100 (`train/m6_pilot.py:98-123`).
7. **Fund stage 1 quirk** (physics): a producer whose cut-off is still NaN gets only the rounding leftover once
   any other producer is priced (`system.py:276-283`).

## Legality / bookkeeping (fine as fixed)
- Replenish quantity is `round(frac × max_replenish)`, with a proportional scale-down when headroom binds
  (`env/decode.py:118-126`). The policy can't choose which input loses under the headroom limit.
- A toggle is ignored while that investment is running or not configured (`decode.py:131-135`). There is no
  fund check in decode.
- K slots are filled in ascending string order id (`rotation_routers.py:47-52`). The even-split fallback beyond K
  (`decode.py:60`) is practically unreachable, because K = TTL + 1 + 3.
- Eligibility comes from `produces`. Single-eligible slots are forced.
- Every stream, including replenishment and investment supplier choice, is routed by the GNN orderer. No default
  router exists.

## Physics (stays fixed)
- Degradation `ceil(rate × max / 100)`.
- The 50/50 emission split; consumers spend their whole balance; refunds.
- Fund stage 1 is proportional to the cut-off.
- The order and reservation TTL.
- An order completes whole or not at all.
- Restoration / upgrade effects.
