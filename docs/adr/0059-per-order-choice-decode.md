# Per-order choice decode and validity layer

Status: accepted (add-m4-orderer-redesign, milestone M4; design D4 and D5, task 4.1). Licensed by
EXP-0057's primary reading `decoder_partial`.

## Context

The legacy ordering action (`logits-v1`, ADR-0025 / ADR-0029, `env/decode.py`) is one raw logit
per producer for each order slot. The decode softmaxes the logits over the order's eligible
producers and splits the quantity by the largest-remainder rule, then splits the tokens following
the integer quantities. Every order is therefore split unless one logit dominates. A policy cannot
say "send this whole order to producer p" exactly, only approximately, and the approximation
depends on the order's quantity.

Two M3/M4 diagnostics say that this is the wrong action for the orderer:

- **EXP-0056 reads `joint`** (3 of 3 shocked worlds). The trained orderer's gap to `round_robin`
  closes only when both of its habits change: splitting orders, and its choice of receivers.
  Whole orders sent to varied receivers recover the gap; splitting under the scorer's rule wasted
  capacity on orders that do not complete.
- **EXP-0057 reads `decoder_partial`**. Drawing a whole receiver per order from the *same* trained
  scores (`trained_sampled_whole`, TS) wins back about 70% of the gap to the reference on the
  12-producer gap worlds, the median world 0.70, short of the 2/3-of-worlds bar for `decoder` by
  one world. The scores carry their target (fidelity `faithful`), so the decode is a bottleneck.
  `decoder_partial` licenses step 2: a per-order choice policy trained on outcomes.

Step 2's policy (design D5) chooses one producer per order. It needs an action that expresses a
choice exactly and stays bounded under bursts (orders 5–10× the usual size), and a decode that
never turns an invalid action into a plausible one.

## Decision

1. **An env option `ordering_decode`**: `"logits-v1"` (the legacy decode, the default) or
   `"choice-v1"`. The default is byte-identical to the code before the option existed: every
   golden, frozen hash and recorded reproduction holds.
   - The option is part of the env config fingerprint (`spaces.config_fingerprint`). It is folded
     in **only when it is not `logits-v1`**, as `producer_obs_version` v1 is left out, so every
     existing fingerprint and checkpoint stamp is unchanged.
   - A driver that can run `choice-v1` merges `spaces.decode_pins(ordering_decode)` into its
     `spec.json` pins and resume check. It is empty at the default and adds `ordering_decode`
     otherwise, so a choice run never resumes or reads as a logits run.

2. **The `choice-v1` action is per-slot shares.** Row k of an ordering agent's (k_stream, P)
   action is slot k's share for each producer, every entry in [0, 1] (`ChoiceShareSpace`, a
   `Box`). Shares, not quantities, so the bound is fixed whatever the burst. Slot k is the k-th
   order in ascending order id, as in the legacy decode.

3. **A validity layer, `allocator.validate_choice`, that checks and never repairs.** A choice is
   valid when its shares are finite and ≥ 0, no positive share sits on a producer that does not
   declare the order's product (ADR-0028), and |Σ − 1| ≤ 1e-6. Quantities come from the legacy
   `_allocate_int` (largest remainder) of the shares × q, and tokens follow the integer quantities.
   - A whole choice, the one-hot share 1.0 on p, is exact: one sub-order of the whole quantity
     with every token.
   - A slot without an order must be all zero, so a policy whose slots are misaligned with its
     orders fails loudly. An order without an eligible producer is valid only as a zero row and is
     not routed, as in the legacy decode.
   - A violation raises `InvalidChoice`. The episode stops, and the gate's contract path records a
     `job` row with check `choice_validity` (`eval/contract.py`), which makes the run invalid. The
     action is never rerouted, clamped or rescaled.
   - Box bounds cannot express "sums to one over the eligible producers", so `ChoiceShareSpace.
     sample(mask)` draws a valid whole choice from a k_stream × P eligibility mask. Under
     `choice-v1` the env publishes that mask as `infos[agent]["action_mask"]` (the PettingZoo
     convention) for the ordering agent about to act. Eligibility is the static `produces` set,
     inside the orderer's information set (ADR-0058).

4. **Overflow orders stay legacy.** An order beyond `k_stream` (ADR-0054, the corrected
   product-stream backlog bound) has no slot and keeps the legacy even split over its eligible
   producers, under both decodes. The routing tracer counts these
   (`overflow_orders`). Giving them a choice is M7 work.

5. **M4's learned policies emit whole (one-hot) shares only** (design D5). The validity layer
   accepts any valid share vector, so a split is representable, but no step-2 or step-3 policy
   emits one, and a test asserts it ("No split emitted", task 4.3). Splitting is out of scope for
   step 2. It may return only through a separate change backed by a diagnosis that a split
   recovers units that whole routing cannot.

No `domain/` change: the decode, the validity layer and the action mask are env-layer code.

## Consequences

- Step-2 policies (`train/per_order_scorer.py`, task 4.3) act under `ordering_decode="choice-v1"`:
  one row per order slot, a single 1.0 on the drawn producer. A composite per-order checkpoint
  carries its own decoder version (`per-order-choice-v1`) and is refused in a `logits-v1` env
  (task 4.3).
- Every existing checkpoint, baseline and router keeps the legacy decode. The ordering baselines
  emit logits and are invalid under `choice-v1`; a `choice-v1` run plays only policies written
  for it.
- A bug in a choice policy becomes a recorded contract failure, not a quietly different routing.
  The cost is that one bad action ends the job; the run is then invalid and must be fixed and
  rerun, which is the intended behaviour for a broken run.
- The fingerprint and pin are omitted at the default. A reader of a pin set or fingerprint
  without `ordering_decode` reads it as `logits-v1`.

## Alternatives rejected

- **Quantities instead of shares.** The bound would scale with the order size, and a burst could
  push a valid action outside a fixed Box.
- **A discrete producer index per slot.** Exact for whole orders, but it cannot express the
  validity layer's general share vector, and it would need a different space type per world size.
- **Repairing invalid choices** (masking ineligible shares and renormalising, as the legacy
  softmax does implicitly). It hides policy bugs and makes an invalid action indistinguishable
  from a valid one in the results.
- **Folding `ordering_decode` into the fingerprint at its default too.** It would change every
  recorded fingerprint and checkpoint stamp for no information.
