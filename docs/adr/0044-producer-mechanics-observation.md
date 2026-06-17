# Producer-mechanics block in the producing observation (scale generalization)

Status: accepted

## Context

The scale work (ADR-0043) produced a policy that **beats the baseline on the fixed big
environment** (227 vs 163 units, 0% death) but **fails to generalize** to held-out scale
seeds (114 units / 33% death vs the obs baseline's 253 / 0%). Domain-randomizing the
topology made held-out *worse*, not better.

Diagnosis (plan, "Recheck / diagnosis"): the **producing** policy is what does not
transfer. Swapping in the obs-baseline producer lifts held-out from 115 to 337 units; the
trained producer lets level-1 producers run a capacity type to zero by cycle ~20.

`build_scale_config(seed)` randomizes, per seed, the producer mechanics that set the value
of every action a producing agent takes:

- per-type **degradation** (rate 3–7, and absolute units scale with max capacity),
- **restoration** amount (40–70) and **cost** (20–50),
- **upgrade** amount (30–50) and **cost** (20–50),
- **max-replenish** per cycle (8–14) and **replenish cost**/unit (2–6),
- **inventory-expansion** increases (15–25) and **cost** (20–40).

None of these were in the producing observation. The obs carried current/max capacity,
inventory, outstanding inbound (ADR-0041), prices, and demand — but not the *action
semantics*. So two scale seeds produced the **same obs vector shape with different optimal
actions**: identical tensor, different decision problem. A neural policy that sees only the
obs cannot adapt its restore/replenish cadence to the seed; it memorizes one operating
point. The obs baseline does not have this handicap — it reads `EnvSpec` (`max_replenish`)
directly for its closed-loop rule.

## Decision

Append a per-agent **producer-mechanics block** to the producing observation, sited as the
last block before the trailing `cycle_progress` scalar (so the front offsets the obs
baselines index — caps, inventory, inbound — are untouched). In order:

1. per-type degradation units this cycle (`num_types`, from `view.degradation`, `/cap_ref`),
2. restoration amount (`view.restoration / cap_ref`),
3. upgrade amount (`view.upgrade / cap_ref`),
4. max-replenish (`spec.max_replenish / inv_ref`),
5. replenish cost/unit (`/token_ref`),
6. restoration cost (`/token_ref`),
7. upgrade cost (`/token_ref`),
8. inventory-expansion increases (`/inv_ref`),
9. inventory-expansion cost (`/token_ref`).

`producing_obs_dim` grows by `num_types + 8`. A new `EnvSpec.producing_mechanics_start`
property (and `producing_mechanics_len`) locates the block self-documentingly.

Sourcing keeps the layering rules: degradation/restoration/upgrade *amounts* already live
on `ProducingAgentView` (domain, pure). The static investment economics (costs,
replenish/inventory-expansion knobs) are folded into a new `ProducerMechanics` dataclass on
`EnvSpec`, built from config in `build_spec` — the env adapter, where normalization belongs.
`None` token costs (unbounded funding target) and a producer with no inventory expansion
read 0 — an absent knob. `domain/` is unchanged.

Checkpoint schema bumped `4 → 5`: the obs vector changed, so older checkpoints are rejected
loudly (ADR-0026).

## Considered options

- **Put the mechanics block before the open-demand block.** Rejected: keeping it as the
  tail block before `cycle_progress` makes the offset math (`producing_mechanics_start`)
  trivial and leaves every front offset used by the obs baselines valid; only the one
  open-demand test that indexed from the tail needed a (correct) re-anchor.
- **Expose only degradation + max-replenish (the minimal set the regression names).**
  Rejected: the costs and amounts also vary per seed and set what each investment toggle is
  worth; a policy that cannot see restoration/upgrade economics still cannot price its
  investments across topologies.
- **Feed `EnvSpec` to the policy network.** Rejected: the trained policy's contract is the
  float observation (CTDE, shared decentralized actors). The fix is to put the missing
  decision variables *in the observation*, not to special-case the net.

## Consequences

- Two same-shape scale seeds now expose **different** mechanics in the obs (regression
  `test_two_scale_seeds_expose_different_mechanics_in_obs`): the seed-varying mechanics are
  no longer hidden state. This is the necessary precondition for a producing policy to
  generalize across the scale envelope; whether it *does* clear the held-out bar is a
  training-budget question (ADR-0043), measured separately.
- Full suite green (192 tests). The pettingzoo `api_test`, the obs baselines, and the
  checkpoint-version guard all still pass; the obs baselines are unaffected (front offsets
  unchanged) and gain no information they did not already read from `EnvSpec`.
- Older checkpoints (v4) no longer load — they were trained on the smaller obs and are not
  comparable on the new economy anyway.
