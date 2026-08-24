# EXP-0042 - Bounded Online Rollout

**Date:** 2026-08-24 | **Status:** IMPLEMENTING

## Question

Can confidence-bounded online scenario search improve closed-loop delivery over
`blend_0.5`, and what is the shallowest sufficient rollout depth in `{1,2,3}`?

## Registered read

The search starts from and falls back to `blend_0.5`. It searches only ambiguous
top-two base routes, samples common future-demand scenarios at checkpoints 4/8/16,
and applies a producer bump only after a paired confidence lower bound clears every
alternative. The shallowest depth that clears all value and safety gates is promoted.

Results pending implementation verification and the registered RunPod CPU run.

Pre-run engineering note: s29 seed 3300 was used for a complete depth-1 timing and
policy smoke (753 floor to 821 search units, 64.7 seconds locally). It is excluded
from the primary gate; the untouched registered s29 block is 3301–3310, and the CLI's
bounded smoke uses non-gating seed 3299.
