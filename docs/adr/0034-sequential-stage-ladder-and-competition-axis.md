# Sequential curriculum ladder and a competition axis

Status: accepted

## Context

ADR-0009 made the curriculum *evidence-gated*: difficulty advances only when the
policy proves survival + fulfilment at the current level. The review of 2026-06-15
(B9 step 5) added a second axis — demand variance — but drove **both** axes
(`scale_degradation` and `scale_demand`) off the *same* scalar simultaneously. Two
problems followed:

1. **Non-attributable stalls.** When the gated run plateaued (it stalled at d≈0.6 on
   `complex.json`), nothing isolated whether degradation, quantity variance, or
   product mixture caused it — all three moved together. `scale_demand` itself
   coupled quantity and mixture under one threshold (`MIXTURE_AT`).
2. **No competition axis.** `complex.json` pits a strong producer against a
   near-dead one (`p5` owns capacity-1=1 while its product needs 2 → zero
   throughput). The curriculum eased degradation and demand but never the
   competition, so the policy faced full routing-against-a-dead-competitor from the
   first iteration. Review item #5: "train first with two viable producers differing
   gradually."

## Decision

**Ramp the curriculum axes sequentially, one stage at a time, and add a competition
axis.** The gated scalar `level()` returns (now `p ∈ [0,1]`) is mapped by a new
`staged_config(base_cfg, p)` through ordered sub-intervals — each ramps the next
axis to full while earlier axes stay maxed and later axes stay fully eased:

| stage | p range      | axis ramped                          |
|-------|--------------|--------------------------------------|
| 1     | p = 0        | (baseline) fixed quantity 1, single product, viable producers, no decay |
| 2     | [0.00, 0.25] | variable quantity (`scale_demand` qty) |
| 3     | p ≥ 0.25     | product mixture on (`scale_demand` pool) |
| 4     | [0.50, 0.75] | competition (`scale_competition`)      |
| 5     | [0.75, 1.00] | full degradation (`scale_degradation`) |

`p = 1` reproduces `base_cfg` (the real task). A gated run starts at `d_start = 0.0`
(stage 1) — the new default.

- **`scale_competition(cfg, d)`** raises every producer's owned capacity that is
  *below* its declared products' recipe requirement up to that requirement at `d=0`
  (made viable), interpolating back to the real value at `d=1`. Only deficient
  capacity *values* change — never the *types* — so actor/critic shapes stay valid,
  exactly like `scale_degradation`/`scale_demand`.
- **`scale_demand(cfg, qty_d, mixture_on=None)`** now takes quantity progress and an
  explicit mixture flag so the two ramp in *separate* stages. `mixture_on=None`
  preserves the legacy single-scalar coupling (`qty_d ≥ MIXTURE_AT`) for callers
  that still pass one argument.

The evidence-gate mechanism (accumulate ≥ `min_episodes`, require survival AND
fulfilment, reset the window per decision) is unchanged — only the scalar's
*meaning* (ladder progress, not degradation fraction) and its *mapping* changed.

## Considered options

- **Keep one scalar, just decouple the gate constants.** Rejected: simultaneous
  ramps remain non-attributable; the review explicitly asked for sequential staging.
- **Ease competition by removing the weak producer early.** Rejected: that changes
  the producer set (and thus obs/action dims and eligibility), breaking shape
  invariants. Boosting capacity keeps the topology fixed and anneals smoothly.
- **Per-axis independent gates.** Deferred: a single monotone progress with ordered
  stages is simpler and already attributes a stall to the stage it halts in; per-axis
  gates can come later if a stage needs independent pacing.

## Consequences

- A stall now localizes to a stage (e.g. "halts entering stage 4" = competition is
  the wall), which the prior coupled scalar could not show.
- `d_start` default changes `0.2 → 0.0`; the scalar is ladder progress, not a
  degradation fraction. Non-gated iter-schedule runs now start fully eased and ramp
  up — re-read any plan numbers that assumed the old `d_start=0.2` start.
- No checkpoint-shape change (config-mapping only); `CHECKPOINT_VERSION` is
  unaffected by this ADR.
- `scale_demand`/`scale_competition`/`scale_degradation` all preserve dims, so the
  actor/critic built on `base_cfg` loads every staged config.
