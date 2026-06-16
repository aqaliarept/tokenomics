# Combined held-out suite + fixed versioned seed manifest

Status: accepted

## Context

NEXT STEPS stage 4 ("combined held-out suite") asks to **combine** randomized demand,
producer competition, capacities, costs, and degradation — the opposite of stage 3,
which changed one axis at a time — under **fixed versioned seed manifests**, and to
require that the trained policy **reliably outperforms the strongest
observation-compatible baseline** (ADR-0036).

Stage 3 gave single-axis, dims-preserving perturbations (`eval/generalization.py`).
Stage 4 needs three things on top: (1) a way to *stack* those axes, (2) a way to *pin
and version* the exact held-out workload so it is reproducible and reviewable, and (3)
a pass/fail bar that is testable without training.

## Decision

- **`generalization.combined_config(cfg, *, degradation_factor, quantity_max,
  capacity_factor, cost_factor)`** — composes the stage-3 `perturb_*` functions;
  neutral defaults (factor 1.0, `quantity_max=None`) leave an axis untouched, so the
  all-default call is the identity and a call names exactly the axes it stacks. A new
  **`perturb_costs`** axis scales every *set* investment cost (restoration / upgrade /
  inventory expansion / per-unit replenishment); a `None` (unbounded) cost stays
  `None`. Every component perturbation is dims-preserving, so the combination is too.

- **`eval/held_out.py::SeedManifest`** — a frozen dataclass (`version`, `base`,
  `seeds`, the four axis factors) that serializes to JSON (`to_json`/`from_json`,
  exact round-trip) so the held-out workload is a checked-in artifact.
  `manifest.config()` loads the base and applies `combined_config`. `load_manifest`
  resolves a relative `base` against the manifest's own directory, so a checked-in
  (manifest, base) pair is portable. The shipped suite is
  `eval/manifests/heldout_v1.json` + `heldout_v1_base.json` (a multi-declarer
  competition base, so producer competition + costs + inventory are exercised), with
  held-out seeds in the `HELD_OUT_SEED_OFFSET` (10000+) range, disjoint from
  training/validation (ADR-0011).

- **`beats_baseline(trained, baseline)` → `SuiteVerdict`** — the stage-4 bar as a
  pure function of two `Aggregated` metric dicts: all of (death no worse, fulfilled
  units no fewer, return strictly higher) must hold; survival is non-negotiable, so a
  return/units win cannot buy back extra death. Returns one reason per failed
  condition. `run_manifest` evaluates a policy pair on the manifest config over its
  fixed seeds — the same harness for the trained policy and the baseline, scored
  head-to-head on the identical workload.

## Considered options

- **Put combined perturbation in `held_out.py` directly.** Rejected: the axis
  transforms belong with the other `perturb_*` in `generalization.py` (one family,
  one place); `held_out.py` owns only the versioning + verdict.
- **Store the manifest's base path as-is (cwd-relative).** Rejected: a checked-in
  manifest must load from any cwd. Resolving against the manifest directory keeps the
  pair portable without absolute paths in the repo.
- **Include `degradation` at the stage-3 2× factor.** Used a *mild* 1.2× instead:
  stage 3 already proved 2× is a catastrophic, separately-tracked failure; a mild
  factor keeps the combined suite from being trivially dominated by that one known
  issue while still exercising the axis.

## Consequences

- The held-out evaluation is now reproducible from a single versioned file; stage 6
  (stress) reuses `SeedManifest` with harsher factors / more seeds.
- `beats_baseline` makes "reliably beats the strongest obs-compatible baseline" a
  concrete, unit-tested gate rather than a judgement call.

### Result (heldout_v1, trained seed 0 vs strongest obs-compatible baseline, 10 held-out seeds, horizon 100)

| scenario             | who      | dead% | units✓ | return |
|----------------------|----------|------:|-------:|-------:|
| unperturbed base     | trained  |     0 |   99.0 | 1224.6 |
| unperturbed base     | obs_base |     0 |   99.0 |  852.1 |
| full heldout_v1      | trained  | **100** | 16.2 |   59.6 |
| full heldout_v1      | obs_base |     0 |  100.3 |  374.6 |

**VERDICT: FAIL** — `beats_baseline` reports all three: higher death (1.00 > 0.00),
fewer units (16 < 100), lower return (60 < 375).

The control comparison is the important part. On the **unperturbed** competition base
over the same held-out seeds, the trained policy is strong and *beats* the baseline
(1225 vs 852 return, equal units) — so it learned the multi-hop topology; this is a
**generalization** failure, not a failure to learn.

Localizing it (reuse the checkpoint, `artifacts/heldout_ablate.py` /
`heldout_isolate.py`):

| perturbation (vs base)      | trained dead% | trained units | obs_base dead% | obs_base units |
|-----------------------------|--------------:|--------------:|---------------:|---------------:|
| none                        |             0 |          99.0 |              0 |           99.0 |
| qty ≤ 3 only                |       **100** |          21.4 |              0 |          105.9 |
| capacity ×0.8 only          |       **100** |          17.0 |              0 |           99.0 |
| cost ×1.3 only              |       **100** |          33.0 |              0 |           99.0 |
| combined, no degradation    |       **100** |          18.4 |              0 |          100.7 |
| full (adds degradation 1.2×)|       **100** |          16.2 |              0 |          100.3 |

**Every single axis — demand, capacity, OR cost — individually collapses the trained
policy (100% death) on this base, while the baseline shrugs all of them off.** This is
*broader* than stage 3, where single axes on the loose phase1 base were survivable: on
the tight multi-hop competition base (e.g. p4 owns 120 capacity-4 at 25/unit ≈ 4
units of headroom), any distribution shift pushes the chain past the policy's
open-loop maintenance cadence. Degradation 1.2× adds nothing — the collapse is already
total without it.

**Conclusion:** the policy memorized the exact training operating point. The
closed-loop observation-compatible baseline (restore/replenish-when-low) tracks any
shift; the open-loop trained policy does not (memory: greedy mean is open-loop). Fix is
the same family flagged in stage 2/3 but now clearly load-bearing: **domain-randomized
training** — train across *ranges* of demand / capacity / cost / degradation (and on
the competition topology), not a single fixed config — so the policy must learn
shift-tracking control. Recorded for the curriculum follow-up; stage 4 infra (manifest
+ verdict) now makes the bar a reproducible gate.
