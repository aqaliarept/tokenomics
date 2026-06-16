# Observation-compatible baselines alongside oracle baselines

Status: accepted

## Context

The Phase 1 NEXT STEPS verification ladder states the release bar four times as
beating *"the strongest **observation-compatible** baseline"* — a policy that acts on
the same observation vector the RL policy sees and nothing more.

Every baseline in `eval/baselines.py` is **white-box (oracle)**: it reads `env.system`
directly (live capacities, recipes, reservations) to make decisions. That is the right
design for an *upper-bound* reference, but it is **not** a fair bar for the RL policy,
which only ever sees the encoded observation vector. Comparing a trained MAPPO policy
against an oracle baseline conflates two questions: "did it learn coordination?" and
"did it overcome the information hidden from it?". Without an observation-compatible
baseline the release bar could not actually be measured.

## Decision

**Add observation-compatible baselines that decide from `env.observe(agent)` + the
static `EnvSpec` alone — never `env.system`.**

The decision logic is a pure function of `(spec, obs)`, so the observation-only
guarantee is *structural* (no `env` handle reaches it, hence no `system` access):

- `decide_producing_obs(spec, obs, pid, *, threshold, stock_target)` — obs-only analog
  of `producing_stock_aware`: restore when an observed capacity ratio (`caps[t]/maxs[t]`,
  read from the producing obs layout) drops below `threshold`, and top each relevant
  input toward `stock_target` from the observed inventory section (denormalized by
  `inv_ref`, mapped back to a raw replenish value like `decode`).
- `decide_ordering_obs(spec, obs)` — obs-only ordering router. The ordering obs hides
  producer capacity (ADR-0029), so the only routing signal is the per-producer cut-off
  price; route toward the lowest (slack capacity).
- Policy wrappers `producing_obs_stock_aware()` / `ordering_obs_price` feed these
  `env.observe(agent)` and register as `obs_stock_aware` / `obs_price` in the baseline
  dicts (picked up by `signal_check`).

The spec supplies only *static* layout + normalization constants (`cap_ref`, `inv_ref`,
`max_replenish`, index maps) — exactly the fixed quantities a trained net learns
implicitly. Using them is not privileged information.

### The undefined-price trap (finding)

A naive `-price` ordering router **collapsed** on `competition.json` (100 % death,
return −489 vs even_split's 229). Root cause: a cut-off price of `0` means *"no
production yet"* (`UNDEFINED_PRICE` → `0.0` in encode), **not** *"cheapest"*. Because
ADR-0029 hides capacity, the unproven producer is frequently the one that *cannot*
produce; routing toward it starved composite assembly. Fix: park price-0 producers at
the logit floor so a *proven* cheap producer always wins; at cold start (all prices 0)
this collapses to an even split. After the fix `obs_price` matches `even_split`
(return 225, 0 % death).

## Considered options

- **Decode the obs inside an `(env, agent)` policy that just avoids touching
  `env.system`.** Rejected: the boundary is then a convention, not enforced. A pure
  `(spec, obs)` decision function makes observation-only a type-level guarantee and is
  directly unit-testable with an encoded obs.
- **Make `obs_price` the strongest observation-compatible ordering baseline.** Rejected:
  with capacity hidden (ADR-0029) price alone is a weak signal; `even_split` is the more
  robust obs-compatible ordering baseline. `obs_price` is kept as a second data point and
  a demonstration that price routing is insufficient under capacity-hiding.
- **Enrich the ordering obs with capacity/throughput so a price router can compete.**
  Out of scope — that would revisit ADR-0029. Recorded as a candidate if the RL policy
  cannot beat `even_split` on competition topologies.

## Consequences

- The fair release bar is **`obs_stock_aware + even_split`**. On `phase1.json` it scores
  return ≈ 1255 / 100 % consumer fulfilment / 0 % death — within ~1 % of the oracle
  `stock_aware` (1267); the gap is the obs blind spot to `outstanding_inbound`. The
  trained policy must clear this number, not (only) the oracle one.
- New baseline category convention: white-box baselines read `env.system`; obs baselines
  read only `env.observe(agent)`. Keep the two clearly separated in `eval/baselines.py`.
- `obs_price`'s competition behavior is a concrete demonstration that the ordering side
  is information-limited by ADR-0029 — useful when interpreting RL ordering performance.
