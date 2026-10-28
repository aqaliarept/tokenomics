# Design

## Context

See `proposal.md` (Why). Requirements: the `stochastic-environment` and `gate-harness`
deltas. Roadmap source: `establish-project-baseline` design D1 (M2 row) and D3. Current
state that shapes the approach (paths under `tokenomics/tokenomics/`):

- **Demand RNG is one sequential stream.** `env/aec_env.py:182` seeds
  `np.random.default_rng(seed)` and draws `horizon` B9 entries per randomized consumer, in
  consumer order. Any new draw on that stream would shift every later consumer's demand.
- **Cycle flow.** `System.__init__` ends with `start_cycle()` (`domain/system.py:307`), so
  cycle 1 is ready at reset. `complete_cycle()` (`:438`) runs every producer's `produce()`,
  resolves orders, then calls `start_cycle()` (`:533`). That call emits, places the next
  cycle's orders and advances `cycle_counter`, even after the last played cycle.
  `start_cycle` moves no capacity today: degradation is step 5 of `produce()`
  (`domain/producing_agent.py:469-471`).
- **In-flight production = reservations.** A composite bid accepted without its inputs
  becomes a `_Reservation`. It assembles when its inputs arrive, or fails after
  `RESERVATION_TTL = 2` cycles (`domain/producing_agent.py:31`, `:427-444`). Primal and
  stocked-composite bids complete when accepted.
- **Where `r` is read.** The env reads the capacity ratio for `r_end` / `min_capacity_ratio`
  after `complete_cycle()` (`env/aec_env.py:394`), so after the trailing `start_cycle`. A
  shock applied in `start_cycle` would leak an unplayed cycle into `r_end`.
- **Observed consequences exist.** The product-stream orderer view carries per-producer
  rejected-or-failed EMAs (`domain/orderer_view.py:96`, `:256`). Rejections caused by an
  outage therefore show up after the fact, without any new observation field.
- **Fingerprint hashes `asdict(config)`** (`env/spaces.py:318-327`, ADR-0035). A new
  `Configuration` field would change every existing fingerprint unless it is left out when
  unset.
- **Gate "shocked" today = B9 only** (`eval/families.py:49-53`). The frozen packaged files
  are `eval/registry/{worlds,roster,side_effect_params}_v1.json`, each versioned by a content
  hash (EXP-0044 attempt 2). EXP-0045 is the M1 reading of the portable stack against them.
- **ADRs.** `docs/adr/` was deleted from this branch in commit `18c717d` ("cleanup"), but
  CLAUDE.md and ARCHITECTURE.md still point to it. The predecessors cited below can be read
  with `git show 18c717d^:docs/adr/<file>`: ADR-0025 (reservations, TTL), ADR-0029 (the
  producer owns the recipe and the auction), ADR-0030 (death detection), ADR-0035 (config
  fingerprint), ADR-0038 (dims-preserving perturbations), ADR-0052 (product streams).
- **Local box**: 2 cores, about 2 GB RAM free, shared. Every EXP run is a pod run.

## Goals / Non-Goals

**Goals:**
- Capacity hits, producer outages and demand drift to the `stochastic-environment`
  contract, with every scenario a pytest.
- Shocked gate worlds that really carry shocks. The v1 files stay frozen, and v2 is frozen
  by a baseline-only screen.
- One small, decision-gating exit run: the champion runs shocked, unretrained, with zero
  contract failures.

**Non-Goals:**
- Supply-link cuts, demand surges and fund shocks (M6). Their schema exists, and validation
  rejects them.
- Any training, candidate selection or margin claim. The exit run is a contract check, not
  an acceptance read.
- New observation features (for example an "in outage" flag). Shocks are seen only through
  their consequences.
- Outages that are partial, or that block a producer's purchasing.
- The champion development report on registry v2. M3 starts with it (Workflow follow-up).

## Decisions

### D1. Shock model is pure configuration data
`domain/shocks.py` defines `ShockKind` (all five kinds), `ShockTarget`, `ShockEvent(kind,
target, start_cycle, duration, magnitude)` and `ShockGenerator`. `Configuration` gains
`shocks: ShockModel | None = None`, where `ShockModel` holds explicit `events` and at most
one `generators` entry per kind. `load_config` reads `"shocks"`. `eval/scale.config_to_dict`
writes it back, and only when it is set. Validation (`Configuration.validate`) applies the
kind table below. A kind outside `IMPLEMENTED_SHOCK_KINDS = {capacity_hit,
producer_outage}`, in an event or in a generator, fails with an error naming the kind.

| Kind | Target | Duration | Magnitude | M2 |
|---|---|---|---|---|
| capacity_hit | producer + ≥ 1 capacity types it owns | 1 | m in (0, 1) | built |
| producer_outage | producer | ≥ 1 | 1.0 | built |
| link_cut | producer + a product it declares | ≥ 1 | 1.0 | rejected (M6) |
| demand_surge | consumer | ≥ 1 | factor > 1 | rejected (M6) |
| fund_shock | the investment fund | ≥ 1 | m in (0, 1] | rejected (M6) |

Every event has `start_cycle ≥ 2`, so the state observed right after reset (where `r_start`
is read) is never shocked. *Alternative:* pass the schedule to the env as a keyword argument,
outside the config. Rejected: a world must be described fully by its config, so that registry
replay, the task archive and the fingerprint all cover it.

### D2. Domain mechanics replay an injected schedule
`System(..., shock_schedule=())` takes a tuple of `ShockEvent`, checks it against the config
and never mutates it. The domain draws nothing.
- **Capacity hit.** The hit is applied in `start_cycle` right after `cycle_counter` advances
  to its start cycle t: before cycle t's observations and auction, after cycle t−1's
  degradation. The removal is `floor(m·c)`, computed exactly with `Fraction(str(m))`, so a
  float such as 0.29·100 cannot round down wrongly. With m < 1 a positive capacity never
  reaches 0, so a hit cannot by itself trigger system death (ADR-0030). Max capacity is
  unchanged, and restoration works as before.
- **Producer outage.** In `complete_cycle`, a producer is out when `start ≤ cycle_counter ≤
  start + duration − 1`. `produce(outage=True)`:
  - re-validates reservations as usual;
  - assembles no reservation: a survivor ages (`cycles_waited += 1`) and fails at its TTL as
    if its inputs were missing;
  - rejects every new bid, while still recording requested capacity;
  - degrades capacity as usual;
  - leaves the cut-off price unchanged and its funds at 0.

  The producer's own investment and replenishment orders go on, so deliveries still reach
  its inventory. *Alternative:* model an outage as capacity set to 0. Rejected: that would
  trigger death detection, and it would destroy the capacity state the spec says keeps
  degrading.
- **Pinned end-of-cycle capacity.** Before its trailing `start_cycle`, `complete_cycle`
  snapshots every producer's (current, max) capacity into `CycleResult.end_capacities`. The
  env computes `r` from that snapshot, so a hit belonging to an unplayed cycle never reaches
  `r_end`. Without shocks the snapshot equals today's read, because `start_cycle` moves no
  capacity. `env.capacity_trace` (behind the runner's `cap_final` / `cap_min`) reads the same
  snapshot, so no episode statistic sees an unplayed cycle's shock. `is_dead()` still reads
  live state, which is safe because a hit cannot empty a capacity.

### D3. Schedules are generated in the env, one derived stream per kind
`env/shock_schedule.py`:
- `derived_rng(seed, stream)` returns `np.random.default_rng(int.from_bytes(sha256(f"tokenomics|{stream}|{seed}").digest()[:8], "big"))`.
  The streams are `shock:capacity_hit`, `shock:producer_outage` and `drift`; M6 kinds will
  add their own. The legacy B9 stream `default_rng(seed)` is untouched, so demand draws are
  preserved, and a new kind cannot shift an existing kind.
- A generator draws `count` events. For each event the draws come in a fixed order: start
  uniform on its window, duration uniform on its range, magnitude uniform on its range
  (rounded to 4 decimals), target producer uniform over its target list (default: every
  producer, in config order), and for a capacity hit one owned capacity type uniform over
  the sorted owned types. Events are sorted by (start, kind, target).
- The generator never sees the horizon: windows live in the config. The env merges the
  explicit events with the generated ones and then drops events whose start is after the
  horizon. The schedule of a shorter horizon is therefore a prefix of a longer one.
- `reset(seed=None)` on a config with a generator or drift raises: an unseeded episode would
  not be reproducible. Legacy configs keep today's `seed=None` behaviour.
- The schedule is fixed at reset, before any action, so it cannot depend on actions. The
  env stat `shock_digest` (sha256 of the canonical JSON of the injected schedule; the empty
  schedule has its own digest) and `shock_events` (the event count) pin it. Nothing in
  `env/encode.py` or the views reads the schedule, which is what the unannounced test checks.

### D4. Drift is a world-level, budgeted redraw
`Configuration.demand_drift: DemandDrift | None` with `j` in [0, 1]. It applies to every
B9-randomized consumer (`is_randomized`, no explicit schedule). N is the number of those
consumers, each placing one order per cycle. Fixed-product and scheduled consumers neither
drift nor count toward N.

Generation, on the `drift` stream, runs cycle by cycle, so it is prefix-stable:
- In cycle 1, each drifting consumer draws an entry with `draw_schedule_entry`, which is the
  B9 distribution.
- At each transition, k = floor(j·N) (computed exactly with `Fraction(str(j))`). k distinct
  consumers are chosen uniformly without replacement. Each redraws uniformly from its pool ×
  quantity range minus its current entry, and keeps its entry if that set is empty.

At most k orders change per transition, and a product switch counts as one change. The
stationary marginal is uniform (redraw-from-the-others on a finite set), the same as B9's,
so the M1 family ratio test is expected to hold; task 5.1 reruns it. Drift stays inside each
consumer's pool, so the product-stream agent set (ADR-0052) is unchanged.

*Alternative:* a per-consumer switching probability. Rejected: it cannot bound the number of
changes per transition, and the spec requires the floor(j·N) cap. Note that at 12 producers
N = 3, so j < 1/3 freezes demand. The profile therefore uses j = 0.34 (D7).

### D5. Legacy invariance and the fingerprint
- With `shocks` and `demand_drift` unset, every code path is the one at base commit
  `c9c7da2` (the fingerprint, restored to pre-M1 values, is the one deliberate exception). `test_legacy_invariance` goldens, the `build_scale_config` output and the v1
  registry replay must not move.
- `_canonical_economy` leaves out `shocks` and `demand_drift` when they are None, so every
  existing config keeps its fingerprint (ADR-0035).
- It also leaves out an empty consumer `schedule`. M1 task 2.2 (`55539bf`) had silently moved
  every fingerprint by hashing that field's empty default: the EXP-0006B champion pins
  `548120bd72921234`, which recomputed as `217bab1c778f7b07` at `c9c7da2`. Omitting it
  restores the pre-M1 values (supervisor decision, task 1.1). A shocked config gets a different
  fingerprint by design. The resulting "fingerprint mismatch" warning on a shocked world is
  expected and is not a contract failure.
- Observation and action spaces are built from topology only (ADR-0038: perturbations
  preserve dims). Shocks and drift add no feature, so checkpoints load unchanged.

### D6. Shock-free twin and the resilience rows
- The **shock-free twin** of a shocked episode is the same world config with `shocks=None`
  (drift kept), run with the same episode seed. Its demand is identical because the streams
  are independent. The harness asserts this through equal demand digests.
- This is how D3's "matched calm episode" is read, following the spec scenario ("the same
  world and episode seed run without shocks"). A calm-regime world is not used: its cyclic
  demand differs, so the dip would mix demand with shock effects.
- The env adds `consumer_fulfilled_units_trace`: the consumer units completed in each played
  cycle.
- `eval/resilience.py` (pure). For a shock with start t and last active cycle e = t + d − 1,
  window w and tolerance τ (both from the shock profile, D7):
  - recovery cycle r = the first c in [e + 1, H − w + 1] whose window sum
    Σ_{c..c+w−1} shocked ≥ (1 − τ)·Σ twin over the same window;
  - `recovery_cycles` = r − (e + 1);
  - `dip_units` = Σ_{c=t}^{r−1} (twin − shocked), signed;
  - `dip_fraction` = dip_units over the twin's sum on the same span (None when that is 0).
  - If there is no such r (or the shocked episode died), the row is unrecovered:
    `recovery_cycles` is None and the dip is summed to the last played cycle.
  - `overlapped` is set when another shock's active span meets [t, r).
- Twins run only for the gate subject, not for roster members. Resilience is a robustness
  report of the candidate, not an acceptance clause or a side effect (side-effect-guards
  "Side-effect definition").
- *Alternative:* one twin per shock, removing only that shock. Rejected for its cost (one
  extra episode per shock); the overlap flag marks the confounded rows instead.

### D7. Shock profile "shocks-i" (pre-registered)
`eval/registry/shock_profiles_v1.json`, versioned by content hash, holds the named profile
that registry v2 references. Counts scale with the producer count P, as `round_half_up(P ×
rate)`, so the shock pressure per producer is the same at both gate sizes:

| Field | Value | At 12 | At 42 |
|---|---|---|---|
| capacity_hit rate | 1/6 per producer per episode | 2 | 7 |
| capacity_hit window, duration, magnitude | cycles 21–90, 1, U[0.3, 0.6] | | |
| producer_outage rate | 1/12 per producer per episode | 1 | 4 |
| producer_outage window, duration, magnitude | cycles 21–90, U{3..6}, 1.0 | | |
| drift j | 0.34 | k = 1 of N = 3 | k = 3 of N = 11 |
| resilience window w, tolerance τ | 5 cycles, 0.05 | | |

The window starts after the warm-up, where most of the best-baseline shortfall sits
(DIAGNOSTICS 18: cycles 1–20). It ends at 90, leaving at least 10 cycles at horizon 100 to
see recovery. `build_family_config(..., shock_profile=None)` resolves the profile into the
world config: the drift, plus generators with the resolved counts and every producer as a
target. With None, the output is byte-identical to v1. The values may change only before the
v2 freeze; after it, a change is a new profile version.

**Smoke profile.** At horizon ≤ 12 the "shocks-i" window (cycles 21–90) produces an empty
schedule, so local tests and the timing probe would exercise no shock at all. They use
`shocks-i-smoke` instead. It is a test-only profile kept in the test fixtures and in the
driver's `--dry-run` path, never in `shock_profiles_v1.json`. It has the same kinds, rates,
durations and magnitudes, with the start window moved to cycles 2–8, so every resolved event
fires inside horizon 12 (an outage of up to 6 cycles starting at 8 still overlaps cycles
8–12). Every smoke test asserts that the injected schedule is non-empty, that it holds both
kinds, and that the first onset is ≤ the horizon. The drivers refuse `shocks-i-smoke` without
`--dry-run`, and every recorded run (EXP-0046, EXP-0047) pins the "shocks-i" profile hash in
its `spec.json`. The probe therefore times the shock mechanics without touching the
pre-registered real-run profile.

### D8. Registry v2, roster v2, parameters v2
- **Registry v2.** `build_worlds_v2()` starts from the v1 draft generator (`build_worlds_v1()`,
  unscreened): same ids, world seeds, episode seeds and reserve order. Calm entries are copied
  unchanged. Shocked entries (gate and reserves) get `"shock_profile": "shocks-i"` in their
  generator params, and the profile's content hash is recorded in the file. The output is
  `eval/registry/worlds_v2.json`, a draft until it is screened.
- **Registry validation adds** (a) calm entries must configure no shocks and no drift, and
  (b) v2 shocked gate entries must name a profile whose hash matches.
- **Roster v2 draft.** `PRODUCING_BASELINES` × (`ORDERING_BASELINES` + `blend_0.5` +
  `shock_reactive`), R = 5, unfrozen. The pruning rule is unchanged.
- **The freeze.** One baseline-only `--mode screen` on registry v2 development (EXP-0047)
  does zero-baseline replacement, the headroom check, pruning and the q = 0.05 X/Y
  calibration. It writes frozen `worlds_v2.json`, `roster_v2.json` and
  `side_effect_params_v2.json`. Each of the three frozen v2 files, the frozen registry
  included, records `screened_on` = the hash of the registry v2 draft that the screen read.
  The frozen registry's own content hash differs from that draft hash, because screening
  writes its record into the file. `screened_on` is therefore the shared key, not the frozen
  registry's hash.
- **Mixed versions.** A real report on a v2+ registry admits the set only when the registry,
  roster and parameter files carry the same `screened_on`. A file without the field, or with a
  different value, is refused, naming it. v1 files have no `screened_on` and pair only with
  each other, as today.
- After installation the CLI default becomes the v2 set. `--mode recompute` keeps using the
  versions pinned in `spec.json`.
- *Alternatives:*
  - Edit the shocked v1 entries in place. Rejected: it silently changes frozen files that
    EXP-0045 is bound to.
  - Reuse EXP-0044's calm rows (calm worlds are identical in v2). Rejected: it would mix
    commits and provenance in one screen to save about a third of a cheap run.

### D9. Shock-reactive ordering baseline
`shock_reactive` starts from `blend_0.5`'s logits (`eval/baselines.py:316-327`), the M1
roster's tempered router. For each order it then sets to `-ACTION_LOGIT_BOUND` the logit of
every eligible producer whose `env.system.telemetry.rejected_or_failed_ema(pid, product)`
exceeds half its `routed_ema(pid, product)`. If that excludes every eligible producer of an
order, the order keeps the base logits.
- **Both contracts.** It reads the same ordering-agent view as `blend_0.5`
  (`env.system.ordering_agents[aid].view(...)`) and the domain-owned telemetry. Both exist
  under the legacy and the joint-stack contract, so roster v2's mandatory pair
  `obs_stock_aware/shock_reactive`, which `eval/roster.py:contract_for` sends to LEGACY
  (`product_stream_ordering=False`), runs as well as the joint-stack pairs.
- **Re-admission (decayed-routes floor).** A producer is excluded only while its evidence is
  fresh: its `routed_ema(pid, product)` must also be at least `(1 − α)^6` (α = the telemetry's
  EMA rate, 0.3, so about 0.118) of the summed `routed_ema` of the order's eligible producers.
  Without the floor an excluded producer is never re-admitted. It then gets almost no units,
  so its routed and rejected EMAs decay at the same rate and their ratio stays over the half
  threshold for the rest of the episode, after an outage and after any ordinary overload
  rejection alike (found in task 5.4: an outage in cycles 2–4 left the producer excluded through
  cycle 8). That weak a bar would recreate the benchmark artifact this baseline exists to rule
  out. n cycles after its last routing, the producer's share is at most `(1 − α)^n` while the
  product's routed flow does not shrink. The floor therefore re-admits it as a probe within 6
  cycles, the longest D7 outage. It is excluded again only if the new routing is again mostly
  rejected or failed.
- **No peeking.** It never reads the product-stream-only view or any shock state, so it can
  react only to observed consequences.
- `eval/signal_check.py` uses a hardcoded `COMBOS` list (`:20-31`), so the task adds
  `("stock_aware", "shock_reactive")` there explicitly.
- The producing side needs no new baseline: `entity_planner` already restores low capacity,
  which is how it responds to a capacity hit.
- *Alternative:* a schedule-reading oracle baseline. Rejected: it would peek. The roster must
  hold a fair, trivial reactive bar so that the shocked margin cannot be a benchmark artifact
  (`establish-project-baseline` Risks).

### D10. Gate changes
- Rows gain `shock_digest` and `shock_events`. The pairing assertion covers both digests.
- For every subject episode on a world with shocks, the driver adds a twin job (subject
  `twin:<subject>`). Twin rows never enter the hindsight-baseline or verdict arithmetic.
- `resilience.csv` is a pure function of subject and twin rows plus the episode's schedule
  (recorded per row as JSON), so `--mode recompute` rebuilds it.
- A `--regimes` filter is added for dry-runs.
- CLAUDE.md "Evaluation" and ARCHITECTURE.md §7.2 document the new stats, files and flags.

**Contract-failure accounting** lives in a shared `eval/contract.py`, used by the gate and by
the exit driver (D11). Every failure goes to `contract_failures.csv`, one row each (scope
`job` or `run`, world, episode, subject, replicate, check, error type, message). The checks:
- **Exceptions.** Any exception inside a job is caught and recorded, the job stops, and the
  other jobs finish.
- **Spaces.** Every observation, and every policy action before the env decodes or clips
  it, is checked for full space membership: shape, dtype, bounds (`space.contains`) and
  finiteness (no NaN or inf). A violation is recorded and the job stops. Decoding
  (`env/decode.py`) would otherwise clip it silently.
- **Terminal outcome.** `eval/runner.run_episode` stops at `max_iter` and returns its stats
  either way. Checking `env.terminations` / `env.truncations` after the loop would not work:
  PettingZoo's dead-step handling (`_was_dead_step`, run on `env.step(None)`) deletes each
  finished agent's entries. The checker therefore **latches** each agent's outcome inside the
  step loop, the first time `env.last()` reports `term` or `trunc` for it, before
  `step(None)` removes it. After the loop:
  - every agent in `possible_agents` must have a latched outcome;
  - latched terminations must agree with `dead`, and latched truncations with
    `cycles == horizon`.

  Otherwise it records "no terminal outcome". `cycles == horizon` with no latched truncation
  is a failure; truncation is never inferred from the cycle count.
- **Run-level failures.** Checkpoint hashing and loading in the parent (`run_report` hashes
  checkpoints with `sha256_file` before any worker starts, `eval/gate_cli.py:570`) and the
  parent's digest-pairing assertion are wrapped too. Each is recorded as a `run` row before
  the run ends with a non-zero exit. A digest mismatch still fails the run, as the spec
  requires; it now also leaves a record.
- **Invalid, never partial.** A run with any failure row has status `invalid` in `spec.json`
  and `verdict.json`; it is not an acceptance read.
  - Resume keeps `contract_failures.csv` append-only. A retried job that later succeeds does
    not delete the earlier failure, so the run stays invalid.
  - `--mode recompute` reports `invalid` whenever failure rows exist or expected rows are
    missing. It never builds a verdict from partial rows.

**Versioned screen outputs.** `--mode screen` today hardcodes the v1 output names
(`side-effect-params-v1` at `eval/gate_cli.py:704`; the `frozen/*_v1.json` paths and spec keys
at `:870-880`). The registry and roster parsers have no `screened_on` field. The screen
therefore takes its output version tag from the input registry's name (`worlds-v1` → `v1`,
`worlds-v2` → `v2`):
- it writes `frozen/{worlds,roster,side_effect_params}_<tag>.json` and names
  `side-effect-params-<tag>`;
- for tag `v2` and later, it records `screened_on` = the draft registry hash in all three
  files;
- for tag `v1` it writes no `screened_on` key at all, so a v1 screen on the v1 fixtures gives
  frozen files byte-identical to the base commit.

Parsers accept a missing `screened_on` on v1 files and on v2+ drafts: the unscreened
registry draft and the unfrozen roster draft, which are the screen's inputs. It is required on
frozen v2+ files and checked at admission. The version-pairing refusal of D8
("Mixed versions refused") is implemented here.

### D11. M2 exit driver
`python -m tokenomics.eval.shock_contract` runs two stacks shocked and unretrained, each
episode with its twin, horizon 100:
- **portable**: the gate subject (M1 D1, EXP-0012 orderer + EXP-0035 producer) on every
  registry v2 draft development shocked gate world at 12 and 42 producers, with every
  registered episode seed: 2 sizes × 4 worlds × 10 = 80 episodes, plus 80 twins.
- **flat**: the `champion.json` stack (`m2_stream_stochastic` + `d3_rung_dagger_v4`, loaded
  as `champion.json` declares: product streams, orderer obs version 3) on its native
  held-out topologies (`build_scale_config(seed, depth=3)`, `held_out_topology_seeds` 6–9)
  with the Shocks I generators
  applied through `apply_shock_profile(cfg, profile, producers)`, which adds drift only when
  the consumers are randomized. It runs 10 hash-derived episode seeds per topology: 40
  episodes, plus 40 twins.

It writes the EXP layout: `spec.json` with pins, `episodes.csv`, `resilience.csv`,
`contract_failures.csv` and a `RESULT.md` skeleton. A recorded run pins the "shocks-i"
profile hash, and `--dry-run` may use `shocks-i-smoke` (D7). The exit holds when
`contract_failures.csv` has no rows. Deaths, units, side effects and resilience are reported
and do not gate. *Alternative:* a gate `--dry-run` report. Rejected: the flat stack cannot run
in the gate (fixed widths, M1 D1), and the exit question is the contract, not the margin. The
driver reuses the gate's job runner and row format for the portable stack.

### D12. Records
- ADR-0055 recreates `/srv/research/tokenomics/docs/adr/` and cites its predecessors by git
  history.
- CLAUDE.md Domain Language gains: Shock, Capacity Hit, Producer Outage, Demand Drift,
  Shock-free Twin, Resilience.
- EXP-0046 (M2 exit) and EXP-0047 (v2 calibration) each get a `RESULT.md` (plain English
  first), a README index row and a HISTORY paragraph in the measurement arc.

### D13. Compute
- Reference: EXP-0045 ran 1,280 horizon-100 episodes in ≈ 5.5 min on 8 vCPU (≈ 0.8 vCPU-h
  billed).
- EXP-0046 is ≈ 240 episodes, expected ≤ 1 vCPU-h.
- EXP-0047 is the size of EXP-0044 attempt 2 (≈ 11 vCPU-h billed) × 100/88 runs per
  world-episode, so ≈ 12.5 vCPU-h. The v1 draft roster's 40 stacks are 28 deterministic + 12
  random, i.e. 28 + 12·R = 88 runs at R = 5. `shock_reactive` adds 8 stacks: 7 deterministic
  and `random/shock_reactive`, which runs R = 5 replicates. That is 12 more runs, 100 in
  total.
- A local timing probe (horizon 12, one seed) confirms both estimates before launch. If it
  extrapolates either run above 20 vCPU-h, the implementer stops and reports to the
  supervisor instead of launching.

## Risks / Trade-offs

- [The champion dies or collapses under shocks] → That is a finding, not a contract failure.
  It is reported in EXP-0046 and diagnosed in M3, before any training.
- [Shocked v2 worlds hit zero baseline or fail headroom] → The v2 screen replaces
  zero-baseline worlds from reserves. A headroom failure stops the run, gets reported, and
  becomes a new code task (no pre-planned retune).
- [floor(j·N) = 0 freezes drift at small N] → j = 0.34 gives k ≥ 1 at both sizes. A test pins
  k at 12 and 42.
- [Hidden information leaks before onset] → The unannounced test compares every agent's
  observation, byte for byte, at every cycle before t with the same episode without that
  shock.
- [An unplayed-cycle hit contaminates `r_end`] → `end_capacities` snapshot (D2), plus its
  fixture.
- [The legacy path drifts] → Goldens, a fingerprint-unchanged test over every packaged config
  and fixture, and a v1 content-hash test.
- [Twins double the subject cost on shocked worlds] → Twins run for the subject only. It is a
  small share of a report, which the roster dominates.
- [Overlapping shocks confound per-shock dips] → Flagged rows. Per-shock twins are rejected
  for cost.

## Migration Plan

All changes are additive: the new fields default to None, the new files are new versions,
and the v1 files and v1 default stay in place until the frozen v2 set is installed. Rollback
is a revert, which leaves EXP-0045's bindings valid because v1 never changes.

## Open Questions

The numeric values in D7 and the D9 threshold are pre-registered here and may be tuned only
before the EXP-0047 freeze. None of them changes the specs or the tasks.
