# Proposal

## Why

The roadmap's milestone M2 "Shocks I + drift" (`establish-project-baseline` design D1, D3)
must add the first supply shocks and smooth demand drift to the environment before the
development audit (M3) and any training (M4). Today the only stochastic element is B9
per-cycle random demand, so the gate's "shocked" regime contains no shocks at all
(`eval/families.py` module docstring, regime `shocked`). The `stochastic-environment`
contract already specifies seeded, unannounced, action-independent shocks. None of that
exists in code yet. M2's exit is: every delivered spec scenario is a pytest, and the current
champion runs shocked, unretrained, with zero contract failures.

## What Changes

- **Shock model (domain, pure data).** There are five shock kinds (capacity hit, producer
  outage, supply-link cut, demand surge, fund shock). Each entry has a kind, a target, a start
  cycle, a duration and a magnitude. A configuration may carry an explicit shock schedule
  and/or one seeded generator per kind. Configuration validation rejects every kind that is not
  built yet (link cut, surge and fund shock until M6), with an error naming the kind.
- **Capacity hit and producer outage mechanics (domain).** The domain replays an injected
  schedule and has no RNG of its own. A capacity hit removes `floor(m·c)` units of current
  capacity at the start of its cycle. During an outage a producer accepts no new bids and
  assembles no reserved (in-flight) production; its capacity keeps degrading and it can
  still buy.
- **Seeded schedule generation (env).** Shock schedules and drifting demand are drawn
  before the episode starts, from world configuration + episode seed only. Each shock kind
  and the drift draw from their own derived stream, so the existing B9 demand draws and any
  earlier kind's schedule are byte-identical when a kind is added. A `shock_digest` episode
  stat pins the schedule. Observation and action spaces do not change, so existing
  checkpoints load unchanged.
- **Smooth demand drift (env).** A world-level drift parameter `j` makes at most
  `floor(j·N)` of the N drifting consumer orders change between consecutive cycles. B9
  per-cycle sampling stays available and unchanged.
- **Resilience report (eval).** Each shocked episode with shocks gets a *shock-free twin*:
  the same world, episode seed and demand schedule, with the shock schedule empty. For each
  shock, the report gives the unit dip against the twin and the cycles until fulfilled units
  recover to within a pre-registered tolerance.
- **Shocked gate worlds get real shocks, under new versions.** The frozen v1 files
  (`worlds_v1.json`, `roster_v1.json`, `side_effect_params_v1.json`) are not changed. Registry
  v2 keeps every v1 world and seed, and gives the shocked worlds a pre-registered "Shocks I"
  profile (capacity hits, outages, drift). Roster v2 adds a shock-reactive ordering baseline.
  A baseline-only screen on v2 freezes registry v2, roster v2 and side-effect parameters v2.
  The gate checks shock digests across subjects, runs twins, and reports contract failures.
- **M2 exit experiment.** The portable champion stack (gate subject, M1 design D1) and the
  flat `champion.json` stack run shocked and unretrained, and zero contract failures are
  required. A local timing probe sizes the run, and the run is kept small (no budget for
  1000+ vCPU-h runs).
- **ADR-0055** records the schedule and shock design.

No **BREAKING** changes. With no shocks and no drift configured, every existing episode
statistic, scale-generator output, default seeding, config fingerprint and frozen v1 file
stays byte-identical.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `stochastic-environment`: makes concrete what M2 delivers. A per-kind derived stream and
  preserved demand draws ("Seeded event schedule"). Kind-specific targets ("Shock kinds").
  Integer rounding for a capacity hit and the pinned end-of-episode capacity ("Capacity hit").
  An outage blocking in-flight completion while capacity still degrades ("Producer outage").
  The M2 kinds and the rejection of link cut / surge ("Shock delivery order"). The shock-free
  twin, tolerance window and censored recovery ("Resilience is reported").
- `gate-harness`: the shocked regime of registry v2 carries Shocks I + drift, and calm worlds
  carry none ("Regimes"). The shock schedule digest is checked across subjects ("Identical
  demand across observation contracts"). Roster v2 adds a shock-reactive baseline ("Frozen
  baseline roster"). The resilience table and contract failures are in the report ("Gate
  report contents"). New: "Versioned registry for shocked worlds" and "Contract failures are
  counted".

## Impact

- Domain: `tokenomics/tokenomics/domain/config.py` (shock and drift schema, validation, JSON
  load), a new `domain/shocks.py` (types, application), `domain/system.py` (schedule
  injection, cycle hooks, post-cycle capacity snapshot), `domain/producing_agent.py` (outage
  in `produce()`).
- Env: a new `env/shock_schedule.py` (derived streams, generators, digest),
  `env/aec_env.py` (`reset` wiring, drift demand, fulfilled-units trace, `shock_digest`),
  `env/side_effects.py` (`r` read from the pre-next-cycle snapshot), `env/spaces.py`
  (fingerprint omits unset shock/drift fields).
- Eval: `eval/families.py` (`shock_profile` param), `eval/registry.py` (`build_worlds_v2`,
  calm-has-no-shocks check), new `eval/registry/{worlds_v2,roster_v2,shock_profiles_v1}.json`,
  `eval/gate.py` / `eval/gate_cli.py` (twins, resilience, shock digest, contract failures),
  a new `eval/resilience.py`, a new `eval/contract.py` (contract-failure checks),
  `eval/baselines.py` (`shock_reactive`), `eval/signal_check.py` (its `COMBOS` row),
  `eval/roster.py` (roster v2, `screened_on`), `eval/scale.py`
  (`config_to_dict` round-trip), a new `eval/shock_contract.py` (exit driver).
- Records: `/srv/research/tokenomics/docs/adr/0055-*.md`, the CLAUDE.md Domain Language and
  Evaluation sections, ARCHITECTURE.md §7.2, EXP-0046 (M2 exit) and EXP-0047 (v2
  calibration) with README rows and HISTORY paragraphs.
- Compute: two small CPU pod runs, sized by a local timing probe (tasks.md).
