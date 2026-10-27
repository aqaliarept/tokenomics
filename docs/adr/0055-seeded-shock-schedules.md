# Seeded, per-kind shock schedules replayed by an RNG-free domain

Status: accepted (add-shocks-i-drift, milestone M2 "Shocks I + drift")

This directory was deleted from this branch in commit `18c717d` ("cleanup") and is recreated
by this ADR. The predecessors cited below can be read with
`git show 18c717d^:docs/adr/<file>`.

## Context

M2 adds the first supply shocks (capacity hits, producer outages) and smooth demand drift
(`stochastic-environment` spec). These constraints shape it:
- The domain must stay RNG-free.
- Every policy comparison must be paired on the same events.
- Existing checkpoints must load unchanged.
- Today's demand comes from one sequential RNG stream (`env/aec_env.py`
  `_build_consumers`), so any extra draw on it would shift every later consumer's demand.

## Decision

1. **Shocks are configuration data.** `domain/shocks.py` defines all five kinds (capacity
   hit, producer outage, supply-link cut, demand surge, fund shock), each with a target,
   start cycle, duration and magnitude. `Configuration.shocks` holds explicit events and at
   most one seeded generator per kind. Validation rejects every kind not yet built (link
   cut, surge and fund shock until M6), naming it. Every shock starts at cycle ≥ 2, so the
   state observed at reset is never shocked. The world stays fully described by its config,
   so registry replay, the task archive and the fingerprint all cover it.
2. **The env draws and the domain replays.** At reset the env draws the schedule from the
   world config and the episode seed only, and injects it. It cannot depend on any action.
   Each shock kind and drift use their own stream, derived from
   `sha256("tokenomics|<stream>|<seed>")`. The legacy demand stream `default_rng(seed)` is
   untouched. Adding a kind therefore never moves an earlier kind's schedule or any demand
   draw. Generator windows live in the config, not in the horizon, so a shorter horizon's
   schedule is a prefix of a longer one.
3. **Mechanics.**
   - A capacity hit removes `floor(m·c)` units, computed exactly, at the start of its cycle;
     a positive capacity never reaches zero, so a hit alone cannot trigger death (ADR-0030).
   - An outage rejects every new bid and assembles no reserved production (ADR-0025
     reservations keep aging to their TTL). Capacity still degrades, and the producer keeps
     buying. The auction stays the producer's own (ADR-0029).
   - The env reads end-of-cycle capacity from a snapshot taken before the next cycle's
     `start_cycle`, so an unplayed cycle's shock never reaches `r_end`.
4. **Unannounced, contract-preserving.**
   - No observation feature exposes the schedule; agents see only consequences, such as
     rejected sub-orders in the existing telemetry.
   - Spaces depend on topology only (ADR-0038: perturbations preserve dims), so checkpoints
     load unchanged.
   - Product streams (ADR-0052) are unchanged, because drift stays inside each consumer's
     pool.
5. **Fingerprint.** `_canonical_economy` leaves out `shocks` / `demand_drift` when unset, so
   every config without them keeps its ADR-0035 fingerprint. A shocked config gets a new one.
   An empty consumer `schedule` is left out too. Hashing that empty default (M1 task 2.2,
   `55539bf`) had silently moved every fingerprint, so pre-M1 checkpoints such as the
   EXP-0006B champion (`548120bd72921234`) no longer matched; omitting it restores the pre-M1
   values.
6. **Resilience against a shock-free twin.** A shock's dip and recovery time are measured
   against the same world and episode seed with the shock schedule emptied. The demand stays
   identical because the streams are independent. A calm-regime world is not used: its
   cyclic demand differs, so the dip would mix demand with shock effects.
7. **Gate worlds get shocks only through new versions.** The frozen v1 registry, roster and
   side-effect parameters never change. Registry v2 keeps every v1 world and seed, and gives
   its shocked worlds the pre-registered "shocks-i" profile. Shock counts scale per producer.
   The window is cycles 21–90, after the warm-up. Drift is `j = 0.34`, so at least one order
   changes per cycle at 12 producers.

## Considered Options

- *Shocks as an env keyword outside the config.* Rejected: a world would no longer be
  described by its config, so replay and fingerprints would miss the shocks.
- *Outage as capacity set to 0.* Rejected: it would trip death detection and destroy the
  capacity state the spec says keeps degrading.
- *Draw shocks from the existing demand RNG.* Rejected: every later demand draw would shift.

## Consequences

- A "fingerprint mismatch" warning on a shocked world is expected and is not a contract
  failure.
- Per-shock dips are confounded when shocks overlap; such rows are flagged rather than
  costing one twin per shock.
