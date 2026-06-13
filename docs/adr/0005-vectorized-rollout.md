# ADR-0005: In-house synchronous vectorized rollout

## Status
Accepted (Phase 1.5 P3).

## Context
P2 gave a working MAPPO loop (ADR-0004) but `collect()` rolls a **single** AEC
env per iteration. More decorrelated samples per update need many envs. The plan
(slice 5) suggested SuperSuit.

SuperSuit fights this env's design:
- Its vector path (`pettingzoo_env_to_vec_env` → `concat_vec_envs`) wants a
  homogeneous ParallelEnv. Ours is AEC, turn-based, with **two different obs
  dims** (producing vs ordering). SuperSuit fits that only by padding all obs to
  a common shape + `black_death` — pad dims the actors must learn to ignore.
- The centralized critic reads `env.global_state()` (ADR-0004). SuperSuit vec
  wrappers expose only the standard obs/reward/done API — no hook for a custom
  per-env global accessor; it would have to be smuggled through `info`.
- Its value-add is parameter sharing into one agent stream, but we keep **two
  role actors + one critic** and demux by agent id anyway.

The one real SuperSuit win is multiprocess wall-clock speedup. For a smoke
trainer proving learnability on a pure-python (fast) env, that is not the
bottleneck.

## Decision
Generalize `collect()` to drive a **list of independent AEC envs** in
round-robin, appending decisions to the same two role buffers. No new dependency.

- `num_envs` envs, each `reset()` with a distinct seed drawn from the run RNG.
  The domain is deterministic; only the **stochastic policy sampling**
  decorrelates trajectories across envs (CLAUDE.md: seed changes stochastic
  policies only).
- Pending-reward bookkeeping is keyed by **`(env_index, agent)`** so identical
  agent names across envs do not collide.
- An env that terminates/truncates is flushed (dead-steps attach final reward /
  bootstrap) then `reset()` and reused, until the iteration's total decision
  budget `steps` is met.
- `steps` keeps its meaning: **total live decisions per iteration**, now spread
  across `num_envs`. `global_state()` and the role split are untouched.

**Synchronous only.** All envs step in this process. No wall-clock speedup over
one core — the deliverable is sample decorrelation and a `num_envs` knob. True
multiprocess throughput (subprocess workers, gym `AsyncVectorEnv` style) is a
later change behind the same `collect()` seam if real runs need it.

## Consequences
- Centralized-critic design (ADR-0004) unchanged; `collect()` is the only seam
  that grew.
- No SuperSuit / extra deps.
- One-core wall-clock unchanged; if throughput becomes the limit, swap the sync
  loop for subprocess workers without touching actors/critic/GAE.
