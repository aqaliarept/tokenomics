# Centralized critic (MAPPO / CTDE) and truncation bootstrapping

Phase 1.5 P2. The smoke trainer (ADR-0003 era) gave each policy its **own**
critic on its **local** observation. With a single shared cooperative reward
that is the wrong baseline: a producing agent's local obs can't see the ordering
phase, an ordering agent's can't see the other producers' investment decisions,
yet the reward they share depends on all of it. The local-obs critic therefore
estimates a baseline conditioned on partial state, inflating advantage variance
and mis-assigning credit — the exact problem CTDE exists to fix.

## Decision — one centralized critic, two decentralized actors

**Actors stay local (decentralized execution):** producing actor on the 16-ish
local obs, ordering actor on its order-slot obs. Unchanged — this is what gets
deployed.

**One shared critic on global state (centralized training):** a single
`V(s_global)` network, trained on transitions from *both* role buffers. Because
the reward is shared and cooperative, every agent's return is the same global
return; one value function is the correct baseline for all of them. Advantages
are still computed per role buffer (each samples the global trajectory at its
own decision points), but against the common global value.

**Global state** = every producer's full state (capacity, max, requested,
degradation, upgrade_running, restoration_running, cut-off price) + cycle
progress. dim = `7·P + 1`. In Phase 1 the producers *are* the system state —
capacity health drives every order outcome — so this is a sufficient global
summary without dumping raw orders. Built in `env/encode.py::encode_global`,
exposed as `env.global_state()`; dim is `EnvSpec.global_state_dim`.

## Decision — GAE distinguishes truncation from termination

The old GAE treated `term or trunc` identically (the bug flagged in the plan).
Now:

- **Termination** (system death — all producers at zero capacity): a true
  terminal. No bootstrap; future value = 0.
- **Truncation** (horizon reached): the episode would have continued. Bootstrap
  the final transition with `V(s_final)` from the central critic.

Concretely each transition carries a `term` flag (kills the value backup) and
the trajectory's last transition carries a `boot` value (0 if terminated, else
`V(s_final)`). Harmless while episodes always hit the horizon; necessary now
that a learning policy can die early and episode length varies.

## Trade-offs

- Global state is hand-rolled, not the raw concat of all local obs. Cheaper and
  Phase-1-sufficient; revisit when multiple producers per type (post-Phase-1)
  make order routing matter.
- Still a *smoke* trainer: single critic MLP, no minibatching across envs. Real
  MAPPO scale comes with P3 vectorization.
