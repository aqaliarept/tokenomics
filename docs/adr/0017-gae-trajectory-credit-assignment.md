# GAE credit-assignment bug — link transitions by trajectory, not buffer order

Track A. The single dominant cause of the "collapse-from-good-init" instability
was not optimization, plasticity, reward design, or the critic. It was a
**credit-assignment bug**: GAE backed up advantages along buffer adjacency, but
the shared role buffer interleaves many trajectories.

## The bug

`collect()` stores transitions in two flat role buffers (all producing agents in
one, all ordering agents in the other). Each buffer interleaves:

- multiple rollout envs (`num_envs=4`),
- multiple agents sharing the role (4 producing, 8 ordering),
- multiple episodes as finished envs reset and reuse.

`gae()` treated adjacent rows as one trajectory:

```python
next_val = val[t + 1] if t + 1 < len(val) else boot[t]   # WRONG
last = delta + gamma * lam * nonterminal * last
```

But `val[t+1]` almost always belongs to a *different env or agent*. The recursive
advantage leaked other agents' rewards across trajectory boundaries; the critic
was then trained on `adv + val` — those same mixed targets. Advantage whitening
cannot repair this: it renormalizes the corrupted values but not their missing
trajectory identity. A low normalized `closs` only meant the critic fit corrupted
targets well.

Second defect: at the rollout cut (`n == steps`), the latest action for every
live `(env, agent)` is still waiting for its AEC reward. Those rows held the
placeholder reward `0.0`, no successor, no bootstrap — yet were fed to PPO as if
they were real zero-reward events (~up to `num_envs × role_agents` rows/batch).

## The fix

Represent the trajectory graph explicitly (`gae.fix` proposal):

1. **`Buf.next_idx`** — `next_idx[i]` = buffer index of the next transition for
   the same `(env, agent)`, or `-1` at trajectory end / batch exit. Every link
   points forward (the successor is appended after the predecessor), so one
   reverse pass suffices.
2. **Link on reward attach** — when an agent is selected again, pop its pending
   row, attach the delayed reward/terminal, record `boot[prev] = V(current
   state)`, then (if the episode continues) append the new row and set
   `next_idx[prev] = cur`.
3. **`gae()` follows links** — `next_val = val[next_idx[t]]` (or `boot[t]` if
   `-1`); `next_adv = adv[next_idx[t]]` (or `0`). Truncation/termination
   semantics unchanged (ADR-0004): bootstrap on cut/truncation, never on death.
4. **`drop_pending()`** — before returning, remove rows still open at the cut and
   remap links. Their predecessors already carry `boot = V(s_next)`, so they get
   the correct finite-rollout target `r + γV(s_next)` without the unfinished
   action.

Because `boot[prev]` is recorded on *every* non-terminal step (not only
truncation), a predecessor whose successor is dropped at the cut still
bootstraps correctly.

## Result

Reproduction was seed 1 — the canonical collapse-from-good-init seed.

- **Before:** good init (iter 0 `dead 0.00, ret 3510`) destroyed by iter 10
  (`dead 1.00`), never recovered.
- **After (KL+low-std config):** clean learn-and-hold — `dead 0.00`, `ret ~3300`
  flat from iter 5 to iter 34.
- **After (clean PPO config, no target-kl, `log_std_init 0`):** `ret 3623,
  cons 377, dead 0.00`, holds to the final iter.

10-seed clean-config sweep (100 iters): every **good** init (1, 2, 5) reaches a
genuine *trained* peak of `ret 3514 / 3623 / 4075` — all beat the `3230` bar — and
these are mid-run trained results, not the iter-0 init. So `best-on-eval`
(ADR-0011) now banks an honestly-trained survivor for the good-init seeds, which
it could not before.

## What it does NOT fix

Residual **peak-then-collapse** persists on some good inits (seeds 2, 5 peak then
die by the final iter). The fix removes the dominant confound, so this residual is
now a *clean* question — correct advantages, yet a known-good policy is still
abandoned. That is the loss-of-plasticity / feature-rank-collapse hypothesis
([No Representation, No Trust](https://arxiv.org/abs/2405.00662)), no longer
maskable by invalid credit assignment.

## Reinterpreting the prior Track-A experiments

ADRs 0010–0016 all ran on corrupted advantages; their conclusions stand only as
*observations on the broken trainer*, not as facts about PPO on this env:

| prior finding | revised reading |
|---------------|-----------------|
| lower exploration (0012) helps | reduced variance in corrupted batches; not proof exploration was the cause |
| target-KL (0015) prevents some collapse | capped the damage of a *wrong* update; did not make the direction right |
| critic warm-up (0013) doesn't fix seed 1 | a warmer critic still fit mixed-trajectory targets |
| no-critic (0014) flips which seeds work | removed one corrupted component, re-rolled the random update field |
| reward shaping (0016) inconsistent | shaped rewards still leaked across unrelated trajectories |
| `closs ≈ 1` (0007) | critic fit *normalized corrupted* targets — not evidence of correct targets |

Target-KL and low exploration are kept as standard PPO safeguards, not as the
load-bearing fixes they appeared to be.

## Follow-up (separate from this patch)

`ValueNorm` ("PopArt-lite", ADR-0007) updates normalization stats without
compensating the critic output layer, so denormalized predictions drift even at
fixed weights. Now that GAE targets are correct, measure critic explained
variance and value drift before deciding whether to freeze stats after warm-up,
implement real PopArt output preservation, or drop value normalization in favour
of `REW_SCALE`.

## Tests

`test_gae_follows_links_not_adjacency` (interleaved A0,B0,A1,B1 must give
per-trajectory advantages `[3, 300, 2, 200]` — the old adjacency code fails it),
`test_drop_pending_removes_open_rows_and_remaps_links`, and collector link
invariants in `test_vectorized_collect_fills_both_buffers`. 27 pass.
