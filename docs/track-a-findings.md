# Track A findings — cracking Phase 1 (beat the baseline bar)

**Goal:** a trained MAPPO policy that beats `restore_when_low` + `even_split` —
`return ≈ 3230`, `cons✓ ≈ 359`, `0 %` death over horizon 100.

**Status (2026-06-14, Session 5):** the corrected trainer now finishes above the
baseline on all three canonical regression seeds. In addition to trajectory,
rollout, and PopArt fixes, PPO now masks producing action aliases, conditions the
critic on the acting agent and local observation, and defaults to the matched
actor controls: producing temperature `0.25`, ordering std `0.14`, target-KL
`0.008`. Final returns are seed 1 `3568`, seed 2 `3438`, seed 5 `4050`, all with
`dead=0`. Seed 5 still has a transient mid-run collapse, so monotonic stability
remains open. Loss of plasticity is not needed to explain the original failure.
The full suite passes (`36 passed`).

Sessions 1-4 are retained as investigation history but their causal conclusions
are superseded where Session 5 says so.

---

## What was changed (in order)

| ADR | Change | Effect |
|-----|--------|--------|
| 0007 | **Critic fix** — value-target normalization (`ValueNorm`, running mean/std), 128-wide critic, 8 epochs, ±5σ target clip | `closs 1008 → ~2`, flat→fitting. Necessary; advantages now trustworthy |
| 0008 | **Schedule** — actor LR anneal→10%, entropy decay→0, longer runs, CLI flags | Helps; not sufficient alone |
| 0009 | **Rollout-gated curriculum** — advance difficulty on completed-episode survival | Fixed in Session 4: no false 100% death when no episode completes, and no iteration floor overriding the gate |
| 0010 | **Capacity-floor shaping** — `surv_coef·(health−1)` on training reward only (eval stays raw) | Finicky: coef 15 too weak, coef 30 *worse* (killed even seed 0). Not the robust lever |
| 0011 | **Best-on-eval checkpoint** — deploy best policy seen, ranked survival-first | Operational safeguard. Note: iter 0 is after one PPO update, not untouched initialization |
| 0012 | **Low exploration** — ordering Gaussian log std reduced from `0`; final matched default `-2.0` (std 0.14) | Keeps stochastic ordering rollouts close to mean-action deployment |
| 0015 | **Target-KL early stop** — break actor epochs when approx-KL > 1.5×target | Bounds repeated PPO epochs; historical conclusions were confounded by later correctness bugs |
| 0013/0014/0016 | critic warm-up, no-critic, reward shaping | Explored, **shelved** — none fix the good-init collapse. See Session 2 ruled-out table |
| **0017** | **GAE credit-assignment fix** — explicit `next_idx` links; drop unfinished frontier | Corrects invalid advantages and returns. Session-3 hold results were still confounded by episode resets |
| **0018** | **Persistent rollout episodes** — rollout cuts bootstrap but do not reset environments | Exposes cycles 22-100; old training repeatedly saw only cycles 0-21 |
| **0019** | **Output-preserving PopArt** — compensate final critic layer when value stats change | Removes value drift at fixed weights; prediction invariance verified below `8e-9` |
| **0020** | **Producing sampling temperature 0.25** | Producing stochastic death 70% → 0% on fixed-logit ablation; aligns training and argmax deployment |
| **0021** | **Producing valid-action mask** | Prevents PPO crediting restoration/upgrade actions that the environment silently stripped because the investment was already running |
| 0022 | **Investment request cost sweep** | 0.5/1/2 did not fix seed 5; retained as an experimental flag, default 0 |
| **0023** | **Agent-conditioned centralized critic** | Adds padded local observation + agent identity; old critic could not see ordering-agent orders or distinguish AEC decision positions |
| **0024** | **Validated actor trust-region defaults** | Temperature 0.25, ordering log std -2.0, target-KL 0.008 finish above the bar on seeds 1/2/5 |

---

## Historical diagnosis before the correctness fixes

The following pattern motivated Sessions 1-2. ADR-0017 through ADR-0024 later
showed that it was a symptom of invalid credit assignment and mismatched policy
updates, not evidence for a standalone loss-of-plasticity mechanism.

Across **every** experiment one pattern held, independent of curriculum,
shaping, and batch size:

- Seeds whose random init was **bad** (greedy dies at iter 0) tended to **learn**
  survival and, with low exploration, **hold** it.
- Seeds whose random init was **good** (greedy survives at iter 0) **collapsed**
  within ~30 iters and usually never recovered.

A surviving policy being *destroyed by training* is the signature of
**optimization instability**, not a missing reward. Confirmed by ADR-0012:
cutting the ordering-actor exploration noise (std 1 → 0.37) changed seed 0 from
*peak-then-collapse* to *stable survival to the final iter*. The std=1 Gaussian
on the ordering logits made stochastic rollouts wildly unlike the deployed
greedy policy — most rollouts died, so the gradient was death-dominated noise.

Why not reward shaping (A4)? Death carries no explicit score penalty (it just
ends the episode), so survival only pays through far-off foregone reward. Adding
a capacity-floor term should help in principle, but empirically it was
magnitude-sensitive and never gave cross-seed robustness — the instability
dominated the signal.

---

## Seed results (no curriculum, A1+A2; 150 iters, 1024 steps)

Final-iter greedy eval (3 seeds each), target = `dead 0%, ret 3230, cons 359`.

| config | seed | final dead | final ret | note |
|--------|------|-----------|-----------|------|
| std=1 (log_std 0) | 0 | 0.00 | 3490 | passes but had drifted in other runs |
| std=1 | 1 | 1.00 | 1705 | collapse from good init |
| std=1 | 2 | 1.00 | 1858 | collapse from good init |
| std=1 | 3 | 1.00 | −303 | never survives at any eval |
| std=1 | 4 | 1.00 | 2030 | never survives |
| std=1 | 5 | 1.00 | 2206 | survives only at iter 0 (init) |
| **std=0.37 (log_std −1.0)** | **0** | **0.00** | **3571** | **learns it90, holds it90→149 — stable** |
| std=0.37 | 1 | 1.00 | 1951 | still collapses from good init |

So: **1/2 target seeds stable.** seed 0 solidly beats the bar and holds; seed 1
unsolved.

Dead ends ruled out: bigger batch (4096) did **not** help (seed 0 collapsed too);
eval-gated curriculum was defeated by its own iter floor; shaping coef sweep
found no robust value.

---

## The unsolved nub + next lever (not yet run)

seed 1 collapses **from a good init** — the first ~30 updates destroy a policy
that already survived at iter 0. Prime suspect: **destructive early actor updates
on not-yet-warm advantages**. Candidate fixes, in priority order:

1. **Critic warm-up** — freeze the actors for the first ~K iters (or until
   `closs` < threshold) so the first policy-gradient steps use a trustworthy
   baseline.
2. **Actor LR warm-up** — ramp actor LR up over the first ~10 iters instead of
   starting at full `3e-4`.
3. Even lower `log_std` init (−1.6, std ≈ 0.2) and/or smaller actor LR overall.

Each targets the early-update instability directly. Recommend (1) first.

---

## Session 2 — the collapse is two mechanisms, one fixed

The "peak-then-collapse" was not one bug. Splitting it was the whole job.

### Mechanism 1 — learn-up collapse → FIXED by target-KL (ADR-0015)

Seeds that start *below* survival and learn up used to peak then drift back. PPO
clipping does not actually bound the step; one over-large update tips the policy
out and plasticity loss blocks recovery ([No Representation, No Trust](https://arxiv.org/abs/2405.00662)).
`--target-kl 0.015` breaks the actor epochs when approx-KL exceeds 1.5×target.
Result: bad-init learners reach `dead 0.00` and **hold to the final iter**, robust
across `target_kl` 0.008/0.015. Seeds 0 and 9 finish at ret 3545 / 3893.

### Mechanism 2 — collapse-from-good-init → DIAGNOSED, unsolved

Seeds whose random init already survives collapse within ~15–30 iters. Halving
the trust region did **not** touch the iter-30 crash → not an over-large step.
**10-seed sweep, critic + KL 0.015, classified by iter-0 greedy survival:**

| init @ iter 0 | seeds | survive @ final |
|---------------|-------|-----------------|
| good (`dead 0`) | 1, 2, 5 | **0/3 — all collapse** |
| bad (`dead 1`)  | 0, 3, 4, 6, 7, 8, 9 | 2/7 (seeds 0, 9 learn-and-hold) |

**The law:** every surviving init is destroyed by training; every learned survivor
comes from a bad init. The collapse is **init-quality-driven, not a
hyperparameter** — a good init is a knife-edge the optimizer walks off.

### Levers ruled out for mechanism 2 (each one run)

| lever | flag / ADR | result |
|-------|-----------|--------|
| critic warm-up | `--critic-warmup` / 0013 | banks a survivor via best-on-eval, doesn't hold; regressed seed 0 |
| drop critic | `--no-critic` / 0014 | *flips* which seeds work (seed 1 ✓ 4399, seed 0 ✗) — critic not the culprit |
| trust-region size | `--target-kl 0.008` / 0015 | no effect on the iter-30 crash |
| death penalty | `--death-penalty` / 0016 | negligible after REW_SCALE; crash is abrupt, not drift |
| graded low-cap penalty | `--low-cap-coef` / 0016 | byte-identical trajectory — too small, fires too late |
| lower exploration | `--log-std-init -1.6/-2.0` | still collapses, arguably faster |

### Where this leaves the deliverable

best-on-eval (ADR-0011) ships a genuine trained survivor for the seeds that learn.
Good-init seeds only degrade from their lucky start, so for them best-on-eval
grabs iter 0 — honest about init quality, not a trained result. Open question:
why does the optimizer leave a known-good policy? (loss of plasticity / feature
rank collapse is the literature's prime suspect — PFO regularization untried.)

## Session 3 — the root cause was a credit-assignment bug, not instability

Sessions 1–2 chased *optimization instability* — the symptom. The cause was a
GAE bug (ADR-0017). `collect()` buckets transitions into two flat role buffers
that interleave 4 envs, 12 agents, and episode resets. `gae()` walked **buffer
adjacency** (`val[t+1]`, recursive `last`), so for almost every row the "next
state" belonged to a different env or agent. PPO advantages contained unrelated
agents' rewards; the critic was trained on the same mixed targets. A surviving
("good") init has something to lose, so a corrupted update direction reliably
pushed it off the narrow survival region — exactly the reported law.

### The fix (ADR-0017)

- `Buf.next_idx[i]` = the next transition for the same `(env, agent)`, set when
  the agent is reselected and its delayed reward is attached.
- `gae()` follows links: `next_val = val[next_idx[t]]` / `boot[t]` at the end.
- `drop_pending()` removes the still-open frontier at the rollout cut (placeholder
  reward `0.0`, no successor); predecessors keep `boot = V(s_next)` so their
  finite-rollout target stays correct.

### Historical result — GAE fixed, but episodes still reset each iteration

| init @ iter 0 | clean PPO (no kl, log_std 0): best / final | takeaway |
|---------------|--------------------------------------------|----------|
| good — seed 1 | **3623 / 3623** | learns and **holds** — law broken |
| good — seed 2 | 3514 / collapse | trained peak beats bar; then peak-collapse |
| good — seed 5 | 4075 / collapse | trained peak beats bar; then peak-collapse |
| bad — 0,3,4,6,7,8,9 | mostly never survive | unchanged |

This table established that the GAE fix mattered, but the apparent seed-1 hold
did not survive ADR-0018. Those runs repeatedly trained only on cycles 0-21.

### Reinterpretation — Sessions 1–2 ran on corrupted advantages

| prior finding | revised reading |
|---------------|-----------------|
| lower exploration helps (0012) | less variance in corrupted batches; not the cause |
| target-KL prevents some collapse (0015) | capped a *wrong* update's size; direction still wrong |
| critic warm-up no help (0013) | warmer critic still fit mixed targets |
| no-critic flips seeds (0014) | re-rolled the random update field |
| reward shaping inconsistent (0016) | shaped rewards leaked across trajectories |
| `closs ≈ 1` (0007) | critic fit *normalized corrupted* targets |

### What remains

At this point residual collapse was not yet a clean plasticity question:
rollout-state resets, value-normalization drift, and stochastic/deployment
mismatch remained. Session 4 addresses those confounds.

## Session 4 — two remaining trainer correctness defects

The GAE fix exposed two independent issues that had to be corrected before
interpreting residual peak-then-collapse as representation loss.

### Rollout cuts reset the episode (ADR-0018)

Every training iteration rebuilt and reset all rollout environments. At the
default `1024 steps / 4 envs / 12 agents`, each environment reached only cycle
21 of the 100-cycle horizon, then restarted from cycle 0. The trainer never saw
the late-cycle states where residual system death occurred.

Rollout environments now persist across PPO updates while their configuration
is unchanged. Batch cuts bootstrap value targets but no longer reset episode
state. A batch with no completed episode reports `roll_dead=NaN`, not the false
`1.0` fallback previously consumed by curriculum gating. Gated curriculum no
longer has an iteration-floor override; it holds difficulty until an actual
completed episode supplies evidence.

### ValueNorm changed V(s) at fixed weights (ADR-0019)

Updating EMA mean/std changed every denormalized value prediction before any
optimizer step. The critic now uses output-preserving PopArt: after statistics
change, its final layer is inversely rescaled so raw-unit `V(s)` is invariant.
The variance update also includes movement of the batch mean.

These are correctness fixes, not hyperparameter changes. Residual-collapse
claims must be rerun after ADR-0018 and ADR-0019.

### Matched rerun after persistent episodes + PopArt

Clean PPO (`100 iters`, no curriculum, no target-KL, `log_std_init=0`) no
longer supports the Session-3 "seed 1 holds" conclusion:

| seed | best observed | final | result |
|------|---------------|-------|--------|
| 1 | dead 0, ret 3548 (iter 0) | dead 1, ret 1754 | collapses |
| 2 | dead 0, ret 3514 (iter 0) | dead 1, ret 2164 | collapses |
| 5 | dead 0, ret 4075 (iter 0); ret 3439 at iter 40 | dead 1, ret 1851 | collapses |

Here "iter 0" is already **after one PPO update**; the trainer does not evaluate
the untouched random initialization. The previous 100-iteration result was
produced by repeatedly training only on cycles 0-21.

The new rollout metric exposes a policy-objective mismatch: completed stochastic
training episodes frequently have `roll_dead=1.00` while the deterministic
mean/argmax evaluation policy has `dead=0.00`. PPO optimizes the stochastic
policy, but deployment evaluates a discontinuous deterministic projection.
This must be measured and addressed before attributing the remaining greedy
collapse to loss of plasticity.

### Producing policy stayed nearly random (ADR-0020)

Role ablation on a greedy-surviving checkpoint isolated the mismatch:

| producing | ordering | stochastic eval death |
|-----------|----------|-----------------------|
| greedy | greedy | 0% |
| stochastic | greedy | 65% |
| greedy | stochastic | 0% |
| stochastic | stochastic | 45% |

Producing entropy was `1.334`, near the four-action maximum `1.386`. Removing
the entropy bonus barely changed it (`1.326`), so the problem was not only
regularization: shared-return PPO learned the correct argmax ordering without
making the Categorical distribution decisive.

Sampling the same fixed logits at temperature `0.25` reduced stochastic
producing death from 70% to 0% and produced return `3410`. Track A now defaults
to `--prod-temperature 0.25`, which leaves greedy argmax unchanged while making
training rollouts resemble deployment.

### 100-iteration result after ADR-0018 through ADR-0020

Configuration: no curriculum, no target-KL, producing temperature `0.25`,
ordering `log_std_init=-2.0`.

| seed | best | final | completed-rollout survival |
|------|------|-------|----------------------------|
| 1 | ret 4232, dead 0 (iter 90) | **ret 3286, dead 0** | 35/40 batches with completed episodes had 0% death |
| 2 | ret 3363, dead 0 (iter 0) | ret 2394, dead 0 | 6/39 survival batches |
| 5 | ret 4151, dead 0 (iter 0); ret 3920 at iter 20 | ret 1631, dead 1 | 4/36 survival batches |

Seed 1 now learns from a genuinely bad untouched initialization (`ret 153`,
dead 1 before training), reaches strong trained policies, and finishes above the
bar. Seeds 2 and 5 remain unstable/low-performing under stochastic rollouts.
That residual is now the next research target; it is not fixed by the accounting
changes above.

### Producing action aliases corrupted policy credit (ADR-0021)

The environment cleared restoration/upgrade bits when the corresponding
investment was already running, but PPO retained the sampled four-way action
and its log probability. Thus action 1 could execute as noop and action 3 could
execute as only-upgrade, while the reward was credited to the original action.
The producing distribution now masks invalid commands from the observation's
running flags in collection, PPO updates, entropy, and greedy evaluation.

### Investment churn remained a local optimum (ADR-0022)

Regressed producing policies requested both investments whenever possible,
creating about 500 completed and 150 rejected investment orders while consumer
completion fell below 300. Costs 0.5/1/2 did not prevent seed-5 collapse, so
this shaping is not enabled by default.

### The critic omitted the acting decision context (ADR-0023)

The centralized critic saw only producer state. It did not know which of the 12
agents was acting and could not see an ordering agent's incoming orders, even
though those orders determine bids and the next score. The critic now receives
global producer state, the padded current local observation, and a one-hot
acting-agent identity.

### Matched cross-seed stability control (ADR-0024)

Temperature `0.25` sharpens producing-logit gradients by `1/T = 4`, and the
valid-action mask concentrates probability on fewer actions. Once the critic
also received the missing decision context, the previously preferred target-KL
`0.015` remained too permissive: it fixed seeds 2 and 5 but seed 1 still
finished dead. Tightening it to `0.008` produced:

| seed | best return | final return | final death |
|------|-------------|--------------|-------------|
| 1 | 3725 | **3568** | 0 |
| 2 | 3514 | **3438** | 0 |
| 5 | 4395 | **4050** | 0 |

Configuration: 100 iterations, 1024 decisions per batch, four rollout
environments, no curriculum, producing temperature `0.25`, ordering
`log_std_init=-2.0`, target-KL `0.008`, greedy evaluation every ten iterations.

This is a final-criterion fix, not monotonic convergence. Seed 5 evaluates dead
from iterations 30 through 60 and recovers at iteration 70. Completed stochastic
rollouts also remain more volatile than greedy deployment.

## Repro

```
# validated defaults; repeat with seeds 2 and 5
python -m tokenomics.train.smoke_ppo --iters 100 --no-curriculum \
    --eval-every 10 --seed 1 --save /tmp/policy.pt

# historical unconstrained-update ablation
python -m tokenomics.train.smoke_ppo --iters 100 --no-curriculum \
    --no-target-kl --eval-every 10 --seed 1

# score a saved policy against the baseline table
python -m tokenomics.eval.signal_check --policy /tmp/policy.pt
```
