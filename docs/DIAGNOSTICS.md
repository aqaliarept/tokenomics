# Diagnostics registry — known problems → verification checks

Companion to the CLAUDE.md **"Root cause before next strategy"** rule. When an
experiment fails or underperforms, scan this registry FIRST: if the symptom matches a
known problem, run the listed check instead of inventing a new diagnostic or jumping to
the next strategy. Each entry states the problem in plain English, the discriminating
check (tool + command shape), and what each outcome means for the next move.

Maintenance:
- New failure cause diagnosed with evidence → add an entry (problem statement in plain
  English, check, outcome branches, EXP reference).
- Humans may add a plain-English problem statement to **"Unverified problems"** at the
  bottom without knowing the check; the agent designs and attaches the check later.
- An entry is evidence-backed: it must cite at least one experiment or task where the check
  discriminated between competing causes.

Run profile: checks marked **pod** follow the CLAUDE.md compute rule (real runs on
Runpod); **local** checks are quick one-shot diagnostics.

History: entries 1–16 were removed from this branch by the `cleanup` commit 18c717d and are
preserved in git (`git show 18c717d^:docs/DIAGNOSTICS.md`). Numbering continues from them.

---

## 17. 120-producer joint-stack episodes cost two orders of magnitude more than 12-producer ones

**Problem (plain English):** under the joint-stack contract (entity producer observation +
product-stream orderers, `eval/joint_stack.joint_stack_env`), the cost of one episode
cycle grew about 150× (sparse-wide) to 175× (dense) from 12 to 120 producers. Ten times
the producers should not cost that much. The legacy contract on the same worlds grew only
about 30×. Projected to the M1 gate runs, this made EXP-0044/0045 cost about 1,600 + 900
vCPU-hours (task 8.3, `openspec/changes/add-measurement-gate-harness/sizing.md`). The
competing explanations call for different responses:
- the simulation itself (auction, production) scales badly. That is a domain-model problem
  and would cost the legacy contract too;
- a per-agent observation cost. Fix the observation builder;
- policy cost or harness overhead (PettingZoo bookkeeping, baselines). Leave the domain
  alone.

**Discriminating check (local, about a minute):** cProfile one joint-stack episode at a
short horizon (h ≤ 4) on a dense-120 and a sparse-120 family world. BLAS pinned to 1
thread, idle box (entry 16). Sort by cumulative time. The registry holds no 120-producer
worlds since task 8.7 (gate sizes 12 and 42), so the check builds the world directly.
```
python -c "import cProfile, pstats; \
from tokenomics.eval.families import build_family_config; \
from tokenomics.eval.joint_stack import joint_stack_env; \
from tokenomics.eval.runner import run_episode; \
from tokenomics.eval.baselines import PRODUCING_BASELINES as P, ORDERING_BASELINES as O; \
env = joint_stack_env(build_family_config(1000, family='dense', producers=120, regime='calm'), 3); \
cProfile.run('run_episode(env, P[\"entity_planner\"], O[\"proportional\"], 0)', '/tmp/p'); \
pstats.Stats('/tmp/p').sort_stats('cumulative').print_stats(25)"
```

**Outcome branches:**
- **(a) `env/aec_env.observe` → `domain/orderer_view.build_product_stream_view` dominates.**
  This is the cause found here. Every product-stream orderer (≈300 sparse, ≈500 dense at 120)
  rebuilt every producer's rows (`_producer_stats`, `_producer_product_stats`). Each
  producer total scanned every telemetry EMA key (`telemetry.sum_by_producer`), so the cost
  was streams × producers × keys per cycle. Sparse-120 at h = 3 before the fix: observe
  96%, stream view 86%, 1,004,640 `sum_by_producer` calls.
  Fixed in task 8.4, with byte-identical observations, actions and episode statistics:
  - per-producer EMA totals indexed once per fold;
  - producer rows memoized while no producer row can change (`StreamViewCache`);
  - one shared row for undeclared, no-history (producer, product) pairs;
  - env producer infos built only for the encoders that read them.
  The guard is `tests/test_orderer_view_perf_equivalence.py`.
  If this branch shows up again, look for a new per-stream × per-producer computation
  added to the view.
- **(b) `System.complete_cycle` (auction / production) dominates.** The domain model scales
  badly. Not this case: env step was 2% of the episode before the fix.
- **(c) Policy calls or PettingZoo bookkeeping (`_accumulate_rewards`, `_clear_rewards`)
  dominate.** Harness overhead. After the 8.4 fix these are the remaining tail (about 20%
  of a dense-120 episode), next to the observation encode (about 30%).

**Seen in.** Task 8.3 sizing (`sizing.md` §5) and task 8.4, before → after, single process
on an idle box. Episode time at h = 12, `entity_planner/proportional`:

| world | before | after |
|---|---|---|
| sparse-120 | 28.8 s | 3.4 s |
| dense-120 | 53.3 s | 5.2–5.4 s |

The portable subject at h = 8 went from 40.0 s to 4.6 s (sparse) and from 47.4 s to 7.0 s
(dense).

---

## Unverified problems (plain-English inbox)

Humans: describe a suspected problem here in plain English — no tooling knowledge
needed. The agent turns it into a check and promotes it to a numbered entry once it has
discriminated a real case.

_(empty)_
