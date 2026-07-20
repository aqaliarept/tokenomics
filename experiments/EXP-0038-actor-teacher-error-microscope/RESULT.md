# EXP-0038 — Actor/Teacher Error Microscope

**Date:** 2026-07-19 · **Driver:** `tokenomics.eval.value_router_errors` · **Reproduce:** see [`spec.json`](spec.json)

**Verdict: SCOPED DIAGNOSIS, not final closure.** The current EXP-0037 actor row is missing
important context: identical 7-feature producer rows receive teacher labels spanning up to 90
units, so the actor cannot fit this target as currently observed (loss flat at ~12.9). This does
**not** prove that every deployable observation would fail. EXP-0038A continues with an augmented
observation audit before any new actor is trained.

## Question

EXP-0037 showed that the value-residual orderer did not beat fixed `blend_0.5`, and its
current producer-only feature row fit the marginal-value labels poorly. What exactly is the
actor getting wrong relative to the teacher?

This experiment is diagnostic-only. It should compare, on the same routing decisions:

```text
teacher marginal values per eligible producer
actor predicted residual scores per eligible producer
state/order/backlog facts visible at that decision
```

The goal is to decide whether the next fix is observation enrichment, a different target, or
stopping learned orderer work.

## Method

Build an error dump over the EXP-0036/0037 gate tasks (`s29`, `s77`, `s45`):

1. Reuse the EXP-0037 `blend_0.5` trajectory and `marginal_value_labels` teacher.
2. Fit or load a residual actor under a fixed seed.
3. For every labeled order decision, log teacher labels and actor scores for each eligible
   producer.
4. Rank mistakes by teacher-regret of the actor's pick.
5. Dump producer-feature deltas and order/backlog metadata for the largest mistakes.
6. Run an alias check: identical or near-identical actor feature rows with conflicting teacher
   labels.

## Required Outputs

```text
decision_rows.csv
producer_rows.csv
top_mistakes.csv
alias_groups.csv
summary.csv
```

## Read Order

1. If actor scores disagree with teacher mainly when missing order/backlog/TTL fields differ,
   observation enrichment is justified.
2. If actor features are identical but teacher labels conflict, the current feature set aliases
   the target; enrich only if the dump identifies the missing fields.
3. If teacher labels are unstable even after visible order/backlog facts are considered, the
   marginal-value target is still noisy and another actor is not licensed.

## Status

Complete for the current-row microscope. Driver `tokenomics.eval.value_router_errors` built (TDD, 15 tests) and run on a fresh
CPU pod over the three gate tasks. Dump pulled to [`dump/`](dump/) (`decisions.csv`,
`producers.csv`, `top_mistakes.csv`, `alias_groups.csv`, `summary.csv`, `spec.json`); 13,940
labeled decisions, seeded actor rerun (matches the EXP-0037 gate actor: seed 0, delta 1.0,
lookahead 8, eval-seeds 3, horizon 100, epochs 300).

## Results

Headline (`dump/summary.csv`): actor top-1 matches the teacher on **48.7%** of all decisions, mean
teacher regret **1.32** units (max 54), across 13,940 decisions.

Read in three steps — each answers one branch of the plan's Read Order:

**1. Most decisions have nothing to learn.** 69.4% of decisions have a **flat teacher** — every
eligible producer earns the same marginal value (regret 0 by construction; `blend_0.5` is already
teacher-optimal there). The marginal-value "value bend" signal is sparse: only **30.6% (4,265)** of
decisions are genuine choices where the actor can be wrong. On those, top-1 is only **0.451** and
mean regret **4.32**; regret is concentrated (top 5% of decisions hold 65% of all regret).

**2. The misses are NOT pure current producer-row aliasing.** Among the 1,975 real mistakes (actor ≠
teacher, regret > 0), the teacher-best and actor-best producers have **identical visible feature
rows only 3.9%** of the time (pure local aliasing). In **96.1%** the two producers already differ on
a feature the actor sees (makeable ~70%, headroom ~65%, reserved ~53%, free-cap ~28%, input ~37%).
That rules out the narrow story that the actor is usually choosing between two locally identical
producer rows. It does **not** rule out missing order/system context: the teacher's label is a
rollout difference, so other orders, other ordering agents, downstream bottlenecks, and global
demand can still change the value of the same local row. The largest mistakes are mostly
single-order local queues (`backlog_size = 1`), so local backlog alone is unlikely to explain them.

**3. The teacher target is not a function of the current actor row.** For every exact actor
feature row seen ≥ 5 times, the teacher label it receives varies wildly across contexts: median
range **10.0** units, mean **14.3**, and **66.9%** of repeated rows have a label range > 1 unit;
worst offenders span **90 units** (std ~9–25) for byte-identical actor features. A single logit
bump swings downstream fulfilment by ±50 units (`dump/top_mistakes.csv`: teacher pick +0, actor
pick −54). The marginal value of a logit nudge is a global, path-dependent quantity — a difference
of two full multi-agent rollouts — so the 7-feature local producer row is too lossy. This is why
the residual fit loss is flat at ~12.9 under the current row. A post-run audit also showed that
adding logged decision context (especially product and quantity) reduces, but does not eliminate,
fixed-feature label conflict. That leaves a real open question for EXP-0038A: which deployable
system/context observations close the conflict, and which teacher information is only available by
rollout emulation?

Consistent with EXP-0009 (value-from-`global_state` did not generalize across these topologies;
held-out R² negative). The finer-grained result here: even *in-sample*, the per-decision
marginal-value label is not a stable function of the **current EXP-0037 actor row**.

## Conclusions

- **Cause diagnosed for the current actor:** the EXP-0037 actor fits the teacher badly because its
  7-feature producer row aliases the marginal-value label. Identical actor rows map to teacher
  labels differing by up to 90 units.
- **Concrete next action (DoD):** continue with
  [`EXP-0038A`](../EXP-0038A-missing-observation-audit/RESULT.md). Add candidate deployable
  observation blocks and rerun the target-learnability check before training any new actor.
- `blend_0.5` remains the orderer floor and EXP-0037's deployed residual gate remains failed. The
  unresolved question is whether a richer **deployable** observation can make the teacher label
  learnable, or whether the teacher's advantage requires rollout emulation that cannot be used in
  the real application.
