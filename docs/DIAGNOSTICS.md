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

## 18. A family fails (or passes) the headroom check, and capacity cuts do not move it

**Problem (plain English):** the gate's headroom check (design D10) compares a world family's
median best-baseline fill rate with the 0.95 ceiling. When a family is too close to the ceiling,
the obvious retune is to give it less capacity. In EXP-0044 that did nothing: dense-12 stayed at
about 0.955 from 1.0× down to 0.8× capacity. The competing explanations call for different levers:
- capacity binds, but the cut was too small: probe deeper cuts;
- the shortfall is not capacity at all but **warm-up / edge loss**: orders placed while the
  supply chain fills, or still in flight at the horizon, expire (TTL 2). Change what the check
  measures, or the demand pressure, not the capacity.

**Discriminating check (pod, cheap):** replay the family's strongest baseline on every
development gate world and split the undelivered units into cycles 1–20 and 21–100. It spans
many worlds at the full horizon and gates a decision, so it runs on a pod (CLAUDE.md compute
rule), not locally. Do it as **one h = 100 run per world with a snapshot of ordered and fulfilled
units at cycle 20**. Do NOT subtract an h = 20 run from an h = 100 run: on shocked worlds each
randomized consumer's horizon-length schedule comes from one shared random stream, so later
consumers get different demand at the two horizons (post-hoc review of EXP-0044). The EXP-0044
scripts used that subtraction and are valid on calm worlds only. They are kept as a record,
not as the recipe: `experiments/EXP-0044-gate-calibration-12-42/retune/warmup_share.py <size>`
and `warmup_share_family.py <family> <size> <member>`, run from `tokenomics/` as
`../.venv/bin/python ../experiments/EXP-0044-gate-calibration-12-42/retune/<script>`.

**Outcome branches:**
- **(a) Most of the shortfall is in cycles 1–20, and the fill rate after cycle 20 is near 1.**
  Warm-up dominates. This is the case found on EXP-0044's calm dense worlds: dense-12 had 54%
  of its shortfall in cycles 1–20 and a 0.975 median fill after; dense-42 had 85% and 0.995.
  Capacity is not binding (demand ≈ 6% of top-layer throughput), so a mild capacity cut cannot
  help. Shocked worlds were not diagnosed (unmatched subtraction).
- **(b) The shortfall is spread over the whole episode and the late fill rate drops as capacity
  is cut.** Capacity binds; a capacity scale is the right lever. Find the size of the cut with
  `gate --mode probe`.

**Seen in.** EXP-0044 retune probe (task 10.3), `experiments/EXP-0044-gate-calibration-12-42/retune/`.

---

## 19. A random baseline beats every hand-written rule

**Problem (plain English):** a baseline that routes orders at random sets the hindsight baseline
on most episodes. It beats rules that look at capacity or prices. That usually means something
is wrong with the rules or with the measurement, not that randomness is good. The competing
explanations call for different fixes:
- (a) **packing**: the rules pile sub-orders on the same producers, and atomic auctions reject
  them. Fix the rule's load awareness;
- (b) **temporal variation**: what helps is changing the receiving producer over time, which a
  deterministic rotation can do. Fix the rule, and no stochastic policy is needed;
- (c) **rule defects**: lowest-index tie-break, rounding of small orders, routes to producers
  that cannot make a unit, slot overflow. Fix that defect;
- (d) **investment-fund channel**: random routing of producers' own purchases changes who gets
  investment;
- (e) **scoring artifact**: the random baseline is a mean of replicates and the rules are single
  runs;
- (f) **split penalty**: the rules split small orders, and a split order completes only if every
  part does.

**Discriminating checks (pod for the full grid; local for the trace):**
1. Full grid with diagnostic routers that change one thing each (EXP-0048 driver):
   `python -m tokenomics.eval.routing_audit --workers 16 --source-commit {SOURCE_COMMIT}
   --out-dir EXP-NNNN-title` on a cpu16 pod (≈ 4 vCPU-h on registry v2). Read `edge.csv`. A
   router *recovers the edge* in a size × regime cell when its equal-weight world-mean gain over
   the best deterministic roster member is at least 0.8 of random's. It recovers it in *most
   cells* at 2 of the 3 random-dominated cells.
2. (a) `proportional_load_aware` against `proportional`; rejection reasons
   (`rejected_bid_units_*`, `lost_units_*`) in `routing_metrics.csv`.
3. (b) `round_robin` and `random_whole` against `random_fixed_episode` (decorrelated and
   mostly whole, but fixed over time: the cleaner time-variation contrast); `round_robin`
   against `proportional_whole` (both whole, but `proportional_whole` also herds every orderer
   onto the argmax producer: check `herfindahl_mean`); the `random_scaled_s` sweep read with its
   split degree. A balanced fixed assignment is not in the router set, so alternation over time
   and the long-run balance of a fixed assignment are not separated.
4. (c) local single-seed trace,
   `openspec/changes/add-m3-development-audit/probes/trace_21.py --world <w> --router <r>`
   (tie decisions, rounding, `zero_throughput_route_share`, `overflow_orders`), plus
   `even_split_rotating` against `even_split`.
5. (d) `random_consumer_only` / `random_producer_only` against `blend_0.5`;
   `fund_share_herfindahl_episode` against 1/N.
6. (e) the `edge_r1` and `random_worst_beats_best_det` columns of `edge.csv` (R = 1 against
   R = 5). (f) the `split_2x2` rows of `edge.csv` (whole/split × deterministic/random at matched
   split degree) and `wasted_units_*`.

**Outcome branches:**
- **(b) confirmed** (`round_robin` recovers; fixed assignments, `random_fixed_episode` and
  `proportional_whole`, do not): variation of the receiver over time is the lever. A deterministic rotating rule is a roster v3 candidate (design D3, a user decision),
  and stochastic deployment is not needed for the edge.
- **(c) confirmed** (a fixed variant recovers): the rule is defective. Fix it under a new name
  through roster v3 (D3); never edit the frozen rule in place.
- **(a)**, **(d)** confirmed: change the rule's load awareness, or read the producer-purchase
  routing together with role attribution (PoC-C).
- **(e)** confirmed (R = 1 removes the edge): no rule change; document the averaging in the gate.
- **(f)** confirmed (at the same kind of variation, routing whole instead of split recovers
  most of the edge; `split_2x2` rows and the matched contrasts in the EXP's RESULT): a
  whole-order rule is a roster v3 candidate, and a learned residual must be able to route an
  order whole. (f) can combine with (b): read the contrasts with one factor held fixed, not only
  the single-factor pattern "whole recovers, split does not".

**Seen in EXP-0048** (`experiments/EXP-0048-random-routing-diagnosis/`): (b), (c) and (f)
confirmed, and they combine. `round_robin` (rotating, whole) recovered 4.72 / 2.51 / 1.62 of the
edge (12-shocked, 42-calm, 42-shocked; +52 to +125 units), and beat `random_fixed_episode` by
89–241 units; `proportional_whole` (whole, not rotating, herding) recovered 0.46 / −3.34 /
−0.73. `even_split_rotating` recovered 0.69 / 0.81 / 0.84, against `even_split`'s −3.74 /
−4.65 / −1.07; a random leftover gains most of the same units. Random variation helps; rotation
helps somewhat more. Routing whole instead of split at the same variation is worth 0.78–4.04 of the
edge. (a), (d) and (e) were ruled out.

**Seen in EXP-0049** (`experiments/EXP-0049-informed-routing-and-trained-trace/`, follow-up to
(b)): on registry v2 (every product has two eligible producers), three rules that read the
producers' observed spare capacity do not improve on blind rotation; fanout ≥ 3 is untested.
Check:
`routing_audit --plan exp0049` (adds `two_choice_spare`, `spare_weighted_whole`,
`spare_rotating_whole`, and the trained subjects, sha256-pinned checkpoints); compare each with
its blind counterpart (`random_whole`, `round_robin`) on units, `receiver_change_share_*` and
`herding_hhi`, and count how often spare binds
(`openspec/changes/add-m3-development-audit/probes/spare_33.py`). Outcome: spare rarely binds
(skip 0–2.7% of decisions; `lost_units_consumer_no_fit` ≈ 0 for every router), and following a
shared capacity signal keeps each orderer on the same producer over time (receiver change
0.17–0.30 for the argmax rule against 0.50 for blind random), losing units as TTL expiries.
Read `receiver_change_share_*`, not `herding_hhi`: receiver change tracks units in every cell,
per-cycle herding does not (`round_robin` herds at 0.89–0.90 on 12-producer worlds and wins).
The trained portable orderer varies receivers like `round_robin` (0.64–0.67) without sampling,
but splits 29–40% of consumer orders; it beats `random` and loses to `round_robin` on shocked
worlds. Caveat: at
fanout 2, "pick two, take the roomier" is an argmax; testing it needs worlds with fanout ≥ 3.

**Seen in EXP-0050** (`experiments/EXP-0050-routing-vs-fanout/`, the same routers on worlds with
2–14 possible makers per product; registry v2 has 2): the diagnosis at fanout 2 understates how
much routing matters. Check: `routing_audit --plan exp0050` and its `fanout_summary.csv` /
`routing_metrics.csv` (split degree, `wasted_units_consumer`, `lost_units_consumer_ttl`,
`lost_units_consumer_no_fit` per cell). Outcome: at 42 producers `round_robin` beats the best
hand-written rule by +116 / +150 units at fanout 2 and +335 / +421 at fanout 7; routers that
split lose more with more makers (losses are TTL expiries, no-fit ≈ 0): `random` splits more at
fanout 14 and falls 268–296 units below `round_robin`, and `random_whole` recovers it
(`even_split`'s fanout-7 collapse comes mostly from its lowest-index tie-break). The spare-capacity routers still lose to blind choice. The trained portable orderer splits
as much as the failing rules but its split orders complete, and it clearly leads at 42
producers with fanout 4 and 14.
Open: why its parts complete. Its distillation target is input-aware (log of
`makeable_qty_now`, capacity and inputs in stock; `proportional` weighs capacity only). Check:
the teacher as a router (split by `makeable_qty_now`), its whole-order version, and a trace of
expired reservations by producer input stock.

---

## Unverified problems (plain-English inbox)

Humans: describe a suspected problem here in plain English — no tooling knowledge
needed. The agent turns it into a check and promotes it to a numbered entry once it has
discriminated a real case.

_(empty)_
