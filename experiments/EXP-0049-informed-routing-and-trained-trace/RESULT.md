# EXP-0049 — Informed routing and the trained orderer's routing

**Status: complete.** 3,968 episode rows over 128 world-episodes, 15 subjects (R = 5 for the
stochastic ones), horizon 100; registry v2 `7b09f536e988`, roster v2 `5050508ecb25`. Run commit
`71d5c3f93c9b1c956db79c11603d5c31e7c10ccb` (clean tree). Produced by
`python -m tokenomics.eval.routing_audit --plan exp0049` (see `spec.json`). Change
`add-m3-development-audit`, task 3.3, design D2 "Informed spreading routers and trained-agent
trace".

**Verdict, in two answers.**

1. **Can a rule that looks at the producers' spare capacity route better than the blind
   `round_robin`? Not the three rules tested, on these worlds, and the reason is diagnosed.**
   In registry v2 every product has exactly two possible producers, so each rule only ever
   chooses between two; informed spreading in general, for example "pick two of many, take the
   roomier", is untested and needs worlds with three or more producers per product. On these
   worlds spare capacity is almost never what limits an order: an order is 1–2 units, the
   less-loaded of its two possible producers has 13–22 units of room on average (fewer than
   2–10 units in one decision of ten), and over all 128 world-episodes almost no consumer
   unit is lost because a producer had no room. So the number tells an ordering agent almost
   nothing about where its order will succeed. Following it does change one thing: an agent
   that always prefers the roomier producer keeps sending to the same producer cycle after
   cycle, which is the opposite of the rotation that won EXP-0048. The rule that follows it
   hardest changes its producer on only 17–30% of decisions (blind random choice: 50%) and
   loses 15–152 units per episode against blind random choice, 94–97% of them as extra
   reservations that expire. The rule that only uses it to skip a full producer is the same as
   `round_robin`, because the skip almost never happens (0–2.7% of decisions).
2. **Did the trained orderer learn random's trick? Partly, and without randomness.** It changes
   the receiving producer from one decision to the next about as often as `round_robin` does,
   which was the main lever in EXP-0048. It is deterministic, so this is not "stochasticity"
   (EXP-0006B's reading). It splits fewer orders than `proportional`, the rule it was distilled
   from, but still about a third of consumer orders. It beats `random` on 126 of 128 episodes
   and every roster member in every cell. `round_robin` still beats it on the shocked worlds.

## How to read this

**The setting.** In every world here, each product can be made by exactly two producers. Every
cycle, many ordering agents (one per buyer and product: a "stream") each decide which of the two
producers gets each of their orders. EXP-0048 found what wins: send each order **whole** to one
producer (a split order completes only if every part does), and **change which producer** from
one cycle to the next. `round_robin` does both with a fixed rotation: cycle 1 → producer A,
cycle 2 → producer B, cycle 3 → producer A, and so on, with each agent starting at its own
offset so that agents do not all pick A at the same time.

**The two questions** (the user's): `round_robin` is blind; it never looks at the producers.
Can a rule that does look (at spare capacity) do better? And the trained portable orderer: did
it beat the hand-written rules by learning the same trick?

**The routers tested.** Each sends every order whole to one producer.
- `two_choice_spare`: pick two eligible producers at random and send to the one with more spare
  capacity. Blind counterpart: `random_whole` (pick one at random).
- `spare_weighted_whole`: pick one producer at random, with odds proportional to its spare
  capacity (a producer with twice the room is twice as likely). Counterpart: `random_whole`.
- `spare_rotating_whole`: `round_robin`, but skip a producer that does not have room for the
  whole order. Counterpart: `round_robin`.

"Spare capacity" is what the ordering agent itself observes: how many units of the product a
producer could still accept this cycle, given its current capacity minus what its pending
reservations hold (`accept_qty_now` in the agent's view). It does not see other agents'
orders placed earlier in the same cycle; no agent does.

**The trained subjects.** `trained_orderer` is the portable orderer (EXP-0012
`orderer_scorer.pt`) with the same producer as every router (`entity_planner`), so its routing
compares directly with theirs. `portable` is the full portable stack with its own trained
producer (EXP-0035 `producer_candidate.pt`). Both are deterministic: same world, same routing.

**Terms.**
- *Units*: consumer units delivered per episode (higher is better), as an equal-weight mean over
  worlds.
- *Gain*: units minus the best deterministic roster v2 member's units on the same episode.
- *Edge*: `random`'s gain. *Recovered fraction*: gain ÷ edge (EXP-0048's rule: a router
  "recovers the edge" in a cell at 0.8 or more, and "in most cells" at 2 of the 3 cells where
  random dominates: 12-shocked, 42-calm, 42-shocked).
- *Split share*: the share of consumer orders sent to more than one producer (0 = never split).
- *Receiver change*: for each stream, the share of consecutive routing decisions that went to a
  different producer than the decision before (0 = always the same producer; `round_robin` with
  one order per cycle and two producers = 1).
- *Herding*: for each product and cycle with at least two orders, how concentrated the orders
  are on one producer, as a Herfindahl index weighted by orders: 1.0 = every order went to the
  same producer, 0.5 = evenly over the two. Random choice between two producers gives about
  0.73–0.74 here. It turned out not to explain units (below): `round_robin` herds at 0.89–0.90
  on 12-producer worlds and is still the best whole-order router there.
- *Wasted units*: units already delivered as part of a split order that then failed.
- *TTL losses*: units of accepted orders that expired waiting for inputs (the main way units are
  lost, EXP-0048).

## Result

### Question 1: informed routers against their blind counterparts

Units per episode (equal-weight world means), with the difference to the blind counterpart.
"Worlds" = worlds where the informed router has more units, "episodes" = episodes better /
equal / worse.

| cell | `random_whole` | `two_choice_spare` (diff; worlds; episodes) | `spare_weighted_whole` (diff; worlds; episodes) | `round_robin` | `spare_rotating_whole` (diff; worlds; episodes) |
|---|---|---|---|---|---|
| 12-calm | 514.9 | 470.8 (−44.1; 0/8; 1/1/22) | 510.5 (−4.4; 3/8; 9/1/14) | 539.5 | 540.8 (+1.2; 3/8; 9/12/3) |
| 12-shocked | 478.3 | 463.0 (−15.3; 0/4; 2/0/38) | 480.6 (+2.4; 3/4; 27/1/12) | 510.8 | 511.4 (+0.6; 2/4; 7/31/2) |
| 42-calm | 1726.4 | 1627.6 (−98.8; 0/8; 0/0/24) | 1705.9 (−20.5; 1/8; 3/0/21) | 1754.8 | 1747.8 (−7.0; 4/8; 12/3/9) |
| 42-shocked | 1646.6 | 1495.0 (−151.6; 0/4; 0/0/40) | 1607.8 (−38.7; 0/4; 1/0/39) | 1673.1 | 1673.5 (+0.4; 3/4; 19/0/21) |

Routing behaviour (42-shocked, the other cells read the same way; receiver change over
consumer orders, herding over all orders of a product, consumer and producer):

| router | receiver change (consumer) | herding (all orders) | units |
|---|---|---|---|
| `random_whole` | 0.50 | 0.74 | 1646.6 |
| `two_choice_spare` | 0.19 | 0.99 | 1495.0 |
| `spare_weighted_whole` | 0.45 | 0.79 | 1607.8 |
| `proportional_whole` (EXP-0048's "always the bigger one") | 0.19 | 1.00 | 1492.4 |
| `round_robin` | 0.90 | 0.72 | 1673.1 |
| `spare_rotating_whole` | 0.90 | 0.73 | 1673.5 |

**`two_choice_spare` turned into "always the producer with more room".** Every product in these
worlds has exactly two eligible producers (checked over all 24 worlds: 648 of 648 products).
"Pick two at random" then always picks both, so the rule always takes the producer with more
spare capacity. That is practically `proportional_whole` (within 0.6–5 units of it in every
cell). The "power of two choices" idea was therefore not tested; it needs worlds with three or
more eligible producers, which registry v2 does not have. This was missed when the router was
designed (task 3.2). What the arm does show is what happens when every agent follows the same
capacity signal: the agents stop changing receivers (0.17–0.30 against `random_whole`'s 0.50),
lose more units to expiring reservations (42-shocked: 394.9 against 251.4), and lose 15–152
units per episode against blind random choice, in every world. They also all pick the same
producer in a cycle (herding 0.99), but that is not what costs the units: no bid is turned
away for lack of room (consumer units lost to `no_fit` ≤ 0.1 per episode for every subject).

**`spare_weighted_whole` varies a little less, and loses a little.** The larger-spare producer
of a pair holds 60–69% of the pair's spare on average (single-episode check below), so it gets
a 60–69% chance instead of 50%, and it stays the larger one from cycle to cycle. Receiver change
falls from 0.50 to 0.45–0.49 and TTL losses rise (42-calm 225.4 against 204.9, 42-shocked 290.2
against 251.4). It loses 20.5 and 38.7 units per episode at 42 producers (1 of 8 and 0 of 4
worlds better) and is about even at 12 (−4.4, +2.4).

**`spare_rotating_whole` is `round_robin`.** Its differences to `round_robin` are +1.2, +0.6,
−7.0 and +0.4 units per episode (0.0–0.4%), with most 12-producer episodes identical (12 of 24,
31 of 40). By the letter of the pre-registered rule it "beats its counterpart" (more units in 2
of the 3 dominant cells: +0.6 and +0.4). The margins are far inside the episode-to-episode
spread (42-shocked: 19 episodes better, 21 worse), and the reason is direct: its skip almost
never fires. We do not read this as "producer data helps". The paired differences per cell (both routers
are deterministic, so the spread is across worlds and episodes) are +1.25 / +0.58 / −7.00 /
+0.38 units with world-clustered standard errors 2.80 / 0.55 / 6.49 / 4.31 (t = 0.45 / 1.05 /
−1.08 / 0.09) and per-episode standard deviations 7.6 / 2.6 / 17.5 / 17.2. No difference is
distinguishable from zero. This departs from the literal pre-registered 2-of-3 rule, which
counts the sign of the mean only.

**Why the signal does not help: a single-episode check** (`probes/spare_33.py`, first episode
of three worlds, `spare_rotating_whole` playing; per routing decision with two eligible
producers):

| world | decisions (consumer / producer) | skip fires | both producers too full | spares tied | larger producer's share of the pair's spare | mean order size | smaller spare (mean) |
|---|---|---|---|---|---|---|---|
| 12-shocked-00 | 300 / 677 | 0.0% / 0.0% | 0% | 1.7% / 1.0% | 0.60 / 0.69 | 1.81 / 1.84 | 22.1 / 13.4 |
| 42-shocked-00 | 1,104 / 2,515 | 0.5% / 2.7% | 0% | 5.2% / 3.1% | 0.64 / 0.66 | 1.62 / 1.67 | 13.2 / 17.2 |
| 42-calm-00 | 1,100 / 2,508 | 0.1% / 0.4% | 0% | 4.0% / 2.1% | 0.61 / 0.67 | 1.92 / 1.89 | 16.7 / 17.8 |

Orders are 1–2 units and the smaller spare is 13–22 units on average (10th percentile 2–10
units; the order is at least as large as the smaller spare on 0.3–10.1% of decisions, most
often for producer orders on 42-shocked), so capacity at the moment of routing rarely decides
whether an order succeeds. The full run agrees: over all 128
world-episodes, consumer units lost to bids turned away for lack of room (`no_fit`,
`displaced`) are at most 0.15 per episode for every subject, cell means. Units are lost mostly
to reservations that expire waiting for inputs (EXP-0048), which spare capacity does not
measure.

**What does track units: receiver change, not herding.** Across the six whole-order routers
(`random_whole`, `two_choice_spare`, `spare_weighted_whole`, `proportional_whole`,
`round_robin`, `spare_rotating_whole`), the rank correlation of units with receiver change is
0.89 / 0.83 / 1.00 / 0.94 (12-calm / 12-shocked / 42-calm / 42-shocked); with herding it is
−0.49 / −0.37 / −0.94 / −0.94. Herding follows units only at 42 producers, and `round_robin`
contradicts it at 12 (herding 0.89–0.90, best whole-order router). How sticking to one producer
over time turns into expiring reservations (presumably by concentrating input purchases) is not
traced.

### Question 2: what the trained orderer learned

Routing behaviour per cell, the trained orderer next to the routers it could resemble (from
left: 12-calm / 12-shocked / 42-calm / 42-shocked; all columns over consumer orders except
herding, which counts all orders of a product, consumer and producer):

| subject | split share | receiver change | herding (all orders) | wasted units | TTL losses | units |
|---|---|---|---|---|---|---|
| `random` (R = 5) | 0.09 / 0.08 / 0.07 / 0.07 | 0.58 / 0.57 / 0.56 / 0.56 | 0.73 / 0.73 / 0.73 / 0.73 | 3.9 / 5.5 / 11.8 / 15.8 | 59.0 / 81.5 / 196.9 / 257.5 | 513.4 / 469.5 / 1723.3 / 1625.3 |
| `round_robin` | 0 / 0 / 0 / 0 | 0.65 / 0.88 / 0.64 / 0.90 | 0.90 / 0.89 / 0.73 / 0.72 | 0 / 0 / 0 / 0 | 37.4 / 46.8 / 178.5 / 228.7 | 539.5 / 510.8 / 1754.8 / 1673.1 |
| `proportional` | 0.57 / 0.59 / 0.52 / 0.45 | 0.40 / 0.26 / 0.43 / 0.21 | 0.79 / 0.82 / 0.83 / 0.85 | 22.3 / 40.4 / 110.6 / 135.4 | 41.1 / 72.1 / 250.1 / 318.7 | 513.0 / 443.1 / 1569.9 / 1438.0 |
| **`trained_orderer`** | 0.40 / 0.31 / 0.35 / 0.29 | 0.67 / 0.66 / 0.67 / 0.64 | 0.82 / 0.89 / 0.86 / 0.86 | 8.1 / 12.2 / 29.0 / 39.8 | 30.4 / 62.2 / 109.0 / 201.5 | 539.4 / 482.5 / 1798.8 / 1658.8 |
| `portable` | 0.40 / 0.32 / 0.35 / 0.29 | 0.67 / 0.66 / 0.68 / 0.64 | 0.82 / 0.89 / 0.86 / 0.86 | 8.1 / 12.1 / 29.4 / 38.5 | 30.4 / 62.6 / 108.8 / 202.2 | 539.4 / 482.3 / 1799.5 / 1659.4 |

Its units against the others, per episode (difference in units; worlds better / all; episodes
better / equal / worse):

| comparison | 12-calm | 12-shocked | 42-calm | 42-shocked |
|---|---|---|---|---|
| `trained_orderer` − `random` | +26.0 (8/8; 24/0/0) | +13.0 (4/4; 40/0/0) | +75.5 (8/8; 24/0/0) | +33.5 (4/4; 38/0/2) |
| `trained_orderer` − `proportional` | +26.4 (5/8; 15/0/9) | +39.4 (4/4; 38/1/1) | +228.9 (8/8; 24/0/0) | +220.8 (4/4; 40/0/0) |
| `trained_orderer` − `round_robin` | −0.1 (2/8; 6/0/18) | −28.3 (0/4; 0/0/40) | +44.0 (7/8; 21/0/3) | −14.2 (0/4; 11/0/29) |
| `portable` − `trained_orderer` | +0.0 (0/8; 0/24/0) | −0.2 (1/4; 1/38/1) | +0.8 (1/8; 3/15/6) | +0.6 (3/4; 11/20/9) |

Recovered fraction of random's edge: `trained_orderer` 2.17 / 4.61 / 1.44 in the three dominant
cells (3 of 3), `portable` 2.15 / 4.65 / 1.44. Against the per-episode hindsight baseline (the
best roster v2 member on that episode, `random` included) `trained_orderer` has more units on
12 of 24, 37 of 40, 21 of 24 and 38 of 40 episodes, a mean ratio of 1.011 / 1.022 / 1.037 /
1.021. That matches EXP-0045's 1.011–1.041 for the portable stack on registry v1.

**Reading, per the pre-registered branches.** The discriminating metrics point different ways,
so the answer is "partly":
- *Receiver change* 0.64–0.67, against `random`'s 0.56–0.58 and `proportional`'s 0.21–0.43, and
  about `round_robin`'s in the calm cells (0.64–0.65). It varies the receiver over time, the main
  lever of EXP-0048. It is deterministic, so the variation does not come from sampling.
- *Split share* 0.29–0.40: below `proportional`'s 0.45–0.59, far above `random`'s 0.07–0.09 and
  `round_robin`'s 0. It wastes 8–40 completed units per episode that a whole-order router does
  not.
- *Herding* 0.82–0.89: like `proportional` (0.79–0.85), above `random` (0.73). Herding does not
  explain units in this run (`round_robin` herds as much at 12 producers and wins), so this is
  reported, not counted against it.

It also loses fewer units to expiring reservations than any router on 12-calm and 42-calm
(42-calm: 109.0 against `round_robin`'s 178.5) and than any but `even_split_rotating` on
42-shocked (201.5 against 186.6); on 12-shocked three routers lose fewer (`spare_rotating_whole`
46.1, `round_robin` 46.8, `even_split_rotating` 57.1, against its 62.2). Where its variation comes from is not traced:
the scorer reads features of its own recent routing (`recent_routed_qty` and similar), which
could make it alternate, but this run does not test that. Why it loses to `round_robin` on
shocked worlds (−28.3 and −14.2 units) is also not diagnosed; its remaining splits are one
candidate, not tested. The portable producer changes almost nothing: `portable` equals
`trained_orderer` on every 12-calm episode and within 0.8 units per cell elsewhere.

### Reproduction checks

Every reference row, 2,304 of them (the six roster members with `random`'s 5 replicates, and
`round_robin`, `even_split_rotating`, `proportional_whole`, `random_whole` × 5), equals
EXP-0048's units and digests for the same world-episode and replicate. Every `portable` row on
the 80 shocked world-episodes equals EXP-0046's `candidate` units and digests (80 of 80). So the
new metrics did not change any episode, and the trained path loads the right checkpoints.

## Conclusions

- **Question 1, diagnosed for the three rules tested.** On registry v2, where every product has
  two eligible producers, the three spare-capacity rules do not beat blind spreading of whole
  orders. Informed spreading in general, including the power of two choices, is untested at
  fanout ≥ 3. Cause, with evidence: spare capacity is not the binding constraint at routing
  time (orders of 1.6–1.9 units against 13–22 spare units on average, 10th percentile 2–10; the
  skip fires on 0–2.7% of decisions; consumer units lost to bids turned away for lack of room
  ≤ 0.15 per episode in every cell), so it carries little information about success. Following
  it makes an agent keep choosing the same producer over time (receiver change 0.17–0.30 for
  the argmax rule, 0.45–0.49 for the weighted draw, against 0.50 for blind random choice and
  0.64–0.90 for `round_robin`), and the units are lost as expiring reservations (−15 to −152
  and −4 to −39 per episode). Receiver change tracks units in every cell (rank correlation
  0.83–1.00 over the whole-order routers); per-cycle herding does not (`round_robin` herds at
  0.89–0.90 on 12-producer worlds and wins). How sticking to a producer leads to expiring
  reservations is not traced. `spare_rotating_whole` meets the literal 2-of-3 "beats" rule by
  +0.6 and +0.4 units; with world-clustered t of 1.05 and 0.09 we treat that as no effect, a
  departure from the literal rule.
  Limitation: `two_choice_spare` degenerated to an argmax because every product has two
  eligible producers; the power-of-two-choices idea needs worlds with fanout ≥ 3.
- **Question 2, partly diagnosed.** The trained portable orderer learned to vary the receiving
  producer over time (receiver change 0.64–0.67, like `round_robin`), deterministically, and to
  split less than `proportional` (0.29–0.40 against 0.45–0.59). It did not learn to route
  whole. It beats `random` on 126 of 128 episodes and every roster v2
  member in every cell, and loses to `round_robin` on shocked worlds. What drives its variation,
  and why it loses there, are not diagnosed.
- **What changes for roster v3 (design D3, the user's decision).** The informed routers add
  nothing over `round_robin`, so they are not candidates; `round_robin` (and the
  `even_split_rotating` fix) remain the candidates EXP-0048 named. Adopting `round_robin` would
  raise the bar above the trained orderer on shocked worlds (12-shocked: 0 of 40 episodes
  better, −28.3 units; 42-shocked: 11 of 40, −14.2) but not on 42-calm (+44.0).
- **What changes for M4's starting point.** The portable orderer already gets the time-variation
  part of the edge without sampling, so M4 does not need stochastic deployment for it. M4's
  residual does not need spare capacity as a routing input at these sizes. The measured gaps to
  close is whole-order routing (split share 0.29–0.40 → 0, 8–40 wasted units); herding is not
  shown to matter. Whether routing whole recovers the shocked-world gap to `round_robin` is the
  next question, not answered here.

## Details

**Command** (repo root; checkpoints staged under `tokenomics/_pod_ckpts/`, deleted after):

```
python3 run_pod_training.py --create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12 \
  --vcpu 16 --mem 32 --disk 30 --pod-name exp0049-informed-routing --verify-source-commit \
  --train "-u -m tokenomics.eval.routing_audit --plan exp0049 \
    --prod-ckpt /workspace/tokenomics/_pod_ckpts/producer_candidate.pt \
    --orderer-ckpt /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt \
    --workers 16 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0049" \
  --pull "/workspace/tokenomics/runs/exp0049:./experiments/EXP-0049-informed-routing-and-trained-trace"
```

`--pull` placed the run under `exp0049/`; its files were moved up unchanged, and `episodes.csv`
into `artifacts/` (git-lfs). A resume in this directory would need it moved back.

**Verification** (`/srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/exp0049_check.py`, run from the repo root as
`python3 openspec/changes/add-m3-development-audit/probes/exp0049_check.py
experiments/EXP-0049-informed-routing-and-trained-trace experiments/EXP-0048-random-routing-diagnosis
experiments/EXP-0046-m2-exit-shocked-champion`; its own arithmetic, not `routing_audit`'s):
- `contract_failures.csv` header-only; `spec.json` status `complete`, commit as above, plan
  `exp0049`.
- 3,968 rows, every planned (world, episode, subject, replicate) once, none extra; one demand and
  shock digest pair per world-episode (128).
- 2,304 reference rows equal EXP-0048 (units and digests), 0 mismatches; 80 of 80 `portable`
  rows equal EXP-0046's `candidate` rows, 0 mismatches.
- Checkpoint pins: producer `8b51b77cb5ac…`, orderer `d25388bfa33e…`; equal to EXP-0046's
  `portable_producer` / `portable_orderer` pins and to the sha256 of the staged files.
- `edge.csv` recomputes from `artifacts/episodes.csv` (equal-weight world means, 0.8 / 2-of-3
  rule): 60 cell rows and 15 rule rows, 0 mismatches.
- The single-episode spare check: `probes/spare_33.py` (run from `tokenomics/`), local, 10 s.

**Compute.** Runpod secure cloud, cpu5c, 16 vCPU / 32 GB, `runpod/base:0.7.0-ubuntu2004`,
python 3.12, torch 2.15.0+cpu. Pod `weo1ikm87rf6ji`, created and terminated by the launcher.
- Pod life: created after 14:49:43 UTC and terminated before 14:59:05 UTC (local clock, stamps
  around the launcher): ≤ 9.4 min ≈ **2.5 vCPU-h billed**.
- Execution (pod clock): `spec.json` `started` 14:50:30 to the last row at 14:58:57 ≈ 8.5 min ≈
  2.25 vCPU-h. `sizing.md` §EXP-0049 expected 2.15 vCPU-h and 8.0 min.
- M3 pod budget: ≈ 6.9 of 40 vCPU-h used (EXP-0048 4.4 + this 2.5). `runpodctl pod list`
  returned `[]` afterwards.

**Files.** `spec.json` (pins incl. plan and checkpoint sha256, provenance),
`artifacts/episodes.csv` (gate row format plus `trace_*` columns, incl. the new
`receiver_change_share_*`, `herding_hhi`, `herding_groups`; git-lfs), `routing_metrics.csv`
(per size × regime and subject: units, death rate, every tracer scalar, equal-weight world
means), `edge.csv` (levels `episode`, `cell`, `rule`, `split_2x2`), `contract_failures.csv`.
