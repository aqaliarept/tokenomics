# EXP-0054 — PoC-C: which role to train first, on mined shocked worlds

**Status: complete, valid. Not an acceptance read.** `python -m tokenomics.eval.teacher_audit
--poc c`, run commit `4e96255becbef380c0ba9ea515a2afd525314c04` (task 9.1 of
`add-m3-development-audit`, design D6), on atlas v2 = EXP-0052 `reclassified/atlas_v2.json`
(content hash `220886a7cf12…`, file SHA-256 `7517314f297a…`), portable producer EXP-0035
`producer_candidate.pt` (SHA-256 `8b51b77cb5ac…`), trained orderer EXP-0012 `orderer_scorer.pt`
(SHA-256 `d25388bfa33e…`). 12 episode rows = 3 shocked world-episodes at 12 producers × 4 arms,
horizon 100. 0 contract failures, 0 deaths. Fresh 4 vCPU secure pod, ≈ 2.6 vCPU-h billed
([Details](#details)).

## In plain English

**The question.** Our deployed agent is the *portable stack*: a trained producer (EXP-0035) and
a trained orderer (EXP-0012). On hard worlds it falls short of a *teacher stack*: the
hand-written producer `entity_planner` together with the PoC-A teacher `d2`, a slow lookahead
search that decides who gets each order (EXP-0053). M4 will train one of the two roles toward
its reference first. This experiment asks which role accounts for more of the shortfall.

**The four arms (the role swaps).** Each world is played four times with the same demand and
shocks (equal digests):
- **S**: the portable stack as deployed (portable producer + portable orderer).
- **A**: swap only the **orderer** for the teacher: portable producer + `d2`. The search's
  rollouts also play the portable producer, so the teacher plans for the producer it really has.
- **B**: swap only the **producer** for the hand-written one: `entity_planner` + portable
  orderer.
- **T**: swap both: `entity_planner` + `d2` (the teacher stack; EXP-0053's `d2` arm).

**What is measured.** Consumer units delivered over 100 cycles, summed over the worlds.
- The **gap** G = Σ (T − S): how far the portable stack is below the teacher stack.
- The **orderer closure** c_O = Σ (A − S) / G: the share of the gap closed by swapping only the
  orderer. 1.0 = the orderer swap alone closes all of it.
- The **producer closure** c_P = Σ (B − S) / G: the same for the producer swap.
- The **interaction** T − A − B + S: how much the two swaps are not additive.
- The pre-registered rule (D6): the role with the larger closure is trained first, unless
  |c_O − c_P| < 0.15, which is a tie and goes to the user; G ≤ 0 would make the reading
  undefined.

**Why mined worlds.** On the gate's easy worlds the portable orderer already beats every
hand-written rule (EXP-0049), so there would be no gap to attribute. D6 therefore uses worlds the
*headroom atlas* (EXP-0051/0052, a classified sample of generated worlds) mined, where a gap
exists: the shocked worlds of PoC-A's sample as EXP-0053 ran it, i.e. 12-0060 (class *hard*: a
cheap informed router already beats the best simple rule there by more than chance), 12-0111 and
12-0344 (class *open*: the analytic ceiling leaves much room above the best simple rule, but no
cheap policy reached it), all with burst demand and `shocks-i` shocks, one episode each at the
confirm stage's first seed.

**How large a difference counts.** Each arm is played once per world, so a difference between
two arms on one world is real only if it is larger than chance alone would produce. The
yardstick is the *noise floor*: the atlas's per-world measure of how large a gain chance
produces on such a world. Atlas v2 gives 145.0 units (12-0060) and 159.5 (12-0111, 12-0344);
those are for a mean of 2 episodes, and the matching floor for one episode of one arm against
one episode of another, the one EXP-0053 used, is 3 × σ_ep = **164.3 units** (σ_ep = the
atlas's pooled episode-to-episode spread, 54.8). Below, a per-world difference "clears the
floor" when it exceeds 164.3.

**What came out.**

| world | S (portable) | A (orderer swapped) | B (producer swapped) | T (teacher stack) | gap T − S | c_O | c_P | interaction |
|---|---|---|---|---|---|---|---|---|
| 12-0060 | 2,094 | 2,580 | 2,155 | 2,617 | 523 | 0.93 | 0.12 | −24 |
| 12-0111 | 2,166 | 3,263 | 2,364 | 3,262 | 1,096 | 1.00 | 0.18 | −199 |
| 12-0344 | 2,592 | 3,505 | 2,621 | 3,413 | 821 | 1.11 | 0.04 | −121 |
| **sum** | **6,852** | **9,348** | **7,140** | **9,292** | **G = 2,440** | **1.02** | **0.12** | **−344** |

Which per-world differences clear the 164.3-unit floor (0060 / 0111 / 0344; ✓ = clears):

| difference | 0060 | 0111 | 0344 | clears on |
|---|---|---|---|---|
| T − S (the gap) | +523 ✓ | +1,096 ✓ | +821 ✓ | 3 / 3 |
| A − S (orderer swap) | +486 ✓ | +1,097 ✓ | +913 ✓ | 3 / 3 |
| A − B (orderer swap vs producer swap) | +425 ✓ | +899 ✓ | +884 ✓ | 3 / 3 |
| B − S (producer swap) | +61 | +198 ✓ | +29 | 0111 only |
| A − T | −37 | +1 | +92 | none |
| interaction T − A − B + S | −24 | −199 ✓ | −121 | 0111 only |
| `round_robin` − B (EXP-0053 rows) | +274 ✓ | +891 ✓ | +589 ✓ | 3 / 3 |
| `d2` − `round_robin` (EXP-0053 rows) | +188 ✓ | +7 | +203 ✓ | 0060, 0344 (by 24 and 39) |

- **The gap is real.** The portable stack delivers 2,440 fewer units than the teacher stack over
  the three worlds (−26%), and it is below the teacher stack on every world by more than the
  floor (by 523 to 1,096).
- **The orderer explains the gap; the producer explains little of it.** Swapping only the
  orderer closes **c_O = 1.02** of the gap; swapping only the producer closes **c_P = 0.12**.
  The difference, 0.90, is six times the 0.15 tie band, each world reads the same way on its own
  (c_O 0.93–1.11, c_P 0.04–0.18), and the per-world A − B differences (425 / 899 / 884) all
  clear the floor. By D6's rule the reading is **orderer: M4 trains the orderer role first.**
  c_O above 1 is not an overshoot: A − T (−37 / +1 / +92) is inside the floor on every world, so
  the reading is c_O ≈ 1, A ≈ T.
- **The producer swap and the interaction are not distinguishable from zero except on
  12-0111.** B − S is +61 / +198 / +29 and the interaction −24 / −199 / −121; only 0111's clear
  the floor. On 0111 the producer swap helps under the portable orderer (+198) but not under
  the teacher (A − T = +1), which is what the negative interaction there means. On the other two
  worlds neither the producer swap nor the interaction is measurable in one episode.
- **Where the orderer gap comes from (context, cross-run arithmetic, not D6's reading).**
  EXP-0053 played `entity_planner` with `round_robin` (the teacher's floor: every order sent
  whole to the next eligible producer, no search) on the same world-episodes (same digests,
  deterministic arms). With that, T − S = 2,440 splits exactly into: the producer swap B − S =
  +288 (12%); `round_robin` instead of the trained orderer under `entity_planner`, 8,894 −
  7,140 = **+1,754 (72%)**, clearing the floor on all three worlds (274 / 891 / 589); and the
  search's gain over its floor, d2 − `round_robin` = +398 (16%), inside the floor on 0111 and
  clearing it narrowly on 0060 and 0344. Most of the orderer gap is thus the trained orderer
  falling behind plain whole-order rotation on these burst worlds, not something only the search
  finds. **Why it falls behind is not diagnosed.** The symptom is DIAGNOSTICS 19's (EXP-0049:
  the trained orderer splits 29–40% of consumer orders and loses to `round_robin` on shocked
  worlds); EXP-0052 left open whether splitting itself costs the trained orderer. EXP-0053's
  trace columns on these world-episodes are consistent with two of entry 19's mechanisms and do
  not separate them (trained orderer / `round_robin`, 0060 / 0111 / 0344):
  - (f) *splitting*: split share of consumer orders 0.36 / 0.23 / 0.26 against 0; units
    wasted in split orders that did not complete 190 / 428 / 345 (sum 963, less than the 1,754);
  - (b) *less receiver rotation*: receiver change share 0.83 / 0.46 / 0.55 against 0.94 / 0.71 /
    0.68; bid units displaced by higher-priced bids 493 / 3,489 / 2,886 against 68 / 1,346 /
    1,749.

  The discriminating check is the trained orderer forced to route each order whole (to its
  argmax receiver), against itself as-is and against `round_robin`, on these 3 world-episodes
  (registered as DIAGNOSTICS 19 "Seen in EXP-0054"). If the whole version recovers most of the
  1,754 units, the cause is splitting (f); if not, its rotation contrast with `round_robin`
  decides (b). No router for it exists yet, so it needs a code task first.
- **T reproduces EXP-0053 exactly.** T had to be rerun, not reused: task 8.1 changed the search
  worker (it now takes the producer from the job, so A's rollouts can play the portable
  producer), so the teacher-path fingerprint (`24cb5a84…`) differs from EXP-0053's registered
  `13e63e4e…` and D6's reuse rule refused reuse. The rerun T units equal EXP-0053's `d2` on all
  three worlds (2,617 / 3,262 / 3,413; `t_reproduction` in `spec.json`), and so do its search
  counts (searched streams 2,522 / 2,406 / 3,724, overrides 159 / 82 / 149, rollouts 90,296 /
  38,924 / 71,852). B also equals EXP-0053's `entity_planner/trained_orderer` rows (2,155 /
  2,364 / 2,621). The code change left the teacher's behaviour unchanged.
- **The teacher behaves the same with either producer.** A overrides `round_robin` on 152 / 80 /
  160 searched streams (T: 159 / 82 / 149), out of 2,518 / 2,524 / 3,805; the posterior resolved
  every decision (0 unresolved). A's search costs 1.4× T's per episode (the portable producer is
  batch-1 torch inside every rollout; 8.2 predicted 1.20–1.43×).

**Limitations.**
- **The producer reference is hand-written, not a teacher.** No producer search exists, so
  c_P = 0.12 measures only how far the portable producer is from `entity_planner`. A stronger
  producer reference could close more; this run cannot say how much.
- **Small sample.** Three shocked 12-producer worlds (one hard, two open), all burst demand, one
  episode each. The main reading is robust to that (A − B clears the 164.3-unit one-episode
  floor on every world, by 2.6× to 5.5×), but the secondary readings (the producer swap, the
  interaction, A against T, the search's gain over `round_robin`) mostly sit inside it. The run
  covers neither calm worlds, other demand patterns, nor 42-producer worlds (a 42-producer
  teacher read needs the user's approval of spend, D5/D8).
- **The orderer reference is mostly its floor.** 72% of the gap closes with whole-order
  `round_robin` alone (cross-run, above), so "train the orderer toward the teacher" here largely
  means "stop losing to whole-order rotation"; only 16% is search beyond it, and that part is
  inside the floor on one of the three worlds.

**What changes.** D6's pre-registered reading is **orderer**, not a tie, so the role choice does
not go to the user: M4's first training target is the ordering role. *How* to train it is not
decided by this run. If the DIAGNOSTICS 19 check above finds splitting (f), the first fix is
routing orders whole, and search labels matter for the remaining ≈ 16%; if it finds rotation
(b), the fix is in how the orderer varies its receivers. Only after that check can M4 decide
whether `d2` is the label source for the orderer on these worlds.

## Verification

Independent check `openspec/changes/add-m3-development-audit/probes/exp0054_check.py` (stdlib
only; D6 arithmetic and the D5 step-2 sample rule re-implemented): **PASS**.
- `contract_failures.csv` header-only.
- The sample equals the shocked worlds of D5's step-2 sample recomputed from atlas v2 (12-0060,
  12-0111, 12-0344), each at EXP-0053's episode seed (1295251592 / 195303600 / 1680874617).
- 12 rows = planned (3 world-episodes × S, A, B, T, replicate 0), each once, nothing else; one
  demand + shock digest pair per world-episode, equal to EXP-0053's.
- B equals EXP-0053's `entity_planner/trained_orderer` units on 3 / 3; T equals EXP-0053's
  `teacher/d2` on 3 / 3; `spec.json` `t_reproduction` recomputes.
- Every `attribution.csv` field (size row and the 3 world rows: S, A, B, T, G, c_O, c_P,
  interaction, closure difference, band, reading) recomputes from `artifacts/episodes.csv`.
- Pins: run commit `4e96255b…` (spec pin and provenance), checkpoint SHA-256s, atlas content
  hash and file SHA-256, horizon 100, not a dry run, no filters, T not reused with its reason
  recorded; the staged files' SHA-256 equal the pins (producer `8b51b77c…`, orderer
  `d25388bf…`, atlas `7517314f…`); one decisions file per A / T job.

## Tables

Search effort per teacher episode (`artifacts/teacher_decisions.csv`, 31,200 decision rows; a
*searched stream* is one order inside an ordering decision with ≥ 2 eligible producers; *no
confident decision* = the stream reached its last scenario checkpoint without a confident gain
or loss and kept the floor):

| world | arm | searched streams | overrides | no confident decision | rollouts | search s (pod) | EXP-0053 `d2` s |
|---|---|---|---|---|---|---|---|
| 0060 | T | 2,522 | 159 (6.3%) | 606 | 90,296 | 1,208 | 937 |
| 0060 | A | 2,518 | 152 (6.0%) | 631 | 91,600 | 1,691 | – |
| 0111 | T | 2,406 | 82 (3.4%) | 280 | 38,924 | 516 | 487 |
| 0111 | A | 2,524 | 80 (3.2%) | 290 | 40,456 | 683 | – |
| 0344 | T | 3,724 | 149 (4.0%) | 742 | 71,852 | 1,148 | 1,117 |
| 0344 | A | 3,805 | 160 (4.2%) | 751 | 73,700 | 1,643 | – |

Context from EXP-0053 (same world-episodes and digests): `entity_planner` + `round_robin` (the
teacher's floor) 2,429 / 3,255 / 3,210; the *atlas bar* (on each world, the best of the 8 blind and
hand-written routers, all with `entity_planner`) 2,473.5 / 3,255 / 3,210. S is below the bar on all
three by more than the floor (−380 / −1,089 / −618 units; −15%, −33%, −19%). A is above the bar by
+106.5 / +8 / +295 units, beyond the floor only on 0344.

## Conclusions

- D6 reading: **orderer** — G = 2,440 > 0, c_O = 1.023, c_P = 0.118, |c_O − c_P| = 0.905 ≥ 0.15;
  per world orderer on 3 / 3, A − B above the 164.3-unit one-episode floor on 3 / 3. M4 trains
  the ordering role first; how to train it waits on the DIAGNOSTICS 19 check (last bullet).
- Secondary readings mostly inside the floor: A ≈ T (c_O ≈ 1, not an overshoot); the producer
  swap (B − S) and the interaction (−344 summed) clear the floor only on 12-0111.
- `t_reproduction`: T rerun (fingerprint changed by task 8.1) equals EXP-0053 on 3 / 3 worlds,
  units and search counts.
- Open, not diagnosed here: why the trained orderer trails whole-order `round_robin` by 274–891
  units on these burst worlds (72% of G). Matching entry: DIAGNOSTICS 19 (splitting (f) vs
  receiver rotation (b); EXP-0053's traces fit both). Discriminating check: the trained orderer
  forced whole (argmax receiver) against itself and `round_robin` on these 3 world-episodes;
  whole recovers most of the 1,754 → (f), else the rotation contrast decides (b). Needs a new
  diagnostic router (code task) first; run before M4 picks how to train the orderer.

## Details

- Command (from the repo root): `python3 run_pod_training.py --create-pod --image
  runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12 --vcpu 4 --mem 8 --disk 30 --pod-name
  exp0054-role-attribution-mined --verify-source-commit --train "-u -m
  tokenomics.eval.teacher_audit --poc c --atlas /workspace/tokenomics/_pod_ckpts/atlas_v2.json
  --prod-ckpt /workspace/tokenomics/_pod_ckpts/producer_candidate.pt --orderer-ckpt
  /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt --reuse-t
  /workspace/tokenomics/_pod_ckpts/exp0053 --workers 4 --source-commit {SOURCE_COMMIT} --out-dir
  /workspace/tokenomics/runs/exp0054" --pull
  "/workspace/tokenomics/runs/exp0054:./experiments/EXP-0054-role-attribution-mined"`.
- **Staging.** All inputs under the git-ignored `tokenomics/_pod_ckpts/` (not `_pod_inputs/`,
  which holds a tracked file; EXP-0053 "Staging path changed"): both checkpoints, `atlas_v2.json`,
  and EXP-0053's `spec.json` + `artifacts/episodes.csv` as `_pod_ckpts/exp0053/` for `--reuse-t`.
  Deleted after the run; `_pod_inputs/` untouched.
- The `lp` extra (scipy 1.18.1) was confirmed in the pod venv (the launcher installs
  `.[train,lp]` on `--cpu`).
- **Pod and spend.** Pod `5tu4na6nhwf6mj`, 4 vCPU `cpu5c`, secure, first attempt; created
  ≈ 16:12:55Z, terminated ≈ 16:52:08Z: ≈ 39.2 min × 4 vCPU = **≈ 2.6 vCPU-h billed** (sized:
  44 min / 3.1 central, 88 min / 6.0 at × 2; hard stop 105 min / 7.0, not reached). Driver
  16:15:43Z → ≈ 16:52:05Z (≈ 36.4 min); the longest job, 0060's A, searched for 1,691 s
  (≈ 28 min; sized 31.0 pod min). Pod list after the run: `[]`.
- **M3 spend** (budget 40 vCPU-h): ≈ 25.0 through EXP-0053 (≈ 27.8 upper) + EXP-0054 ≈ 2.6 =
  **≈ 27.6 vCPU-h billed (≈ 30.4 upper)**.
- Teacher rows carry empty `trace_*` columns by design (the routing tracer is off on teacher
  episodes; DIAGNOSTICS 22).
- Files: `spec.json` (pins, provenance, `t_reuse`, `t_reproduction`), `attribution.csv`,
  `contract_failures.csv`; `artifacts/episodes.csv`, `artifacts/teacher_decisions.csv` (31,200
  rows, 5.6 MB), `artifacts/decisions/*.csv` (per teacher job), all under git-lfs.
