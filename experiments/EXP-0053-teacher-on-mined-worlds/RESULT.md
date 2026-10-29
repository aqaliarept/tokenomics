# EXP-0053 — PoC-A: the teacher on mined worlds

**Status: complete, valid. Not an acceptance read** (no verdict; the gate margins are reference
ratios only). `python -m tokenomics.eval.teacher_audit --reduction step2`, run commit
`6abb218d3c2c04b9ab6331c07bfdf73239abd06a` (task 7.1 of `add-m3-development-audit`, design D5,
ADR-0056), on atlas v2 = EXP-0052 `reclassified/atlas_v2.json`, content hash
`220886a7cf12aaac0cc0a48dc03c57f2dc2484bb2af6185a6160ec509c8a2a23` (file SHA-256 `7517314f297a…`),
trained orderer EXP-0012 `orderer_scorer.pt` (SHA-256 `d25388bfa33e…`). 133 episode rows over 8
world-episodes at 12 producers, horizon 100. 0 contract failures. Fresh 8 vCPU secure pod,
≈ 3.1 vCPU-h billed ([Details](#details)).

## In plain English

**The question.** The headroom atlas (EXP-0051/0052) found 12-producer worlds where the
analytic ceiling leaves a lot of room above the best simple rule, but no cheap policy got into it
(class *open*), and a few where a cheap informed router already beats the simple rules (class
*hard*). Before M4 builds a self-improvement loop around a *teacher* (a slow, expensive policy
whose decisions a fast learner imitates), this experiment asks: on those worlds, does a lookahead
search reach more consumer units than the cheap policies do? If it does, it can label decisions
the learner cannot yet make on its own; if not, M4 has no teacher on these worlds.

**The three teacher arms.** All use the same hand-written producer (`entity_planner`); only the
ordering side (who gets each order) differs.
- **`floor` = `round_robin`**: send each order whole to the next eligible producer in rotation.
  It is the search's default action, and also what the teacher does whenever it cannot decide.
  It replaced EXP-0042's floor (`blend_0.5`) because EXP-0048–0050 showed that splitting orders
  across producers loses 100–400 units per episode to plain whole-order rotation; a search
  starting that far below the bar would mostly measure its own bad starting point.
- **`d2`, the teacher**: at every ordering decision with two or more possible producers, it
  tries each alternative (the whole order to each eligible producer, plus what the trained
  orderer would do), plays each forward two cycles followed by the floor, and switches away from
  `round_robin` only when the gain is confidently positive across scenarios. On worlds whose
  future is random (shocks, random demand), the scenarios are *posterior forks*: futures drawn to
  be consistent with everything observed so far (ADR-0056), 4, 8 or 16 of them.
- **`ub`, the clairvoyant bound**: the same search, but given the true future (the real pending
  shocks and demand) as its single scenario. It reads hidden information, so it is only a
  reference; it is never a teacher, label source or candidate. It was meant to show how much a
  perfect forecast would help this search, but it turned out **not to be an upper bound** on it
  (see below). It runs only on the 5 worlds whose future is random (on a cyclic,
  shock-free world it equals `d2` by construction).

**What is compared, per world.** The *bar* is the best of the 8 blind and hand-written routers
on the same world-episode (the atlas bar: five roster v2 orderers, `random`, `round_robin`,
`random_whole`, all rerun here). The *reach arm* is the best cheap informed router
(`makeable_split`, `makeable_weighted_whole`, the trained orderer). The *ceiling C* is the
largest number of consumer units the world can deliver by the atlas's bounds (task 4.7), or the
best played arm if higher. The *noise floor* is the atlas's per-world measure of how large a gain
chance alone produces on such a world (atlas v2, 3 standard errors from pooled median-based
spreads); a gain counts only above it. The sample is the pre-registered one: 4 hard and 4 open
train-split worlds, the confirm stage's first episode seed each (D5 reduction step 2, `ub` kept;
`sizing.md` §EXP-0053).

**What came out.**

| world | class | shocks / demand | bar (arm) | reach (arm) | ceiling C | noise floor | floor | **d2** | d2 − bar | d2 / bar | `ub` (hidden info) | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12-0060 | hard | shocks, burst | 2,473.5 (`random_whole`) | 2,628.5 (`mww`) | 4,759 | 145.0 | 2,429 | **2,617** | +143.5 | 1.058 | 2,606 | candidate/depth-limited |
| 12-0176 | hard | cyclic | 1,814 (`round_robin`) | 1,969.5 (`mww`) | 3,254 | 86.7 | 1,814 | **1,916** | +102 | 1.056 | – | candidate/depth-limited |
| 12-0005 | hard | cyclic | 834 (`round_robin`) | 932 (`mww`) | 1,172 | 93.4 | 834 | **940** | +106 | 1.127 | – | beyond trained |
| 12-0348 | hard | cyclic | 861 (`round_robin`) | 940 (trained) | 983 | 8.6 | 861 | **925** | +64 | 1.074 | – | candidate/depth-limited |
| 12-0111 | open | shocks, burst | 3,255 (`round_robin`) | 3,071 (`mww`) | 13,803 | 159.5 | 3,255 | **3,262** | +7 | 1.002 | 3,261 | ceiling-limited |
| 12-0344 | open | shocks, burst | 3,210 (`round_robin`) | 3,133.5 (`mww`) | 11,429 | 159.5 | 3,210 | **3,413** | **+203** | 1.063 | 3,383 | **search headroom** |
| 12-0308 | open | seasonal | 2,736 (`round_robin`) | 2,734.5 (`mww`) | 8,159 | 159.5 | 2,736 | **2,917** | **+181** | 1.066 | 2,904 | **search headroom** |
| 12-0392 | open | drift | 4,215 (`random_whole`) | 3,522.5 (`mww`) | 12,299 | 131.4 | 4,039 | **4,333** | +118 | 1.028 | 4,234 | ceiling-limited |

(`mww` = `makeable_weighted_whole`; bar and reach for stochastic arms are means of 2 replicates.)

- **The teacher never does worse than its floor or the bar.** `d2` beats `round_robin` on all 8
  worlds (+7 to +294 units) and the bar on all 8 (+7 to +203); it is never below the bar. No
  episode of any teacher arm, and no bar arm, died.
- **Open worlds: the pre-registered reading holds, narrowly.** On 2 of the 4 open worlds
  (12-0344, 12-0308) `d2` beats the bar by more than the noise floor, which meets D5's "at least
  half" rule: **search finds headroom the cheap arms miss there**, and these two worlds are
  teacher-label candidates for M4. The call is thin. The atlas floor in the table is for a
  *mean of 2 episodes* of a deterministic bar against a stochastic reach arm (3 × √(σ_ep² +
  σ_rep² / 2) / √2 = 159.5 on these sparse_f2 worlds); this run compares *one* episode of two
  deterministic arms (`d2` against `round_robin`), whose matching floor is 3 × σ_ep = **164.3**.
  Under it the margins are +38.7 (12-0344) and +16.7 (12-0308) units, and both still pass
  (12-0392, bar `random_whole` R = 2: 3 × √(σ_ep² + σ_rep² / 2) = 185.8, +118 still fails;
  12-0111 unchanged). Two of four is the minimum for "at least half", each passing world rests on
  one episode, and the posterior search's own sampling variation is not in any floor: one of the
  two falling inside noise on a second seed would flip the reading. On the other two (12-0111, 12-0392)
  neither `d2` nor `ub` beats the bar by more than the noise floor; D5's rule labels them
  "ceiling-limited", but D5 means by that "not reachable at depth 2 even with the future known",
  and this `ub` cannot show that (it is not an upper bound, below), so the label here only says
  that neither arm beat the bar. On all four the teacher closes very little of the room: 0.1–3.3%
  of the distance from the bar to the ceiling. The open worlds' ceiling is essentially their
  demand (9,600–13,900 units ordered against 2,900–4,300 delivered by `d2`; consumer fill
  24–35%), and the ceiling ignores order lifetimes (TTL), routing, funding and degradation, so
  how much of that room any policy can reach is unproven.
- **Hard worlds: the teacher beats the bar but does not exceed the best cheap router.** On 3 of
  4 hard worlds `d2` beats the bar by more than the noise floor (12-0176, 12-0005, 12-0348;
  12-0060 misses by 1.5 units, and by 42 under its one-episode floor of 185.8). The three cyclic,
  shock-free worlds are deterministic, so their atlas floor holds only the reach arm's replicate
  noise (or 1% of the bar) and `d2` against `round_robin` there has no episode noise at all. Against the reach arm, by D5's pre-registered comparison (`d2` ≥
  reach, no noise margin) only 12-0005 reads "beyond trained" (+8 units over
  `makeable_weighted_whole`); the other three read "candidate- or depth-limited". On 12-0060 and
  12-0176 the shortfall (−11.5, −53.5) is inside the noise floor, so these are ties; only on
  12-0348 is it larger than that world's (very small) floor: `d2` 925 against the trained
  orderer's 940, although the trained orderer's action is one of the teacher's candidates.
  The evidence is that the search rarely leaves its floor: on 12-0348 it overrode `round_robin`
  on 21 of 1,322 searched streams (1.6%), and on 381 (29%) it reached no confident decision
  and kept the floor; across all eight worlds overrides are 1.6–6.3% of searched streams (a
  *stream* is one order inside an ordering decision; a decision with ≥ 2 eligible producers
  searches 1–3 of them). That
  is consistent with a two-cycle lookahead against a `round_robin` continuation seeing too little
  of the advantage a whole informed policy builds up, but it is not diagnosed: whether the
  candidate set, the depth or the two-cycle objective is the limit is not separated by this run.
- **The clairvoyant arm is not an upper bound for this search.** `ub` is 1 to 99 units *below*
  `d2` on all 5 worlds. The cause is structural, not a defect (review concern #358, DIAGNOSTICS
  23). The search scores a candidate by the consumer units it adds within a two-cycle window,
  played on with the floor. On the true future most decisions are exactly neutral inside that
  window, so `ub` takes the "no difference" exit (`deterministic_zero`: every candidate gains
  exactly 0 → keep `round_robin`) more often than `d2` on every world (0060 70.0% vs 66.9% of
  searched streams, 0111 82.5 / 81.3, 0308 89.0 / 86.0, 0344 74.2 / 68.9, 0392 79.7 / 78.8).
  Posterior forks perturb demand and shocks, so for some weakly better actions a few of the 16
  futures show a small positive window gain and none a negative one; the one-sided confidence
  test then passes and `d2` switches, and those switches pay off beyond the window. The two arms
  first diverge exactly like this (0060 cycle 2, 0308 cycle 8, 0392 cycle 8: `ub` keeps the floor
  by `deterministic_zero`, `d2` overrides; the reviewer's repro at 0060 found nonzero gains in 3 of
  16 posterior futures, mean 0.56 units, never negative), and the clairvoyant fork does carry the
  recorded episode's demand and shock digests. So `ub` differs from `d2` both in information and
  in how it reads the gains (one scenario with a zero-variance rule against up to 16 with a
  confidence bound), and more information about a myopic objective can lower total units. **D5's
  fork-limited reading is therefore not determinable from this run** (the rule's count of 0 is
  uninformative), and nothing here says whether better forks would help.
- **The posterior always resolved.** Every one of the 25,200 posterior decisions (500 cycles on
  the 5 posterior worlds) found 16 of 16 consistent futures: unresolved fraction 0, acceptance rate 1.0 everywhere.
  By ADR-0056 the acceptance rate is binary (1.0 when a feasible plan exists, 0 and unresolved
  otherwise), so 1.0 means "always feasible", not "a high sampling rate".
- **Gate margins, as references only.** `d2` / bar reaches the calm reference 1.05 on 4 of the 5
  calm worlds (12-0176, 12-0005, 12-0348, 12-0308; not 12-0392 at 1.028) and the shocked
  reference 1.10 on none of the 3 shocked worlds (12-0060 1.058, 12-0111 1.002, 12-0344 1.063). This is not an acceptance read and changes no margin.

**What it means for M4's teacher.** A depth-2 lookahead over `round_robin` is a *safe* teacher on
these worlds (never below the bar, +6% on average, at 1.002–1.127 × the bar) and the D5 reading
says it finds headroom on half the open worlds, so the self-evolution loop has a teacher in the
pre-registered sense, with 12-0344 and 12-0308 as label candidates. It is a *weak* one: one
episode per world, margins of 17–39 units over the one-episode noise floor (2 of 4 open worlds,
the minimum for D5's rule), 2–3% of the room closed, and on
the hard worlds it ties or trails the cheap informed routers that already exist. What limits it
is open: the candidate set, the depth, the two-cycle objective (a longer window or a non-myopic
score may change the picture, including what a clairvoyant reference would show), or the
forks; this run does not separate them, and its `ub` gives no evidence for or against the forks.
That needs its own discriminating check before M4 relies on this teacher or picks what to
strengthen.

## Verification

Independent check `openspec/changes/add-m3-development-audit/probes/exp0053_check.py` (stdlib
only; rules re-implemented from D5 / D12): **PASS**.
- `contract_failures.csv` header-only.
- The sample equals D5's step-2 rule recomputed from atlas v2 (hard by gain: 0060, 0176, 0005,
  0348; open by room, best 2 shocked + 2 unshocked: 0111, 0344, 0308, 0392); all train split, 12
  producers, patterns cyclic / burst / seasonal / drift; each at the confirm's first episode seed.
- 133 rows = planned (8 × (14 blind / reach + `floor` + `d2`) + 5 `ub`), each (world-episode,
  subject, replicate) once; one demand + shock digest pair per world-episode; `bound/ub` rows
  flagged hidden information on exactly the 5 worlds with shocks or stochastic demand.
- All 112 blind / reach rows reproduce EXP-0052's units for the same world, episode, subject and
  replicate; `teacher/floor` = `round_robin` units on all 8 worlds.
- 0 B2 violations; every `ratios.csv` field (bar, reach, C, d2 deltas and ratios, room closed,
  beats / ≥ / below flags, reference ratios, readings, `ub` columns) recomputed from
  `episodes.csv`; unresolved fraction recomputed from `teacher_decisions.csv`; acceptance rates
  binary.
- Pins: `spec.json` records the run commit, the atlas content hash, the orderer SHA-256 and
  `reduction: step2`; the staged files' SHA-256 equal the pins (atlas `7517314f…`, orderer
  `d25388bf…`).

## Tables

Search effort per teacher episode (`teacher_decisions.csv`; a *searched stream* is one order
inside an ordering decision with ≥ 2 eligible producers, counted from `searched_streams`; of the
65,700 decision rows, 34,175 search 0 streams, 28,748 one, 2,424 two and 353 three; *no
confident decision* = the stream reached its last scenario checkpoint without a confident gain or
loss and kept the floor):

| world | arm | scenarios | searched streams | overrides | no confident decision | rollouts | search s (pod) |
|---|---|---|---|---|---|---|---|
| 0060 | `d2` | posterior | 2,522 | 159 (6.3%) | 606 | 90,296 | 937 |
| 0060 | `ub` | clairvoyant | 2,502 | 105 | 605 | 12,391 | 174 |
| 0176 | `d2` | deterministic | 1,785 | 52 (2.9%) | 251 | 8,809 | 92 |
| 0005 | `d2` | deterministic | 2,132 | 71 (3.3%) | 467 | 10,542 | 120 |
| 0348 | `d2` | deterministic | 1,322 | 21 (1.6%) | 381 | 6,603 | 65 |
| 0111 | `d2` | posterior | 2,406 | 82 (3.4%) | 280 | 38,924 | 487 |
| 0111 | `ub` | clairvoyant | 2,428 | 85 | 243 | 7,067 | 97 |
| 0344 | `d2` | posterior | 3,724 | 149 (4.0%) | 742 | 71,852 | 1,117 |
| 0344 | `ub` | clairvoyant | 3,726 | 134 | 539 | 10,839 | 229 |
| 0308 | `d2` | posterior | 2,605 | 74 (2.8%) | 228 | 38,592 | 483 |
| 0308 | `ub` | clairvoyant | 2,626 | 73 | 146 | 7,500 | 98 |
| 0392 | `d2` | posterior | 3,468 | 159 (4.6%) | 493 | 98,480 | 1,145 |
| 0392 | `ub` | clairvoyant | 3,409 | 153 | 471 | 16,639 | 181 |

Posterior: 25,200 decisions over 500 cycles, 16 / 16 accepted draws on every one (403,200
draws), 0 unresolved. Room closed by `d2` = (d2 − bar) / (C − bar): 0060 0.063, 0176 0.071, 0005 0.314,
0348 0.525, 0111 0.001, 0344 0.025, 0308 0.033, 0392 0.015. `d2` / C: 0.55, 0.59, 0.80, 0.94,
0.24, 0.30, 0.36, 0.35.

## Conclusions

- D5 per-class reading: **open — search headroom on 2 of 4 (12-0344, 12-0308; teacher-label
  candidates); on 12-0111 and 12-0392 neither `d2` nor `ub` beats the bar (the rule's
  "ceiling-limited"); fork-limited not determinable (`ub` is not an upper bound for this search,
  DIAGNOSTICS 23); hard — beyond trained on 1 of 4 (12-0005), candidate- or depth-limited on 3
  (12-0060, 12-0176 within noise; 12-0348 below the trained orderer by more than its floor).**
  `d2` ≥ bar everywhere, 0 deaths, 0 unresolved posterior labels; `ub` 1–99 units below `d2` on
  5 / 5 (structural: the two-cycle objective plus the one-scenario zero rule).
- The open-world reading does not fall short, so no failure diagnosis is required by D5; the
  hard-world limit (candidate set, depth or two-cycle objective) and the small share of room
  closed are recorded above with their evidence, not diagnosed further.
- For M4: the depth-2 search over `round_robin` is a safe but weak teacher on 12-producer mined
  worlds. What limits it (candidates, depth, the myopic window, forks) is open; this run gives no
  evidence for or against better forks. Margins unchanged; 42-producer teacher reads stay unrun
  (D5, user approval of spend needed).

## Details

- Command (from the repo root): `python3 run_pod_training.py --create-pod --image
  runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12 --vcpu 8 --mem 16 --disk 30 --pod-name
  exp0053-teacher-on-mined-worlds --verify-source-commit --train "-u -m
  tokenomics.eval.teacher_audit --reduction step2 --atlas
  /workspace/tokenomics/_pod_ckpts/atlas_v2.json --orderer-ckpt
  /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt --workers 8 --source-commit {SOURCE_COMMIT}
  --out-dir /workspace/tokenomics/runs/exp0053" --pull
  "/workspace/tokenomics/runs/exp0053:./experiments/EXP-0053-teacher-on-mined-worlds"`.
- **Staging path changed from tasks.md.** tasks.md 7.1 stages `atlas_v2.json` under
  `tokenomics/_pod_inputs/`; that directory is not git-ignored (it holds the tracked
  `gate_archive.jsonl`), so an untracked file there fails `--verify-source-commit`. Both inputs
  were staged under the git-ignored `tokenomics/_pod_ckpts/` instead and deleted after the run;
  `_pod_inputs/` was left untouched.
- The `lp` extra (scipy) was installed on the pod (`run_pod_training` installs `.[train,lp]` on
  `--cpu`); the driver's real-run check, which refuses without it, passed and B_lp ran.
- **Pod and spend.** Pod `n0x6h11yovc2u6`, 8 vCPU `cpu5c`, secure, 16 GB; created ≈ 12:42:51Z
  (first attempt), terminated ≈ 13:05:42Z: ≈ 22.9 min × 8 vCPU = **≈ 3.1 vCPU-h billed**
  (sized: 33 min / 4.4; hard stop 75 min / 10). Driver 12:43:50Z → ≈ 13:05:37Z (21.8 min); the
  longest job, 0392's `d2`, searched for 1,145 s (≈ 19 min, sized 30.5 pod min). Pod list after
  the run: `[]`.
- **M3 spend** (budget 40 vCPU-h): ≈ 21.9 through EXP-0052 (≈ 24.7 upper, if a failed pod's
  initialization was billed) + EXP-0053 ≈ 3.1 = **≈ 25.0 vCPU-h billed (≈ 27.8 upper)**.
- Arms (17 rows per shocked / stochastic world-episode, 16 otherwise): roster v2 (`entity_planner`
  with `blend_0.5`, `even_split`, `obs_price`, `proportional`, `random` R = 2, `shock_reactive`),
  `round_robin`, `random_whole` R = 2 (blind); `makeable_split`, `makeable_weighted_whole` R = 2,
  `trained_orderer` (reach); `teacher/floor`, `teacher/d2`, `bound/ub`. Bar / reach = this run's
  argmax on the world-episode (driver choice; D5).
- Teacher rows carry empty `trace_*` columns by design (the routing tracer is off on teacher
  episodes; DIAGNOSTICS 22).
- Files: `spec.json` (pins, provenance, readings), `ratios.csv`, `contract_failures.csv`;
  `artifacts/episodes.csv`, `artifacts/teacher_decisions.csv` (65,700 decision rows, 10 MB, moved
  under git-lfs `artifacts/`), `artifacts/decisions/*.csv` (per teacher job).
