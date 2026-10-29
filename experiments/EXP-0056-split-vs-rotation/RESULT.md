# EXP-0056 — DIAGNOSTICS 19 split check: splitting or receiver rule?

**Status: complete, valid. Not an acceptance read. Reading: `joint` (3 of 3 worlds).**
`python -m tokenomics.eval.teacher_audit --poc split`, run commit
`40dc1873ab9fd7a03ebc028eca1b5f2142331088` (task 11.3 of `add-m3-development-audit`, design
D13), on atlas v2 = EXP-0052 `reclassified/atlas_v2.json` (content hash `220886a7cf12…`, file
SHA-256 `7517314f297a…`), trained orderer EXP-0012 `orderer_scorer.pt` (SHA-256
`d25388bfa33e…`), references EXP-0053 and EXP-0054. 18 episode rows = 3 shocked world-episodes
at 12 producers × (TR, TW, RS, RR + RW × 2), horizon 100, `entity_planner` producer, routing
tracer on. 0 contract failures, 0 deaths. Fresh 4 vCPU secure pod, ≈ 0.1 vCPU-h billed
([Details](#details)).

## In plain English

**The question.** On the three shocked, burst-demand worlds of PoC-C (12-0060, 12-0111,
12-0344), the trained orderer (the EXP-0012 scorer, with the hand-written producer
`entity_planner`) delivers 274 / 891 / 589 fewer consumer units than the plain rule
`round_robin`, which sends every order whole to the next eligible producer in turn. That
shortfall is 72% of the gap between our deployed agent and the teacher stack (EXP-0054). Before
M4 decides how to train the orderer, we need to know *why* it loses. Two explanations fit the
earlier traces (DIAGNOSTICS 19):
- **(f) splitting**: the scorer splits 23–36% of consumer orders across producers, and a split
  order completes only if every part does;
- **(b) the receiver rule**: *which* producers it sends orders to. It sends all concurrent
  orders of a product to the same producer (spread across concurrent orders) and changes the
  receiver over time less than `round_robin` does (rotation over time).

**Why a 2 × 2 and not just a "forced whole" version.** Making the scorer send each order whole
to its top-scored producer removes splitting, but it can also change how its receivers rotate.
If that version failed, we could not tell whether splitting is harmless or the rotation change
hurt. So the check crosses the two rules, each arm using the same scorer:

| | receivers chosen by the scorer | receivers chosen by `round_robin`'s rotation |
|---|---|---|
| the scorer's own split | **TR**: the trained orderer as is | **RS**: the scorer's split sizes, handed out in `round_robin`'s order (largest part to `round_robin`'s pick) |
| every order whole | **TW**: each order whole to the scorer's top producer | **RR**: `round_robin` |

RW (`random_whole`, each order whole to a random eligible producer, 2 replicates) is a
reference only. TW − TR is what removing splitting is worth while keeping the scorer's receiver
rule; RS − TR is what changing the receiver rule is worth while keeping the scorer's split.
RR − RS and RR − TW read the same two changes at the other level of the other factor. The
*interaction* TR − TW − RS + RR says how much the two changes are worth together beyond the sum
of each alone.

**How large a difference counts.** Each arm is played once per world (the arms are
deterministic), so a difference between two arms on one world counts when it exceeds the
atlas's one-episode noise floor, 3 σ_ep = **164.3 units** (the yardstick EXP-0053 and EXP-0054
read; the atlas's per-world floors for a 2-episode mean are 145.0 / 159.5 / 159.5).

**The pre-registered rule (D13, fixed before any horizon-100 data).** On a world, a change
*recovers the gap* (gap = RR − TR) when the gap clears the floor, the change's gain from TR
clears the floor, and that gain is at least 2/3 of the gap. Over the 3 worlds (≥ 2 needed):
TW recovers → **(f)**; else RS recovers → **(b)**; else the gap clears and neither single change
recovers → **`joint`** (only both changes together, i.e. RR, close it); else **`neither`**; a gap
clearing on fewer than 2 worlds → **`no_gap`**.

**First: how each change worked (realized levels, `manipulation.csv`; context, never a gate).**
The factors are rules held by construction; how often orders then split or receivers rotate
follows each arm's own trajectory.
- RS kept the scorer's split: split share of consumer orders 0.57 / 0.22 / 0.28 against TR's
  0.36 / 0.23 / 0.26 (degree 1.82 / 1.22 / 1.28 against 1.48 / 1.23 / 1.26). On 12-0111 and
  12-0344 equal within 0.02; on 12-0060 RS split *more* (+0.21) than TR did.
- RS took `round_robin`'s receiver behaviour on both sides: concurrent top share (the share of a
  cycle's concurrent orders of a product sent to the group's most common producer; 1 = all to
  one) 0.54 / 0.73 / 0.75 against RR's 0.52 / 0.73 / 0.74 and TR's 0.98 / 0.99 / 0.99; lead
  receiver change (how often the producer receiving an order's largest part changes between
  consecutive orders of a stream) 0.90 / 0.69 / 0.63 against RR's 0.94 / 0.71 / 0.68 and TR's
  0.66 / 0.36 / 0.43.
- TW kept the scorer's receiver behaviour: concurrent top share 0.99 on all three (TR 0.98–0.99);
  lead receiver change 0.51 / 0.35 / 0.48 against TR's 0.66 / 0.36 / 0.43 — lower on 12-0060
  (−0.15), within 0.05 on the other two. (The 11.2 horizon-12 dry run's "0.0" did not carry over
  to horizon 100.)
- The tracer's set change share (how often the *set* of producers receiving an order differs
  from the stream's previous order): TR 0.83 / 0.46 / 0.55, TW 0.51 / 0.35 / 0.48, RS 0.95 /
  0.75 / 0.76, RR 0.94 / 0.71 / 0.68 (RW 0.75 / 0.50 / 0.50). It equals the lead receiver change
  for arms that only route whole (TW, RR, RW, every row). For the splitting arms (TR, RS) it is
  higher on every world, because a change in a split order's minor parts changes the set while
  the lead stays (it can also miss a lead change within an unchanged set; here the net effect is
  upward). So it ranks RS above RR on all three worlds, and TR above TW (12-0344) and above RW
  (12-0060), where lead change ranks them the other way; read rotation from the lead receiver
  change.
- TR differs from RR on both receiver sides (concurrent spread by 0.25–0.45, rotation by
  0.24–0.34), so the receiver factor here moves both at once.

**What came out (consumer units over 100 cycles).**

| world | TR | TW | RS | RR | RW (mean of 2) | gap RR − TR | TW − TR (share of gap) | RS − TR (share) | RR − RS (share) | RR − TW (share) | interaction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12-0060 | 2,155 | 2,024 | 2,149 | 2,429 | 2,473.5 | 274 ✓ | −131 (−0.48) | −6 (−0.02) | 280 ✓ (1.02) | 405 ✓ (1.48) | 411 ✓ |
| 12-0111 | 2,364 | 2,908 | 2,566 | 3,255 | 3,038.5 | 891 ✓ | +544 ✓ (0.61) | +202 ✓ (0.23) | 689 ✓ (0.77) | 347 ✓ (0.39) | 145 |
| 12-0344 | 2,621 | 2,823 | 2,673 | 3,210 | 3,105 | 589 ✓ | +202 ✓ (0.34) | +52 (0.09) | 537 ✓ (0.91) | 387 ✓ (0.66) | 335 ✓ |
| **sum** | **7,140** | **7,755** | **7,388** | **8,894** | | **1,754** | **+615 (0.35)** | **+248 (0.14)** | **1,506 (0.86)** | **1,139 (0.65)** | **891 (0.51)** |

✓ = clears the 164.3-unit floor.

- **The gap reproduced** on all three worlds (274 / 891 / 589, each above the floor).
- **Neither single change recovers it on any world.** Removing splitting while keeping the
  scorer's receivers (TW) recovers −0.48 / 0.61 / 0.34 of the gap — it *loses* units on
  12-0060, and on 12-0111, the nearest, it falls short of the 2/3 bar (544 of the 594 needed).
  Changing the receiver rule while keeping the scorer's split (RS) recovers −0.02 / 0.23 / 0.09.
  So every world is `joint`, and by D13's rule **the reading is `joint` (3 of 3 worlds)**.
  The reading does not hinge on 12-0111's near miss: (f) needs TW to recover on 2 worlds, and
  on the other two it recovers −0.48 and 0.34.
- **Each change pays far more once the other is made.** With `round_robin`'s receivers,
  removing splitting (RS → RR) is worth 280 / 689 / 537 units (0.86 of the summed gap); with the
  scorer's receivers (TR → TW) it is worth 0.35. With whole orders, changing the receiver rule
  (TW → RR) is worth 405 / 347 / 387 (0.65); with the scorer's split (TR → RS) it is worth 0.14.
  The interaction is positive on every world, +411 / +145 / +335 (51% of the summed gap), and
  clears the floor on 12-0060 and 12-0344 (also against a √2-wider floor of ≈ 232 for a
  four-arm contrast, our arithmetic, not pre-registered); on 12-0111 it is inside the floor and
  the two changes look closer to additive there. That is the complement pattern D13 names:
  each change helps mainly when the other is also made. **But 12-0060's interaction is not
  clean consumer-side evidence.** There TW also fails on its producers' own purchases: its
  investment rejection rate is 0.56 (TR 0.02, RS 0.06, RR 0.02) and its mean capacity over the
  episode (`trace_capacity_ratio_trace`) is 194.7 against TR 227.9, RS 227.4, RR 242.3. TW's
  units, and so RR − TW and the interaction (both contain −TW), absorb that drop, which runs
  through the investment-routing channel (DIAGNOSTICS 19 (d)), not consumer receivers. The
  complement evidence above the floor that is not carrying such a failure is 12-0344 alone
  (+335; there TW's investment rejection 0.025 and mean capacity 241.0 are in line with TR's
  0.006 / 240.2 and the other arms' 0.011–0.021 / 234.2–238.8).
- **What the traces show (description, not a causal test).** The arms with the scorer's
  receivers (TR, TW) send almost every concurrent order of a product to one producer, and they
  have more bid units displaced by higher-priced bids (TR 493 / 3,489 / 2,886, TW 1,010 / 3,827 /
  3,172 against RS 81 / 1,356 / 1,483 and RR 68 / 1,346 / 1,749) and more consumer units lost to
  expiry *while the receiver had the inputs* (TR 692 / 1,653 / 2,718, TW 712 / 1,740 / 2,305
  against RS 102 / 1,015 / 1,260 and RR 121 / 962 / 1,100) — consistent with congestion at the
  one producer the scorer picks. Removing splitting under those receivers (TR → TW) raises
  displacement further (e.g. 493 → 1,010 on 12-0060): the split's minor part was sending some
  load elsewhere. Under `round_robin`'s spread, splitting mostly wastes capacity: units produced
  for split orders that did not complete are RS 260 / 310 / 508, close to RR − RS on 12-0060 and
  12-0344 (280, 537) and about half of it on 12-0111 (689). Expiries for lack of inputs are
  higher with `round_robin`-style receivers (RS / RR ≈ 2,180 / 8,700–9,100 / 6,800–6,900 against
  TR 1,585 / 8,203 / 5,282): the scorer's input-aware picks avoid input shortages (as EXP-0052
  saw), but its congestion costs more than that saves.
- **TW on 12-0060 also rejects 56% of its producers' investment orders** (one RW replicate there
  0.26, every other row ≤ 0.07; TR 0.02), and its producers run at lower capacity (mean 194.7
  against 208.6–242.3 for every other row on that world). Routing producers' own purchases whole
  to the scorer's top pick fails there, which is consistent with TW losing units on that world
  (−131 against TR); this run does not show how much of the loss it accounts for.
- **Reproduction.** TR, RR and RW equal EXP-0053's rows exactly (units on all 12 compared rows)
  and TR equals EXP-0054's B on all three worlds (2,155 / 2,364 / 2,621); demand and shock
  digests equal both references on every world-episode (`spec.json` `reproduction`: 15 of 15
  equal). The code path for these arms is unchanged.

**Limits (D13; they bound what `joint` means).**
- (o) A cause outside the two rule changes cannot be detected here: every unit of the gap is
  assigned to splitting, the receiver rule or their interaction. `manipulation.csv` describes
  how the changes worked; it does not test for another cause.
- (i) `joint` is the remainder: a world whose gap clears and where no single change recovers it
  is `joint` by construction. The branch alone is not evidence that the two changes are
  complements; only the interaction's sign and size speak to that (positive on 3 worlds, above
  the floor on 2, of which 12-0060's also carries TW's investment failure, above; so above the
  floor and clean on 12-0344 only).
- (ii) The "receiver rule" factor is every way `round_robin`'s receivers differ from the
  scorer's: spread across concurrent orders, rotation over time, and also *which* producers get
  orders (their capacity and stock). This run cannot say whether spread or rotation is the part
  that matters; the trained orderer differs from `round_robin` on both.
- (iii) Removing splitting was tested as a *rule* (route whole to the scorer's top pick, or
  whole by rotation), not as "incomplete split orders are the mechanism". The wasted-unit and
  displacement columns above describe the mechanism; they do not isolate it.
- Both factors change how *producers' own purchases* (investment orders) are routed, not only
  consumer orders: the orderer routes both, and TW / RS / RR apply their rule to both. So a
  contrast can carry an investment-channel effect (12-0060's TW above), and the reading is
  about the orderer's rule on all its orders, not about consumer routing alone.
- Small sample: three shocked 12-producer worlds with burst demand, one episode each.

**What it means for M4.** On these three shocked, burst-demand 12-producer worlds (one episode
each), the cheapest rule changes that close the trained orderer's gap to `round_robin` are both
together: whole orders *plus* `round_robin`-like receivers. Either alone recovers a third of the
summed gap (whole orders, scorer's receivers) or a seventh (rotated receivers, scorer's split).
So M4's orderer needs both abilities available: route an order whole, and vary its receivers.
Which receiver property matters — spread across concurrent orders, rotation over time, or which
producers are picked — is open (limit (ii)); the scorer's tiling of one score row over all its
slots (D13) is one candidate for the spread side, not a tested cause. The reference arm RW hints
that rotation as strong as `round_robin`'s is not required: it routes whole to random receivers
with less rotation than RR (lead change 0.75 / 0.50 / 0.50 against 0.94 / 0.71 / 0.68) and still
recovers 1.16 / 0.76 / 0.82 of the gap (RW − TR = 318.5 / 674.5 / 484), but with spread like RR's
(concurrent top share 0.57 / 0.72 / 0.70); one reference arm, not a contrast. Hypothesis, not
tested here (only the EXP-0012 scores were played): improving the scorer's scores under its
current architecture, without whole routing and varied receivers, would leave most of the gap.
Whether search labels (`d2`) add beyond that remains the ≈ 16% EXP-0054 attributed to the
search. In DIAGNOSTICS 19's terms this is (f) and (b) combined.

## Details

- **Command** (repo root; staged inputs under the git-ignored `tokenomics/_pod_ckpts/`):
  `python3 run_pod_training.py --create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu --python
  python3.12 --vcpu 4 --mem 8 --disk 30 --pod-name exp0056-split-vs-rotation
  --verify-source-commit --train "-u -m tokenomics.eval.teacher_audit --poc split --atlas
  /workspace/tokenomics/_pod_ckpts/atlas_v2.json --orderer-ckpt
  /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt --reference
  /workspace/tokenomics/_pod_ckpts/exp0053 --reference /workspace/tokenomics/_pod_ckpts/exp0054
  --workers 4 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0056"
  --pull "/workspace/tokenomics/runs/exp0056:./experiments/EXP-0056-split-vs-rotation"`.
- **Staging.** `orderer_scorer.pt` (SHA-256 `d25388bfa33ea3fc…`, = pin), `atlas_v2.json`
  (file SHA-256 `7517314f297ac39e…`, content hash `220886a7cf12aaac…`, = pins), EXP-0053's and
  EXP-0054's `spec.json` + `artifacts/episodes.csv` as `_pod_ckpts/exp0053/`, `exp0054/`.
  Deleted after the run.
- **Pod and spend.** Pod `m4m36h27agvcrk`, 4 vCPU `cpu5c`, secure, first attempt; launched
  21:13:28Z, terminated before 21:14:57Z (launcher exit): ≤ 1.5 min × 4 vCPU = **≈ 0.1 vCPU-h
  billed** (sized 0.3–0.5; hard stop 1.0, not reached). Driver started 21:14:39Z, done
  ≈ 21:14:52Z. Pod list after the run: `[]`.
- **M3 spend.** ≈ 27.8 through EXP-0055 (≈ 30.6 upper) + ≈ 0.1 = **≈ 27.9 vCPU-h billed
  (≈ 30.7 upper)** of the original 40; against the fresh 40 vCPU-h counter the user set before
  this task, ≈ 0.1.
- **Independent check:** `python3
  openspec/changes/add-m3-development-audit/probes/exp0056_check.py
  experiments/EXP-0056-split-vs-rotation --atlas
  experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json --exp0053
  experiments/EXP-0053-teacher-on-mined-worlds --exp0054
  experiments/EXP-0054-role-attribution-mined --ckpt orderer=… --ckpt atlas=…` (stdlib only):
  contract failures header-only; 18 planned rows each once; one digest pair per world-episode,
  equal to both references; `split_check.csv` (floors from the atlas, every per-world quantity,
  sums, counts, reading) and `manipulation.csv` recomputed from `episodes.csv`; `reproduction`
  recomputed; pins; staged SHA-256. All pass, reading `joint`.
- **Files:** `spec.json` (pins, provenance, `pre_registered`, `reproduction`, `result`),
  `split_check.csv`, `manipulation.csv`, `contract_failures.csv`; `artifacts/episodes.csv`
  (gate + `trace_*` + `lead_*` columns, git-lfs).
