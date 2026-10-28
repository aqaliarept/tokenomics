# EXP-0046 — M2 exit: champion stacks shocked and unretrained

**Contract: HOLDS.** 0 contract failure(s); 120 shocked episode(s) and
120 shock-free twin(s) on stacks portable, flat.

Shock profile `shocks-i` (profile file c92d066085c0), horizon 100. Produced by
`python -m tokenomics.eval.shock_contract` (see `spec.json`) at commit
`d04ee94f711b790822f6226da22cb2feedaa5054` (OpenSpec change `add-shocks-i-drift`, task 8.1;
2026-10-28).

## Question

Do the current champion checkpoints load and complete shocked episodes, unretrained, with zero
contract failures (project-mission "Champion runs on a shocked ruleset unretrained")?

## Result

**What was asked.** M2 added shocks to the economy: unannounced capacity hits (at the start of one
cycle a producer loses 30–60% of its current capacity of some types and has to restore it by
normal investment) and producer outages (a producer accepts and completes nothing for 3–6
cycles). The exit question for M2 is whether the
policies we already have, trained before shocks existed and not retrained, still run cleanly
when shocks happen. "Cleanly" means no **contract failure**: no crash, no observation or action
outside its declared space, no episode that ends without every agent being marked done, and no
checkpoint that fails to load or match its pinned hash. That is the only thing this run gates.

**What was run.** Two stacks, both unretrained, horizon 100, shock profile `shocks-i`:
- the **portable stack** (EXP-0035 producer `producer_candidate.pt` + EXP-0012 orderer
  `orderer_scorer.pt`) on all 8 shocked development gate worlds of the registry v2 draft
  (sparse-wide, 4 worlds with 12 producers and 4 with 42), 10 episode seeds each: 80 episodes;
- the **flat champion** from `champion.json` (EXP-0007A orderer `m2_stream_stochastic.pt` +
  EXP-0004 producer `d3_rung_dagger_v4.pt`) on its own held-out economies, depth 3, topologies
  6–9, with `shocks-i` applied, 10 episode seeds each: 40 episodes.

Every shocked episode was paired with its **shock-free twin**: the same world and seed with the
shocks removed, so the demand is identical (all 120 pairs have equal demand digests, and all 120
twins have empty shock schedules). Comparing an episode with its twin isolates what the shocks
cost. 240 jobs, all present, none duplicated.

**Contract: it holds.** `contract_failures.csv` has only its header (0 failures), no row is
missing, and the driver exited 0. The M2 exit criterion is met.

**Deaths: none.** 0 of 120 shocked episodes and 0 of 120 twins ended in system death.

**Units against the twin.** The "unit ratio" below is total consumer units the shocked episodes
delivered divided by what their twins delivered (1.0 = shocks cost nothing).

| stack | producers | episodes | shocks per episode | shocked units (mean) | twin units (mean) | unit ratio | worst episode | episodes below twin |
|---|---|---|---|---|---|---|---|---|
| portable | 12 | 40 | 3 (2 hits, 1 outage) | 482.3 | 485.4 | 0.994 | 0.955 | 23 of 40 |
| portable | 42 | 40 | 11 (7 hits, 4 outages) | 1659.4 | 1669.5 | 0.994 | 0.966 | 33 of 40 |
| flat | 12 | 40 | 3 (2 hits, 1 outage) | 479.1 | 479.7 | 0.999 | 0.957 | 20 of 40 |

In plain terms, the shocks in `shocks-i` cost these stacks well under 1% of delivered units:
0.6% for the portable stack at both sizes, 0.1% for the flat champion. No single episode lost
more than 4.5%. Some episodes deliver more than their twin (up to +5%), so the
shocked run's path is not strictly worse. Per world the ratio ranges 0.987–1.002 (portable)
and 0.989–1.007 (flat).

**Resilience per kind.** For each shock, `resilience.csv` records two things against the twin:
the **dip** (consumer units the twin delivered minus what the shocked run delivered, from the
shock's start until recovery), and the **recovery time** (cycles after the shock ends until a
5-cycle window of delivered units is back within 5% of the twin's same window; 0 means
recovered at once). A row is **overlapped** when another shock was active during its span, so
its dip is shared.

| stack | producers | kind | shocks | recovered | median / max recovery (cycles) | mean dip (units) | overlapped |
|---|---|---|---|---|---|---|---|
| portable | 12 | capacity hit | 80 | 80 | 0 / 7 | 0.94 | 8 |
| portable | 12 | producer outage | 40 | 40 | 0 / 5 | 5.62 | 6 |
| portable | 42 | capacity hit | 280 | 280 | 0 / 6 | 1.34 | 88 |
| portable | 42 | producer outage | 160 | 159 | 0 / 4 | 5.54 | 75 |
| flat | 12 | capacity hit | 80 | 80 | 0 / 7 | 0.71 | 4 |
| flat | 12 | producer outage | 40 | 39 | 0 / 9 | 3.08 | 1 |

678 of 680 shocks recovered; 550 of them within the cycle the shock ended. Outages cost
noticeably more than capacity hits (mean dip 3–6 units vs about 1), which fits what they do: an
outage stops a producer's output for several cycles, while a one-off capacity hit leaves
capacity above what the sparse-wide and flat economies need in most cycles (the median hit dip is
0). This last point is a reading of the numbers, not a diagnosed cause.

The two unrecovered shocks are both producer outages that started at cycle 90, the latest onset
the profile allows, and ended at cycle 95 (portable-42, `dev-sparse_wide-42-shocked-02`, seed
417658043, producer p1_4) and 94 (flat, topology 7, seed 1117556223, producer p0_0). With a
5-cycle window and a 100-cycle episode, only one or two recovery windows remain after such a
late outage. They fell short by 2.9 and 0.8–1.8 units of the 95% threshold (75 vs 77.9; 22 and
21 vs 22.8). They are censored by the horizon, not shocks the stack failed to absorb in a way
that was observed.

Per-shock dips do not add up to the total loss: overlapped shocks share cycles, and after
recovery a shocked run can run ahead of its twin. The sums of dips (300 / 1,262 / 180 units for
portable-12 / portable-42 / flat) exceed the total shortfalls (126 / 405 / 22 units). Use
`summary.csv` / the unit ratio for the total loss.

**What changes.** M2's exit holds: both champion stacks run on the shocked ruleset unretrained
with zero contract failures and zero deaths, so the shock machinery is safe to put under the
measurement gate. Resilience and units are reported, never gating. The run also shows that
`shocks-i` is a mild perturbation for these stacks (≤ 0.6% of units). Whether that is mild enough
to stop shocked worlds from separating policies is not answered here. The registry v2
calibration screen (task 9.1, EXP-0047) measures the baseline roster on the same worlds.

## Details

**Command** (repo root):

```
python run_pod_training.py --create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12 \
  --vcpu 8 --mem 16 --disk 30 --pod-name exp0046-m2-exit --verify-source-commit \
  --train "-u -m tokenomics.eval.shock_contract \
    --prod-ckpt /workspace/tokenomics/_pod_ckpts/producer_candidate.pt \
    --orderer-ckpt /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt \
    --workers 8 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0046" \
  --pull "/workspace/tokenomics/runs/exp0046:./experiments/EXP-0046-m2-exit-shocked-champion"
```

The pull landed in `EXP-0046-m2-exit-shocked-champion/exp0046/` (source path without a trailing
slash); its files were moved up one level, unchanged.

**Inputs (pinned in `spec.json` `pins.checkpoints`).** The portable pair was staged from the
git-lfs files into `tokenomics/_pod_ckpts/` (deleted after the run); the flat pair and
`champion.json` are in the synced tree (`tokenomics/m2_inputs/`, `tokenomics/champion.json`).
Local SHA-256 of the staged / synced files, equal to the pins:
- `portable_producer` `8b51b77cb5ac763114120d3bcbcbc9cd2c566362fe6e599d67e8c8ac883e36ea`
  (EXP-0035 `producer_candidate.pt`);
- `portable_orderer` `d25388bfa33ea3fc82aebbac48c75c886ed5c091f6016f1ce36eeb4293744a8c`
  (EXP-0012 `orderer_scorer.pt`);
- `flat_producer` `16e470eb808297d893c8a1999dd9c9baa68b90a3e370a1eecaf61b89ec33fa20`
  (`m2_inputs/d3_rung_dagger_v4.pt`);
- `flat_orderer` `7d3f494784001f5a1ab1178db709e2960e3d702a736e79d3326bb613a7ab049c`
  (`m2_inputs/m2_stream_stochastic.pt`);
- `champion` `3f400116259d195a2f9be92d91ae8d765aa33d07e389fceac2f570a8aa5076ed`.

Registry `worlds-v2` draft, content hash `3032147f1f17…`; shock profiles `c92d066085c0…`;
source commit `d04ee94f711b790822f6226da22cb2feedaa5054`, tree sha256 `2e9473eb25a0…`
(59 files); `dry_run: false`, `acceptance_read: true`, `resumes: []`.

**Outputs.**
- `spec.json`: pins (grid, checkpoints, registry, profile, horizon, source), environment, result.
- `episodes.csv`: 240 rows (gate row format; subject `candidate` / `twin:candidate`), with
  `shock_schedule` and `fulfilled_trace` on every row.
- `resilience.csv`: 680 rows, one per shock that took effect.
- `summary.csv`: per stack (portable mixes both sizes; the table above splits them).
- `contract_failures.csv`: header only.

The per-size and per-kind tables above were computed from `episodes.csv` and `resilience.csv`
with a read-only script (completeness against `spec.json` grid, demand-digest equality, per
stack × size unit ratios, per stack × size × kind resilience).

**Compute.** Runpod secure cloud, cpu5c, 8 vCPU / 16 GB, `runpod/base:0.7.0-ubuntu2004`,
python 3.12.11, torch 2.14.1+cpu, pod `qgq3r6wly1y207` (created and terminated by the launcher).
- Pod life: created after 06:50:44 UTC, terminated before 06:53:49 UTC: ≤ 3.1 min ≈
  **0.41 vCPU-h billed** (sizing.md expected 0.6–0.8).
- Execution: `spec.json` `started` 06:51:41.8 → last output written 06:53:34.4 ≈ 1.9 min.
- After the run `runpodctl pod list --all` returned `[]`.

## Conclusions

- **M2 exit holds.** Both champion stacks (portable at 12 and 42 producers on registry v2
  shocked development worlds; flat champion on topologies 6–9) run `shocks-i` unretrained with
  0 contract failures, 0 deaths and every planned episode and twin present.
- `shocks-i` costs these stacks 0.6% (portable) and 0.1% (flat) of consumer units against the
  shock-free twins; worst episode −4.5%.
- 678 of 680 shocks recovered, median 0 cycles. Outages cost more than capacity hits. The two
  unrecovered rows are cycle-90 outages censored by the horizon.
- Not a value or acceptance reading of the gate; the shocked-world headroom question goes to
  EXP-0047.
