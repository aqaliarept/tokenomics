# EXP-0045 — Champion development report at 12 and 42 producers (task 12.1)

**Verdict: REJECT.** Failing clauses: calm margin, shocked margin, no losing world, run-down,
starvation. Zero deaths. A development report, not an acceptance read: an honest starting point
for M1, and nothing is promoted. Run 2026-10-27 at commit
`f2ea280aac1d3eefb182d1cdba71fb1e4fd5b2f8`.

## In plain English

**The question.** How does the current portable agent stack do against the measurement gate,
before any further training? The stack is the gate subject (design D1): the EXP-0035 promoted
producer `producer_candidate.pt` and the EXP-0012 portable orderer scorer `orderer_scorer.pt`.
The gate compares a candidate, episode by episode, with the **hindsight baseline**: the best of
the frozen roster's hand-written baselines on that same episode. After the EXP-0044 screen that
roster is the `entity_planner` producer paired with five orderers (blend_0.5, even_split,
obs_price, proportional, random). The candidate passes
only if it:
- beats that baseline by a margin on average: ≥ 5% on calm worlds and ≥ 10% on shocked worlds,
  at each economy size;
- never loses a world on average;
- never lets the economy die;
- avoids two side effects:
  - *run-down*: ending an episode with less than X = 73% of its starting capacity ratio;
  - *starvation*: filling less than Y = 77% of any one consumer's orders.

**What was measured.** The candidate and the five baselines (the random one 5 times) played all
24 sparse-wide development worlds at 12 and 42 producers, every registered episode (3 per calm
world, 10 per shocked world). That is 1,280 episodes at horizon 100.

**The numbers.** "Ratio" is the candidate's delivered units divided by the hindsight baseline's:
1.00 means as good as the best hand-written baseline on that episode.

| size | regime | mean ratio | required | margin |
|---|---|---|---|---|
| 12 | calm | 1.011 | 1.05 | fail |
| 12 | shocked | 1.033 | 1.10 | fail |
| 42 | calm | 1.038 | 1.05 | fail |
| 42 | shocked | 1.041 | 1.10 | fail |

- *Overall the stack is slightly better than the best baseline:* world ratios run 0.94–1.10 at
  12 producers (median 1.012) and 0.99–1.07 at 42 (median 1.041). It wins 1–4% on average but
  nowhere reaches the required margins.
- *Losing worlds: 5 of 24*, all calm. At 12 producers: calm-01 (0.994), calm-02 (0.998), calm-03
  (0.941), calm-07 (0.990). At 42: calm-03 (0.986). The 12-producer calm-03 loss is the only one
  larger than 2%.
- *Deaths: none.* The economy survived every episode.
- *Run-down: 8 failures*: 8 of the 10 episodes of one world, 12-shocked-00. There the candidate
  ends episodes at 63–70% of its starting capacity ratio, under X = 73%.
- *Starvation: 22 consumer-level failure records in 19 episodes* (an episode fails once
  for each starved consumer):
  - at 12 producers: all 3 episodes of calm-05;
  - at 42 producers: all 3 episodes of calm-00, -01, -03, -04 and -05, plus 1 of shocked-02.
  In each, some consumer gets under Y = 77% of its orders filled (e.g. 0.71 at 42-calm-00). The
  stack's routing serves total demand slightly better than the best baseline but leaves some
  consumers clearly short.
- *Contract failures: none.* Every job ran; the demand digests matched across subjects; no world
  was non-gating.

**What changes.** M1's measurement gate is now in place and has a recorded starting point. The
current portable stack is about +1% (12 calm) to +4% (42) better than the strongest hand-written
producer. That is well short of the 5%/10% margins, with a handful of losing worlds and clear
starvation on calm 42-producer worlds. This is the bar that the next milestones (development
audit, evolution-strategies training on whole-episode outcomes) must move. Whether the
starvation comes from the orderer's split rule or from the producer's investment choices is not
diagnosed here.

## Details

**Command** (repo root). It was the first use of the task 11.2 launcher fixes: `--create-pod`
worked with no curl or shim, the pod image was resolved, and the launcher terminated its pod.

```
python run_pod_training.py --create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12 \
  --vcpu 8 --mem 16 --disk 30 --pod-name exp0045-champion-report --verify-source-commit \
  --train "-u -m tokenomics.eval.gate --mode report --subject portable \
    --prod-ckpt /workspace/tokenomics/_pod_ckpts/producer_candidate.pt \
    --orderer-ckpt /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt --split development \
    --workers 8 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0045" \
  --pull "/workspace/tokenomics/runs/exp0045/:./experiments/EXP-0045-champion-dev-report-12-42/"
```

**Inputs (pinned in `spec.json`).**
- Packaged frozen files from EXP-0044 attempt 2 (task 11.1):
  - registry `566946fe0c0363ec888f9a58bfe7bd480bcc948e049c93e9923088e2cab911e3` (screened,
    sparse-wide);
  - roster `c2832dbb44d26601f279fc04c0c483c543c5a5fcdc8cb2231090fc03f76a3731` (5 entity_planner
    members, frozen, R = 5);
  - side-effect parameters `e6651b2ebe541b18ade0a8904f27b0a08844e2a4f8ee0e231ca062514472f2c1`
    (X 0.7297, Y 0.7681).
- Checkpoints, staged from git-lfs into `tokenomics/_pod_ckpts/` (deleted after the run); their
  SHA-256 equal the `spec.json` pins and the EXP-0012 / EXP-0035 files:
  - producer `8b51b77cb5ac763114120d3bcbcbc9cd2c566362fe6e599d67e8c8ac883e36ea` (EXP-0035);
  - orderer `d25388bfa33ea3fc82aebbac48c75c886ed5c091f6016f1ce36eeb4293744a8c` (EXP-0012).
- Source tree sha256 `54627ee9…`, 52 files; split development, horizon 100, no filters
  (`dry_run: false`).

**Outputs.**
- `spec.json`, `episodes.csv` (1,280 rows).
- `worlds.csv`: per world, the candidate mean, baseline, ratio, deaths, side-effect failures and
  monitored metrics.
- `regimes.csv`, `verdict.json`.

**Local verification.** `python -m tokenomics.eval.gate --mode recompute --out-dir
experiments/EXP-0045-champion-dev-report-12-42` → `recomputed verdict reject; MATCH`.

**Compute.** Runpod secure cloud, cpu5c, 8 vCPU, `runpod/base:0.7.0-ubuntu2004`, pod
`ofeql4525xrg4j`.
- Pod life ≈ 17:09:52 → 17:16:05 UTC: ≈ 6.2 min ≈ 0.8 vCPU-h billed, ≈ $0.03.
- Report execution 17:10:33 → ≈ 17:16 (≈ 5.5 min).
- The launcher stopped and terminated the pod; `runpodctl pod list --all` returned `[]`.

## Conclusions

- The measurement gate runs end to end on frozen inputs. The current portable stack is
  rejected: +1–4% over the hindsight baseline (margins need +5% calm, +10% shocked), 5 losing
  calm worlds, run-down on one shocked 12-producer world, and starvation on six calm worlds
  (mostly at 42) plus one shocked episode. Zero deaths.
- This is the M1 baseline reading for the measurement arc; nothing is promoted.
