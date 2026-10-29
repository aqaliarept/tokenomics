# Experiments

Empirical training/eval runs and their results. **ADRs** (`docs/adr/`) record *design
decisions*; **experiments** record *what happened when we ran it*.

**Read [HISTORY.md](HISTORY.md) first** — it is the compacted narrative of the whole
record, grouped by arc, with the causes behind every closed branch. The table below is
only a lookup index into the per-experiment directories.

## Index

| ID | Date | Result |
|---|---|---|
| [EXP-0051](EXP-0051-headroom-atlas-screen/RESULT.md) | 2026-10-29 | Headroom atlas screen (M3 D12: world-space Latin hypercube, 512 worlds at 12 producers + 128 at 42 + 8 EXP-0050 anchors, 1 episode, 8 blind/hand-written bar arms + 3 reach arms with `entity_planner`, B2 / B_base ceiling, provisional classes with EXP-0050's spreads at 2 × SE; run commit 92f3177) — **SATURATION IS RARE OUTSIDE THE EASY CORNER; MEASURED HARD ≈ 13% AT 12, ≈ 27% AT 42 (PROVISIONAL)**: labels 12 producers saturated 21 / hard 213 / open 274 / flat 4; 42 producers hard 68 / open 58 / flat 1 / collapse 1, none saturated; the hard labels are an upper bound: the pre-registered floor used EXP-0050's easy-corner σ_rep (2.6 / 10–21), 18–35× below the screen's own `random_whole` R = 2 replicate SD (46–77 at 12, 156–343 at 42); with it 63–71 / 34 stay hard at 2 × SE (DIAGNOSTICS 21); saturation = low-ρ small-order corner (10 by fill, ρ ≤ 0.23) + 11 over-demanded worlds by the ceiling; hard worlds at lower ρ and, at 42, wide choice (f14 22/26 hard, f2 2/26); open at high ρ (bar fill ≈ 0.4); `makeable_weighted_whole` is the reach argmax on 226 of 281 hard worlds (median gain ≈ 5–6% of the bar, 12% at 42-f14); anchors reproduce EXP-0050 (80/80 rows; 12-f4, 42-f4/f7/dense saturated, dense-12 hard by +19 at fill 0.946); 0 contract failures, 0 B2 violations; selection 64 + 32 (72 hard, 24 open), split train 60 / validation 16 / held-out 16 / held-out-region 4; confirm (EXP-0052) decides; ≈ 7.3 vCPU-h |
| [EXP-0050](EXP-0050-routing-vs-fanout/RESULT.md) | 2026-10-28 | Routing versus producers per product (M3 D2 follow-up: EXP-0049 arms + `proportional_whole`, `even_split_rotating` on family worlds with 2–14 makers per product: sparse_wide 12 × f2/f4, 42 × f2/f4/f7/f14, dense 12/42; calm + `shocks-i`; 4 worlds/cell, 96 world-episodes) — **ROUTING MATTERS MORE WITH FANOUT; TRAINED ORDERER LEADS AT f4/f14; MECHANISM UNDIAGNOSED**: `round_robin` − best hand-written rule at 42 producers +116/+150 (f2) → +335/+421 (f7); routers that split lose more with fanout (`random` 1.07 → 1.42 makers per order, `random_whole` +254/+261 at f14; losses are TTL expiries, no-fit ≈ 0; `even_split`'s f7 collapse is likely its tie-break defect, not counted); spare-capacity routers still lose to blind choice (`two_choice_spare` − `random_whole` < 0 in 14/16 cells); trained orderer beats every v2 rule in 16/16 cells by mean (clearly in 12), clearly beats `round_robin` at 42-f4/f14 (+153/+151 at 42-f14), loses at f2 (clearly in the 2 shocked cells); splits as much as the failing rules but its parts complete (10–15 wasted vs 126–271); its distillation target is input-aware (`makeable_qty_now`), unlike `proportional`; deciding check: the teacher as a router + its whole-order version + TTL-expiry trace; ≈ 3.9 vCPU-h |
| [EXP-0049](EXP-0049-informed-routing-and-trained-trace/RESULT.md) | 2026-10-28 | Informed routing and the trained orderer's routing (M3 D2 follow-up: 3 spare-capacity whole-order routers, `entity_planner/trained_orderer`, `portable`, with roster v2 + 4 references, all 128 registry v2 dev world-episodes, R = 5) — **DIAGNOSED (Q1), PARTLY (Q2)**: the 3 spare-capacity rules do not beat blind whole-order spreading on registry v2, fanout 2 everywhere (fanout ≥ 3 untested; spare rarely binds: 1.6–1.9-unit orders vs 13–22 spare on average, no-fit losses ≈ 0; following it keeps agents on the same producer over time: `two_choice_spare` = argmax at fanout 2, receiver change 0.17–0.30 vs 0.50, −15 to −152 units vs `random_whole`, lost as TTL expiries; per-cycle herding does not track units; `spare_rotating_whole` = `round_robin`, skip fires 0–2.7%); trained orderer varies receivers like `round_robin` (0.64–0.67) deterministically, splits 29–40% of consumer orders; beats `random` on 126/128 episodes, loses to `round_robin` on shocked worlds; references reproduce EXP-0048 (2,304 rows), `portable` EXP-0046 (80/80); ≈ 2.5 vCPU-h |
| [EXP-0048](EXP-0048-random-routing-diagnosis/RESULT.md) | 2026-10-28 | Random-routing diagnosis (M3 D2: 21 routers incl. 15 diagnostic, `entity_planner`, all 128 registry v2 dev world-episodes, R = 5, tracer on) — **DIAGNOSED: rotation + whole orders, not randomness**: deterministic `round_robin` (rotating, whole) recovers random's edge in 3/3 dominated cells (4.72 / 2.51 / 1.62) and beats `random` on all 104 of their episodes; fixed random weights (`random_fixed_episode`) among the worst, −89 to −241 units vs `round_robin`; `proportional_whole` 0/3 (also herds); whole vs split at matched variation worth 0.78–4.04 of the edge; rule defect `even_split` lowest-index tie-break (`even_split_rotating` 2/3); packing, fund channel, scoring artifact ruled out; D3 roster-v3 decision → user; 4.4 vCPU-h |
| [EXP-0047](EXP-0047-registry-v2-calibration/RESULT.md) | 2026-10-28 | Registry v2 calibration screen (sparse-wide 12/42, `shocks-i`, roster v2 with `shock_reactive`) — **HEADROOM PASSED**: median best fill 0.904 / 0.875 (v1 0.934 / 0.886), no reserves; roster pruned 48 → 6 (v1's five `entity_planner` members + `entity_planner/shock_reactive`); X 0.752 (v1 0.730), Y 0.730 (v1 0.768); frozen v2 files written |
| [EXP-0046](EXP-0046-m2-exit-shocked-champion/RESULT.md) | 2026-10-28 | M2 exit, champion stacks shocked (`shocks-i`) and unretrained — **CONTRACT HOLDS**: 0 contract failures, 0 deaths, 120 episodes + 120 twins (portable 12/42 on registry v2 shocked dev worlds, flat topologies 6–9); units vs shock-free twin 0.994 portable / 0.999 flat; 678/680 shocks recovered (2 horizon-censored cycle-90 outages) |
| [EXP-0045](EXP-0045-champion-dev-report-12-42/RESULT.md) | 2026-10-27 | Champion development report (portable stack, sparse-wide 12/42) — **REJECT** (dev reading, not acceptance): mean ratio 1.011–1.041 vs 1.05/1.10 margins, 5 losing calm worlds, run-down 1 world, starvation 6 calm worlds, zero deaths |
| [EXP-0044](EXP-0044-gate-calibration-12-42/RESULT.md) | 2026-10-27 | Gate calibration screen at 12/42 producers — **HEADROOM FAILED**: dense-12 saturated (median best fill 0.954 > 0.95); other 3 cells pass; no frozen files. Retune probe: capacity scale 0.8–1.0 **does not fix it** (warm-up-dominated shortfall). **Attempt 2 (sparse-wide only): PASSED**, roster pruned to 5 entity_planner members, X 0.730 / Y 0.768, frozen files written |
| [EXP-0043](EXP-0043-scenario-rollout-surrogate/RESULT.md) | 2026-08-27 | Scenario rollout surrogate — **CANCELLED, NOT RUN** (rejected by the baseline plan; code on `archive-1`) |
| [EXP-0042](EXP-0042-bounded-online-rollout/RESULT.md) | 2026-08-24 | Confidence-bounded online rollout — **PASS**, depth 2 promoted (+15.3%, death-safe) |
| [EXP-0041](EXP-0041-gnn-generalization-residual-router/RESULT.md) | 2026-08-09 | Pooled MPNN residual router — **CANCELLED BY EXP-0040**, not run |
| [EXP-0040](EXP-0040-rollout-target-stability/RESULT.md) | 2026-08-01 | Marginal-value label stability audit — **SEARCH_ONLY** (stability failed, usefulness passed) |
| [EXP-0039](EXP-0039-gnn-observation-probe/RESULT.md) | 2026-07-31 | Relational GNN observation probe — **FAIL**, target not recoverable |
| [EXP-0038B](EXP-0038B-producer-product-observation-ladder/RESULT.md) | 2026-07-28 | Per-producer + system observation ladder — **GATE FAILED**, target retired |
| [EXP-0038A](EXP-0038A-missing-observation-audit/RESULT.md) | 2026-07-27 | Missing-observation audit — enrichment **NOT LICENSED** |
| [EXP-0038](EXP-0038-actor-teacher-error-microscope/RESULT.md) | 2026-07-19 | Actor/teacher error microscope — **SCOPED DIAGNOSIS**: current row aliases the label |
| [EXP-0037](EXP-0037-value-routed-orderer/RESULT.md) | 2026-07-15 | Value-routed orderer residual — **BRANCH CLOSED** (1/3 tasks) |
| [EXP-0036](EXP-0036-track-a-round2-primal-tight/RESULT.md) | 2026-07-04 | Primal-tight round 2 — headroom **LARGE and ORDERER-owned**; choice learnability failed 3/3 |
| [EXP-0035](EXP-0035-distill-epochs/RESULT.md) | 2026-07-04 | Distill epochs 400→2000 — **FIDELITY WALL BROKEN**, all gates pass, **PROMOTED** |
| [EXP-0034](EXP-0034-stage4b-track-a/RESULT.md) | 2026-07-03 | Capacity-binding Track A — premise validated, tail-trades, ~15% harvest |
| [EXP-0033](EXP-0033-offline-ceiling-bound/RESULT.md) | 2026-07-03 | Offline ceiling bound — champion at **98.3% of exact demand** (near-bound) |
| [EXP-0032b](EXP-0032b-orderer-headroom-scan/RESULT.md) | 2026-07-03 | Orderer headroom scan over all 26 tasks — **DECISIVE: orderer lever spent** |
| [EXP-0032](EXP-0032-orderer-oracle-representability/RESULT.md) | 2026-07-03 | Orderer oracle representability — **INVALID PROBE** (superseded by 0032b) |
| [EXP-0031](EXP-0031/RESULT.md) | 2026-07-03 | Stage-4 gap-attribution diagnostic — producer replay closure **~5%** on the saturated family (dead lever) |
| [EXP-0030](EXP-0030-stage4-freeze-orderer/RESULT.md) | 2026-07-03 | Stage-4 freeze-orderer producer augment — **PROMOTED** by human override |
| [EXP-0029](EXP-0029-stage4-augment/RESULT.md) | 2026-07-02 | Augment (not replace) mined replay — **AUGMENT WORKS**, producer regression closed |
| [EXP-0028](EXP-0028-topology-grid-mining/RESULT.md) | 2026-07-02 | Topology-grid mining — **BLOCKER FIXED**, diverse 26-task archive |
| [EXP-0027](EXP-0027-stage4-replay-fixed/RESULT.md) | 2026-07-02 | Bug-fixed replay rerun — **REJECT stands**, now scientifically valid |
| [EXP-0026](EXP-0026-stage4-prioritized-replay/RESULT.md) | 2026-07-02 | Stage-4 prioritized replay — **REJECT**, single-cell archive closes nothing |
| [EXP-0025](EXP-0025-stage3-random-mining/RESULT.md) | 2026-07-02 | Stage-3 random mining — pipeline works at scale; noise floor load-bearing |
| [EXP-0024](EXP-0024-curriculum-reference-strength/RESULT.md) | 2026-07-01 | Short-rollout search reference bar — **BLOCKER RESOLVED**, first in-claim weakness |
| [EXP-0023](EXP-0023-curriculum-task-archive-and-stress-suite/RESULT.md) | 2026-07-01 | Curriculum task archive + stress suite — scaffold works; bar was the blocker |
| [EXP-0022](EXP-0022-ablation-flat-vs-row/RESULT.md) | 2026-07-01 | Flat-vs-row ablation — **EDGES = VALUE LEVER, NOT PORTABILITY LEVER** |
| [EXP-0021](EXP-0021-producer-broaden-training/RESULT.md) | 2026-07-01 | Broadened producer distillation — **PRIMARY TARGET CLOSED** (fork/join s9 0.955→1.000) |
| [EXP-0020](EXP-0020-broad-fixture-matrix/RESULT.md) | 2026-06-30 | Broad 22-fixture matrix — **ZERO schema/contract failures**; deepjoin-d5 = TTL wall |
| [EXP-0019](EXP-0019-dynamic-baseline-audit/RESULT.md) | 2026-06-30 | Dynamic baseline audits — **BOTH CLEAN**, contract vs policy stays separable |
| [EXP-0018](EXP-0018-producer-planner-gate-acceptance/RESULT.md) | 2026-06-30 | Deterministic planner investment gates — producer-alone gap **materially reduced** |
| [EXP-0017](EXP-0017-out-of-family-recipe/RESULT.md) | 2026-06-30 | Out-of-family fork/join recipe — **CONTRACT PASS**, joint 2.86×, producer-alone caveat |
| [EXP-0016](EXP-0016-size-transfer/RESULT.md) | 2026-06-30 | Size transfer (24 products / 24 producers) — **ACCEPT** |
| [EXP-0015](EXP-0015-verification-matrix/RESULT.md) | 2026-06-30 | Verification matrix runner — **ACCEPT**; contract ⊥ value |
| [EXP-0014](EXP-0014-portable-stack-smoke/RESULT.md) | 2026-06-29 | Portable stack smoke — **ACCEPT**, both modules coexist |
| [EXP-0013](EXP-0013-portable-producer-scorer/RESULT.md) | 2026-06-29 | Portable producer scorer — **ACCEPT** (P1 value + producer-count transfer) |
| [EXP-0012](EXP-0012-portable-orderer-scorer/RESULT.md) | 2026-06-28 | Portable orderer scorer — **ACCEPT** (O2 value) |
| [EXP-0011A](EXP-0011A-deterministic-producer-champion-audit/RESULT.md) | 2026-06-26 | Deterministic producer champion audit — **REJECT** replacement promotion |
| [EXP-0010D](EXP-0010D-producer-size-holdout-audit/RESULT.md) | 2026-06-26 | Producer size-holdout audit — **ACCEPT** gates, no learned-scorer promotion |
| [EXP-0010C](EXP-0010C-learned-producer-scorer/RESULT.md) | 2026-06-25 | Learned producer scorer — **ACCEPT** infra, **REJECT** controller |
| [EXP-0010B](EXP-0010B-deterministic-producer-planner/RESULT.md) | 2026-06-25 | Deterministic producer planner — **ACCEPT** controller slice |
| [EXP-0010A](EXP-0010A-producer-entity-view/RESULT.md) | 2026-06-25 | Entity-factored producer view — **ACCEPT** (obs-contract slice, ADR-0053) |
| [EXP-0001](EXP-0001-boundary-grid/RESULT.md) | 2026-06-21 | Boundary grid — wall is chain **DEPTH**, blocker is the **producer** |
| [EXP-0002](EXP-0002-producer-warm-start/RESULT.md) | 2026-06-22 | Producer warm start — **PASS**; plain MSE fails on the gated decode |
| [EXP-0003](EXP-0003-producer-standalone-improvement/RESULT.md) | 2026-06-23 | Producer-only RL on the clone — **REJECT** (plateaus, critic EV≈0) |
| [EXP-0004](EXP-0004-rung-quantity-weakness/RESULT.md) | 2026-06-23 | Rung-staged DAgger — weakness is **cloning fidelity**, not the expert; v4 accepted |
| [EXP-0009](EXP-0009-value-guided-planner/RESULT.md) | 2026-06-25 | Value-guided planner — **ACCEPT** compute lever (3.82×), but **value lever STOPPED** |
| [EXP-0008](EXP-0008-elite-set-distillation/RESULT.md) | 2026-06-25 | Elite-set distillation — **REJECT** promotion |
| [EXP-0007B](EXP-0007B-champion-iteration-r1/RESULT.md) | 2026-06-25 | Champion-iteration driver — **ACCEPT** automation, no promotion |
| [EXP-0007A](EXP-0007A-product-stream-ordering/RESULT.md) | 2026-06-24 | Product-stream ordering (ADR-0052) — **ACCEPT**, current champion orderer |
| [EXP-0006B](EXP-0006B-orderer-stochastic/RESULT.md) | 2026-06-24 | Stochastic orderer — **ACCEPT (strict)**, first to clear the hindsight baseline 4/4 |
| [EXP-0006A](EXP-0006A-orderer-observation/RESULT.md) | 2026-06-24 | Enriched orderer observation — **REJECT (strict)**, partial lever |
| [EXP-0005](EXP-0005-learning-vs-baseline/RESULT.md) | 2026-06-23 | Learning vs baseline + headroom audit — improvement **PROVEN**, partial vs strict |

## Layout & conventions

Each experiment is a self-contained dir `EXP-NNNN-kebab-title/` (numbering parallels ADRs —
sortable, stable, citable):

```
EXP-NNNN-title/
  spec.json        # declarative reproducer: driver, params, seeds, configs, compute, date
  RESULT.md        # the finding — question, method, result table, conclusions, follow-up
  summary.csv      # machine-readable per-cell/per-run metrics
  configs/         # materialized INPUT configs (the experiment definition)
  artifacts/       # checkpoints (.pt), raw logs, per-iter curves  → git-lfs (see .gitattributes)
```

- **`spec.json` / `RESULT.md` / `summary.csv` / `configs/`** are normal git (small, diff-able —
  they *are* the durable record).
- **`artifacts/`** is tracked via **git-lfs** (`.gitattributes`). Run `git lfs install` once
  after cloning.

### Artifact retention

Bulk artifacts (raw label dumps, per-episode shards, prediction CSVs, training logs) have been
pruned from the working tree — they remain in git history. What is kept in-tree is the record
(`spec.json` / `RESULT.md` / `summary.csv` / `configs/`) plus the small set of checkpoints that
something live still reads:

| Kept artifact | Why |
|---|---|
| `EXP-0004/.../d3_rung_dagger_v4.pt` | champion producer (`champion.json`) |
| `EXP-0007A/.../m2_stream_stochastic.pt` | champion orderer (`champion.json`) |
| `EXP-0006B/.../m2_v3_obs_stochastic.pt` | `test_champion_regression` provenance guard |
| `EXP-0012/.../orderer_scorer.pt` | portable orderer scorer (champion stack, HISTORY §0) |
| `EXP-0021/.../producer_scorer_broad.pt` | portable producer scorer (champion stack, HISTORY §0) |
| `EXP-0035/.../producer_candidate.pt` | promoted producer; SHA-pinned by the EXP-0042 teacher (`REGISTERED_INPUT_SHA256`) |
| `EXP-0036/.../task_archive.jsonl` | task archive; SHA-pinned by the EXP-0042 teacher (staged copy: `tokenomics/_pod_inputs/gate_archive.jsonl`) |
| `EXP-0037/gate/residual_scorer.pt` | actor floor of the closed graph gate; no live reader, kept to reproduce from `archive-1` |
| `EXP-0038B/audit_sys/spec.json` | actor-floor artifact of the closed graph gate; no live reader, kept to reproduce from `archive-1` |
| `EXP-0038/dump/` | EXP-0038 decision dump; no live reader, kept to reproduce from `archive-1` |

Before deleting anything else here, check it is not SHA-pinned: the EXP-0042 teacher
verifies its inputs by hash and will refuse to run against a substitute.

### Code for closed arcs

Drivers for closed or rejected arcs (PPO trainer, curriculum game-master, marginal-value
router, graph branch, EXP-0043 surrogate) were removed from `new-model` on 2026-10-25 and
are preserved on branch `archive-1` (commit `7aeef1f`). A `spec.json` whose `driver` names
a module missing from the tree reproduces from that branch.

## Producing a new experiment

Follow the dir contract above (see `_TEMPLATE/`). Point the driver's `--out-dir` at a fresh
`EXP-NNNN-<title>/`, then add an index row and, if the result closes or opens an arc, a
paragraph in `HISTORY.md`. For off-box runs, wrap with `run_pod_training.py` and `--pull` the
EXP dir back.
