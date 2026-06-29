# EXP-0014 — Portable stack smoke (C1)

Status: **DONE (PASS)** — the two portable modules run JOINTLY in one env. Producer
scorer/planner (EXP-0013) + orderer scorer/allocator (EXP-0012), no re-train. On
held-out depth-3 seeds 6–9 the joint stack **beats** the deterministic
`entity_planner + proportional` bar (mean ratio **1.109**, min **1.042**, zero death),
and every seed is **byte-identical under a capacity relabel** (4/4). Contract gates
(feasibility, relabel invariance, ratio harness) unit-tested
(`tests/test_joint_stack.py`, 3 passed; full suite 432 passed). Local eval only.

## Question
Do the two portable modules — producer `ProducerScoreModel → plan_producer_command`
and orderer `OrdererScoreModel → allocate_suborders` — coexist in one env and hold
value when run as the full portable stack, with no feasibility failure and no relabel
mismatch? See `spec.json`.

## Setup
- **Stack** (no re-train, both distilled checkpoints reused unchanged):
  - producer = `scorer_producing_policy` → `plan_producer_command` (EXP-0013 v2 gate-head
    ckpt; planner owns producer feasibility incl. restoration),
  - orderer = `scorer_ordering_policy` → `decode_ordering` → `allocate_suborders`
    (EXP-0012 ckpt; allocator owns eligibility + conservation).
- **Env** — `entity_producer_obs=True` *and* `product_stream_ordering=True,
  corrected_stream_k=True`. The producer entity view (v2) and the corrected v4
  product-stream ordering are **orthogonal** constructor flags; this is the first env
  that turns both on at once. Each learned policy reads its own white-box view off
  `env.system`, so the obs encodings don't interfere.
- **Bar** — deterministic `entity_planner + proportional` on the *same* joint env.
- **Eval** — held-out `build_scale_config` seeds 6–9, depth 3 (P=12), 5 episodes each,
  horizon 100. Reports value (units ratio, death) and a per-seed capacity-relabel gate.

## Result

| seed | P | joint_units | bar_units | ratio | dead | bar_dead | relabel |
|------|----|-------------|-----------|-------|------|----------|---------|
| 6 | 12 | 500.4 | 466.2 | 1.073 | 0 | 0 | identical |
| 7 | 12 | 502.4 | 482.0 | 1.042 | 0 | 0 | identical |
| 8 | 12 | 600.6 | 512.2 | 1.173 | 0 | 0 | identical |
| 9 | 12 | 554.6 | 482.8 | 1.149 | 0 | 0 | identical |

Mean ratio **1.109** · min **1.042** · max joint dead-rate **0.000** ·
relabel gate **4/4 byte-identical**. Full rows in `summary.csv`.

## Conclusion
**PASS.** All C1 gate conditions hold:

1. **Feasibility / coexistence** — both portable policies run together in one env with
   no feasibility failure; the deterministic planner (producer) and masked allocator
   (orderer) keep the joint command valid for any scores (proven for untrained models in
   `test_joint_portable_stack_drives_feasible_episode`).
2. **Value** — the full portable stack beats the deterministic `entity_planner +
   proportional` bar on every held-out seed (mean 1.109, min 1.042), zero death.
3. **Relabel** — a pure capacity-type rename leaves the joint episode byte-identical on
   all four seeds (units, cycles, death) — no relabel mismatch. The id-free contract
   carries through both modules at once.

**Reading the number.** The lift is **orderer-driven**: EXP-0012 showed the orderer
scorer beats `proportional` ~1.28× at depth 3, while the producer scorer *ties*
`entity_planner` (EXP-0013). The joint ratio (1.109) sits below the orderer-alone 1.28×
precisely because the bar here pairs the *strong* `entity_planner` producer rather than
`stock_aware` — i.e. the portable producer matches the planner and the portable orderer
adds the routing lift on top. The combined ladder's gateway (C1) is cleared; **out-of-
family recipe transfer (C3 / the open P2/O3 axis) is NOT covered here.**

## Rerun command
```
python -m tokenomics.eval.joint_stack \
  --producer-ckpt experiments/EXP-0013-portable-producer-scorer/artifacts/producer_scorer.pt \
  --orderer-ckpt experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt \
  --depth 3 --seeds 6-9 --eval-seeds 5 --horizon 100
```
Local eval (evaluation of existing checkpoints — no training). Reuses the EXP-0012 +
EXP-0013 artifacts in place; no new checkpoint emitted.

## Artifacts
- `summary.csv` — per-seed value + relabel-gate rows.
- (checkpoints live in `EXP-0012-.../artifacts/orderer_scorer.pt` and
  `EXP-0013-.../artifacts/producer_scorer.pt`; this experiment adds no new ckpt.)
