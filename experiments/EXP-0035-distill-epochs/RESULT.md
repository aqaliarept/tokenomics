# EXP-0035 — Distill fix trial: epochs 2000 breaks the fidelity wall

Fix trial licensed by the EXP-0034 root-cause chain (low harvest → not generalization
→ not representability/capacity → **multi-task optimization/interference**). Cheapest
licensed knob first: rerun the EXP-0034 augment-replay **identically except
`--epochs 2000`** (was 400), hidden 64 = champion architecture unchanged.

## Question

Does a 5× fit budget let one producer scorer absorb the per-task investment behavior
of 121 topologies (57 mined + 64 broad augment) that 400 epochs averaged away — i.e.
does train/val/blind closure of the mined (bar − champion) gap jump well above the
EXP-0034 candidate's ~0.16, judged by the `eval.train_closure` standing gate
(DIAGNOSTICS entry 1)?

## Setup

Same recipe as `EXP-0034 replay/` byte-for-byte except epochs: archive =
EXP-0034 `recoverable_archive.jsonl` (113 tasks, split_seed 0 → same 57/28/28
train/val/blind), champion producer = EXP-0030, orderer frozen = EXP-0012,
`--augment --freeze-orderer --depth 3`. Fit MSE kept falling past the old cutoff
(0.045 @ 400 → 0.027 @ 600 → converged by 2000) — the under-training signature,
visible in the live log. One fresh 16-vCPU CPU pod per stage, auto-terminated.

## Result — fidelity wall BROKEN; all gates pass

Closure = (candidate − champion) / (bar − champion), unit sums per split:

| split | EXP-0034 (400 ep) | **EXP-0035 (2000 ep)** | candidate − champion |
|---|---|---|---|
| train (57) | 0.159 | **1.096** (median/task 1.04) | +83.1u |
| val (28)   | 0.19  | **0.860** | +68.3u |
| blind (28) | 0.12  | **1.000** | +68.4u |

- Per-task: train 3/57 worse than champion (was 26/57), 42/57 closure > 0.7 (was 11);
  quartiles 0.70 / 1.04 / 1.55 (was −0.51 / 0.10 / 0.57). Val+blind: 1/28 worse each
  (was 12 and 10).
- **The EXP-0034 tail-gate task healed**: s54 (−107u, upgrade misallocation at the
  layer-0 primal — diagnosed in EXP-0034 RESULT) now +61.7u (939.67 vs 878.0), above
  its own mined bar.
- **Objective gate PASSES** (first Stage-4 replay to do so): val 1106.7 > 1038.4,
  blind tail 813.33 > 772.33, 0 contract failures, 0 transfer regressions (broad
  seeds 6–9), 0 deaths. `promote_recommended` stays false by design (human call).
- Candidate exceeds the bar on train (1.096) and matches it on blind (1.000): the
  distilled scorer generalizes the planner's behavior slightly beyond the search bar
  on in-family tasks.

**Verdict: the harvest bottleneck was exactly the diagnosed cause — under-training at
multi-task scale. One knob (epochs 400 → 2000) took harvest from ~1/6 of mined
headroom to ~all of it. No width increase, no recipe change, no staging needed.**

Consequences:
- **Promotion candidate**: `replay/artifacts/producer_candidate.pt` beats the EXP-0030
  incumbent on every split of the capacity-binding family with zero broad-family cost —
  strictly dominates the EXP-0034 advisory-fail candidate. Human promotion call open.
- **Mining unblocked**: Track-A round 2 (primal-tight cells) and Track B (scripted
  demand/shock traces) are off HOLD — the refinery is fixed; `--epochs 2000` is the
  new replay default posture, `train_closure` stays the standing gate per round.
- DIAGNOSTICS entries 1–2 discriminated the cause in two pod runs; registry pattern
  validated on first use.

## Artifacts

- `replay/` — full stage-4 replay layout (spec.json, summary.csv, RESULT.md,
  `artifacts/producer_candidate.pt` + val/blind/transfer CSVs)
- `train_closure/` — standing-gate output (57-task CSV + summary JSON)

## Reproduce

Two pod runs (`run_pod_training.py --create-pod --cpu --python python3.12`, staging
champion ckpts + archive + `gap_attribution.csv` under `tokenomics/_pod_ckpts/`):
`eval.curriculum_replay --augment --freeze-orderer --depth 3 --epochs 2000`, then
`eval.train_closure --split train` on the pulled candidate.
