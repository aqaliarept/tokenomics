# EXP-0009 value-guided planner — short rollout + learned terminal value vs full rollout

ACCEPT the compute lever; the win is **short-rollout sufficiency, NOT the learned value.**

The orderer search teacher (`eval.orderer_oracle`) scores every candidate routing by a full
`lookahead=12`-cycle deepcopy rollout — the dominant label-gen cost. EXP-0009 replaces the
tail: roll only `short=3` cycles under a candidate, then add `V(global_state)` as the learned
estimate of the remaining `window=9` cycles' consumer units. `V` reads the existing
centralized-critic system state (`env.global_state()`, ADR-0004) — no new state contract.

Run: depth-3, v4 producer held fixed (`d3_rung_dagger_v4.pt`), orderer continuation
`proportional`, candidates `{even_split, proportional, obs_price}` + 12 sampled routings.
Value model fit on 1200 (state, units-to-go) rows from train topologies 0–5 (2 episodes
each); planner compared on held-out topologies 6–9 (5 episode seeds each). Runpod CPU.

## Held-out (value-guided vs rollout-only teacher, 5 episode seeds)

| seed | value_units | rollout_units | ratio | value_cycles | rollout_cycles | agreement |
|---|---|---|---|---|---|---|
| 6 | 503.2 | 499.4 | **1.008** | 4455 | 17010 | 0.242 |
| 7 | 492.8 | 486.8 | **1.012** | 4455 | 17010 | 0.234 |
| 8 | 588.8 | 589.2 | 0.999 | 4455 | 17009 | 0.228 |
| 9 | 543.0 | 543.8 | 0.999 | 4455 | 17010 | 0.134 |

Compute: **3.82× fewer simulated cycles** (4455 vs ~17010) on every seed.

## Conclusions

**The compute lever holds: at 3.82× fewer simulated cycles the value-guided planner is units-
neutral-or-better in aggregate.** Under the strict `value_units >= rollout_units` criterion it
**beats on 2/4** held-out (s6 +3.8, s7 +6.0) and **falls within 0.15% on the other 2** (s8 −0.4
= −0.07%, s9 −0.8 = −0.15%) — i.e. practical ties, not strict wins. **Held-out mean +2.15 units
(+0.4%)** (531.95 vs 529.80). So the accurate claim is *mean-positive / practical-tie at lower
compute*, not strict match-or-beat on every seed. As a cheaper teacher for label generation,
the short-window planner is a drop-in.

**But the mechanism is not the one the experiment set out to demonstrate.** The value model
did **not** learn a useful state-dependent tail:

- The fit MSE plateaued at **146.60 immediately** (flat from epoch ~187 of 1500; rmse ≈ 12.1
  units) — `V` collapsed to ≈ the mean target. A near-constant terminal value cannot change
  the short-rollout ranking (a constant shifts every candidate's score equally — the exact
  property pinned in `test_valued_scoring_reduces_to_rollout_and_composes_terminal_value`).
- **Ranking agreement with the full rollout is low: 0.13–0.24.** The per-cycle argmax under
  the value-guided scorer rarely matches the full-rollout argmax.

So the savings come from the **short 3-cycle rollout being nearly outcome-sufficient at this
depth/candidate set**, not from `V` carrying the tail. The many per-cycle disagreements
(low agreement) are near-ties in episode value, so they wash out and final units land equal.
This is the honest finding: at depth-3, a 3-cycle horizon already separates the routings that
matter for the episode outcome; the learned value adds essentially nothing on top.

**Verdict:** the value-guided planner is **accepted as a cheaper teacher** (3.82× fewer
simulated cycles, held-out mean +0.4% units — 2/4 strict beats, 2/4 sub-0.15% ties), but the
**learned value model is not yet contributing** and
its near-constant fit is a real weakness. Before claiming `V` substitutes for the rollout
(which is what would extend the *effective horizon* beyond what a short rollout sees), the
regressor needs work: input feature scaling / normalization of `global_state`, target
normalization, capacity, or lr schedule (the instant plateau looks like a mean-collapse, not
a capacity ceiling). The terminal-value tail is also correctly **zeroed at terminal states**
(system death / horizon) so terminal candidates get no phantom score
(`test_value_tail_is_zeroed_at_terminal_states`, review fix).

**Next:** (a) diagnose/fix the value-model mean-collapse (normalize `global_state` inputs +
targets; sanity-check target variance) and re-run — does a non-trivial `V` lift ranking
agreement and let a *shorter* `short` (1–2) or *longer effective window* hold units? (b) only
then re-test the original EXP-0009 promise (extend planning horizon at fixed compute).

## Artifacts
- `summary.csv` — the held-out table above (per-seed units, cycles, agreement).
- Producer held fixed: `m2_inputs/d3_rung_dagger_v4.pt` (EXP-0004 v4 champion clone).
- Machinery (TDD): `tokenomics/train/value_model.py`, planner-side in
  `tokenomics/eval/orderer_oracle.py` (`_rollout_window`, `_score_candidates_valued`,
  `planner_orderer_episode`, `ranking_agreement_episode`), tests in
  `tests/test_value_planner.py`.

## Reproduce
```bash
python run_pod_training.py --pod-id <cpu-pod> --cpu --python python3.12 \
  --train "-u -m tokenomics.train.value_model --depth 3 \
    --prod-ckpt /workspace/tokenomics/m2_inputs/d3_rung_dagger_v4.pt \
    --train-seeds 0,1,2,3,4,5 --held-out-seeds 6,7,8,9 \
    --lookahead 12 --short 3 --samples 12 --csv-out /workspace/tokenomics/heldout.csv" \
  --pull /workspace/tokenomics/heldout.csv:./experiments/EXP-0009-value-guided-planner/summary.csv
```
