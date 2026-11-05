# EXP-0091: M5 L3 pilot on k5_rf5, 20 generations, seed 0 (task 5.1)

**Question.** Does ES on the producer residual over the final producer k5_rf5 improve beyond its floor?
- The pilot: 20 generations, opt seed 0, σ 0.05, pre-registration v6 `5cd0e302…`.
- Pre-registered L3: g_tr = centre − k5_rf5 on the 16 monitor world-episodes (replicates 0 / 1, equal weight)
  at generation 20, with F_tr = the floors rule; `feasible` if g_tr > F_tr, else `unresolved`.
- This is a training-set measure, not an acceptance read; L5 validation decides that.

**Setup.**
- Code b3af893, run from HEAD 2e6ec61 (records only). `producer_es --mode train --generations 20 --opt-seed 0`.
- Inputs: EXP-0090's bar, normaliser, margins and re-keyed rows (T = 5/3).
- Pod: secure 16 vCPU in **EUR-IS-1** (`j67gg4q7nhwko6`), 22:06–22:25 UTC. 0 contract failures.
- Read: `producer_read --mode pilot` → `pilot_read/`.
- Reported-only analyses (supervisor, before launch): `reported_only.py` → `reported_only.json`.
- The k = 6 arm was replayed locally with `k6_monitor_local.py`: 32 episodes, 1 worker, the monitor's job path
  and keys. The pod's replay of the same jobs failed, because its spawned workers re-ran the runpy-launched
  script.

## Registered reading: **`feasible`**: g_tr = **+467.5** vs F_tr 125.6

| monitor (16 world-episodes × 2 replicates) | generation 0 | 10 | 20 |
|---|---|---|---|
| centre units (k5_rf5: 3,871.4) | 3,871.4 | 4,276.5 | **4,338.9** |
| g_tr (centre − k5_rf5) | 0 | +405.1 | **+467.5** (+12.1%) |
| centre held units / cycle (k5_rf5: 282.3) | 282.3 | 446.5 | **499.2** (1.77×) |
| centre investment rejection (k5_rf5: 0.099) | 0.099 | 0.317 | **0.330** |
| centre deaths / episodes | 0 / 32 | 0 / 32 | 0 / 32 |

- **Training:** minibatch fitness rises from 1.003 (generation 1) to 1.11 (generations 13–20), with 0–1
  disqualified members per generation.

## Reported only: is the centre just "k ≈ 6"? No

**Centre vs `entity_planner_buyahead_k6_rf5` at generation 20** (same 32 monitor episodes and keys; digests
equal):

| | units | held units / cycle | deaths |
|---|---|---|---|
| k5_rf5 (base) | 3,871.4 | 282.3 | 0 |
| k6_rf5 | 3,936.5 (+65.1) | 307.9 | 0 |
| **centre (generation 20)** | **4,338.9** (+467.5; **+402.4 over k6**) | 499.2 | 0 |

**θ norms and the effective buffer per generation:**
- The mean k_g is evaluated on EXP-0090's recorded θ = 0 feature rows. It is an approximation: the centre's own
  features differ.

| generation | ‖θ_k‖ | ‖θ_b‖ | ‖θ_e‖ | mean k_g | median k_g | share k_g > 5.5 |
|---|---|---|---|---|---|---|
| 1 | 0.060 | 0.060 | 0.045 | 5.26 | 5.15 | 24% |
| 5 | 0.210 | 0.171 | 0.096 | 6.41 | 5.71 | 59% |
| 10 | 0.342 | 0.278 | 0.157 | 8.10 | 6.48 | 75% |
| 15 | 0.464 | 0.295 | 0.196 | 10.45 | 7.39 | 81% |
| 20 | 0.559 | 0.295 | 0.238 | **12.16** | **8.02** | 82% |

**θ_k at generation 20 (normalised features):**
- bias +0.28;
- shortfall EMA +0.29, headroom +0.27, blocked cover +0.15, log use +0.11;
- depth −0.18, inbound cover −0.09.

**What this says:**
- Part of the change is a uniform buffer increase (the bias, about ×1.3).
- Most of it is state-dependent: much more stock for inputs that recently fell short, have room or have
  blocked reservations, and less for deep inputs. The mean k_g of 12 against a median of 8 shows how skewed it
  is.
- θ_b (extra buying) plateaus from generation 13. θ_e (expansion) keeps growing, and investment rejections
  triple, presumably from the extra expansion orders (inferred).

## Conclusions
- **L3 `feasible`.** The pre-registered next step is L4: seed 0 resumed to 40 generations, then seed 1 from
  θ = 0 to 40.
- **The learned centre is not "k ≈ 6".** It beats k6_rf5 by +402 on the monitor, while k6_rf5 itself is only
  +65 over the base there. It learned a selective, much larger buffer.
- **Two flags for the reads:**
  - Held stock is 1.77× the base's. That is above v6's `stock_heavy` threshold (1.5×), which is reported, never
    deciding, and goes to the user at L5.
  - Investment rejections triple.
- These are training-set numbers. Generalization is L5's question.
- **Spend:** ≈ 5.0 vCPU-h billed (stop 14). M5-P ≈ 13.3 / 80.
