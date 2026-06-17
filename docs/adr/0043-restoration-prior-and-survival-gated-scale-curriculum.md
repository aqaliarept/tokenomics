# Survive-first scale curriculum: survival-gated difficulty + restoration prior

Status: accepted

## Context

After the ADR-0041 (outstanding-inbound obs) and ADR-0042 (fulfilment-only reward)
fixes made the scale environment a valid learning problem, the open task was to
**train a policy that beats the strongest observation-compatible baseline on the big
(depth-3 `build_scale_config`) economy**. The obs-compatible bars (the fair target —
ADR-0029 hides producer capacity from the ordering agent, so white-box routing is an
oracle, not a bar):

- **fixed seed 0**: `obs_stock_aware + even_split` = **163 units / 0% death**.
- held-out (seeds 10000+): `obs_price` = **253 units**.

Two failures blocked it, both about *getting the policy to train at full difficulty*:

1. **The gated curriculum (ADR-0009) stalled at difficulty 0.30.** The gate advances
   only when rollout fulfilment ≥ `advance_fulfil` (then default 0.5–0.7). At scale a
   competent policy fulfils only ~25% of demand (deep chains, TTL=2), so the
   fulfilment threshold was *structurally unreachable* — difficulty pinned at 0.30
   forever. The policy trained on eased demand and was evaluated at full difficulty, a
   train/eval gap: it either idled to survive (~12 units) or, when forced higher, died.
   Two 250–400 iter runs confirmed the pin (rollout fulfilment ~0.24 every iter).

2. **Greedy eval never restored capacity.** Greedy eval (the deployed policy) emits the
   Gaussian *mean*; `decode_producing` fires restoration only when the restoration
   toggle dim > 0; its init bias was 0. So the deterministic policy did not restore
   until it *learned* to via toggle exploration — and at full degradation it died of a
   **localized** capacity-type collapse (one producer loses one required type while
   total/mean capacity stays high — `capmin` ~100 at death), i.e. a per-product
   `is_dead` (ADR-0030), not global starvation.

## Decision

Two changes, mirroring fixes already proven on the smaller configs:

1. **Survival-gated difficulty for scale.** Gate on survival, not fulfilment: set
   `advance_fulfil` low (0.15) so difficulty climbs one `d_step` per evidence window as
   long as the rollout *survives* (`death_rate ≤ advance_below`). The sequential ladder
   (`staged_config`: qty → mixture → competition → degradation) climbs through the
   non-lethal axes quickly, then slows at the degradation axis where survival evidence
   actually gates it. The policy reaches and *trains at* full difficulty, where the
   fulfilment reward pulls fulfilment up. This is a hyperparameter choice, not a code
   change — the gate already supported it (ADR-0009).

2. **Restoration prior (`prod_restore_bias`, code change).** A positive output-bias
   init on the restoration toggle (symmetric with the P1a replenishment bias) so the
   greedy mean restores from iter 0 — a *survive-first prior*: PPO learns fulfilment
   from a living baseline instead of discovering survival by exploration. Extracted
   `producing_mean_bias(num_inputs, action_dim, replen_bias, restore_bias)` (pure,
   tested) and wired it into `train()` + a `--prod-restore-bias` CLI flag.
   **`restore_bias` defaults to 0.0, so the canonical phase1 run is bit-unchanged.**

## Consequences

- **Fixed big environment: BEATEN and verified.** Survival-gated training (400 iters,
  `advance_fulfil=0.15`, `low_cap_coef=3.0`, `death_penalty=10.0`; no DR, no restore
  prior needed) deploys a checkpoint scoring **227 units / return 2157 / 0% death** on
  fixed seed 0 vs the baseline's 163 / 1588 / 0% — **+39% units, higher return, equal
  survival**. noop is 0 units / 100% death. The survival-gate fix alone cleared the bar.
- **Generalization gap remains (the next rung).** The same checkpoint gets 136.8 units
  / 40% death on neighbour seeds 0–4 and 114.7 / 33% death on the held-out suite — it
  overfits the trained topology. The fix is domain-randomized training over scale seeds
  (ADR-0042 reward + the `dr=seeds` lever); the restoration prior targets the
  unseen-topology death. Held-out bar 253 units / <5% death is not yet cleared.
- `restore_bias=3.0` ≈ always-restore behaviour in greedy eval (a known survivable
  baseline). If it over-invests and starves replenishment off the shared fund, widen
  the restoration toggle's `log_std` so PPO can learn to switch it off near max — not
  yet observed to be necessary.
