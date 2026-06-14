# Phase 1 Implementation Plan — Python PettingZoo AEC Environment

Port the Go domain (`emulation/domain/`) to pure Python, expose as a PettingZoo AEC environment for MARL training. Per ADR-0001 and ADR-0002.

The env, port, and a MAPPO smoke loop are built (ADR-0001–0006): observable per-iter eval curves, centralized critic, vectorized rollout, difficulty curriculum.

> **Superseded by the ADR-0025 supply-chain redesign (2026-06).** Track A below proved the *old* flat model (single-capacity producers, capacity-only products, recipe-splitting ordering agents) poses a learning problem. The domain is being rewritten around whole-product orders, multi-capacity producers, composite assembly, and inventory pipelining. **The Track A target numbers no longer apply** — baselines and targets must be re-derived against the new model. Track A results are kept below as the historical record of the old model. See **Track B** for the redesign rollout.

**Status (old model): bar cleared.** The trained policy survived the full horizon and beat the strongest baseline on the canonical seeds.

**Old target (no longer valid): beat `restore_when_low` + `even_split` — return ≈ 3230, ≈ 359 consumer completions, 0 % death over horizon 100.**

## Result

Greedy-eval of the trained MAPPO policy (best-on-eval, ADR-0011) on the `signal_check` harness, validated config (no curriculum, prod temperature `0.25`, `log_std_init=-2.0`, target-KL `0.008`, 100 iters):

| policy | cycles | dead% | cons✓ | cons✗ | return | capmin |
|---|---|---|---|---|---|---|
| restore_when_low + proportional (best baseline) | 100 | 0 | 359 | 41 | 3238 | 99 |
| **trained (mappo), seed 1** | **100** | **0** | **377** | **22** | **3725** | **156** |

Canonical seeds 1 / 2 / 5 finish alive: best return 3725 / 3514 / 4395, final 3568 / 3438 / 4050, 0 % death (ADR-0024). Repro in `docs/track-a-findings.md`.

## Track A — DONE (beat the baseline bar)

The original collapse (`dead 1.00`) was **not** an instability/tuning problem — it was a stack of trainer correctness defects. Diagnosed and fixed across `docs/track-a-findings.md` sessions 3–4:

| Defect | Fix |
|--------|-----|
| GAE walked buffer adjacency, not the real `(env,agent)` trajectory | ADR-0017 explicit successor links |
| PPO rollout cuts reset live episodes | ADR-0018 persistent rollout episodes |
| ValueNorm changed V(s) at fixed weights | ADR-0019 output-preserving PopArt |
| Producing policy stayed near-random | ADR-0020 temperature |
| Action aliases corrupted producing credit | ADR-0021 producing action mask |
| Critic omitted the acting decision context | ADR-0023 agent-conditioned critic |
| Trust region too permissive | ADR-0024 target-KL `0.008` |

The original Track A levers (A1 critic fix, A2 long-run + LR schedule, A3 eval-gated curriculum, A4 reward-shaping) were explored along the way. **The winning config uses no curriculum and no shaping** — A3/A4 were tried and ruled out, not needed (ADR-0009/0010/0016/0022 record the negative results). The real wins were the correctness fixes above.

## Open items (not Phase-1 blockers)

1. **Monotonic training stability — unresolved.** Convergence is non-monotonic: e.g. seed 5 evaluates dead iters 30–60, recovers ~70. best-on-eval ships the survivor and protects deployment, but the training curve still shows the dip. **No established mechanism** — loss-of-plasticity / feature-rank collapse is unmeasured speculation (no rank measurements, no PFO experiment). Open research.
2. **Full-seed robustness — unproven.** Only seeds 1 / 2 / 5 validated under the final config. Bad-init seeds (0,3,4,6,7,8,9) were never re-swept post-ADR-0018. The win is "survives on canonical seeds," not "survives on all seeds."

**Post-Phase-1 (separate arc):** the deep-research 6-phase roadmap — RL-per-decision-block, graph/history critic, investment & durability, planner/regulator. Not Track A.

## Track B — supply-chain redesign (ADR-0025)

Rewrite the domain around whole-product orders, multi-capacity producers, composite assembly, and inventory pipelining. Sequence bottom-up so each layer is testable before the next depends on it. `domain/` stays pure; all reward/space logic stays in `env/`.

**B1 — Domain: products & recipes.** Extend the process sheet to list input products (and quantities) alongside the capacity bundle. Mark products primal vs composite. Config (`config/phase1.json`, `domain/config.py`) grows an inputs map. A composite test fixture (a 2-level recipe) lands here. *Done when:* domain encodes a primal+composite product set and the recipe graph is queryable.

**B2 — Domain: multi-capacity producers + bundle auction.** `ProducingAgent` owns a set of capacity types; `produce()` tracks per-type remaining capacity and accepts bids atomically (clear only if every required type fits *this cycle*). Price = `tokens / total-capacity`. Cut-off stays scalar. **Remove** the single-type multi-cycle `in_progress` carry-over — a bundle that does not fit is rejected, not booked across cycles (large demand is handled by the ordering quantity split + TTL re-orders). *Done when:* `test_domain_*` covers atomic multi-type acceptance, single-cycle rejection of an over-large bundle, and per-type degradation.

**B3 — Domain: inventory + composite assembly (capacity reserved at acceptance).** Add the inventory subsystem: stock of inputs + finished goods, unit-count limit, assembly consumes inputs and produces finished goods. A composite bid that clears the auction **reserves its capacity at acceptance**. Inputs in stock that cycle → assemble immediately (consume inputs + capacity) → complete. Inputs missing → hold the reserved capacity, wait in-progress up to TTL; inputs arriving in time → assemble, TTL expiring first → **fail (rejected, −5)**. Reserved capacity blocks other bids — the intended pressure to keep headroom and not over-accept; replenishment removes the block by stocking inputs ahead. No assembly-time revalidation (capacity already reserved → no overcommit). Degradation is unchanged; reservations are **componentwise** `reservation[t]`; available-to-new-bids per type = `capacity[t] − Σ reservation[t]`. Each cycle after degradation, re-validate live reservations by re-running the atomic price-greedy auction over them against post-degradation `capacity[t]`; any that no longer fit → fail (−5). *Done when:* a composite completes same-cycle when inputs are pre-stocked; one waiting for inputs holds capacity, blocks a competing bid, and fails on TTL; degrading one type below its reservation fails only the reservation needing that type (cheapest first) and spares disjoint-type reservations; the capacity-unavailable overcommit case cannot arise.

**B4 — Domain: investment (inventory expansion) + system death + per-producer fund.** Add inventory-expansion investment alongside restoration/upgrade. Replace `distribute_investment_fund` with **two-stage** allocation: per producer by scalar cut-off price, then within each producer's budget by fixed priority **restoration → replenishment → upgrade → inventory expansion**, greedy to budget, unfunded requests expiring. Redefine termination: a capacity type at zero across all producers. *Done when:* fund is producer-first (issuing more orders does not raise a producer's share); episode ends on the new death condition; inventory limit grows on completed expansion.

**B5 — Env: spaces + encode/decode.** Ordering action → per-order simplex over eligible producers: apportion **quantity first** via largest-remainder (`_allocate_int`), then **tokens following the realized integer quantities**, drop zero-quantity sub-orders. Producing action → `Box(G + 3)`, bounds `[−10, 10]`: `G` = **global** input-product vocabulary (sorted union over all composite recipes), then `restoration, upgrade, inventory_expansion`. Replenish decode is **joint over shared headroom** `H = inventory_limit − current_inventory_units − outstanding_inbound_units`: per-dim desired `d_g = round(clip(raw,0,BOUND)/BOUND × max_replenish)` (`raw ≤ 0 → 0`); if `Σ d_g ≤ H` use as-is, else scale to exactly `H` via largest-remainder. `max_replenish` = config constant (per-input per-cycle ceiling, default = initial inventory limit). Irrelevant-to-this-producer dims are no-ops. Toggle decode: `raw > 0 → request`, gated if already running. Producing obs gains multi-capacity + per-input current stock + inventory state; resize critic global state. *Done when:* `pettingzoo.test.api_test` passes; encode/decode round-trip tests pass; conservation tests hold (Σ sub-quantities = Q, Σ sub-tokens = order tokens, no zero-quantity bid); a producer's irrelevant replenish dims emit no order; replenish dims whose desired sum exceeds `H` order exactly `H` total, proportionally split.

**B6 — Env: reward.** Add the flat per-unit inventory holding penalty to the env reward (keep `+10/−1/−5`). *Done when:* reward unit test shows the penalty scales with units held.

**B7 — Eval: re-derive baselines + targets.** Update `eval/baselines.py` (ordering baselines now produce quantity splits; add a stock-aware producing baseline). Re-run `signal_check` to find the new strongest baseline = the new RL bar. Keep `test_signal_exists` green. *Done when:* a sensible baseline clearly beats do-nothing on the new model, and a new target row is recorded here.

**B8 — Train: rewrite producing actor + re-validate MAPPO (ADR-0026).** Not a plain rerun. Switch the producing actor `CategoricalActor → GaussianActor` (both policies now continuous; drop the discrete rollout branch). Retire the producing action mask (ADR-0021); reinterpret temperature as Gaussian `log_std` (ADR-0020) and `investment_request_cost` as toggle count ≤3 (ADR-0022). **All** model + optimizer state starts fresh (both actors, critic, PopArt, normalizer) behind a checkpoint version tag that rejects old/missing-version loads. Carry forward the model-agnostic fixes (ADR-0017/18/19/23/24). *Done when:* trainer builds against the new spaces, old checkpoints are rejected loudly, and the trained policy beats the B7 baseline on canonical seeds.
