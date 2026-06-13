# RL environment design: observations, actions, reward, and policies

We made several interconnected design decisions for how the tokenomics simulation is exposed as an RL environment. These are recorded together because they form a coherent package — changing one would likely require reconsidering the others.

## Agent types and shared policies

Two policy networks total: one shared across all producing agents, one shared across all ordering agents (both consumer and producer ordering agents). This follows the CTDE (Centralized Training with Decentralized Execution) pattern — all agents of the same type share weights but act on their own local observations. We considered separate policies for consumer vs producer ordering agents, but with shared reward their training signal is identical, and more data for one network beats less data for two.

## Observation spaces

**Producing agent (16 floats):** Own capacity state (capacity, max_capacity, requested_capacity, degradation, upgrade_running, restoration_running — all normalized) plus global producer summary (capacity and cut-off price for every producer, normalized) plus cycle_progress. We initially considered a purely local observation (7 floats, own state only) but the investment decision is nearly meaningless without knowing if upstream producers can fulfill the restoration/upgrade product.

**Ordering agent (K order slots):** Up to K order slots (each: active flag, normalized token budget, capacity needed per type) plus producer summary (capacity and cut-off price per producer) plus cycle_progress. Ordering agents see producer state but not other ordering agents' orders — preserving the decentralized information design from DR001.

> **Correction (impl):** This ADR originally set K=2, reasoning a consumer agent has ≤1 order and a producer agent ≤2 (restoration + upgrade). That ignored **carry-over**: a still-processing order stays in the system and is re-placed next to a freshly placed order. With TTL=2 an order lives across cycles with counters {0,1,2} before timeout, so a consumer agent can hold up to **3** at once. K is now **3** (ordering obs = 3·(2+T)+2P+1 = 27 for this config). `decode` additionally even-splits any orders beyond K so the domain's "all orders addressed" invariant can never break.

## Action spaces

**Producing agent:** `Discrete(4)` — four combinations of (doRestoration, doUpgrade) booleans.

**Ordering agent:** Continuous simplex-normalized token distribution. The network outputs raw weights (one per producer-capacity slot per order). These are softmax-normalized per order to produce proportions summing to 1.0, then multiplied by the order's token budget to get actual token amounts. This converts a variable "split N tokens across M producers" problem into a fixed-size `Box` action space.

### How simplex normalization works (detailed)

The problem: an ordering agent must split an order's token budget across the producers that supply the required capacity types. For example, an order for Product 1 has 100 tokens and needs capacity-1 and capacity-3 — so the agent must decide how many tokens go to P1 (capacity-1) vs P3 (capacity-3). Valid splits: (60, 40), (25, 75), (100, 0), etc. — any non-negative pair summing to exactly 100.

RL neural networks output fixed-size vectors of real numbers — they can't directly output "integers summing to a constraint." The simplex normalization bridges this gap:

1. **Network outputs raw logits.** For K=2 order slots, each with up to `num_capacity_types` producer slots, the network outputs a flat vector. Example with 4 capacity types and 2 order slots: 8 raw numbers like `[-0.5, 2.1, 0.0, 1.3, 0.8, -1.0, 0.4, 0.2]`.

2. **Inactive slots are masked.** If order slot 2 is unused (padded), its 4 values are ignored. If an order only needs 2 of 4 capacity types, the other 2 are masked out.

3. **Softmax per order.** The remaining active values for each order are passed through softmax, which converts any real numbers into positive proportions summing to 1.0. Example: raw values `[-0.5, 2.1]` for order 1's two required capacity types become `[0.07, 0.93]`.

4. **Scale by token budget.** Proportions are multiplied by the order's token budget. `[0.07, 0.93] x 100 tokens = [7, 93]` tokens allocated to each producer.

The agent never learns token amounts directly — it learns ratios. Shifting a logit higher means "send more tokens this way." The environment handles the conversion to valid allocations.

## Reward design

Shared per-cycle reward across all agents: `+10 per completed order, -1 per in-progress order, -5 per rejected/timed-out order`. We designed this from scratch rather than using the Go emulator's penalty-only score (which was unverified). Key properties:

- **Positive signal exists** — completing orders is visibly good, not just "less bad"
- **Dense** — every cycle produces a meaningful signal
- **No capacity health shaping** — degradation pressure is captured indirectly through order failures, forcing agents to discover why investment matters

## Episode termination

Fixed horizon (configurable, default 100 cycles) with early termination on system death (all producers at zero capacity). No safety net against death spirals — this is a legitimate failure mode agents must learn to avoid.
