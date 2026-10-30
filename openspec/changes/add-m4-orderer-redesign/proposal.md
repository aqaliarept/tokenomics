# Proposal: M4 orderer redesign (gated ladder)

Revision 3 (2026-10-30). It answers review rounds 1 and 2 and applies the user's decisions on the
orderer information set (ADR-0058) and on deferring a visible outage status.

## Why

M3 chose the ordering role for M4: in EXP-0054, swapping the orderer closed c_O = 1.02 of the gap
against c_P = 0.12 for the producer. M3 also diagnosed why the deployed orderer trails plain
whole-order rotation:

- **What the learned part does.** It only ranks producers per (source, product) stream, once per
  cycle, from shared producer rows. Every agent of a product reads the same rows.
- **What fixed code does.** The routing: a softmax, a largest-remainder split, and one score row
  tiled over every slot (`train/orderer_scorer.py` `scorer_ordering_policy`, `env/allocator.py`).
- **The result.** About 98–99% of a product's concurrent orders, across all its agents, lead to
  one producer (EXP-0056).
- **How it was trained.** By regression onto a formula, log `makeable_qty_now`, never on outcomes.
- **What closes the gap.** EXP-0056 reads `joint`: whole orders and varied receivers together,
  which this decoder cannot express.

The atlas results point the same way (EXP-0051 addenda):

| Router | Beats `round_robin` on | Median vs `round_robin` |
|---|---|---|
| trained orderer | 13% of 639 worlds | worse with load and with makers per product |
| `makeable_split` (its own target, as a router) | | −26% |
| `makeable_weighted_whole` (the same signal, whole, one producer drawn per order) | 62% | +1.8% |

M4 (`establish-project-baseline` D1 / D5) is "bounded ES feasibility → candidate trained on ≤ 12
producers; exit: sealed 12-producer gate pass, both regimes". On 2026-10-30 the user chose a
gated ladder for the ordering role. The user also decided that orderers decide only from previous
cycles' information, and that diversity across agents comes from each agent's own history, not
from coordination.

## What Changes

- **ADR-0058, orderer information set.** An orderer decides from:
  - its own orders;
  - everything written before cycle t's `start_ordering` (incl. cycle t's consumer orders and
    producing phase); nothing written by any ordering agent in cycle t (`requested_load` is last
    cycle's value);
  - its own outcome history;
  - its own sampling stream.

  It never sees other agents' same-cycle bids, which would mean recursive re-planning; collisions
  are accepted. One shared portable network with no agent ids. It cites ADR-0056 for the
  teacher's information set.
- **Pre-pod local checks (before any pod run).**
  - Restore DIAGNOSTICS entries 6 / 9 / 11 / 12 from git history.
  - Offline, from EXP-0052 rows: the X / Y violation rates of the reference arms on loaded worlds,
    and the step-1 reading with `makeable_weighted_whole` substituted, written down as the
    prediction.
  - A minimal local trace of orders per agent turn, the composition of concurrent groups (within
    one agent or across agents), and whether per-agent outcome histories differ today.
- **Step 1, decoder check, no training.**
  - The current EXP-0012 scores, one producer drawn per order (keyed inverse CDF of softmax), each
    order routed whole.
  - Compared with the trained orderer, its whole argmax, `round_robin`, and
    `makeable_weighted_whole` with an atlas draw and with the same key (paired).
  - Read on the 12-producer loaded atlas train worlds; 42 producers reported.
  - Two pre-registered readings: the decoder reading (is the decode the bottleneck?) and a
    fidelity reading (do the scores carry their target?), plus score-fidelity metrics.
- **Step 2, per-order portable scorer trained on outcomes.**
  - New input: **agent-local history per producer**, the agent's own past bid outcomes by kind
    only (routed, accepted, completed, failed with dropped and TTL merged, rejected), plus latency
    and recency (EMAs). No diagnostic reason field is read. The shared rows stay.
  - **Whole orders only:** one producer per order. Splitting is out of scope for M4 step 2
    (EXP-0056 `joint`) and could come back only with a separate diagnosis.
  - Per-order sampling is kept, so that agents with empty histories still differ.
  - A versioned share-encoded choice decode with a validity-only allocator; the legacy decode
    stays byte-identical.
  - Bounded ES (D5) on a residual over the frozen EXP-0012 scores, with relative
    disqualification.
  - A gradient-signal probe comes before the pilot. Seed 0 runs first; the pilot is resumed, not
    re-run.
  - Readings with a fixed precedence. 4 or 8 episodes per world, by the paired detectable-effect
    rule.
  - A pre-registered herding check (cycle-to-cycle share swing; agent-local vs shared-only
    inputs).
  - A held-out confirmation, guarded by a tracked ledger.
- **Step 3, GNN, only if step 2 plateaus.**
  - First an env-side relational check (reservations, inbound and input cover; no same-cycle
    bids), with no archive restore.
  - Only after a `relational` reading is `graph_primitives_v1` restored from `archive-1`, with its
    `pending_bid` relations dropped (ADR-0058) and `reservation` relations kept.
- **Imitation guard.** Any imitation target first passes the EXP-0040 label-stability check
  (winner overlap ≥ 0.80) and DIAGNOSTICS 12's target-is-a-function check. The default plan uses
  none.
- **Budget.** Re-derived:
  - tranche A: 95 vCPU-h authorisation (53–83 expected);
  - tranche B1: 65 (relational check, released after `plateau`);
  - tranche B2: sized and approved at GNN licensing;
  - a launch-fit rule throughout (design D10).
- **Records.**
  - ADR-0058 (written in this revision);
  - ADR-0059 (choice decode);
  - DIAGNOSTICS entries;
  - EXP-0057 onward, the README and HISTORY §14.
- **Not changed.** `domain/` stays pure. Gate registry, roster, parameters and gate specs are
  human gates. Nothing above 42 producers.
- **Deferred: visible outage status (user decision 2026-10-30).** Orderers do not see producer
  outage status in M4. ADR-0055's "consequences only" stands, and agents demote failing producers
  from their own failure history. An explicit, visible producer outage status for all policies,
  the hand-written ones included, is deferred to its own ADR (M6 shocks II or a separate small
  change).

## Open user decisions (listed, not decided here)

1. **Roster v3.** Either add `round_robin` (and `even_split_rotating`) as members through a roster
   v3 draft and a baseline-only re-screen (≈ 4 vCPU-h), which raises the gate's hindsight
   baseline, or keep roster v2.
2. **Atlas worlds or families in a gate registry**, and which regions: the 42-producer
   wide-choice (f14) hard worlds, the 12-producer dense / f4 hard worlds, or none. Registry v2 is
   fanout 2 and light-load today.
3. **A 42-producer teacher read** (PoC-A / PoC-C at 42) and its spend. The ladder does not need
   it.
4. **How the sealed 12-producer gate exit is read for a redesigned orderer.** This covers:
   - which registry and roster (decisions 1–2);
   - candidate replicates for a sampled policy;
   - which producer the gate candidate carries (`entity_planner` is white-box; today's gate
     subject carries the EXP-0035 producer).

   It is asked only after a `confirmed` held-out read.
5. **Which producer the orderer trains with** (`entity_planner` / EXP-0035 / both). The default
   is `entity_planner`. It must be answered **before group 6**, the first ES pod run. If the
   answer is EXP-0035 or both, an EXP-0035-producer validation column becomes part of `success`
   and `confirmed`, and the ES runs are re-sized.
6. **The M4 budget** (design D10): tranche A 95 before the first pod run, tranche B1 65 after
   `plateau`, tranche B2 at GNN licensing.

## Capabilities

### New Capabilities
- `learned-orderer`: how the learned ordering policy decides, decodes, is trained and is judged.
  It covers:
  - the information set (everything written before cycle t's `start_ordering`, incl. cycle t's
    consumer orders and producing phase; nothing written by any ordering agent in cycle t) and
    agent-local history;
  - per-order choices behind a validity layer, with the legacy decode unchanged;
  - the decoder and fidelity check before training;
  - outcome training by bounded ES with pre-registered, ordered readings and relative
    disqualification;
  - the label-stability gate on imitation targets;
  - atlas-split evaluation with paired floors, a power rule, fixed references and held-out
    confirmation;
  - the herding check;
  - the gated GNN step;
  - per-decision cost.

### Modified Capabilities
- None. Gate, roster, acceptance and atlas requirements are unchanged. `development-audit`'s
  "Atlas evidence does not change the gate" covers the ladder's atlas reads.

## Impact

- **Code, all outside `domain/`:**
  - `env/decode.py` / `env/allocator.py`: the choice decode and the validity layer.
  - New `env/agent_history.py`.
  - `eval/routing_diagnostics.py`: the TS and MWWk arms.
  - `eval/routing_trace.py`: share swing, concurrent composition, history dispersion.
  - New `eval/decoder_check.py`.
  - New `train/per_order_scorer.py`.
  - New `train/order_es.py` (worlds / bar, loop / resume, validate / readings).
  - New `eval/m4_read.py`.
  - `eval/deployed_cost.py`: accepts the new subject.
  - The tracked pre-registration file in `train/registry/`, and the held-out ledger
    `experiments/m4_heldout_ledger.json`.
  - Only if the GNN is licensed: restore from `archive-1` `7aeef1f` without the `pending_bid`
    relations.
  - If agent-local history needs a per-sub-order outcome the domain lacks, it is added as additive
    pure data.
- **Docs:** `docs/adr/0058-orderer-information-set.md` (new), `docs/adr/0059-*`,
  `docs/DIAGNOSTICS.md` (entries 6 / 9 / 11 / 12 restored, plus new entries).
- **Records:** `experiments/EXP-0057…`, the README, HISTORY §14.
- **Compute:** Runpod secure CPU pods only. Every run is probe-sized, with a hard stop of 2 ×
  estimate (≤ 40), under the launch-fit rule.
