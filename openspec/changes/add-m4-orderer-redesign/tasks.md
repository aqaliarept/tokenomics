# Tasks

Local budget (probed 2026-10-30): 2 cores, 3814/1810 MB RAM (total/available), 24 GB free disk, load 0.34.
Shared by supervisor + implementer + reviewer. Classes: local | local-exclusive | remote:<profile>.
Paths are under /srv/research/tokenomics/tokenomics/ unless absolute. "Full suite" = `cd /srv/research/tokenomics/tokenomics && ../.venv/bin/python -m pytest -q` (~2.5 min on this box, local-exclusive). "Goldens" = `/srv/research/tokenomics/tokenomics/tests/test_legacy_invariance.py`, green and unedited in every code task. "Frozen hashes" = the v1 / v2 registry, roster and parameter files under `/srv/research/tokenomics/tokenomics/tokenomics/eval/registry/` keep their recorded content hashes. "Atlas" = `/srv/research/tokenomics/experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json` (content hash `220886a7…`). "Checkpoints" = `/srv/research/tokenomics/experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt` (`d25388bf…`) and `/srv/research/tokenomics/experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt` (`8b51b77c…`). "ADR-0058" = `/srv/research/tokenomics/docs/adr/0058-orderer-information-set.md` (no same-cycle cross-agent input; agents diverge by agent-local history and sampling).
Remote profiles (secure cloud only, never community): `cpu8` = `--create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12 --vcpu 8 --mem 16 --disk 30`; `cpu16` = the same with `--vcpu 16 --mem 32`; `cpu32` = `--vcpu 32 --mem 64`. Largest shape first (32 → 16 → 8) whose workers stay busy, with `--workers` = vCPU and longest-first job submission. `--pull` nests the run in a subdirectory: move the files up and `episodes.csv` into `artifacts/` (git-lfs). Delete staged `tokenomics/_pod_ckpts/` and `_pod_inputs/` after each run.
Budget (design D10), awaiting user approval: tranche A 95 vCPU-h (groups 3–9; 53.9 expected on the seed-0 path, 82.9 on the full success path; largest spend + stop 91.9, so a **3.1 vCPU-h margin**, 0.6 under 8-episode reads; the approval question also offers 100), tranche B1 65 (groups 10–11, released only after `plateau`), tranche B2 approved at GNN licensing (task 12.1). **Launch-fit rule:** a run launches only if the billed spend so far in its tranche plus its hard stop is ≤ the tranche. The supervisor records the check in the task. Hard stop = 2 × the probe estimate, ≤ 40 per run. A probe above its stop after its pre-registered reduction, or a launch-fit miss, goes to the user. Nothing above 42 producers. Gate, roster and registry files are never edited.
Ladder gates (design D1): a group marked **Gated** starts only when the named reading is recorded in its EXP `RESULT.md`. Any other reading stops the ladder: the supervisor inserts a diagnosis task (root-cause rule) and asks the user. EXP numbers are planned; an inserted diagnosis takes the next free number.

## 1. Pre-pod local checks (code; before any pod run)

- [x] 1.1 [code][local] Restore DIAGNOSTICS entries 6 ("Model looks good — is it memorizing?"), 9 ("Pod run stalls / crawls at high load"), 11 ("Distill-from-search fails — is the TEACHER SIGNAL even learnable?") and 12 ("Actor fits the teacher badly — missing observation, or an unlearnable target?") verbatim from `git show 18c717d^:docs/DIAGNOSTICS.md` into `/srv/research/tokenomics/docs/DIAGNOSTICS.md`. They keep their numbers and are placed before entry 17. Each gets a one-line "Restored 2026-10-30 from 18c717d^ (referenced by add-m4-orderer-redesign)" note. Update the "History:" line to say which entries are back. Verify: `diff` of each restored section's body against the `git show` text is empty apart from the note line, and the numbering is monotonic (`grep -n '^## ' /srv/research/tokenomics/docs/DIAGNOSTICS.md`).
- [x] 1.2 [code][local] Offline checks (c), (d) and (e) (design D11), no episodes. Write `/srv/research/tokenomics/openspec/changes/add-m4-orderer-redesign/probes/pre_pod_offline.py`, which reads `/srv/research/tokenomics/experiments/EXP-0052-headroom-atlas-confirm/artifacts/episodes.csv` and the Atlas.
  - (c): on loaded (ρ > 0.28) non-collapse world-episodes at 12 and 42, the shares of rows of `entity_planner/round_robin`, `makeable_weighted_whole` and the trained orderer that die, violate run-down (r_end < 0.7516·r_start) or leave any consumer below Y = 0.7302 (per-consumer `consumer_fill_rate`), plus the share where one arm violates and another does not.
  - (d): the step-1 primary reading of design D3, with its precedence (`no_gap` first) and the n = 1 floor rule, computed with MWW in place of TS on the 23 loaded 12-producer train worlds and the confirm's seeds. It is recorded as the written prediction for EXP-0057 under `faithful` (gap worlds, rec_w, branch).
  - (e): the paired per-episode MWW − RR differences on the 9 loaded 12-producer validation worlds, with the implied detectable effect at 4 and 8 episodes as a share of the mean validation bar (a first look at D7's power rule; task 5.4 recomputes it).
  - Output: `/srv/research/tokenomics/openspec/changes/add-m4-orderer-redesign/notes/pre_pod_checks.md` §c, §d and §e.

  Verify: the script reruns to identical output in seconds; it lists the subject names it matched and the world counts (23 / 14 / 9); §d names the predicted branch; §e states the mean bar it used.
- [x] 1.3 [code][local] Minimal local trace for (a) and (b) (design D11). The recorded rows carry only per-episode aggregates, with no per-agent, per-turn or per-agent-producer fields, so a trace is needed. Write `/srv/research/tokenomics/openspec/changes/add-m4-orderer-redesign/probes/trace_agents.py`, which wraps the ordering policies and reads only outcome kinds (accepted / completed / failed / rejected) after each `complete_cycle`, never a reason field. It replays 12-0060, 12-0111 and 12-0344 (first confirm episode, horizon 100, `entity_planner` with TR, RR and MWW at R = 1) and one loaded 42-producer `sparse_f14` train world (TR only), with `--workers 1`.
  - (a): orders per agent turn by source class (consumer / producer stream), and the across-agent share of concurrent same-product order groups.
  - (b), read under MWW, the A0-like sampling regime (TR and RR are context only):
    - `agent_history_dispersion`: the median, over products with ≥ 2 agents and producers that ≥ 2 agents routed ≥ 3 units to, of the across-agent SD of own completion rate;
    - `routed_share_dispersion`: the median across-agent total-variation distance between agents' routed-share vectors.
  - Output in notes §a and §b, with the statement that this is descriptive (EXP-0057's dispersion reading repeats it at scale under TS).

  After 1.2. Verify: each command finishes in < 2 min; the 12-producer TR / RR units equal EXP-0056's rows for those world-episodes (the trace is neutral); notes §a and §b give the numbers. **Numeric triggers** (design D11): if the across-agent share in (a) is < 0.5, or under MWW `agent_history_dispersion` < 0.05 or `routed_share_dispersion` < 0.10 on ≥ 2 of the 3 12-producer worlds, the task ends by reporting to the supervisor, who asks the user before group 3.

## 2. Step 1 — decoder check instruments (code)

- [x] 2.1 [code][local-exclusive] Add TS and MWWk, plus the new trace columns (design D3).
  - In `/srv/research/tokenomics/tokenomics/tokenomics/eval/routing_diagnostics.py`, add TS `trained_sampled_whole` and MWWk `mww_keyed`. Both take a per-slot inverse-CDF draw from the SHA-256(world, episode, replicate, agent, cycle, slot) uniform: TS over softmax(EXP-0012 score) at T = 1, MWWk ∝ `makeable_qty_now`, uniform when all are zero. Each routes whole via `_whole`. TS goes in `TRAINED_WRAPPERS` and MWWk in `DIAGNOSTIC_ROUTERS`, both outside the roster.
  - In `/srv/research/tokenomics/tokenomics/tokenomics/eval/routing_trace.py`, add:
    - `lead_concurrent_top_share`, ported from `teacher_audit`'s split-check tracer so that every driver reports it;
    - `share_swing_consumer` (design D9 formula);
    - `concurrent_cross_agent_share`;
    - `orders_per_turn_{consumer,producer}`;
    - `agent_history_dispersion` and `routed_share_dispersion` (design D3 definitions, from outcome kinds only);
    - per-decision score-fidelity accumulators for TS / TR: Spearman correlation against log `makeable_qty_now`, top-1 agreement, KL(softmax ‖ makeable share).

  After 1.3. Verify: in `/srv/research/tokenomics/tokenomics/tests/test_routing_diagnostics.py`:
  - TS and MWWk conserve quantity and tokens, route only to eligible producers, never split, and replay identically;
  - with scores = log makeable, TS and MWWk give identical sub-orders for the same keys (paired);
  - a two-producer (log 3, log 1) fixture gives a 0.75 ± 0.03 draw share over 4,000 keys;
  - **ADR-0058 permutation test**: permuting or altering the actions of the ordering agents that act before agent a in a cycle leaves TS's and MWWk's inputs (rows, scores, makeable units) and sub-orders for a byte-identical;
  - hidden shock state does not change actions (sentinel);
  - `share_swing` is 0 for constant shares and 1 for a full switch;
  - `lead_concurrent_top_share` reproduces EXP-0056's value on its first-episode TR rows of 12-0344;
  - `concurrent_cross_agent_share` is 1 for two consumer streams of one product;
  - the dispersion measures are 0 for identical agents and positive on a two-agent fixture with different outcomes;
  - "Tracing changes no statistic" still holds;
  - "Diagnostic router added" holds (frozen hashes, baseline keys, `signal_check` rows).

  Goldens and the full suite are green.
- [x] 2.2 [code][local-exclusive] Build `/srv/research/tokenomics/tokenomics/tokenomics/eval/decoder_check.py` (design D3) on `routing_audit`'s job path, tracer on, `entity_planner` for every arm.
  - Sample: Atlas train split, ρ > 0.28, not `collapse` (23 at 12, 14 at 42), at the confirm's episode seeds.
  - Arms: TR, TS (R = 2), MWWk (R = 2), MWW (R = 2), TW, RR.
  - Floors are per comparison: max(atlas `noise_floor` at the run's counts, 3·SE of the run's own paired per-world differences). The own SE is used only with ≥ 2 paired observations, else the atlas floor (the n = 1 rule).
  - Readings, each with its D3 precedence:
    - primary: `no_gap` → `decoder` → `decoder_partial` → `not_decoder`;
    - fidelity: `faithful` → `mixed` (both or neither of the next two) → `fidelity_loss` → `beyond_target`;
    - dispersion: `history_differs` / `history_flat`.

    The licence comes from the primary reading alone. The D3 interpretation table is written into `reading.json`.
  - Outputs:
    - `decoder_check.csv` (per world: G_w, per-comparison floors, rec_w, TS − MWWk, TS − TW, cell, ρ quartile);
    - `fidelity.csv`, `dispersion.csv`;
    - `reading.json` (12 producers only; the 42-producer table marked reported);
    - `reproduction.csv` (TR / TW / RR vs EXP-0056 on the three worlds; TR / RR / MWW vs EXP-0052);
    - `routing_metrics.csv`, `contract_failures.csv`;
    - a `RESULT.md` skeleton that quotes the task 1.2 prediction.
  - Exit 2 on any mismatch, digest difference or contract failure.
  - A real run needs the registered hashes, a 40-hex commit and no filters. `--dry-run` allows `--worlds --max-worlds --arms --horizon --shock-profile shocks-i-smoke`.

  After 2.1. Verify: new `/srv/research/tokenomics/tokenomics/tests/test_decoder_check.py` covers:
  - sample selection (23 + 14, the three EXP-0056 worlds present, no validation or held-out world);
  - the primary arithmetic on hand-built rows: each branch, the 2/3 and 1/3 edges, a gap exactly at the floor counting as none, "No gap takes precedence" (rows where `decoder`'s rule also holds), and 42-producer rows never changing a reading;
  - the fidelity arithmetic, including "Fidelity readings both hold" (8 / 8 / 7 of 23 → `mixed`);
  - "Agent histories do not differ";
  - the n = 1 rule (a single-episode world with two deterministic arms uses the atlas floor) and the own-spread floor overriding a smaller atlas floor;
  - real-run refusal with filters;
  - a 1-world horizon-12 `shocks-i-smoke` `--dry-run` with `--workers 1`.

  The full suite is green.
- [ ] 2.3 [code][local] Timing probe for EXP-0057: one horizon-12 episode each of TR, TS, MWWk and RR on one loaded 12-producer and one loaded 42-producer f14 sample world (single seed, < 2 min each). Extrapolate with `/srv/research/tokenomics/openspec/changes/archive/2026-10-30-add-m3-development-audit/sizing.md` §EXP-0051 (pod ≈ 0.646 × local, plus ≈ 10 min setup) into `/srv/research/tokenomics/openspec/changes/add-m4-orderer-redesign/sizing.md` §EXP-0057. Pre-registered reduction if the estimate is > 4 vCPU-h: MWW (the unkeyed reproduction arm) at R = 1. After 2.2. Verify: sizing.md lists the measured seconds, the billed estimate, wall time on cpu16 and a stop of ≤ 8.

## 3. Step 1 — decoder check run (experiment)

- [ ] 3.1 [experiment][remote:cpu16] Stage the Checkpoints, the Atlas and EXP-0052 / EXP-0056 `episodes.csv` under `/srv/research/tokenomics/tokenomics/_pod_ckpts/`. Run `python -m tokenomics.eval.decoder_check --atlas /workspace/tokenomics/_pod_ckpts/atlas_v2.json --orderer-ckpt /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt --reference /workspace/tokenomics/_pod_ckpts/exp0052_episodes.csv --reference /workspace/tokenomics/_pod_ckpts/exp0056_episodes.csv --workers 16 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0057` into `/srv/research/tokenomics/experiments/EXP-0057-decoder-check/`. After 2.3 and the user's approval of tranche A. Verify: `contract_failures.csv` is header-only; all planned rows are present once with equal digests; `reproduction.csv` is all MATCH; `reading.json` recomputes from `decoder_check.csv`, `fidelity.csv` and `dispersion.csv`; the staged SHA-256 match the pins; the records cite the commit.
  - Resources: cpu16 fresh pod (--create-pod), ≈ 4.0 vCPU-h billed (1.3 compute + 2.7 setup), ≈ 15 min wall (design D10; re-sized by task 2.3), hard stop 8 vCPU-h, launch-fit 0 + 8 ≤ 95; runs commit after 2.3; stage Checkpoints + Atlas + EXP-0052 / EXP-0056 episodes under tokenomics/_pod_ckpts/; pull /workspace/tokenomics/runs/exp0057 → experiments/EXP-0057-decoder-check/; --verify-source-commit yes.

  Records:
  - `RESULT.md`, plain English first: is the fixed decoder the bottleneck, and do the scores carry their target? It covers each arm, the gaps, recovered shares and per-comparison floors at 12, the 42-producer table, the primary / fidelity / dispersion readings and their interpretation, the task 1.2 prediction against the outcome, the fidelity metrics, and the concentration / herding / composition columns;
  - the README row;
  - new HISTORY §14 "M4: the ordering ladder";
  - a DIAGNOSTICS entry 19 note;
  - a new DIAGNOSTICS entry on `not_decoder` / `no_gap`.

  Stop and report unless the primary reading is `decoder` or `decoder_partial`. On `history_flat`, the supervisor asks the user before group 4.

## 4. Step 2 — choice decode, agent-local history, per-order policy (code). Gated: EXP-0057 reading `decoder` or `decoder_partial`

- [ ] 4.1 [code][local-exclusive] Choice decode and validity layer (design D4).
  - Env option `ordering_decode` (`"logits-v1"` default, `"choice-v1"`), hashed into the env config fingerprint and every driver's pins.
  - Under `choice-v1` the action is per-slot shares in [0, 1]. `validate_choice` in `/srv/research/tokenomics/tokenomics/tokenomics/env/allocator.py` checks support ⊆ eligible, ≥ 0 and |Σ − 1| ≤ 1e-6, gets quantities from `_allocate_int`, apportions tokens ∝ quantity and raises on violation, which `/srv/research/tokenomics/tokenomics/tokenomics/eval/contract.py` records as a `job` row.
  - Overflow orders stay legacy.
  - Write `/srv/research/tokenomics/docs/adr/0059-per-order-choice-decode.md`. It records that M4's learned policies emit one-hot (whole) shares only, and that splitting is out of scope for step 2 (design D5). No `domain/` change.

  After 3.1. Verify:
  - new `/srv/research/tokenomics/tokenomics/tests/test_per_order_decode.py` covers "Whole choice", "Ineligible producer chosen", "Conservation under bursts" (a property test over valid share vectors at q up to 10× the world's max consumer quantity) and the fingerprint changing with `ordering_decode`;
  - `pettingzoo` `api_test` passes on both decodes;
  - "Legacy checkpoint replay" holds (goldens unedited);
  - the `domain/` purity test and the full suite are green.
- [ ] 4.2 [code][local-exclusive] Agent-local history (design D5; ADR-0058) in a new `/srv/research/tokenomics/tokenomics/tokenomics/env/agent_history.py`.
  - The history is built **only from bid-outcome kinds** of the agent's own sub-orders: accepted, completed, failed (the domain's dropped and TTL failures merged) and rejected (all kinds merged). No `reason` field is read.
  - Features per (ordering agent, eligible producer of its stream product): EMAs of routed, accepted, completed, failed and rejected units; completion latency; completion rate with a 1-unit prior; log recency; a never-routed flag (9 features). α is the domain telemetry's.
  - Folded only after `complete_cycle`, reset at reset.
  - If a needed per-sub-order outcome kind is missing from the domain's per-cycle result, add it as additive pure data in `domain/`, with no RNG, IO or logging.

  After 4.1. Verify: new `/srv/research/tokenomics/tokenomics/tests/test_agent_history.py` covers:
  - "Different histories, different inputs";
  - **"Reasons scrambled"**, on a `shocks-i-smoke` episode (horizon 12, a 12-producer Atlas world, TS as the ordering policy):
    - the test first asserts that each of `outage`, `no_fit`, `displaced`, `dropped` and `ttl` occurs at least once among the traced agents' own sub-orders (if a world lacks one, the fixture picks another seeded world, and the chosen world is pinned in the test);
    - it then substitutes values (a seeded permutation and random strings) in both `BidOutcome.reason` and the `ProductionResult` rejection / failure reason lists, never removing an element, so the domain's `zip(..., strict=True)` holds;
    - every agent's history must be byte-identical to the unscrambled episode's;
  - **static no-reference test**: the source of `/srv/research/tokenomics/tokenomics/tokenomics/env/agent_history.py` contains no `reason`, `REJECT_` or `FAIL_` identifier or string (an AST and text check);
  - "Outage not revealed": an outage rejection vs a no-fit rejection, and a TTL failure during an outage vs a dropped reservation, give identical history on fixtures;
  - "Earlier ordering agents' same-cycle routing is invisible": permuting or altering the actions of the ordering agents before agent a leaves a's history and rows byte-identical;
  - "Last cycle's load only": `requested_load` is unchanged across the ordering phase;
  - hidden shock state does not change history (sentinel);
  - the history resets at reset.

  The `domain/` purity test, goldens and the full suite are green.
- [ ] 4.3 [code][local-exclusive] Per-order residual policy `/srv/research/tokenomics/tokenomics/tokenomics/train/per_order_scorer.py` (design D5, whole orders only).
  - Frozen EXP-0012 base, one forward per stream per cycle.
  - φ = shared row (31) ⊕ agent-local history (9) ⊕ own-turn assigned (2); ψ = product ⊕ order features; a fixed median / IQR normaliser, hashed.
  - Bounded Δ (B = 2.0) and τ = exp(θ_τ). No split action: every choice is a one-hot share.
  - Arms A0 (θ = 0) / A1 (43 parameters) / A2 (R = 4: 263, or R = 2).
  - Per-parameter σ scales stored with the arm. Keyed draws exclude the member.
  - `choice-v1` share actions. A composite checkpoint (θ, base SHA-256, normaliser hash, arm, decoder / architecture versions).
  - A shared-only ablation switch (history replaced by the mean over the product's agents).

  After 4.2. Verify: new `/srv/research/tokenomics/tokenomics/tests/test_per_order_scorer.py` covers:
  - "Zero residual equals the decoder check": A0 under `choice-v1` gives the same sub-orders as TS under `logits-v1` on a horizon-12 `shocks-i-smoke` episode of a 12-producer Atlas world;
  - **ADR-0058 permutation test**: permuting or altering the actions of the ordering agents before agent a leaves a's φ, ψ, b_i, own-turn features and resulting sub-orders byte-identical;
  - "No split emitted": every routed order has one sub-order, across random θ;
  - "Own earlier orders visible" and "Fixed decision order";
  - "Empty histories still differ" and "Relabelled agents";
  - |Δ| ≤ B; A1 and A2 at θ = 0 equal A0; A2 with v = 0 equals A1;
  - "Composite checkpoint in a legacy env refused" and "Frozen base loads";
  - the shared-only ablation gives identical history inputs across a product's agents;
  - the shock sentinel.

  The full suite is green.

## 5. Step 2 — ES driver, pre-registration and sizing (code)

- [ ] 5.1 [code][local-exclusive] Training worlds and the bar precompute in `/srv/research/tokenomics/tokenomics/tokenomics/train/order_es_worlds.py` (design D6).
  - Worlds: the 23 loaded non-collapse 12-producer Atlas train worlds plus 41 fresh `eval/world_space` draws (disjoint seed stream; the ρ > 0.28 / fill < 0.95 / no-dead-bar filters; a fixed redraw order), each pinned.
  - `--mode bar` plays RR, MWW R = 2, TR, the roster v2 deterministic members and A0 on every training world-episode with the chosen producer, records per-job timings (for longest-first) and applies the "A0 infeasible" branch (replace worlds where A0 dies; `no_feasible_start` at > 1/4 replaced or < 50 feasible).
  - The 16-world-episode monitor set and the minibatch schedule come from a recorded seed stream.

  After 4.3. Verify: new `/srv/research/tokenomics/tokenomics/tests/test_order_es_worlds.py` covers determinism, 64 worlds, disjoint seeds, filters, the "No feasible start" branch on a forced-death fixture, and the bar table schema with a 2-world horizon-12 `--dry-run`. The full suite is green.
- [ ] 5.2 [code][local-exclusive] ES loop in `/srv/research/tokenomics/tokenomics/tokenomics/train/order_es.py` (design D6 / D7).
  - `--mode train --arm --opt-seed --generations`: antithetic pairs, centered ranks, Adam with per-parameter σ, CRN minibatches, fitness U / B − h·H / D, relative disqualification, and the monitor-set eval every 10 generations.
  - `--resume-from <ckpt>`, accepted only on matching pins, so the main run continues the pilot.
  - `--mode probe`: the gradient-signal probe, with two disjoint minibatches per arm, cosine, the paired / unpaired variance ratio and the D7 `signal` / `weak` / `no_signal` reading.
  - Longest-first submission. Process pool with one job per member × world-episode, BLAS and torch at 1 thread.
  - A real run refuses a pre-registration mismatch, a training size other than 12, or reads above 42.

  After 5.1. Verify: new `/srv/research/tokenomics/tokenomics/tests/test_order_es.py` covers:
  - members of a pair see identical world-episodes and keys;
  - on a quadratic toy the update moves θ toward the optimum;
  - "Control also starves a consumer";
  - the probe cosine is ≈ 1 on a noiseless toy and ≈ 0 on pure noise;
  - resume equivalence (2 + 1 generations = 3);
  - "Unregistered change refused";
  - a 2-world, 2-generation, 2-pair horizon-12 `--dry-run` with `--workers 1` in seconds.

  The full suite is green.
- [ ] 5.3 [code][local-exclusive] Validation readings in `/srv/research/tokenomics/tokenomics/tokenomics/train/order_es_validate.py` (design D7 / D9).
  - `--mode validate --runs <dirs>` plays every staged candidate, A0, RR, MWW and TR, plus the shared-only ablation of each candidate, on all Atlas validation worlds (4 episodes, or 8 by the power rule; stochastic arms R = 2). It adds the EXP-0035-producer column when the producer decision says so.
  - Gains and floors: g_tr = center − A0 on the monitor set; g_val = selected candidate − A0 on the 12-producer loaded validation world mean. Floors are per comparison, with the n = 1 rule.
  - Readings:
    - D6 selection, then the D7 precedence (`unresolved` → `kill` → `success` → `plateau` → `budget_limited` → `other`). `budget_limited` is triggered by **any** seed-0 arm still improving at the seed-0 read, and by the seed-1 run still improving at the seed-1 read. At the seed-1 read, `plateau` clause (i) also needs the seed-1 run's own last-10-generation improvement < F_tr;
    - with `--seed-combine`, the D7 seed combination table (both `success` → `success`; both `plateau` → `plateau`; a mix → `seed_disagreement`; any other seed-1 reading carries through; clauses (i) and (ii) from seed 0).
  - Labels and checks:
    - the concentration check (`lead_concurrent_top_share`, `routed_share_dispersion` for the candidate, A0, MWW, RR and the ablation) and the `not_via_deconcentration` label (under `success`);
    - the D7 label table: which labels go with `success` and which with `plateau` (`history_inert` and `herding` with both);
    - the `history_inert` label;
    - `share_swing` with the `herding` flag.
  - Tables per ρ quartile, 42-producer, light and collapse.
  - Writes `validation.csv`, `concentration.csv`, `herding.csv` and `reading.json`.
  - Freeze `/srv/research/tokenomics/tokenomics/tokenomics/train/registry/es_prereg_m4_v1.json` (arms, seeds, n, σ scales, lr, weight decay, generations, minibatch, h, B, floors and the n = 1 rule, the power rule, readings, seed table, labels, seed streams, producer, α).

  After 5.2. Verify: new `/srv/research/tokenomics/tokenomics/tests/test_order_es_validate.py` covers every reading on hand-built numbers, including:
  - "Cross-world failure" (success and kill both true → `kill`);
  - "Any arm still improving" (the non-selected arm improving → `budget_limited`, not `other`);
  - "Seed 1 still improving": seed-1 inputs that meet every `plateau` clause except the seed-1 run's own last-10-generation improvement ≥ F_tr → not `plateau`, but `budget_limited`;
  - seed 1 reading `budget_limited` from its own still-improving run while every seed-0 arm has stopped improving;
  - `history_inert` produced under `plateau` as well as under `success`; `not_via_deconcentration` only under `success`;
  - every row of the seed combination table ("Seed disagreement", "Seed-1 failure carries through");
  - g_val computed against A0;
  - the per-comparison floors with the n = 1 rule;
  - "Gain without de-concentration", "History adds nothing measurable" and "Agents switch together";
  - validation rows never containing held-out worlds.

  The full suite is green.
- [ ] 5.4 [code][local] Timing probe, detectable effect and sizing for EXP-0058–0060.
  - Time horizon-12 episodes of A0 and A2 against TR on 2 loaded 12-producer training worlds (single seed, < 2 min each). Extrapolate to horizon 100 for s per generation, the bar precompute, the monitor and validation reads.
  - Compute the minimum detectable effect (design D7) from **paired per-episode differences** in the EXP-0052 confirm rows on the 9 loaded validation worlds (MWW − RR, TR − RR, each on the same world-episodes). It is 3 × the SE of the world mean at 4 and at 8 episodes (stochastic arms R = 2), as a share of the **mean** validation bar. The single-arm `se` column and the atlas formula are reported, not used. Choose 4 or 8 episodes by the D7 rule.
  - Write `/srv/research/tokenomics/openspec/changes/add-m4-orderer-redesign/sizing.md` §EXP-0058–0060 with estimates, stops and the launch-fit table (the 8-episode variant if chosen). Record the pre-registration file's SHA-256 there.

  After 5.3. Verify: sizing.md lists the measured seconds, the per-order / TR ratio, the paired MDE values with the mean bar and the chosen episode count, and stops ≤ 40 (≤ 4 for the probe). If the MDE is > 3% at 8 episodes, or the 8-episode launch-fit exceeds 95, the task ends by reporting to the supervisor.

## 6. Step 2 — gradient-signal probe and pilot (experiment). Gated: the user's producer decision (open decision 5) and tranche A approval recorded

- [ ] 6.1 [experiment][remote:cpu8] Stage the Checkpoints and the Atlas. Run `python -m tokenomics.train.order_es --mode probe --arms A1,A2 --workers 8 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0058` on its own cpu8 pod (design D10: placement justified) into `/srv/research/tokenomics/experiments/EXP-0058-es-gradient-probe/`. After 5.4. Verify: `contract_failures.csv` is header-only; the cosine recomputes from the per-member fitness rows; pins match.
  - Resources: cpu8 fresh pod (--create-pod), ≈ 2.0 vCPU-h billed (0.7 compute + 1.3 setup), ≈ 15 min wall (sized by task 5.4), hard stop 4 vCPU-h, launch-fit 4.0 + 4 ≤ 95; runs commit after 5.4; stage Checkpoints + Atlas under tokenomics/_pod_ckpts/; pull /workspace/tokenomics/runs/exp0058 → experiments/EXP-0058-es-gradient-probe/; --verify-source-commit yes.

  Records:
  - `RESULT.md`, plain English first: does one ES generation carry a usable direction? It gives the cosine per arm and the paired / unpaired variance ratio, the reading `signal` / `weak` / `no_signal`, and A2's rank;
  - the README row and the HISTORY §14 paragraph.

  On `weak`, update sizing.md §EXP-0059/0060 (minibatch 24) and re-check launch-fit. On `no_signal`, add a DIAGNOSTICS entry and stop.
- [ ] 6.2 [experiment][remote:cpu32] Run `order_es_worlds --mode bar` and then `order_es --mode train --arm A1 --opt-seed 0 --generations 20 --workers 32 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0059` into `/srv/research/tokenomics/experiments/EXP-0059-es-pilot/`. Records:
  - Resources: cpu32 fresh pod (--create-pod), ≈ 13 vCPU-h billed (7.4 compute + 5.3 setup), ≈ 25 min wall (sized by task 5.4 / 6.1), hard stop 26 vCPU-h, launch-fit 6.0 + 26 ≤ 95; runs commit after 5.4; stage Checkpoints + Atlas; pull /workspace/tokenomics/runs/exp0059 → experiments/EXP-0059-es-pilot/; --verify-source-commit yes.
  - `RESULT.md`, plain English first: cost per generation and worker utilisation, the monitor-set curve, g_tr against F_tr, and the reading `feasible` / `unresolved` (or `no_feasible_start`);
  - the README row and the HISTORY §14 paragraph;
  - an update to sizing.md §EXP-0060 from the measured rate (cpu32 vs cpu16).

  On `unresolved` / `no_feasible_start`, add a DIAGNOSTICS entry and stop. After 6.1 (reading `signal` or `weak`). Verify: `contract_failures.csv` is header-only; the bar table is complete; each generation's members share digests; g_tr recomputes; the generation-20 checkpoint and pins are recorded for resume.

## 7. Step 2 — main ES, seed 0 first (experiment). Gated: EXP-0059 reading `feasible`

- [ ] 7.1 [experiment][remote:cpu32] Stage the Checkpoints, the Atlas, EXP-0059's bar table and its generation-20 checkpoint. Run `order_es --mode train --arm A1 --opt-seed 0 --resume-from /workspace/tokenomics/_pod_inputs/a1-s0-g20 --generations 40 --workers 32 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0060-a1-s0` into `/srv/research/tokenomics/experiments/EXP-0060-es-main/artifacts/a1-s0/`. After 6.2. Verify: header-only `contract_failures.csv`; generations 21–40 are present; the resume pins equal EXP-0059's; the final θ hash is recorded.
  - Resources: cpu32 fresh pod (--create-pod), ≈ 12 vCPU-h billed (6.7 + 5.3), ≈ 25 min wall (sized by task 6.2's measured s / generation), hard stop 24 vCPU-h, launch-fit 19.0 + 24 ≤ 95; runs commit after 5.4; stage Checkpoints + Atlas + EXP-0059 bar table + checkpoint under tokenomics/_pod_inputs/; pull /workspace/tokenomics/runs/exp0060-a1-s0 → experiments/EXP-0060-es-main/artifacts/a1-s0/; --verify-source-commit yes.
- [ ] 7.2 [experiment][remote:cpu32] Run `order_es --mode train --arm A2 --opt-seed 0 --generations 40` (with R as set by EXP-0058) into `artifacts/a2-s0/`, with staging as 7.1 but without the checkpoint. After 6.2. Verify: as 7.1, for generations 1–40.
  - Resources: cpu32 fresh pod (--create-pod), ≈ 19 vCPU-h billed (13.3 + 5.3), ≈ 40 min wall (sized by task 6.2), hard stop 38 vCPU-h, launch-fit 31.0 + 38 ≤ 95 (81.0 if launched together with 7.1); runs commit after 5.4; stage as 7.1 minus the checkpoint; pull /workspace/tokenomics/runs/exp0060-a2-s0 → experiments/EXP-0060-es-main/artifacts/a2-s0/; --verify-source-commit yes.
- [ ] 7.3 [experiment][remote:cpu16] Run `order_es_validate --mode validate --runs a1-s0,a2-s0 --workers 16 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0060-val0` into `/srv/research/tokenomics/experiments/EXP-0060-es-main/val-seed0/`. This is the seed-0 reading: selection, the D7 precedence and the herding flag. Records:
  - Resources: cpu16 fresh pod (--create-pod), ≈ 3.9 vCPU-h billed (1.2 + 2.7), ≈ 15 min wall (sized by task 5.4), hard stop 8 vCPU-h (11 under 8-episode reads), launch-fit 50.0 + 8 ≤ 95; runs commit after 5.4; stage Checkpoints + Atlas + the two θ files; pull /workspace/tokenomics/runs/exp0060-val0 → experiments/EXP-0060-es-main/val-seed0/; --verify-source-commit yes.
  - `RESULT.md` draft, plain English first: what was trained; training and validation gains against A0 / RR / MWW / TR with floors; the 42-producer, light and ρ-quartile tables; herding; the reading and what it licenses;
  - the README row and the HISTORY §14 paragraph.

  On `unresolved` / `kill` / `other`, add a DIAGNOSTICS entry and stop. On `budget_limited`, ask the user. After 7.1 and 7.2. Verify: header-only `contract_failures.csv`; equal digests per world-episode; `reading.json` recomputes from `validation.csv` and the runs' logs; no held-out world in any row.
- [ ] 7.4 [experiment][remote:cpu32] Only if 7.3 reads `success` or `plateau` (the seed-1 rule): run `order_es --mode train --arm <selected> --opt-seed 1 --generations 40` into `artifacts/<selected>-s1/`, with staging as 7.2. After 7.3. Verify: as 7.1.
  - Resources: cpu32 fresh pod (--create-pod), ≈ 19 vCPU-h billed, ≈ 40 min wall (sized by task 6.2), hard stop 38 vCPU-h, launch-fit 53.9 + 38 = 91.9 ≤ 95 (93.1 under 8-episode reads); runs commit after 5.4; stage as 7.2; pull /workspace/tokenomics/runs/exp0060-<selected>-s1 → experiments/EXP-0060-es-main/artifacts/<selected>-s1/; --verify-source-commit yes.
- [ ] 7.5 [experiment][remote:cpu16] Run `order_es_validate --mode validate --seed-combine --runs <selected>-s0,<selected>-s1 --reuse-references val-seed0` into `/srv/research/tokenomics/experiments/EXP-0060-es-main/`. This is the final reading (D7 table on both seeds, `seed_disagreement`). Finish the EXP-0060 `RESULT.md`, README row, HISTORY §14 paragraph and DIAGNOSTICS entries (as 7.3), with the D7 label table (which labels go with the reading). On `success` with a `herding` flag or a `not_via_deconcentration` label, stop and report to the supervisor, who inserts a diagnosis task (root-cause rule; DIAGNOSTICS entry); group 8 starts only once each such label is diagnosed. On `plateau`, record `history_inert` and `herding` and ask the user to release tranche B1. After 7.4. Verify: as 7.3; the reused reference rows' digests equal val-seed0's.
  - Resources: cpu16 fresh pod (--create-pod), ≈ 3.0 vCPU-h billed (0.3 + 2.7), ≈ 12 min wall (sized by task 5.4), hard stop 6 vCPU-h (7 under 8-episode reads), launch-fit 72.9 + 6 ≤ 95; runs commit after 5.4; stage Checkpoints + Atlas + both θ + val-seed0 rows; pull /workspace/tokenomics/runs/exp0060-val1 → experiments/EXP-0060-es-main/; --verify-source-commit yes.

## 8. Step 2 — held-out and cost drivers (code). Gated: EXP-0060 final reading `success`, and no undiagnosed `herding` or `not_via_deconcentration`

- [ ] 8.1 [code][local-exclusive] Held-out confirmation `/srv/research/tokenomics/tokenomics/tokenomics/eval/m4_read.py` (design D9).
  - Plays the selected θ (SHA-256 pinned) with A0, RR, MWW and TR, all with the recorded producer, plus the EXP-0035-producer cross-check (gating only under open decision 5's EXP-0035 / both answer) and `portable`.
  - All Atlas held-out worlds (12 at 12, 4 at 42) plus held-out-region (4 at 42, reported apart), fresh seeds, the D7 episode count, R = 2.
  - Writes the `confirmed` / `not_confirmed` reading (per-comparison floors, n = 1 rule), the concentration / history / herding tables and the ρ-quartile tables. `RESULT.md` is marked "not an acceptance read".
  - **Ledger** (design D9, the C0 / C1 rule). A real run needs `--ledger <path>`. It refuses unless the ledger holds exactly one entry for (atlas hash, change, split), with `status: pending`, a matching θ hash, and a recorded package source hash (design D9: SHA-256 over sorted paths and contents of `tokenomics/tokenomics/`, `tokenomics/tests/` and `pyproject.toml`, without `__pycache__`) equal to the synced tree's, which the launcher's `--verify-source-commit` guarantees is the clean launch commit. The entry's code commit C0 may differ from the launch commit C1 when only files outside `tokenomics/` changed (the ledger). A `done` or `invalid` entry always refuses. The driver never writes the ledger.
  - Add a small CLI `python -m tokenomics.eval.m4_read --ledger-pending ... | --ledger-close done|invalid ...` that the experiment task runs **locally**. `--ledger-pending` records C0 = HEAD, the package source hash at C0, the θ hash and the date. `--ledger-close` records C1 and the EXP path. The tracked ledger is `/srv/research/tokenomics/experiments/m4_heldout_ledger.json`, outside the package; create it empty here.
  - Extend `/srv/research/tokenomics/tokenomics/tokenomics/eval/deployed_cost.py` to accept the per-order subject (registered θ hash).
  - Real runs refuse filters and > 42 producers.

  After 7.5. Verify: new `/srv/research/tokenomics/tokenomics/tests/test_m4_read.py` covers:
  - "No pending entry", "Already read" and the closing CLI (pending → done / invalid; no transition out of `done` / `invalid`);
  - **"Pending entry committed, then launched"**: in a temporary git repo, write the pending entry at C0, commit it as C1, run the driver's ledger check on C1's package tree with `--source-commit` C1 → accepted; a C1 that also changes a file under `tokenomics/` → refused; the package source hash ignores `__pycache__` and the git-ignored staging dirs;
  - "Not confirmed", "Light corner excluded", "Digest mismatch";
  - a 1-world horizon-12 `--dry-run` that needs no ledger and touches none;
  - `deployed_cost --dry-run --cells P12-E2-k1` with the per-order subject.

  The full suite is green.
- [ ] 8.2 [code][local] Timing probe for EXP-0061 / EXP-0062: horizon-12 episodes of the selected θ on one held-out world at 12 and at 42, and one `deployed_cost --dry-run` cell. Write sizing.md §EXP-0061 / §EXP-0062 with stops and the launch-fit check. After 8.1. Verify: sizing.md lists the measured seconds, the estimates and stops (≤ 11, ≤ 3).

## 9. Step 2 — held-out confirmation and per-order cost (experiment)

- [ ] 9.1 [experiment][remote:cpu16] Held-out confirmation, in three steps (design D9).
  - Resources: cpu16 fresh pod (--create-pod), ≈ 5.4 vCPU-h billed (2.7 + 2.7; 8.1 under 8-episode reads), ≈ 22 min wall (sized by task 8.2), hard stop 11 vCPU-h (17 under 8-episode reads), launch-fit 75.9 + 11 ≤ 95; runs C1 (= the commit after 8.2 plus the ledger-pending commit); stage Checkpoints + Atlas + selected θ under tokenomics/_pod_ckpts/ and the committed experiments/m4_heldout_ledger.json under tokenomics/_pod_inputs/; pull /workspace/tokenomics/runs/exp0061 → experiments/EXP-0061-m4-heldout-confirmation/ (the ledger is not pulled; it is closed locally); --verify-source-commit yes.
  - **Before launch (local):** at the code commit C0 (the commit after 8.2), run `m4_read --ledger-pending --ledger /srv/research/tokenomics/experiments/m4_heldout_ledger.json --atlas-hash 220886a7… --theta-hash <θ>`. It records C0, the package source hash and the θ hash. Commit the ledger alone as C1 and launch at C1 (`{SOURCE_COMMIT}` = C1). Stage the Checkpoints, the Atlas, the selected θ and the committed ledger.
  - **On the pod:** run `python -m tokenomics.eval.m4_read --ledger /workspace/tokenomics/_pod_inputs/m4_heldout_ledger.json --atlas /workspace/tokenomics/_pod_ckpts/atlas_v2.json --theta /workspace/tokenomics/_pod_ckpts/selected_theta.npz --prod-ckpt /workspace/tokenomics/_pod_ckpts/producer_candidate.pt --orderer-ckpt /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt --workers 16 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0061` into `/srv/research/tokenomics/experiments/EXP-0061-m4-heldout-confirmation/`.
  - **After (local):** close the entry with `--ledger-close done` on exit 0 with valid rows. On exit 2, a crash, a lost pod or a missing pull, close it with `--ledger-close invalid`. **Pre-registered:** an `invalid` read counts as used, goes to the user and is never re-run automatically.

  After 8.2. Verify: the pending entry is in git (C1) before the pod is created; C1 differs from C0 only in the ledger; header-only `contract_failures.csv`; all rows are present with equal digests; `reading.json` recomputes; the closed entry records C0, C1 and the θ hash; pins match.

  Records:
  - `RESULT.md`, plain English first: held-out results at 12 and, unretrained, at 42 against each reference with floors; concentration, history and herding; the light and ρ tables; the EXP-0035 cross-check; `confirmed` / `not_confirmed` and what it means for the gate-exit decision;
  - the README row and the HISTORY §14 paragraph.

  On `not_confirmed`, add a DIAGNOSTICS entry and stop. On `confirmed`, the supervisor asks open decision 4.
- [ ] 9.2 [experiment][remote:cpu8] Run `python -m tokenomics.eval.deployed_cost --orderer-subject per-order --theta /workspace/tokenomics/_pod_ckpts/selected_theta.npz --orderer-ckpt /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt --workers 6 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0062` (PoC-B's full grid, horizon 20) into `/srv/research/tokenomics/experiments/EXP-0062-per-order-cost/`. Records: `RESULT.md`, plain English first, with the per-orderer-decision slopes on the fanout and consumer axes against 1.1; the README row; a HISTORY §14 line. After 8.2. Verify: `slopes.csv` recomputes from `timings.csv`; no stage is flagged, or the flag is diagnosed; pins match.
  - Resources: cpu8 fresh pod (--create-pod), ≈ 1.6 vCPU-h billed, ≈ 12 min wall (EXP-0055: 0.23 vCPU-h compute; sized by task 8.2), hard stop 3 vCPU-h, launch-fit 81.3 + 3 ≤ 95; runs commit after 8.2; stage Checkpoints + selected θ; pull /workspace/tokenomics/runs/exp0062 → experiments/EXP-0062-per-order-cost/; --verify-source-commit yes.

## 10. Step 3 gate — env-side relational arm (code). Gated: EXP-0060 final reading `plateau` and tranche B1 released by the user

- [ ] 10.1 [code][local-exclusive] Add arm A2+R to `/srv/research/tokenomics/tokenomics/tokenomics/train/per_order_scorer.py` (design D8), with no archive restore. φ_i is extended by per-producer aggregates computed in `env/` from domain state as of the start of the ordering phase: `reservation` units and cycles waited for the product, inbound units and inventory cover of the recipe's inputs. No pending-bid or same-cycle aggregate (ADR-0058). Add a pre-registration entry `es_prereg_m4_r_v1.json`. After 7.5. Verify: new tests show that the aggregates equal hand counts of reservations, inbound and inventory on fixtures; the ADR-0058 permutation test passes for A2+R inputs; A2+R with zero extra weights equals A2; the shock sentinel; the full suite is green.
- [ ] 10.2 [code][local] Timing probe for EXP-0063: horizon-12 A2+R vs A2 on 2 training worlds. Write sizing.md §EXP-0063 with stops ≤ 40 and the tranche B1 launch-fit table. After 10.1. Verify: sizing.md lists the measured seconds, the feature-cost ratio and the stops.

## 11. Step 3 gate — relational check (experiment)

- [ ] 11.1 [experiment][remote:cpu32] Run `order_es --mode train --arm A2R --opt-seed 0 --generations 40` into `/srv/research/tokenomics/experiments/EXP-0063-relational-check/artifacts/a2r-s0/`, with staging as 7.2. After 10.2. Verify: as 7.1.
  - Resources: cpu32 fresh pod (--create-pod), ≈ 21 vCPU-h billed, ≈ 45 min wall (sized by task 10.2), hard stop 40 vCPU-h, launch-fit B1 0 + 40 ≤ 65; runs commit after 10.2; stage Checkpoints + Atlas + EXP-0059 bar table; pull /workspace/tokenomics/runs/exp0063-a2r-s0 → experiments/EXP-0063-relational-check/artifacts/a2r-s0/; --verify-source-commit yes.
- [ ] 11.2 [experiment][remote:cpu16] Run `order_es_validate --mode validate --runs a2r-s0 --compare <EXP-0060 A2 s0 θ>` into `/srv/research/tokenomics/experiments/EXP-0063-relational-check/val-seed0/`, giving the seed-0 `relational` / `not_relational` reading (design D8). Records as 7.3. On `not_relational`, add a DIAGNOSTICS entry and stop. After 11.1. Verify: as 7.3; the A2 reference rows reproduce EXP-0060's validation rows.
  - Resources: cpu16 fresh pod (--create-pod), ≈ 3.9 vCPU-h billed, ≈ 15 min wall (sized by task 10.2), hard stop 8 vCPU-h, launch-fit B1 21 + 8 ≤ 65; runs commit after 10.2; stage Checkpoints + Atlas + both θ; pull /workspace/tokenomics/runs/exp0063-val0 → experiments/EXP-0063-relational-check/val-seed0/; --verify-source-commit yes.
- [ ] 11.3 [experiment][remote:cpu32] Only if 11.2 reads `relational`: train A2+R seed 1 (40 generations) and, on the same pod, run the final validation read on both seeds (`relational` needs both). Finish the EXP-0063 records and add a DIAGNOSTICS entry. On `relational`, ask the user to release tranche B2 (and to choose between A2+R and a GNN if A2+R also meets `success`). After 11.2. Verify: as 7.5.
  - Resources: cpu32 fresh pod (--create-pod), ≈ 24 vCPU-h billed (21 + 3.0), ≈ 55 min wall (sized by task 10.2), hard stop 40 vCPU-h, launch-fit B1 24.9 + 40 = 64.9 ≤ 65; runs commit after 10.2; stage as 11.2; pull /workspace/tokenomics/runs/exp0063 → experiments/EXP-0063-relational-check/; --verify-source-commit yes.

## 12. Step 3 — GNN per-order router (code). Gated: EXP-0063 reading `relational` and the user's choice of a GNN

- [ ] 12.1 [code][local] Run `/opsx:update add-m4-orderer-redesign` to fix the step-3 optimiser: ES over a reduced GNN (hidden ≤ 16), or a score-function estimator on the sampled head with paired-episode baselines; no critic; no per-decision labels unless they pass "Imitation targets pass label stability". The update also fixes its pre-registration, cost model, tranche B2 budget (for the user's approval) and the EXP-0064 / EXP-0065 tasks. EXP-0063 is the diagnosed reason. It **cites step 2's `history_inert` label** (EXP-0060) and states whether agent-local history enters the graph as candidate-edge attributes, and why. After 11.3. Verify: `openspec validate add-m4-orderer-redesign --strict` passes; the design names EXP-0063 as the reason and cites the `history_inert` label; tranche B2 is listed for approval.
- [ ] 12.2 [code][local-exclusive] Restore from `archive-1` `7aeef1f`: `domain/graph_view.py`, the `graph_primitives_v1` schema / encoder parts of `eval/graph_value_router_dump.py`, and the MPNN parts of `train/graph_router_probe.py`, with their tests.
  - **Drop the `pending_bid` relations and their reverses** (ADR-0058) and keep `reservation` relations. The schema version becomes `graph_primitives_v1-adr0058` with a new hash.
  - Agent-local history enters as candidate-edge attributes only as 12.1 decided.
  - The head feeds D5's whole-order choice rule (one producer per order, no split) and the D4 validity layer.
  - Implement the 12.1 trainer.

  After 12.1. Verify:
  - the restored schema, identity-isolation, relabel-invariance and mixed-size tests pass;
  - a schema test asserts that no `pending_bid` relation exists;
  - the ADR-0058 permutation test passes on graph inputs and sub-orders;
  - "No split emitted" holds for the GNN policy;
  - the `domain/` purity test passes;
  - a per-order GNN episode at horizon 12 under `choice-v1` has zero contract failures;
  - goldens and the full suite are green.

## 13. Step 3 — GNN training and read (experiment)

- [ ] 13.1 [experiment][remote:cpu32] Run the 12.1-specified GNN training into `/srv/research/tokenomics/experiments/EXP-0064-gnn-per-order/`, then the validation read against the step-2 best, A0, RR, MWW and TR. The reading is D7's success rule, plus beating the step-2 best by more than F_val. Records as 7.5. After 12.2. Verify: as 7.5.
  - Resources: cpu32 fresh pod (--create-pod), ≈ 20 vCPU-h billed placeholder (re-sized by 12.1's probe), hard stop ≤ 40 vCPU-h, launch-fit within tranche B2; runs commit after 12.2; stage Checkpoints + Atlas + EXP-0059 bar table; pull /workspace/tokenomics/runs/exp0064 → experiments/EXP-0064-gnn-per-order/; --verify-source-commit yes.
- [ ] 13.2 [experiment][remote:cpu16] On `success`, run `m4_read` for the GNN candidate into `/srv/research/tokenomics/experiments/EXP-0065-gnn-heldout-confirmation/`, with the three ledger steps and records as 9.1. The ledger key for this change's held-out split is already closed after EXP-0061. A GNN confirmation therefore needs the user's decision and a new ledger key (for example `held-out/gnn`), written as a pending entry before launch. After 13.1. Verify: as 9.1.
  - Resources: cpu16 fresh pod (--create-pod), ≈ 5.4 vCPU-h billed (as EXP-0061; re-sized in 12.1), hard stop 11 vCPU-h, launch-fit within tranche B2; runs C1 (= the commit after 12.2 plus the ledger-pending commit, as 9.1); stage Checkpoints + Atlas + GNN checkpoint + the committed ledger (as 9.1); pull /workspace/tokenomics/runs/exp0065 → experiments/EXP-0065-gnn-heldout-confirmation/ (the ledger is closed locally); --verify-source-commit yes.

## 14. Ladder record (code)

- [ ] 14.1 [code][local] When the ladder stops (any terminal reading, or after group 9 / 13), write the HISTORY §14 summary and update §0. Put in plain English the readings per step, spend against each tranche with the launch-fit log, and the inputs to the open user decisions: roster v3; atlas worlds in a gate registry; the 42-producer teacher read; how the sealed 12-producer exit is read (producer, candidate replicates). After the last executed task. Verify: every EXP-0057+ record has a README row and a §14 paragraph; spend totals equal the sum of the records' billed vCPU-h.

## Workflow follow-up

- Each task goes through the cross-harness review (`supervise` / `review-orchestrator`) before it is ticked.
- AskUserQuestion points:
  - tranche A approval before task 3.1, stating the success-path margin: 3.1 vCPU-h under 95 (0.6 under 8-episode reads), with the options 95 and 100 (margin 8.1 / 5.6);
  - open decision 5 (the training producer) before group 6;
  - `history_flat` after 3.1 (before group 4); the pre-pod triggers after 1.3 (before group 3);
  - `budget_limited` after 7.3;
  - tranche B1 release after a `plateau` final reading;
  - tranche B2 / A2+R-vs-GNN after 11.3;
  - open decision 4 (gate exit) after a `confirmed` held-out read.
- Sync the `learned-orderer` spec and archive the change once 14.1 is written.
- Pushing `new-model` and champion promotion stay human gates.
