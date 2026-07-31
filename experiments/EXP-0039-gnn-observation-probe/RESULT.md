# EXP-0039 — GNN Observation Probe (graph_primitives_v1)

**Status:** local build + TDD complete; real s29/s77/s45 dump + probe staged for a Runpod CPU
pod (Phases 7–8). Gate not yet read — the numbers below are filled after the pod run.

## Question

Can a graph model over raw deployable economy state recover the EXP-0037 marginal-value
teacher's within-decision producer preference that the EXP-0038B hand-metric ladder missed?
Supervised representability probe only. A pass **licenses** a later deployed-GNN orderer
experiment; it does not itself prove deployed improvement.

## What was built (all TDD, green locally)

| Layer | Module | Tests |
|---|---|---|
| Domain graph view (`graph_primitives_v1`) | `tokenomics/domain/graph_view.py` | `tests/test_graph_view.py` (7) |
| Schema + encoder + trajectory dump + NPZ loader | `tokenomics/eval/graph_value_router_dump.py` | `tests/test_graph_dump.py` (13) |
| Pure-Torch relational GNN + leave-one-seed-out gate | `tokenomics/train/graph_router_probe.py` | `tests/test_graph_probe.py` (19) |
| End-to-end smoke (dump → probe → gate) | — | `tests/test_graph_probe_smoke.py` (1) |
| Launcher clean-commit binding (Finding-9.1) | `run_pod_training.py` | `tests/test_run_pod_detached.py` (+1) |

Graph tests total **40** (view 7 + dump 13 + probe 19 + smoke 1), plus the launcher binding test.

Design decisions (accepted before implementation, per ARCH.md Phase 0):
- **Commitment visibility = visible.** Other orderers' current-cycle `pending_bid`s and live
  `reservation`s are part of the deployable observation contract, exposed through the
  domain-owned `EconomyGraphView` (they exist before the current action → not future leakage).
  Recorded in `graph_schema.json` `commitment_visibility`.
- **Model = minimal pure-Torch relational message passing** (no PyG dependency): 4 layers,
  hidden 64, schema-role embeddings only (node-type + relation-type — no per-identity
  embeddings), relation-specific edge encoders + message transforms, sum aggregation, residual
  + layer-norm, one shared candidate head over `[decision, current-order, target-product,
  candidate-producer, candidate-relation]`.

Contract guarantees locked by tests: schema-ordered feature columns; `log1p`/`signed_log1p`/binary
transforms; candidate↔eligible alignment; structural-zero inventory/telemetry/capacity relations; no
raw domain ID / teacher label / actor score in any model tensor (identity isolation); pure read
(no env mutation); exact NPZ round-trip with offset isolation; producer-relabel invariance;
mixed-size batching through one model; deterministic training for a fixed seed.

## Pre-flight corrections (review-driven — before any pod spend)

A pre-run review caught three input defects that each bias a supervised probe toward a **false FAIL**
(see DIAGNOSTICS #13). All fixed test-first; the schema hash changed as a result (recompute with
`python -c "from tokenomics.eval import graph_value_router_dump as gd; print(gd.schema_hash())"`).

1. **Finite blocker.** `free = current − reserved` is signable (a prior-cycle reservation can outlive
   degraded capacity) but was declared `log1p` → `log1p(x≤−1) = NaN/−inf`. Now `free` and `headroom`
   use **`signed_log1p`** = `sign(x)·log1p(|x|)`; `run_graph_dump` asserts every tensor finite before
   serializing (`_assert_finite`), and `_load_dump` re-asserts finiteness on load.
2. **Flat-decision domination.** The EXP-0038 dump is ~69% flat decisions (all candidate labels equal
   → no within-decision preference). The gate now trains **primarily on non-flat decisions** (flats
   excluded from each training fold, count reported as `n_train_flat_excluded`); held-out scoring is
   unchanged (real-choice filter already lives in `_routing_metrics`). HANDOVER.md:204–207.
3. **Fold-local standardization** (ARCH.md:290–291, previously unimplemented). `FoldNormalizer` fits
   mean/std on each fold's training graphs' continuous (`log1p`/`signed_log1p`) columns and applies to
   train + held-out; binary/one-hot columns pass through untouched.

A **second review round** caught five more issues, all fixed test-first:

4. **Verdict could false-PASS an unstable result.** Stability now applies a dispersion rule to
   **every** gate metric: the four numeric conditions must hold for the seed-mean **and**
   independently for each of the ≥3 seeds (any seed failing ⇒ no-pass). Regression caught: per-seed
   r² −100/−100/201 averages positive but is unstable ⇒ now no-pass. HANDOVER.md:241.
5. **Dump validation was shallow.** `_load_dump` now **requires** `graph_schema.json` and validates
   one graph per decision with exact per-decision candidate coverage / `candidate_index` 0..k-1 /
   count vs `n_eligible` / producer-id alignment (via `topologies.jsonl`) / task-seed consistency, plus
   optional expected-count stop-checks — not just the global row total. ARCH.md:343.
6. **Persisted normalization ≠ trained normalization.** Dropped the misleading all-graph refit;
   `spec.json` now stores the **actual per-fold** train stats used, alongside each fold's loss history
   and checkpoint provenance (`folds`).
7. **Missing audit artifacts.** The probe now writes `predictions.csv` (task, held-seed, model-seed,
   decision/candidate, producer, pred, teacher, actor) — every gate clause is recomputable from it
   **without a checkpoint or the simulator** (HANDOVER.md:408) — plus per-fold `checkpoints/*.pt`.
8. **Device correction bypassed by the launch path.** (superseded by round-3 #12 below).

A **third review round** caught four more, all fixed test-first:

9. **Metadata alignment against the authority.** `_load_dump` now checks each candidate row's
   `task/seed/cycle/order_id` against the **decisions.csv** row (not just intra-group consistency), so
   a label can never bind to the wrong graph or leave-one-seed-out split. HANDOVER.md.
10. **Pre-registered shape enforced.** `run_probe(require_shape=True)` requires exactly **3 tasks × ≥3
    trajectory seeds × ≥3 model seeds** (a partial dump can no longer earn a mechanical PASS), and
    `_verdict` carries a `three_tasks` condition. `--expect-decisions/--expect-candidates` are
    validated in `_load_dump` (aliases reconciled). ARCH.md, HANDOVER.md.
11. **Determinism certified on the target device.** `train_gnn` pins BLAS to 1 thread ⇒ CPU is
    bitwise-exact; a `_determinism_preflight` trains a subset twice and aborts if `max_abs_diff`
    exceeds the declared tol (0 on CPU, 1e-4 on CUDA); device/torch/tol recorded. HANDOVER.md:381.
12. **Actor-floor fallback now affects the verdict.** `--actor-floor-json` (`{task: mean_regret}`)
    **replaces** the derived floor in the computed verdict (both recorded). The deterministic gate
    read runs on a **CPU pod** (see “Determinism vs device”); GPU is exploration-only.

A **fourth review round** caught four more, all fixed test-first (schema hash **unchanged** — these
touch audit binding + validation, not the observation):

13. **Graph↔decision binding.** Swapping two structurally-identical graph payloads (same candidate
    set, different live features) previously passed every check — nothing bound `graphs[di]` to
    `decisions.csv` row `di`. The dump now stamps each graph with a `decision_digest` of
    `(task_id, seed, cycle, order_id)` as a **separate shard array** (never a model tensor); `_load_dump`
    recomputes it from `decisions.csv` and rejects any mismatch. Finding-4.1.
14. **Pre-registered dump config enforced.** `_load_dump` now reads the dump's **own `spec.json`** and
    rejects any deviation from the pre-registered gate config — exact `task_ids` set, `delta`,
    `lookahead`, `horizon`, `eval_seeds`, `floor`, `has_actor` (a wrong teacher/task set preserves the
    decision count while changing every label). `require_shape` now requires **exactly 3 anchored
    trajectory seeds** (not ≥3) and the exact s29/s77/s45 task ids. Finding-4.2.
15. **Effective floor is consistent everywhere.** `metrics_by_task.csv` now reports
    `verdict_floor_regret` + `floor_source` (the floor the verdict actually compared against) alongside
    the derived actor floor, and a new `floor_by_task.csv` durably records derived + override + source —
    so an overridden verdict is reproducible from artifacts and the table never contradicts the read.
    Finding-4.3.
16. **Determinism certified on a real full-epoch fold.** `_determinism_preflight` now trains one genuine
    leave-one-seed-out fold at the run's **full `epochs`** twice and compares held-out predictions, not a
    40-decision × 10-epoch subset; `epochs`/`n_train_decisions`/`fold` recorded in `spec.json`. This is a
    **representative** certification (one real fold), not a re-run of all 27 fits — sound because CPU
    determinism is bitwise **by construction** (single-thread BLAS + per-fold seed), so one fold
    exercising every op certifies the path. Finding-4.4 (claim scope corrected in round 5, Finding-5.4).

A **fifth review round** caught four more (input provenance), all fixed test-first (schema hash still
unchanged):

17. **Input files fingerprinted + validated.** The dump previously recorded only `has_actor: true`, so a
    wrong producer (changes labels) or wrong actor (weakens the floor) — same shape — passed validation.
    The dump now records **SHA-256** of the archive + producer ckpt + actor ckpt (`inputs` manifest,
    `sha256_file`); the probe validates them against the pinned `GATE_INPUT_SHA256` (the frozen
    pre-registered files' hashes) and rejects a dump built from a wrong or unfingerprinted input.
    Finding-5.1.
18. **Exact task→seed sets enforced.** "Exactly three seeds" checked only the count. The dump records the
    anchored `task_seeds` and the probe enforces the exact sets **s29={29,30,31}, s77={77,78,79},
    s45={45,46,47}** (`GATE_TASK_SEEDS`) — validated both against the dump manifest and directly on the
    scored `candidates.csv` seeds. Finding-5.2.
19. **Floor CSV reproduces the verdict independently.** `floor_by_task.csv` rounded floors to 4 dp
    (`1.000049`→`1.0` could flip PASS↔FAIL). It now writes derived + verdict floors at **full precision**
    (`repr(float)`, round-trippable) plus the override JSON's **source path + sha256**, so the verdict is
    reproducible from that CSV alone. The display `metrics_by_task.csv` stays rounded. Finding-5.3.
20. **Determinism claim scoped.** The preflight is a **representative** full-epoch fold, not a whole-gate
    re-run; wording corrected (see #16). Finding-5.4.

A **sixth review round** caught three more (immutable source + authoritative floor + early input
check), all fixed test-first (schema hash still unchanged):

21. **Immutable source revision recorded** (scope widened + **enforced** in round 7, see #24). The
    observation-schema hash cannot detect a change to training/metric/verdict code, and the pod rsync
    excludes `.git/`, so nothing bound a run's artifacts to the code that produced them. Both specs now
    carry a `source` fingerprint — a git-independent **SHA-256 over the bytes of the whole synced
    `tokenomics` package tree** (every `.py`, so all transitive producers of the trajectory + labels
    are covered) plus a git **commit** via `--source-commit` (launcher has git; the pod does not).
    HANDOVER.md:413. Finding-6.1. **Commit the EXP-0039 source and pass its commit before launch.**
22. **Actor-floor fallback is authoritative, not arbitrary** (tightened to exact + pod-safe in round 7,
    see #25). `--actor-floor-json` accepted any `{task: regret}` — an inflated floor silently makes the
    licensing bar easier (verdict needs GNN regret < floor). The canonical floor is pinned:
    `GATE_ACTOR_FLOOR_0038B` (EXP-0038B `audit_sys` deployed-actor `mean_regret`), and
    `canonical_actor_floor()` derives it **mechanically** from the artifact gated by
    `GATE_FLOOR_SHA256`. On a `require_shape` run any override must **equal** the pinned floor or the
    gate aborts. Finding-6.2.
23. **Input hashes checked before spend, not after.** The dump recorded input SHA-256 but only the
    probe (after the finished dump) compared them — a wrong staged checkpoint burned the full ~4
    core-hour dump first. The dump CLI now takes `--require-gate-inputs`, calling `assert_inputs_match`
    against the pins **before** `run_graph_dump`, so a mismatch fails at launch. Finding-6.3.

A **seventh review round** caught that round 6's provenance was **audit-only**, plus two fallback
defects, all fixed test-first (schema hash still unchanged):

24. **Source identity is now a REQUIRED cross-stage invariant, not metadata.** `_load_dump` never
    validated the dump's `source` — a dump with a zeroed hash and a bogus commit was accepted, so a
    dump from source A could earn a verdict under source B. Now: a `require_shape` run **requires a
    valid git commit** (`_validate_commit` rejects `None`/`not-a-git-object`), and `_load_dump` calls
    `assert_source_matches` **before any training** — the dump's `tree_sha256` must equal the running
    probe's live fingerprint and its commit must equal the probe's `--source-commit`, else abort. The
    fingerprint was also widened from four hand-picked modules to the **whole synced tree** (the four-
    module subset silently ignored `marginal_value_labels`, `joint_stack_env`, curriculum, actor
    scoring, domain deps). Finding-7.1.
25. **Override validation is exact, and the verdict uses the pinned values.** `assert_override_is_canonical`
    used `tol=1e-6` — a `+5e-7` inflation passed and would become the verdict floor (a strict `<` gate
    can flip inside that interval). Now it is **exact equality**, and `run_probe` additionally
    **replaces** the validated override with the pinned `GATE_ACTOR_FLOOR_0038B` constants so no drift
    survives into the verdict. Finding-7.2.
26. **Pod-safe canonical fallback.** `--actor-floor-exp0038b` defaults to the artifact under
    `experiments/…`, which is **outside** the synced pod tree (`/workspace/tokenomics`), so the
    documented fallback would fail on a clean pod. New `--actor-floor-canonical` uses the in-code
    pinned floor directly (identical values, no artifact file); the staged probe command documents it.
    Finding-7.3.

An **eighth review round** caught that the recorded commit was still forgeable and the runtime env was
uncaptured, both fixed test-first (schema hash still unchanged):

27. **Commit is bound to a real, clean revision (launcher-side).** `_validate_commit` accepted any
    7–40 hex string — `a1b2c3d`, `deadbee`, forty zeros all passed — so a dirty tree could be stamped
    with a bogus id whose tree hash still matched. Now `_validate_commit` requires a **full 40-char**
    sha (rejects abbreviations), and new `resolve_verified_commit` (launcher-side, needs git) resolves
    HEAD, verifies the object **exists** (`git cat-file -e`) and the producing tree is **clean** (no
    modified/untracked files → the live fingerprint corresponds to that commit), and is wired as
    `--verify-source-commit` on both CLIs. The dump CLI now also validates the commit shape **before**
    `run_graph_dump` (under `--require-gate-inputs`), so a missing/malformed commit fails at launch,
    not after the dump budget. Finding-8.1.
28. **Runtime environment captured + compared.** Deps are open lower bounds (`torch>=2.2`, …) installed
    fresh on the pod; the probe recorded only `torch_version`. Both specs now carry an `env` manifest
    (`environment_manifest`: python/platform/implementation + numpy/torch/pettingzoo/gymnasium), and
    `_load_dump` compares the gated version keys exactly on a `require_shape` run — a probe cannot score
    a dump built under a different numpy/torch/etc. HANDOVER.md:413. Finding-8.2.

A **ninth review round** caught that the clean-commit binding was still manual and the env gate ignored
architecture/backend, plus stale test counts — all fixed test-first (schema hash still unchanged):

29. **Clean-commit binding is now launcher-enforced, not manual.** `resolve_verified_commit` existed but
    `run_pod_training.py` never called it — the staged commands still took a hand-typed `--source-commit`,
    and the pod (no `.git`) only checks the 40-char shape, so forty zeros passed end-to-end. The launcher
    now has `--verify-source-commit`: **before rsync** (the only place with `.git`) it resolves+verifies
    the clean 40-char HEAD (`resolve_clean_commit`, inline so the launcher needs no numpy/torch) and
    **substitutes it for a `{SOURCE_COMMIT}` placeholder** in `--train`; a dirty/non-repo tree aborts the
    launch. The binding no longer depends on following prose. Finding-9.1.
30. **Env gate now covers architecture + numeric backend.** `_ENV_GATED_KEYS` compared only versions —
    identical `numpy`/`torch` on a different CPU arch or BLAS is not the same numerical environment (a
    `machine="other-arch"`/`blas="different"` manifest was accepted). The manifest now also records
    `machine` (arch), `numpy_blas`, `torch_build` (hash of `torch.__config__`), and an `image` tag/digest
    (from `EXP0039_POD_IMAGE`, audit); the first three are **gated**. Finding-9.2.
31. **Test counts corrected.** The recorded 14/20/42 were wrong; actual collection is **dump 13, probe
    19, view 7, smoke 1 = 40 graph tests**, plus the new launcher binding test. The repo suite is the
    source of truth. Finding-9.3.

A **tenth review round** closed two remaining launcher gaps, both fixed test-first (schema hash still
unchanged):

32. **Cleanliness now covers the launcher itself.** `resolve_clean_commit` checked only the synced
    `tokenomics/` subtree — but `run_pod_training.py` lives outside it and *is* producing code (it
    resolves the commit, rewrites the command, launches). A launcher modified after HEAD (subtree clean)
    still returned the "clean" commit. It now checks the synced tree **and** the launcher file
    (`Path(__file__)`) against HEAD, aborting if either is dirty — but **not** the whole repo, since
    Phase-7 pulls intentionally dirty `experiments/` before Phase-8. Placeholder injection is now a
    tested helper (`apply_source_commit`). Finding-10.1.
33. **The pod image is actually recorded.** `environment_manifest` read `EXP0039_POD_IMAGE`, but nothing
    set it — the artifact recorded `image: null`. The launcher now resolves the pod image
    (`resolve_pod_image` → REST `GET /v1/pods/<id>` `imageName`) and **exports it into the remote run**
    (`EXP0039_POD_IMAGE=<img>` prefix on the python invocation, both detached + foreground paths), so the
    `env` manifest records the image the code ran under. Best-effort (unresolved → `null`, never fatal;
    the source/version gates stay the hard controls). Finding-10.2.

Also hardened: `_load_dump` validates the dump **schema hash** against the probe's; `run_probe` emits a
**computed pass/fail verdict** from the five conditions (no human reinterpretation).

**Compute budget (Finding-4.5, not an invalidation).** Single-thread BLAS (required for the bitwise-exact
CPU determinism above) makes each fit single-core; the gate is 27 sequential fits × 200 epochs ≈ **~4
core-hours** on the CPU pod (≈2.8 s/epoch on ~1k representative graphs). Launch **detached** (the Runpod
driver already does) and budget the pod accordingly; the determinism preflight adds ~2 more full-epoch
fits up front.

## Staged pod run (Phases 7–8)

**Step 1 (dump) → CPU pod** (label rollouts are CPU-bound; GPU buys nothing). **Step 2 (probe)
→ CPU pod** — the deterministic read must be **bitwise-exact**, which holds only on single-thread CPU
(declared tol 0); CUDA `index_add_` is order-nondeterministic (tol 1e-4), so GPU is exploration-only,
not the licensing read. Budget ~4 core-hours for the 27 fits (Finding-4.5). Stage the producer + actor
checkpoints into the synced package tree first (they live under `experiments/`, outside it):

```bash
mkdir -p tokenomics/_pod_ckpts
cp experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt tokenomics/_pod_ckpts/
cp experiments/EXP-0037-value-routed-orderer/gate/residual_scorer.pt         tokenomics/_pod_ckpts/
```

### 1. Graph dump (s29/s77/s45, real trajectory)

```bash
python run_pod_training.py --pod-id <id> --cpu --python python3.12 --verify-source-commit \
  --train "-u -m tokenomics.eval.graph_value_router_dump \
    --archive /workspace/tokenomics/_pod_inputs/gate_archive.jsonl \
    --task-ids <s29_id>,<s77_id>,<s45_id> \
    --prod-ckpt /workspace/tokenomics/_pod_ckpts/producer_candidate.pt \
    --actor-ckpt /workspace/tokenomics/_pod_ckpts/residual_scorer.pt \
    --delta 1.0 --lookahead 8 --eval-seeds 3 --horizon 100 --require-gate-inputs \
    --source-commit {SOURCE_COMMIT} \
    --out-dir /workspace/tokenomics/graph_runs/dump_graph" \
  --pull "/workspace/tokenomics/graph_runs/dump_graph:./experiments/EXP-0039-gnn-observation-probe/dump_graph"
```

`--verify-source-commit` resolves + verifies the clean 40-char HEAD **on the launcher before rsync**
(the only place with `.git`) and substitutes it for the `{SOURCE_COMMIT}` placeholder; a dirty/non-repo
tree aborts the launch (Finding-9.1). No manual sha, no forty-zeros bypass.

Stop-immediately checks after pull: `spec.json` `n_decisions == 13940`, `n_candidates == 37804`,
and `dump_graph/decisions.csv` **byte-identical** to
`experiments/EXP-0038-actor-teacher-error-microscope/dump/decisions.csv`
(`cmp` them). These counts are also enforced mechanically — pass `--expect-decisions 13940
--expect-candidates 37804` to the probe (below) and `_load_dump` aborts on mismatch. `--require-gate-inputs`
makes the **dump** abort at launch if a staged input's SHA-256 does not match the pins (Finding-6.3),
before ~4 core-hours of rollout. If the checked-in EXP-0037 `residual_scorer.pt` is pre-P2 and its
recomputed actor floor disagrees with EXP-0038B, add **`--actor-floor-canonical`** to the probe command
(pod-safe: uses the in-code pinned EXP-0038B floor, needs no artifact file — the `--actor-floor-exp0038b`
artifact lives outside the synced pod tree, Finding-7.3). It **replaces** the derived floor **in the
computed verdict** (both recorded in `spec.json`); on a `require_shape` run any override must **exactly**
equal the pinned floor or the gate aborts (Finding-6.2/7.2). Do NOT copy EXP-0038B's stale `source_dump`
provenance bug.

**Both stages carry the same clean 40-char commit** (the committed EXP-0039 revision): the dump stamps
it and the probe's `_load_dump` aborts **before any training** unless the dump's whole-tree source hash
AND commit match the probe's (Finding-7.1). With `--verify-source-commit` the **launcher** resolves it —
verifying HEAD exists and both the producing tree **and the launcher file** are **clean** before rsync
(the pod has no `.git`, Finding-8.1/9.1/10.1) — and injects it for `{SOURCE_COMMIT}`; a dirty tree/launcher
aborts the launch, so there is no manual sha to get wrong. The launcher also resolves the pod image and
exports `EXP0039_POD_IMAGE` into the run (Finding-10.2). `_load_dump` compares the dump's and probe's
**`env` manifest** (versions + **machine arch + numpy BLAS + torch build**, Finding-9.2); run both stages
on the **same image** so they match.

### 2. GNN probe + gate (CPU pod for the deterministic read)

```bash
python run_pod_training.py --pod-id <id> --cpu --python python3.12 --verify-source-commit \
  --train "-u -m tokenomics.train.graph_router_probe \
    --dump-dir /workspace/tokenomics/graph_runs/dump_graph \
    --out-dir  /workspace/tokenomics/graph_runs/probe \
    --epochs 200 --model-seeds 0,1,2 --source-commit {SOURCE_COMMIT} \
    --expect-decisions 13940 --expect-candidates 37804" \
  --pull "/workspace/tokenomics/graph_runs/probe:./experiments/EXP-0039-gnn-observation-probe/probe"
```

**Input provenance (validated by default).** On a `require_shape` run the probe rejects the dump unless
its recorded input SHA-256s match the pinned frozen inputs (`GATE_INPUT_SHA256`: archive
`bc6ca4eb…`, producer `8b51b77c…`, actor `666533e4…`) and its `task_seeds` are exactly
s29={29,30,31}/s77={77,78,79}/s45={45,46,47}. The dump records these from `sha256_file` of `--archive`,
`--prod-ckpt`, `--actor-ckpt`, so the frozen files must be the ones staged on the pod. If a frozen input
is ever deliberately regenerated, re-pin `GATE_INPUT_SHA256` (a conscious act). An `--actor-floor-json`
override records its own path+sha256 in `floor_by_task.csv`/`spec.json`.

**Determinism vs device.** HANDOVER.md:381 requires two same-seed runs to produce identical
predictions within a declared tolerance. `train_gnn` pins BLAS to a single thread
(`torch.set_num_threads(1)`), so on **CPU** the path is **bitwise-exact** (declared tol 0) — without
that pin, multithreaded FP reduction order jitters ~5e-4. CUDA `index_add_` is order-nondeterministic
regardless, so the probe declares a tol of `1e-4` on GPU. The gate read therefore runs on a **CPU
pod** for an exact, reproducible verdict; `run_probe`'s `_determinism_preflight` trains **one real
leave-one-seed-out fold at the full `epochs` twice** up front and **aborts** if `max_abs_diff` exceeds
the declared tol (fold/epochs/diff recorded in `spec.json`).
The probe is still device-aware (`--device cuda` works); use GPU only for exploration, not the read. The pull brings
back `spec.json` (verdict + device/torch/determinism + per-fold normalization/loss + checkpoint
provenance), `metrics_by_task.csv`, `predictions.csv` (gate recomputable without a checkpoint), and
`checkpoints/`.

## Predeclared gate (apply without reinterpretation)

Pass = **all** of: GNN mean regret beats the deployed actor floor on ≥2/3 tasks; centered
within-decision r² > 0 on ≥2/3 tasks; average relative regret reduction vs actor floor ≥10%
across the three tasks; no task worse than actor floor by >5%; stable across ≥3 training seeds.
Beating only random (not the actor floor) is a **fail**.

## Baselines to report in the same table

random floor · deployed actor floor · cheap current-row ridge/MLP · EXP-0038B full-tabular
`producer_product_system_v1` numbers (copied from `EXP-0038B/audit_sys/`) · primary graph probe ·
optional `graph_plus_actor_score` ablation (only after reading the primary result).

## Read (fill after pod run)

_Pass:_ hand metrics were too compressed; relational deployable state carries the teacher's
within-decision preference → open a follow-up to train + deploy a GNN orderer against `blend_0.5`.

_Fail:_ signal is not a stable function of deployable decision-time state even represented
relationally → retire the marginal-value actor target for learned orderers; move to
rollout/search as a compute lever or a directly-observable, held-out-learnable objective.

Topology limitation (state regardless of outcome): the gate holds out **trajectories**, not
unseen topology. A pass establishes within-topology, held-out-trajectory representability only.
Do not overclaim "not a function of any observation" — the defensible claim is "not learnable
under `graph_primitives_v1` and this probe gate".

## Results

Run: dump + probe on RunPod CPU pod (`runpod/base:0.7.0-ubuntu2004`, 16 vCPU / 32 GB), commit
`7c80a88`, image-matched across both stages. Dump = 13940 decisions / 37804 candidates; decisions.csv
teacher/context columns **byte-identical to the EXP-0038 dump across all 13940 rows** (0 diffs). Floor =
`--actor-floor-canonical` (pinned EXP-0038B `GATE_ACTOR_FLOOR_0038B`, sha `44d6d28f…`) — the checked-in
EXP-0037 `residual_scorer.pt` is pre-P2, so its recomputed actor picks drift from the EXP-0038 dump
(6715/13940 actor-column rows) while the teacher target is unchanged; the pinned floor bypasses the drift.
Determinism preflight: **bitwise-exact** (`max_abs_diff 0.0 ≤ tol 0.0`, CPU, 200 epochs). Metrics are the
held-out (leave-one-seed-out) mean over 3 model seeds.

| metric | s29 | s77 | s45 |
|---|---|---|---|
| random floor mean regret | 5.380 | 6.241 | 5.239 |
| actor floor mean regret (pinned EXP-0038B; derived-this-run) | 4.120 (4.005) | 4.754 (4.833) | 3.935 (4.374) |
| EXP-0038B tabular ladder (ridge, pooled) | \*best ≈ **5.30**, r²≈0, beats actor **0/3** — never cleared the floor | | |
| **GNN mean regret (mean ± std over seeds)** | **5.290 ± 0.084** | **6.231 ± 0.124** | **5.250 ± 0.026** |
| GNN centered r² (held-out) | **−0.759** | **−0.811** | **−0.720** |
| relative regret reduction vs actor | **−28.4 %** | **−31.1 %** | **−33.4 %** |

\* EXP-0038B `audit_sys/block_summary.csv` reports pooled (not per-task) ridge regret 5.30–5.68 across the
feature ladder, `ridge_beats_actor_ntasks = 0` at every block — the hand-metric wall this probe re-tests.

Gate conditions (all must hold): `three_tasks ✓` · `beats_actor_2of3 ✗ (0/3)` · `r2_pos_2of3 ✗ (0/3)` ·
`rel_reduction_10pct ✗ (avg −31 %)` · `no_task_worse_5pct ✗` · `stable_ge3_seeds ✗ (per-seed 0/3)`.

_Verdict:_ **FAIL.** A relational GNN over raw deployable state (`graph_primitives_v1`) does **not** recover
the EXP-0037 marginal-value teacher's within-decision producer preference on held-out trajectory seeds.
On every task the GNN's held-out regret sits **at the random-routing level** (5.29 vs 5.38; 6.23 vs 6.24;
5.25 vs 5.24) and its centered r² is **negative** — its predicted teacher-values transfer worse than
predicting the mean. Training loss fell steeply (≈49→1.5 per fold), so the model **fits the training
seeds but does not generalize**: a within-topology generalization gap, not an optimization failure. This
is the same wall the EXP-0038B tabular ladder hit; richer relational observation did not license the
target either.

**What it licenses:** nothing new — do **not** open a deployed-GNN orderer experiment on this target. The
defensible claim is scoped: the marginal-value teacher preference is not learnable under
`graph_primitives_v1` and this gate (held-out **trajectory**, fixed topology). Consistent with
DIAGNOSTICS #12 / EXP-0038 (target not a stable function of decision-time observable state). Next lever is
**not** a richer orderer observation — it is rollout/search as a compute lever, or a directly-observable,
held-out-learnable objective. See DIAGNOSTICS #14 for the `load_shard` OOM found + fixed mid-run.
