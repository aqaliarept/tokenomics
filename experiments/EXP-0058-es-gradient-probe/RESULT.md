# EXP-0058 — Bar precompute and ES gradient-signal probe

**Status: complete, valid. Not an acceptance read. Bar: complete (64 training worlds, 128
world-episodes, 0 replaced). Probe reading: `no_signal` (A1 cosine −0.07; A2's undefined): diagnose,
stop.** Task 6.1 of `add-m4-orderer-redesign` (design D6 / D7). Both steps ran at commit
`f3c895e258cb…`, which was verified clean, from a clean worktree. Inputs:
- atlas v2 = EXP-0052 `reclassified/atlas_v2.json` (content hash `220886a7cf12…`);
- EXP-0012 `orderer_scorer.pt` (`d25388bfa33e…`), A0's base and the TR bar arm;
- frozen roster v2;
- pre-registration `es_prereg_m4_v1.json` (`b9d449c15db6…`).

`entity_planner` producer, horizon 100. There were two fresh secure 16 vCPU pods, one per step
([Cost](#cost-and-the-stall)), 0 contract failures in either step, ≈ 6.9 vCPU-h billed.

## In plain English

**What this step is for.** Step 2 of the M4 orderer ladder trains a small correction ("residual")
on top of the trained orderer's scores. It uses evolution strategies (ES), which need no
gradients:
- each generation plays 12 pairs of slightly perturbed copies of the policy (+ε and −ε);
- it ranks them by how well they do on 12 training world-episodes;
- it moves the parameters toward the better-ranked perturbations.

Before paying for the pilot (EXP-0059, ≈ 12 vCPU-h), two things were needed:
1. The **bar precompute** fixes the training worlds and, on each world-episode, the *training
   bar*: the best of the hand-written routers and the trained orderer. Fitness is measured
   against it. The precompute also fits the feature normaliser, from the rows of A0 (the
   uncorrected policy).
2. The **gradient-signal probe** asks whether one ES generation points in a usable direction at
   all. It scores the same 12 perturbation pairs on two disjoint minibatches of 6
   world-episodes each and compares the two gradient estimates.
   - The **cosine** between the two estimates is near 1 if both minibatches agree on the
     direction, and near 0 if the estimates are noise.
   - Pre-registered (design D7): cosine ≥ 0.3 is `signal` (pilot as registered); 0.1–0.3 is
     `weak` (pilot with minibatch 24); < 0.1, or undefined, is `no_signal` (diagnose, stop).
   - It reads on arm A1 (a 44-parameter row-only residual). A2 (268 parameters, a low-rank
     bilinear term) is reported; its cosine is structurally lower at the start.

**Terms.**
- **ρ (tightness)**: consumer demand against the producers' starting throughput. A *loaded*
  world has ρ > 0.28.
- **Disqualified member**: a perturbed copy that, on some minibatch world-episode, does
  something A0 does not do on the same episode and random keys. That means it dies, runs down
  its capacity (r_end < 0.7516 · r_start), or starves a consumer (fill < 0.7302) that A0 does
  not starve. A disqualified member takes the worst rank (design D6, the *relative* rule).

**What happened.**

1. **The bar is complete and clean.**
   - 81 candidate worlds were screened (`bar/candidates.csv`): the 23 loaded atlas training
     worlds and 58 fresh draws. 12 fresh draws were dropped before play as not loaded
     (ρ ≤ 0.28). The other 69 were played, and 5 of those were dropped because a bar arm died
     there.
   - That leaves the target **64 training worlds, 128 world-episodes**. A0 died on none, so none
     was replaced; 1,380 episode rows; 0 contract failures. The mean training bar is 2,057.4
     consumer units.
   - A0 runs down on 78.1% of the world-episodes and starves some consumer on 98.4%. This is why
     D6 uses a *relative* rule: an absolute X / Y rule would disqualify almost everything.
   - The normaliser hash is `2d574ec8a3df…`. **The between-steps check passed**: the
     42-producer static-count clip flag (`static_psi_clip_42`) is not raised, no column clips
     heavily, and the completion-rate clip share is 0. Two columns are constant in the fit rows
     (`eligible`, `recipe_capacity_type_count`).
2. **The probe reads `no_signal`.**

   | arm | cosine between the two minibatch estimates | estimate norms (minibatch 1 / 2) | paired / unpaired variance ratio |
   |---|---|---|---|
   | **A1 (reads)** | **−0.070** | 5.21 / 6.80 | 0.046 |
   | A2 (reported) | **undefined** | **0.00** / 15.00 | 0.014 |

   A1's two estimates point in unrelated directions (cosine below 0.1). A2's first-minibatch
   estimate is exactly zero, so its cosine is undefined, which D7 reads as no signal. A2 stays at
   its registered rank 4: the rank-2 switch applies only when A1 ≥ 0.1. **Next, as
   pre-registered: diagnose, stop. The pilot (EXP-0059) does not run.**
3. **Almost every member was disqualified.** This is what the run itself records. It is not
   yet a diagnosis.

   | arm, minibatch | members disqualified (of 24) | flags | world-episodes that disqualify |
   |---|---|---|---|
   | A1, 1 | 21 | run-down 26, starvation 16 | 12-0014, 12-0005, 12-0139 |
   | A1, 2 | 16 | run-down 17, starvation 7, death 2 | 12-0000, 12-0044 (+12-0155 twice) |
   | A2, 1 | **24** | run-down 25, starvation 17 | 12-0014, 12-0139, 12-0005 |
   | A2, 2 | 18 | run-down 18, starvation 9 | 12-0044, 12-0000 |

   - A disqualified member takes the worst rank, so when every member of a minibatch is
     disqualified all ranks tie, and the ES estimate is exactly zero (A2, minibatch 1). With 16–21
     of 24 tied, A1's estimates rest on a handful of members.
   - The flags concentrate on 6 of the 12 world-episodes.
   - The paired variance ratio is small (0.05 / 0.01). The common random numbers work: a +ε / −ε
     pair differs far less than two unpaired members.
   - Task 5.2's horizon-12 dry run had already shown this pattern (4 of 6 members disqualified in
     generation 1), and it was carried forward to watch here.
   - Raw fitness spans 0.45–0.76 (A1) and 0.46–0.77 (A2).

   Whether the relative rule disqualifies on draw noise alone, the landscape is flat at the
   probe's step size, or 6 world-episodes are too few to agree on a direction is **not decided by
   this run**. `docs/DIAGNOSTICS.md` entry 25 names the discriminating checks.

**What changes because of it.** As pre-registered: `no_signal` → diagnose, stop. No ES pilot or
main run launches until the cause is diagnosed with evidence (CLAUDE.md, "Root cause before next
strategy"). The bar stays valid and reusable (supervisor decision 1): any later probe or pilot
reads this bar, so it is not run again.

## Details

### Method
- **Bar** (`order_es_worlds --mode bar --workers 16`, out `bar/`).
  - Candidate worlds are taken in a fixed order: the 23 atlas training worlds, then the fresh
    `world_space` draws, each screened (ρ, bar arms alive, bar fill < 0.95).
  - Every training world-episode plays 10 arms: RR, MWW × 2, TR, the 5 deterministic roster v2
    members and A0. That is 1,380 jobs including the dropped draws.
  - B_we = the max over the bar arms. The normaliser is fitted on A0's rows (median / IQR,
    log1p on size-scaled counts, clip ±5; the frozen `median-iqr-log1p-v2` rule).
- **Probe** (`order_es --mode probe --arms A1,A2 --bar-dir … --workers 16`).
  - Both arms start at their registered initial θ (Δ = 0), from the frozen pre-registration
    (`probe.theta`).
  - 12 antithetic pairs per arm, scored on two disjoint minibatches of 6 training world-episodes
    each (common random numbers: every member plays the same world-episodes and keys).
  - The estimate uses centered ranks per minibatch, with disqualified members ranked worst.
  - 576 jobs.

### Bar summary (`bar/bar_summary.json`)

| field | value |
|---|---|
| candidates screened / played | 81 / 69 (`bar/candidates.csv`; `bar_summary.json`'s `considered` = 69 counts the played ones) |
| training worlds / world-episodes | 64 / 128 |
| dropped before play, ρ ≤ 0.28 | 12 |
| dropped after play, a bar arm died | 5 |
| replaced, A0 died | 0 |
| A0 run-down share / starvation share | 0.781 / 0.984 |
| mean bar (B_we) | 2,057.4 |
| normaliser | `2d574ec8a3dfcac9fe399906a8cc093bbd2551ec9ba6d9495e238dabf3a96065` |
| normaliser flags | `static_psi_clip_42` [], `heavy_clip` [], completion-rate clip 0.0; constant columns `phi[16] eligible`, `psi[7] recipe_capacity_type_count` |
| schedule / training worlds SHA-256 | `122ebe64f117…` / `02abf7f7e247…` |

### Probe (`probe.json`, `probe_fitness.csv`)

| arm | cosine | disqualified (of 48) | non-finite | variance ratio | fitness min / median / max |
|---|---|---|---|---|---|
| A1 | −0.0696 | 37 | 0 | 0.0463 | 0.451 / 0.549 / 0.755 |
| A2 | undefined | 42 | 0 | 0.0144 | 0.463 / 0.607 / 0.767 |

- Minibatch 1: 12-0014, 12-0004, 12-0005, 12-0003, 12-0051 (fresh) and 12-0139 (atlas).
- Minibatch 2: 12-0000, 12-0044, 12-0002, 12-0033 (fresh) and 12-0155, 12-0308 (atlas).
- **Recomputed locally** from `probe_fitness.csv` with `order_es.probe_from_rows`: the estimate
  norms, variance ratios, disqualified / non-finite counts, A2's undefined cosine and the reading
  are equal. A1's cosine is equal to 6 × 10⁻¹⁷ (float rounding between pod and local).
  `probe.json` SHA-256 is `facfacfa8e75…`.

### Cost and the stall

**Billed ≈ 6.9 vCPU-h**: the bar pod ≈ 6.04 plus the probe pod ≈ 0.87. Tranche A billed so far is
≈ 12.0 of 100. Planned: one cpu8 pod, ≈ 3.0, stop 6.0. The stall and a supervisor decision made
it two pods.

| pod | shape | life (UTC) | billed | what ran |
|---|---|---|---|---|
| `fu1pfac93s95su` | secure 16 vCPU | 13:17:03 → 13:39:42 (22.65 min) | 6.04 | setup ≈ 1 min; the bar 13:18 → 13:23; pull; then idle until deleted at the hard stop |
| `d753h4sqnck39v` | secure 16 vCPU | 13:42:45 → 13:46:01 (≈ 3.3 min) | ≈ 0.87 | setup ≈ 1 min; the probe 13:43:36 → ≈ 13:45:50; pull, stop, terminate |

- **The stall, in plain words.** A local check ran between the two pod steps, on a thrashing
  shared box, while the pod sat idle and billed.
  - The plan ran the bar, then a quick check of its output, then the probe, on the same pod.
  - The check (reading three files of the pulled bar) ran on the local machine, which other
    agents' work was overloading (load 8.8). A two-second read took ≈ 14 minutes (13:24 → 13:38).
  - By then the pod was at 21.3 of its 22.5 allowed minutes (6.0 vCPU-h on 16 vCPU), with the
    probe still to run. It was deleted at the stop (the first delete timed out, hence 0.04 over),
    and the probe ran on its own pod (supervisor decision; stop 4.0, launch-fit 11.1 + 4.0 =
    15.1 ≤ 100).
  - Rule since then: no local step runs while a pod bills. Checks run on the pod or after the
    pull. See `docs/DIAGNOSTICS.md` entry 26.
- **Deviation 1 (approved by the supervisor): the bar pod ran with `--no-stop`**, not the plan's
  "`--keep-pod`, then `--pod-id`". `run_pod_training.py`'s `--volume-size` defaults to 0, so
  `/workspace` lives on the container disk, which RunPod clears when a pod stops. A stop and
  restart between the steps would have lost the bar and the venv.
- **Deviation 2 (recorded): the bar was staged under the git-ignored `_pod_ckpts/exp0058_bar/`**
  (without `fit_rows/`), not `_pod_inputs/`. `_pod_inputs/` is tracked and not ignored, so new
  files there make the tree dirty and `--verify-source-commit` refuses the launch.
  - The staged normaliser hash was checked (`2d574ec8a3df…`).
  - The driver accepted the bar before launch: 0 real-run problems.

### Files

| file | what |
|---|---|
| `spec.json`, `probe.json`, `probe_fitness.csv`, `contract_failures.csv` | the probe: pins (bar, normaliser and prereg hashes, source tree), the reading, per-member fitness / rank / disqualification reasons |
| `bar/` | the bar run as `order_es` reads it (`spec.json`, `bar.csv`, `training_worlds.json`, `schedule.json`, `normaliser.json`, `normaliser_report.csv`, `normaliser_static_42.csv`, `candidates.csv`, `timings.csv`, `bar_summary.json`, `world_records.json`, `contract_failures.csv`) |
| `artifacts/` (git-lfs) | `bar_episodes.csv` (the bar's 1,380 rows), `bar_fit_rows/` (A0's 138 fit-row files, 78 MB), `probe_episodes.csv` (the probe's 576 rows) |

Moving `artifacts/bar_episodes.csv` and `artifacts/bar_fit_rows/` back into `bar/` (as
`episodes.csv` / `fit_rows/`) restores the layout that `order_es_worlds --mode fit-normaliser
--bar-dir bar/` reads. `order_es` and 6.2's staging need neither.

## Conclusions

- **Bar.** Complete and valid: 64 training worlds, 128 world-episodes, 0 A0 deaths, the
  normaliser `2d574ec8a3df…` with no clip flag. It is reusable by any later probe or pilot.
- **Probe: `no_signal`.** A1's cosine is −0.07, and A2's is undefined (a zero estimate in one
  minibatch). As pre-registered: diagnose, stop. The pilot does not launch.
- **What the run shows, not yet a diagnosis.** 37 of 48 A1 and 42 of 48 A2 member-minibatch
  scores were disqualified by the relative run-down / starvation rule. That ties most ranks, and
  in A2's first minibatch all of them. DIAGNOSTICS entry 25 lists the discriminating checks:
  - the cosine with disqualification off, from the recorded fitness;
  - A0 against itself at another key, i.e. the rule's disqualification rate on noise alone;
  - pair differences against replicate noise.

  Their outcome decides the next step. No strategy is proposed before that.
- **Cost.** ≈ 6.9 vCPU-h billed. The stall came from a local check between pod steps, on an
  overloaded shared box, while the pod billed idle. It is fixed by rule (DIAGNOSTICS 26).
