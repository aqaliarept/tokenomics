# EXP-0036 Stage A0b pre-A1b checks (blend value + choice learnability)

A0's `named_capture_frac` proved the orderer headroom is REACHABLE inside a named-policy
vocabulary via the lookahead oracle. It did NOT prove (a) that the alpha-blend family adds
anything over hard selection, nor (b) that the committed choice is predictable from the
stream observation — the two assumptions Stage A1b's head rests on. This run tests both.

**Read order:**

1. **Blend value (D2):** `blend_capture_frac` vs `named_capture_frac`. A clear gain => the
   continuous residual is blend-recoverable => keep the alpha head. ~equal => plain
   categorical selection suffices => drop alpha.
2. **Choice learnability (D1):** held-out `choice_top1` vs `choice_majority` (and the
   decisive-margin split). Near-majority => the categorical target inherits the same
   unpredictability wall (A1b would fail; fall back to A1 denoise / A2 deploy-time search).
   Well above majority => A1b is green-lit.
3. **Global-state probe (A0c, read only if D1 failed):** held-out `gchoice_top1` vs
   `gchoice_majority` — the same protocol from the ADR-0004 global snapshot, one sample per
   searched cycle. Beats its majority floor => the D1 wall is missing CONTEXT (enrich the
   orderer obs with global features). Fails too => enrichment with THESE features is dead —
   but see Conclusions: the snapshot is a lossy roll-up, so a fail does NOT prove the choice
   is unpredictable from any observation. *(This paragraph originally overclaimed "rollout
   NOISE; no observation can predict it" — corrected 2026-07-05, emitter template fixed.)*

**Means:** named_capture_frac=0.7839, blend_capture_frac=0.9008, choice_top1=0.3400446576781035, choice_majority=0.5191927114636595, choice_top1_decisive=0.27742211830071123, choice_majority_decisive=0.4139432909595889, gchoice_top1=0.3333333333333333, gchoice_majority=0.4966666666666667, gchoice_top1_decisive=0.23737867239919225, gchoice_majority_decisive=0.41870236466679694.

## Per-task

```json
[
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "search_cu": 1070.67,
    "even_split_cu": 809,
    "search_named_cu": 1070.67,
    "search_blend_cu": 1062.67,
    "named_capture_frac": 1.0,
    "blend_capture_frac": 0.9694,
    "choice_top1": 0.27058823529411763,
    "choice_majority": 0.5294117647058824,
    "choice_top1_decisive": 0.13123844731977818,
    "choice_majority_decisive": 0.4269870609981516,
    "n_choice_decisive": 541,
    "n_choice_train": 2414,
    "n_choice_heldout": 1275,
    "gchoice_top1": 0.31,
    "gchoice_majority": 0.53,
    "gchoice_top1_decisive": 0.19047619047619047,
    "gchoice_majority_decisive": 0.40476190476190477,
    "n_gchoice_train": 200,
    "n_gchoice_heldout": 100,
    "choice_heldout": true,
    "vocab": [
      "even_split",
      "blend_0.75",
      "blend_0.5",
      "blend_0.25",
      "proportional"
    ]
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "search_cu": 997.33,
    "even_split_cu": 513.67,
    "search_named_cu": 847.67,
    "search_blend_cu": 972.33,
    "named_capture_frac": 0.6906,
    "blend_capture_frac": 0.9483,
    "choice_top1": 0.41720683559222155,
    "choice_majority": 0.5615792575132587,
    "choice_top1_decisive": 0.43031784841075793,
    "choice_majority_decisive": 0.5293398533007335,
    "n_choice_decisive": 818,
    "n_choice_train": 3510,
    "n_choice_heldout": 1697,
    "gchoice_top1": 0.29,
    "gchoice_majority": 0.54,
    "gchoice_top1_decisive": 0.19607843137254902,
    "gchoice_majority_decisive": 0.5490196078431373,
    "n_gchoice_train": 200,
    "n_gchoice_heldout": 100,
    "choice_heldout": true,
    "vocab": [
      "even_split",
      "blend_0.75",
      "blend_0.5",
      "blend_0.25",
      "proportional"
    ]
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "search_cu": 981.67,
    "even_split_cu": 199.67,
    "search_named_cu": 716.67,
    "search_blend_cu": 813.33,
    "named_capture_frac": 0.6611,
    "blend_capture_frac": 0.7847,
    "choice_top1": 0.33233890214797135,
    "choice_majority": 0.4665871121718377,
    "choice_top1_decisive": 0.27071005917159763,
    "choice_majority_decisive": 0.28550295857988167,
    "n_choice_decisive": 676,
    "n_choice_train": 3110,
    "n_choice_heldout": 1676,
    "gchoice_top1": 0.4,
    "gchoice_majority": 0.42,
    "gchoice_top1_decisive": 0.32558139534883723,
    "gchoice_majority_decisive": 0.3023255813953488,
    "n_gchoice_train": 200,
    "n_gchoice_heldout": 100,
    "choice_heldout": true,
    "vocab": [
      "even_split",
      "blend_0.75",
      "blend_0.5",
      "blend_0.25",
      "proportional"
    ]
  }
]
```

## Conclusions

**A0c verdict (CORRECTED on review, 2026-07-05 — the original verdict overclaimed): the
ADR-0004 rolled-up global snapshot cannot predict the committed choice at ~200 train
cycles/task; whether ANY observation can is still OPEN.** Held-out gchoice_top1 vs
gchoice_majority:

- s29: 0.31 vs 0.53 (decisive 0.19 vs 0.40) — clear FAIL
- s77: 0.29 vs 0.54 (decisive 0.20 vs 0.55) — clear FAIL
- s45: 0.40 vs 0.42 (decisive 0.33 vs 0.30) — at the floor (within n=100 binomial noise)

2/3 clear fails, 1/3 statistical tie; nowhere does the probe beat its majority floor.
D2 and D1 reproduce A0b exactly (deterministic domain — same search_cu / captures; stream
choice_top1 varies only with torch init).

**Why the original "no observation can predict it / branch (b) picked" conclusion was
withdrawn:** the probe input `env.global_state()` (`encode_global`, ADR-0004) is NOT the
full env state — it is a lossy roll-up:

- per-producer capacity is **summed across types** (`sum(v.capacities.values())`) — but
  bottlenecks and system death on these primal-tight tasks are per-type;
- **no order-queue state** (no in-progress orders, no per-order TTL/age, no backlog
  composition — the stream obs carries per-slot TTL; the global vector does not);
- inventory appears as a **unit count**, not composition by product;
- **no history** — an instantaneous snapshot; given a lossy snapshot, temporal aggregates
  are exactly how hidden state gets recovered.

On top of the lossiness, the probe trained on only ~200 cycles/task (n_gchoice_train=200,
held-out 100): failing to beat a ~50% majority floor from 200 samples of a high-dim 5-way
problem is weak evidence of impossibility. Since domain + search are deterministic
(n_samples=0), the committed choice IS an exact function of the FULL env state — the target
function provably exists; A0c only shows the ADR-0004 roll-up at n=200 does not learn it.

**What A0c DID close:** enriching the orderer obs by concatenating the *existing* global
feature vector — that exact information content was tested and failed.

**Three causes still standing (root-cause rule: run the discriminating check before
committing to a branch):**

1. **lossy snapshot** → probe from a lossless env-state serialization (per-type capacities,
   queue/TTL detail, inventory composition by product);
2. **missing temporal features** → k-cycle history stack / hand aggregates (degradation
   trend, rejection rate by layer, capacity deltas) — backward-looking proxies for the very
   dynamics the lookahead unrolls forward;
3. **genuine incompressibility** at feasible sample sizes → A2 deploy-time light search
   (realizes blend capture 0.90 mean by construction; EXP-0009 showed short rollouts
   suffice) or A1 soft/advantage labels (gated on a critic held-out ranking check —
   decisive-cycle fails suggest ranking from state is hard too).

**Next (A0d): complete-state probe** — same protocol, lossless env-state serialization,
± k-cycle history stack, ~10× mined cycles. Still fails → incompressibility is diagnosed
with evidence and A2 is confirmed. Passes → enrichment is back alive; the follow-up is
which compact features carry the signal (temporal aggregates first).

Open caveats: per-sample margins still not persisted ("decisive" = margin > 0, which can
sit within rollout-noise scale). No trace-level teacher-vs-probe comparison was done (only
aggregate top-1); decisive-cycle accuracy being WORSE than overall (0.19 vs 0.31 on s29)
says the errors are not near-tie noise, but which cycles fail and why has not been read.
