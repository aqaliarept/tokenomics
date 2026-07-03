# EXP-0029 — Stage 4 v2: Augment (not replace) Prioritized Replay

Controlled flip of **EXP-0027** to `augment=True` on the **diverse** EXP-0028 topology-grid
archive. The EXP-0026/0027 conclusion was: mined-*only* distill trained the producer from
scratch on a narrow set, **replacing** the champion's broad coverage and regressing the four
uneven-strength transfer cells. v2 tests the prescribed fix: **augment** the champions' original
broad train sets with the prioritized mined replay, keeping the coverage.

## What changed vs EXP-0027 (only this)

- **Augment train sets** (`augmented_train_configs`): producer = `broadened_producer_train_configs(0–5, d3, w4)` = 54 economies (scale + forkjoin + sharedcap + **uneven** + 5 scarcity per seed — the EXP-0021 champion set) **+** the mined replay; orderer = `build_scale_config(0–5, d3)` (6, the EXP-0012 set) **+** mined replay. This run: **producer 62 topologies (broad 54 + mined 8)**, orderer 14 (broad 6 + mined 8).
- **Archive** = EXP-0028 (26 tasks, 8 topology cells) instead of the EXP-0025 single cell.
- **Leakage guard**: mining seeds (0–29) overlap the held-out eval seeds (6–9); a mined scale cell at a held-out seed can reconstruct a held-out target. The driver drops replay tasks at held-out seeds — here **seeds 7 and 9 dropped**, so the candidate is never trained on a topology it is then evaluated on.
- **Hyperparams matched to the champions** (producer epochs 300 / lr 0.01 = EXP-0021; orderer iters 400 = EXP-0012) so the broad half reproduces the champion — any residual gap is then about data/augment, **not** under-training.

## Gate (advisory — promotion human-approved)

```json
{
  "val_candidate_units": 493.945, "val_champion_units": 494.055, "val_improves_or_ties": false,
  "blind_candidate_tail": 486.0, "blind_champion_tail": 484.0, "blind_tail_not_worse": true,
  "new_contract_failures": 0,
  "transfer_regressions": 2, "no_transfer_regression": false,
  "promote_recommended": false
}
```

Mined-VAL: champ 494.06 / cand 493.94 (tie, −0.11u). BLIND: champ mean 523.11 tail 484 / cand
mean 523.78 tail 486 (**candidate better**). Regressions **4 → 2**.

## The result — the producer uneven-strength regression is CLOSED

Joint held-out decomposition (scale s6–9), the attribution:

| stack | s6 | s7 | s8 | s9 |
|---|---|---|---|---|
| champion | 491.3 | 496.7 | 589.3 | 544.7 |
| **candidate_producer_alone** (cand prod + champ ord) | **491.3** | **496.7** | **589.3** | **544.7** |
| candidate_orderer_alone (champ prod + cand ord) | 493.3 | 496.7 | 591.7 | 532.3 |
| candidate_joint | 493.3 | 496.7 | 591.7 | 531.7 |

- **The augmented producer reproduces the champion exactly on held-out scale** (`producer_alone`
  == champion, every seed) — a clean no-regression retrain.
- On the **four uneven-strength cells** the joint candidate now **beats** champion:
  s6 584.0→585.67 (+1.67), s7 584.0→586.0 (+2.0), s8 487.67→491.67 (+4.0), s9 537.33→542.0
  (+4.67). **All four EXP-0027 regressions (−16 to −128u) are gone** — the augment kept the
  uneven family the mined-only run dropped, exactly as the EXP-0027 conclusion predicted.
- Elsewhere the candidate ties or beats champion: scarcity high_backlog +28.3, degen no_backlog
  +15.0, sharedcap/scale s6/s8 +2, uneven all +; 0 contract failures, 0 deaths.

## The residual — one orderer-side scale-s9 locus

The 2 remaining regressions are **scale-d3-s9** and **sharedcap-d3-w4-s9** (identical values —
the same underlying economy), 544.7 → 531.7 (**−13.0u, −2.4%**). The decomposition puts it on the
**orderer**: `candidate_orderer_alone` carries the −13 on s9 (532.3) while `producer_alone` is
champion-exact. It is a single held-out seed, and the orderer's broad set is scale-only, so the
mined replay perturbed the orderer's s9 allocation. This is orthogonal to the augment hypothesis
(which is about producer coverage) and much milder than the EXP-0027 uneven regressions it replaced.

## Conclusions

**Augment works — the producer regression is closed; verdict is REJECT-for-promotion but a
decisive positive for the mechanism.**

1. **Confirms the EXP-0026/0027 diagnosis and fix.** The uneven-strength regressions were a
   coverage hole from *replace-not-augment*. Augmenting (broad + mined) closes all four and turns
   them into small wins, with the augmented producer reproducing the champion on scale exactly.
   This is the first Stage-4 candidate with **no producer-side transfer regression**.

2. **Not promotable as-is** (advisory gate, human-approved): mined-val is a dead tie (−0.11u), and
   a new **orderer-side** −13u regression appears on scale-s9 (with its sharedcap twin) — 2
   regressions, so `no_transfer_regression=false`. Net trade vs EXP-0027: 4 producer-side uneven
   regressions (−16..−128u) → 1 orderer-side scale-s9 locus (−13u). Strictly better, not clean.

3. **Next lever is the orderer, not the producer.** The producer half is solved (augment reproduces
   champion + adds uneven wins). To clear the gate, stabilize the orderer retrain on s9 — options:
   lower the mined-replay weight into the orderer (the s9 perturbation tracks the mined draw, which
   was novelty-dominated / seed-14-heavy), or exclude the orderer from mined replay entirely (its
   scorer distills only from `proportional` and can't absorb the mined search-oracle headroom
   anyway — the standing EXP-0026 orderer-architecture limit). No promotion; champion untouched.

Candidate ckpts `artifacts/producer_candidate.pt` + `artifacts/orderer_candidate.pt` are **not
promoted** (git-lfs on commit).
