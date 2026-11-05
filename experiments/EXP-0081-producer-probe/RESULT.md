# EXP-0081: M5 L2, the gradient probe on rf3 (task 4.2)

**Question.** At θ = 0 on rf3, do the ES gradient estimates on two disjoint minibatches of 24 world-episodes
point the same way? Arms S (σ 0.05) and S2 (2σ). The pre-registered reading is v2 `5887b2b2…`: cosine ≥ 0.3 is
`signal`, ≥ 0.1 is `weak`, otherwise `no_signal`. This is not an acceptance read.

**Setup.**
- Commit 2687c55, `producer_es --mode probe`.
- Inputs: EXP-0080's bar, normaliser and margins.
- 2 arms × 2 minibatches × 24 members × 24 world-episodes = 2,304 episodes, horizon 100, MWWk.
- Pod: secure 8 vCPU in EUR-IS-1 (`aiqj4crcpytzse`; 16 vCPU was unavailable), 18:16–18:25 UTC.

## Registered reading: none (cosine undefined), so the step reading is `no_signal`. Stop branch.

| arm | minibatch A disqualified | minibatch B disqualified | ‖estimate‖ A / B | cosine |
|---|---|---|---|---|
| S (σ 0.05) | 3 / 24 | **24 / 24** | 10.95 / **0** | undefined |
| S2 (σ 0.10) | 8 / 24 | **24 / 24** | 4.18 / **0** | undefined |

There were 0 contract failures and all 2,304 rows are present.

## Diagnosis: DIAGNOSTICS 25, branch **disqualification-driven** (both discriminating checks)

**Check 1, the rule off** (`checks/check1_rule_off.py`; the recorded rows and noise stream, centred ranks of raw
fitness):

| arm | cosine, rule on | cosine, rule off |
|---|---|---|
| S | undefined | **0.840** |
| S2 | undefined | **0.902** |

The landscape has a strong, consistent direction. The rule removes it.

**Check 2, the rule's self-rate** (`checks/check2_self_flags.py`). No pod was needed: EXP-0080's r1–r4 re-keys of
rf3 at θ = 0 cover the probe's world-episodes. Each re-key is scored as a "member" over the minibatch:

- **Minibatch A:** 0 / 4 re-keys are disqualified.
- **Minibatch B:** **4 / 4** are disqualified.
- **Per episode on B:** rf3's re-keys flag 12 / 96 (12.5%); the members flag 116 / 1,152 (10.1%). So the members
  flag no more often than rf3's own draw noise does.
- **Where the flags come from:**
  - **Knife-edge world-episodes.** Train 12-0044 and 12-0007 (rf3 re-keys flag 4 / 4 on starvation; members
    35 / 48 and 23 / 48) and 12-0013 (2 / 4; members 17 / 48). On these, r0 is an outlier draw: any other key,
    including rf3's own, starves a consumer beyond the pooled margin.
- **Real deaths:**
  - On 12-0016, 24 / 48 members die, of both signs and both arms, while rf3's re-keys die 0 / 4.
  - On 12-0004, 5 / 48 members die.
  - These are θ-caused but not one-directional.
- **Why every member falls on B:** a member is disqualified if any of its 24 episodes flags. On minibatch B two
  world-episodes flag for almost any key, so every member ties at the worst rank and the estimate is exactly zero.
- **The calibration targeted the wrong level.** The 4.69% self-flag rate (EXP-0080) is per episode. Per member
  over 24 episodes, rf3 itself is disqualified on B with certainty.
- **This was seen before.** The same knife-edge world (12-0044) appears in DIAGNOSTICS 25 (EXP-0058) and 28
  (EXP-0070).

## Conclusions
- **The probe does not license the pilot under pre-registration v2.** The ladder stops (task 4.2 stop branch).
- **The cause is diagnosed.** The relative disqualification rule, aggregated per member with "any episode flags",
  fires on rf3's own draw noise on minibatch B. The policy class shows a strong gradient with the rule off.
- **Response per DIAGNOSTICS 25:** a rule redesign, which changes a registered read. This goes to the user, as a
  new pre-registration version. v2 is not edited.
- **Spend:** ≈ 1.1 vCPU-h billed (stop 4). M5-P ≈ 2.7 / 80.
