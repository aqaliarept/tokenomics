# Critic-free policy gradients: multi-rollout baselines and paired (common-random-number) comparisons

Scope: methods that replace a learned value function with baselines computed from several rollouts, with emphasis on paired / CRN comparisons on identical scenarios. Target use: cooperative team (12-42 producers + ~50 orderers, shared GNN policy per role), deterministic seeded simulator, 100-cycle episodes, no centralized critic. Research date 2026-11-06.

## 1. GRPO: group-relative advantage, group sizes, failure modes (Dr. GRPO), use outside LLMs

### Takeaway
GRPO (DeepSeekMath, 2024) drops PPO's critic. It samples G outputs for the same prompt and uses the group's mean (and std) reward as the baseline, so it is a shared-start (same "scenario") multi-sample baseline. The std and per-response length normalizations bias it, and Dr. GRPO (2025) removes them, which makes it RLOO up to a constant. Outside LLMs, grouping unrelated parallel episodes (no shared start) fails on long-horizon control. Per-time-step group normalization (GR2PO, 2026) brings it close to PPO in MuJoCo. Multi-agent uses so far are LLM-team work (MAGRPO), not classical MARL.

### Cited Findings
- **GRPO advantage, outcome supervision:** r̃_i = (r_i − mean(r)) / std(r) over the G outputs of one question. Every token of output i gets Â_{i,t} = r̃_i. With process supervision, each token's advantage is the sum of the normalized step rewards at or after it. The KL term (Schulman's unbiased estimator π_ref/π_θ − log(π_ref/π_θ) − 1, coefficient 0.04) is added to the loss, not to the reward. Other settings: policy LR 1e-6, batch 1024, max length 1024, a single policy update per exploration stage. — [Shao et al. 2024, DeepSeekMath, arXiv 2402.03300](https://arxiv.org/html/2402.03300v3)
- **GRPO group size:** 64 outputs per question in DeepSeekMath. (The HTML text renders this as "6464", an OCR artifact for 64.) — [DeepSeekMath](https://arxiv.org/html/2402.03300v3)
- **Why GRPO drops the critic:** PPO's value function is "typically another model of comparable size as the policy model," which adds "a substantial memory and computational burden." No numeric saving is given. — [DeepSeekMath](https://arxiv.org/html/2402.03300v3)
- **GRPO results:** DeepSeekMath 7B scores 51.7% on MATH, and 60.9% with 64-sample self-consistency. The gain is credited to GRPO together with the data pipeline, so GRPO's share is not isolated. Process supervision (GRPO+PS) beats outcome supervision (GRPO+OS). Iterative RL (reward model retrained with 10% replay) helps most in the first iteration. — [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- **Dr. GRPO, two biases:**
  - Length bias: the 1/|o_i| factor favors short correct answers and penalizes long wrong ones less, so wrong answers grow longer.
  - Difficulty bias: dividing by the std of the group's rewards up-weights questions with low reward variance (very easy or very hard ones).
  - The fix: remove both normalizations, use a constant denominator, and set Ã_i = R_i − mean(R). This equals RLOO up to a factor G/(G−1).
  - Results: G = 8, Oat-Zero-7B scores 43.3% on AIME 2024, about 27 h on 8×A100. The paper also finds that several open-source PPO implementations carry the same length-normalization bias.
  — [Liu et al. 2025, arXiv 2503.20783](https://arxiv.org/html/2503.20783)
- **GRPO on classical control fails without shared starts:**
  - Setup: groups are the G concurrent episodes from parallel envs with independent resets (not a shared start), and one trajectory-level normalized return is applied to every timestep.
  - Result: all critic-free variants (batch mean, EMA, GRPO-batch) perform alike and "substantially below PPO" on long-horizon tasks (Acrobot, MountainCarContinuous, HalfCheetah). CartPole is the exception, where critic-free often beats PPO.
  - Group size: G=8 beats {16, 32, 64}. The authors suspect larger batch-groups mix unrelated episodes.
  - Discount: γ≈0.9–0.95 is best on HalfCheetah, which has no termination. γ=1 is worst there.
  - Stated cause: an episodic scalar baseline "cannot distinguish transitions of different value within one trajectory", and the signal is diluted over long non-terminating episodes.
  - Recommendation: use a critic for long horizons. Future work: state-based grouping.
  — [de Oliveira et al., Nov 2025, arXiv 2511.03527](https://arxiv.org/html/2511.03527)
- **GR2PO (Sep 2026), per-time-step group normalization:**
  - Method: compute per-env return-to-go G_t with no bootstrap, normalize across the N parallel envs at the same time index t: A_t = (G_t − μ_t)/(σ_t+δ), clip to [−c, c]. Uses the clipped surrogate, an entropy bonus and KL early stopping. States in a group are explicitly not required to be equal.
  - Results (final eval return, 5 seeds): Ant 6335 vs PPO 6610 / SAC 6847 / GRPO 1013; Walker2d 5024 vs 5965 / 5294 / 130; Humanoid 8331 vs 9484 / 7279 / 87. Budgets: 80M transitions (Ant, Walker2d), 400M (Humanoid).
  - Sensitivities: a GAE-style variant failed, and removing clipping hurt. A medium group size gave the best final return.
  — [Wang et al. 2026, arXiv 2609.19850](https://arxiv.org/html/2609.19850)
- **Continuous-control GRPO theory:** this framework paper says GRPO's use in continuous control is "unexplored". — [arXiv 2507.19555](https://www.arxiv.org/pdf/2507.19555)
- **MAGRPO (Liu, Chen, Liang, Lyu, Amato, 2025):** multi-turn, multi-agent GRPO for cooperating LLMs, formalized as a Dec-POMDP. It uses centralized group-relative advantages with decentralized execution and was tested on writing and coding collaboration. The authors report better response efficiency and quality and learned cooperation schemes. — [arXiv 2508.04652](https://arxiv.org/html/2508.04652v2)
- **GiGPO (Feng et al., NeurIPS 2025):** for multi-turn LLM agents.
  - Method: an episode-level group-relative advantage plus a step-level advantage. Step-level groups are built retroactively from "anchor states" that recur across the group's trajectories, and the actions taken from the same state are compared.
  - Results: more than 12% over GRPO on ALFWorld and more than 9% on WebShop, with no extra rollouts, no auxiliary model, the same GPU memory and little or no extra time.
  — [arXiv 2505.10978](https://arxiv.org/abs/2505.10978)

### Inferences
- Classical-control GRPO failed for two reasons: its groups did not share a starting scenario, and its advantage was trajectory-level. Our simulator can share both scenario and seed across a group. That is the LLM "same prompt" condition and the property the 2025 critique says classical GRPO lacked.
- With 100 cycles and a team reward, a single per-episode scalar advantage will likely dilute just as on HalfCheetah. Some per-time-index normalization (GR2PO) or state-anchored step groups (GiGPO, or explicit forks, see §3) are needed.
- Use the Dr. GRPO form (mean-centred, no std division), or report results with both forms. Dividing by std up-weights near-deterministic worlds, for example saturated or collapsed ones where every rollout gets almost the same reward.

### Gaps
- No primary source found for GRPO/RLOO in classical (non-LLM) cooperative MARL such as SMAC, MPE or supply chains. The emergentmind claim that multi-context GRPO was "validated in MARL" could not be traced to a primary paper.
- MAGRPO's group sizes and numeric results were not extracted.

## 2. RLOO (Kool et al. 2019; Ahmadian et al. 2024), ReMax and other multi-sample baselines: variance vs PPO with a critic

### Takeaway
RLOO uses k samples per input, and each sample's baseline is the mean of the other k−1. In RLHF with k=2–4 it beat PPO-with-critic by 10–32 win-rate points while loading one fewer model. ReMax uses a single greedy rollout as the baseline and cuts GPU memory by about 46%. These gains come from a regime that is effectively one-step: the full completion is one action with a terminal reward.

### Cited Findings
- **Origin:** Kool, van Hoof and Welling, "Buy 4 REINFORCE Samples, Get a Baseline for Free!" (ICLR 2019 DeepRLStructPred workshop). Drawing several samples per datapoint gives a free baseline. They also derive a version with samples drawn without replacement, which needs importance weights. Experiments: TSP tours. — [poster](https://wouterkool.github.io/pdf/poster-buy-4-samples-free-baseline.pdf), [abstract](https://sotaverified.org/papers/buy-4-reinforce-samples-get-a-baseline-for); follow-up: [Kool et al. ICLR 2020, arXiv 2002.06043](https://ar5iv.labs.arxiv.org/html/2002.06043)
- **RLOO estimator:** (1/k) Σ_i [R(y_i) − (1/(k−1)) Σ_{j≠i} R(y_j)] ∇ log π(y_i|x), with the y_i drawn i.i.d. from π_θ(·|x). — [Ahmadian et al. 2024, "Back to Basics", arXiv 2402.14740](https://arxiv.org/html/2402.14740)
- **Win rates (simulated GPT-4 judge):**

  | Method | TL;DR | HH-Pythia | HH-Llama |
  |---|---|---|---|
  | RLOO k=4 | 77.9 | 43.7 | 64.1 |
  | RLOO k=2 | 74.2 | 47.6 | 62.2 |
  | RAFT k=4 | 73.2 | 42.1 | 63.3 |
  | REINFORCE w/ baseline | 70.7 | 37.9 | 55.3 |
  | Vanilla PG | 70.4 | 36.4 | 52.3 |
  | PPO | 67.6 | 29.2 | 32.0 |
  | DPO | 66.6 | 39.0 | 61.9 |

  — [Ahmadian et al. 2024](https://arxiv.org/html/2402.14740)
- **Sample efficiency:** RLOO k=2 matches or beats RAFT k=4 with half the online samples. — [Ahmadian et al. 2024](https://arxiv.org/html/2402.14740)
- **PPO's machinery barely acts here:** PPO's clipping fired on fewer than 5% of samples per batch, and removing it slightly helped. Treating the whole completion as one action works at least as well as per-token modeling. — [Ahmadian et al. 2024](https://arxiv.org/html/2402.14740)
- **Variance numbers (Anthropic-HH, Table 2):** reward variance 3.1 for RLOO k=4, 3.0 for RLOO k=2, 3.7 for Vanilla PG and 2.3 for PPO. REINFORCE with a baseline shows about 27% less variance than Vanilla PG. PPO's lower reward variance did not translate into a better policy. — [Ahmadian et al. 2024](https://arxiv.org/html/2402.14740)
- **Memory:** PPO loads up to 4 models (policy, reference, reward, value). RLOO loads 3. — [Ahmadian et al. 2024](https://arxiv.org/html/2402.14740)
- **ReMax (Li et al., 2023/2024):** REINFORCE whose baseline is the reward of the greedy (argmax) rollout. It uses about 46% less GPU memory than PPO on 7B, needs no offloading, and removes more than 4 PPO hyperparameters. Mistral-7B reaches 94.78% on AlpacaEval and 7.739 on MT-bench. — [arXiv 2310.10505](https://arxiv.org/abs/2310.10505)
- **POMO (Kwon et al., NeurIPS 2020):** the same idea in combinatorial optimization (routing). It runs N rollouts per problem instance from different start nodes, and the baseline is the mean return of the N. It needs no critic and no extra greedy-rollout pass. Reported: 0.14% optimality gap on TSP100 and inference more than 10× faster. — [arXiv 2010.16011](https://arxiv.org/pdf/2010.16011)
- **Dr. GRPO ≡ RLOO** up to G/(G−1) once std/length normalization is removed. — [arXiv 2503.20783](https://arxiv.org/html/2503.20783)

### Inferences
- POMO and Kool's routing work are the closest non-LLM precedent for our setting. They are deterministic simulators: one instance, several rollouts, a shared baseline. The difference is horizon: their episodes are tens to hundreds of decisions with a terminal reward, while ours has dense per-cycle rewards and many agents.
- Expect the RLHF result (critic-free ≥ PPO) to carry over only where the effective decision is short-horizon or where per-step credit is recovered another way (§3, §5). §1's classical-control evidence points the other way for long horizons.
- A ReMax-style baseline costs 1 extra rollout per scenario. For us that could be the current deterministic policy (argmax actions), or a hand-written reference such as `entity_planner` + `round_robin` on the same seed. The reference variant is no longer "self-baseline", but it is still unbiased if the reference does not depend on the sampled actions.

### Gaps
- No source gives gradient-variance numbers for RLOO vs a critic in a long-horizon multi-agent simulator. The RLHF variance figures are reward variance, not gradient variance.
- ReMax's wall-clock speedup was not given in the abstract.

## 3. VinePPO and branch/vine rollouts from shared intermediate states; TRPO "vine"

### Takeaway
If the simulator can reset to any intermediate state, a critic can be replaced by Monte Carlo value estimates from K forked continuations. VinePPO (K=9, step-level advantages) is 2–5× slower per iteration than PPO but reaches PPO's peak accuracy in 1.5–3× less wall-clock. The learned critic it replaced was "near chance" at ranking actions. TRPO's vine procedure (2015) does the same and adds common random numbers across the K branches.

### Cited Findings
- **VinePPO method (Kazemnejad et al., ICML 2025):** for each state s_t in a training trajectory, re-feed the prefix and sample K continuations from the current policy. V̂(s_t) is their mean return, with default K = 9. The continuations serve only value estimation, never the policy update. Advantages are per reasoning step: steps are split on newlines and punctuation, and steps over 100 characters are split further. — [arXiv 2410.01679](https://arxiv.org/html/2410.01679)
- **VinePPO cost and gains:**
  - Per step: "up to 5x [slower] for RhoMath 1.1B and 2x for DeepSeekMath 7B compared to PPO".
  - Time to PPO's peak: about 3.0× and 1.51× less wall-clock, and 9× and 2.8× fewer gradient steps.
  - Memory: avoids the value network, which roughly doubles GPU memory in PPO.
  - Accuracy: beats PPO, GRPO, RLOO, RestEM and DPO+ on MATH and GSM8K, with a larger gap on MATH. Reference point: PPO reaches 42.8% on MATH with DeepSeekMath 7B.
  — [arXiv 2410.01679](https://arxiv.org/html/2410.01679)
- **Critic diagnosis:**
  - PPO's value network has explained variance 0.7–0.9, so it is not undertrained.
  - It is still biased and "performs near chance levels" at identifying the top action for most of training.
  - Its error grows with depth into the trajectory, while VinePPO's MC error shrinks.
  - Accuracy within 0.05 of the true value: PPO rises to about 65%, VinePPO stays at 70–90%.
  — [arXiv 2410.01679](https://arxiv.org/html/2410.01679)
- **Effect of K:** accuracy improves from K = 1 to 3 to 9. Higher K also improves wall-clock efficiency, and small K already works. — [arXiv 2410.01679](https://arxiv.org/html/2410.01679)
- **VinePPO's limitation:** it needs reset-to-any-state, which the authors call rare in Gym-style RL. — [arXiv 2410.01679](https://arxiv.org/html/2410.01679)
- **TRPO vine (Schulman et al., ICML 2015):**
  - Procedure: choose a "rollout set" of N states along the sampled trajectories, sample K actions at each, and estimate each action's Q by a short rollout.
  - CRN: the same random-number sequence is reused across the K rollouts from a state, which reduces the variance of the Q differences.
  - Trade-off: much lower variance than single-path for the same number of Q samples, but the system must be resettable to arbitrary states.
  — [TRPO arXiv 1502.05477](https://arxiv.org/pdf/1502.05477) (procedure confirmed via search-result extraction; [summary](https://medium.com/@hsinhungw/trust-region-policy-optimization-explained-c2671542c329))

### Inferences
- Our domain is deterministic and copyable, so the reset assumption holds for us. The repo already deep-copies systems per candidate in `eval/bounded_rollout_orderer` (EXP-0042) and in `teacher_audit`.
- A vine/fork estimator is therefore implementable. At a sampled cycle t, fork the system, run K continuations (or K alternative actions with shared demand and shock draws), and take the advantage of the realized action against the fork mean.
- Cost scales as (fork points per episode) × K × (rollout length to horizon or to a truncation depth). Truncated continuations need a bootstrap or a bias acceptance. VinePPO uses full continuations.
- With 100 cycles and about 60–90 agents acting per cycle, forking at every cycle is expensive. Sample a subset of fork cycles, as in TRPO's "rollout set".
- Multi-agent caveat: one fork compares joint actions, not individual agents. Per-agent credit would need per-agent alternative-action forks, which are difference rewards (§4) and cost O(agents × K) per fork point.

### Gaps
- TRPO's paper gives no quantified variance reduction from CRN in the vine procedure, at least none found via search.
- VinePPO's MATH/GSM8K per-method numbers live in figures only.

## 4. Common random numbers, paired rollouts, antithetic sampling (PEGASUS, ES, team credit)

### Takeaway
Fixing the scenario randomness (PEGASUS) turns policy evaluation into a deterministic function of the policy, so candidate and reference can be compared on identical scenarios. A paired difference has variance Var(A) + Var(B) − 2Cov(A, B), so it helps only when the paired outcomes stay positively correlated. Trajectories diverging under different actions erode that correlation. A 2026 study found that lower reward-contrast variance does not imply lower gradient variance or faster learning.

### Cited Findings
- **PEGASUS (Ng & Jordan, UAI 2000):** any (PO)MDP can be converted into an equivalent POMDP with deterministic transitions by fixing the random-number sequences ("scenarios"). Policy search then maximizes the empirical value over a fixed set of scenarios. Sample complexity is polynomial, not exponential, in the horizon (cf. Kearns, Mansour and Ng 1999), and the method applies to infinite state and action spaces. Demonstrations: a small discrete problem and bicycle riding. — [arXiv 1301.3878](https://arxiv.org/abs/1301.3878)
- **CRN lineage:** Kahn & Marshall (1953) introduced coupling the random draws of alternatives. Glasserman & Yao (1992) showed its benefit depends on the covariance it induces. TRPO's vine uses CRN for action comparisons. — cited in [Sakib 2026, arXiv 2609.24144](https://arxiv.org/html/2609.24144)
- **"Luck Is Not Skill" (Sakib, Sep 2026), paired rollouts in group-relative RL:**
  - Setup: Qwen3.5-2B with LoRA, a 13-tool fault-injected back-office simulator, 24 prompts × 8 rollouts, 100 steps, 3 seeds.
  - Pairing: all members of a group share one event-keyed noise schedule (key = tool + resource + repeat index). Each rollout's marginal distribution is unchanged, so leave-one-out estimators stay unbiased.
  - Theory: pairing removes the between-schedule part of reward-contrast variance, but "need not reduce gradient variance". Proposition 3 gives the exact condition, which depends on score-function geometry.
  - Counterexample (one-step bandit, G=8): contrast variance falls 9%, but gradient variance rises 2.2× (0.0026 → 0.0058).
  - Gradient probe: the covariance trace under pairing is 21–30% lower (grader flips) and 40–63% lower (tool faults).
  - Success: 66.8% vs 61.8% (+5.1 points, with one seed's AUC negative, so the pre-registered criterion was missed). No demonstrated speed-up in steps, tokens or wall time.
  - Cost: 52–54% of paired groups had constant reward (zero signal) vs about 13% unpaired.
  - Divergence: alignment weakens as history-dependent state diverges.
  - Advice: measure gradient-trace variance directly, and do not assume a gain.
  — [arXiv 2609.24144](https://arxiv.org/html/2609.24144)
- **Seeded vs independent ES (2026, LLM reasoning):** common seeds gave κ_pair = 0.819 at σ = 1e-4. At larger perturbation scales, paired subtraction did not reduce raw variance under either seed mode. — [arXiv 2608.27351](https://arxiv.org/pdf/2608.27351)
- **Antithetic / mirrored sampling in ES:** Salimans et al. (2017) evaluate each perturbation as a pair θ ± ε (Geweke 1988; Brockhoff et al. 2010). This is the most commonly applied ES variance reduction. Structured control variates (Tang, Choromanski and Kucukelbir, AISTATS 2020) are reported to beat it in RL. QMC and orthogonal directions are alternatives. — [Salimans et al. arXiv 1703.03864](https://arxiv.org/pdf/1703.03864), [Tang et al. arXiv 1906.08868](https://arxiv.org/pdf/1906.08868)
- **Coupling validity risks (imperfect-information games, 2026):** an invalid coupling can leak hidden state, synchronize policy randomness, or misalign chance events after histories diverge. The paper argues the right target is return covariance projected through the policy-gradient geometry, not contrast variance, and its selector falls back to independent sampling unless a lower confidence bound certifies improvement. — via search extraction of [arXiv 2512.24145](https://arxiv.org/pdf/2512.24145v2) (not fetched directly; treat as secondary)
- **Single-seed difference as an accept/reject test for candidate updates:** robotics paper; the variance reduction requires positive covariance (stated as a lemma). — [DC-Ada, arXiv 2604.03905](https://arxiv.org/pdf/2604.03905) (search snippet only)
- **Difference rewards (Wolpert & Tumer; Agogino & Tumer):** credit agent i with D_i = G(z) − G(z with i's action replaced by a default). When G is a known function this needs no learned estimator. — [Tumer publications](https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-holmesparker_aamas14.html), [AAMAS 2009 abstract](https://www.cs.huji.ac.il/~jeff/aamas09/pdf/02_Extended_Abstract/C_SP_0765.pdf)
- **Dr.Reinforce (Castellini, Devlin, Oliehoek, Savani; AAMAS 2021 best-paper award, NCA 2022):** policy gradient with difference rewards computed by differencing the known reward function, "avoid[ing] difficulties associated with learning the Q-function" in COMA's centralized critic. It is tested against REINFORCE, Q-AC, COMA and Colby on Multi-Rover and Predator-Prey. Dr.ReinforceR learns a reward network when the reward is unknown. — [arXiv 2012.11258](https://arxiv.org/pdf/2012.11258v2.pdf), [TU Delft](https://research.tudelft.nl/en/publications/difference-rewards-policy-gradients)

### Inferences
- **Two noise sources:** our episode seed fixes demand, drift, patterns and shocks, and the orderer's keyed draws (`set_draw_key`) already provide event-keyed CRN of the "Luck Is Not Skill" type. So "same seed" fixes the exogenous scenario completely, and the only remaining noise is policy sampling. Grouping by world-seed is exactly PEGASUS. It removes between-scenario variance, which is likely the dominant term because world difficulty varies widely (saturated vs hard vs collapse worlds in the atlas).
- **Candidate vs reference:** pairing a candidate against a reference (or the previous policy) on the same seed removes the world-difficulty term. The project's gate already uses a hindsight baseline on matched demand digests, so the eval side does this already. In training, the same logic gives a per-seed baseline.
- **Divergence risk:** in a coupled economy, small routing changes cascade through capacity, investment and death. Positive correlation between candidate and reference returns may decay across 100 cycles, especially on shocked worlds. Measure Cov(candidate, reference) per cycle offset before relying on it.
- **Dead groups:** expect zero-variance groups on saturated or collapse worlds, the analogue of the 52–54% constant-reward groups. Filter or down-weight them rather than dividing by std.
- **Per-agent credit:** difference rewards by counterfactual re-simulation (replace agent i's action by a default, re-run from the fork) are critic-free and exact in a deterministic simulator. Cost is one extra rollout per agent per fork point, so it suits sampled agents and fork cycles, not all of them.

### Gaps
- No primary source found that quantifies CRN gains specifically in policy-gradient training of a cooperative multi-agent simulator. The "Using Common Random Numbers for Simulation-based Planning with Rollouts" paper surfaced only via secondary review sites and was not verified.
- Gradient-variance numbers for PEGASUS-style fixed-scenario PG were not found.

## 5. Per-step (temporal) credit without a critic; long horizons (~100 steps)

### Takeaway
A pure trajectory-level baseline (GRPO/RLOO) fails at 100+ steps in classical control. Three critic-free ways to recover per-step credit are evidenced:
1. group statistics per time index on return-to-go (GR2PO: near PPO on MuJoCo);
2. state-anchored step groups (GiGPO: +9–12% over GRPO in multi-turn agents);
3. Monte Carlo forks from intermediate states (VinePPO / TRPO vine: best accuracy, 2–5× per-iteration cost).

Process-level rewards also beat outcome-only rewards in GRPO (DeepSeekMath).

### Cited Findings
- **Failure of episodic baselines:** an episodic scalar baseline "cannot distinguish transitions of different value within one trajectory". Critic-free variants lose to PPO-with-GAE on long-horizon Gymnasium tasks (1M steps, 8 envs). — [arXiv 2511.03527](https://arxiv.org/html/2511.03527)
- **Per-time-index normalization:** normalizing return-to-go across parallel envs at the same t reaches about 84–96% of PPO's eval return (Walker2d 5024/5965, Humanoid 8331/9484, Ant 6335/6610) on Ant, Walker2d and Humanoid. Immediate-reward GRPO collapses (eval 87–1013). — [arXiv 2609.19850](https://arxiv.org/html/2609.19850)
- **Step-level groups from recurring states:** ALFWorld +12%, WebShop +9% vs GRPO at no extra rollout cost. — [arXiv 2505.10978](https://arxiv.org/abs/2505.10978)
- **Process vs outcome rewards:** with a per-step return-to-go of normalized step rewards, GRPO+process supervision beats GRPO+outcome supervision. — [DeepSeekMath](https://arxiv.org/html/2402.03300v3)
- **MC value from forks:** step-level MC advantages beat a learned critic. The critic's error grows along the trajectory, while the MC error shrinks. — [arXiv 2410.01679](https://arxiv.org/html/2410.01679)
- **Discounting:** γ≈0.9–0.95 helps critic-free learning on non-terminating long tasks. — [arXiv 2511.03527](https://arxiv.org/html/2511.03527)

### Inferences
- Our reward is dense per cycle (units completed, run-down, deaths), so per-cycle return-to-go is available. The natural critic-free estimator is: for a group of R rollouts on the same world-seed, advantage at cycle t = (G_t^i − mean_j G_t^j), optionally with γ < 1. This is GR2PO, but with groups sharing the scenario, which GR2PO's groups do not, so its variance should be lower.
- The per-cycle paired difference (candidate units at cycle t − reference units at cycle t, same seed) is the temporal analogue. It isolates when the gain or loss happens but still mixes which agent caused it.
- The rollouts in a group share only cycle 0's state; states at the same cycle t diverge afterwards. A per-index baseline is then a state-free baseline: unbiased but higher-variance. GiGPO's anchor-state matching is the fix in discrete text environments. Our states are continuous and rarely recur exactly, so periodic forks (VinePPO) are the closer analogue.

### Gaps
- No source found that runs critic-free PG on a ~100-step cooperative multi-agent economy or supply-chain simulator.
- No head-to-head of GR2PO-style per-index baselines vs forked MC baselines.

## 6. Sample cost multipliers vs actor-critic PPO

### Takeaway
There is no universal multiplier. Rollouts per scenario run from k=2 (RLOO) through G=8 (Dr. GRPO, best in classical control) to G=64 (DeepSeekMath). Per-iteration cost runs up to 2–5× PPO for K=9 vine forks (VinePPO), offset by 1.5–3× faster time-to-quality. Memory saving is 1 model (RLOO, 3 vs 4) or about 46% (ReMax). In classical control, critic-free methods lost to PPO at equal environment steps.

### Cited Findings
- **Rollouts per scenario:**
  - RLOO k=2 and 4; k=2 matches RAFT k=4 at half the samples. — [arXiv 2402.14740](https://arxiv.org/html/2402.14740)
  - GRPO G=64. — [DeepSeekMath](https://arxiv.org/html/2402.03300v3)
  - Dr. GRPO G=8, about 27 h on 8×A100 for a 7B model. — [arXiv 2503.20783](https://arxiv.org/html/2503.20783)
  - Classical control: G=8 beats 16–64. — [arXiv 2511.03527](https://arxiv.org/html/2511.03527)
- **VinePPO:** K=9. Per step up to 5× (1.1B) / 2× (7B) slower than PPO. Time to PPO's peak 3.0× / 1.51× less, with 9× / 2.8× fewer gradient steps. — [arXiv 2410.01679](https://arxiv.org/html/2410.01679)
- **ReMax:** +1 greedy rollout per prompt, about 46% less GPU memory than PPO. — [arXiv 2310.10505](https://arxiv.org/abs/2310.10505)
- **GR2PO:** no step-efficiency gain over PPO, which learns faster early on some tasks. Wall-clock is shorter than SAC on all three tasks and shorter than PPO on Walker2d and Humanoid. — [arXiv 2609.19850](https://arxiv.org/html/2609.19850)
- **Paired rollouts:** no demonstrated step, token or wall-time gain over unpaired GRPO. — [arXiv 2609.24144](https://arxiv.org/html/2609.24144)

### Inferences
- **Cost per update for us:** (number of world-seeds) × R rollouts × 100 cycles, plus the fork overhead if vines are used. In this repo, rollouts are batch-1 sequential Python, one process per core, and the expensive part is the simulator, not the network. A critic's main cost here would be learning difficulty (credit across 60–90 agents and 100 cycles), not compute. VinePPO's finding that a converged critic can be near chance at ranking actions applies directly.
- **Starting point:** R = 4–8 per seed with the RLOO / Dr. GRPO mean-centred baseline and per-cycle return-to-go. Sparse fork points (for example 4–8 cycles per episode, K = 3–9) only if per-index baselines prove too noisy. Measure gradient-trace variance with and without pairing before committing pod budget.

### Gaps
- No source compares environment-step cost of critic-free vs critic PG in cooperative MARL. Any multiplier for our setting must be measured.
