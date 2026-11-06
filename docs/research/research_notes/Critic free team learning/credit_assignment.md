# Critic-free multi-agent credit assignment: difference rewards and counterfactuals computed by simulation

Context: 12-42 producer agents and about 50 ordering agents. Each role shares one GNN policy. The team reward is consumer units delivered per cycle. The simulator is deterministic, seeded and can be forked. The question is how to give each agent a lower-variance signal than the shared team return without a learned value network.

Research note, 2026-11-06. Some primary PDFs (AAMAS 2007 air traffic, the SCC full text) could not be text-extracted, so some numbers below come from abstracts or publication-page summaries. Each case is flagged.

## 1. Difference rewards / Wonderful Life Utility (Wolpert & Tumer; Agogino & Tumer)

### Takeaway
A difference reward is D_i = G(z) − G(z with agent i's action replaced by a counterfactual). It stays aligned with the team objective ("factored") and is much more sensitive to agent i's own action ("learnable"). In the Tumer group's domains it gave large gains over the shared global reward, with reported scaling to 1,000-10,000 agents. Its practical weakness is computing G(z_{−i} + c_i) for every agent. Each agent needs one extra evaluation of the world utility, or |A_i| evaluations for the policy-weighted "aristocrat" form. The Tumer group's own later work turned to estimated or modelled difference rewards because the exact form is hard to compute in many domains.

### Cited Findings
- **Definition (WLU / clamping).** Wolpert & Tumer define the Wonderful Life Utility for agent η as G(z) − G(z_η̂, CL_η). The agent's move is replaced by a fixed "clamping" choice CL_η, and the result is factored whatever clamp is used. Earlier COIN work set the clamp to zero. The 2001 paper "Optimal Payoff Functions for Members of Collectives" (Advances in Complex Systems, vol. 4, pp. 265-279) derives the optimal clamp as the one giving a mean-field approximation to the Aristocrat Utility (AU). Bar Problem experiments corroborated the predicted gain over clamping to zero. — [Wolpert & Tumer 2001, NASA NTRS copy](https://ntrs.nasa.gov/api/citations/20010066523/downloads/20010066523.pdf); [Tumer publication page](https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-wolpert_acs01.html); [IDEAS/RePEc record](https://ideas.repec.org/a/wsi/acsxxx/v04y2001i02n03ns0219525901000188.html)
- **Default-action form vs aristocrat (policy-expectation) form**, as formalised by Castellini et al.:
  - Default-action form (Eq. 4): ΔR^i = R(s,a) − R(s, ⟨a^{−i}, c^i⟩), with a fixed default action c^i.
  - Aristocrat-utility form (Eq. 5): ΔR^i = R(s,a) − E_{b^i∼π_i}[R(s, ⟨a^{−i}, b^i⟩)]. This removes the need to choose a default action.
  — [Castellini et al., Difference Rewards Policy Gradients, arXiv 2012.11258v2](https://arxiv.org/html/2012.11258v2)
- **"Aligned and sensitive" = factoredness and learnability.** Agogino & Tumer (JAAMAS 2008, vol. 17(2), pp. 320-338) measure how well an agent's reward aligns with the global reward (factoredness) and how sensitive it is to the agent's own actions (learnability). The two trade off: a highly factored reward tends to be poorly learnable, and vice versa. The paper compares three rewards on a continuous, noisy multi-rover domain (static and dynamic):
  - team game reward (each agent gets G)
  - "perfectly learnable" local reward
  - difference reward

  In the dynamic version, the team-game and perfectly-learnable rewards perform poorly and the difference reward performs well. — [Tumer publications page, JAAMAS 2008](https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-agogino_jaamas08.html); [typeset.io copy](https://typeset.io/pdf/analyzing-and-visualizing-multiagent-rewards-in-dynamic-and-3rti1jfwux.pdf). The reward-analysis idea first appeared in AAMAS 2005 ("Multi Agent Reward Analysis for Learning in Noisy Domains") and GECCO 2004 (multi-rover evaluation functions). Seen via the search summary only; full text not verified.
- **Air traffic flow management (FACET simulator, AAMAS 2007).** Agents with personalised (difference) rewards cut congestion by up to 45% compared with agents using the global reward. They cut it by up to 67% compared with the industry approach (Monte Carlo estimation). — [Tumer & Agogino, AAMAS 2007](https://www.ifaamas.org/Proceedings/aamas07/html/pdf/AAMAS07_0453_bd798e682c64b239665c2e4d0efcd1df.pdf). Abstract figure only; the PDF body could not be extracted.
- **Air traffic, evolutionary version (GECCO 2007).** The scenario had 300 aircraft and two congestion points. Agents evolving against the system evaluation gave about a 40% gain over a direct evolutionary algorithm. Agent-specific difference evaluation functions added a further ~30%. — [Tumer & Agogino, GECCO 2007](https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-agogino_gecco07.html)
- **Journal version** (Autonomous Agents and Multi-Agent Systems, DOI 10.1007/s10458-010-9142-5). It compares the system reward, the difference reward and a difference reward *estimated* to lower computation cost. — [AAMAS journal PDF mirror](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/s10458-010-9142-5.pdf). Search summary only.
- **Computing difference rewards gets expensive.** Curran, Agogino & Tumer (AAMAS 2013) report "severe computational issues" when the difference reward is combined with a greedy scheduler in air-traffic hard-constraint handling. — [AAMAS 2013 p1281](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2013/docs/p1281.pdf). Search summary.
- **Scaling to many agents (Tumer group):**
  - Holmes-Parker & Tumer compare an agent-centric difference-reward method against global and local rewards with up to 10,000 agents. A workshop paper applies hierarchical difference rewards to a 10,000-sensing-agent Defect Combination Problem.
  - CLEAN rewards (Agogino & Tumer) are reported to significantly outperform other methods with up to 1,000 agents in a congestion problem.
  - Difference rewards beat global rewards with 1,000 agents in the Gaussian Squeeze domain.

  — [Tumer & Holmes-Parker, ALA 2012](https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-holmesparker_ala12.html); [Holmes-Parker et al., AAMAS 2013 structure paper](https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-holmesparker-structure_aamas13.html); [AAMAS 2012 Z6_8](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2012/papers/Z6_8.pdf). All from search summaries; per-experiment numbers were not verified.
- **When G is unavailable or costly, model the difference reward.** Proper & Tumer (AAMAS 2012) learn a reward model to estimate D. In a very large-scale air-traffic simulation, the estimate greatly outperforms the other methods. Proper & Tumer (AAAI 2013) address the criticism that D is hard to use without an equation for G(z) by working from a noisy global reward. — [Proper & Tumer, AAMAS 2012](https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-proper_aamas12.html); [Proper & Tumer, AAAI 2013](https://mlanthology.org/aaai/2013/proper2013aaai-multiagent)
- **Collective (large-population) setting.** Nguyen, Kumar & Lau (NeurIPS 2018) extend difference rewards to collective multiagent RL with a non-decomposable global reward. They test on taxi-fleet optimisation, multiagent patrolling and grid navigation, and report significantly better solutions than prior methods. One reviewer judged the contribution incremental over existing difference-reward work. — [NeurIPS 2018 paper](https://papers.NeurIPS.cc/paper_files/paper/2018/file/94bb077f18daa6620efa5cf6e6f178d2-Paper.pdf); [Reviews](https://papers.neurips.cc/paper_files/paper/2018/file/94bb077f18daa6620efa5cf6e6f178d2-Reviews.html)

### Inferences
- **What one cycle costs here.** With a forkable deterministic simulator, the exact default-action D_i for every agent at one decision cycle costs N extra counterfactual evaluations. That is about 54-92 forks per cycle (12-42 producers plus ~50 orderers), each played to wherever the reward is read.
  - The reward is units delivered per cycle. Orders have TTL 2, and investments act with a delay. So a one-cycle counterfactual misses most of an investment action's effect.
  - The fork would have to run H cycles, giving roughly N·H extra simulator cycles per decision cycle.
  - The policy-weighted (aristocrat) form multiplies this by K sampled alternative actions per agent.
- **The counterfactual choice matters.** WLU stays factored for any clamp, but learnability does not. Wolpert & Tumer show the clamp choice changes performance, and the mean-field aristocrat clamp beat zero-clamping. Candidate defaults here:
  - Producer: no-op.
  - Orderer: the role's hand-written floor, e.g. `round_robin` routing.
  - Or the policy's own mean action, which approximates AU.

  The default should be one whose removal does not trigger system death. Otherwise D_i measures "avoided catastrophe", not marginal contribution.
- **Evidence at large N is mostly from simple utilities.** The 1,000-10,000-agent results come from congestion, Gaussian-squeeze and sensing domains, where G is cheap to re-evaluate in closed form. They are not evidence that forking a full simulator scales the same way.

### Gaps
- Exact per-domain numbers and agent counts for the AAMAS 2004 / GECCO 2004 multi-rover papers and the JAAMAS 2010 air-traffic journal. The PDFs could not be extracted.
- No source found that reports wall-clock cost of difference rewards computed by full-simulator re-runs as a function of N.

## 2. COMA counterfactual baseline, and variants that compute the counterfactual by real reward or simulation (Dr.Reinforce)

### Takeaway
COMA computes an AU-style baseline: it marginalises agent i's action under its policy while the others stay fixed. The baseline is read from a learned centralised Q critic, which is exactly what the user wants to avoid. Dr.Reinforce takes the same aristocrat difference directly on the true reward function. That needs reward or simulator access and costs about |A_i| reward evaluations per agent per step. Dr.Reinforce beat COMA on multi-rover and predator-prey, but its advantage over baselines shrank as N grew from 3 to 8.

### Cited Findings
- **COMA** (Foerster, Farquhar, Afouras, Nardelli, Whiteson; arXiv 1705.08926, May 2017; AAAI 2018). A centralised critic estimates Q, and decentralised actors are trained against a counterfactual baseline that "marginalises out a single agent's action, while keeping the other agents' actions fixed". The critic's representation computes this baseline in one forward pass. On decentralised StarCraft micromanagement, COMA significantly improves average performance over other multi-agent actor-critic methods, and its best agents are competitive with centralised full-state controllers. — [arXiv 1705.08926](https://arxiv.org/abs/1705.08926)
- **Dr.Reinforce** (Castellini, Devlin, Oliehoek, Savani). arXiv 2012.11258, Dec 2020. Extended abstract at AAMAS 2021; full version in Neural Computing and Applications 2022 (DOI 10.1007/s00521-022-07960-5), ALA'21 best paper. It combines REINFORCE with the aristocrat difference reward computed on the *known* reward function. Differencing the reward directly avoids the difficulty of learning COMA's Q-function. — [arXiv 2012.11258](https://arxiv.org/abs/2012.11258v2); [Oliehoek publication page](https://www.fransoliehoek.net/pub-html/htmlfiles/b2hd-Castellini22NCA.html)
- **Cost: linear in the local action set.** The policy-weighted term costs a number of evaluations "linear in the size of the local action set", about |A_i| per agent per step instead of the exponential joint-action count. Dr.Reinforce needs the true R, so it needs reward access or a resettable simulator. — [arXiv 2012.11258v2 (HTML)](https://arxiv.org/html/2012.11258v2)
- **Experiments:**
  - Multi-Rover (10×10 grid, N ∈ {3, 5, 8}) and Predator-Prey (10×10, N ∈ {3, 5, 8}), 30 runs each.
  - SMAC "Very Hard" maps, 10 runs. Plain Dr.Reinforce could not run on SMAC because rewards for all alternative actions were not obtainable, so only the learned-reward variant ran there.
  - Baselines: independent REINFORCE, CentralQ, COMA, and Colby et al.'s local approximation. QMIX and IAC were not compared.
  - Results are reported as curves, without tabulated returns.
  - On Multi-Rover, Dr.Reinforce and Dr.ReinforceR beat all baselines, with the gap largest at N = 3 and shrinking as N grows. The authors attribute this to crowding.
  - On Predator-Prey, COMA performs "extremely poorly".
  - On SMAC, Dr.ReinforceR is "almost never underperforming", with significant gains over COMA on 3s5z, 1c3s5z and MMM. COMA was better on so_many_baneling.

  — [arXiv 2012.11258v2 (HTML)](https://arxiv.org/html/2012.11258v2)
- **Dr.ReinforceR** learns a centralised reward network R_ψ(s, a) by MSE regression, with no bootstrapping and no moving target, and plugs it into the same policy-weighted difference. It generally matches Dr.Reinforce on Multi-Rover. Its prediction error is lower than COMA's critic error. Masking noise hurts more than symmetric noise. — [arXiv 2012.11258v2 (HTML)](https://arxiv.org/html/2012.11258v2)
- **Stated limitations:**
  - Under partial observability the baseline is generally not unbiased, and the method has no convergence guarantee.
  - Performance "could be unstable or arbitrarily bad".
  - The reward network's input grows with the number of agents.
  - Difference rewards need accurate per-action reward estimates.

  — [arXiv 2012.11258v2 (HTML)](https://arxiv.org/html/2012.11258v2)

### Inferences
- **Dr.Reinforce with a simulator is the closest published match to the user's setup.** The variant is critic-free, uses the true reward, and gets the counterfactual by evaluation. It is one-step, though: it differences the *immediate* reward R(s,a). In a tokenomics cycle, a producer's restoration or investment and an orderer's routing affect units in later cycles. So a one-step reward difference is biased toward myopic credit unless each counterfactual is rolled forward H cycles.
- **The action space breaks the "|A_i| evaluations" count.** Here the actions are continuous or simplex (token splits, capacity decisions), so the count does not apply directly. The policy-expectation term has to be Monte Carlo sampled with K draws per agent, giving N·K forks per cycle.
- **The shrinking gap is a warning, not a proof.** Dr.Reinforce's gap over baselines narrows by N = 8 in crowded multi-rover. This is the only direct evidence found on how simulated difference rewards scale with N in a policy-gradient setting, and it is small-N.

### Gaps
- Dr.Reinforce does not report returns as numbers in the body text, and no wall-clock or evaluation-count figures were found.
- Only the abstract was found for **CCPO** ("Counterfactual Credit Policy Optimization"), which builds dynamic baselines by simulating with an agent's contribution removed. No arXiv ID, venue or numbers were verified. — [catalyzex listing](https://www.catalyzex.com/paper/counterfactual-credit-policy-optimization-for)

## 3. Shapley-value credit (Shapley Q-value / SQDDPG, Shapley Counterfactual Credits, SHAQ) and its cost

### Takeaway
Shapley credit averages an agent's marginal contribution over all coalitions. Exact computation is factorial in N, so every published method uses Monte Carlo sampling of coalitions or permutations. The published methods all evaluate coalitions through a learned critic, not by simulation. Done by forking instead, each sampled permutation costs N coalition evaluations, so M permutations cost M·N forks per credited decision. That is at least as expensive as N difference rewards, with an extra factor M.

### Cited Findings
- **Shapley Q-value / SQDDPG** (Wang, Zhang, Kim, Gu; arXiv 1907.05707, July 2019; AAAI 2020). The paper frames global-reward games as an "extended convex game" and proposes Shapley Q-value, which distributes the global reward to reflect each agent's contribution. It is used as the critic in SQDDPG. On Cooperative Navigation, Prey-and-Predator and Traffic Junction, it converges faster than MADDPG, COMA and independent DDPG/A2C. — [arXiv 1907.05707](https://arxiv.org/abs/1907.05707). The abstract page does not state the sampling scheme or sample count. Separately, the Markov Shapley value line lists SHAQ, SQDDPG and SMFPPO as algorithms built on it. — [search summary of SHAQ, Wang et al., NeurIPS](https://arxiv.org/pdf/2106.00285)
- **Shapley Counterfactual Credits** (Li et al., KDD 2021, arXiv 2106.00285). Exact Shapley computation grows factorially with the number of agents. The paper uses Monte Carlo sampling to cut the growth "from factorial to polynomial", includes an ablation on the number of MC samples, and reports beating existing cooperative MARL methods on SMAC, with larger margins on harder maps. — [arXiv 2106.00285](https://arxiv.org/pdf/2106.00285); [emergentmind abstract](https://www.emergentmind.com/papers/2106.00285.md). The full text was not extracted, so the sample count M, the counterfactual mechanism and the exact numbers are unverified.
- **Additive decomposition by regression.** COSAC (Deshmukh, Subramanian, Addanki, Vlassis; arXiv 2604.17693, v2 May 2026) avoids coalition enumeration entirely. It fits an additive per-agent decomposition of the team reward with one ridge regression over the rollout batch, as a critic-free Sequential Aristocrat Utility estimator. — [arXiv 2604.17693](https://arxiv.org/html/2604.17693)

### Inferences
- **Forked Shapley is likely too expensive at this scale.** With 54-92 agents, simulator-evaluated Shapley at M = 10 permutations means about 540-920 forks per credited cycle, and more if each fork must be rolled forward H cycles. The extra benefit of Shapley over a difference reward is accounting for interaction and synergy, such as producers whose capacities are complements in a recipe. That benefit is plausible here, but no source quantifies it against D_i.
- **Two cheaper ways to get Shapley-like information:**
  - Group-level Shapley over a few coarse groups, e.g. producers by capacity type or orderers by consumer cluster.
  - COSAC-style regression decomposition fitted on ordinary on-policy rollouts.

### Gaps
- Sample counts, ablation curves and wall-clock costs for SQDDPG, SCC and SHAQ were not extracted from the full texts.
- No source was found that computes Shapley credit by actual simulator rollouts in a MARL training loop.

## 4. Credit assignment via forked or replayed simulator states: cost vs N, and grouping

### Takeaway
The cleanest recent evidence is C3 (2026). It resets to a recorded prefix, injects alternative actions, rolls out to the terminal reward, and uses a leave-one-out baseline among the alternatives. It needs no critic and gets an advantage ranking far better than a critic trained on the same rollouts. Its signal reliability, however, collapses for long agent chains, and its cost grows with continuation count. COSAC is the main critic-free alternative that avoids extra environment calls, at the price of an additive approximation. Neither paper tests the user's regime: many simultaneous agents over a long horizon in a stateful simulator.

### Cited Findings
- **C3** (Chen et al.; arXiv 2603.06859, v4 Oct 2026, "The Trace Is the State: Exact Credit Assignment for LLM Agent Teams"):
  - **Method.** At each decision point it samples n alternative actions, resets to the recorded prefix, injects each alternative and rolls the downstream agents to the terminal reward. The advantage is the alternative's mean return minus the rollout-weighted mean of the other alternatives. No learned critic.
  - **Exactness condition.** Exact only if "the trace is the state": every causally relevant input is recorded or resampled. Hidden state voids the guarantee.
  - **Training budget.** 8 rollouts per instance, arranged as 2 alternatives × 4 continuations. It writes 10 messages per instance against 16 for 8 independent runs.
  - **Ranking agreement.** On the 2-agent chain, C3 with 4 continuations reaches 0.69 rank correlation with a 16-continuation reference. Two such references agree at 0.73. A decision-value critic trained on the same rollouts reaches 0.29.
  - **Training results** (Qwen3-4B, 5 seeds), MATH500: C3 82.8 vs MAGRPO 74.5 vs MAPPO 69.3 (Holm-corrected p < 0.001 vs MAGRPO). C3 also used 37% fewer training tokens than MAPPO in the long-generation setting.
  - **Variance.** Across 2-10-agent workflows, per-advantage noise is 0.90-1.07× the derived law, which has no agent-count term.
  - **Reliability.** Split-half reliability falls from 0.64 (2-agent chain) to 0.01 (10-agent chain). The authors attribute this to shrinking signal variance, not rising noise.
  - **Agent ablation.** Removing an agent shifts the downstream observation distribution by a bias that more rollouts cannot shrink.

  — [arXiv 2603.06859](https://arxiv.org/html/2603.06859)
- **COSAC on what replay costs.** C3-style replay costs N·K·L extra reward evaluations per iteration, and its per-iteration cost grows as O(K²) in team size K. COSAC instead uses "fictitious" continuations, which are policy forward passes with no environment calls, plus a ridge-regression additive decomposition. Results:
  - Sequential cooperative bandit, K = 2-16 agents, at K = 16: advantage MSE about 2× lower than C3, 30× lower than MA-GRPO (shared advantage) and 46× lower than HA-GRPO.
  - Regret at K = 10: COSAC 0.102, MA-GRPO 0.125, HA-GRPO 0.253, C3 0.273.
  - At K = 2, the shared-advantage MA-GRPO ties COSAC.
  - ARC with 4 LLM agents: AUC 0.721 vs 0.663 for MA-GRPO.

  Stated limits: the additive surrogate may fail for strongly interaction-dependent rewards; the experiments are one-pass bandits, not stateful MDPs; actions must be discrete. — [arXiv 2604.17693](https://arxiv.org/html/2604.17693)
- **Learned difference-reward models** are the Tumer group's answer for when G is costly to re-evaluate (Proper & Tumer 2012), and CLEAN rewards scale to 1,000 agents (section 1). — [Proper & Tumer, AAMAS 2012](https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-proper_aamas12.html)

### Inferences
- **Cost model for this simulator.** Let N be the number of agents, K the alternatives per agent, H the fork horizon in cycles and T the number of cycles credited per episode. Per episode:
  - per-agent difference rewards by forking cost about T·N·K·H extra simulator cycles;
  - group-level counterfactuals over G groups cost T·G·K·H.

  Worked example: N = 92, K = 2, H = 5, T = 100 gives about 92,000 extra cycles per episode, roughly 920× a plain 100-cycle episode. Grouping into G = 8 groups gives about 8,000, or 80×. Crediting only a random subset of S cycles per episode cuts it by a further factor S/T. These are arithmetic estimates, not measured.
- **Shared policies cut the need for N per-agent counterfactuals.** With one GNN per role, the policy-gradient update pools every agent's (observation, action, advantage) samples. An unbiased per-agent advantage on a *random subset* of agents and cycles may therefore suffice, and the cost then scales with the subset size, not with N. This is inferred from how parameter sharing pools gradients; no source tests it.
- **Forking must copy all state.** C3's "trace is the state" condition maps onto the forkable seeded simulator. The fork must copy all causally relevant state: RNG streams, pending shocks, order TTLs and inventories. The project has already seen a routing-tracer closure shared through `deepcopy` (DIAGNOSTICS 22), which is the kind of hidden state that breaks exact counterfactuals.
- **Agent ablation is not a neutral counterfactual.** C3's finding that ablation adds a bias more rollouts cannot remove argues for a policy-sampled or feasible default action (aristocrat form) over removing the agent.
- **Deep coupling may drown the per-agent signal.** C3's reliability collapse over 10-agent chains suggests that, under deep causal coupling, the per-agent signal can be tiny next to the team signal. Multi-cycle producer→orderer→consumer chains here may behave the same way, making short H and immediate-ish credit attractive.

### Gaps
- No published work found that does forked-simulator credit for more than about 16 simultaneous agents in a stateful, long-horizon MDP. All replay or fork evidence above is from LLM pipelines or bandits.
- No published grouping study (counterfactual over agent groups with varying group count) with numbers was found.

## 5. When is the shared team reward alone enough? (MAPPO, IPPO)

### Takeaway
MAPPO and IPPO both train on the shared team reward without explicit per-agent credit and are strong on SMAC, MPE, GRF and Hanabi. Both still use a learned value function as a baseline: MAPPO's is centralised, IPPO's is local. So they do not show that a critic-free shared return is enough. Shared-reward evidence also comes from small teams. The largest SMAC map, 27m_vs_30m, has 27 agents; this map size is from background knowledge and is not sourced here. COSAC's bandit result fits this: the shared advantage ties per-agent credit at K = 2 but falls behind as K grows.

### Cited Findings
- **MAPPO** (Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu; arXiv 2103.01955, Mar 2021; NeurIPS 2022 Datasets & Benchmarks):
  - Shared reward: "both MAPPO and IPPO operate in settings where agents share a common reward".
  - SMAC: matched or beat RODE on 10 of 14 maps with equal samples.
  - GRF 3v1: MAPPO 88.03% vs QMIX 8.12%.
  - Hanabi, 5 players: MAPPO 23.04, IPPO 20.75, SAD 22.06, VDN 21.28.
  - Value normalisation "never hurts training and often improves the final performance", and is critical in Spread.
  - In Hanabi, MAPPO's margin over IPPO widens with more players, suggesting "a centralized critic input can be crucial".
  - The paper does not analyse credit assignment or test per-agent credit methods.

  — [arXiv 2103.01955v4](https://arxiv.org/html/2103.01955v4); [abstract](https://arxiv.org/abs/2103.01955)
- **IPPO** (Schroeder de Witt et al.; arXiv 2011.09533, Nov 2020). Independent PPO, where each agent estimates only a local value function, performs as well as or better than state-of-the-art joint-learning methods on SMAC with little tuning. The authors suggest this may come from robustness to some forms of environment non-stationarity. — [arXiv 2011.09533](https://arxiv.org/abs/2011.09533)
- **Shared advantage vs per-agent credit, critic-free.** MA-GRPO (shared, critic-free) ties COSAC at K = 2 agents and trails at K = 10 (regret 0.125 vs 0.102) and K = 16 (advantage MSE 30× higher). — [arXiv 2604.17693](https://arxiv.org/html/2604.17693). In LLM math pipelines, C3 beats MAGRPO (shared advantage) by +8.28 MATH500. — [arXiv 2603.06859](https://arxiv.org/html/2603.06859)
- **Shared reward vs difference rewards at large N.** Tumer-group domains with 100s-1,000s of agents report the global (team-game) reward performing poorly against difference rewards (sections 1 and 2). — [JAAMAS 2008 page](https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-agogino_jaamas08.html); [AAMAS 2007](https://www.ifaamas.org/Proceedings/aamas07/html/pdf/AAMAS07_0453_bd798e682c64b239665c2e4d0efcd1df.pdf)

### Inferences
- **The literature splits by team size.**
  - With small teams (tens of agents at most) and a learned baseline, the shared reward is usually enough (MAPPO, IPPO).
  - With many agents and a shared scalar, the signal-to-noise per agent falls roughly as 1/N in the agent's own contribution to G. Difference rewards were designed for this regime, and the Tumer group's large-N results show it.

  This setup, with 54-92 agents sharing a units-per-cycle scalar, sits in the second regime. Weight-sharing per role partly offsets it, because gradients pool across agents.
- **Cheap discriminating check before building per-agent forks.** Measure the actual variance of the policy-gradient estimate:
  - (a) shared per-cycle team return (critic-free, batch-mean baseline);
  - (b) forked D_i on a sampled subset of agents and cycles.

  Compare gradient signal-to-noise (or the C3-style split-half reliability of the advantages) at 12 vs 42 producers. If (a)'s reliability at 42 is already similar to (b), shared reward suffices. If it collapses, per-agent forked credit is justified. This follows the project's root-cause-first rule.

### Gaps
- No source found that tests a critic-free shared-return policy gradient (REINFORCE/GRPO-style) against difference rewards at 50-100 agents in a stateful simulator.
- MAPPO's SMAC success comes with a centralised critic, and its benefit cannot be separated from credit assignment using the sources here.
