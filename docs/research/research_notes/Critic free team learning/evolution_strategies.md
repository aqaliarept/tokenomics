# Evolution strategies and neuroevolution as critic-free learners for shared multi-agent / GNN policies on CPU

Scope: can ES (OpenAI-ES, NES, CMA-family, PGPE, GA) train one shared GNN policy (thousands to tens of thousands of parameters) for a cooperative team of 12-42 producers + ~50 orderers, 100-cycle episodes, deterministic seeded simulator, CPU pods, ~5 s per episode at 12 producers. Research date 2026-11-06.

Method note: arXiv PDFs could not be text-extracted in this session (no PDF tooling), so some well-known paper details were confirmed only through abstracts, ar5iv HTML, or search summaries that quote the paper. Details recalled but not re-verified are listed under each section's Gaps, not as findings.

## 1. OpenAI ES (Salimans et al. 2017): scale, data efficiency, variance-reduction tricks

### Takeaway
OpenAI-ES matched TRPO on MuJoCo while using up to about 10x more environment data, and was up to 3x better on simple tasks. It won on wall-clock only by spending a great deal of hardware: 3D Humanoid in under 10 minutes on 1,440 CPU workers, against about 11 hours on one 18-core machine. Its engineering tricks are antithetic (mirrored) sampling and shared random seeds, so workers exchange only scalar returns. ES's advantage is parallelism and indifference to horizon and credit assignment. It is not sample efficiency.

### Cited Findings
- Abstract claim: ES matches TRPO's MuJoCo performance using "no more than 10x" as much data. The body reports up to 3x *better* sample complexity than TRPO on simple environments, with a larger penalty on Hopper and Walker2d. The common "3-10x" summary comes from slides and secondary write-ups, not from the abstract's wording. — [Salimans et al. 2017, arXiv 1703.03864](https://arxiv.org/abs/1703.03864); [Salimans, ES slides (SC for DL 2017)](https://supercomputersfordl2017.github.io/Presentations/Salimans_ES.pdf)
- 3D Humanoid solved in under 10 minutes with 1,440 workers (OpenAI's blog: 1,440 CPUs across 80 machines). The same task takes about 11 hours with ES on one 18-core machine, "on par with RL". — [arXiv 1703.03864](https://arxiv.org/abs/1703.03864); [OpenAI blog](https://openai.com/index/evolution-strategies/)
- Variance reduction: the paper says "To reduce variance, we use antithetic sampling" (mirrored ±ε pairs). Workers share random seeds and exchange only scalar returns, so each worker rebuilds every perturbation locally. This communication scheme scales to more than a thousand workers. — [arXiv 1703.03864](https://arxiv.org/pdf/1703.03864) (via search summary quoting the paper)
- Later critique: Augmented Random Search (ARS) uses linear policies and random search over parameters (the ES family). It reports matching state-of-the-art MuJoCo sample efficiency and being "at least 15 times more efficient" computationally than the fastest competing model-free methods. Its table lists ES at 10 min on 1,440 cores, against ARS figures on 18 cores (657 min is quoted for 18 cores). ARS also found high variability across "hundreds of random seeds" and argues that the usual sample-efficiency estimates mislead. — [Mania, Guy, Recht 2018, arXiv 1803.07055](https://arxiv.org/abs/1803.07055)
- 2026 replication-style study: ES "do not consistently train faster than DRL" on Flappy Bird, Breakout and MuJoCo. Used as a pretraining stage for DRL, ES helped only in the simplest environment. This is a course project (EE-568), so its evidence is weak. — [Martínez et al. 2026, arXiv 2604.00066](https://arxiv.org/abs/2604.00066)

### Inferences
- The "3-10x more data than PPO/TRPO" figure is the right prior for *single-agent* dense-control tasks. In a team setting the gap is likely *larger*. An on-policy actor-critic gets one training sample per agent-decision: about 62 agents × 100 cycles ≈ 6,200 decisions per episode here, with per-agent advantages from a critic. ES gets one scalar fitness per episode for the whole shared parameter vector. Parameter sharing keeps the dimension d small but does not give ES more signal per episode.
- What ES offers this project is not efficiency. It is (a) no critic, so none of the critic-fitting failure modes, (b) indifference to the 100-cycle horizon and delayed outcomes, (c) exact use of determinism (see §5), and (d) embarrassingly parallel episodes with one process per core, which fits the repo's "fan out independent runs" rule.

### Gaps
- Not re-verified this session (PDF not extractable); recalled from the paper, check before citing: MuJoCo policy = MLP with 2 hidden layers of 64 tanh units; virtual batch normalization used for Atari only (for policies sensitive to the perturbation); fitness shaping = centered rank transform; L2 weight decay; σ = 0.02; Atari network = the DQN architecture; Atari data comparison against A3C with 1 hour on 720 CPUs vs about 1 day for A3C on 32 cores; the claimed invariance to action frequency (frame-skip) and to long horizons.
- The exact population (perturbations per update) used for the Humanoid and Atari runs was not confirmed.

## 2. Scaling ES: GA to millions of parameters, NES / CMA limits, PGPE, low-rank ES, libraries

### Takeaway
Gradient-free search scales to millions of parameters (a GA trained 4M+ Atari parameters) if you pay with large populations: 1,000 to 20,000 per generation and about 1B frames. Full-covariance CMA-ES is infeasible beyond a few thousand dimensions. For a policy of a few thousand to tens of thousands of parameters, the practical choices are diagonal or isotropic search: OpenAI-ES / PGPE / SNES, sep-CMA, LM-MA-ES (O(n log n)), or the 2025 low-rank EGGROLL for matrix-heavy nets. Accelerator libraries (EvoJAX, evosax) give large speedups only when the simulator itself runs in JAX, which this Python simulator does not.

### Cited Findings
- Deep GA (Such et al. 2017): Atari network "contains over 4M parameters"; population 1,000 (+1 elite), truncation 20, mutation σ = 0.002; 1B frames (also run to 6B); about 4 hours on one desktop (4 GPUs, 48 CPU cores), about 1 hour on 720 CPU cores, against about 7-10 days (DQN) and about 4 days (A3C). At 1B frames the GA beats ES, A3C and DQN on 6 of 13 games each. Random search beats DQN on 3 games, ES on 3 and A3C on 6. — [Such et al. 2017, arXiv 1712.06567](https://ar5iv.labs.arxiv.org/html/1712.06567)
- Same paper, Humanoid (about 167k parameters): population 12,500, and the GA needed about 1,500 generations against about 100 for ES. It took "∼15 times longer to perform slightly worse than ES". The GA is not uniformly better than ES on continuous control. — [arXiv 1712.06567](https://ar5iv.labs.arxiv.org/html/1712.06567)
- Seed-chain encoding compresses final Atari networks "8,000-50,000-fold" (28-bit seeds). This is the same idea as ES's shared-seed communication. — [arXiv 1712.06567](https://ar5iv.labs.arxiv.org/html/1712.06567)
- GA-based novelty search solved Image Hard Maze (4M+ parameters, population 20,000), where plain GA failed. — [arXiv 1712.06567](https://ar5iv.labs.arxiv.org/html/1712.06567)
- LM-MA-ES reduces MA-ES/CMA-ES's O(n²) time and storage to O(n log n) for "efficient zeroth order large-scale optimization", with "state-of-the-art performance on a set of established large-scale benchmarks". — [Loshchilov, Glasmachers, Beyer 2017, arXiv 1705.06693](https://arxiv.org/abs/1705.06693)
- In a 50-agent ES study, full second-order GASS and CMA-ES were *not run* because of computational cost at 70k+ parameters. — [Fan, Theodorou, Reeder 2018, arXiv 1803.01106](https://arxiv.org/html/1803.01106v2)
- EGGROLL ("Evolution Strategies at the Hyperscale", Nov 2025): low-rank perturbations ABᵀ (rank r ≪ min(m,n)) cut per-layer storage from mn to r(m+n). The low-rank update converges to the full-rank one at O(1/r), and the aggregate update has rank min(Nr, m, n). Claimed to be competitive with GRPO for LLM reasoning. — [Sarkar et al. 2025, arXiv 2511.16652](https://arxiv.org/pdf/2511.16652v1)
- Critique (2026): "EGGROLL, Unrolled" argues that rank-one perturbations lie in a zero-volume subset of the matrix space and can make the mean field nonconservative and reverse an optimum's local stability. It proposes LOO-ROLL, one evaluation per direction instead of two antithetic ones, which halves estimator MSE at equal evaluation cost in transformer blocks. — [Kaya & Hashemi 2026, arXiv 2609.10980](https://arxiv.org/pdf/2609.10980)
- evosax (2022): JAX library of "30 evolutionary optimization algorithms" (finite-difference ES, estimation-of-distribution ES, GAs) with an ask-evaluate-tell API and JIT, vectorization and multi-device parallelism. — [Lange 2022, arXiv 2212.04180](https://arxiv.org/abs/2212.04180)
- EvoJAX (GECCO 2022): solves most of its tasks "within minutes on a single accelerator" against "hours or days when using CPUs". The speedup needs the task itself compiled in JAX. — [Tang, Tian, Ha 2022, arXiv 2202.05008](https://arxiv.org/abs/2202.05008)

### Inferences
- At d ≈ 2k-30k, full CMA-ES (O(d²) memory, about 10⁸-10⁹ covariance entries at the top of the range) is impractical. sep-CMA, LM-MA-ES, SNES/PGPE or plain OpenAI-ES with Adam are the realistic options. EGGROLL matters only for much larger matrix-heavy models.
- EvoJAX / evosax speedups do not transfer, because the simulator is sequential Python (domain/ is pure Python, no numpy). On this project the ES math is negligible next to episode cost: the evaluator is the bottleneck. evosax could still serve as the optimizer library through ask/tell with an external CPU evaluator.
- The GA results show that gradient-free search on millions of parameters needs populations of 10³-10⁴ and about 10⁹ frames. None of the cited successes ran on a budget of the order of 10²-10³ vCPU-h with 5 s episodes.

### Gaps
- Not re-verified this session: NES (Wierstra et al., JMLR 2014) natural-gradient formulation; sep-CMA-ES (Ros & Hansen 2008) diagonal covariance and linear time; PGPE (Sehnke et al. 2010) symmetric sampling and baseline; Guided ES (Maheswaranathan et al. 2019, arXiv 1806.10230), which biases sampling toward a surrogate-gradient subspace; the Nesterov-Spokoiny result that Gaussian-smoothing gradient-estimator variance grows linearly with dimension d. All are standard, but their numbers were not confirmed here.
- No published evosax / EvoJAX benchmark tables (speedup factors per task) were retrieved.

## 3. ES and evolutionary hybrids in multi-agent RL

### Takeaway
Pure ES on a shared team policy has been done: a 50-UAV team with one shared 70k-parameter policy, population 300 per iteration, 2 days on a 244-thread Xeon Phi. But no 2018-2026 paper found in this search benchmarks pure ES head-to-head against MAPPO/IPPO on SMAC, MPE or JaxMARL. Where pure evolutionary MARL appears as a baseline, it is reported as sample-inefficient. The literature mostly uses evolution as a *complement* to gradient MARL: MERL (team reward evolved, agent reward by gradient), ERL / CEM-RL in single-agent, and CMA-MAPPO / EECG in 2025-2026.

### Cited Findings
- Fan et al. 2018: ES framed as a special case of model-based stochastic search (GASS). Cooperative scenario: 50 fixed-wing UAVs attack a base defended by 20 quadcopters. Competitive scenario: team against team. **One shared policy per team** (MLP 102→200→200→50→3, "more than 70,000 parameters"); N = 300 samples per iteration; 200 s episodes at 0.1 s steps (2,000 steps); training took **two days** fully parallel on a Xeon Phi (244 threads). The cross-entropy method was "significantly worse" than ES. — [arXiv 1803.01106](https://arxiv.org/html/1803.01106v2)
- MERL (ICML 2020): many cooperative tasks have a sparse team reward and dense per-agent rewards. MERL evolves a population of *teams* by neuroevolution on the team reward, trains policies by gradient on the agent-specific rewards, and periodically injects the gradient-trained policies into the population. Reported to "significantly outperform" MADDPG on several coordination benchmarks. — [Khadka et al. 2019/2020, arXiv 1906.07315](https://arxiv.org/abs/1906.07315)
- CEM-RL (ICLR 2019): combines the cross-entropy method with TD3, presented as a "satisfactory trade-off between performance and sample efficiency", motivated by deep neuroevolution being "stable but sample-inefficient". — [Pourchot & Sigaud 2018, arXiv 1810.01222](https://arxiv.org/abs/1810.01222)
- ACE-MAPPO (2026 preprint, cooperative air combat) compares against MAPPO, RMAPPO, IPPO and an evolutionary MARL baseline (EMARL). It reports that EMARL's pure evolutionary approach has low sample efficiency, so its learning lags well behind. — [arXiv 2605.25091](https://arxiv.org/html/2605.25091)
- CMA-MAPPO (Swarm and Evolutionary Computation, 2026): CMA-ES hybridized with MAPPO for exploration under sparse rewards. Baselines are MAPPO, IPPO, MAPPO-RND, MERL, pure CMA-ES and differential evolution, and the hybrid is claimed best. — [ScienceDirect S2210650226000507](https://www.sciencedirect.com/science/article/abs/pii/S2210650226000507)
- EECG (AAMAS 2025; Peng, Li, Hao): combines evolutionary algorithms, episodic memory and graph-based coordination for cooperative MARL. Reported to beat graph-based, value-based and evolutionary baselines on SMAC, and also tested in SUMO traffic. Code: MercyM/EECG. — [AAMAS 2025 p1623](https://aamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p1623.pdf); [GitHub](https://github.com/MercyM/EECG)
- Survey statement: evolutionary algorithms "reuse past interaction data less efficiently than many gradient-based methods", hence the hybrids. — [Evolutionary MARL: Advances and Challenges (ResearchGate)](https://www.researchgate.net/publication/414987442_Evolutionary_Multi-Agent_Reinforcement_Learning_Advances_and_Challenges)
- DC-Ada (2026): keeps a pretrained *shared* multi-robot policy frozen and adapts small per-robot observation transforms by gradient-free accept/reject random search on short **common-random-number** rollouts, using only scalar team returns. Deterministic 2D simulator; 200,000 joint env steps per run; 5 seeds. Mixed results: no single mitigation wins everywhere. — [Alqithami 2026, arXiv 2604.03905](https://arxiv.org/abs/2604.03905)
- Project's own ES record (this repo): outcome-trained ES on small residual corrections over frozen hand-written bases.
  - M4 step 2 (orderer residual, EXP-0058 → EXP-0071): learns and generalises (+45 to +70 consumer units on validation) but plateaus at the simple-rule level.
  - On the MWWk base (EXP-0070): +25.8 against a floor of 92.3 after 40 generations, and 542 in 1,000 perturbed members disqualified for real run-down / starvation.
  - M5 producer residual (EXP-0090 → EXP-0097): +2-4% units at 12 producers, `plateau` on the run-down clause, ≈ 65.2 vCPU-h.
  - Source: /srv/research/tokenomics/experiments/HISTORY.md (sections "Does the ES pilot clear its floor? (EXP-0059)", "Step 2 closes", "Learning a producer correction on k5_rf5").

### Inferences
- The closest published analogue to this project's setup (Fan et al.: about 50 agents, one shared policy of about 70k parameters, ES, CPU-only) needed about 2 days on 244 hardware threads. That is roughly 10⁴ thread-hours, an order of magnitude above a 150 vCPU-h tranche. Network size and episode cost differ, so this is an order-of-magnitude anchor, not a prediction.
- The MERL pattern fits this project's reward structure: sparse or noisy team outcomes (consumer units, run-down, deaths) plus dense local signals. Evolve on the team objective; take gradient or imitation steps on local, per-agent targets. That is also how the M6 plan (imitate k5_rf5+MWWk → MAPPO) could be made critic-free: ES fine-tunes an imitation-initialised GNN on team fitness.
- The repo's own ES results show the binding constraint was *fitness noise / disqualification against a noisy floor*, not optimizer capacity. A larger GNN under ES inherits the same signal-to-noise problem with a larger d, so fewer samples per dimension.

### Gaps
- I found no paper that benchmarks pure OpenAI-ES (or CMA-ES) against MAPPO/IPPO on SMAC, MPE, MAgent or JaxMARL with reported sample counts. Searches returned hybrids only. The "Co-evolution as more than a scalable alternative for MARL" OpenReview submission looked relevant but was behind a verification wall (not read).
- ERL (Khadka & Tumer, NeurIPS 2018, arXiv 1805.07917) was not fetched. Recalled, unverified: a GA population plus a DDPG learner sharing a replay buffer, with the RL actor periodically copied into the population. Likewise CEM-RL's mechanism (half of a CEM population takes TD3 critic-gradient steps each iteration) was not confirmed from the abstract.
- Head-to-head numbers for EECG and CMA-MAPPO (win rates, sample counts) were not extracted (PDF / paywall).

## 4. ES / evolution for GNN and combinatorial-optimization policies (routing, scheduling)

### Takeaway
There is a small but real literature training neural combinatorial-optimization policies with evolutionary methods: EA-trained Pointer Networks on TSP, and ES-optimized graph-RL for dynamic and flexible job-shop scheduling. The claim is better solution quality than gradient-trained counterparts at equal compute. All found cases are single-decision-maker CO problems, not multi-agent teams with a shared GNN.

### Cited Findings
- EA-trained Pointer Networks: up to "30.21% improvement in quality of the solution with the same computational time" over gradient-descent-trained PtrNets on TSP. Claimed first PtrNet trained at 1,000-node TSP; outperforms eight methods. Preprint only. — [Zhong et al. 2023/2024, arXiv 2312.01150](https://arxiv.org/abs/2312.01150)
- "Evolution strategies-based optimized graph reinforcement learning for solving dynamic job shop scheduling problem" (Su et al., Applied Soft Computing 2023): combines GNNs and DRL with ES optimization for dynamic JSSP. Only the title and description were retrieved, not how ES is applied. — [researchr / BAAI listing](https://hub.baai.ac.cn/view/29233)
- "Evolutionary Strategies with Dual Graph Reinforcement Learning for Flexible Job Shop Scheduling Problem" (Yang et al. 2025): title only. — [researchr](https://researchr.org/publication/YangWX25-4)
- A search fragment mentions a GNN branching policy (improving Gasse et al.'s GNN) learned with ES in branch-and-bound, but the source paper was not identified. — [search result, unidentified]; treat as a lead only.
- NEORL toolkit implements PPO-ES hybrids (PPO guides GA/ES), with TSP demos. — [arXiv 2112.07057](https://arxiv.org/pdf/2112.07057)

### Inferences
- GNN policies are not special for ES: ES sees only a flat parameter vector, so message-passing weight sharing just keeps d small (a GNN with 2-4 layers of 32-64 hidden units is about 5k-30k parameters). The claimed ES advantage in CO (escaping the local optima of REINFORCE-style training) is plausible for this project's whole-order routing decisions, which are discrete and non-smooth. But the repo's own evidence (§3) is that small ES residuals plateau near the hand-written rules.

### Gaps
- No paper found that trains a *shared multi-agent* GNN policy with pure ES and reports parameter counts, population and wall-clock. The ES-GNN branching paper could not be identified.
- The training details of Su et al. 2023 and Yang et al. 2025 (population, parameters, compute) were not retrieved.

## 5. Common random numbers and baseline-relative fitness

### Takeaway
On a deterministic, seeded simulator, ES should evaluate every perturbation (and both halves of each antithetic pair) on the same scenario seeds. The fitness should be the paired difference to a reference policy on those seeds. This is common random numbers (CRN). It removes scenario variance from the ranking exactly when trajectories stay correlated. Evidence that CRN helps is strong in some noise models and negative in others, so it should be measured, not assumed.

### Cited Findings
- OpenAI-ES's shared-seed scheme is a CRN device for the *perturbations* (every worker regenerates the same ε). Antithetic pairs are a variance-reduction device. — [arXiv 1703.03864](https://arxiv.org/pdf/1703.03864)
- GECCO 2015 (noisy population-based optimization): CRN were "very efficient for some artificial noise models, and detrimental for others". In the unit-commitment application CRN provided "by far, most of the improvement". — [Variance reduction in population-based optimization: application to unit commitment (core.ac.uk)](https://core.ac.uk/download/49487579.pdf); [Lingnan listing](https://scholars.ln.edu.hk/en/publications/variance-reduction-in-population-based-optimization-application-t/)
- DC-Ada (2026) uses "short common-random-number rollouts" for budgeted accept/reject of candidate against incumbent in a deterministic multi-robot simulator. The search summary states the argument: positive covariance between baseline and candidate rollouts makes the paired improvement estimate strictly lower-variance than independent seeds, cutting false accepts and false rejects. — [arXiv 2604.03905](https://arxiv.org/abs/2604.03905)
- Caveat from the same search: paired evaluation equals a CRN estimator whose gain is set by the covariance of the two outcomes. If the policies' trajectories diverge quickly, the reduction shrinks. — [arXiv 2512.24145](https://arxiv.org/pdf/2512.24145v3)
- Project evidence: the repo's ES runs already use a precomputed per-world-episode bar (A0) and paired ranking. Disqualification rules relative to A0 tied most ranks (EXP-0058: 37/48 and 42/48 members disqualified), and the gradient-direction probe read `no_signal` (cosine −0.07) until the base and rules changed (cosine 0.97 on k5_rf5, EXP-0086). — /srv/research/tokenomics/experiments/HISTORY.md

### Inferences
- In this simulator demand and shocks are drawn from the episode seed before play, and the domain is deterministic. So for a fixed seed set, fitness F(θ) is a deterministic function and all residual noise is *scenario sampling*. CRN plus antithetic pairs then makes the per-pair difference F(θ+σε) − F(θ−σε) free of scenario noise *within* the minibatch. What remains is generalisation across seeds, which is why the minibatch of worlds must be resampled each generation (or rotated) to avoid overfitting a fixed set.
- A baseline-relative fitness (candidate − reference on the same world-episode, possibly normalised per world as the repo does with its bar) is a rank-preserving shift within a generation when the bar is shared. It matters only if worlds are mixed in a minibatch with very different scales. Then per-world normalisation, or a rank transform within world, stops the high-unit worlds from dominating.
- Hard thresholds (death / run-down disqualification) are what destroyed ES signal in EXP-0058/0070. Smooth penalties inside the fitness keep the ranking informative. EXP-0096/0097's λ run-down term shows the trade-off: a noisy smooth term is barely seen by the ranks.

### Gaps
- No primary ES-for-RL paper was found that quantifies the CRN gain (variance ratio) for policy-search fitness on a deterministic multi-agent simulator. The GECCO 2015 and DC-Ada sources are the closest. The 2512.24145 source's exact context (title, domain) was not read.

## 6. Episodes per update and CPU wall-clock for a few-thousand-parameter shared GNN here

### Takeaway
Published ES successes used hundreds to thousands of perturbations per update: OpenAI-ES 1,440 workers, Fan et al. 300 per iteration, GA 1,000-20,000. A few-thousand-parameter GNN needs on the order of 10²-10³ episodes per update and 10²-10³ updates. At 5 s per 12-producer episode that is roughly 10²-10³ vCPU-h per run, and several times more at 42 producers. The project's own small-residual ES used population 24 at about 0.36-0.39 vCPU-h per generation. A full shared GNN is 1-2 orders of magnitude more expensive per useful gradient than those runs, unless it starts from imitation (BC) weights and ES only fine-tunes.

### Cited Findings
- Population anchors:
  - OpenAI-ES Humanoid: 1,440 workers. — [arXiv 1703.03864](https://arxiv.org/abs/1703.03864)
  - 50-agent ES with a 70k-parameter shared policy: N = 300 per iteration, 2 days on 244 threads. — [arXiv 1803.01106](https://arxiv.org/html/1803.01106v2)
  - GA: 1,000 (Atari, 4M parameters), 12,500 (Humanoid, about 167k parameters). — [arXiv 1712.06567](https://ar5iv.labs.arxiv.org/html/1712.06567)
- Project anchors (CPU pods, same simulator):
  - EXP-0059: 20 generations at 24 members per generation, ≈ 7.8 vCPU-h on a secure cpu8.
  - EXP-0070: 40 generations, ≈ 14.3 vCPU-h.
  - M5-P producer residual arc: ≈ 65.2 vCPU-h in total.
  - Source: /srv/research/tokenomics/experiments/HISTORY.md
- ARS showed that random search on *linear* policies can be computationally 15x cheaper than competitors on MuJoCo. Small-d policies are where ES/random search competes best. — [arXiv 1803.07055](https://arxiv.org/abs/1803.07055)

### Inferences
These are back-of-envelope estimates, not sourced results. Assumptions: 5 s per episode at 12 producers; about 15-20 s at 42 producers (assumed roughly linear in agent count); one episode per core; no ES-math overhead.

| Setting | Episodes / update | vCPU-h / update (12 P) | Wall per update, 32-vCPU pod | 200 updates (12 P) | 200 updates (42 P, ×3.5) |
|---|---|---|---|---|---|
| Lean: 32 antithetic pairs × 2 CRN seeds | 128 | 0.18 | ≈ 20 s | ≈ 36 vCPU-h, ≈ 1.1 h wall | ≈ 125 vCPU-h |
| Medium: 64 pairs × 4 seeds | 512 | 0.71 | ≈ 80 s | ≈ 142 vCPU-h, ≈ 4.4 h wall | ≈ 500 vCPU-h |
| Fan-et-al-like: 150 pairs × 4 seeds | 1,200 | 1.67 | ≈ 3.1 min | ≈ 333 vCPU-h | ≈ 1,170 vCPU-h |

- Dimension scaling: the ES gradient-estimator variance grows roughly with d / (number of perturbations) (standard Gaussian-smoothing result; see Gaps). At d ≈ 5k with 64 pairs, each update sees about 1% as many directions as there are parameters. So progress per update is slow unless (a) the parameter count actually being searched is cut, or (b) the search starts near a good solution.
  - Ways to cut it: search only a head or residual of a few hundred parameters on top of a frozen BC-trained GNN, as the repo's residual ES did; or use low-rank (EGGROLL-style) or guided-subspace perturbations.
  - Starting near a good solution: imitation-initialised weights.
- On a single 32-vCPU pod, PPO / MAPPO gets about 6,200 agent-decisions of training signal per 5 s episode, against one scalar for ES. For the same ~150 vCPU-h tranche, MAPPO sees about 100k episodes' worth of per-decision samples, while ES gets about 100k scalar fitnesses spread over a d-dimensional search. ES is competitive only if the critic is the failure mode (non-stationary team credit, 100-cycle delayed outcomes, hard constraints like death), which is the stated motivation for "critic-free" here.
- Recommended ES design if pursued, inferred from the sources above:
  - OpenAI-ES with Adam, antithetic pairs, a centered-rank fitness, and CRN over a per-generation resampled world-seed minibatch.
  - Fitness = paired difference to the frozen base (BC GNN or k5_rf5+MWWk) per world-episode, normalised per world, with smooth penalties instead of hard disqualification.
  - Searched dimension kept at ≤ 1-2k: a head, layer-norm gains or a low-rank adapter on the BC GNN.
  - Population of about 32-64 pairs × 2-4 seeds; one process per core, BLAS pinned to 1 thread.

### Gaps
- No primary source gives a population-size rule for a given d in RL ES. The d/n variance scaling (Nesterov & Spokoiny 2017; Berahas et al. 2019 on finite-difference vs Gaussian-smoothing gradients) was not verified this session.
- The episode cost at 42 producers (assumed 15-20 s) is not measured. EXP-0055's deployed-path cost slopes (/srv/research/tokenomics/experiments/, PoC-B) should replace the ×3.5 assumption.
- No source was found that reports ES vs PPO sample counts on a *multi-agent shared-policy* task. The 3-10x single-agent ratio is the only published number, and it likely understates the gap for 60+ agents with per-agent credit.
