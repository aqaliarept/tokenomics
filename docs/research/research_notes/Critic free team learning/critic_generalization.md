# Centralized critics / value functions: generalization across topologies and sizes (GNN critics in MARL), failure modes, mitigations

Context for the writer. The project plans a GNN-based centralized critic (CTDE, MAPPO-style) trained on about 30 supply-network topologies with 12 to 42 producers. Two past attempts failed. A regression V(global_state) to units-to-go, trained on 6 economies, gave negative held-out R^2. An earlier PPO critic stalled near 0. These notes collect primary-source evidence on whether such a critic can be expected to give useful advantages.

Method note. Numbers below come from full-text extraction of the arXiv PDFs (Procgen, IDAAC, PPG, MAPPO, IPPO, Papoudakis benchmark, PIC, UPDeT, MAT, PopArt-multitask, Observational Overfitting, Gammelli ICML 2023, HDPO, POMO, Kool), unless the item says "abstract" or "search snippet". Accessed 2026-11-06.

## Q0 (objective): Can a GNN centralized critic trained on ~30 topologies (12–42 producers) be expected to give useful advantages?

### Takeaway
Not by default. The literature shows that value functions generalize worse than policies across instances. A value function must encode instance-specific facts, such as remaining steps, total scale and topology-specific bottlenecks, that the policy does not need. Robust reports of GNN or transformer *critics* transferring zero-shot to unseen graphs or sizes are rare. The transfer successes are almost all *policy* transfer (shared local operators). Thirty training topologies is far below the instance counts where Procgen agents stop overfitting (about 10,000 levels). With per-instance value normalization, a decoupled critic and the critic used only at train time on the training topologies, a critic can still reduce variance *on the training set*. The cheaper and better-evidenced alternative is a per-instance, sampled baseline (POMO/GRPO-style shared baseline, greedy-rollout baseline).

### Cited Findings
- On Procgen, PPO agents "strongly overfit to small training sets in almost all cases. To close the generalization gap, agents need access to as many as 10,000 levels". Training sets were 100 to 100,000 levels, with 200M steps each — [Cobbe et al. 2020, Procgen (ICML 2020)](https://arxiv.org/abs/1912.01588)
- IDAAC's central claim is that accurate value prediction needs more information than the optimal policy needs. In a Ninja example, two semantically identical initial observations have different true values (reward discounted over 24 vs 50 steps), so "the agent must memorize the number of remaining steps in that particular level", using spurious features like background colour — [Raileanu & Fergus 2021, DAAC/IDAAC (ICML 2021)](https://arxiv.org/abs/2102.10330)
- In the same paper, six PPO agents were trained on 200 to 10,000 levels. Agents trained on more levels generalize better *and* have **higher** end-of-training value loss, so value loss correlates *positively* with test performance (Maze r = 0.96). On one Ninja level, PPO-200 has value loss 0.2 and test score 5.9, PPO-10k has value loss 0.3 and test score 8.8, and decoupled DAAC-200 has value loss 0.2 and test score 7.3 — [Raileanu & Fergus 2021](https://arxiv.org/abs/2102.10330)
- Interference between the policy and value objectives hurts when parameters are shared. Value optimization "tolerates a significantly higher level of sample reuse than policy optimization" — [Cobbe et al. 2021, Phasic Policy Gradient (ICML 2021)](https://arxiv.org/abs/2009.04416)

### Inferences
- Units-to-go is exactly the kind of quantity IDAAC describes: it depends on horizon remaining, the size of the economy (12 vs 42 producers), demand scale and the topology's bottleneck capacity. A critic that fits it on 6 or 30 training economies is incentivized to memorize economy-identifying features. Negative held-out R^2 is the expected outcome, not an anomaly.
- The useful question is not "does the critic generalize to held-out topologies?" but "does the critic reduce advantage variance on the topologies it is trained on, without leaking topology-specific features into the actor?". The critic is discarded at execution in CTDE, so held-out critic R^2 matters only if training itself covers new topologies (online curriculum over new worlds).

### Gaps
- I found no paper that reports held-out-topology R^2, or explained variance, of a centralized critic in a supply-chain or production-network MARL setting. The negative held-out R^2 cannot be benchmarked against a published number.

## Q1: Evidence on value-function overfitting vs policy generalization

### Takeaway
Single-agent generalization work shows asymmetric overfitting. Value heads latch onto instance-specific features, and sharing those features with the policy damages policy generalization. Decoupling, and training the policy representation on advantages rather than values, gives the best Procgen test scores of its time. Multi-agent evidence that independent (local-value) learners match centralized-critic learners in many tasks is consistent with this picture.

### Cited Findings
- Procgen easy mode (200 train levels, 25M steps), PPO-normalized scores in Train / Test form: PPG 144.5 / 152.2, DAAC 131.0 / 162.3, IDAAC 132.2 / 163.7, UCB-DrAC 118.9 / 139.7, PLR 106.7 / 128.3. The decoupled methods trade a little train score for the best test score and the smallest generalization gap (10 seeds) — [Raileanu & Fergus 2021, Table 1](https://arxiv.org/abs/2102.10330)
- DAAC trains the policy network with an auxiliary head that predicts GAE advantages, with a *separate* value network. "The advantage is a relative measure of an action's value while the value is an absolute measure… the advantage can be expected to vary less with the number of remaining steps". IDAAC adds an adversarial loss so that the policy encoder cannot tell which of two states came first, which removes remaining-steps information. That loss applies to the policy network only — [Raileanu & Fergus 2021](https://arxiv.org/abs/2102.10330)
- DAAC's value network still shows the episode-step-linear (memorizing) trend, "but… this does not negatively affect the policy since DAAC uses separate networks". The value head can memorize as long as it does not share features with the policy — [Raileanu & Fergus 2021, Sec. 5.5](https://arxiv.org/abs/2102.10330)
- PPG uses disjoint policy and value networks plus periodic distillation. Its key observations are (1) policy/value interference when parameters are shared and (2) a higher tolerable sample reuse for the value function — [Cobbe et al. 2021](https://arxiv.org/abs/2009.04416)
- "Observational overfitting": agents correlate reward with spurious observation features even when the MDP dynamics are fixed, and implicit regularization of the architecture matters for RL generalization — [Song et al. 2020, Observational Overfitting (ICLR 2020)](https://arxiv.org/abs/1912.02975)
- Independent PPO, with each agent estimating a *local* value function, "can perform just as well as or better than state-of-the-art joint learning approaches" on SMAC, possibly because it is robust to some forms of non-stationarity — [de Witt et al. 2020, IPPO](https://arxiv.org/abs/2011.09533)
- Across 25 tasks, centralized critics (MAA2C/MAPPO vs IA2C/IPPO) help mainly when agents' observations "lack important information about other agents"; independent learning "is competitive… in fully-observable tasks of MPE and LBF". COMA, MADDPG, VDN and QMIX "are unable to learn effective behaviour" in sparse-reward RWARE. The hypothesis is that large joint-input critics need sufficient training signal — [Papoudakis et al. 2021, EPyMARL benchmark (NeurIPS D&B)](https://arxiv.org/abs/2006.07869)

### Inferences
- The project's critic regressing V(global_state), trained on few economies, fits the "value needs instance-specific information" pattern. An actor that shares a trunk with that critic risks inheriting topology-memorizing features, which is the PPG/DAAC interference.
- The Papoudakis result suggests a centralized critic pays off only when local observations miss decision-relevant global information. If producers' local observations plus team reward already carry most of the signal, the critic's added value is small and its fitting cost is high, especially with sparse or delayed signals like units-to-go.

### Gaps
- No primary multi-agent study directly replicates IDAAC's value-loss-vs-generalization correlation for centralized critics across topologies. The analogy is inferred.
- Cobbe et al. 2019 ("Quantifying generalization in RL", CoinRun) was not fetched. The specific CoinRun level-count numbers are not included.

## Q2: GNN / transformer / permutation-invariant critics that transfer across team sizes or graphs

### Takeaway
Permutation-invariant or GNN critics clearly help *training at a fixed size* (PIC), and set critics enable variable-roster training. Reported cross-size and cross-graph *zero-shot* successes are mainly policy-side (shared local operators). Zero-shot transfer to larger or harder instances often fails outright (MAT: 0% win rate on 5 of 6 harder SMAC maps). A 2026 study explicitly argues that a set-compatible critic alone does not transfer coordination to the policy. A fixed-width MAPPO critic was worst at held-out roster sizes on RWARE.

### Cited Findings
- **PIC (2019):** a graph-convolution critic with max-pooling gives a permutation-invariant critic and handles heterogeneous agents through node attributes. The abstract reports "30 times more agents" (up to N = 200) and a 15–50% test-reward gain on MPE. Cooperative Navigation final reward, MLP critic vs PIC: N=15 −6,490 vs −1,999; N=30 −20,722 vs −11,363; N=100 −128,100 vs −71,495; N=200 −502,349 vs −436,215. Each N is trained separately; in the full text I extracted, I found no zero-shot cross-N critic transfer experiment — [Liu et al. 2019, PIC (CoRL 2019)](https://arxiv.org/abs/1911.00025)
- **UPDeT (2021):** a transformer per-agent Q-network with policy decoupling, combined with value-decomposition mixers (VDN/QMIX/QTRAN) on SMAC. Transfer experiments go 7m→5m→3m and 3m→5m→7m, loading at step 0 and 500k and fine-tuning. The paper says the architecture makes zero-shot transfer *possible*, with performance without fine-tuning shown only as points in Fig. 5 (no extractable numbers). The "vanilla transformer fails to beat the enemies", and an aggregation (pooled global embedding) transformer was "worse than… the GRU-based model" — [Hu et al. 2021, UPDeT (ICLR 2021)](https://arxiv.org/abs/2101.08001)
- **MAT (2022):** pre-trained on 8 SMAC tasks (10M samples) and tested on 6 harder unseen tasks. Zero-shot median win rates are **0%** on 5m_vs_6m, 27m_vs_30m, 2s_vs_1sc and MMM2, 3.1% on 1c3s5z, and 100% only on 8m (an easy marine map close to the training set). With 10% fine-tuning data: 27m_vs_30m 71.2% (MAT) vs 26.8% (MAPPO); MMM2 62.5% vs 36.2%. Multi-agent MuJoCo (HalfCheetah with a disabled joint): zero-shot MAT beats zero-shot MAPPO on 4 of 6 variants. MAT's critic is a per-agent observation-value head on the shared encoder — [Wen et al. 2022, MAT (NeurIPS 2022)](https://arxiv.org/abs/2205.14953)
- **PC3D (May 2026):** zero-shot held-out roster sizes (Spread {9,10}, LBF {7,8}, RWARE {9,10}), 5 seeds. Held-out mean return, given as IPPO / MAPPO (fixed-width masked critic) / PIC-MAPPO (set critic) / PC3D: Spread −103.43 / −86.13 / −84.42 / −79.18; LBF (×10²) 8.24 / 7.91 / 8.40 / 8.98; RWARE 6.17 / **2.77** / 5.67 / 7.73. The authors argue a set-compatible critic "improves the centralized training signal but does not automatically provide the decentralized policy with a reusable notion of coordination across roster sizes" — [PC3D, arXiv 2605.10377 (2026)](https://arxiv.org/abs/2605.10377)
- **Graph-RL for network control (2023):** shared local GNN operator with A2C (separate actor and critic decoders), bi-level with a one-step LP. Dynamic vehicle routing, zero-shot New York→Shenzhen: 36,100 vs 36,918 when trained in-city (≈97.8% retained; 89.5% of oracle 40,332). End-to-end RL zero-shot reaches 18,568. Supply-chain inventory (SCIM) Graph-RL reaches 96.8% / 96.0% / 99.5% of oracle on 1F2S / 1F3S / 1F10S (in-distribution). Transfer is credited to "the locality of graph network-based policies… a shared, local operator" — [Gammelli et al. 2023, Graph RL for Network Control via Bi-Level Optimization (ICML 2023)](https://proceedings.mlr.press/v202/gammelli23a/gammelli23a.pdf)
- **Inventory networks, HDPO + GNN (2023–2025):** critic-free pathwise gradients through a differentiable simulator. A shared-weight GNN's optimality gap "remains flat" as stores go from 3 to 50, while a vanilla NN deteriorates. The authors report "30% performance gains with just 128 training s[amples]" from GNN structure. HDPO beats generalized-newsvendor heuristics by up to 24% on real data — [Alvo, Russo, Kanoria, Lee, arXiv 2306.11246](https://arxiv.org/abs/2306.11246)
- **GNN + MARL inventory control (2024/2025):** a GNN state representation with global mean pooling under CTDE, tested on 4 supply-chain configurations. The abstract gives no held-out-topology test and no numbers — [Kotecha & del Rio Chanona, arXiv 2410.18631](https://arxiv.org/abs/2410.18631) (abstract only)
- **Traffic signal control:** TransferLight (2024) trains one weight-tied GNN policy on random networks and tests zero-shot on 1 to 21 intersection real networks. It still reports that trained baselines "failed to control a subset of intersections" on Cologne8, and its own run diverges on Ingolstadt21 (no numbers). The paper does not describe how the value function handles different network sizes — [TransferLight, arXiv 2412.09719](https://arxiv.org/abs/2412.09719). One 2025 study reports that MPLight and AttendLight "fail to converge" when transferred from Jinan/Hangzhou to a 196-intersection New York grid — [arXiv 2503.08728 (search snippet, not fetched in full)](https://arxiv.org/abs/2503.08728)

### Inferences
- The positive transfer evidence (Gammelli, HDPO, TransferLight) is for **local, weight-shared policies**, sometimes combined with an explicit per-instance optimizer (LP) or a differentiable simulator. None of them shows a pooled global value head extrapolating across graph sizes. A pooled-embedding critic (the shape of the project's planned "pooled per-agent embeddings" critic) has the weakest support. UPDeT's pooled "aggregation transformer" underperformed a GRU, and global pooling is where scale information (12 vs 42 producers) must be represented explicitly.
- PC3D's RWARE row shows that a centralized-critic method can do *worse* than independent learners at held-out sizes (2.77 vs 6.17), while a set critic recovers most of the gap (5.67). The architecture of the critic matters, but even a good critic does not beat IPPO by much there.
- MAT's table shows that zero-shot transfer to harder, bigger instances (27m_vs_30m, MMM2) is usually near 0. Few-shot fine-tuning on the target is where the gains appear. For the project, the 42-producer worlds need to be in the critic's training distribution; extrapolating from 12 is not supported.

### Gaps
- CoLight / MPLight original papers were not fetched. Their train-vs-test intersection counts and numbers are not included.
- No power-grid MARL critic-transfer paper was found within the search budget.
- No mean-field-critic transfer numbers were found within the search budget.
- UPDeT's zero-shot win rates exist only as figure points and could not be extracted.

## Q3: Reported failure modes: value scale across instances, reward normalization, PopArt

### Takeaway
Return scale differs across instances: larger economy means more units-to-go, and different horizons and tightness add to that. A single regression head then has its loss and gradients dominated by large-scale instances. This is the multi-task problem PopArt was built for. MAPPO's own ablation finds value normalization "never hurts" and is "critical" when returns span a wide range. Two further failure modes come with large joint-input critics: high input dimensionality, and missing agent-specific features.

### Cited Findings
- In multi-task IMPALA, "the magnitude of updates" depends on "1) the scale of rewards, 2) the sparsity of rewards, 3) the competence of the agent". Even with reward clipping, update magnitudes "can still differ significantly between" tasks. PopArt-IMPALA reached a **110% median** human-normalized score on Atari-57 with one agent, vs **59.7%** median for multi-task IMPALA, and 72.8% mean on DMLab-30 — [Hessel et al. 2019, Multi-task Deep RL with PopArt (AAAI 2019)](https://arxiv.org/abs/1809.04474)
- In MAPPO, value targets "drastically change due to differences in the realized returns, leading to instability". MAPPO standardizes targets with running mean and std, regresses to normalized targets and denormalizes for GAE (PopArt in the algorithm box). "Value normalization never hurts training and often improves the final performance of MAPPO significantly"; in MPE Spread, with returns from below −200 to 0, it is "critical" — [Yu et al. 2022, The Surprising Effectiveness of PPO in Cooperative MARL (NeurIPS 2022 D&B)](https://arxiv.org/abs/2103.01955)
- MAPPO critic-input ablation on SMAC: concatenated-local-observations state (CL) "is ineffective, particularly in maps with many agents". The environment-provided state (EP) underperforms on hard maps because it lacks agent-specific information. Agent-specific global state (AS) and its feature-pruned version (FP) work best. "Suggestion 2: include both local, agent-specific features and global features… check that these features do not unnecessarily increase the input dimension" — [Yu et al. 2022](https://arxiv.org/abs/2103.01955)
- A fixed-width MAPPO critic must be sized to the maximum roster with masked slots, and it performed worst at held-out sizes on RWARE (2.77 vs IPPO 6.17) — [PC3D 2026](https://arxiv.org/abs/2605.10377)
- Partial observability makes value targets look like instance memorization (remaining-steps dependence), which damages a shared encoder — [Raileanu & Fergus 2021](https://arxiv.org/abs/2102.10330)
- Learned-critic baselines for policy gradient on combinatorial instances underperform per-instance baselines. Kool et al. found a greedy-rollout baseline "more efficient than using a value function". It improves both the Attention Model and the Pointer Network, and getting actor-critic "to work is non-trivial" — [Kool et al. 2019, Attention, Learn to Solve Routing Problems! (ICLR 2019)](https://arxiv.org/abs/1803.08475)

### Inferences
- Failure modes that likely apply to the project's past attempts:
  - **Scale mismatch.** 42-producer economies produce far larger units-to-go than 12-producer ones. Without per-instance normalization the critic fits the big worlds and is noise on the small ones, or the reverse.
  - **Remaining-horizon memorization.** Units-to-go is roughly linear in cycles remaining, the IDAAC Ninja pattern.
  - **Too few instances** (6, or 30) relative to Procgen's 10k-level regime.
  - **Pooled global input** that drops agent-specific features, or concatenation that explodes dimension.
- A critic "stalled near 0" is consistent with unnormalized targets plus a sparse or delayed signal. It is also consistent with a critic whose variance-reduction gain is near zero (explained variance ≈ 0). Which of the two applies is a diagnosable question: log explained variance per world and per size bucket.

### Gaps
- No source found that quantifies how per-instance return normalization changes held-out critic R^2 specifically. PopArt and MAPPO report effects on policy return, not on critic R^2.

## Q4: Mitigations and what the evidence supports

### Takeaway
Ranked by strength of evidence for the project's case:
1. **Per-instance / per-world value normalization** (PopArt per world or size bucket; at minimum MAPPO-style running normalization). Strong evidence for multi-task stability.
2. **Decouple** critic and actor (separate networks, no shared trunk), and optionally train the actor trunk on advantages (DAAC) or use PPG distillation. Strong single-agent evidence for generalization.
3. **Critic used only at train time, on the training worlds**: measure its explained variance there, not its held-out R^2.
4. **Agent-specific plus global critic input** with controlled dimensionality (MAPPO AS/FP), local weight-shared GNN operators, explicit scale features (producer count, cycles remaining).
5. **Replace the learned critic with a per-instance sampled baseline** (POMO shared baseline: the mean return of N rollouts on the same instance; or a greedy-rollout baseline), which is well suited to a deterministic simulator where re-rolling one world is cheap.
6. **IPPO / local values** as a control arm.

I found no primary evidence specific to critic *ensembles* for cross-topology generalization.

### Cited Findings
- PopArt adaptively rescales value targets per task and preserves the unnormalized outputs, giving scale-invariant updates for actor-critic. Results: 110% vs 59.7% median human-normalized score on Atari-57 — [Hessel et al. 2019](https://arxiv.org/abs/1809.04474)
- MAPPO Suggestion 1: "Utilize value normalization to stabilize value learning" — [Yu et al. 2022](https://arxiv.org/abs/2103.01955)
- DAAC/IDAAC: separate value network, advantage-prediction head for the policy, adversarial step-order invariance. Best Procgen test score at publication (163.7 PPO-normalized) — [Raileanu & Fergus 2021](https://arxiv.org/abs/2102.10330)
- PPG: separate phases, distillation of value into the policy trunk under a KL constraint, higher sample reuse for the value. "Significantly improves sample efficiency" over PPO on Procgen — [Cobbe et al. 2021](https://arxiv.org/abs/2009.04416)
- The POMO shared baseline is b_shared(s) = (1/N) Σ_j R(τ_j) over N rollouts of the same instance. The advantage has zero mean per instance, the baseline needs no forward pass of a "separately trained network (Critic)", and it makes training "highly resistant to local minima". TSP100 optimality gap 0.14% (abstract) — [Kwon et al. 2020, POMO (NeurIPS 2020)](https://arxiv.org/abs/2010.16011)
- Greedy-rollout baseline beats the critic baseline for the Attention Model and the Pointer Network; the critic baseline is about 13% slower per epoch — [Kool et al. 2019](https://arxiv.org/abs/1803.08475)
- Critic-free pathwise optimization (HDPO) with a weight-shared GNN on inventory networks keeps a flat optimality gap from 3 to 50 stores — [Alvo et al., arXiv 2306.11246](https://arxiv.org/abs/2306.11246)
- Independent local values (IPPO) are competitive on SMAC and on fully-observable MPE/LBF tasks — [de Witt et al. 2020](https://arxiv.org/abs/2011.09533); [Papoudakis et al. 2021](https://arxiv.org/abs/2006.07869)
- Set or permutation-invariant critics fix the fixed-width problem at variable sizes (PIC-MAPPO 5.67 vs MAPPO 2.77 on held-out RWARE), but they do not transfer coordination to the actor by themselves — [PC3D 2026](https://arxiv.org/abs/2605.10377); [Liu et al. 2019](https://arxiv.org/abs/1911.00025)

### Inferences
- **Concrete recipe if a critic is kept:**
  - (a) Normalize targets per world, or per size bucket (12 vs 42), with PopArt. Better still, regress *per-world-normalized* return-to-go, e.g. divided by that world's floor-policy units over the same remaining cycles.
  - (b) Feed explicit cycles-remaining and size features, so the critic need not infer them from spurious topology cues.
  - (c) Use separate actor and critic networks with no shared GNN trunk, or PPG-style distillation.
  - (d) Use per-agent critic inputs: agent-local plus pooled global (AS-style).
  - (e) Monitor explained variance per world; if it is ≤ 0 on the training worlds, the critic is adding noise.
  - (f) Train on all ~30 topologies, including 42-producer ones. Do not rely on 12→42 extrapolation (MAT/PC3D evidence).
- **The critic-free option is especially well matched to this project.** The domain is deterministic and seeded, so K re-rollouts of the *same* world and seed give an exact per-instance baseline (POMO/GRPO-style): no value generalization is required at all. The counterfactual per-agent version (re-roll with one agent's action swapped) gives credit assignment directly. Its cost is K× rollouts, which fits a CPU-pod fan-out workflow.
- Ensembles could expose critic uncertainty on unfamiliar topologies, but I found no primary source showing that this fixes cross-topology value generalization. Treat it as unproven.

### Gaps
- No primary source in the search budget evaluates critic ensembles for cross-instance generalization in MARL.
- No source compares PopArt-per-instance vs global normalization on held-out graphs.
- GRPO / RLOO (LLM-era critic-free group baselines) and counterfactual-baseline MARL (COMA) were not fetched here; presumably they are covered by the adjacent "critic-free" research notes.
