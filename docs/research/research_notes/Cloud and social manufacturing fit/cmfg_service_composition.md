# Service composition and optimal selection (SCOS) in cloud manufacturing, including multi-tier / composite-product supply

Evidence grades used throughout:
- **[A]** full text (or most of it) read by this researcher.
- **[B]** abstract, index record or publisher listing only. The details come from search-engine summaries of those pages, not from the full text.
- **[C]** described secondhand, in another paper's related-work section or a review.
- **[U]** unverified. From background knowledge, no page fetched. Do not cite without checking.

Access limits: PeerJ, PMC, MDPI and Bohrium refused automated fetches (403/404/captcha), and Springer, Elsevier and T&F full texts are paywalled. The one SCOS full text read is the 2024 arXiv review (Li, Liu, Shi). Every per-paper record below is therefore [B] unless marked otherwise. The numbers are the papers' own claims and have not been reproduced.

## Q1. Do SCOS methods assume a central composer with full information? Which decentralised or self-organising composition methods exist?

### Takeaway
Mainstream SCOS is centralised. A cloud platform decomposes a task into a fixed chain of subtasks, knows every candidate service's QoS vector (time, cost, reliability, and so on), and runs a metaheuristic or a single-agent DRL policy to choose one provider per subtask. Decentralised variants exist but are thin and weakly evaluated:
- blockchain or peer-to-peer partitioning of the problem (Block-SC);
- game-theoretic models with providers as players;
- contract-net style self-organising agents, mostly from cloud computing rather than manufacturing;
- one recent multi-agent RL paper under centralised training, decentralised execution (CTDE).

None of these matches our setting: many independent requesters, each routing its own order with only local information.

### Cited Findings

**Surveys (frame of the field)**
- **Li, Liu, Shi (2024), "Review of Cloud Service Composition for Intelligent Manufacturing", arXiv:2408.01795 [A]** (preprint, not peer reviewed).
  - Resource services are "centralized under the management of the cloud platform"; the platform decomposes tasks, picks candidate services and builds the composition. In single-participant studies the service demander "typically plays a central role in decision making" — [arXiv HTML](https://arxiv.org/html/2408.01795v1)
  - Of the 78 papers counted, the optimisation indicators are overwhelmingly demander-side: cost 68, time 66, reliability 28, quality 28, reputation 26, availability 16, maintainability 4. Provider-side load balancing appears in only 5 and resource utilisation in 9 — [arXiv HTML](https://arxiv.org/html/2408.01795v1)
  - Task structures are sequential, parallel, selective and circular. Zhou & Yao (2017) showed the last three reduce to the sequential model — [arXiv HTML](https://arxiv.org/html/2408.01795v1)
  - The algorithm taxonomy is heuristic (ACO, PSO, WOA, NSGA-II, GWO, chaotic optimisation, SA, hybrids) vs RL (DDPG, PD-DQN, AV-MPO/GTrXL, REINFORCE attention model). The review cites **no multi-agent RL paper** and **does not discuss decentralised composition** — [arXiv HTML](https://arxiv.org/html/2408.01795v1)
  - Typical capacity constraint: "a provider's service performs at most one subtask at a time". Load balancing is the standard deviation of provider workloads — [arXiv HTML](https://arxiv.org/html/2408.01795v1)
  - Stated gaps: too little attention to providers and platforms, no real data or standards, no common simulation platform, weak theory. RL decisions are hard to interpret and its training is unstable and sample-hungry — [arXiv HTML](https://arxiv.org/html/2408.01795v1)
- **"Service optimal selection and composition in cloud manufacturing: a comprehensive survey", Int J Adv Manuf Technol 97:795–808 (2018), DOI 10.1007/s00170-018-1910-4 [B].**
  - Frames SCOS as NP-hard. Organises the literature by selection criteria, algorithms, optimisation functions, correlation consideration, subtask-to-service mapping and dynamic composition — [Peeref](https://www.peeref.com/works/5379635), [bib-bvb record](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_proquest_journals_2622990097/Details)
  - Usually attributed to Bouzary & Chen [U]. The fetched records did not confirm the authors; the one result naming Bouzary & Chen was their separate 2019 grey-wolf SCOS paper.
- **Hayyolalam et al. (2019), IJAMT vol. 105 [B].** Describes itself as the first systematic review of CMfg-SCOS methods — [search record](https://searchworks.jioinstitute.edu.in/articles/asx__130205409). The mapping of this record to the paper was not verified.

**Approach 1: centralised metaheuristic SCOS (the canonical model)**
- **Tao, Laili, Xu, Zhang (2013), "FC-PACO-RM: a parallel method for service composition optimal-selection in cloud manufacturing system", IEEE TII 9(4):2023–2033, DOI 10.1109/TII.2012.2232936 [B].**
  - Model: multi-objective, constrained SCOS solved by the platform.
  - Method: adaptive chaos optimisation with roulette-wheel selection, an island-model parallel GA (full connections) and reflex migration.
  - Evaluation: compared against 3 serial and 7 parallel algorithms on 3 cases; claims effectiveness on complex SCOS — [Peeref](https://peeref.com/works/10291566), [Tao homepage](https://shi.buaa.edu.cn/taofei/en/lwcg/189185/content/3154.htm), [Unpaywall](https://unpaywall.org/10.1109%2FTII.2012.2232936)
  - Assumptions (typical of the line): the full QoS of every candidate is known, one provider per subtask, the platform decides the whole chain at once, and there are no dynamics within an episode [C, from the review's description of the class](https://arxiv.org/html/2408.01795v1).

**Approach 2: multi-task SCOS with enterprise capacity (closest centralised model to ours)**
- **Liu, Xu, Zhang, Wang, Zhong (2017), "Workload-based multi-task scheduling in cloud manufacturing", RCIM 45:3–20, DOI 10.1016/j.rcim.2016.09.008 [B].**
  - Model: many tasks share services. It adds service quantity, service efficiency, task workload and **enterprise capacity**, plus logistics.
  - Finding (simulation): scheduling larger-workload tasks first shortens makespan and raises service utilisation without lowering fulfilment quality when there is no time constraint. Time-constrained cases were studied separately.
  - The authors name continuous task arrival as future work — [SciSpace](https://scispace.com/papers/workload-based-multi-task-scheduling-in-cloud-manufacturing-401ft54py1), [Bohrium](https://bohrium.dp.tech/paper/arxiv/599de00c5d763cf99f7398b7)
  - Follow-up (Ping, Liu, Zhang, Wang, Xu, 2023): DQN scheduling with dynamic task arrival, reported best among the approaches it compared — [BUAA](https://research.buaa.edu.cn/zh/publications/deep-reinforcement-learning-based-multi-task-scheduling-in-cloud-/) [B]

**Approach 3: single-agent DRL composer (logistics-aware)**
- **Liang, Wen, Liu et al. (2021), "Logistics-involved QoS-aware service composition in cloud manufacturing with deep reinforcement learning", RCIM [B].**
  - Method: PD-DQN (dueling DQN with prioritised replay). Motivated by the parameter tuning that GA/PSO/ACO need and their poor adaptation to dynamics.
  - Compared only with DQN and Q-learning; claims "effective, robust, scalable" — [Bohrium](https://bohrium.dp.tech/paper/arxiv/6578feab939a5f4082b17a6b), [Peeref](https://peeref.com/works/21854175)
- **Liu et al. (2022), "Logistics-involved service composition in a dynamic cloud manufacturing environment: a DDPG-based approach", RCIM 76 [B].**
  - DDPG vs ACO: **ACO converged faster and reached ~0.997% higher steady-state QoS**; DDPG was more adaptable and robust.
  - DDPG vs DQN: ~3.249% higher QoS and faster, more stable convergence — [Bohrium](https://bohrium.dp.tech/paper/arxiv/5ce3a99bced107d4c6560026), [Peeref](https://www.peeref.com/works/25475421)
- "Service composition in cloud manufacturing: a DQN-based approach" (BUAA group) [B] — [BUAA](https://research.buaa.edu.cn/en/publications/service-composition-in-cloud-manufacturing-a-dqn-based-approach/)
- In every case a single agent composes the whole chain, i.e. the central composer is learned.

**Approach 4: dynamic re-composition (reconfiguration after service exceptions)**
- **Wang et al., JIM (2020), SCEHAA [B].** Improved ACO. Handles service exceptions under strict deadlines by "adaptive service execution path reconfiguration".
- **Wang et al., RCIM (2021), SCRIHHO [B].** Improved Harris Hawks optimiser. Constraints: exceptions, service occupancy time, the original strict time limit, dynamic QoS change. Claims to beat PSO and GWO.
- **Wang et al., Knowledge-Based Systems (2021), AMPOWOA [B].** Reconfiguration model "DSCRPC" with objectives time, cost and product service quality. Trigger: equipment faults invalidate a service.
- **Zhao, RCIM (2025), MOSCRMA [B].** Reconfiguration may use only providers' **available service time windows** given their occupied status. Memetic algorithm with flexible lot splitting and intermingling.
- Sources for the four papers above: [x-mol JIM](https://www.x-mol.com/paper/1311860491742842880), [BUAA](https://research.buaa.edu.cn/en/publications/enhancing-reconfiguration-of-cloud-manufacturing-service-composit/), [Peeref](https://www.peeref.com/works/23840352), [CVTISR record](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c328t-3b7f0ea77f7a9fd320160a9e2b147a12d938e13881336bba30b1092ea42443bf3). The per-paper mapping of these links comes from a search summary and was not individually verified.
- All are **reactive** central re-optimisation after a failure. The search found no proactive (failure-anticipating) composition in this line.

**Approach 5: game-theoretic composition (providers as players)**
- **Vaziri Goudarzi et al. (2021), "Equilibrial service composition model in Cloud manufacturing (ESCM) based on non-cooperative and cooperative game theory for healthcare service equipping", PeerJ CS [B].**
  - Model: a non-cooperative game in which providers join the composite service as players, then a cooperative game with coalitions.
  - The Nash point gives the composite QoS and each player's payoff. Cooperative play gave better payoffs and better QoS than non-cooperative play.
  - Case: equipping 24 hospitals. Whether the data are real or synthetic was not confirmed — [PeerJ](https://peerj.com/articles/cs-410), [PubMed](https://pubmed.ncbi.nlm.nih.gov/33817051/)
- **Distributed scheduling game [B].** A static scheduling sub-game plus a dynamic selection sub-game; the Nash equilibrium is computed by double ACO (DACO) — [McMaster](https://experts.mcmaster.ca/scholarly-works/2658488)
- **Non-cooperative dynamic game for distributed 3D-printing service selection (BUAA, AsiaSim 2019) [B].** Equilibrium found by an evolutionary algorithm — [BUAA](https://research.buaa.edu.cn/en/publications/distributed-3d-printing-services-in-cloud-manufacturing-a-non-coo/)
- Note: in these papers the equilibrium is still computed centrally by a metaheuristic. The game is a modelling device, not a protocol run by distributed agents.

**Approach 6: decentralised and self-organising composition**
- **Aghamohammadzadeh & Fatahi Valilai (2020), "A novel cloud manufacturing service composition platform enabled by Blockchain technology" (Block-SC), IJPR 58(17):5280–5298, DOI 10.1080/00207543.2020.1715507 [B].**
  - Explicitly "challenges the centralised mechanism". Splits the composition problem into many sub-problems, each holding a small fraction of the service/task pool.
  - Motivation: parameters that "change rapidly over time". The abstract gives no numbers — [RePEc](https://ideas.repec.org/a/taf/tprsxx/v58y2020i17p5280-5298.html)
  - Follow-up: Radmanesh, Haji, Fatahi Valilai (2021), PeerJ CS, blockchain run by "an autonomous peer-to-peer network" — [PeerJ cs-743](https://peerj.com/articles/cs-743)
  - Another paper's related work calls Block-SC "a decentralized peer-to-peer strategy" — [PeerJ cs-461](https://peerj.com/articles/cs-461) [C]
- **Gutierrez-Garcia & Sim (2013), self-organising agents for service composition (cloud computing, not manufacturing), Applied Intelligence, DOI 10.1007/s10489-012-0380-x [C].** Agents self-organise through interaction and negotiation, using a contract-net bidding scheme: managers announce tasks and contractors bid — [Kent repository](https://kar.kent.ac.uk/32021/). The full text was not readable.
- **MARL with CTDE for multi-task CMfg service composition, MDPI *Symmetry* 18(2):238 (2026) [B].** Independent task agents plus a global coordination agent, targeting inter-task fairness. Argues that single-agent RL cannot balance global efficiency with fairness — [MDPI](https://www.mdpi.com/2073-8994/18/2/238). Full text refused (403); sizes, baselines and gains unknown.
- **Patent CN121981485A [B].** CMfg scheduling with discrete-event simulation and MARL under CTDE. A patent claim, not peer-reviewed evidence — [patent](https://www.goveda.com/patent/CN-121981485-A)
- A Chinese-language review finds multi-agent simulation better suited than analytic models to the distribution, dynamics and diversity of service composition — [kjglyj](https://kjglyj.ijournals.cn/kjglyj/article/abstract/20221007017) [B]

### Inferences
- The SCOS decision unit is **one composite task, composed once, end to end, by one decider**. Ours is **many concurrent orders, each routed one tier at a time by a different, myopic decider**. In our domain the consumer's ordering agent chooses only the final assembler. The assembler's own ordering agent chooses its input suppliers later, under TTL 2. So our system performs *recursive, decentralised, sequential composition*, and the composite chain is never chosen as a whole.
- The decentralised SCOS work (Block-SC, the games) decentralises **computation or ownership** of the problem, not **information**. Each piece still optimises over a known QoS table. None models agents that cannot see same-cycle orders from others. Only the CTDE MARL paper is structurally close to our CTDE setup, and its evidence could not be read.
- The field's indicator mix (cost/time/reliability in more than 80% of papers, load balancing in about 6%) shows SCOS optimises the **requester's** QoS. Our objective is **system-level**: total fill without running capacity down. SCOS objectives are therefore misaligned with ours by construction.

### Gaps
- Could not read the Block-SC full text: how sub-problems are formed, whether peers hold only local information, the results.
- Could not read the MARL-CTDE (*Symmetry* 2026) paper: observation spaces, reward, sizes, baselines.
- Did not find any CMfg paper in which **every requester independently routes its own order** with local information (the setup closest to MWW). It may exist under "distributed order allocation" or "supply chain formation" labels, which this search did not cover.

## Q2. How do they handle multi-tier dependencies (a service's inputs come from other providers), cascading failure, and capacity limits over time?

### Takeaway
Poorly. "Multi-tier" in SCOS usually means a **service hierarchy** (resource → function → process) or a **BOM decomposed by the platform into subtasks**. It does not mean providers who must themselves procure inputs from other providers whose capacity is shared and degrading. The usual capacity limit is "one subtask at a time" or a fixed enterprise capacity. Cascading failure appears only in complex-network robustness simulations, not in composition algorithms. Dynamics appear as reactive re-composition after a single service exception.

### Cited Findings
- **Multi-level SCOS (Ding et al. 2019/2020, Beihang journal) [B].**
  - Three service levels: resource, function, process. The 2020 journal version is "A niching behaviour-based algorithm for multi-level manufacturing service composition optimal-selection" — [BUAA journal](https://bhxb.buaa.edu.cn/bhzk/en/article/doi/10.13700/j.bh.1001-5965.2018.0630), [BUAA (zh)](https://research.buaa.edu.cn/zh/publications/%E4%B8%80%E7%A7%8D%E5%A4%9A%E5%B1%82%E6%AC%A1%E5%88%B6%E9%80%A0%E6%9C%8D%E5%8A%A1%E5%BB%BA%E6%A8%A1%E5%92%8C%E7%BB%84%E5%90%88%E4%BC%98%E9%80%89%E6%96%B9%E6%B3%95/)
  - A hierarchy of service granularity, not a supply dependency between independent providers.
- **BOM as a bill of manufacturing services (Qanbari et al. 2014, IEEE Internet Computing, TOSCA) [B].** Hierarchy from end product to services, components, resources and materials, with parent–child (assembly–component) relations — [TU Wien PDF](https://dsg.tuwien.ac.at/Staff/sd/papers/Zeitschriftenartikel%20S.%20Qanbari%202014%20Toward%20Portable.pdf). A modelling and portability paper; no optimisation results found.
- **Three-tier programming model (Lim, Xiong, Wang 2022) [B/C].**
  - Tiers: demanders (lower), platform operators (middle), resource providers (upper), each a different decision maker — reported in the [same search set](https://arxiv.org/pdf/2408.01795)
  - The arXiv review lists Lim et al. (2022) with an improved NSGA-II (a-i-NSGA-II) — [arXiv HTML](https://arxiv.org/html/2408.01795v1)
  - The tiers are *stakeholder* tiers, not supply tiers.
- **Supplier-network logistics plus manufacturing composition: Aghamohammadzadeh, Malek, Fatahi Valilai (2020), IJPR 58(7):1987–2015, DOI 10.1080/00207543.2019.1640406 [B].**
  - Each production task and logistics operation is a service, arranged by operation process chart.
  - Objective: lower operation and logistics cost subject to a "configured cloud entropy" criterion. Evaluated on a numerical example only — [RePEc](https://ideas.repec.org/a/taf/tprsxx/v58y2020i7p1987-2015.html)
- **Correlation-aware composition** is a standard survey category: QoS of one service depends on which other services are co-selected — [survey abstract](https://www.peeref.com/works/5379635) [B]. It is the closest SCOS concept to "my input supplier's state matters", but no detail was retrieved.
- **Capacity over time:**
  - Most models: "a provider's service performs at most one subtask at a time"; Zhang et al. (2018) used a load-queue model for dynamic QoS under load — [arXiv HTML](https://arxiv.org/html/2408.01795v1) [A]
  - Liu et al. (2017) add enterprise capacity and workload — [SciSpace](https://scispace.com/papers/workload-based-multi-task-scheduling-in-cloud-manufacturing-401ft54py1) [B]
  - Zhao (2025) uses providers' available time windows when re-composing — [CVTISR](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c328t-3b7f0ea77f7a9fd320160a9e2b147a12d938e13881336bba30b1092ea42443bf3) [B]
  - **No retrieved SCOS paper models capacity that degrades and must be restored by buying products from other providers.**
- **Cascading failure / robustness:**
  - Entropy (MDPI, 2023), "Robustness of Cloud Manufacturing System Based on Complex Network and Multi-Agent Simulation" [B]. Couples a complex-network model with multi-agent simulation and scores robustness by service time, cost, reliability and order completion rate — [MDPI](https://mdpi.com/1099-4300/25/1/45). Full text refused; failure model and numbers not retrieved.
  - MBE (2023) [B]: multi-agent simulation of CMfg service processes to test task rescheduling under disturbances — [AIMS](https://www.aimspress.com/article/doi/10.3934/mbe.2023147)
- **Re-composition** after an exception is reactive and central (Wang 2020/2021, Zhao 2025, under Q1 Approach 4). The review notes the claim that RL "often adapts without requiring a complete re-optimization" — [arXiv HTML](https://arxiv.org/html/2408.01795v1) [A]

### Inferences
- Our hardest failure mode is a *producer-side* allocation failure that propagates through supply: early PPO producers misallocated restoration and starved the middle layer of depth-3 chains, ending in system death. SCOS has no counterpart. In SCOS a provider never needs inputs from other providers to remain able to serve, so starvation cascades cannot arise in its models. The literature therefore offers no tested mechanism for our cascade problem. Only the complex-network robustness simulations touch it, and only descriptively.
- The correlation-aware idea is the right *concept* for multi-tier supply: the value of choosing assembler A depends on A's suppliers. We lack evidence on how SCOS operationalises it beyond pairwise QoS correlation tables.

### Gaps
- No retrieved paper treats circular supply dependencies (A needs B's product, B needs A's), which our domain has.
- The Entropy 2023 failure model (random vs targeted node removal, load redistribution) and its results are unknown (403).
- Multi-tier SCOS with **provider-side input procurement** may exist under supply chain configuration or supply chain formation (e.g. agent-based supply chain formation). It was outside this search and is partly auction-based, which the brief excludes.

## Q3. Which ideas (upstream-aware selection, reserving input capacity, composition-level feasibility checks) could translate to per-order routing by independent agents in our domain? Assumption-by-assumption fit and offline checks

### Takeaway
Most SCOS machinery does not transfer: central composer, full QoS table, single-task objective, metaheuristic search. Three ideas do transfer as local routing signals:
1. **Composition-level feasibility, one tier deeper ("deep makeable").** Weight an assembler by whether its missing inputs are obtainable upstream, not only by its own stock.
2. **Exception-triggered re-composition.** Stop routing to producers in outage, or whose recent sub-orders failed.
3. **Multi-task load awareness.** Liu 2017's workload/capacity view, which MWW already approximates.

A key fact from our code: MWW's weight `makeable_qty_now` already equals min(capacity-makeable, input-inventory-makeable) (`/srv/research/tokenomics/tokenomics/tokenomics/domain/metrics.py:119-127`). MWW is therefore already one-tier upstream-aware through the assembler's own stock. The open question is whether **tier-2+ awareness** would add anything, and recorded data can test that before any build.

### Cited Findings
- Our MWW weight: `complete_qty_now(p, x) = min(capacity_makeable_qty(p, x), input_makeable_qty(p, x))`. `makeable_qty_now` returns it. Bottleneck helpers separate the two limits: `capacity_bottleneck_ratio`, `input_bottleneck_ratio`, `producer_product_bottleneck_ratio` — `/srv/research/tokenomics/tokenomics/tokenomics/domain/metrics.py:119-155` (repo code, read directly).
- SCOS evaluations compare only against other metaheuristics or DRL variants. The two DRL papers compare to DQN, Q-learning and ACO, and ACO even edged DDPG on steady-state QoS — [Bohrium DDPG](https://bohrium.dp.tech/paper/arxiv/5ce3a99bced107d4c6560026) [B]. None compares to a simple capacity-proportional rule like MWW (no such baseline appears in the review's account) — [arXiv HTML](https://arxiv.org/html/2408.01795v1) [A]
- Liu 2017: larger workloads first raises utilisation without lowering quality (no time constraint) — [SciSpace](https://scispace.com/papers/workload-based-multi-task-scheduling-in-cloud-manufacturing-401ft54py1) [B]
- Re-composition papers trigger on service exceptions (equipment faults) and re-plan within available time windows — [CVTISR](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c328t-3b7f0ea77f7a9fd320160a9e2b147a12d938e13881336bba30b1092ea42443bf3) [B]

### Inferences — fit by approach, assumption by assumption

**A1. Centralised metaheuristic SCOS (Tao 2013 line)**

| Assumption | Fit | Note |
|---|---|---|
| One composer decides the whole chain | **Breaks** | Each tier is routed by a different ordering agent at a different time. |
| Full, static QoS table per candidate | **Breaks** | Agents see local state only. Our closest QoS analogue is `makeable_qty_now`, plus recent fill history if recorded. |
| One provider per subtask | **Holds** | Whole-order rule. EXP-0066 tested whether splitting is needed, giving reading `split_needed` / `whole_suffices`. |
| Objective is the requester's cost/time/QoS | **Breaks** | Our objective is system fill, no run-down, no death. Tokens are not cost. |
| No dynamics within an episode | **Breaks** | Degradation, TTL 2, shocks. |

Verdict: not transferable as an algorithm. At best it is a hindsight bound, solving the composition offline for a recorded cycle, which overlaps the existing B2 / B_lp bounds.

**A2. Multi-task capacity-aware scheduling (Liu 2017, Ping 2023)**

| Assumption | Fit | Note |
|---|---|---|
| Many tasks compete for shared capacity | **Holds** | The closest SCOS model to ours. |
| Central scheduler orders tasks by workload | **Breaks** for routers | Task ordering in our domain is the producer's price-sorted auction, so it is mechanism design (out of scope). |
| Fixed enterprise capacity | **Adaptable** | Ours degrades and is restored. |

Transferable idea: load awareness. MWW's makeable-proportional draw already is a decentralised load balancer.

Offline check: on the recorded EXP-0057 / EXP-0066 MWWk rows, correlate per-cycle routing concentration (`lead_concurrent_top_share`, recorded by the split-check tooling) with that cycle's `no_fit` loss. If loss does not rise with concentration, there is no load-balancing headroom beyond MWW.

**A3. Single-agent DRL composer (Liang 2021, Liu 2022)**

| Assumption | Fit | Note |
|---|---|---|
| One agent, global state, composes the full chain | **Breaks** | Our CTDE routers see local obs. |
| Reward = composite QoS | **Adaptable** | Consumer units filled. |
| Synthetic instances, weak baselines | **Evidence gap** | Gains are only vs DQN / Q-learning, and ACO beat DDPG on steady-state QoS. |

Verdict: no evidence this beats a strong hand rule. It matches our own record: learned routers reached round-robin level, not MWW.

**A4. Exception-triggered re-composition (Wang 2020/2021, Zhao 2025)**

| Assumption | Fit | Note |
|---|---|---|
| Failure is observed (service invalidated) | **Adaptable** | Our agents see outages only through consequences: rejected or expired sub-orders. |
| Central re-plan within time windows | **Breaks** | Re-route locally instead. |

Transferable idea: a local exception memory, i.e. down-weight a producer whose recent sub-orders from this agent failed.

Offline check (cheap; needs recorded traces):
- On recorded shocked episodes (gate / EXP-0057 rows carry `shock_schedule` and traces), count sub-orders that MWW sent to a producer **during its outage**.
- Is `makeable_qty_now` blind to outage? It is computed from capacity and inventory, which an outage does not zero. **Unverified; check the outage handling in `domain/`**.
- Measure the consumer units lost on those sub-orders. If the loss is material, an outage-aware or failure-memory weight is a diagnosed fix. If it is near zero, drop the idea.

**A5. Game-theoretic composition (ESCM 2021, DACO game)**

| Assumption | Fit | Note |
|---|---|---|
| Providers are utility-maximising players with payoffs | **Breaks** | Tokens are not money; providers have no profit. |
| Coalitions form | **Breaks** | No contracting in our domain. |
| Equilibrium computed centrally by metaheuristic | **Breaks** | Not a distributed protocol. |

Adaptable only as a common-payoff or potential-game framing of the router objective. A related repo proposal exists at `/srv/research/tokenomics/openspec/changes/add-m5-potential-game-gnn/` (not read for this note). SCOS game papers offer no reusable mechanism for it.

**A6. Decentralised composition (Block-SC, contract-net agents, MARL-CTDE)**

| Assumption | Fit | Note |
|---|---|---|
| Partition the service/task pool so each peer solves a small sub-problem (Block-SC) | **Adaptable** | Our ordering agents already decide independently; the partition is by requester. |
| Announce-and-bid (contract net) | **Breaks** | Requires same-cycle communication, which our agents lack, and leans into auction design (excluded). |
| MARL-CTDE with a global coordination agent | **Partially holds** | Matches our CTDE. A global coordinator would break the local-info constraint at execution unless it is training-only. |

No evidence strong enough to act on (all [B], no numbers retrieved).

**A7. Multi-tier / correlation-aware composition (Ding multi-level, Qanbari BOM, correlation-aware SCOS)**

| Assumption | Fit | Note |
|---|---|---|
| The platform sees the whole BOM and every tier's providers | **Breaks** | Our ordering agents see only their own order's product. |
| QoS of a choice depends on co-selected upstream services | **Holds in substance** | An assembler's ability to deliver depends on its suppliers' capacity. |

Transferable idea: "deep makeable", a recursive weight equal to own makeable plus what the missing inputs' eligible suppliers could make now, discounted for TTL 2.

Discriminating offline check (run before any build):
- On recorded MWWk rows of EXP-0057 / EXP-0066, take each consumer unit of a **composite** product lost to `no_fit` or expiry.
- Classify whether the chosen assembler was **input-bound** (`input_bottleneck_ratio` < `capacity_bottleneck_ratio`) or **capacity-bound** at decision time.
- For input-bound losses, check whether **another eligible assembler had stock** (then the loss is choice-fixable at tier 1; MWW already sees that, so the loss is draw noise) or whether **no assembler had stock but upstream suppliers had makeable inputs** (only a tier-2-aware weight could help).

Outcome branches:
- (a) Most losses capacity-bound → upstream awareness is irrelevant.
- (b) Most input-bound but choice-fixable at tier 1 → the issue is MWW's randomness, not depth.
- (c) A material share is input-bound with upstream supply available → "deep makeable" is a diagnosed candidate.

The bottleneck ratios are probably not in the recorded CSVs. If they are not, this needs a read-only observer replay like EXP-0066's (`eval/split_need.py` observer pattern), which by CLAUDE.md rules runs on the pod if it is a full sweep.

**Reserving input capacity (from the SCOS / time-window idea)**
- Breaks the whole-order, TTL-2, no-reservation domain contract, and moves into mechanism design.
- Offline proxy: for composite orders that failed because inputs arrived late (expired after TTL 2 with inputs in transit), count how many would have completed with one extra cycle. This only bounds the value of reservation and is not actionable for routers.

### Gaps
- Whether recorded traces carry per-decision bottleneck type (input- vs capacity-bound) and outage status per sub-order is not checked. Inspect the `trace_*` and `sn_*` columns before planning a replay.
- Whether `makeable_qty_now` is zeroed during a producer outage is not verified (check the outage code in `domain/`).
- No SCOS paper reports gains of upstream-aware or correlation-aware selection vs a non-aware baseline in quantitative form that this search could retrieve.

## Q4. What evidence quality do these papers have (real cases vs synthetic instances; baselines)?

### Takeaway
Low to moderate. The field's own 2024 review says QoS data "rely heavily on randomly generated data from simulation experiments, lacking support from real datasets", and that there is no common simulation platform for comparison. Baselines are other metaheuristics or DRL variants, never simple dispatch rules. Reported gains are small and instance-specific: for example, ACO beat DDPG by about 1% steady-state QoS. Decentralised and game-theoretic papers report few or no numbers in their abstracts.

### Cited Findings
- QoS data mostly random simulation, no real datasets. No comprehensive simulation platform, which hinders comparison and reproducibility. Real cases are few: cement equipment (Wang 2018), a flight data recorder (Cai 2019), automotive engine parts (Liu 2022). The QWS dataset is not mentioned — [arXiv HTML](https://arxiv.org/html/2408.01795v1) [A]
- FC-PACO-RM: 3 cases, compared against 3 serial and 7 parallel algorithms (metaheuristics only) — [Peeref](https://peeref.com/works/10291566) [B]
- DDPG (2022): ACO faster and about 0.997% higher steady-state QoS than DDPG; DDPG about 3.249% above DQN — [Bohrium](https://bohrium.dp.tech/paper/arxiv/5ce3a99bced107d4c6560026) [B]
- PD-DQN (2021): baselines DQN and Q-learning only — [Bohrium](https://bohrium.dp.tech/paper/arxiv/6578feab939a5f4082b17a6b) [B]
- Re-composition: SCRIHHO "superior to PSO, GWO"; AMPOWOA beats other optimisers on its own model — [CVTISR](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c328t-3b7f0ea77f7a9fd320160a9e2b147a12d938e13881336bba30b1092ea42443bf3) [B]
- ESCM game: 24-hospital equipping case. The comparison is cooperative vs non-cooperative variants of the same model — [PeerJ](https://peerj.com/articles/cs-410) [B]
- Block-SC, IJPR 2020 logistics-entropy: the abstracts report no numbers; one numerical example — [RePEc Block-SC](https://ideas.repec.org/a/taf/tprsxx/v58y2020i17p5280-5298.html), [RePEc logistics](https://ideas.repec.org/a/taf/tprsxx/v58y2020i7p1987-2015.html) [B]
- Evidence-grade summary per approach:

| Approach | Data | Baselines | Grade of our access |
|---|---|---|---|
| A1 Tao 2013 metaheuristic | cases, likely synthetic [U] | other metaheuristics | B |
| A2 Liu 2017 multi-task | simulation | two task orderings | B |
| A3 PD-DQN / DDPG | synthetic | DQN, Q-learning, ACO | B |
| A4 re-composition (Wang, Zhao) | simulation / own model | PSO, GWO, other optimisers | B |
| A5 ESCM game | 24-hospital case | own variants | B |
| A6 Block-SC / MARL-CTDE | numerical example / unknown | none stated / unknown | B |
| Surveys | — | — | A (Li 2024), B (2018, 2019) |

### Inferences
- No SCOS result supports a claim like "method X beats a capacity-proportional dispatch rule". Simple rules are not benchmarked. Any SCOS idea we adopt is an **untested hypothesis in our setting** and must pass our own offline check, then a pod run against MWW on the atlas worlds.
- The single most useful methodological warning from the field: without a shared simulator and strong baselines, metaheuristic and DRL papers report small, non-comparable gains. Our gate, roster and atlas setup is already stronger evaluation practice than the SCOS literature.

### Gaps
- Could not verify whether the FC-PACO-RM "cases" and the ESCM hospital data are real or synthetic.
- Citation counts and replication status were not gathered.
- Full texts of the 2018 and 2019 surveys were not read; their treatment of dynamic composition is unknown beyond the topic list.
