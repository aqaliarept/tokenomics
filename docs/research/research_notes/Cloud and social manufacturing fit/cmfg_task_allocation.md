# Task allocation, scheduling and service selection in cloud manufacturing (CMfg): many deciders, local information, fit to the tokenomics domain

**Evidence grades used below.** A = full text read. B = the abstract or bibliographic record of the primary source was read (OpenAlex, the publisher or the institutional repository). C = only a search-engine snippet or secondary summary was read, so the claim is **unverified**. **No full text could be read in this pass.** ScienceDirect, IEEE Xplore and Taylor & Francis returned 403 or JS-only pages, and Semantic Scholar rate-limited. Every paper card is therefore grade B or C. Instance sizes and percentage gains are missing for most papers, and that is recorded as a gap rather than filled in.

**Local data used for fit checks.** I computed quick means from `/srv/research/tokenomics/experiments/EXP-0057-decoder-check/artifacts/episodes.csv`, which holds 639 rows: arms MWW, MWWk, RR, TS, TR, TW, with the `entity_planner` producer, sizes 12 and 42. These are **my unpaired arm means, not the experiment's pre-registered reading**. They are labelled "local data" wherever they are used.

Abbreviations: MARL = multi-agent RL; CTDE = centralised training, decentralised execution; DA = deferred acceptance; GNN = graph neural network; MWW = makeable-weighted whole routing (our best ordering rule); RR = round robin.

---

## Q1. Which CMfg methods handle many independent decision-makers with local information, and how do they avoid conflicts (herding) without a central coordinator?

### Takeaway
Very little CMfg work is truly decentralised. The field is dominated by a central platform that solves a combinatorial assignment, mostly with bio-inspired metaheuristics (more than 50% of papers per Halty et al. 2020). The "multi-agent" methods that do exist avoid conflicts in one of two ways:
- **Iterative negotiation** inside a decision epoch: contract-net bid, award and reject rounds, or deferred-acceptance proposal and rejection rounds.
- **Centralised training with decentralised execution**, where a shared critic or a "global coordination agent" sees the joint state.

Both rely on something our domain forbids: within-cycle visibility or communication across agents. No CMfg method I found is shown to avoid herding with purely local, one-shot, simultaneous decisions.

### Cited Findings

**Survey 1. Liu, Wang, Wang, Xu, Zhang (2019), "Scheduling in cloud manufacturing: state-of-the-art and research challenges", IJPR 57(15-16):4854-4879. Grade B (abstract).**
- About 158 CMfg scheduling articles had been published over the preceding ~8 years, analysed from Scopus data — [Liu et al. 2019, IJPR](https://doi.org/10.1080/00207543.2018.1449978).
- The survey "systematically summarises" the typical characteristics of CMfg scheduling. It compares CMfg scheduling with cloud-computing, workshop and supply-chain scheduling and lists future challenges — [Liu et al. 2019](https://doi.org/10.1080/00207543.2018.1449978); also [IDEAS record](https://ideas.repec.org/a/taf/tprsxx/v57y2019i15-16p4854-4879.html).
- The specific characteristic taxonomy (centralised vs distributed, splittable vs not, static vs dynamic) and the shares per category were **not readable** (full text paywalled).

**Survey 2. Halty, Sánchez, Vázquez, Viana, Piñeyro, Rossit (2020), "Scheduling in cloud manufacturing systems: Recent systematic literature review", Math. Biosci. Eng. 17(6):7378-7397. Grade B (abstract + landing page).**
- "Methods based on bio-inspired metaheuristics are by far the most widely used (they represent more than 50% of the articles found)" — [Halty et al. 2020](http://www.aimspress.com/article/doi/10.3934/mbe.2020377).
- The authors name multi-objective approaches (conflicting objectives) and "decentralized approaches such as those based on game theory" as the main future lines. This implies decentralised CMfg scheduling was under-developed as of 2020 — [Halty et al. 2020](https://doi.org/10.3934/mbe.2020377).
- Agent and game-theory papers in its reference list include:
  - Liu et al. 2018, multi-agent with dynamic task arrivals
  - Xiao et al. 2019, game-theory multi-task scheduling
  - Bai et al. 2018, public-goods game
  - Liu et al. 2019, iterative double auction
  - Chen et al. 2019, cooperative service booking
  - Zhou et al. 2018, event-triggered dynamic scheduling
  - Zhu et al. 2020, DRL real-time scheduling

  Whether each was in the review sample is not stated — [Halty et al. 2020 landing page](http://www.aimspress.com/article/doi/10.3934/mbe.2020377).

**Approach 1. Multi-agent contract-net scheduling with dynamic task arrivals: Liu, Wang, Wang, Wang, Zhang (2018), Procedia CIRP 72. Grade B (abstract only).**
- *Problem model / who decides:* A multi-agent architecture for CMfg scheduling with dynamic task arrivals. Decisions come from "many-to-many negotiations based on an extended contract net protocol" — [Liu et al. 2018](https://doi.org/10.1016/j.procir.2018.03.138).
- *Method:* Task-side and provider-side agents negotiate. Under contract net, a manager announces, contractors bid, and the manager awards. "Many-to-many" means several tasks negotiate with several providers at once — [Liu et al. 2018](https://doi.org/10.1016/j.procir.2018.03.138).
- *Evaluation:* "Simulation results indicate the feasibility of the model". The abstract reports no baseline and no quantitative gain — [Liu et al. 2018](https://doi.org/10.1016/j.procir.2018.03.138).
- *Information per agent, capacity model, splitting:* not readable (gap).
- *Conflict / herding avoidance:* This is inferred from the contract-net protocol, not read from the paper. Providers bid with their current availability and the task side awards one bid, so two tasks cannot both be promised the same capacity. That requires at least one message round trip before commitment.

**Approach 2. Self-organizing scheduling with MAS + multi-agent DRL: Li, Liu, Li, Gao (2025), J. Manuf. Syst. 79:179-198. Grade B (abstract summary via bibliography page).**
- *Problem model:* Manufacturing resources (machine agents, AGV agents) and a cloud agent are modelled as agents. Production-logistics collaboration runs through a "self-organizing contract network protocol". AGV agents bid on logistics tasks with heuristics, and machine agents bid on processing tasks with multi-agent DRL. A three-priority action space "ensures correct bidding" — [Li et al. 2025 (bib page)](https://gpbib.cs.ucl.ac.uk/gp-html/Li_2025_jmsy.html); [DOI](https://doi.org/10.1016/j.jmsy.2025.01.004).
- *Training:* "centralized training and decentralized execution" for the machine agents — [Li et al. 2025](https://gpbib.cs.ucl.ac.uk/gp-html/Li_2025_jmsy.html).
- *Disturbances:* It handles "new job arrivals and machine breakdowns" — [Li et al. 2025](https://gpbib.cs.ucl.ac.uk/gp-html/Li_2025_jmsy.html).
- *Baselines:* scheduling rules, genetic programming and three DRL methods. The authors report better performance "through competition among heterogeneous resource agents". No numbers were available — [Li et al. 2025](https://gpbib.cs.ucl.ac.uk/gp-html/Li_2025_jmsy.html).
- *Scope note:* the setting is a smart-factory shop floor with a cloud layer, not an inter-enterprise CMfg platform.
- *Decision direction:* This one matters for us. Here the **resource (provider) side bids** for tasks, and the task side does not pick a provider. That is the mirror of our ordering agents.

**Approach 3. Distributed real-time scheduling by distributed dueling DQN (D3QN): Zhang, Yang, Yan, Hu (2022), IEEE TII 18(12):8999-9007. Grade B (repository abstract summary).**
- *Problem model:* Real-time CMfg production scheduling with logistics (AGVs) for dynamic customised orders, modelled as a semi-Markov decision process. The objective is to minimise weighted tardiness — [Zhang et al. 2022 (BIT Pure)](https://pure.bit.edu.cn/en/publications/distributed-real-time-scheduling-in-cloud-manufacturing-by-deep-r/); [DOI](https://doi.org/10.1109/TII.2022.3178410).
- *Who decides:* "distributed actors and one centralized learner" are deployed in the edge and cloud layers. The actors decide, and a single learner trains the shared network — [Zhang et al. 2022](https://pure.bit.edu.cn/en/publications/distributed-real-time-scheduling-in-cloud-manufacturing-by-deep-r/).
- *Gains:* Better weighted tardiness and flow time than "other state-of-the-art algorithms". Baselines and numbers were not readable — [Zhang et al. 2022](https://pure.bit.edu.cn/en/publications/distributed-real-time-scheduling-in-cloud-manufacturing-by-deep-r/).
- *"Distributed" here:* This is Ape-X-style distributed acting for throughput, not independent deciders with conflicting local views. This is my inference from the actor/learner description.

**Approach 4. Two-sided matching game with deferred acceptance for public CMfg platforms: Delaram, Houshmand, Ashtiani, Fatahi Valilai (2021), Procedia Manufacturing. Grade B (abstract).**
- *Problem model:* On public CMfg platforms, service allocation is "a matching game between manufacturing providers and consumers", unlike private platforms, where it is an optimisation matching problem. The solution is the stable outcome of Gale-Shapley deferred acceptance — [Delaram et al. 2021](https://doi.org/10.1016/j.promfg.2021.10.043).
- *Evaluation:* a case study of automotive spare parts in Iran. No baseline numbers are in the abstract — [Delaram et al. 2021](https://doi.org/10.1016/j.promfg.2021.10.043).
- *Conflict avoidance:* This follows from DA itself, not from the abstract. Providers tentatively hold their preferred proposals and reject the rest, and rejected consumers propose to their next choice. Over-subscription is resolved through repeated rounds of proposal and rejection.

**Related (not a paper card, dissertation): Pahwa (2025), "Decentralized Decision Making in Two Sided Manufacturing-as-a-Service Marketplaces", arXiv:2506.12730. Grade B (abstract).**
- It argues current MaaS platforms are centralised and proposes:
  - a reverse auction
  - a stable matching mechanism built with mechanism design and mathematical programming
  - **online matching in a dynamic, stochastic setting where demand and supplier capacity arrive over time**

  It is evaluated by simulating a 3D-printing marketplace. Numbers were not readable — [Pahwa 2025](https://arxiv.org/abs/2506.12730). (The auction and pricing parts are out of scope per the brief.)

**Related, CTDE with a central coordinator: Fang, Ying, Cao, Fang, Lu (2026), "A Multi-Task Service Composition Method Considering Inter-Task Fairness in Cloud Manufacturing", Symmetry 18(2):238. Grade B (abstract).**
- Independent task agents (one per manufacturing task) run under CTDE, plus "a centralized global coordination agent ... to dynamically regulate resource selection probabilities". Two-layer attention covers intra-task subtasks and global resource state — [Fang et al. 2026](https://doi.org/10.3390/sym18020238).
- It explicitly names the failure our domain shows: single-task-centric optimisation causes resource "starvation" when concurrent tasks compete for shared resources. The authors report "significantly" better inter-task fairness at maintained QoS. No numbers were available — [Fang et al. 2026](https://doi.org/10.3390/sym18020238).

### Inferences
- **Every conflict-avoidance mechanism found needs a channel our domain does not have within a cycle:**
  - contract-net award or reject
  - DA proposal and rejection rounds
  - a central coordination agent that rescales selection probabilities (Fang 2026)
  - a centralised critic

  In our domain, the closest structural analogue is the **producer auction plus TTL = 2**. A producer's greedy price-ordered auction is the "award/reject" step. A rejected order can be re-sent next cycle, which is one more DA-like round, at most once. So our system already *is* a truncated DA / contract-net with a 1-round budget. CMfg mechanisms that need more rounds per epoch do not transfer directly.
- **Fit of Approach 1 (contract net) to our domain, assumption by assumption:**

  | Assumption | Holds / breaks / adaptable |
  |---|---|
  | Multi-round negotiation per epoch | **breaks**: one-shot routing, no same-cycle visibility |
  | Providers report availability truthfully in bids | adaptable: `makeable_qty_now` is already the availability signal MWW reads |
  | Whole tasks, no splitting | holds |
  | Capacity static within the horizon | **breaks**: degradation every cycle, capacity hits, outages |
  | One decision layer (task→provider) | **breaks**: our providers are also buyers (inputs, investment), so allocation is coupled to supply |

  *Could it beat MWW?* Only if the negotiation round is emulated by the retry after TTL. That is a change to the environment protocol, not a routing rule. As a pure ordering rule it reduces to "pick by reported availability", which is MWW-like. *Offline check:* measure how many consumer units are lost to `displaced` / `no_fit` on the first attempt and then succeed on the second. This needs per-order traces; episode-level `trace_lost_units_consumer_displaced` and `trace_lost_units_consumer_no_fit` give only an upper bound on what extra negotiation could recover. From local data, MWW at 12 producers loses on average 521.5 units to `displaced` and 303.8 to `no_fit`, against 5,327.8 to `ttl` and 2,105.1 fulfilled per episode. So a better first-shot award would recover at most ~13% of lost units at 12 producers. That is my unpaired mean computation (local data).
- **Fit of Approach 2 (provider-side bidding with CTDE):**

  | Assumption | Holds / breaks / adaptable |
  |---|---|
  | Resources bid for tasks (pull) rather than tasks choosing resources (push) | **breaks** our protocol (ordering agents push whole orders) but is *adaptable* as a thought experiment: our producer auction is already a pull by price |
  | CTDE with shared critic | adaptable: our policies are already CTDE (two shared policies) |
  | Disturbances: arrivals and breakdowns | holds: maps to orders and outages |
  | Shop-floor scale, one firm | **breaks**: we have inter-producer supply dependencies |

  *Could it beat MWW?* Not as a router: it changes who decides. *Offline check:* none possible on recorded data without per-decision provider state.
- **Fit of Approach 4 (DA matching):** Stability needs complete preference lists and repeated rounds. Our TTL gives 2 rounds and our agents have no preference over providers beyond availability. *Could it beat MWW?* Not plausibly as a one-shot rule. *Offline check:* same as for contract net (first-attempt failure vs second-attempt success share).
- **Fang 2026's "global coordination agent that regulates selection probabilities"** is the CMfg analogue of a central bias on MWW's sampling weights. Our domain forbids a central agent at execution. A CTDE critic that only trains is allowed, and its effect would be learned in the routing policy. Our ES residual on top of MWW gained nothing, so this direction has weak prior support here.

### Gaps
- No full text read. For every approach above, the per-agent observation spaces, instance sizes (number of providers or tasks) and quantitative gains are unknown.
- I found **no CMfg paper that evaluates decentralised one-shot routing with no communication against a capacity-proportional random rule like MWW**. Absence after ~25 searches and abstract pulls is weak evidence that such work is rare, not proof that it does not exist.
- Zhou et al. (2021, RCIM), where multiple AI schedulers learn from each other's experience and are tested on rush orders and machine failures, surfaced only as a search snippet ([x-mol record](https://www.x-mol.com/paper/1405011479891488768)). It is unverified, and its venue relevance to inter-enterprise CMfg is unclear.

---

## Q2. Which methods are capacity-aware beyond "proportional to current available capacity" (look-ahead, reservation, load forecasting, degradation-aware)?

### Takeaway
The capacity-aware CMfg rules found go one step past "current capacity". They use **queue state or workload**: the expected completion time from a provider's subtask queue, workload-ordered task priority, and queueing-theoretic load balancing. Some also use **simulation look-ahead to pick among rules in real time** (dynamic data-driven simulation). I found **no CMfg method that models provider capacity degradation or restoration investment**. Local data suggest our binding loss is not the receiving producer's capacity at all: most TTL losses happen while the producer is **short of input products**. That points to *upstream supply* awareness, which CMfg service-composition work treats centrally, not per order.

### Cited Findings

**Approach 5. Real-time scheduling by dynamic data-driven simulation (DDDS): Zhou, Zhang, Ren, Wang (2019), IEEE TII 15(9):5042-5051. Grade B (abstract).**
- *Problem model:* Dynamic CMfg service scheduling for just-in-time delivery. Tasks arrive over time and subtasks go to candidate services — [Zhou et al. 2019](https://doi.org/10.1109/TII.2019.2894111); [IEEE Xplore](https://ieeexplore.ieee.org/document/8618441/).
- *Method:* Three single and three combined scheduling rules "based on service time, logistics time, and subtask queue status of candidate services". A simulation fed with **real-time task and service information** selects the better rule online. Models are built in Simio and connected to a real-time database — [Zhou et al. 2019](https://doi.org/10.1109/TII.2019.2894111).
- *Evaluation:* a numerical-control machining case study. Results are described only as "promising". The abstract gives no instance sizes, baselines or numbers — [Zhou et al. 2019](https://doi.org/10.1109/TII.2019.2894111).
- *Who decides:* the platform, centrally. It uses full real-time state of all services' queues.

**Approach 6. Workload-based multi-task scheduling: Liu, Xu, Zhang, Wang, Zhong (2017), Robot. Comput.-Integr. Manuf. 45 (online 2016). Grade C/B (abstract fragments via search snippet; OpenAlex record without abstract).**
- *Problem model:* Multi-task CMfg scheduling where "task workload provides an important basis for task scheduling". The model includes workload modelling, service efficiency and service quantity — [Liu et al. 2017 (ScienceDirect abstract page)](https://www.sciencedirect.com/science/article/abs/pii/S0736584516303210); [DOI](https://doi.org/10.1016/j.rcim.2016.09.008).
- *Finding:* "scheduling larger-workload tasks with a higher priority can lead to better system performance such as a shorter makespan and higher service utilization" — [Liu et al. 2017](https://www.sciencedirect.com/science/article/abs/pii/S0736584516303210). (Abstract snippet only; the magnitude was not readable.)
- 237 citations per OpenAlex, making it one of the most cited CMfg multi-task scheduling papers — [OpenAlex via DOI](https://doi.org/10.1016/j.rcim.2016.09.008).

**Related: queueing-theory load balancing with logistics: Jafarnejad Ghomi, Rahmani, Qader (2019), Concurrency Computat. Pract. Exper. 31:e5329. Grade B (abstract).**
- It proposes a CMfg queuing system (CMfgQS) and a load-balancing heuristic based on task process times (LBPT). These are embedded in an MILP and solved with a GA for large instances. The authors claim it is among the first CMfg works applying queueing theory to service selection, scheduling and load balancing — [Ghomi et al. 2019](https://doi.org/10.1002/cpe.5329).

**Local data (EXP-0057, my unpaired means per episode; the producer is `entity_planner` throughout).**
- Consumer units lost to TTL expiry dominate every arm. For MWW at 12 producers: 5,327.8 lost to `ttl` vs 2,105.1 fulfilled. At 42 producers: 11,313.1 vs 5,922.4 — [local file](/srv/research/tokenomics/experiments/EXP-0057-decoder-check/artifacts/episodes.csv).
- Of MWW's TTL-lost consumer units, `trace_ttl_lost_units_consumer_inputs_short` = 3,990.3 of 5,327.8 at 12 producers (75%) and 8,677.1 of 11,313.1 at 42 (77%). For RR: 4,786.0 of 5,580.6 (86%) at 12 and 10,648.7 of 11,824.7 (90%) at 42 — [local file](/srv/research/tokenomics/experiments/EXP-0057-decoder-check/artifacts/episodes.csv).
- MWW has *fewer* input-short TTL losses than RR but *more* `displaced` losses: 521.5 vs 315.6 units at 12 producers, 464.8 vs 311.1 at 42 — [local file](/srv/research/tokenomics/experiments/EXP-0057-decoder-check/artifacts/episodes.csv).

### Inferences
- **"Subtask queue status" (Zhou 2019) and "workload" (Liu 2017)** are, in our terms, the producer's already-accepted backlog against capacity. Our auction clears every cycle (TTL = 2, greedy acceptance), so there is no persistent FIFO queue. The closest signal is in-progress accepted orders awaiting inputs. MWW's `makeable_qty_now` is **already backlog-net of current capacity and inventory**, if it accounts for reservations. That depends on the definition in `routing_diagnostics`, which I did not verify. So the step "proportional to capacity → queue-aware" is largely what MWW already is.
- **The diagnostic gap is upstream.** About three quarters or more of lost consumer units die as TTL expiry while the receiving producer is short of inputs. A CMfg idea that targets this is *service composition with precedence*: scheduling a composite task's subtasks jointly so upstream suppliers are booked in time. Every version found is centralised and assumes full visibility of supplier states.
- **The decentralised analogue for us** would be routing by "makeable quantity *including expected inputs within TTL*": a one-cycle look-ahead on input arrival. Equivalently, the *producer-side* ordering agents buying inputs earlier. That is a producer or input-purchasing question as much as a consumer-routing one. It is consistent with PoC-C's interest in role attribution.
- **Fit of Approach 5 (DDDS rule selection) to our domain:**

  | Assumption | Holds / breaks / adaptable |
  |---|---|
  | Central platform with real-time state of all services | **breaks**: local info only |
  | Simulation look-ahead per decision | **breaks at execution** (agents cannot simulate others), *adaptable offline*: use simulation to choose which rule each agent class runs, i.e. a rule portfolio picked per world class |
  | Deterministic or known processing and logistics times | partially holds: domain deterministic, shocks unannounced |
  | Static capacity | **breaks**: degradation, hits, outages |
  | Queue-based providers (FIFO) | **breaks**: we have a per-cycle price auction with TTL |

  *Could it beat MWW?* Possibly as a **rule selector**: pick MWW vs RR vs another rule per world class, offline. That is a portfolio, not a better router. *Offline check:* on EXP-0057's recorded worlds, compute the per-world best arm among {MWW, MWWk, RR} and the gain of the per-world oracle choice over always-MWW (the "selector headroom"). If it is below the noise floor, rule selection cannot beat MWW. Local data hint that RR's mean fulfilled at 12 producers (2,163.3) is not below MWW's (2,105.1) on these worlds. That is an unpaired mean and must be checked against the pre-registered paired reading before use.
- **Fit of Approach 6 (workload priority):** Who sets priority **breaks** the transfer. In our domain the producer's auction orders bids by token price per capacity unit, and the ordering agent cannot reprioritise. The token signal *is* the priority. The adaptable version is producer-side and must not touch domain rules. *Could it beat MWW?* Not as a routing rule. *Offline check:* needs a per-order trace (order size, loss class) to test whether large orders are disproportionately lost to `no_fit`. Not in the episode-level CSV. EXP-0066's `sn_*` columns may partly cover it: `/srv/research/tokenomics/experiments/EXP-0066-split-need`.
- **Degradation-aware allocation:** I found nothing in CMfg. The closest are "machine breakdown" disturbance handling (Li 2025) and fuzzy or robust uncertainty in processing times (references in Halty 2020). None model capacity that wears down every cycle and is restored by buying products from other providers.

### Gaps
- I could not verify how Zhou 2019 defines "subtask queue status", or whether its rules include expected completion time. The full text is needed.
- Reservation or advance-booking in CMfg (Chen et al. 2019, "cooperative service booking", cited in Halty 2020) was not read.
- The definition of `makeable_qty_now` (whether it nets out already-accepted bids and in-transit inputs) is not verified in code. That is the deciding point for whether "queue-aware" adds anything over MWW.

---

## Q3. What MARL / GNN methods exist for CMfg scheduling; do they generalise across numbers of providers or topologies; what baselines did they beat, by how much, on what instances?

### Takeaway
MARL in CMfg is mostly CTDE with agents defined as *tasks*, *subproblems* or *resources*. GNN work is mostly **centralised constructive scheduling**: a heterogeneous task-location graph, or FJSP disjunctive graphs. Reported baselines are other MARL or DRL methods, greedy algorithms, GAs and dispatching rules. **None I found compares against a capacity-proportional stochastic dispatcher like MWW.** Generalisation is claimed (Wang 2022 MARL "generalizability", DOAGNN "excellent generalization", HGAT up to 20 locations) but not quantified in anything I could read. The largest CMfg GNN instance found had **20 locations**, below our 42 producers.

### Cited Findings

**Approach 7. Dynamic CMfg task scheduling with MARL: Wang, Zhang, Liu, Li, Chen, Zhao, Bai (2022), J. Manuf. Syst. 65:130-145. Grade C (search snippets only; OpenAlex has no abstract; ScienceDirect 403).**
- Bibliographic: DOI 10.1016/j.jmsy.2022.08.004, 98 citations per OpenAlex — [DOI](https://doi.org/10.1016/j.jmsy.2022.08.004); [Peeref record](https://www.peeref.com/works/25954479).
- **Unverified (snippet):**
  - The scheduling problem is "formulated and trained by multi-agent reinforcement learning".
  - "Contrast experiments on a case study" are reported to show it beating six other MARL-based scheduling algorithms in scheduling performance and generalizability — search-result summary of [ScienceDirect S0278612522001327](https://www.sciencedirect.com/science/article/abs/pii/S0278612522001327).

  A follow-up search could not reproduce the "group-service" or "six algorithms" wording. Treat the comparison claim as unverified.
- Same group: Wang, Zhang, Liu, Zhao, Wang (2022), "Solving task scheduling problems in cloud manufacturing via attention mechanism and deep reinforcement learning", JMS 65 — single-agent attention DRL — [DOI](https://doi.org/10.1016/j.jmsy.2022.08.013). No abstract was readable.

**Approach 8. Heterogeneous graph attention network (HGAT) for CMfg + logistics: Fomin, Makarov, Voronina, Strimovskaya, Pozdnyakov (2024), IEEE Access. Grade B (abstract).**
- *Problem model:* Task scheduling in CMfg with logistics cost. It uses a heterogeneous graph whose "nodes denote tasks and locations", and it minimises manufacturing plus logistics cost — [Fomin et al. 2024](https://doi.org/10.1109/access.2024.3522020).
- *Gains:* "significant performance improvements over greedy algorithms and comparable results to strong genetic algorithms in large-scale scenarios with up to 20 locations" — [Fomin et al. 2024](https://doi.org/10.1109/access.2024.3522020).
- *Who decides:* a single central policy that constructs the schedule. The training algorithm and capacity model were not readable.

**Approach 9 (adjacent, the FJSP core of CMfg). Dual Operation Aggregation GNN (DOAGNN): Zhao, Zhou, Wang, Cao, et al. (2025), WWW '25. Grade B (abstract).**
- It decouples the disjunctive graph into a machine-operation graph and an operation-precedence graph, with separate aggregation. The reward combines machine efficiency, schedule balance and makespan — [Zhao et al. 2025](https://doi.org/10.1145/3696410.3714616).
- It "outperforms state-of-the-art models and exhibits excellent generalization capabilities" on well-known FJSP datasets. The numbers and size-transfer details were not readable — [Zhao et al. 2025](https://doi.org/10.1145/3696410.3714616).

**Approach 10. Hierarchical RL + attention for CMfg with uncertain events: Zhang, Jiang, Guo, Liu, Hu, Zhang, Deng, Wang (2025), Knowledge-Based Systems 113335. Grade C (search snippet).**
- Multi-objective dynamic scheduling is decomposed into three interdependent subproblems: optimisation objective, manufacturing task, and service selection. Each subproblem is treated as an agent, using attention and pointer networks, and **six types of uncertain events** are handled — search summary of [ScienceDirect S095070512500382X](https://www.sciencedirect.com/science/article/abs/pii/S095070512500382X); [DOI](https://doi.org/10.1016/j.knosys.2025.113335). Unverified beyond the snippet.

**Other single-agent DRL in CMfg (context for baselines):**
- Zhu, Li, Tang, Sun (2020), "Sharer", IEEE Access. It turns multi-resource CMfg scheduling into one learning target. It is "comparable to the latest heuristics", which is parity, not superiority — [Zhu et al. 2020](https://doi.org/10.1109/access.2020.2964955).
- Dong, Xue, Xiao, Li (2020), "RLTS", Concurrency Computat. A DQN assigns precedence-constrained tasks to cloud servers to minimise execution time, and it beat four heuristics in simulation — [Dong et al. 2020 (BJUT repository)](https://ir.bjut.edu.cn/item/3771).

### Inferences
- **Who the agent is matters for the transfer.** CMfg MARL agents are tasks (Fang 2026), subproblems (Zhang 2025) or resources (Li 2025). The paradigm closest to our ordering agents is **one agent per task, choosing services under CTDE with a global coordinator** (Fang 2026). Even there, a centralised coordination agent acts at execution time, which our domain forbids.
- **Baseline strength is the main reason not to expect these methods to beat MWW.** Their reported wins are over greedy algorithms, GAs, dispatching rules and other DRL or MARL variants (and parity with heuristics for Sharer). MWW is a stochastic, availability-proportional, whole-order rule. In queueing terms it is a "power-of-d / join-the-least-loaded" style randomised dispatcher, which is known to be strong under decentralised arrivals. *(This last characterisation is general queueing knowledge, not from a cited CMfg source; treat it as an inference.)* Our own result fits this: an ES-learned per-order router reached RR level but not MWW.
- **Generalisation 12→42:** HGAT's largest instance was 20 locations. No CMfg MARL paper read reports training on one provider count and testing on another. Our scale target is beyond reported CMfg evidence.
- **Fit of Approach 7/10 (CMfg MARL) to our domain, assumption by assumption:**

  | Assumption (typical CMfg MARL) | Holds / breaks / adaptable |
  |---|---|
  | Central platform state available to the critic | adaptable (CTDE training only) |
  | Decentralised execution with local obs | holds for Li 2025 and Fang 2026, except Fang's coordinator |
  | Tasks decomposed into subtasks routed to different services | **breaks for consumer orders**: ours go whole to one producer that does its own decomposition |
  | Static provider capacity within an episode | **breaks**: degradation, hits, outages |
  | Providers are pure servers | **breaks**: our providers are also buyers, and their capacity depends on investment products from peers, so routing has second-order effects on run-down |
  | Objective = makespan, tardiness or QoS | **breaks**: ours is units filled subject to no run-down, no starvation, no system death; MWW-perturbation results show the side effects bind |

  *Could it beat MWW?* Low plausibility. A learned residual on MWW already gained nothing, and these papers give no evidence against an MWW-strength baseline. *Offline check:* a "feature sufficiency" test on routing traces. Regress per-decision acceptance (accepted vs `no_fit` / `displaced` / `ttl`) on the receiving producer's `makeable_qty_now` alone, then add local features a MARL agent could see (recent rejection history per producer, own order size, token price rank). If the added features do not raise held-out predictive power beyond noise, no local-observation policy has information MWW lacks. **This needs per-decision logs.** The episode-level `episodes.csv` columns do not support it, so it requires a traced replay. EXP-0057's dispersion reading (`history_differs` vs `history_flat` in `/srv/research/tokenomics/experiments/EXP-0057-decoder-check/dispersion.csv`) is the closest recorded evidence on whether agent-local history carries contrast.
- **Fit of Approach 8/9 (GNN):**
  - Centralised constructive scheduling: **breaks** at execution.
  - A heterogeneous graph of tasks and providers: *adaptable* as a per-agent encoder over the eligible-producer set. This is what our M5 potential-game + GNN proposal would need.
  - Size generalisation: unproven in CMfg beyond 20 nodes.

  *Could it beat MWW?* Not supported by any CMfg evidence found. *Offline check:* none on recorded data. Any GNN router needs the same feature-sufficiency test first.

### Gaps
- Agent definitions, observation and action spaces, and instance sizes for Wang 2022 (JMS 65:130-145) could not be read. This is the most cited CMfg MARL paper found and should be pulled through institutional access.
- No CMfg MARL or GNN paper read reports cross-size (provider count) transfer with numbers.
- No CMfg paper found tests against a randomised availability-proportional dispatcher.

---

## Q4. Which assumptions (central scheduler, full information, splittable tasks, static capacity, deterministic processing) break in our domain?

### Takeaway
Across the approaches found, the assumptions that break most often are:
1. **a central scheduler or coordinator at execution**, in nearly all metaheuristic, DDDS and GNN work and in Fang 2026's coordinator;
2. **static provider capacity**;
3. **providers as pure servers** with no supply dependency on peers;
4. **makespan, tardiness or QoS objectives** instead of fill-without-run-down.

Whole (unsplit) assignment generally **holds**, since CMfg usually assigns each (sub)task to one service. Determinism partly holds: our domain is deterministic, but shocks are unannounced. The decentralised CMfg mechanisms (contract net, DA matching) keep local information but need **within-epoch multi-round communication**, which our one-shot routing with TTL = 2 does not allow.

### Cited Findings
- More than 50% of CMfg scheduling papers use bio-inspired metaheuristics, which implies a central optimiser over a known instance. Decentralised (game-theoretic) approaches are named as future work — [Halty et al. 2020](https://doi.org/10.3934/mbe.2020377).
- Private CMfg platforms treat allocation as an optimisation matching problem. Public platforms are a two-sided matching game — [Delaram et al. 2021](https://doi.org/10.1016/j.promfg.2021.10.043).
- Current MaaS platforms are centralised, and decentralisation is argued for transparency — [Pahwa 2025](https://arxiv.org/abs/2506.12730).
- Single-task-centric optimisation ignores competition among coexisting tasks and causes "starvation". Traditional heuristics need full re-execution for new scenarios — [Fang et al. 2026](https://doi.org/10.3390/sym18020238).
- DDDS needs a real-time database of all tasks and services feeding a simulator — [Zhou et al. 2019](https://doi.org/10.1109/TII.2019.2894111).
- The distributed DRL is actors at the edge with **one centralised learner** — [Zhang et al. 2022](https://pure.bit.edu.cn/en/publications/distributed-real-time-scheduling-in-cloud-manufacturing-by-deep-r/).
- Uncertainty handling in CMfg is via event types such as new arrivals and machine breakdowns ([Li et al. 2025](https://gpbib.cs.ucl.ac.uk/gp-html/Li_2025_jmsy.html)), or via six uncertain-event types ([Zhang et al. 2025, snippet, unverified](https://www.sciencedirect.com/science/article/abs/pii/S095070512500382X)). None found models per-cycle capacity degradation with restoration investment.

### Inferences

**Assumption matrix** (H = holds in our domain, B = breaks, A = adaptable; "?" = not readable):

| Approach (grade) | Central decider at execution | Full / global info | Splittable task | Static capacity | Deterministic times | Providers pure servers | Multi-round comms per epoch | Objective matches ours |
|---|---|---|---|---|---|---|---|---|
| 1 Contract net, Liu 2018 (B) | H (decentral) | H (local bids) | ? | ? (likely B) | ? | B | **B** (needs rounds) | ? |
| 2 MAS+DRL bidding, Li 2025 (B) | H (CTDE) | A | H (whole jobs) | B (breakdowns modelled) | ? | B | **B** (contract net) | B (shop KPIs) |
| 3 Distributed D3QN, Zhang 2022 (B) | A (actors decide, central learner) | ? | ? | ? | ? | B | H | B (tardiness) |
| 4 DA matching, Delaram 2021 (B) | H (mechanism) | B (full preference lists) | H | B | — | B | **B** | B |
| 5 DDDS rules, Zhou 2019 (B) | **B** | **B** | subtasks | B | partly | B | n/a | B (JIT delivery) |
| 6 Workload priority, Liu 2017 (C/B) | **B** | **B** | subtasks | ? | ? | B | n/a | B (makespan, utilisation) |
| 7 MARL, Wang 2022 (C) | ? (MARL) | ? | ? | ? | ? | ? | ? | ? |
| 8 HGAT, Fomin 2024 (B) | **B** | **B** | ? | ? | ? | B | n/a | B (cost) |
| 9 DOAGNN, Zhao 2025 (B) | **B** | **B** | ops on machines | B (FJSP static) | H | B | n/a | B (makespan) |
| 10 HRL, Zhang 2025 (C) | B (hierarchical) | ? | subtasks | B (uncertain events) | B | ? | n/a | multi-objective |
| Fang 2026 CTDE fairness (B) | **B** (coordinator agent) | A | subtasks | ? | ? | B | n/a | partial (fairness ≈ our starvation) |

- **Structural mismatch common to all:** In our domain, routing a consumer order changes which producers burn capacity, which in turn changes who buys restoration products from whom. Allocation and capacity dynamics are therefore coupled through the investment market. This explains why "perturbing MWW worsened run-down/starvation". No CMfg allocation method found models this coupling. Most CMfg scheduling is a one-sided assignment over fixed providers.
- **Most transferable idea:** not a router, but the **diagnosis frame**. Fang 2026's "inter-task fairness to prevent starvation" and Zhou 2019's "queue status" correspond to two of our side-effect clauses (starvation, run-down). Local data say the dominant loss is TTL expiry with inputs short, which is an upstream-supply problem. That suggests the next routing gain, if any, depends on producers' input procurement. This agrees with the role-attribution framing in PoC-C. It is an inference from my unpaired means and should be confirmed with paired, per-world numbers.
- **Plausibility ranking for beating MWW** (my judgement, given the evidence above):
  1. A **rule selector** (DDDS-style, offline per world class): plausible only if per-world oracle headroom among existing rules exceeds the noise floor. Testable now on EXP-0057 rows.
  2. **Input-aware look-ahead availability** (makeable within TTL including expected input arrivals): plausible in principle because it targets the dominant loss class. It needs per-decision traces, and probably producer-side changes.
  3. **CMfg MARL/GNN routers**: low. No evidence against MWW-strength baselines, and our own learned residual on MWW gained nothing.
  4. **Contract-net / DA negotiation**: requires protocol changes, i.e. extra rounds, which change the environment, not the policy. Out of scope for an ordering rule.

### Gaps
- No full text read, so the matrix contains "?" wherever the abstract was silent. Splittability and the capacity model are the most often unknown.
- No CMfg paper found models provider capacity that degrades and is restored through purchases from other providers. This could be a real gap in the literature or an artefact of abstract-level search.
- The local-data loss decomposition uses unpaired arm means over different numbers of rows per arm (MWW R = 2 vs RR R = 1). A paired, per-world version and the definitions of the `trace_*` loss classes should be checked before any decision rests on them.
