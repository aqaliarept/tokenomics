# Capacity sharing, maintenance/restoration and disruption resilience in cloud / distributed manufacturing

Evidence grades used throughout:
- **[A]** I read the full text, or a full HTML version, and checked the details.
- **[B]** Only the abstract or publisher landing page was read. Mechanism and claims are taken from it, with no numbers or only partial ones.
- **[C]** Only a bibliographic listing or search snippet. The venue, the results or both are unverified.

Our-domain facts used in the fit assessments come from the task brief and the project glossary. Degradation is a percentage of max capacity per cycle, so it does not depend on load. Restoration and upgrade are bought as investment products from other producers through the investment fund. Orders go whole to one eligible producer. Each producer runs a greedy price-per-capacity-unit auction. TTL is 2. Shocks are capacity hits and outages, never observed in advance. The data we record (per the project CLAUDE.md) includes per-episode `r_start`/`r_end`, `min_capacity_ratio`, `min_spare_fraction`, `investment_rejection_rate`, `consumer_rejection_rate`, `consumer_fulfilled_units_trace`, `resilience.csv` (dip/recovery per shock), routing-tracer `trace_*` columns, and `capacity_trace` from `run_episode`.

## Q1. Which joint production–maintenance or degradation-aware policies (and their decision rules) could inform when and where producers should restore capacity?

### Takeaway
The most transferable ideas are two. First, a shared dual price on the coupling constraint lets each unit decide maintenance and production locally (Rokhforoz & Fink, [A]). Second, maintenance priority follows the current bottleneck: fix the bottleneck, and use its downtime as an "opportunity window" to service other machines (Zhou et al. 2015 and Durandt et al. 2016, [B]). Both make restoration a decision about the system's constraint, not about each unit's own condition. That matches our failure mode, where the middle supply layer starved. Neither paper models maintenance that must be bought from other agents, which is how restoration works in our domain.

### Cited Findings

**Approach 1: Rokhforoz & Fink (2020), distributed joint dynamic maintenance + production scheduling, MPC + Benders + dual decomposition [A]**
- Source: P. Rokhforoz, O. Fink, arXiv:2011.13743, preprint, no venue on the arXiv page — [arXiv abs](https://arxiv.org/abs/2011.13743); full text [ar5iv HTML](https://ar5iv.arxiv.org/html/2011.13743). A search listing ties this line of work to *Journal of Manufacturing Systems* vol. 59 (2021) — [Peeref listing](https://www.peeref.com/works/22103409). That journal version is [C]: I did not confirm it is the same paper.
- **Who decides / objective:** several manufacturing units, each with its own capacity and degradation dynamics, sell production capacity and schedule maintenance by degradation level. They must jointly meet system demand exactly in every period, which is the coupling constraint. Each agent maximises price × production minus quadratic and linear production cost minus a quadratic penalty on its degradation state. The price P(t) is a given profile, not set inside the model. — [ar5iv](https://ar5iv.arxiv.org/html/2011.13743)
- **Degradation model:** x_n(t+1) = (1 − Z_n(t))(A·x_n(t) + B·q_n(t)), so degradation grows linearly with production q_n. Maintenance (Z = 1) resets the state to zero, which is perfect maintenance, and forces production to zero in that period. In the case study A = 1 and B is 0.0031–0.0090. — [ar5iv](https://ar5iv.arxiv.org/html/2011.13743)
- **Method:** a Benders master problem sets every agent's maintenance schedule. Given that schedule, each agent solves its own production problem with distributed MPC. The demand constraint is handled by dual decomposition: the centre broadcasts a multiplier λ(t), agents reply with q_n(t), and λ is updated by subgradient. Only demand, λ, q_n and Benders cut information are shared. Each agent's dynamics and costs stay private, though the privacy claim is not tested. — [ar5iv](https://ar5iv.arxiv.org/html/2011.13743)
- **Evaluation and gains:** 7 units over 196 days. All methods reach more than 90% of optimal. Centralised MPC is about 2.66% better than the distributed method, and the distributed method is "significantly faster", shown only in a figure. Convergence takes "about 55 iterations" as stated, which may be a typo. — [ar5iv](https://ar5iv.arxiv.org/html/2011.13743)
- **Assumptions:** deterministic linear scalar dynamics, no uncertainty in demand or degradation (stochastic extension left as future work), demand met exactly, perfect maintenance, strongly concave subproblems, one case study. — [ar5iv](https://ar5iv.arxiv.org/html/2011.13743)

**Approach 2: Bottleneck-based opportunistic maintenance / maintenance opportunity windows in serial lines with buffers [B]**
- Zhou, Yu, Shao, Trentesaux (2015), *J. Quality in Maintenance Engineering* 21(1):70–88. The bottleneck sets system throughput, so each preventive action on the bottleneck is a chance to maintain other machines. A cost-saving function decides whether to do opportunistic maintenance, depending on whether the machines ahead of the bottleneck also do it. Maintenance is imperfect, and machines are of several types. In a numerical simulation of a series system with buffers, it reportedly beats (i) maintaining each machine separately and (ii) maintaining everything when the bottleneck hits its reliability threshold. No figures are visible without the paywall. — [Emerald](https://www.emerald.com/insight/content/doi/10.1108/JQME-09-2013-0059/full/html)
- Durandt et al. (2016), *South African J. Industrial Engineering*. Machines next to the constraint have locked-in "free time" (maintenance opportunity windows, MOWs), computed from buffer levels and throughput rates. Shown to operators in real time, it lets them schedule preventive maintenance at no throughput cost. Bottleneck downtime adds free time at other stations, and the bottleneck shifts with breakdowns. The same group found deterministic MOWs fragile: buffers are nearly empty when the window opens, so a further stop can break the flow. — [SAJIE](https://sajie.journals.ac.za/pub/article/view/1370); [Stellenbosch repository](https://scholar.sun.ac.za/handle/10019.1/98099) (via search summary)
- A 2025 *J. Shanghai Jiao Tong Univ.* paper uses variable rather than fixed maintenance windows for series-parallel systems with buffers, estimates average buffer WIP, and reports lower total maintenance cost than fixed windows. [C], abstract via search. — [SJTU](https://xuebao.sjtu.edu.cn/EN/abstract/article/1006-2467/49818)

**Survey anchors for joint production–maintenance**
- Geurtsen et al. (2023), "Production, maintenance and resource scheduling: A review", *EJOR* 305(2):501–529. About 250 papers to the end of 2020, the first survey of all three combined. Work integrating all three is sparse. [B] — [IDEAS/RePEc](https://ideas.repec.org/a/eee/ejores/v305y2023i2p501-529.html)
- Older reviews: Budai, Dekker & Nicolai (2006), maintenance and production planning models — [EUR repository](https://repub.eur.nl/pub/8022/ei2006-44.pdf) [C]. Hadidi, Al-Turki & Rahim (2012), integrated production–maintenance–quality — [IDEAS](https://ideas.repec.org/a/ids/ijisen/v10y2012i1p21-50.html) [C].

### Inferences

**Fit of Approach 1 (Rokhforoz & Fink) to our domain, assumption by assumption:**
- *Degradation driven by load (B·q):* **breaks.** Our degradation is a fixed % of max capacity per cycle, whatever the load. Producing more does not wear capacity faster. Run-down in our domain therefore cannot come from load wearing machines. It must come from restoration not being bought or not being fulfilled.
- *Maintenance is a local action costing only lost production:* **breaks.** Our restoration is an investment product made by *other* producers, using their capacity and paid from the shared fund. Maintenance in our domain is itself a cross-agent order that competes in other producers' auctions with consumer orders. This coupling is absent from the paper and is the key structural difference.
- *Perfect maintenance, reset to as-good-as-new:* **adaptable.** Restoration moves capacity toward max, so it is partial in practice.
- *Demand met exactly each period:* **breaks.** We maximise fill subject to run-down and death constraints, and unfilled orders are allowed.
- *Deterministic, known dynamics:* **breaks partly.** The domain is deterministic, but shocks are unannounced and agents see only local information.
- *Shared coupling price λ(t):* **adaptable, and the most useful idea.** Our cut-off price is already a local, per-producer market-tightness signal (glossary). The paper suggests a *system-level* shadow price on the scarce resource, broadcast to everyone, so that each unit's local trade-off between maintaining and producing accounts for system demand. An analog would be a shadow price on each capacity type or product layer, for example from the investment fund's cut-off-price distribution, that producers use to rank restoration against fulfilment.
- *Centralised master for maintenance timing:* **breaks** CTDE and local-only execution. Usable only as a training-time teacher or an analysis tool.

**Fit of Approach 2 (bottleneck-based opportunistic maintenance) to our domain:**
- *Series line with a single throughput-limiting bottleneck:* **adaptable.** Our supply graph is layered with circular dependencies. A bottleneck generalises to the product or capacity type whose available capacity binds first. The "middle supply layer starved" failure is exactly a bottleneck that was not restored.
- *Bottleneck priority for maintenance:* **holds as a decision rule.** Restore the constraint first. In our domain a restoration investment on a non-bottleneck producer spends fund tokens and other producers' capacity without raising throughput.
- *Opportunity window = slack at non-bottleneck stations:* **adaptable.** The analog is spare capacity, which we record as `min_spare_fraction`. Producers with spare capacity this cycle could make restoration products for others at no consumer cost. Ordering restoration from producers with slack is the "maintenance opportunity window" in our terms.
- *Buffers decouple stations:* **partly holds.** Our inventory is limited and penalised, so buffers are thin. The MOW-fragility finding warns that thin buffers make opportunistic restoration windows risky right after a shock.
- *Operators know the line state:* **breaks.** Our agents see only local information. Bottleneck identity would have to be inferred from local signals such as one's own cut-off price, rejection rates or input starvation.

**Offline checks on recorded data:**
- *Bottleneck-restoration check:* from `capacity_trace` per producer per cycle (confirm that `run_episode` keeps per-producer, per-type resolution), identify each cycle's binding product or capacity type: the one with the most `no_fit`/rejected consumer units in the routing trace. Then measure what share of restoration investments landed on producers in that binding layer, comparing `entity_planner` with the failed joint-RL producers. Prediction from Approach 2: `entity_planner` concentrates restoration on the binding layer, and the RL producers do not.
- *Coupling-price check:* compute per cycle the fund-weighted cut-off price per capacity type. Test whether it leads run-down, i.e. whether rising cut-off prices on a type predict a later fall in that type's capacity ratio. If it does, it can serve as the λ-like shadow price for a restoration rule.
- *Slack-window check:* correlate per-producer spare capacity (1 − accepted/capacity) in cycle t with fulfilment of restoration orders placed to that producer in t. If restoration orders sent to slack producers fill more often, routing restoration to slack is an available lever.

### Gaps
- I found no CMfg or shared-manufacturing paper where maintenance is purchased from other network members, as our restoration is. All the joint production–maintenance work found treats maintenance as a local action. This may be a real literature gap; the search was limited.
- Quantitative gains for Zhou et al. 2015 and Durandt et al. 2016 are behind paywalls or were not extracted.
- Whether Rokhforoz & Fink's JMS 2021 version matches the arXiv preprint was not verified.
- The 2025 branch-and-price production–maintenance paper with interdependent degradation ([arXiv 2511.07143](https://arxiv.org/pdf/2511.07143v1)) was found but not read [C].

## Q2. How do CMfg/distributed-manufacturing papers handle provider outages and capacity loss, and with what measured gains?

### Takeaway
CMfg handles disruptions mostly *reactively*. It re-composes or re-schedules services after a provider fails, using metaheuristics or DRL. The main design choice is internal substitution versus transferring the task to another provider. One agent-based study favours external transfer (Zhang & Ren 2023, [B]), but reported gains are mostly qualitative or relative to other metaheuristics. Supply-chain work on proactive redundancy (backup capacity, more complete paths) gives the clearer structural result: redundancy and path count drive resilience. Its evidence is [B]/[C]. I found no CMfg paper with a strong, quantified head-to-head of proactive versus reactive strategies.

### Cited Findings

**Approach 3: Zhang & Ren (2023), task rescheduling with resource substitution in CMfg, multi-agent simulation [B]**
- X. Zhang, D. Ren, *Mathematical Biosciences and Engineering* 20(2):3120–3145, doi 10.3934/mbe.2023147. When CMfg spans many users, service agents and regions, disturbances increase. Task exceptions need fast rescheduling. — [AIMS Press](https://aimspress.com/article/doi/10.3934/mbe.2023147)
- Strategies: *internal transfer* (substitute resources inside the provider) and *external transfer* (to other providers, which depends on logistics distance). A third strategy appears in the figures but is not named on the landing page. — [AIMS Press](https://aimspress.com/article/doi/10.3934/mbe.2023147)
- Setting: a multi-agent simulation of a complex electronic product's service process, three orders, several dynamic environments, averaged over 10 runs. Metrics: QoS and a new "flexibility of service" (FoS) index. — [AIMS Press](https://aimspress.com/article/doi/10.3934/mbe.2023147)
- Result: external transfer had higher QoS and FoS. Sensitivity analysis shows resource-matching rate (internal) and logistics distance (external) strongly affect the indices. No numbers on the landing page. The landing page also does not say whether the disturbances include machine or provider failures. — [AIMS Press](https://aimspress.com/article/doi/10.3934/mbe.2023147); [PDF](https://aimspress.com/aimspress-data/mbe/2023/2/PDF/mbe-20-02-147.pdf)

**Approach 4: DRL rescheduling under machine breakdown in CMfg [C]**
- "Cloud Manufacturing Task Scheduling Under Machinery Breakdown Based on Deep Reinforcement Learning" proposes an LSTM-DoubleDQN scheduler plus a breakdown rescheduling method. Its QoS reportedly beats alternatives, and the rescheduling method is "verified". **Unverified:** the indexing page mixes this abstract with another paper's authors and venue (Dong et al., *CCPE* 2020). — [ScienceGate](https://boch.sciencegate.app/works/W4392942263)
- Wang et al. (2024, *IJPR*), DRL dynamic task scheduling in CMfg, reportedly beats other DRL methods by at most 32.8% (solution quality) and 28.6% (generalisation). [C], from a search summary; the link was not resolved to the primary source.
- Related outside CMfg: dynamic flexible job shop under machine breakdown, Double DQN with attention, *Processes* 2025 — [MDPI PDF](https://mdpi-res.com/d_attachment/processes/processes-13-01246/article_deploy/processes-13-01246.pdf) [C]. Negotiation-based multi-agent RL for reconfigurable manufacturing: breakdowns raise the variability of makespan, throughput and tardiness — [arXiv 2511.07707](https://arxiv.org/pdf/2511.07707) [C].
- CMfg service-composition *reconfiguration* after service exceptions: one model adds occupancy-time and changing-QoS constraints and is solved with an enhanced Harris Hawks optimiser, reportedly beating PSO and GWO — [BUAA](https://research.buaa.edu.cn/en/publications/enhancing-reconfiguration-of-cloud-manufacturing-service-composit/) [C].

**Approach 5: Network redundancy and ripple-effect resilience (supply-chain / production-network literature) [B/C]**
- Ivanov (2017), *IJPR* 55(7):2083–2101. The ripple effect is a disruption spreading from the point of failure through supply, production and distribution, modelled by multi-stage simulation with capacity disruptions. — [IDEAS](https://ideas.repec.org/a/taf/tprsxx/v55y2017i7p2083-2101.html) [B]
- A Monte Carlo study reports that adding backup suppliers and factories to a multiple-allocation network nearly eliminates shortages under severe disruption. Resilience rises with network size, density and the number of *complete paths*. — [Synapse listing](https://synapsesocial.com/papers/W4402128191) [C]
- Agent-based backup-plant redundancy, noting the conflict with lean practice (excess capacity) — [Petra repository](https://repository.petra.ac.id/21287/1/Publikasi1_96007_9484.pdf) [C]. A discrete-event dairy-chain study finds a pre-negotiated emergency outsourcing contract limits service-level loss under production disruption — [JII](https://jii.araku.ac.ir/article_733728_en.html) [C].

**Survey anchors for CMfg scheduling**
- Liu, Wang, Wang, Xu, Zhang (2019), "Scheduling in cloud manufacturing: state-of-the-art and research challenges", *IJPR* 57(15–16):4854–4879, about 158 articles. Compares CMfg scheduling with cloud computing, workshop and supply-chain scheduling. [B] — [IDEAS](https://ideas.repec.org/a/taf/tprsxx/v57y2019i15-16p4854-4879.html)
- Halty et al. (2020), "Scheduling in cloud manufacturing systems: Recent systematic literature review", *MBE*. Focuses on optimisation methods and names decentralised and game-theoretic approaches as promising. [B] — [CAS catalogue](https://katalog.lib.cas.cz/KNAV/EdsRecord/edsdoj,edsdoj.bcc3bd06ec5d405ba17ceb592d68c47d)
- A 2010–2020 CMfg scheduling review argues metaheuristics have "inherent weaknesses" for dynamic CMfg scheduling and names DRL as underexplored — [UTS repository](https://opus.lib.uts.edu.au/handle/10453/162393) [C, attribution via search summary].

### Inferences

**Fit of Approach 3 (internal vs external transfer):**
- *A central platform sees the failure and re-dispatches:* **breaks.** Shocks are unobserved, and ordering agents see only local information. The analog is an ordering agent detecting failure through its own rejected sub-orders and re-routing within TTL 2.
- *External transfer to another provider:* **holds.** This is re-routing to another eligible producer, which our whole-order router can already do in the next cycle.
- *Internal substitution:* **breaks.** Capacity types are not interchangeable, and `produces` is authoritative.
- *Logistics distance as transfer cost:* **breaks.** We have no distance. The closest analog is TTL consumed by a failed attempt.
- *Lesson:* how well re-routing works depends on the *resource-matching rate*, i.e. how many alternative eligible producers have spare capacity. That is our "choice" axis (producers per product) combined with spare capacity.

**Fit of Approach 4 (DRL rescheduling):**
- *Central scheduler with full state:* **breaks.** Most CMfg DRL is a single centralised agent.
- *Breakdown visible to the scheduler:* **breaks.** We have no advance or direct observation.
- These results are not strong enough evidence to borrow a method, only the framing.

**Fit of Approach 5 (redundancy / complete paths):**
- *Backup capacity reserved ahead:* **adaptable.** The analog is keeping spare capacity, which `min_spare_fraction` already measures, or keeping the restoration-producing layer above a capacity floor.
- *Resilience grows with the number of complete paths:* **holds structurally.** Our system death is "some product has no declaring producer with all required types", i.e. zero complete paths. The ripple effect maps onto our cascade through circular dependencies. Complete-path count per product is computable from the config and the capacity trace.
- *Excess capacity conflicts with efficiency:* **holds.** This is the same tension as fill versus run-down.

**Offline checks on recorded data:**
- *Re-routing value after outages:* in `resilience.csv` and the routing trace, for outage shocks, measure the share of orders initially sent to the out-of-service producer that were re-routed and filled within TTL. Compare across worlds with different choice (producers per product). Prediction from Approach 3: dip shrinks as choice × spare capacity grows.
- *Complete-path margin as a death predictor:* per cycle, count for each product the producers that declare it and still hold every required type above some threshold. Test whether the minimum count, or its fall, predicts system death and `min_capacity_ratio` in recorded episodes. If it does, a path-margin feature could drive proactive restoration.
- *Proactive vs reactive:* compare `entity_planner` episodes split by `min_spare_fraction` before the shock, against the shock's `dip_units` and recovery. If more spare capacity before the shock means smaller dips, buffer capacity is the lever, not better reactive routing.

### Gaps
- I found no CMfg paper with quantified gains for provider-outage handling that I could verify at full-text level. Zhang & Ren gives no numbers on the landing page, and the DRL-breakdown paper's authorship is unverified.
- I found no CMfg study comparing proactive (robust/buffered) with reactive scheduling head-to-head under provider outages. The proactive evidence comes from supply-chain simulation.
- Ivanov 2017's quantitative results were not extracted.

## Q3. Which methods coordinate capacity across providers (capacity sharing / pooling) with local information?

### Takeaway
The main CMfg/networked-enterprise line is Renna & Argoneto's game-theoretic capacity sharing (2010–2016). Independent plants with excess demand or excess capacity trade capacity through bilateral, decentralised mechanisms: Nash bargaining, Owen-value cooperative games, Gale–Shapley matching, and a fuzzy engine in CMfg, all implemented as multi-agent systems [C]. Platform-adoption studies show firms use capacity sharing to absorb downtime and use idle capacity [B]. I could not verify quantitative gains. Mechanisms built on prices or incentives are out of scope as a primary topic, so they are only noted here.

### Cited Findings

**Approach 6: Renna & Argoneto, game-theoretic capacity sharing among independent plants [C]**
- 2010, *Int. J. Advanced Manufacturing Technology* 47:1241–1252: Nash bargaining solution for trading capacity among plants — [listing](https://iris.unibas.it/cris/rp/rp03624); the [Springer page](https://link.springer.com/article/10.1007/s00170-009-2254-x) redirected to login and was not read.
- 2011, "Capacity sharing in a network of independent factories: A cooperative game theory approach": a coordination mechanism based on Owen's theorem with a multi-agent architecture, studied under different environmental conditions. Venue unconfirmed: the brief suggested Robotics & CIM, which I could not verify. — [IRIS Unibas listing](https://iris.unibas.it/handle/11563/5451) (403 on fetch; content from search snippet)
- 2013, "Capacity sharing in a network of enterprises using the Gale–Shapley model": stable matching between plants short of capacity and plants with excess — [IRIS listing](https://iris.unibas.it/cris/rp/rp03624).
- 2016, framework for capacity sharing in cloud manufacturing: Gale–Shapley cooperative game plus a fuzzy engine — [IRIS listing](https://iris.unibas.it/cris/rp/rp03624?type=all).
- Renna's research page describes cooperation mechanisms for capacity sharing among networked enterprises, supported by a multi-agent architecture — [Renna research page](https://sites.google.com/site/paolorenna/attivita-di-ricerca). A Palermo repository describes the planning horizons: long-term capacity availability negotiation, medium-term allocation, short-term workload assignment — [UniPA](https://fitforthem.unipa.it/rep:6533b86ffe1ef96bd12cd7d3).

**Platform / practice evidence [B]**
- SINTEF (Arica & Oliveira 2019): interviews with 34 companies, mostly manufacturers, in 5 European countries, on what firms need before adopting B2B capacity-finding and capacity-sharing platforms — [SINTEF Brage](https://sintef.brage.unit.no/sintef-xmlui/handle/11250/2761459).
- Horizontal cooperation platforms let firms "flexibly counteract production downtimes" and use unused machine capacity. Platform goals are understudied — [AIS eLibrary, ICIS 2019](https://aisel.aisnet.org/icis2019/is_heart_of_innovation_ecosystems/innovation_ecosystems/15) [B]. A 3D-printing capacity-trading prototype — [Würzburg](https://www.wiwi.uni-wuerzburg.de/fileadmin/12021300/2019/DIHPOkt19_Towards-Open-Production.pdf) [B].
- A 2024 *IMDS* model: when production or maintenance cost is high, sharing capacity on a third-party platform benefits the equipment supplier, the platform and manufacturers with high earnings per unit of capacity used — [Emerald](https://www.emeraldinsight.com/insight/content/doi/10.1108/IMDS-05-2023-0318/full/html) [B]. Economic and incentive focus, so outside primary scope.

### Inferences

**Fit of Approach 6 (capacity sharing):**
- *Plants trade capacity itself (one plant's machine hours serve another's orders):* **breaks.** In our domain capacity cannot be transferred. Producers can only accept bids for products they declare. "Sharing" exists only implicitly, through the router sending an order to whichever eligible producer has room.
- *Bilateral matching of capacity-short and capacity-rich agents (Gale–Shapley):* **adaptable** for *routing*. A whole-order router that matches orders to eligible producers by spare capacity is a matching problem. Stable matching with local preferences is a known decentralised alternative to price-only routing, and uTWF/MWW are related queue-aware heuristics already tested (EXP-0067 per the project CLAUDE.md).
- *Agents declare excess capacity / shortage to each other:* **breaks partly.** Our agents do not broadcast capacity, only local observations. Any declared-slack signal would be a new observation channel.
- *Objective is profit or cost sharing:* **breaks.** Tokens are not value, and the goal is cooperative fill.
- *Multi-horizon planning (long/medium/short term):* **adaptable.** It maps onto restoration and upgrade (long horizon) versus routing (short horizon), suggesting the two be decided at different timescales.

**Offline check:** per cycle, compute "stranded capacity": spare capacity at producers declaring product p, in cycles where p's consumer orders were rejected elsewhere for `no_fit`. EXP-0066's split-need machinery already classifies `no_fit` units as choice-fixable, split-needed or unfillable. The choice-fixable share is exactly the gain a perfect capacity-matching router could reach. If it is small at the 5% bar used there, capacity sharing through better routing has little to gain, which agrees with the finding that the capacity side, not routing, binds.

### Gaps
- No quantitative results for any Renna & Argoneto paper could be verified. Springer redirected to login and IRIS returned 403.
- I found no capacity-sharing paper where shared capacity is *degrading* and must be restored by the network. Sharing and maintenance appear as separate literatures.

## Q4. Which ideas could explain or fix the finding that order-routing gains cause capacity run-down (coupling routing with restoration)?

### Takeaway
The joint production–maintenance literature makes one consistent point: production and maintenance draw on the same resource, so optimising production alone sacrifices maintenance. The fix is to decide them jointly through a shared price or schedule (Rokhforoz & Fink [A]) or to maintain the bottleneck first (Zhou et al. [B]). In our domain the coupling is sharper. Restoration products are made with other producers' capacity and compete with consumer orders in the same greedy auctions. A router that wins more consumer fill can therefore crowd out restoration orders. This mechanism is an **inference, not yet diagnosed**, and the offline checks below can test it.

### Cited Findings
- Joint production–maintenance models put both decisions in one problem because maintenance removes production time: production is forced to zero in a maintenance period, and the objective trades production revenue against the degradation penalty — [Rokhforoz & Fink, ar5iv](https://ar5iv.arxiv.org/html/2011.13743).
- The coupling between agents is resolved by a broadcast dual price λ(t) on the shared demand constraint, with agents responding locally. Distributed solutions come within about 2.66% of centralised MPC — [ar5iv](https://ar5iv.arxiv.org/html/2011.13743).
- Maintenance timed to the bottleneck, with opportunistic servicing of other machines, beats both independent maintenance and "maintain all at the bottleneck threshold" policies in series systems with buffers — [Zhou et al. 2015, Emerald](https://www.emerald.com/insight/content/doi/10.1108/JQME-09-2013-0059/full/html) (no figures seen).
- Non-bottleneck machines hold "free time" that can be spent on maintenance at no throughput cost. Thin buffers make these windows fragile — [Durandt et al. 2016, SAJIE](https://sajie.journals.ac.za/pub/article/view/1370).
- Integrating production, maintenance and resource scheduling is sparsely studied — [Geurtsen et al. 2023, IDEAS](https://ideas.repec.org/a/eee/ejores/v305y2023i2p501-529.html).
- Ripple-effect work shows a capacity disruption at one node spreads through production layers — [Ivanov 2017, IDEAS](https://ideas.repec.org/a/taf/tprsxx/v55y2017i7p2083-2101.html). This is consistent with a starved middle supply layer cascading into system death.

### Inferences

**Candidate explanations to test (hypotheses, not diagnoses):**
- **H1, auction crowd-out of restoration.** Restoration and upgrade orders are investment orders from producers, backed by fund tokens, bidding in the same greedy auctions as consumer orders. If a better router concentrates high-price consumer orders on the producers that make restoration products, the restoration bids lose and `investment_rejection_rate` rises. Capacity then runs down after a lag. This is our version of "production time removes maintenance time" from Approach 1, except that the competition happens at *another* producer's auction.
- **H2, routing loads the bottleneck layer.** Routing gains come from filling orders at the binding layer, and that layer's degradation is not restored because its own restoration inputs come from a layer the router also loads. This is Approach 2's lesson in reverse: the router "maintains" non-bottlenecks and starves the bottleneck.
- **H3, the fund-distribution feedback.** The fund goes to producers in proportion to their cut-off prices. Routing changes the cut-off prices, which reshapes who can afford restoration. Gains that lower cut-off prices at the restoration-critical producers starve them of funds.

**Fixes suggested by the literature, each needing a diagnosed cause first (project hard rule):**
- *If H1:* a shadow-price coupling in the spirit of Approach 1. Producers, or the router, see a price on capacity reserved for restoration products and decline consumer bids below it. Alternatively, restoration orders get auction priority, the analog of maintenance windows that cannot be pre-empted.
- *If H2:* bottleneck-first restoration (Approach 2). Restoration is ordered for the binding layer first, from producers with slack (the maintenance opportunity window). The router avoids the binding layer's restoration suppliers when alternatives exist.
- *If H3:* change which signal routes the fund. This touches domain or fund rules, which are a human gate.
- *Across all three:* evaluate the router with an objective that includes run-down (`r_end`/`r_start`), as the joint models do with a degradation penalty, not on fill alone.

**Offline checks discriminating H1, H2 and H3 (recorded EXP-0057/EXP-0070/EXP-0071 arm pairs such as TR vs MWW, if per-cycle traces exist):**
- *H1:* per world-episode, paired difference (learned router − MWW) in `investment_rejection_rate` against paired difference in `r_end/r_start`. H1 predicts a positive correlation between rejection increase and run-down. Also check, in the routing trace, whether the rejected investment bids lost at auctions where the competing accepted bids were consumer orders.
- *H2:* identify the binding product or capacity type per cycle (most `no_fit`/rejected consumer units). Compare capacity-ratio trajectories of that layer's producers, and of their restoration suppliers, between arms.
- *H3:* compare per-producer fund receipts, from cut-off price shares, between arms for the producers that make restoration products. H3 predicts lower receipts for those producers under the learned router.
- Before any check, look in `docs/DIAGNOSTICS.md` for an existing entry on run-down or starvation (the project HARD RULE). DIAGNOSTICS 28 was recently "diagnosed" per the commit log and may already cover part of this. I did not read it.

### Gaps
- I found no paper in CMfg or distributed manufacturing that explicitly studies a *router* or dispatcher whose throughput gains cause degradation run-down at a different agent. The coupling literature is single-firm or uses local maintenance.
- H1–H3 are hypotheses derived by analogy. Whether the recorded episode data has per-cycle, per-producer investment rejections and fund receipts was not verified in this research pass. The checks may need a re-run with tracing.
- I did not search the reinforcement-learning-for-maintenance literature, e.g. multi-agent RL for joint maintenance and production, beyond CMfg/DFJSP. It may hold more directly usable decision rules.
