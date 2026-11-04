# Social manufacturing (SocialM) and decentralised manufacturing paradigms: task/order allocation and capacity coordination among many autonomous providers, with fit to the tokenomics domain

Scope note. Sources were read at three depths, and each claim is graded by depth:
- **[FT]** full text read: Szaller/Egri/Kádár 2020 (IJCIM), Szaller/Kádár 2021 (CIRP Annals), Ghahramani/Qiao/Zhou 2026 (arXiv HTML).
- **[ABS]** abstract or bibliographic record only: the Jiang/Leng/Guo SocialM line, ManuChain, Makerchain, Shi et al. 2024, the Xiong et al. survey.
- **[SNIP]** search snippet only: Sari et al. review table, Kaihara 2017, MDPI Systems 2023. Treat these as unverified.

Excluded by constraint (pricing, incentive or mechanism design as the primary topic): Chida et al. 2020, strategyproof matching in crowdsourced manufacturing ([IJAT](https://www.fujipress.jp/ijat/au/ijate001400050734/)); Chen et al. 2024, Stackelberg task-service matching ([Omega](https://ideas.repec.org/a/eee/jomega/v129y2024ics0305048324001221.html)); Stackelberg order allocation in SocialM ([Peeref](https://www.peeref.com/works/17530419)).

Throughout, "OUR" refers to the tokenomics domain: whole-order routing by per-agent ordering agents, a per-producer greedy auction, TTL 2, capacity degradation, shocks, 12/42 producers, MWW and entity_planner as the best rules.

---

## Q1. What coordination mechanisms does SocialM (and its close relatives) propose for many autonomous, heterogeneous providers, and are any evaluated quantitatively against baselines?

### Takeaway
The core SocialM school (Pingyu Jiang, Jiewu Leng, Wei Guo and colleagues, Xi'an Jiaotong) proposes four mechanism families: similarity-based community clustering (GHSOM) followed by multi-objective order allocation; relationship mining on interaction networks; blockchain or smart-contract self-organisation with an upper-level optimiser; and order-driven multi-agent software. In the sources found, this school validates by case study or prototype; no baseline-controlled numbers were retrievable. The quantitatively evaluated work on the same problem (autonomous providers sharing capacity for incoming orders) comes from the adjacent "distributed / crowdsourced manufacturing" line at SZTAKI. That line shows trust/reputation-weighted partner selection cutting average lateness about 23% against a no-history baseline, and a platform-matching variant raising utilisation about 30% at the cost of 4–5% service level. A 2026 arXiv paper on decentralised bidding shows that a learned *ranking* correction on top of a heuristic bid beats both the heuristic and a learned *imitation* of it.

### Cited Findings

#### Approach A: Trust- and reputation-based resource sharing (Szaller, Egri, Kádár 2020) [FT], quantitative
- Citation: Szaller, Á., Egri, P., Kádár, B. "Trust-based resource sharing mechanism in distributed manufacturing", *Int. J. Computer Integrated Manufacturing* 33(1):1–21, 2020, DOI 10.1080/0951192X.2019.1699257 — [SZTAKI record](https://eprints.sztaki.hu/9871/); [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf); [DOI](https://doi.org/10.1080/0951192X.2019.1699257)
- **Problem model.** Autonomous company agents receive external orders. Agents with resource shortages request resources from all other federation members, divide requests among several agents, ask partners to reorganise production, and may cancel undertaken tasks. Who decides: the requesting agent picks among offers. Information: its own subjective trust per partner plus a public reputation per partner kept by a Federation Centre (FC). Objective: service level, meaning tasks completed on time, not throughput — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)
- **Coordination mechanism.** Offer score = equally weighted sum of unit price and trust/reputation. Trust and reputation are exponentially smoothed ratings of past task success and due-date keeping: trust_new = rating·β + trust_prev·(1−β) and rep_new = rating·γ + rep_prev·(1−γ), with β = γ = 0.1 and initial value 80 on a 0–100 scale. Trust is asymmetric. Cancellations are penalised. Requests that get no offer are split into up to d parts — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)
- **Evaluation setting.** AnyLogic multi-agent simulation. 10 companies: 6 "non-reliable" (mean processing overrun 20%, 20% cancellation rate) and 4 "reliable" (5% overrun, 2% cancellation). 20 resource types, 750 model time units, 50 runs per setting, 95% CIs reported. All unit prices are equal, so the baseline ("normal model") effectively picks the first received offer — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)
- **Reported gains.**
  - Average task lateness: 8.52 with no trust or reputation, 6.56 with reputation only, 6.80 with trust only, 6.62 with both. That is roughly a 23% reduction, and reputation alone is about as good as both; 95% CIs are 0.03–0.04 time units — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)
  - The gain grows with federation size (10→100 agents) and shrinks under high load: "if the load of the federation increases, taking trustfulness into account has less impact" — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)
  - Splitting requests: with no division 15–16% of tasks fail; with division into 3 parts failures fall to 6%, with no further gain beyond 3 parts — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)
  - Load concentration: under the baseline all companies run at 40–50% utilisation; with trust, reliable companies' utilisation is clearly higher (Fig. 11, values read from the figure only qualitatively) — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)
  - Shock-like experiment: a reliable agent's reputation and trust are dropped to 10 at t = 300. Reputation recovers in about 100 time units. Utilisation takes "hundreds" of time units, because the subjective trust each partner holds recovers more slowly than the public reputation — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)
- **Assumptions.** Reliability is a *static* agent trait, independent of workload. Agents are honest. Prices are equal. Jobs run FCFS. The order stream is fixed. One job needs one resource type. Partners can "reorganise" up to 20% extra capacity — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)

#### Approach B: Platform vs direct-exchange resource sharing (Szaller and Kádár 2021) [FT], quantitative
- Citation: Szaller, Á., Kádár, B. "Platform and direct exchange-based mechanisms for resource sharing in distributed manufacturing: a comparison", *CIRP Annals* (2021; year from the first author's thesis reference list; volume/pages not verified) — [OA PDF](https://eprints.sztaki.hu/10244/1/Szaller_407_32090056_ny.pdf); year per [Szaller PhD thesis, ref. 124](https://eprints.sztaki.hu/10639/1/Szaller_1_34502438_ny.pdf)
- **Problem model.** The same federation as in A. Each facility decides on its own whom to outsource to.
  - Model A (direct exchange): broadcast requests and receive offers. Selection uses trust and reputation. Splitting uses equal-sized fragments, because the requester does not know partners' free capacity.
  - Model B (platform): facilities publish offers of free capacity (40-unit lookahead) to the FC. The FC pre-filters offers that meet the request and combines up to 3 offers fitted to actual free capacity. Offers are anonymous, so only reputation, not trust, is usable. The final choice stays with the requester — [OA PDF](https://eprints.sztaki.hu/10244/1/Szaller_407_32090056_ny.pdf)
- **Evaluation setting.** AnyLogic. 10 facilities (6 non-reliable, 4 reliable). 20 resource types; each facility has 10–20 types at 8–12 units each. One order every 1.5 time units per facility: 8 units for 20 time units. Federation size swept from 3 to 50, 10 runs each, 500 time units — [OA PDF](https://eprints.sztaki.hu/10244/1/Szaller_407_32090056_ny.pdf)
- **Reported results.**
  - The platform gives about 30% higher average utilisation at every federation size, attributed to capacity-fitted fragment combination.
  - Direct exchange gives about 4–5% higher service level beyond size 10, because with lower utilisation the reliable partners have spare capacity.
  - Platform message count is an order of magnitude lower; direct exchange grows superlinearly (quadratic fit R² 0.999) — [OA PDF](https://eprints.sztaki.hu/10244/1/Szaller_407_32090056_ny.pdf)
  - The authors call this "a kind of trade-off between resource utilization and service level" — [OA PDF](https://eprints.sztaki.hu/10244/1/Szaller_407_32090056_ny.pdf)

#### Approach C: SocialM community formation plus order allocation (Guo and Jiang 2018; Jiang book chapters) [ABS], case study only
- Citation: Guo, W., Jiang, P. "An investigation on establishing small- and medium-sized enterprises communities under the environment of social manufacturing", *Concurrent Engineering: Research and Applications* 26(3):251–264, 2018, DOI 10.1177/1063293X18770499 — [XJTU record](https://scholar.xjtu.edu.cn/zh/publications/an-investigation-on-establishing-small-and-medium-sized-enterpris/)
- **Mechanism, step 1.** Describe SMEs' socialised manufacturing resources, cluster *similar* resources with a growing hierarchical self-organising map (GHSOM), select suitable resource communities and map them to SMEs — [XJTU record](https://scholar.xjtu.edu.cn/zh/publications/an-investigation-on-establishing-small-and-medium-sized-enterpris/)
- **Mechanism, step 2.** SMEs jointly complete orders. Order allocation uses product cost and delivery time as indicators and is solved by a multi-objective algorithm — [XJTU record](https://scholar.xjtu.edu.cn/zh/publications/an-investigation-on-establishing-small-and-medium-sized-enterpris/)
- **Evaluation.** One case from a professional printing firm, "to derive rules for practical production". No baselines or numbers are visible in the abstract — [XJTU record](https://scholar.xjtu.edu.cn/zh/publications/an-investigation-on-establishing-small-and-medium-sized-enterpris/)
- The book chapter version states that "different order allocation strategies lead to different forms of communities", with product cost and delivery time as selection indicators — [Springer: Social Manufacturing: Fundamentals and Applications](https://www.springerprofessional.de/social-manufacturing-fundamentals-and-applications/15839686); [chapter record](https://scholar.xjtu.edu.cn/en/publications/social-business-relationship-and-organizational-network/)
- A review table reportedly lists a variant pairing GHSOM with a "modified multi-objective bird swarm algorithm" for SME clustering and order allocation [SNIP, unverified] — [Sari et al. 2023 PDF](https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-d103bf3b-16b0-45a4-a476-2e87d8abbf8a/c/Sari_i_in._mspe-2023-0003.pdf)
- **Who decides.** In this formulation a platform-level optimiser runs the clustering and allocation using global resource descriptions. This is inferred from the abstract's two-step centralised procedure; providers do not decide locally.

#### Approach D: Blockchain / smart-contract self-organisation (Leng, Jiang et al.: GDUT 2017, Makerchain 2019, ManuChain 2020) [ABS], prototype only
- **2017 position paper.** A centralised platform cannot meet prosumer communities' self-organising needs. The proposed blockchain configuration covers decentralised consensus, smart contracts, multi-form typology and dynamic evolution of prosumer relationships — [J. Guangdong Univ. Technol. 34(5):1–9](https://xbzrb.gdut.edu.cn/EN/Y2017/V34/I05/1)
- **Makerchain** (Leng, Jiang, Xu, Liu, Zhao, Bian, Shi; *J. Cleaner Production* 234:767–778, 2019, DOI 10.1016/j.jclepro.2019.06.265). A blockchain manages "cyber-credit" (trust and transactions) among makers, plus a chemical-signature anti-counterfeit scheme. It is demonstrated with a DApp in which "clustered makers" self-organise around personalised demands. The abstract shows no quantitative evaluation — [XJTU record](https://scholar.xjtu.edu.cn/en/publications/makerchain-a-blockchain-with-chemical-signature-for-self-organizi/)
- **ManuChain** (Leng, Yan, Liu, Xu, Zhao, Shi, Wei, Zhang, Chen; *IEEE TSMC: Systems* 50(1):182–192, 2020, DOI 10.1109/TSMC.2019.2930418; 247 citations per OpenAlex). An iterative bi-level model. Lower-level smart contracts "proactively decentralize fine-grained and individualized task execution among machine tools" via Raspberry Pi gateways. The results feed an upper-level digital twin that runs "coarse-grained holistic optimization". The stated goal is removing "unbalance/inconsistency between holistic planning and local execution". Evaluation is a prototype; no baselines are visible in the abstract — [OpenAlex record](https://api.openalex.org/works/doi:10.1109/TSMC.2019.2930418); [ResearchGate PDF (403 to fetch)](https://researchgate.net/profile/Jiewu-Leng/publication/335007418_ManuChain_Combining_Permissioned_Blockchain_With_a_Holistic_Optimization_Model_as_Bi-Level_Intelligence_for_Smart_Manufacturing/links/5e631c4ca6fdcc37dd0b505f/ManuChain-Combining-Permissioned-Blockchain-With-a-Holistic-Optimization-Model-as-Bi-Level-Intelligence-for-Smart-Manufacturing.pdf)
- **Tokens.** In this line, tokens and credit act as a trust/credit ledger and a contract-execution substrate, not as a demand-priority signal like OUR tokens.

#### Approach E: Order-driven multi-agent SocialM software model (Shi, Dong, Yang, Guo, Jiang 2024) [ABS], simulation demo
- Citation: *J. Computing and Information Science in Engineering* 24(8):081003, 2024, DOI 10.1115/1.4065016. Communities "emerge autonomously through business interactions, resource self-organization, and production information sharing, all driven by product orders". It is implemented with multi-agent plus MQTT, business-interaction software as the control point and an "industrial app store", and tested with a 3D-printing simulation. Sizes, task-allocation rule and results were not retrievable — [XJTU record](https://scholar.xjtu.edu.cn/en/publications/a-novel-industrial-software-model-for-social-manufacturing/)

#### Approach F: Relationship mining on SocialM interaction networks (Leng and Jiang 2017, 2018) [ABS], real data, but descriptive only
- Leng and Jiang 2017 (*IEEE TSMC* 47(2):276–288, DOI 10.1109/TSMC.2016.2623630) extract relationships from text interaction contexts, then match group-level relationships, using loopy belief propagation on a multi-attributed graph — [XJTU record](https://scholar.xjtu.edu.cn/en/publications/mining-and-matching-relationships-from-interaction-contexts-in-a-/)
- Leng and Jiang 2018 (*IJPR* 56(15):5131–5146, DOI 10.1080/00207543.2018.1430903) analyse a real collaborative-manufacturing-network dataset (industrial printing machinery, from a web-based producing/outsourcing platform). They compare "relational capability" (interaction-based) with "node capability" (attribute-based) metrics and test how far they correlate. Network size and coefficients are not visible in the abstract — [EconPapers](https://econpapers.repec.org/RePEc:taf:tprsxx:v:56:y:2018:i:15:p:5131-5146)

#### Approach G (adjacent, not SocialM-branded): Learned ranking correction on decentralised bidding (Ghahramani, Qiao, Zhou 2026) [FT], quantitative
- Citation: "Edge-AI-Driven Learning-to-Rank for Decentralized Task Allocation in Circular Smart Manufacturing", arXiv:2605.16433 (v4, 27 Oct 2026; under review at IEEE IoT J.) — [arXiv abs](https://arxiv.org/abs/2605.16433); [HTML](https://arxiv.org/html/2605.16433)
- **Problem model.** Machines bid for arriving tasks using local information only: queue time, processing time, load, a slack-based urgency term and a contention proxy for one shared, unit-capacity, reusable resource. The task-originating agent picks the minimum bid; there is no global scheduler. Objective, in lexicographic order: tasks completed, then deadline misses, tardiness and energy — [HTML](https://arxiv.org/html/2605.16433)
- **Mechanism.** Three stages: (1) an analytical heuristic bid b = αq + βp + δυ + ηℓ + λχu(t); (2) a regression model imitating the bid; (3) a 337-parameter pairwise ranker whose centred, clipped score is *added* to the heuristic bid (ψ = 20). The ranker is trained on counterfactual preference pairs labelled by downstream outcomes — [HTML](https://arxiv.org/html/2605.16433)
- **Results** (45 tasks, 30 held-out seeds; deltas computed from the reported table):
  - At p_r = 0.7: completed tasks 19.03 → 21.17 (+11.2%), tardiness 22.15 → 19.44 (−12.2%), miss rate 0.79 → 0.74.
  - At p_r = 0.9: completed tasks 14.73 → 15.73 (+6.8%).
  - The regression stage is neutral (19.07 and 14.70 completed).
  - Paired p < 0.05 for completed tasks, tardiness and misses; the energy gain is not claimed significant — [HTML](https://arxiv.org/html/2605.16433)
- **Key claim.** "Minimizing absolute prediction error is neither necessary nor sufficient to improve allocation performance" unless the candidate ordering changes. Under a min-bid rule any monotone transform gives the same allocation — [HTML](https://arxiv.org/html/2605.16433)
- **Limits.** One shared resource, gains shrinking under heavier contention, no full ablation, machine count not stated in the text read — [HTML](https://arxiv.org/html/2605.16433)

#### Other quantitative lead, not verified
- Kaihara, Katsumura, Suginishi, Kádár 2017, "Simulation model study for manufacturing effectiveness evaluation in crowdsourced manufacturing", *CIRP Annals* 66(1):445–448, DOI 10.1016/j.cirp.2017.04.094. Per snippets it is an agent-based negotiation simulation evaluated on delivery and machine usage. The full text is restricted; details unverified — [SZTAKI record](https://eprints.sztaki.hu/id/eprint/9275)
- SocialM crowdsourced task scheduling framed as a flexible job shop over decentralised "social manufacturing units" (MDPI *Systems* 11(8):383, 2023) [SNIP; fetch failed] — [MDPI](https://www.mdpi.com/2079-8954/11/8/383)

#### Surveys
- Xiong, Tamir, Shen, Shang, Wu, Wang, "A Survey on Social Manufacturing: A Paradigm Shift for Smart Prosumers", *IEEE Trans. Computational Social Systems* 10(5):2504–2522 (online 2022), DOI 10.1109/TCSS.2022.3180201 [ABS]. It covers the definition and theory of SM, comparison with other paradigms, enabling technologies (blockchain, 3-D printing, big data), applications, challenges and trends. It frames SM as the manufacturing solution of the CPSS / "Society 5.0" era — [OpenAlex](https://api.openalex.org/works/doi:10.1109/TCSS.2022.3180201); [KFUPM record](https://pure.kfupm.edu.sa/en/publications/a-survey-on-social-manufacturing-a-paradigm-shift-for-smart-prosu/)
- Sari et al. 2023, "Social Manufacturing on Integrated Production System: A Systematic Literature Review", *Management Systems in Production Engineering* [SNIP]. A Scopus-based mapping by concepts, frameworks and technologies — [UGM repository](https://ir.lib.ugm.ac.id/id/eprint/411/)
- Jiang (ed.), *Social Manufacturing: Fundamentals and Applications*, Springer, the reference monograph — [Springer Professional](https://www.springerprofessional.de/social-manufacturing-fundamentals-and-applications/15839686)

### Inferences

These are fit assessments against OUR domain, assumption by assumption.

**A. Trust/reputation selection**
- *"Provider quality is a stable latent trait learnable from history"*: **breaks.** OUR producers do not differ by a fixed lateness distribution. Whether a sub-order is accepted depends on current capacity, the competing bids in that cycle's auction, inputs and shocks, so "reliability" is state-dependent and non-stationary. Szaller's own result that gains shrink under high load points the same way, since in tight worlds the reliable providers saturate.
- *"Requester sees offers before committing"*: **breaks.** OUR ordering agent commits a whole sub-order blind, with no offer round. The analogue is acceptance feedback after the fact, within TTL 2.
- *"Local history usable"*: **holds.** OUR ordering agents could keep a per-producer EWMA of acceptance or fill.
- *"Equal prices, honest agents"*: **adaptable.** OUR producers are cooperative and tokens are not prices.
- *Herding risk*: **live concern.** Their Fig. 11 shows trust concentrating load on reliable providers. In OUR domain concentration on high-acceptance producers is the herding / run-down pathway, and learned corrections on MWW already worsened run-down and starvation.
- *Net reading*: an EWMA acceptance score would most plausibly help only as a *tie-breaker within MWW's draw*, not as a replacement.
- *Splitting*: their 16% → 6% failure reduction from splitting into 3 is the closest published analogue to OUR split-vs-whole question, but it comes with explicit free-capacity mismatch and no auction.

**B. Platform vs direct exchange**
- MWW's use of each producer's makeable quantity *now* is functionally a "published offer" (Model B's capacity visibility) without the combination step. Szaller's result that capacity-fitted matching beats blind equal fragments by about 30% in utilisation is consistent with MWW beating blind rules such as round-robin in OUR data. That is an inference across domains, not a tested equivalence.
- Their utilisation-vs-service trade-off parallels OUR filled-orders-vs-run-down tension: packing load onto the best providers raises throughput but depletes their slack.
- *"FC can combine fragments"*: **breaks** under OUR whole-order rule.
- *"Anonymous offers"*: irrelevant to OUR domain.

**C. Community clustering plus allocation**
- *"Communities group similar resources"*: **holds trivially.** OUR `produces` sets already define the substitute groups per product, so GHSOM similarity clustering adds nothing to eligibility.
- *"A central multi-objective optimiser allocates orders"*: **breaks.** OUR agents are local-only. The nearest local analogue is already the LP / B2 ceiling bounds used in the atlas, which are upper references rather than deployable rules.
- *"Cost and delivery-time objectives"*: **adaptable** to fill and run-down.

**D. Blockchain self-organisation**
- The bi-level idea, local self-organisation plus periodic coarse global re-optimisation, **breaks** OUR "agents see only local information" premise as a deployable policy.
- It is **adaptable** as a diagnostic. ManuChain's "holistic vs local inconsistency" maps to OUR gap between MWW and the LP-based ceiling (B_lp) in the headroom atlas.
- Ledger/credit tokens are a trust record. They are not comparable to OUR emission/burn demand tokens.

**E/F. Order-driven software model and interaction-network mining**
- Neither gives an allocation rule testable in OUR domain. F's "relational capability" (interaction history) vs "node capability" (attributes) contrast mirrors history-based vs state-based routing signals.

**G. Learned ranking correction**
- *"Local features only"*: **holds.**
- *"Single-step task, one shared resource, no degradation"*: **breaks.** OUR routing choices have delayed effects through capacity degradation, investment and system death, which is where OUR learned corrections on MWW failed (run-down and starvation).
- *"Correction added to the heuristic, trained on counterfactual outcome-labelled pairs"*: **adaptable.**
- The paper's lesson matches OUR history: imitation reaches heuristic level only (OUR learned routers at round-robin level), and gains require changing the ordering. Its labels are short-horizon outcome comparisons, which OUR multi-cycle side effects would make unsafe without run-down/starvation terms in the label.

**Offline checks on OUR recorded data**

All of these are cheap, local and read-only; verify which columns exist before use.
1. *(A) Trust signal: does history add information?* Using a recorded run with the routing tracer, such as EXP-0057 or EXP-0066 `episodes.csv` with `trace_*` columns, compute per (ordering agent, producer, product) a lagged EWMA (β = 0.1) of sub-order acceptance. Regress or rank-correlate next-cycle acceptance on that EWMA *conditional on* the producer's `makeable_qty_now` (MWW's signal). Read: no incremental predictive power means trust cannot help beyond MWW. The per-sub-order outcomes may not be stored per producer; if not, this needs a tracer re-run, which falls under the pod rules.
2. *(A/B) Herding cost of concentration.* In the same traces, measure the per-producer share of routed units against subsequent capacity-ratio decline, to test whether routing toward high-acceptance producers precedes their run-down. This quantifies the downside Szaller observed as utilisation concentration.
3. *(G) Ordering check for any learned correction.* For previously trained corrections on MWW, measure on recorded decisions how often the correction changes the argmax or draw-mode producer, against the change in fill and run-down. This applies the "ranking, not regression" lesson as a diagnostic.
4. *(D) Bi-level headroom.* MWW vs B_lp per atlas world is already recorded in the headroom atlas. It bounds what an upper-level holistic optimiser could add.

### Gaps
- No full text was obtained for any Jiang/Leng/Guo SocialM paper. Whether Guo and Jiang 2018, ManuChain or Shi 2024 report baseline comparisons with numbers is **unverified**; only abstracts were read.
- Kaihara et al. 2017 (agent-based negotiation in crowdsourced manufacturing) could not be read; its mechanism and numbers are unknown.
- The CIRP Annals volume and pages for Szaller and Kádár 2021 were not verified; the year comes from the author's thesis.
- No source found evaluates SocialM-style coordination under capacity *degradation* or *shocks* comparable to OUR outages and hits. Szaller's reputation-shock experiment is the only dynamic-disruption analogue, and it perturbs beliefs, not capacity.

---

## Q2. Which mechanisms use only local or historical information (reputation, past performance, community membership), directly comparable to OUR agent-local history?

### Takeaway
Only the SZTAKI trust line (A, B) and the arXiv ranking paper (G) are both local-information and quantitatively evaluated. A uses each requester's own interaction history (trust) plus a centrally aggregated public reputation, and found reputation, which pools all agents' history, about as good as reputation plus trust and better than private trust alone. G uses only instantaneous local state. The SocialM community and blockchain works rely on global clustering or upper-level optimisation (C, D) or a shared ledger (D).

### Cited Findings
- In Szaller et al., private trust alone gave lateness 6.80, public reputation alone 6.56, and both 6.62. Reputation "more accurately determines the reliability of an agent because it is calculated on the base of a higher number of tasks" — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)
- After a reputation shock, subjective trust "increasing much slower than the public reputation", so the agent's utilisation recovered only after hundreds of time units. Private history is slow to re-learn — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)
- The smoothing factor sets "a trade-off between appreciating the positive long-term performance and penalizing the temporary bad performance" — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)
- In the platform model, anonymity removes private trust and only public reputation is used — [OA PDF](https://eprints.sztaki.hu/10244/1/Szaller_407_32090056_ny.pdf)
- G's machines use queue time, processing time, load, slack urgency and a contention proxy, all local, with no historical memory described — [HTML](https://arxiv.org/html/2605.16433)
- Makerchain keeps maker "cyber-credit" on a blockchain, a shared history ledger rather than a local one — [XJTU record](https://scholar.xjtu.edu.cn/en/publications/makerchain-a-blockchain-with-chemical-signature-for-self-organizi/)
- Leng and Jiang 2018 contrast interaction-history metrics with attribute metrics on real platform data, which is descriptive and not used for allocation — [EconPapers](https://econpapers.repec.org/RePEc:taf:tprsxx:v:56:y:2018:i:15:p:5131-5146)

### Inferences
- Mapping to OUR domain: private trust = an ordering agent's own record of its sub-orders to each producer. Public reputation = a pooled acceptance record across all ordering agents, which would *break* OUR agent-local-information premise unless treated as an environment-provided signal.
- Szaller's finding that pooled history beats private history suggests a private-only EWMA in OUR domain would be noisy. Each consumer's ordering agent sees few sub-orders per producer per product over a 100-cycle episode, with TTL 2. This is an inference; check sample counts per (agent, producer, product) in recorded traces before building it.
- MWW's makeable-quantity signal is *current state*, which A's static-trait world lacks. In a non-stationary system, state plausibly dominates history. Offline check 1 in Q1 is the discriminating test.

### Gaps
- No SocialM paper was found that evaluates *purely local, decentralised* history-based routing at scale (≥ 40 providers) against a capacity-aware heuristic. Szaller's largest sweeps (50 and 100 agents) duplicate 10 agent profiles.

---

## Q3. Do community/cluster formation ideas (grouping providers that make complementary products) translate to OUR multi-tier supply (composite products)?

### Takeaway
Mostly no, as evidenced. The SocialM community work clusters *similar* resources (substitutes), not complementary chains, and allocates centrally. "Complementary" grouping appears only as concept ("clustered makers self-organise around personalised demands", "order-driven dynamic community"), without a mechanism or evaluation found. Szaller's own future work names multi-resource, interdependent-job orders as not yet modelled.

### Cited Findings
- Guo and Jiang cluster "similar resources" with GHSOM, then allocate orders across SMEs — [XJTU record](https://scholar.xjtu.edu.cn/zh/publications/an-investigation-on-establishing-small-and-medium-sized-enterpris/)
- The book chapter groups "similar prosumers" into one community via GHSOM — [Springer Professional](https://www.springerprofessional.de/social-manufacturing-fundamentals-and-applications/15839686)
- Leng and Jiang (2019 book chapter) on "generation of order-driven dynamic community of mass individualization" [SNIP; content not read] — [chapter record](https://scholar.xjtu.edu.cn/en/publications/social-business-relationship-and-organizational-network/)
- Shi et al. 2024: communities emerge "driven by product orders" through business interaction and information sharing — [XJTU record](https://scholar.xjtu.edu.cn/en/publications/a-novel-industrial-software-model-for-social-manufacturing/)
- Szaller and Kádár 2021, future work: "more complex orders containing interdependent jobs that may require different types of resources" — [OA PDF](https://eprints.sztaki.hu/10244/1/Szaller_407_32090056_ny.pdf)
- In Szaller et al. 2020, each job requires a single resource type — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf)

### Inferences
- In OUR domain, substitute groups are given by `produces`. Complementary structure is given by process sheets: composite products' input purchases plus investment products (restoration and upgrade) bought from other producers. A "community" would be a supplier–assembler subgraph.
- No reviewed mechanism shows that routing *within* such a subgraph beats MWW.
- A plausible adaptation, untested anywhere found, is to route upstream input purchases preferentially to producers whose downstream assemblers are current customers, which stabilises supply chains. It would trade against MWW's capacity-proportional spreading.
- Offline check: from traces, compute the share of composite-product `no_fit` or rejection losses attributable to input shortage at the assembler versus capacity at the assembler. Only if input-supply losses are material would chain-aware ("community") routing have a target. Check `docs/DIAGNOSTICS.md` for an existing entry on input-starvation before building a new diagnostic.

### Gaps
- No evaluated complementary-community allocation mechanism was found in SocialM. Whether the Leng and Jiang "order-driven dynamic community" chapter contains one is unknown; the full text was not read.

---

## Q4. Where is the evidence conceptual only vs quantitatively validated?

### Takeaway
SocialM-branded literature is overwhelmingly conceptual, architectural, case-study or prototype work. The surveys present SM as a paradigm and a set of enabling technologies, and the SocialM-community IEEE call itself says basic CPSS definitions are still developing. Quantitative, baseline-controlled evidence on allocation among autonomous providers is found in the adjacent distributed/crowdsourced-manufacturing simulation line (SZTAKI) and in decentralised task-allocation ML (arXiv 2026), and it remains small-scale: 10 base agents, single-resource jobs, no degradation.

### Cited Findings

| Work | Evidence grade | Detail |
|---|---|---|
| Guo and Jiang 2018 | Case study | Validated by "a case from a professional printing firm"; no numbers in the abstract — [XJTU record](https://scholar.xjtu.edu.cn/zh/publications/an-investigation-on-establishing-small-and-medium-sized-enterpris/) |
| ManuChain 2020 | Prototype | Permissioned blockchain with Raspberry Pi gateways; no metrics in the abstract — [OpenAlex](https://api.openalex.org/works/doi:10.1109/TSMC.2019.2930418) |
| Makerchain 2019 | Demonstration | DApp demonstration — [XJTU record](https://scholar.xjtu.edu.cn/en/publications/makerchain-a-blockchain-with-chemical-signature-for-self-organizi/) |
| GDUT 2017 | Conceptual | Blockchain configuration proposal — [JGDUT](https://xbzrb.gdut.edu.cn/EN/Y2017/V34/I05/1) |
| Shi et al. 2024 | Simulation demo | 3D-printing simulation; no metrics retrieved — [XJTU record](https://scholar.xjtu.edu.cn/en/publications/a-novel-industrial-software-model-for-social-manufacturing/) |
| Leng and Jiang 2018 | Real data, descriptive | Platform dataset; metric comparison, not allocation — [EconPapers](https://econpapers.repec.org/RePEc:taf:tprsxx:v:56:y:2018:i:15:p:5131-5146) |
| Xiong et al. survey | Survey | Definition, theory, technologies, applications, challenges; no meta-analysis of quantitative results evident from the abstract — [OpenAlex](https://api.openalex.org/works/doi:10.1109/TCSS.2022.3180201) |
| Szaller et al. 2020 | Quantitative simulation | 50 runs/setting with CIs; lateness 8.52 → 6.56–6.80 — [OA PDF](https://eprints.sztaki.hu/9871/1/Szaller_1_31011885_ny.pdf) |
| Szaller and Kádár 2021 | Quantitative simulation | 10 runs × 48 sizes per model; about 30% utilisation, 4–5% service level, 10× messages — [OA PDF](https://eprints.sztaki.hu/10244/1/Szaller_407_32090056_ny.pdf) |
| Ghahramani et al. 2026 | Quantitative simulation | 30 held-out seeds, paired significance; preprint, not yet peer-reviewed — [HTML](https://arxiv.org/html/2605.16433) |

- The IEEE TCSS call "Social Manufacturing after ChatGPT" notes that definitions of basic CPSS concepts are still developing — [IEEE SMC CFP](https://www.ieeesmc.org/wp-content/uploads/2023/07/CFP-Special-Issue-on-Social-Manufacturing-after-ChatGPT.pdf)

### Inferences
- For OUR purposes the actionable, evidence-backed ideas are narrow: (i) history-weighted provider choice, with a known herding/concentration side effect and a gain that shrinks under high load; (ii) capacity-visible matching beats blind splitting, which MWW already embodies; (iii) learned corrections must be judged by whether they change the ordering, not by fit.
- None was validated in a setting with capacity degradation, investment loops, TTL, or system death. Transfer to OUR domain is therefore a hypothesis for the offline checks above, not an expected gain.
- Per the project's root-cause rule, none of these should be proposed as a next strategy without naming a diagnosed cause. The fitting diagnosed cause would be "MWW misallocates in a way that history predicts", which offline check 1 tests.

### Gaps
- No head-to-head SocialM benchmark exists across mechanisms (community, trust, blockchain, platform) on a common simulator.
- No SocialM study at 40+ heterogeneous providers with multi-tier products and disruptions was found.
- Whether the Jiang school's journal papers, beyond the abstracts read, contain quantitative baseline comparisons remains open; full texts were behind paywalls or returned 403.
