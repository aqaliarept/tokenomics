# EXP-0115 pre-registration: why A1 acceptance is harmful (task 4.1; frozen before data)

**Question.** EXP-0098 found A1 (`ttl_feasible`: also accept a bid that fits the next cycle after the queued claims)
`harmful` at 12 producers. Its offered mechanism (claims dying with their order) was refuted by the traced counters
(EXP-0098 correction, 2026-11-06). Which mechanism costs the units? The arms are diagnostics only and never
candidates (user rule 2026-11-06).

**Setup.**
- Same worlds and stacks as EXP-0098: `supply_trace --order-ttl 2 --producer entity_planner_buyahead_k5_rf5`,
  orderers MWWk and RR, 12 and 42 producers, horizon 100, the observer on.
- Code at the commit that pins this note; secure 8 vCPU pod, EUR-IS-1, watchdog, stop 6 vCPU-h. Tranche M6-A.

**Arms (`--acceptance`):**
- `auction`: today's auction as a policy. Its rows must equal EXP-0088's default rows on every column but
  `acceptance_stats` (neutrality). It also gives the default's cut-off trace.
- `ttl_feasible` (A1): its rows must equal EXP-0098's A1 rows on every column but `acceptance_stats`
  (reproduction).
- `ttl_feasible_life` (A1-life): A1, but accept-later only when the bid's order has a later cycle
  (`remaining_ttl ≥ 1`). This tests the information hypothesis (the order's age).
- `ttl_feasible_new_first` (A1-new-first): A1's decisions, but this cycle's accepted bids run before the carried
  pending claims. This tests the displacement hypothesis (step 3b runs pending claims first).

**Measured** (per size × orderer; `read_exp0115.py`):
- units and paired differences per world (EXP-0098's rule: the per-world mean over (episode, replicate); the atlas
  v2 noise floors from EXP-0073 ttl2);
- deaths and run-down failures;
- consumer units lost per supply-trace group and class (A1 − default);
- **displacement:**
  - new bids accepted-but-waiting or declined that would have run had this cycle's bids taken the budget before
    the carried claims did (`displaced_*`);
  - the share of them priced above the cheapest claim that ran;
  - their mean price against the claims that ran;
- **cut-off / fund:** the mean cut-off (lowest accepted price per producer-cycle with an acceptance), declines below
  the cut-off, and the fund share moved against the default (EXP-0098's measure);
- **late inputs:** b2_late units, and the replenishment lag by Little's law (Σ inbound units / Σ delivered units
  over producer-cycles).

**Validity.** The run is invalid, with no reading, if any of these holds:
- a contract failure;
- a missing or extra row against EXP-0088's row set;
- a demand or shock digest that differs from EXP-0088's;
- the `auction` neutrality or the A1 reproduction fails.

**Reading** (12 producers, MWWk decides; RR and 42 reported):
- **Loss and recovery:**
  - L = mean_w(default − A1). The default is EXP-0088's k5_rf5 rows, equal to the `auction` arm under neutrality.
  - For arm X in {life, new_first}: R_X = mean_w(X − A1) / L, the share of A1's loss the arm recovers.
- **Order of the rules:**
  1. `not_reproduced` if L ≤ mean_w(floor_w), that is, A1 is not beyond the floor below the default in this run.
  2. `lifetime` if R_life ≥ 0.5 and R_new_first < 0.5.
  3. `displacement` if R_new_first ≥ 0.5 and R_life < 0.5.
  4. `both` if both are ≥ 0.5.
  5. Otherwise `neither:<group>/<class>`. The group carrying the largest share of A1's extra lost consumer units
     (A1 − default, summed over episodes) is named, with its largest class.
- **Always reported, so the reading has its evidence, never gating:**
  - the full R table;
  - the extra-loss decomposition by group and class;
  - the displacement counters (`displaced_*`), the cut-off and fund-share shift, and the lag.
- The reading names the cause and adds a DIAGNOSTICS entry.
