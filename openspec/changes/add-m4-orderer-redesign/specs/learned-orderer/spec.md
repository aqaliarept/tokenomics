# Spec Delta

## Purpose

Defines how the learned ordering policy chooses producers for each order, what information it may
use, how its choices are decoded and validated, how it is trained on whole-episode outcomes, and
how each step of the M4 ordering ladder is judged before the next one starts.

## ADDED Requirements

### Requirement: Orderer information set
An ordering agent's decision in cycle t SHALL depend only on what was written before cycle t's ordering phase began, its own orders, its own outcome history and its own sampling stream. The earlier writes include previous cycles' outcomes and cycle t's consumer orders and producing phase. Nothing written by any ordering agent in cycle t, the agent's position in the agent order, hidden shock state or any diagnostic reason field SHALL be an input (ADR-0058).

#### Scenario: Earlier ordering agents' same-cycle routing is invisible
- **WHEN** the actions of the ordering agents that act before agent a in the same cycle are permuted or altered
- **THEN** agent a's policy inputs (shared rows, base scores, order features, own-turn features, agent-local history) and its sub-orders are byte-identical

#### Scenario: Last cycle's load only
- **WHEN** an orderer reads a producer's requested load during the ordering phase
- **THEN** the value is the one written by the previous cycle's auction, and it does not change while ordering agents route in the current cycle

### Requirement: Agent-local history
For each eligible producer, the learned orderer SHALL receive the agent's own past bid outcomes with that producer, as moving averages. These cover routed, accepted, completed, failed (dropped and expired merged) and rejected units, plus completion latency and recency. The history SHALL be built only from outcome kinds, never from a reason field. It SHALL be reset at episode start, folded only after a cycle completes, and kept beside the unchanged shared rows.

#### Scenario: Different histories, different inputs
- **WHEN** two agents of the same product have routed different units to a producer in past cycles
- **THEN** their history inputs for that producer differ while their shared producer rows are identical

#### Scenario: Reasons scrambled
- **WHEN** the reason strings of every rejection and failure in an episode are scrambled or removed
- **THEN** every agent's history is byte-identical to the unscrambled episode's

#### Scenario: Outage not revealed
- **WHEN** one sub-order is rejected because its producer is in an outage and another for lack of room, or one reservation fails by TTL during an outage and another is dropped
- **THEN** each pair is counted identically, as rejected or as failed

### Requirement: One shared network without agent identity
Every ordering agent SHALL run the same portable network with the same parameters, and no agent identifier SHALL be a model input. Differences between agents' choices SHALL come only from their own orders, their agent-local history and their sampling streams. An agent identifier MAY key a sampling stream.

#### Scenario: Empty histories still differ
- **WHEN** two agents of the same product with empty histories each hold one order at the first cycle of an episode
- **THEN** their producer draws come from different sampling streams and may differ

#### Scenario: Relabelled agents
- **WHEN** the ordering agents are relabelled with the same histories and sampling keys reassigned accordingly
- **THEN** each agent's decision is unchanged

### Requirement: Per-order producer choice
The learned ordering policy SHALL make one producer choice per order, deciding an agent's orders one at a time in a fixed, recorded order. Each decision SHALL see the units the agent's own earlier orders of the same turn assigned to each producer.

#### Scenario: Own earlier orders visible
- **WHEN** an agent holds two orders in one turn and the first is assigned to producer p
- **THEN** the second decision's input shows the first order's units on p

#### Scenario: Fixed decision order
- **WHEN** the same state is replayed with the same parameters and sampling keys
- **THEN** the orders are decided in the same sequence and give byte-identical sub-orders

### Requirement: Whole orders only in step 2
Every choice of a step-2 or step-3 learned orderer SHALL route the whole order to one eligible producer. Splitting is out of scope for M4 step 2, as EXP-0056's `joint` reading attributes the gap to whole orders with varied receivers. A split action SHALL return only through a separate change backed by a diagnosis that a split recovers units whole routing cannot.

#### Scenario: Whole choice
- **WHEN** the policy chooses producer p for an order of quantity q
- **THEN** exactly one sub-order of quantity q, carrying all of the order's tokens, goes to p

#### Scenario: No split emitted
- **WHEN** a step-2 policy is played on any world-episode with any parameters
- **THEN** every order it routes has exactly one sub-order

### Requirement: Validity layer
The choice decode SHALL check only validity. Shares SHALL go only to eligible producers, be non-negative and sum to one. Quantities SHALL follow from the shares exactly and sum to the order quantity, and tokens SHALL sum to the order's tokens. An invalid choice SHALL be a contract failure, never silently rerouted or rescaled.

#### Scenario: Ineligible producer chosen
- **WHEN** a choice gives a share to a producer that does not declare the order's product
- **THEN** the job is recorded as a contract failure row and the run is invalid

#### Scenario: Conservation under bursts
- **WHEN** any valid choice is decoded for an order of any quantity, including a burst of 5–10× the usual size
- **THEN** the action stays within its bounds and the sub-orders conserve the order's quantity and tokens exactly

### Requirement: Legacy decode unchanged
The legacy logits decode SHALL remain the default. Without the choice decode, every existing checkpoint, baseline and router SHALL give byte-identical actions and `episode_stats`. The decode setting SHALL be part of the env config fingerprint and of every run's pins. A composite per-order checkpoint SHALL carry its own decoder version.

#### Scenario: Legacy checkpoint replay
- **WHEN** the legacy-invariance fixtures run after the choice decode is added
- **THEN** every golden passes unedited

#### Scenario: Composite checkpoint in a legacy env refused
- **WHEN** a per-order checkpoint is loaded into an env configured for the legacy decode
- **THEN** loading is refused, naming the decoder version

#### Scenario: Frozen base loads
- **WHEN** a per-order policy is built over the legacy-versioned EXP-0012 scorer
- **THEN** the scorer loads as the policy's frozen base component

### Requirement: Decoder check precedes training
Before any orderer training run, the current trained scores SHALL be played with one producer drawn per order and each order routed whole. They SHALL be compared with the trained orderer, `round_robin` and a `makeable_weighted_whole` drawn with the same keys. Readings SHALL use only pre-registered 12-producer worlds, with a fixed precedence. The licence SHALL come from the primary reading alone.

#### Scenario: No gap takes precedence
- **WHEN** fewer than 5 of the 12-producer worlds have a trained-vs-`round_robin` gap above their floor
- **THEN** the primary reading is `no_gap` whatever the recovered shares, and the cause is diagnosed

#### Scenario: Decoder is the bottleneck
- **WHEN** there is a gap and the sampled whole decode recovers at least 2/3 of it on at least 2/3 of the gap worlds
- **THEN** the primary reading is `decoder` and per-order training is licensed

#### Scenario: Decoder not the bottleneck
- **WHEN** there is a gap and the median recovered share is below 1/3
- **THEN** the reading is `not_decoder`, no training starts, and the cause is diagnosed first

#### Scenario: Fidelity readings both hold
- **WHEN** the sampled decode falls below the keyed `makeable_weighted_whole` beyond the floor on at least 1/3 of worlds and exceeds it beyond the floor on at least 1/3
- **THEN** the fidelity reading is `mixed` and the licence is unchanged

#### Scenario: Agent histories do not differ
- **WHEN** under the sampled decode the across-agent dispersion of own completion rates or of routed shares stays below its pre-registered threshold on more than 1/3 of the worlds
- **THEN** the dispersion reading is `history_flat` and the user is asked before step-2 code starts

### Requirement: Outcome training by bounded ES
Per-order orderer training SHALL optimise whole-episode outcomes with a bounded ES residual over the frozen deployed scores. Population members SHALL share world-episodes and sampling keys within a generation. Arms, seeds, scales, budget, fitness and readings SHALL be recorded before the first ES run. No per-decision label SHALL be used unless it passes label stability.

#### Scenario: Zero residual equals the decoder check
- **WHEN** the residual parameters are zero
- **THEN** the policy's choices equal the decoder check's sampled whole decode for the same keys

#### Scenario: Unregistered change refused
- **WHEN** an ES run is launched with arms, seeds, scales or rules differing from the recorded pre-registration
- **THEN** the run refuses to start as a real run

### Requirement: Relative disqualification
An ES population member SHALL be disqualified on a world-episode only when it violates a guard that the zero-residual control does not violate there. The guards are death, run-down below X and consumer starvation below Y. A disqualified member SHALL take the worst rank. Training worlds on which the control dies SHALL be replaced before training, and too many replacements SHALL stop the run.

#### Scenario: Control also starves a consumer
- **WHEN** a member and the zero-residual control both leave a consumer below Y on a world-episode
- **THEN** the member is not disqualified for that consumer

#### Scenario: No feasible start
- **WHEN** more than a quarter of the candidate training worlds need replacement because the control dies, or fewer than 50 feasible worlds can be drawn
- **THEN** the run stops as `no_feasible_start` and goes to the user

### Requirement: Disqualification noise margin
Under pre-registration v2, a run-down or starvation flag SHALL also require the member to fall below the control's value on that world-episode by more than a registered margin, one per flag kind, from the control's re-keyed spread (EXP-0068). Death SHALL keep no margin. The margin SHALL apply only to training-time disqualification.

#### Scenario: Member within the control's re-keyed noise
- **WHEN** under pre-registration v2 a member's run-down ratio is below X, the control's is not, and the control's ratio exceeds the member's by no more than the registered run-down margin
- **THEN** the member is not disqualified for run-down on that world-episode

#### Scenario: Member beyond the control's re-keyed noise
- **WHEN** under pre-registration v2 a member leaves a consumer below Y, the control does not, and the control's fill of that consumer exceeds the member's by more than the registered starvation margin
- **THEN** the member is disqualified on that world-episode

#### Scenario: v1 unchanged
- **WHEN** a run loads pre-registration v1
- **THEN** disqualification applies no margin and its results are byte-identical to before

### Requirement: Larger-step probe arm
Under pre-registration v2 the gradient-signal probe SHALL also play A1 at twice its perturbation scale on the same noise directions, minibatches and draw keys. A pre-registered step reading SHALL compare the two arms with the probe's thresholds. The larger-step arm SHALL be diagnostic only and SHALL NOT be trained.

#### Scenario: Only the larger step finds a direction
- **WHEN** A1's cosine is below 0.1 and the larger-step arm's is at least 0.1
- **THEN** the step reading is `flat_at_sigma`, no pilot launches and the step size goes to the user

#### Scenario: Both steps find a direction
- **WHEN** both cosines are at least 0.1
- **THEN** the step reading is `heterogeneity` and training proceeds at the registered step with minibatch 24

### Requirement: Re-based step 2 on MWW
Under pre-registration v3 the per-order policy's frozen base SHALL be MWW's keyed whole-order choice, with the residual added to its log-weights, and the zero-residual control SHALL draw exactly as MWWk. Only arm A1 SHALL train. The bar SHALL be recomputed for that control, and a run SHALL refuse a bar whose control played another base.

#### Scenario: Zero residual equals MWWk
- **WHEN** the residual parameters are zero under pre-registration v3
- **THEN** the policy's sub-orders, episode statistics and draws equal MWWk's for the same keys

#### Scenario: Reported arm stays out of the reading
- **WHEN** a v3 validation read plays the registered reported arm (EXP-0060's A1-s0)
- **THEN** its comparisons are reported and it is never selected or used in a reading clause

### Requirement: Ordered ES readings
Step-2 readings SHALL use consumer units, with the training gain g_tr (center − control, monitor set) and the validation gain g_val (selected candidate − control, loaded 12-producer validation worlds). They SHALL be assigned by a fixed precedence: `unresolved`, `kill`, `success`, `plateau`, `budget_limited`, `other`. A pre-registered table SHALL combine the two seeds' readings.

#### Scenario: Cross-world failure
- **WHEN** the selected candidate's g_tr clears the training floor but its g_val is below one third of it
- **THEN** the reading is `kill` even if g_val also clears its floor, and the cause is diagnosed

#### Scenario: Any arm still improving
- **WHEN** no success holds and any seed-0 arm's monitor-set improvement over its last 10 generations is at least the training floor
- **THEN** the reading is `budget_limited` and further generations need the user's approval

#### Scenario: Seed disagreement
- **WHEN** the selected arm's seed-0 reading is `success` and its seed-1 reading is `plateau`, or the reverse
- **THEN** the final reading is `seed_disagreement` and the cause is diagnosed

#### Scenario: Seed 1 still improving
- **WHEN** the seed-0 reading is `success` or `plateau` and the seed-1 run's own monitor-set improvement over its last 10 generations is at least the training floor
- **THEN** the seed-1 run cannot read `plateau`; without success it reads `budget_limited`

#### Scenario: Seed-1 failure carries through
- **WHEN** the seed-0 reading is `success` or `plateau` and the seed-1 reading is `unresolved`, `kill`, `budget_limited` or `other`
- **THEN** the final reading is the seed-1 reading

### Requirement: Detectable effect pre-registered
Before the first ES run, the minimum detectable effect of the validation read SHALL be computed from paired per-episode differences between arms on the same world-episodes. It SHALL be recorded as a share of the mean validation bar, at the read's world, episode and replicate counts. Above 2% the read SHALL use 8 episodes, and above 3% at 8 episodes it SHALL go to the user.

#### Scenario: Underpowered read
- **WHEN** the recorded detectable effect at 4 episodes exceeds 2% of the mean validation bar
- **THEN** the validation and held-out reads are played with 8 episodes per world

### Requirement: Imitation targets pass label stability
Any per-decision imitation target, including one used only to warm-start, SHALL first pass an EXP-0040-style stability check on replayed states. Winner overlap SHALL be at least 0.80 at the registered continuation count. It SHALL also pass DIAGNOSTICS 12's target-is-a-function check (`/srv/research/tokenomics/docs/DIAGNOSTICS.md` §12).

#### Scenario: Unstable target
- **WHEN** a candidate target's winner overlap across continuations is below 0.80
- **THEN** it is not used as a training target, and the result is recorded

### Requirement: Ladder evaluation worlds
Each ladder step SHALL be judged on loaded 12-producer atlas v2 worlds (ρ > 0.28), with 42-producer, light and collapse worlds reported apart. Method choices and training SHALL use the train split, selection the validation split, and confirmation the held-out split. Results SHALL also be tabulated by ρ quartile.

#### Scenario: Light corner excluded
- **WHEN** a candidate wins only on worlds with ρ ≤ 0.28
- **THEN** the step's reading is unchanged by those worlds

#### Scenario: Larger transfer reported
- **WHEN** a 12-producer-trained candidate is read on 42-producer worlds
- **THEN** it runs unretrained and the result is reported without changing the reading

### Requirement: Held-out confirmation read once
A candidate reading `success` SHALL be confirmed on the held-out split by a pre-registered rule before any gate-exit question. Before launch, the read SHALL be registered as a pending entry in a tracked ledger committed to the repository. The driver SHALL refuse unless the staged ledger holds that matching pending entry. An invalid or crashed read SHALL count as used and go to the user.

#### Scenario: No pending entry
- **WHEN** a real held-out read starts and the staged ledger has no pending entry matching its atlas, change, split, candidate and package tree
- **THEN** the read refuses to run

#### Scenario: Pending entry committed, then launched
- **WHEN** a pending entry naming code commit C0, its package tree and the candidate is committed as C1, and the read is launched at C1
- **THEN** the read is accepted, because C1's package tree equals the entry's, and C1 is recorded when the entry is closed

#### Scenario: Already read
- **WHEN** the ledger entry for that atlas, change and split is `done` or `invalid`
- **THEN** the read refuses to run

#### Scenario: Crashed read
- **WHEN** a held-out read exits with an invalid run, crashes, or its pod is lost before the results are pulled
- **THEN** the entry is set to `invalid`, the case goes to the user, and the read is not re-run automatically

#### Scenario: Not confirmed
- **WHEN** the candidate fails the held-out confirmation rule
- **THEN** the reading is `not_confirmed`, the cause is diagnosed, and no gate-exit question is asked

### Requirement: Paired references and noise floors
Every ladder reading SHALL compare a candidate with `round_robin`, `makeable_weighted_whole` and the current trained orderer, using the same producer policy on world-episodes with equal demand and shock digests. Each comparison SHALL count only when it clears its own floor. A floor's own-spread term SHALL be used only for pairs with at least two paired observations on a world, and the atlas floor otherwise.

#### Scenario: Digest mismatch
- **WHEN** two arms of one world-episode have different demand or shock digests
- **THEN** the run is invalid

#### Scenario: Single-episode world
- **WHEN** a world has one episode and both compared arms are deterministic
- **THEN** that comparison's floor on the world is the atlas floor

#### Scenario: Gain inside the floor
- **WHEN** a candidate beats a reference by less than that comparison's floor
- **THEN** the reading treats it as a tie

### Requirement: Herding and concentration are measured
Every validation and held-out read SHALL report two measures for the candidate, its shared-only ablation, the zero-residual control, `makeable_weighted_whole` and `round_robin`. One is each producer's cycle-to-cycle swing in its share of a product's orders. The other is the share of a product's concurrent orders led by its top producer, with the across-agent dispersion of routed shares. Pre-registered flags SHALL insert a diagnosis before the held-out read.

#### Scenario: Agents switch together
- **WHEN** the candidate's share swing exceeds `makeable_weighted_whole`'s by more than three standard errors on at least a third of the loaded validation worlds
- **THEN** the `herding` flag is recorded and a diagnosis precedes the held-out read

#### Scenario: Gain without de-concentration
- **WHEN** a candidate reads `success` but its concurrent top-producer share is not below the zero-residual control's by more than three standard errors
- **THEN** the `not_via_deconcentration` label is recorded and a diagnosis precedes the held-out read

### Requirement: History effect is recorded
Each `success` and each `plateau` reading SHALL record whether agent-local history changes results. The `history_inert` label SHALL apply when the candidate and its shared-only ablation differ by no more than their floor on the 12-producer validation world mean. The label SHALL NOT change the licence, and SHALL be cited before agent-local history becomes an input of a later model.

#### Scenario: History adds nothing measurable
- **WHEN** the candidate and its shared-only ablation are within the floor on the validation world mean
- **THEN** `history_inert` is recorded and the licence is unchanged

### Requirement: Training producer fixed before training
The producer policy used for orderer training and validation SHALL be recorded before the first ES run, as a user decision with `entity_planner` as the default. If the EXP-0035 producer is chosen alone or with `entity_planner`, the validation and held-out readings SHALL include that producer's column.

#### Scenario: Portable producer chosen
- **WHEN** the user chooses the EXP-0035 producer for training
- **THEN** `success` and held-out confirmation require their clauses in the EXP-0035-producer column

### Requirement: GNN step only after a plateau and a diagnosed reason
A graph-network orderer SHALL be built only after step 2 reads `plateau` and an env-side relational check reads `relational`, i.e. relational state written before ordering adds validation value above the floor. Archive code SHALL NOT be restored before that reading. The restored graph SHALL drop every relation carrying other agents' same-cycle bids. Agent-local history SHALL enter it only after step 2's `history_inert` label is cited.

#### Scenario: Plateau without relational evidence
- **WHEN** step 2 plateaus but the relational state does not raise validation results above the floor
- **THEN** no GNN is built and no archive code is restored, and the cause is diagnosed

#### Scenario: Pending bids excluded from the graph
- **WHEN** `graph_primitives_v1` is restored for the GNN step
- **THEN** its pending-bid relations are absent from the schema, and its reservation relations remain

### Requirement: Ladder sizes stay at 12 and 42
Ladder training SHALL use only economies of 12 producers. Ladder reads SHALL use only 12- and 42-producer economies, and the 42-producer reads SHALL run unretrained.

#### Scenario: Larger world requested
- **WHEN** a ladder driver is asked for a world above 42 producers
- **THEN** it refuses, with no override

### Requirement: Per-order decision cost stays bounded
A per-order orderer SHALL be timed on PoC-B's fanout and consumer axes before any gate read. Its per-orderer-decision cost slope SHALL stay at or below 1.1 on both axes, and it SHALL NOT roll out the simulator at decision time.

#### Scenario: Superlinear per-order cost
- **WHEN** a per-orderer-decision slope exceeds 1.1 on the fanout or the consumer axis
- **THEN** the stage is flagged and the candidate does not go to a gate read until the cause is diagnosed

### Requirement: Ladder evidence does not change the gate
Ladder runs SHALL NOT edit any frozen registry, roster or parameter file, and SHALL NOT produce an acceptance read. A sealed gate read of a ladder candidate SHALL wait for the user's decision on how that read is made.

#### Scenario: Held-out confirmation passes
- **WHEN** a candidate reads `confirmed` on the held-out split
- **THEN** the gate files are unchanged and the sealed 12-producer read waits for the user's decision
