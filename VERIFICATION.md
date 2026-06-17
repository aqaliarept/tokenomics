The current evidence says learning starts failing much earlier than the full complex environment.

  The first documented failure point is:

  fixed depth-3 topology, no topology randomization, no degradation, no weak-producer competition, single product, variable
  quantity near full range
  That is roughly --max-diff 0.24.

  At that point:

  - topology is fixed;
  - product mixture is still off;
  - producer competition is still off;
  - degradation is still off;
  - survival is fine;
  - critic can look healthy in some probes;
  - but the trained policy fulfills almost nothing.

  The policy does not fail because the world is fully randomized. It fails because it does not discover the depth-3
  production chain.

  In simple words:

  Consumer wants final product.
  Final product requires intermediate product.
  Intermediate product requires lower product.
  Producers must replenish inputs, hold inventory, reserve capacity, assemble, and route supply in time.

  The learned producer does not learn that chain. It mostly learns “do not spend capacity.” So it survives, but does not
  fulfill.

  The strongest evidence is the policy-swap test:

  baseProd + baseOrd      157 units
  trainProd + trainOrd     22 units
  trainProd + baseOrd       1 unit
  baseProd + trainOrd     339 units

  Meaning:

  - trained ordering is actually good;
  - trained producing is the blocker;
  - if we keep baseline producers and use trained ordering, performance is excellent;
  - if we use trained producers, fulfillment dies.

  So the real failure is not “RL cannot order.” It is:

  RL cannot learn producer-side replenishment / inventory / assembly behavior in the depth-3 chain from sparse final
  consumer reward.

  DAgger was added because PPO from scratch could not discover this behavior. DAgger means Dataset Aggregation: run the
  learner, see the states it reaches, ask the handwritten policy what it would do there, add those examples, retrain. It is

  Your concern is valid: DAgger is not yet “trained agent discovers better-than-human policy.” It is closer to “trained
  agent learns to imitate a working policy across states.” It is useful because it proved the network can execute much of
  the behavior, but it does not satisfy the larger research goal by itself.

  - Simple/shallow environment: PPO can learn.
  - Big fixed depth-3, easiest-ish demand rung: PPO fails on producing.
      - held-out: 214 < 242/253

  - Still not beating handwritten held-out policy.

  What we do not yet know precisely:

  - whether failure starts at depth-2 or only depth-3;
  - whether fixed quantity p=0.00 is learnable at depth-3;
  - whether failure starts exactly when quantity randomization increases, or already with depth-3 fixed quantity.

  So the next clean experiment should map the boundary:

  depth 1, p=0.00
  depth 2, p=0.00
  depth 3, p=0.00
  depth 3, p=0.10
  depth 3, p=0.24
  depth 3, p=0.25

  For each, run from-scratch PPO and policy-swap. That will tell us whether the first hard wall is chain depth, variable
  quantity, or product mixture.

  My current read: the first real wall is probably depth-3 producer credit assignment, made worse by variable quantity.
  Product mixture and topology randomization are later generalization problems, not the original learning failure.