# Reachable producer hard-case mining before adversarial generation

Status: accepted

## Context

ADR-0049 switched the next phase from end-to-end PPO from scratch to producer-first
training. The immediate scaffold is BC/DAgger from `obs_stock_aware`, gated by the
producer-isolated `train+base` swap column.

One risk remains: even if BC/DAgger learns the local producer rule, later PPO
fine-tuning can reintroduce fixed-topology dependence. A fixed-vector MLP can exploit
product/layer positions in the observation, and sparse final consumer reward may push
the producer back toward the observed failure: outer-layer over-investment,
middle-layer death, and blocked reservations.

An AlphaGo-like "second agent creates tasks" idea is useful, but a learned adversarial
generator is too large a first step. It can easily create impossible states, invalid
simulator corners, or examples that make the learner fail without improving real
environment performance.

The safer first version is **reachable hard-case mining**:

```text
run the current producer in real environments
collect reachable states where it fails or differs from obs_stock_aware
label those states with obs_stock_aware
retrain the producer on normal DAgger data plus mined hard cases
validate only with real swap_eval and producer_trace
```

## Decision

Add reachable producer hard-case mining as the next training stage after BC/DAgger and
before PPO fine-tuning. Defer any learned adversarial task generator until mined
reachable cases are exhausted or prove insufficient.

A mined case is accepted only when:

1. it comes from an actual rollout or from a replayable rollout prefix,
2. it targets a real observed failure mode,
3. `obs_stock_aware` gives a valid expert action for the producer observation,
4. the current learned producer fails, underperforms, or takes an action far from the
   expert label,
5. the case improves real-environment gates when added to training.

Training on mined cases is supervised/DAgger-style by default: the miner finds hard
reachable states, and the expert supplies the target action. PPO remains optional and is
only attempted after mined-case training passes the same real-environment gates.

## Implementation Direction

- Create a producer hard-case mining module that rolls the current learner in fixed
  scale environments with the baseline orderer and records producer observations.
- Represent each accepted case as `(config seed, episode seed, cycle, producer id,
  observation, expert action, learner action, diagnostics, failure tag)`.
- Start with rollout mining and trace-derived rules. Do not synthesize arbitrary raw
  domain states in the first version.
- Score cases by practical usefulness: expert and learner disagree on a reachable
  failure-mode state, and adding the case improves `train+base` or the producer trace.
- Train the producer actor with supervised loss on aggregated mined cases, mixed with
  normal BC/DAgger rollout data so it does not overfit rare failure states.
- Validate only in the real environment using `swap_eval` and `producer_trace`; mined
  case accuracy is a training diagnostic, not the release metric.

## Considered Options

- **Proceed directly from DAgger to PPO.** Deferred. PPO may improve fixed-topology
  return while relearning topology-specific behavior.
- **Build a learned adversarial generator first.** Rejected for the next phase. It adds
  a second learning problem before the simpler reachable-mining path is tested.
- **Generate arbitrary synthetic producer states.** Rejected for v1. Raw synthetic states
  can violate simulator invariants or teach recovery from impossible situations.
- **Use mined-case accuracy as the final metric.** Rejected. The final metric remains
  `d3_p00 train+base`, producer traces, and held-out depth/topology checks.
- **Jump straight to GNN.** Deferred. Mined hard cases improve the training signal
  regardless of whether the actor is an MLP or a future graph policy.

## Consequences

- PPO fine-tuning is no longer the default immediate step after DAgger. It becomes a
  controlled optional phase after mined hard-case supervised training.
- The project gains a reusable curriculum dataset made from reachable producer
  failures, which can later train an MLP, a set policy, or a GNN.
- A learned adversarial generator remains a later option, but it must be justified by
  mined-case saturation or failure.
- The real environment gate remains unchanged: `train+base >= 0.8 * base+base` with
  near-zero death, plus producer-trace checks that middle-layer capacity survives.
