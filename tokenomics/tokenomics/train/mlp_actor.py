"""Flat MLP actors: the champion checkpoint family, its checkpoint IO, and cloning.

Holds what the shipped champion (`champion.json`: EXP-0004 v4 producer + EXP-0007A
stochastic orderer) needs to load and run, plus the behaviour-cloning / DAgger fits
that produced it (EXP-0002 → 0004):

  - `GaussianActor`: diagonal-Gaussian MLP over a Box action (ADR-0026);
  - action masks over the dims decode can actually change;
  - `save_policies` / `load_policies` / `load_order_actor`: versioned checkpoints;
  - `behavior_clone` / `dagger_clone`: supervised fits from a baseline expert;
  - `staged_config`: the sequential difficulty ladder used to stage economies.

The MAPPO trainer that used to live here (`smoke_ppo`) was removed: PPO was rejected
for this stage (experiments/HISTORY.md §2, EXP-0003). It is preserved on the
`archive-1` branch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ..domain.config import Configuration
from ..domain.ordering_agent import OrderingAgentView
from ..domain.producing_agent import ProducingAgentView
from ..env import TokenomicsAECEnv
from ..env.spaces import config_fingerprint
from ..eval.baselines import Policy


# ADR-0020 reinterpreted for the continuous producing actor (ADR-0026):
# exploration is the Gaussian log_std, not a softmax temperature. Track A's 0.25
# temperature has no mapping; this is a fresh hyperparameter, symmetric with the
# ordering actor by default.
DEFAULT_PROD_LOG_STD = -2.0
# P1a (review 2026-06-15): replenishment quantity dims need wide exploration and a
# positive mean init — at prod_log_std=-2 (σ=0.135) and a ~0 mean they round to
# zero ~99.99% of samples, so the consumer-reward path is never explored. σ=exp(0)=1
# and a +3 raw bias center replenishment where decode fires it (raw>~0.5).
DEFAULT_PROD_REPLEN_LOG_STD = 0.0
# 2.0 clears the B7 bar (seed 1: return 1330 > stock_aware 1267, 100/100 consumer,
# 0% death); 3.0 over-stocks (holding penalty drags return to 1248, just under).
DEFAULT_PROD_REPLEN_BIAS = 2.0
# Restoration-toggle bias (ADR-0043). 0.0 = the phase1 default: greedy eval (the
# action MEAN) never fires restoration until learned. At deep-chain scale, absent
# restoration causes a *localized* capacity-type collapse (one producer loses one
# required type while total capacity stays high) -> system death even as the policy
# fulfils. A positive bias mirrors the P1a replenishment prime: the deterministic
# policy restores from iter 0 (a survive-first prior), so PPO learns fulfilment from
# a living baseline instead of having to discover survival by toggle exploration.
DEFAULT_PROD_RESTORE_BIAS = 0.0
DEFAULT_ORDER_LOG_STD = -2.0
# Checkpoint schema version (ADR-0026). Bumped for the continuous-producing /
# producer-simplex-ordering redesign; the loader rejects older/missing versions.
# v3 (B9): ordering obs gained a per-slot product one-hot -> ordering_obs_dim grew,
# so a v2 ordering actor's input layer no longer matches.
# v4 (#2): producing obs + critic global_state gained a 2P open-consumer-demand
# block (quantity + tokens per product), so v3 producing/critic layers no longer match.
# v5 (ADR-0044): producing obs gained the per-agent producer-mechanics block.
# v6 (plan item 1): critic global_state gained a per-producer mechanics block, so the
# critic input dim changed -> a v5 critic layer no longer matches.
CHECKPOINT_VERSION = 6


def mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


# --- decentralized actors (act on local obs) ---
# Both policies are diagonal-Gaussian over a Box (ADR-0026): the producing actor
# emits raw reals and decode owns all gating, so no action mask survives here.
class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        log_std_init: float | np.ndarray = 0.0,
        mean_bias: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.actor = mlp(obs_dim, act_dim)
        # P1a (review 2026-06-15): the replenishment dims round to zero unless the
        # raw action clears ~0.5, so a near-zero mean + std exp(-2)=0.135 explores
        # the multi-hop replenishment path ~0.01% of samples — no consumer-reward
        # gradient. Allow a per-dim log_std (wider exploration on quantity dims)
        # and a positive output-bias init (mean starts where replenishment fires),
        # so the path is sampled and greedy eval emits non-zero replenishment from
        # iter 0. Both are pure init — load_state_dict overwrites them, so trained
        # checkpoints are unaffected.
        if mean_bias is not None:
            with torch.no_grad():
                self.actor[-1].bias.add_(torch.as_tensor(mean_bias, dtype=torch.float32))
        init = torch.as_tensor(log_std_init, dtype=torch.float32)
        if init.ndim == 0:
            init = init.expand(act_dim).clone()
        self.log_std = nn.Parameter(init)

    def dist(self, obs: torch.Tensor) -> Normal:
        return Normal(self.actor(obs), self.log_std.exp())


def producing_mean_bias(
    num_inputs: int,
    action_dim: int,
    replen_bias: float,
    restore_bias: float = 0.0,
) -> np.ndarray:
    """Output-bias init for the producing actor's final layer (P1a + ADR-0043).

    Action layout (decode_producing): `[0:g]` replenishment quantities, then the
    three investment toggles `[g]=restoration, g+1=upgrade, g+2=inventory]`. Greedy
    eval emits the Gaussian mean, so a dim only fires deterministically when its
    bias is positive. Prime replenishment (the consumer-reward path) and, at scale,
    restoration (the survive-first prior). Upgrade + inventory stay at 0 — learned,
    not primed. `restore_bias=0` reproduces the phase1 init exactly."""
    bias = np.zeros(action_dim, dtype=np.float32)
    bias[:num_inputs] = replen_bias
    bias[num_inputs] = restore_bias
    return bias


def producing_action_mask(spec, view: ProducingAgentView) -> np.ndarray:
    """Active producing-action dims for one transition (P1b, review 2026-06-15).

    `decode_producing` no-ops every dim that cannot change this producer's command:
    inputs it does not assemble, and a toggle for an unavailable / already-running
    investment. Summing log-prob / entropy / KL over those dims injects gradient
    noise and spends the target-KL budget without altering behavior. The mask keeps
    the likelihood over only the dims the action actually exercises this step."""
    g = spec.num_inputs
    mask = np.zeros(g + 3, dtype=np.float32)
    # P2 (review 2026-06-15): replenishment dims are behavioral only when there is
    # shared inventory headroom and a positive replenish ceiling — decode emits no
    # replenishment otherwise (joint scaling to headroom 0 / max_replenish 0 yields
    # all-zero quantities). Masking them at zero headroom keeps the likelihood off
    # dims that cannot change the command this step.
    if view.headroom > 0 and spec.max_replenish[view.id] > 0:
        for prod in spec.producer_inputs[view.id]:
            mask[spec.input_index[prod]] = 1.0
    mask[g] = 0.0 if view.restoration_running else 1.0
    mask[g + 1] = 0.0 if view.upgrade_running else 1.0
    mask[g + 2] = (
        1.0 if (view.has_inventory_expansion and not view.inventory_expansion_running) else 0.0
    )
    return mask


def ordering_action_mask(spec, view: OrderingAgentView) -> np.ndarray:
    """Active ordering-action dims for one transition (P1, review 2026-06-15).

    The ordering Box is K*P raw logits (slot k × producer). `decode_ordering`
    softmaxes slot k over only the *eligible* producers of the k-th held order, so
    two dim classes are behavioral no-ops: (a) every dim of a slot whose order has
    ≤1 eligible producer — softmax over one is always 1, there is no allocation
    choice; (b) the non-eligible producer dims of a multi-eligible slot. In the
    canonical Phase 1 config every product has a single declarer (ADR-0028), so the
    whole ordering actor is inert — without the mask it still spends gradient and
    the target-KL budget with zero effect on the env. Slots past `spec.k` (decode
    falls back to an even split, no action effect) stay masked too."""
    mask = np.zeros(spec.k * spec.num_producers, dtype=np.float32)
    for k, oid in enumerate(sorted(view.incoming)):
        if k >= spec.k:
            break
        eligible = sorted(view.eligible[oid].keys())
        if len(eligible) <= 1:
            continue
        for pid in eligible:
            mask[k * spec.num_producers + spec.producer_ids.index(pid)] = 1.0
    return mask


def greedy_eval_policies(
    prod: GaussianActor,
    order: GaussianActor,
    bound: float,
) -> tuple[Policy, Policy]:
    """Wrap the trained actors as deterministic eval `Policy` closures (ADR-0003).

    Greedy = the *deployed* policy: the Gaussian mean (clipped to the action
    bound) for both actors now (ADR-0026). They read the obs via `env.observe`
    so the eval harness measures them exactly like a baseline."""

    def greedy(actor: GaussianActor, env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        o = torch.as_tensor(env.observe(agent), dtype=torch.float32)
        with torch.no_grad():
            mean = actor.actor(o).numpy()
        return np.clip(mean, -bound, bound).astype(np.float32)

    def prod_policy(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        return greedy(prod, env, agent)

    def order_policy(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        return greedy(order, env, agent)

    return prod_policy, order_policy


def save_policies(
    prod: GaussianActor,
    order: GaussianActor,
    bound: float,
    path: str,
    config: Configuration | None = None,
    orderer_obs_version: int = 1,
) -> None:
    """Checkpoint the deployed actors + the dims needed to rebuild them, so the
    eval harness (signal_check) can score the trained policy against the baseline
    table without re-training. Stamped with `CHECKPOINT_VERSION` (ADR-0026) and, when
    `config` is given, the training config's fingerprint (#7) so an eval can detect a
    wrong-config mismatch before the actor crashes on a dim error. Pass
    `orderer_obs_version=2` for an enriched-orderer checkpoint (EXP-0006A) so its
    fingerprint cannot collide with a v1 model on the same economy."""
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "config_fingerprint": (
                config_fingerprint(config, orderer_obs_version) if config is not None else None
            ),
            "prod_obs": prod.actor[0].in_features,
            "prod_act": prod.log_std.numel(),
            "order_obs": order.actor[0].in_features,
            "order_act": order.log_std.numel(),
            "bound": bound,
            "prod": prod.state_dict(),
            "order": order.state_dict(),
        },
        path,
    )


def checkpoint_fingerprint(path: str) -> str | None:
    """Read the training-config fingerprint stamped in a checkpoint (#7), or None if
    the checkpoint predates fingerprinting. Lets `signal_check` warn when a policy is
    scored on a config it was not trained on."""
    ckpt = torch.load(path, weights_only=True)
    return ckpt.get("config_fingerprint")


def checkpoint_obs_dims(path: str) -> dict[str, int]:
    """Read the actor input/output dims a checkpoint's tensors were built for
    (`prod_obs`/`prod_act`/`order_obs`/`order_act`). `signal_check` (review P2)
    compares these to the eval spec and SKIPS the trained row when they differ,
    instead of crashing the actor on a matrix-shape mismatch."""
    ckpt = torch.load(path, weights_only=True)
    return {k: ckpt[k] for k in ("prod_obs", "prod_act", "order_obs", "order_act")}


def load_policies(path: str) -> tuple[Policy, Policy]:
    """Rebuild the greedy eval `Policy` closures from a `save_policies`
    checkpoint — the trained policy as the eval harness sees a baseline.

    Rejects a checkpoint with a missing or older schema version loudly (ADR-0026):
    the producing action, ordering action meaning, and critic state all changed in
    the redesign, so a stale checkpoint holds semantically wrong tensors."""
    ckpt = torch.load(path, weights_only=True)
    version = ckpt.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint schema version {version!r} != {CHECKPOINT_VERSION} "
            "(ADR-0026 redesign); retrain — old checkpoints are not loadable"
        )
    prod = GaussianActor(ckpt["prod_obs"], ckpt["prod_act"])
    order = GaussianActor(ckpt["order_obs"], ckpt["order_act"])
    prod.load_state_dict(ckpt["prod"])
    order.load_state_dict(ckpt["order"])
    return greedy_eval_policies(prod, order, ckpt["bound"])


def load_order_actor(path: str) -> tuple[GaussianActor, float]:
    """Rebuild the orderer GaussianActor + action bound from a `save_policies` checkpoint
    (mirrors `clone_fidelity.load_producer_actor`). `load_policies` returns the greedy
    closures; this exposes the raw actor so a *stochastic* deploy (EXP-0006B) can sample
    from its learned `log_std` instead of collapsing to clip(mean)."""
    ckpt = torch.load(path, weights_only=True)
    if ckpt.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"checkpoint version {ckpt.get('version')!r} != {CHECKPOINT_VERSION}")
    order = GaussianActor(ckpt["order_obs"], ckpt["order_act"])
    order.load_state_dict(ckpt["order"])
    return order, ckpt["bound"]


@dataclass
class BCData:
    """Per-role (obs, action) pairs recorded from a baseline rollout for supervised
    behavior cloning (VERIFICATION §11.4). Arrays are stacked rows: obs widths match
    the actor input dims, action widths the actor output dims, so the cloned net is
    checkpoint-schema compatible."""

    prod_obs: np.ndarray
    prod_act: np.ndarray
    order_obs: np.ndarray
    order_act: np.ndarray


def collect_bc_dataset(
    config: Configuration,
    prod_policy: Policy,
    order_policy: Policy,
    seeds: "range | list[int]",
    horizon: int = 100,
) -> BCData:
    """Roll out a baseline policy pair and record what each actor *should* output per
    observation (VERIFICATION §11.4). The producing actor is the fulfillment blocker
    (§12: it hoards capacity, never replenishes), so cloning the baseline's
    obs→action map gives a policy that already fulfills, instead of requiring random
    discovery of the multi-hop replenishment→assembly chain."""
    po: list[np.ndarray] = []
    pa: list[np.ndarray] = []
    oo: list[np.ndarray] = []
    oa: list[np.ndarray] = []
    for s in seeds:
        env = TokenomicsAECEnv(config=config, horizon=horizon)
        env.reset(seed=s)
        for a in env.possible_agents:
            env.action_space(a).seed(s)
        max_iter = env.horizon * len(env.possible_agents) * 3 + 100
        for agent in env.agent_iter(max_iter=max_iter):
            _, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            obs = np.asarray(env.observe(agent), dtype=np.float32)
            if env._role[agent][0] == "prod":
                act = prod_policy(env, agent)
                po.append(obs)
                pa.append(np.asarray(act, dtype=np.float32))
            else:
                act = order_policy(env, agent)
                oo.append(obs)
                oa.append(np.asarray(act, dtype=np.float32))
            env.step(act)
    return BCData(
        prod_obs=np.stack(po), prod_act=np.stack(pa),
        order_obs=np.stack(oo), order_act=np.stack(oa),
    )


def _build_clone_actors(
    config: Configuration,
    *,
    prod_log_std: float,
    prod_replen_log_std: float,
    prod_replen_bias: float,
    prod_restore_bias: float,
    log_std_init: float,
) -> "tuple[GaussianActor, GaussianActor, float]":
    """Build the producing/ordering actors with the exact train-time init (per-dim
    log_std + replenishment/restore mean bias) so a clone is checkpoint-schema
    compatible (VERIFICATION §11.4). Shared by behavior_clone and dagger_clone."""
    env = TokenomicsAECEnv(config=config, horizon=1)
    bound = float(env.action_space(env._order_agents[0]).high[0])
    prod_obs = env.observation_space(env._prod_agents[0]).shape[0]
    order_obs = env.observation_space(env._order_agents[0]).shape[0]
    g = env.spec_.num_inputs
    pdim = env.spec_.producing_action_dim
    prod_logstd = np.full(pdim, prod_log_std, dtype=np.float32)
    prod_logstd[:g] = prod_replen_log_std
    prod_bias = producing_mean_bias(g, pdim, prod_replen_bias, prod_restore_bias)
    prod = GaussianActor(prod_obs, pdim, log_std_init=prod_logstd, mean_bias=prod_bias)
    order = GaussianActor(order_obs, env.spec_.ordering_action_dim, log_std_init=log_std_init)
    return prod, order, bound


def _fit_clone(
    actor: GaussianActor, obs: np.ndarray, act: np.ndarray,
    bound: float, epochs: int, lr: float,
) -> float:
    """MSE-fit a Gaussian actor's mean to clipped target actions (greedy deploys
    clip(mean), ADR-0026). Full-batch Adam; returns the final loss."""
    X = torch.as_tensor(obs, dtype=torch.float32)
    Y = torch.clamp(torch.as_tensor(act, dtype=torch.float32), -bound, bound)
    opt = torch.optim.Adam(actor.actor.parameters(), lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((actor.actor(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    return float(loss.detach())


def _fit_producer_clone(
    actor: GaussianActor,
    obs: np.ndarray,
    act: np.ndarray,
    masks: np.ndarray,
    weights: np.ndarray,
    *,
    g: int,
    bound: float,
    epochs: int,
    lr: float,
    toggle_weight: float = 5.0,
) -> float:
    """Decode-aware producer clone fit (ADR-0049 Phase 1). Plain full-vector MSE
    (`_fit_clone`) ignores that `decode_producing` is gated at 0 — toggles fire for
    raw>0, replenish for raw>0, masked dims are no-ops — so a low-MSE clone decodes
    the wrong command (empirically ~85% restoration false positives, decoded match
    0.15). This loss matches the decode boundary:

      - replenishment dims: weighted MSE over decode-ACTIVE dims only (`masks`);
      - toggle dims: BCE on the SIGN (target = label>0), exactly decode's threshold,
        so the optimizer is penalized for crossing it rather than for L2 distance.

    Per-sample `weights` upweight the middle layer + positive examples (the depth-3
    wall is L1 restoration/replenishment, EXP-0001 trace)."""
    X = torch.as_tensor(obs, dtype=torch.float32)
    Y = torch.clamp(torch.as_tensor(act, dtype=torch.float32), -bound, bound)
    M = torch.as_tensor(masks, dtype=torch.float32)
    W = torch.as_tensor(weights, dtype=torch.float32).unsqueeze(1)
    tog_target = (Y[:, g : g + 3] > 0).float()  # decode fires a toggle iff raw > 0
    opt = torch.optim.Adam(actor.actor.parameters(), lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        opt.zero_grad()
        out = actor.actor(X)
        rep_w = M[:, :g] * W
        rep_loss = (((out[:, :g] - Y[:, :g]) ** 2) * rep_w).sum() / rep_w.sum().clamp_min(1.0)
        tog_w = M[:, g : g + 3] * W
        tog_bce = nn.functional.binary_cross_entropy_with_logits(
            out[:, g : g + 3], tog_target, reduction="none"
        )
        tog_loss = (tog_bce * tog_w).sum() / tog_w.sum().clamp_min(1.0)
        loss = rep_loss + toggle_weight * tog_loss
        loss.backward()
        opt.step()
    return float(loss.detach())


def behavior_clone(
    config: Configuration,
    prod_policy: Policy,
    order_policy: Policy,
    *,
    seeds: "range | list[int]" = range(2),
    horizon: int = 100,
    epochs: int = 1000,
    lr: float = 3e-3,
    seed: int = 0,
    prod_log_std: float = DEFAULT_PROD_LOG_STD,
    prod_replen_log_std: float = DEFAULT_PROD_REPLEN_LOG_STD,
    prod_replen_bias: float = DEFAULT_PROD_REPLEN_BIAS,
    prod_restore_bias: float = DEFAULT_PROD_RESTORE_BIAS,
    log_std_init: float = DEFAULT_ORDER_LOG_STD,
    log_std_save: float | None = None,
    save: str | None = None,
    log: Callable[[str], None] = print,
) -> str | None:
    """Supervised pre-train of both actors to a baseline's obs→action map, saved as a
    versioned checkpoint (VERIFICATION §11.4). Actors are built with the exact train
    init (per-dim log_std + replenishment/restore mean bias) so the saved state is
    schema-compatible. Fits the Gaussian *mean* via MSE to the clipped baseline action
    (greedy deploys clip(mean), ADR-0026), giving a policy that already fulfills
    instead of needing to discover the multi-hop chain by chance
    (§12: the from-scratch producing actor hoards capacity and never replenishes)."""
    torch.manual_seed(seed)   # reproducible actor MLP init -> reproducible clone (§13.3)
    prod, order, bound = _build_clone_actors(
        config, prod_log_std=prod_log_std, prod_replen_log_std=prod_replen_log_std,
        prod_replen_bias=prod_replen_bias, prod_restore_bias=prod_restore_bias,
        log_std_init=log_std_init,
    )
    data = collect_bc_dataset(config, prod_policy, order_policy, seeds, horizon)
    pl = _fit_clone(prod, data.prod_obs, data.prod_act, bound, epochs, lr)
    ol = _fit_clone(order, data.order_obs, data.order_act, bound, epochs, lr)
    log(f"bc fit: prod_mse {pl:.4f} ({data.prod_obs.shape[0]} rows)  "
        f"order_mse {ol:.4f} ({data.order_obs.shape[0]} rows)")
    # Tighten exploration for warm-start fine-tuning (§13.3): the wide init log_std
    # makes stochastic rollouts drift the policy off the cloned greedy optimum.
    if log_std_save is not None:
        with torch.no_grad():
            prod.log_std.fill_(log_std_save)
            order.log_std.fill_(log_std_save)
    if save is not None:
        save_policies(prod, order, bound, save, config=config)
    return save


def dagger_relabel_dataset(
    config: Configuration,
    learner_prod: Policy,
    learner_order: Policy,
    expert_prod: Policy,
    expert_order: Policy,
    seeds: "range | list[int]",
    horizon: int,
    bound: float,
) -> BCData:
    """One DAgger relabel pass (VERIFICATION §13.11 fix). Roll out the LEARNER (its
    action drives the env), but at every visited state record the EXPERT's action as
    the supervised label. Aggregating these over rounds trains the clone on the state
    distribution it actually reaches — fixing the compounding distribution shift that
    plain BC suffers (§13.5): plain BC only ever sees the expert's own trajectory, so
    a small mean error walks the learner into states the expert never visited and the
    error snowballs over the 100-cycle chain.

    Actions are clipped to the action bound to match the deployed greedy policy
    (clip(mean), ADR-0026), so labels are exactly what the learner should output."""
    po: list[np.ndarray] = []
    pa: list[np.ndarray] = []
    oo: list[np.ndarray] = []
    oa: list[np.ndarray] = []
    for s in seeds:
        env = TokenomicsAECEnv(config=config, horizon=horizon)
        env.reset(seed=s)
        for a in env.possible_agents:
            env.action_space(a).seed(s)
        max_iter = env.horizon * len(env.possible_agents) * 3 + 100
        for agent in env.agent_iter(max_iter=max_iter):
            _, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            obs = np.asarray(env.observe(agent), dtype=np.float32)
            if env._role[agent][0] == "prod":
                label = np.clip(np.asarray(expert_prod(env, agent), dtype=np.float32), -bound, bound)
                po.append(obs); pa.append(label)
                step_act = learner_prod(env, agent)
            else:
                label = np.clip(np.asarray(expert_order(env, agent), dtype=np.float32), -bound, bound)
                oo.append(obs); oa.append(label)
                step_act = learner_order(env, agent)
            env.step(step_act)
    return BCData(
        prod_obs=np.stack(po), prod_act=np.stack(pa),
        order_obs=np.stack(oo), order_act=np.stack(oa),
    )


def dagger_clone(
    configs: "list[Configuration]",
    expert_prod: Policy,
    expert_order: Policy,
    *,
    dagger_iters: int = 3,
    seeds: "range | list[int]" = range(2),
    horizon: int = 100,
    epochs: int = 1500,
    lr: float = 3e-3,
    seed: int = 0,
    prod_log_std: float = DEFAULT_PROD_LOG_STD,
    prod_replen_log_std: float = DEFAULT_PROD_REPLEN_LOG_STD,
    prod_replen_bias: float = DEFAULT_PROD_REPLEN_BIAS,
    prod_restore_bias: float = DEFAULT_PROD_RESTORE_BIAS,
    log_std_init: float = DEFAULT_ORDER_LOG_STD,
    log_std_save: float | None = None,
    save: str | None = None,
    log: Callable[[str], None] = print,
) -> str | None:
    """Multi-topology DAgger (VERIFICATION §13.11 fix). Round 0 is plain BC on the
    experts' own rollouts across every config in `configs`; each later round rolls
    out the CURRENT clone, relabels the states it visits with the experts, AGGREGATES
    into the dataset, and refits from a fresh (seeded) init. This attacks both held-out
    failures at once: multiple topologies give generalization breadth, and the relabel
    fixes the compounding error that made plain multi-topology BC worse (§13.11).

    Actors/dims come from `configs[0]`; the dims-preserving rule (ADR-0039) guarantees
    every scale topology shares obs/action dims, so one net spans the suite."""
    pf, of = [], []  # per-(config) aggregated obs/act, grown each round
    pa_all, oa_all = [], []

    def aggregate(d: BCData) -> None:
        pf.append(d.prod_obs); pa_all.append(d.prod_act)
        of.append(d.order_obs); oa_all.append(d.order_act)

    # round 0: plain BC on expert trajectories across all topologies
    for cfg in configs:
        aggregate(collect_bc_dataset(cfg, expert_prod, expert_order, seeds, horizon))

    prod = order = None
    bound = 0.0
    for it in range(dagger_iters):
        torch.manual_seed(seed)  # refit from a fixed init on the grown dataset
        prod, order, bound = _build_clone_actors(
            configs[0], prod_log_std=prod_log_std, prod_replen_log_std=prod_replen_log_std,
            prod_replen_bias=prod_replen_bias, prod_restore_bias=prod_restore_bias,
            log_std_init=log_std_init,
        )
        PO, PA = np.concatenate(pf), np.concatenate(pa_all)
        OO, OA = np.concatenate(of), np.concatenate(oa_all)
        pl = _fit_clone(prod, PO, PA, bound, epochs, lr)
        ol = _fit_clone(order, OO, OA, bound, epochs, lr)
        log(f"dagger round {it}: prod_mse {pl:.4f} order_mse {ol:.4f} "
            f"({PO.shape[0]} prod rows, {len(configs)} topos)")
        # relabel the CURRENT clone's visited states with the experts (skip after the
        # final fit — nothing would consume the new data).
        if it < dagger_iters - 1:
            lp, lo = greedy_eval_policies(prod, order, bound)
            for cfg in configs:
                aggregate(dagger_relabel_dataset(cfg, lp, lo, expert_prod, expert_order,
                                                 seeds, horizon, bound))

    if log_std_save is not None:
        with torch.no_grad():
            prod.log_std.fill_(log_std_save)
            order.log_std.fill_(log_std_save)
    if save is not None:
        save_policies(prod, order, bound, save, config=configs[0])
    return save


def scale_degradation(cfg: Configuration, d: float) -> Configuration:
    """Copy `cfg` with every producer's degradation scaled by `d` (ADR-0006).
    `d=1.0` is the real config; smaller `d` slows decay so survival is reachable.
    No dimension changes, so the actor/critic shapes stay valid."""
    producers = [
        replace(p, degradation=max(0, round(p.degradation * d)))
        for p in cfg.producer_configs
    ]
    return replace(cfg, producer_configs=producers)


# Demand-variance curriculum axis (review 2026-06-15): below this difficulty a
# multi-product consumer demands only its single primary product; at/above it the
# full product mixture is enabled. The quantity range anneals continuously and
# independently. One gated scalar drives both this and degradation together.
MIXTURE_AT = 0.5


def scale_demand(cfg: Configuration, qty_d: float, mixture_on: bool | None = None) -> Configuration:
    """Copy `cfg` easing consumer demand *variance* — a curriculum axis alongside
    `scale_degradation` (review 2026-06-15).

    `qty_d` (0..1) sets the per-cycle quantity range: `0` collapses it to
    `quantity_min`, `1` restores the real `quantity_max`. `mixture_on` controls the
    multi-product pool: `True` keeps the real pool, `False` collapses it to the
    single primary `product`. When `mixture_on` is None it falls back to the legacy
    single-scalar coupling (`qty_d >= MIXTURE_AT`); `staged_config` passes it
    explicitly so quantity and mixture ramp in *separate* stages. Dims are unchanged
    — the ordering one-hot spans process-sheet products, not the consumer pool — so
    actor/critic shapes stay valid (cf. `scale_degradation`)."""
    consumers = [
        replace(
            c,
            products=c.products
            if (len(c.product_pool) > 1 and (qty_d >= MIXTURE_AT if mixture_on is None else mixture_on))
            else (),
            quantity_max=c.quantity_min + int(round((c.quantity_max - c.quantity_min) * qty_d)),
        )
        for c in cfg.consumers
    ]
    return replace(cfg, consumers=consumers)


# Sequential 5-stage ladder (#1-proper, review 2026-06-15). The gated difficulty
# scalar p in [0,1] ramps ONE axis at a time instead of all axes at once, so a stall
# is attributable to a single axis. Stage 1 (fixed qty1) is the p=0 baseline; each
# quarter ramps the next axis to full while earlier axes stay maxed and later axes
# stay fully eased: qty (variable quantity) -> mixture (product pool) -> competition
# (viable->weak) -> degradation (decay). p=1 is the real config.
_QTY_STAGE = (0.00, 0.25)
_MIX_AT = 0.25  # product mixture switches on once the quantity ramp completes
_COMP_STAGE = (0.50, 0.75)
_DEGR_STAGE = (0.75, 1.00)


def _sub(p: float, lo: float, hi: float) -> float:
    """Local 0..1 progress of `p` within sub-interval [lo, hi] (clamped)."""
    return min(1.0, max(0.0, (p - lo) / (hi - lo)))


def staged_config(base_cfg: Configuration, p: float) -> Configuration:
    """Map the gated difficulty scalar `p` (0..1) to a config that ramps the
    curriculum axes *sequentially* (#1-proper). See the stage constants above. At
    `p=0` the config is the easiest (fixed qty1, single product, viable producers,
    no degradation); at `p=1` it is `base_cfg` (the real task). Dims are preserved
    by every axis, so the actor/critic built on `base_cfg` stays valid."""
    cfg = scale_demand(base_cfg, _sub(p, *_QTY_STAGE), mixture_on=p >= _MIX_AT)
    cfg = scale_competition(cfg, _sub(p, *_COMP_STAGE))
    cfg = scale_degradation(cfg, _sub(p, *_DEGR_STAGE))
    return cfg


def scale_competition(cfg: Configuration, d: float) -> Configuration:
    """Copy `cfg` easing *competition* by difficulty `d` (#1-proper, review item #5).

    Some configs pit a strong producer against a near-dead one (e.g. complex.json
    `p5` owns capacity-1=1 while its product needs 2 -> zero throughput). Routing
    against a competitor that cannot produce is the hard case; the policy should
    first learn to route among *viable* producers. At `d=0` every producer's owned
    capacity is raised to at least its declared products' recipe requirement (made
    viable); at `d=1` the real (possibly weak) capacity is restored. Only deficient
    types are touched; capacity *types* are unchanged, so actor/critic shapes stay
    valid (cf. `scale_degradation`/`scale_demand`)."""
    require = {ps.product: ps.require for ps in cfg.process_sheets}
    producers = []
    for p in cfg.producer_configs:
        needed: dict[str, int] = {}
        for product in p.produces:
            for t, req in require.get(product, {}).items():
                needed[t] = max(needed.get(t, 0), req)
        caps = dict(p.capacities)
        for t, req in needed.items():
            real = caps.get(t, 0)
            if real < req:  # deficient -> interpolate real..req as difficulty eases
                caps[t] = round(real + (req - real) * (1.0 - d))
        producers.append(replace(p, capacities=caps))
    return replace(cfg, producer_configs=producers)


