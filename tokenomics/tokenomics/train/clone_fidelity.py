"""Clone-fidelity diagnostics for the producer warm-start (ADR-0049 Phase 1).

The warm-start gate fails (`clone+base` 82 << 290) while the `obs_stock_aware` expert
itself clears it: a clone-fidelity problem, not a weak expert. Raw MSE hides it —
`decode_producing` is gated (replenish only for raw>0, a toggle only for raw>0, masked
dims no-op), so a low-MSE clone can still decode the wrong command. This module scores
the clone on the thing the gate exercises: the DECODED command, broken down per the
traced failure surface (toggles, replenish quantities, supply-chain layer)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.producing_agent import ProducingAgentView
from ..env import TokenomicsAECEnv
from ..env.decode import decode_producing
from ..env.spaces import EnvSpec
from ..eval.baselines import Policy


def _level_of(pid: str) -> int:
    """Scale topology ids are `p{level}_{k}` (eval.scale)."""
    return int(pid.split("_")[0][1:])


@dataclass
class ProducerSample:
    """One producer decision recorded under the expert: the observation the learner
    would see, the expert's action label, and the producer view + layer needed to
    decode the gated command and group metrics by supply-chain level."""

    obs: np.ndarray
    expert_action: np.ndarray
    view: ProducingAgentView
    producer_id: str
    layer: int


def collect_producer_samples(
    config,
    prod_expert: Policy,
    order_policy: Policy,
    seeds: "range | list[int]",
    horizon: int = 100,
    *,
    step_prod: "Policy | None" = None,
) -> list[ProducerSample]:
    """Drive episodes under a fixed orderer, recording every producer transition (obs,
    EXPERT action label, view, layer). The view is decode's gating state captured at
    the acting moment, so the recorded label decodes to the exact expert command.

    `step_prod` controls which producer action actually drives the env: by default the
    expert (on-policy BC). For DAgger, pass the current learner as `step_prod` so the
    env visits the learner's state distribution while labels stay the expert's
    (relabel) — and keep `order_policy` at the base orderer to match deployment."""
    out: list[ProducerSample] = []
    driver = step_prod or prod_expert
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
            role, did = env._role[agent]
            if role == "prod":
                obs = np.asarray(env.observe(agent), dtype=np.float32)
                view = env.system.producing_agents[did].view()
                label = np.asarray(prod_expert(env, agent), dtype=np.float32)
                out.append(ProducerSample(obs, label, view, did, _level_of(did)))
                env.step(driver(env, agent))
            else:
                env.step(order_policy(env, agent))
    return out


def decoded_command_match(
    spec: EnvSpec, samples: list[ProducerSample], pred_actions: np.ndarray
) -> float:
    """Fraction of samples whose PREDICTED action decodes to the same producing
    command as the expert's, using each sample's recorded gating view. This is the
    behavior the swap gate sees — not raw action distance."""
    if not samples:
        return 0.0
    hits = 0
    for s, pred in zip(samples, pred_actions):
        want = decode_producing(spec, s.expert_action, s.view)
        got = decode_producing(spec, pred, s.view)
        hits += int(want == got)
    return hits / len(samples)


def predict_actions(actor, samples: list[ProducerSample], bound: float) -> np.ndarray:
    """Greedy clone predictions for the samples: the actor mean clipped to the action
    bound — exactly what `greedy_eval_policies` deploys (ADR-0026)."""
    import torch

    X = torch.as_tensor(np.stack([s.obs for s in samples]), dtype=torch.float32)
    with torch.no_grad():
        return torch.clamp(actor.actor(X), -bound, bound).numpy()


def _fit_arrays(
    spec: EnvSpec, samples: list[ProducerSample], l1_weight: float, pos_weight: float
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Build (obs, clipped-target act, decode mask, per-sample weight) for one config's
    samples. The mask uses THIS config's spec so a permuted/varied topology contributes
    its own active dims; weights upweight the middle layer (L1) + positive examples."""
    from .smoke_ppo import producing_action_mask

    obs = np.stack([s.obs for s in samples])
    act = np.stack([s.expert_action for s in samples])
    masks = np.stack([producing_action_mask(spec, s.view) for s in samples])
    g = spec.num_inputs
    positive = (
        (act[:, g : g + 3] > 0).any(axis=1) | (act[:, :g] > 0).any(axis=1)
    ).astype(np.float32)
    layer = np.array([s.layer for s in samples], dtype=np.float32)
    weights = 1.0 + l1_weight * (layer == 1.0) + pos_weight * positive
    return obs, act, masks, weights


def fit_producer_clone(
    actor,
    samples: list[ProducerSample],
    spec: EnvSpec,
    bound: float,
    *,
    epochs: int = 2000,
    lr: float = 3e-3,
    toggle_weight: float = 5.0,
    l1_weight: float = 2.0,
    pos_weight: float = 1.0,
) -> float:
    """Decode-aware producer clone fit over recorded samples (ADR-0049 Phase 1). Builds
    the per-sample decode mask + the layer/positive upweighting, then delegates the
    masked-MSE + BCE-toggle loss to `smoke_ppo._fit_producer_clone`. Middle-layer (L1)
    and positive (restore/replenish) examples are upweighted — the depth-3 wall."""
    from .smoke_ppo import _fit_producer_clone

    obs, act, masks, weights = _fit_arrays(spec, samples, l1_weight, pos_weight)
    return _fit_producer_clone(
        actor, obs, act, masks, weights, g=spec.num_inputs, bound=bound,
        epochs=epochs, lr=lr, toggle_weight=toggle_weight,
    )


def load_producer_actor(path: str):
    """Rebuild the producer GaussianActor + action bound from a `save_policies`
    checkpoint, for scoring a saved clone's decoded fidelity (the gate uses the greedy
    closures from `load_policies`; this exposes the raw actor for `predict_actions`)."""
    import torch

    from .smoke_ppo import CHECKPOINT_VERSION, GaussianActor

    ckpt = torch.load(path, weights_only=True)
    if ckpt.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"checkpoint version {ckpt.get('version')!r} != {CHECKPOINT_VERSION}")
    prod = GaussianActor(ckpt["prod_obs"], ckpt["prod_act"])
    prod.load_state_dict(ckpt["prod"])
    return prod, ckpt["bound"]


def dagger_producer_first(
    configs,
    expert_prod: Policy,
    expert_order: Policy,
    base_order: Policy,
    *,
    dagger_iters: int = 3,
    seeds: "range | list[int]" = range(2),
    horizon: int = 100,
    epochs: int = 2000,
    lr: float = 3e-3,
    seed: int = 0,
    log_std_save: "float | None" = None,
    toggle_weight: float = 5.0,
    l1_weight: float = 2.0,
    pos_weight: float = 1.0,
    save: "str | None" = None,
    log=print,
) -> "str | None":
    """Producer-first DAgger with the decode-aware loss (ADR-0049 Phase 1 fix), over one
    or more same-shape scale topologies (Phase 2 multi-topology sampling).

    Round 0 is BC on the expert's own rollout under the BASE orderer, on every config;
    each later round rolls out the CURRENT clone producer (still under the base orderer
    — the gate's deployment condition), records the states it visits per config,
    relabels them with the expert, aggregates, and refits the producer from a fresh
    seeded init with the masked-MSE + BCE-toggle loss. Each config contributes its own
    decode masks (`_fit_arrays`), so varied/permuted topologies pull the net off slot
    memorization. The orderer is cloned plainly on the first config (it is the baseline
    in the gate's `clone+base` column). Saves a PPO-loadable checkpoint."""
    import torch

    from .smoke_ppo import (
        _build_clone_actors,
        _fit_clone,
        _fit_producer_clone,
        collect_bc_dataset,
        greedy_eval_policies,
        save_policies,
    )
    from .smoke_ppo import (  # init defaults (kept in sync with behavior_clone)
        DEFAULT_ORDER_LOG_STD,
        DEFAULT_PROD_LOG_STD,
        DEFAULT_PROD_REPLEN_BIAS,
        DEFAULT_PROD_REPLEN_LOG_STD,
        DEFAULT_PROD_RESTORE_BIAS,
    )
    from ..env.spaces import build_spec

    cfg_list = list(configs) if isinstance(configs, (list, tuple)) else [configs]
    specs = [build_spec(c) for c in cfg_list]
    g = specs[0].num_inputs
    samples_by_cfg = [
        collect_producer_samples(c, expert_prod, base_order, seeds, horizon)
        for c in cfg_list
    ]

    prod = order = None
    bound = 0.0
    for it in range(dagger_iters):
        torch.manual_seed(seed)  # refit from a fixed init on the grown dataset
        prod, order, bound = _build_clone_actors(
            cfg_list[0], prod_log_std=DEFAULT_PROD_LOG_STD,
            prod_replen_log_std=DEFAULT_PROD_REPLEN_LOG_STD,
            prod_replen_bias=DEFAULT_PROD_REPLEN_BIAS,
            prod_restore_bias=DEFAULT_PROD_RESTORE_BIAS, log_std_init=DEFAULT_ORDER_LOG_STD,
        )
        arrs = [_fit_arrays(sp, smp, l1_weight, pos_weight)
                for sp, smp in zip(specs, samples_by_cfg)]
        obs = np.concatenate([a[0] for a in arrs])
        act = np.concatenate([a[1] for a in arrs])
        masks = np.concatenate([a[2] for a in arrs])
        weights = np.concatenate([a[3] for a in arrs])
        loss = _fit_producer_clone(
            prod, obs, act, masks, weights, g=g, bound=bound, epochs=epochs, lr=lr,
            toggle_weight=toggle_weight,
        )
        matches = [decoded_command_match(sp, smp, predict_actions(prod, smp, bound))
                   for sp, smp in zip(specs, samples_by_cfg)]
        log(f"pf-dagger round {it}: loss {loss:.4f} "
            f"decoded_match[min/mean] {min(matches):.3f}/{np.mean(matches):.3f} "
            f"({len(obs)} rows, {len(cfg_list)} topos)")
        if it < dagger_iters - 1:
            clone_prod, _ = greedy_eval_policies(prod, order, bound)
            for i, c in enumerate(cfg_list):
                samples_by_cfg[i] += collect_producer_samples(
                    c, expert_prod, base_order, seeds, horizon, step_prod=clone_prod
                )

    # orderer: plain BC to the expert orderer on the first config (gate-irrelevant).
    od = collect_bc_dataset(cfg_list[0], expert_prod, expert_order, seeds, horizon)
    _fit_clone(order, od.order_obs, od.order_act, bound, epochs, lr)

    if log_std_save is not None:
        with torch.no_grad():
            prod.log_std.fill_(log_std_save)
            order.log_std.fill_(log_std_save)
    if save is not None:
        save_policies(prod, order, bound, save, config=cfg_list[0])
    return save


@dataclass
class FidelityStats:
    """Clone fidelity over one group of samples. `raw_mse`/`active_mse` are the
    optimizer's view (full vector vs decode-active dims only); the rest score the
    DECODED command — restoration confusion counts, replenish quantity accuracy on
    the expert's positive orders, and the decoded-command mismatch rate. `replen_*`
    are NaN when the group has no positive expert replenishment (e.g. primal layer 0)."""

    n: int
    raw_mse: float
    active_mse: float
    decoded_mismatch: float
    restore_tp: int
    restore_fp: int
    restore_fn: int
    replen_exact: float
    replen_within1: float
    raw_mse_by_dim: np.ndarray


@dataclass
class FidelityReport:
    overall: FidelityStats
    by_layer: dict[int, FidelityStats]


def _stats(
    spec: EnvSpec, samples: list[ProducerSample], preds: np.ndarray
) -> FidelityStats:
    from .smoke_ppo import producing_action_mask  # lazy: keep module torch-free

    g = spec.num_inputs
    n = len(samples)
    sq = np.zeros(spec.producing_action_dim, dtype=np.float64)
    active_sq = 0.0
    active_cnt = 0
    tp = fp = fn = 0
    rex = rw1 = rden = 0  # replenish exact / within-1 / denominator (expert positives)
    mismatch = 0
    for s, pred in zip(samples, preds):
        diff = (pred - s.expert_action).astype(np.float64)
        sq += diff * diff
        mask = producing_action_mask(spec, s.view)
        active_sq += float((diff * diff * mask).sum())
        active_cnt += int(mask.sum())

        want = decode_producing(spec, s.expert_action, s.view)
        got = decode_producing(spec, pred, s.view)
        mismatch += int(want != got)
        if want.do_restoration and got.do_restoration:
            tp += 1
        elif got.do_restoration and not want.do_restoration:
            fp += 1
        elif want.do_restoration and not got.do_restoration:
            fn += 1
        for prod, q in want.replenishments.items():  # expert-positive orders only
            rden += 1
            pq = got.replenishments.get(prod, 0)
            rex += int(pq == q)
            rw1 += int(abs(pq - q) <= 1)

    return FidelityStats(
        n=n,
        raw_mse=float(sq.sum() / (n * spec.producing_action_dim)) if n else 0.0,
        active_mse=active_sq / active_cnt if active_cnt else 0.0,
        decoded_mismatch=mismatch / n if n else 0.0,
        restore_tp=tp, restore_fp=fp, restore_fn=fn,
        replen_exact=(rex / rden) if rden else float("nan"),
        replen_within1=(rw1 / rden) if rden else float("nan"),
        raw_mse_by_dim=(sq / n) if n else sq,
    )


def clone_fidelity_report(
    spec: EnvSpec, samples: list[ProducerSample], pred_actions: np.ndarray
) -> FidelityReport:
    """Score a clone's predicted actions against the expert labels on raw MSE AND on
    decoded behavior, overall and per supply-chain layer (the depth-3 wall is L1, so
    per-layer breakdown localizes where fidelity is lost)."""
    preds = np.asarray(pred_actions, dtype=np.float32)
    layers = sorted({s.layer for s in samples})
    by_layer: dict[int, FidelityStats] = {}
    for lvl in layers:
        idx = [i for i, s in enumerate(samples) if s.layer == lvl]
        by_layer[lvl] = _stats(spec, [samples[i] for i in idx], preds[idx])
    return FidelityReport(overall=_stats(spec, samples, preds), by_layer=by_layer)
