"""M6 RL stage: MAPPO under CTDE on the team potential Φ (add-m6-joint-mpg-gnn D1 / D5, task 2.3).

`python -m tokenomics.train.m6_mappo --mode {calibrate,train}`

- **Actors:** the Stage I GNNs (`m6_gnn`), shared per role, each reading only its own ADR-0058 view, made
  stochastic for RL:
  - producer: 3 Bernoulli gates on the toggle logits; per indexed input row, a Gaussian on the replenish
    pre-sigmoid logit (one learned log-std, shared); the env action = sigmoid(sample) x BOUND;
  - orderer: per order slot, a categorical over its eligible producers (softmax of the logits), drawn with
    MWWk's keyed uniform (inverse CDF), so a rollout is reproducible from its DrawKey and the order is routed
    whole (ADR-0059).
  Deployment stays the Stage I decode (`gnn_producing_policy` / `gnn_ordering_policy`): gate logit > 0, the
  replenish mean, the keyed draw.
- **Critic (training only):** `CentralCritic` on the global view of a cycle: every producer's entity view
  and every stream view acting in that cycle, embedded by its own producer / orderer trunks (initialised from
  the imitated actors), mean- and max-pooled, plus the cycle fraction t / H. It is the CTDE critic of D5; the
  supply-link edges of D3's global graph are not yet in it (reported in the diff).
- **Reward:** the identical-interest team reward Φ_t (`env/potential.py`), scaled by `reward_scale`; every
  decision of cycle t gets that cycle's GAE advantage (an exact potential game: one shared Φ).
- **Update:** PPO clip on the joint log-probability of each decision, value loss on the cycles' returns,
  entropy bonus, global gradient clipping; synchronous data parallel (`_sync_steps`: the workers' shard
  gradients averaged in worker order, one Adam step; deterministic, as Stage I v2's fit).
- **Checkpoints and pins:** each iteration writes `latest.pt` (actors, critic, log-std, optimiser, rng
  state, iteration) and `curve.jsonl`; `spec.json` pins the config, the init models' SHA-256, the potential
  weights, the source commit and the RL pre-registration. A rerun on the same out dir resumes only if the
  pins match.
- **Monitor:** every `monitor_every` iterations, the deployed GNN stack against k5_rf5 + MWWk on the 16
  monitor world-episodes (Stage I's check: `m6_imitation.play_job` + M5's floors), g_tr / F_tr logged.
- `calibrate` (L1): k5_rf5 + MWWk on the training world-episodes at re-keys r1..r4 with the potential
  tracker; λ = σ_units / σ_rundown and μ = σ_units / σ_starving, pooled within-world-episode SDs of the
  episode term totals (M5's noise-scaled rule, training rows only).

A real `train` needs the frozen RL pre-registration (task 2.5b); until then only `--dry-run` runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch import nn

from ..env.potential import PotentialTracker, PotentialWeights
from ..env.spaces import ACTION_LOGIT_BOUND
from . import m6_gnn as g
from . import m6_imitation as im
from . import producer_es as pe

DRIVER = "tokenomics.train.m6_mappo"
RL_PREREG_SHA256: str | None = None  # task 2.5b: frozen before L2; real runs refuse until then
TRAIN_REPLICATE_BASE = 10  # rollout DrawKey replicates start here (the monitor uses 0 / 1)
CALIBRATION_REPLICATES = (1, 2, 3, 4)
LOGIT_EPS = 1e-4


@dataclass(frozen=True)
class MappoConfig:
    iterations: int = 40
    episodes_per_iter: int = 32
    horizon: int = pe.HORIZON
    gamma: float = 1.0  # finite horizon; t / H is a critic input
    gae_lambda: float = 0.95
    clip: float = 0.2
    ppo_epochs: int = 4
    batch: int = 64  # decisions per worker per step (cycles for the critic: the same)
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    grad_clip: float = 1.0
    init_log_std: float = -1.5
    reward_scale: float = 1.0 / 40.0  # ≈ 1 / base units per cycle at 12 producers
    seed: int = 0
    workers: int = 16
    monitor_every: int = 5


# --- the stochastic actors ---


class ActorExtras(nn.Module):
    """The RL-only actor parameters: the replenish noise's log-std (logit space)."""

    def __init__(self, init_log_std: float) -> None:
        super().__init__()
        self.log_std = nn.Parameter(torch.tensor(float(init_log_std)))


def _indexed(env: Any, inputs: Sequence[Any]) -> np.ndarray:
    return np.asarray([gp in env.spec_.input_index for gp in inputs], dtype=bool)


def producer_dist(pm: g.ProducerGNN, ex: ActorExtras, f: dict
                  ) -> tuple[torch.distributions.Bernoulli, torch.distributions.Normal]:
    gates, logit = pm.heads(f)
    return (torch.distributions.Bernoulli(logits=gates),
            torch.distributions.Normal(logit, ex.log_std.exp().expand_as(logit)))


def producer_logp(pm: g.ProducerGNN, ex: ActorExtras, f: dict, bits: np.ndarray, z: np.ndarray,
                  mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint log-probability and entropy of one producer command (masked input rows excluded)."""
    gd, rd = producer_dist(pm, ex, f)
    lp = gd.log_prob(torch.from_numpy(bits.astype(np.float32))).sum()
    ent = gd.entropy().sum()
    if mask.any():
        m = torch.from_numpy(mask)
        lp = lp + rd.log_prob(torch.from_numpy(z.astype(np.float32)))[m].sum()
        ent = ent + rd.entropy()[m].sum()
    return lp, ent


def orderer_logp(om: g.OrdererGNN, f: dict, picks: Sequence[int]
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint log-probability and entropy of one stream decision (slots with >= 2 eligible)."""
    lp, ent = torch.zeros(()), torch.zeros(())
    for lg, k in zip(om(f), picks):
        if k >= 0 and lg.shape[0] > 1:
            ls = torch.log_softmax(lg, 0)
            lp = lp + ls[k]
            ent = ent - (ls.exp() * ls).sum()
    return lp, ent


class RecordingPolicies:
    """The rollout's stochastic policies: they act like the deployed decode but sample, and record
    each decision (features, sampled action, old log-prob, cycle) and each cycle's critic input."""

    def __init__(self, pm: g.ProducerGNN, om: g.OrdererGNN, ex: ActorExtras,
                 rng: np.random.Generator) -> None:
        self.pm, self.om, self.ex, self.rng = pm, om, ex, rng
        self.decisions: list[dict] = []
        self.cycle_inputs: dict[int, dict[str, list]] = {}

    def _cycle(self, env: Any) -> dict[str, list]:
        return self.cycle_inputs.setdefault(int(env.current_cycle), {"producer": [], "orderer": []})

    def producing(self, env: Any, agent: str) -> np.ndarray:
        pid = env._role[agent][1]
        f = g.producer_features(env.system.producer_entity_view(pid), g.producer_context(env, pid))
        with torch.no_grad():
            gd, rd = producer_dist(self.pm, self.ex, f)
            p = torch.sigmoid(gd.logits).numpy()
            mu, sd = rd.loc.numpy(), float(rd.scale[0]) if rd.loc.shape[0] else 0.0
        bits = self.rng.random(3) < p
        z = mu + sd * self.rng.standard_normal(mu.shape[0])
        mask = _indexed(env, f["inputs"])
        with torch.no_grad():
            lp, _ = producer_logp(self.pm, self.ex, f, bits, z, mask)
        sp = env.spec_
        act = np.zeros(sp.producing_action_dim, dtype=np.float32)
        for gp, zz, ok in zip(f["inputs"], z, mask):
            if ok:
                act[sp.input_index[gp]] = float(1.0 / (1.0 + math.exp(-zz))) * ACTION_LOGIT_BOUND
        act[sp.num_inputs:sp.num_inputs + 3] = np.where(bits, ACTION_LOGIT_BOUND,
                                                         -ACTION_LOGIT_BOUND)
        c = int(env.current_cycle)
        self.decisions.append({"role": "p", "f": f, "bits": bits, "z": z, "mask": mask,
                               "logp": float(lp), "cycle": c})
        self._cycle(env)["producer"].append(f)
        return act

    def ordering(self, env: Any, agent: str) -> np.ndarray:
        from ..eval.rotation_routers import _flat, _matrix, _slots, _whole
        from ..eval.routing_diagnostics import draw_key, inverse_cdf, keyed_uniform

        key, cycle = draw_key(env), env.system.cycle_counter
        aid = env._role[agent][1]
        slots = _slots(env, agent)
        mat = _matrix(env)
        if not slots:
            return _flat(mat)
        f = g.orderer_features(env.system.product_stream_view(aid), slots, env.spec_.producer_ids)
        with torch.no_grad():
            logits = self.om(f)
        col = {p: i for i, p in enumerate(env.spec_.producer_ids)}
        picks: list[int] = []
        for (k, _oid, q, el), lg, cols in zip(slots, logits, f["elig"]):
            if el and len(cols):
                w = torch.softmax(lg, 0).numpy().astype(np.float64)
                pick = inverse_cdf(w, keyed_uniform(key, agent, cycle, k))
                mat[k, el] = _whole(el, el.index(col[f["pids"][cols[pick]]]), q)
                picks.append(pick)
            else:
                picks.append(-1)
        with torch.no_grad():
            lp, _ = orderer_logp(self.om, f, picks)
        c = int(env.current_cycle)
        self.decisions.append({"role": "o", "f": f, "picks": picks, "logp": float(lp),
                               "cycle": c})
        self._cycle(env)["orderer"].append(f)
        return _flat(mat)


# --- the centralised critic (training only) ---


class CentralCritic(nn.Module):
    """V(cycle) from the global view: all producers' and acting streams' embeddings, pooled."""

    def __init__(self, hidden: int = g.HIDDEN, rounds: int = g.ROUNDS) -> None:
        super().__init__()
        self.p = g.ProducerGNN(hidden, rounds)
        self.o = g.OrdererGNN(hidden, rounds)
        self.dp, self.do = 4 * hidden, 3 * hidden
        self.head = nn.Sequential(nn.Linear(2 * self.dp + 2 * self.do + 1, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, cyc: Mapping[str, list], tfrac: float) -> torch.Tensor:
        parts = []
        for feats, net, d in ((cyc["producer"], self.p, self.dp), (cyc["orderer"], self.o, self.do)):
            if feats:
                e = torch.stack([net.embed(f) for f in feats])
                parts += [e.mean(0), e.max(0).values]
            else:
                parts += [torch.zeros(d), torch.zeros(d)]
        return self.head(torch.cat([*parts, torch.tensor([float(tfrac)])])).squeeze(-1)

    @classmethod
    def from_actors(cls, pm: g.ProducerGNN, om: g.OrdererGNN) -> "CentralCritic":
        c = cls(pm.hidden, len(pm.upd))
        c.p.load_state_dict(pm.state_dict())
        c.o.load_state_dict(om.state_dict())
        return c


# --- rollouts ---


def _models_from(state: Mapping) -> tuple[g.ProducerGNN, g.OrdererGNN, ActorExtras, CentralCritic]:
    pm, om = g.ProducerGNN(), g.OrdererGNN()
    pm.load_state_dict(state["producer"])
    om.load_state_dict(state["orderer"])
    ex = ActorExtras(0.0)
    ex.load_state_dict(state["extras"])
    cr = CentralCritic()
    cr.load_state_dict(state["critic"])
    return pm, om, ex, cr


def rollout_job(job: dict) -> dict:
    """One keyed training episode with the stochastic actors and the potential tracker: the
    decisions, the per-cycle critic inputs, rewards Φ_t (scaled) and values V_t, and the stats."""
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    state = torch.load(job["snapshot"], weights_only=True)
    pm, om, ex, cr = _models_from(state)
    for m in (pm, om, cr):
        m.eval()
    env = im._episode_env(job)
    tracker = PotentialTracker(PotentialWeights(**job["weights"]))
    tracker.attach(env)
    rng = np.random.default_rng([int(job["seed"]), int(job["iteration"]),
                                 int(job["episode"]) % (1 << 31), int(job["replicate"])])
    pol = RecordingPolicies(pm, om, ex, rng)
    run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None)
    rewards = [r * job["reward_scale"] for r in tracker.rewards()]
    cycles = sorted(pol.cycle_inputs)
    h = int(job["horizon"])
    with torch.no_grad():
        values = [float(cr(pol.cycle_inputs[c], c / h)) for c in cycles]
    st = env.episode_stats
    return {"decisions": pol.decisions, "cycle_inputs": [pol.cycle_inputs[c] for c in cycles],
            "cycles": cycles, "rewards": rewards, "values": values, "horizon": h,
            "dead": bool(st["dead"]), "units": float(st["consumer_completed_units"]),
            "terms": tracker.totals(), "world": job["world"], "episode": job["episode"]}


def gae(rewards: Sequence[float], values: Sequence[float], gamma: float, lam: float
        ) -> tuple[list[float], list[float]]:
    """Advantages and returns over one episode's played cycles; the episode end (death or the
    horizon) is terminal: V after the last cycle = 0 (finite horizon, t / H in the critic)."""
    n = len(rewards)
    adv = [0.0] * n
    last = 0.0
    for t in reversed(range(n)):
        nxt = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, [a + v for a, v in zip(adv, values)]


def build_batch(trajs: Sequence[dict], cfg: MappoConfig) -> tuple[list[dict], list[dict]]:
    """(decision items with normalised advantages, critic items with returns)."""
    dec, cri = [], []
    for tr in trajs:
        if len(tr["rewards"]) != len(tr["values"]):
            raise AssertionError("rewards and critic inputs disagree on the played cycles")
        adv, ret = gae(tr["rewards"], tr["values"], cfg.gamma, cfg.gae_lambda)
        pos = {c: i for i, c in enumerate(tr["cycles"])}
        for d in tr["decisions"]:
            dec.append({**d, "adv": adv[pos[d["cycle"]]]})
        for c, ci, r in zip(tr["cycles"], tr["cycle_inputs"], ret):
            cri.append({"inputs": ci, "tfrac": c / tr["horizon"], "ret": r})
    a = np.asarray([d["adv"] for d in dec], dtype=np.float64)
    mu, sd = (float(a.mean()), float(a.std()) + 1e-8) if len(a) else (0.0, 1.0)
    for d in dec:
        d["adv"] = (d["adv"] - mu) / sd
    return dec, cri


# --- the PPO loss and the synchronous data-parallel update ---


def ppo_loss(mods: Mapping[str, nn.Module], dec: Sequence[dict], cri: Sequence[dict],
             cfg: MappoConfig) -> tuple[torch.Tensor, dict[str, float]]:
    pm, om, ex, cr = mods["producer"], mods["orderer"], mods["extras"], mods["critic"]
    pol, ent = torch.zeros(()), torch.zeros(())
    kl, clipped = 0.0, 0
    for d in dec:
        if d["role"] == "p":
            lp, e = producer_logp(pm, ex, d["f"], d["bits"], d["z"], d["mask"])
        else:
            lp, e = orderer_logp(om, d["f"], d["picks"])
        ratio = torch.exp(lp - d["logp"])
        a = float(d["adv"])
        pol = pol - torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
        ent = ent + e
        kl += float(d["logp"] - lp.detach())
        clipped += int(abs(float(ratio.detach()) - 1) > cfg.clip)
    vl = torch.zeros(())
    for c in cri:
        vl = vl + (cr(c["inputs"], c["tfrac"]) - c["ret"]) ** 2
    n, m = max(1, len(dec)), max(1, len(cri))
    loss = pol / n - cfg.ent_coef * ent / n + cfg.vf_coef * vl / m
    return loss, {"policy": float(pol.detach()) / n, "entropy": float(ent.detach()) / n,
                  "value": float(vl.detach()) / m,
                  "approx_kl": kl / n, "clip_frac": clipped / n}


def _flat_grad(params: Sequence[nn.Parameter]) -> torch.Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in params])


def _shard_worker(w: int, mods: Mapping[str, nn.Module], dec: list, cri: list, cfg: MappoConfig,
                  steps: int, epoch_seed: int, gbuf: torch.Tensor, sbuf: torch.Tensor,
                  ready: Any, done: Any) -> None:
    torch.set_num_threads(1)
    params = [p for m in mods.values() for p in m.parameters()]
    for ep in range(cfg.ppo_epochs):
        rng = np.random.default_rng([epoch_seed, ep, w])
        idec, icri = rng.permutation(len(dec)), rng.permutation(len(cri))
        for b in range(steps):
            ready.wait()
            for p in params:
                p.grad = None
            sl = slice(b * cfg.batch, (b + 1) * cfg.batch)
            loss, stats = ppo_loss(mods, [dec[i] for i in idec[sl]], [cri[i] for i in icri[sl]],
                                   cfg)
            if loss.requires_grad:
                loss.backward()
            gbuf[w] = _flat_grad(params)
            sbuf[w] = torch.tensor([stats[k] for k in STAT_KEYS])
            done.wait()


STAT_KEYS = ("policy", "entropy", "value", "approx_kl", "clip_frac")


def _sync_steps(mods: Mapping[str, nn.Module], opt: torch.optim.Optimizer, dec: list, cri: list,
                cfg: MappoConfig, epoch_seed: int, workers: int) -> dict[str, float]:
    """PPO epochs over (dec, cri): shard w = items[w::workers]; per step the shard gradients are
    averaged in worker order, clipped, and one Adam step updates the shared weights."""
    params = [p for m in mods.values() for p in m.parameters()]
    sizes = [p.numel() for p in params]
    steps = max(1, max(len(dec), len(cri)) // (cfg.batch * workers))
    tot = dict.fromkeys(STAT_KEYS, 0.0)
    if workers <= 1:
        rng_seed = epoch_seed
        for ep in range(cfg.ppo_epochs):
            rng = np.random.default_rng([rng_seed, ep, 0])
            idec, icri = rng.permutation(len(dec)), rng.permutation(len(cri))
            for b in range(steps):
                sl = slice(b * cfg.batch, (b + 1) * cfg.batch)
                opt.zero_grad()
                loss, stats = ppo_loss(mods, [dec[i] for i in idec[sl]],
                                       [cri[i] for i in icri[sl]], cfg)
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                    opt.step()
                for k in STAT_KEYS:
                    tot[k] += stats[k]
        return {k: v / (steps * cfg.ppo_epochs) for k, v in tot.items()}
    for m in mods.values():
        m.share_memory()
    gbuf = torch.zeros(workers, sum(sizes)).share_memory_()
    sbuf = torch.zeros(workers, len(STAT_KEYS)).share_memory_()
    ctx = torch.multiprocessing.get_context("fork")
    ready, done = ctx.Barrier(workers + 1, timeout=1800), ctx.Barrier(workers + 1, timeout=1800)
    procs = [ctx.Process(target=_shard_worker, args=(w, mods, dec[w::workers], cri[w::workers],
                                                     cfg, steps, epoch_seed, gbuf, sbuf, ready,
                                                     done), daemon=True)
             for w in range(workers)]
    for pr in procs:
        pr.start()
    try:
        for _ep in range(cfg.ppo_epochs):
            for _b in range(steps):
                ready.wait()
                done.wait()
                grad = gbuf[0].clone()
                for w in range(1, workers):
                    grad += gbuf[w]
                grad /= workers
                for p, chunk in zip(params, torch.split(grad, sizes)):
                    p.grad = chunk.view_as(p).clone()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                for i, k in enumerate(STAT_KEYS):
                    tot[k] += float(sbuf[:, i].mean())
    finally:
        for pr in procs:
            pr.join(timeout=60)
            if pr.is_alive():
                pr.terminate()
    if any(pr.exitcode for pr in procs):
        raise RuntimeError("a PPO shard worker failed")
    return {k: v / (steps * cfg.ppo_epochs) for k, v in tot.items()}


# --- checkpoints, pins ---


def _state(mods: Mapping[str, nn.Module]) -> dict:
    return {k: m.state_dict() for k, m in mods.items()}


def save_checkpoint(path: Path, mods: Mapping[str, nn.Module], opt: torch.optim.Optimizer,
                    iteration: int) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({**_state(mods), "optimizer": opt.state_dict(), "iteration": iteration,
                "norm_format": g.NORM_FORMAT}, tmp)
    tmp.replace(path)


def _open(out: Path, pins: dict) -> int:
    """Start or resume: returns the first iteration to run; refuses a pin mismatch."""
    out.mkdir(parents=True, exist_ok=True)
    spec = out / "spec.json"
    if spec.exists():
        have = json.loads(spec.read_text())["pins"]
        if have != pins:
            raise ValueError("the out dir's pins differ from this run's; use a fresh --out-dir")
        ck = out / "latest.pt"
        return int(torch.load(ck, weights_only=False)["iteration"]) + 1 if ck.exists() else 0
    spec.write_text(json.dumps({"driver": DRIVER, "pins": pins}, indent=1, sort_keys=True) + "\n")
    return 0


# --- modes ---


def iteration_jobs(ts: Any, train: Sequence, cfg: MappoConfig, it: int, snapshot: str,
                   weights: PotentialWeights, n: int | None = None) -> list[dict]:
    rng = np.random.default_rng([cfg.seed, it, 7])
    k = min(n or cfg.episodes_per_iter, len(train))
    pick = [train[i] for i in sorted(rng.choice(len(train), size=k, replace=False))]
    return [{**pe.make_job(ts, kind="mappo", producing=im.BASE_PRODUCER, theta=None,
                           generation=it, member="mappo", world=w, episode=e,
                           replicate=TRAIN_REPLICATE_BASE + it, horizon=cfg.horizon),
             "snapshot": snapshot, "weights": asdict(weights), "seed": cfg.seed,
             "iteration": it, "reward_scale": cfg.reward_scale} for w, e in pick]


def run_train(*, training_dir: str, atlas: str, init_models: str, out_dir: str,
              cfg: MappoConfig, weights: PotentialWeights, dry_run: bool,
              max_world_episodes: int | None = None, log: Callable[[str], None] = print) -> int:
    from ..eval.provenance import sha256_file

    if not dry_run:
        p = []
        if RL_PREREG_SHA256 is None:
            p.append("no frozen RL pre-registration (task 2.5b)")
        if sha256_file(init_models) != im.EXP0100_MODELS_SHA256:
            p.append("the init models are not EXP-0100's")
        if max_world_episodes:
            p.append("--max-world-episodes is set")
        im._refuse(p)
    torch.manual_seed(cfg.seed)
    ts, train, mon = im._training(training_dir)
    pm, om = im.load_models(init_models)
    mods: dict[str, nn.Module] = {"producer": pm, "orderer": om,
                                  "extras": ActorExtras(cfg.init_log_std),
                                  "critic": CentralCritic.from_actors(pm, om)}
    opt = torch.optim.Adam([
        {"params": [*pm.parameters(), *om.parameters(), *mods["extras"].parameters()],
         "lr": cfg.lr_actor},
        {"params": list(mods["critic"].parameters()), "lr": cfg.lr_critic}])
    out = Path(out_dir)
    pins = {"config": asdict(cfg), "weights": asdict(weights),
            "init_models_sha256": sha256_file(init_models), "rl_prereg": RL_PREREG_SHA256,
            "source_commit": im.SOURCE_COMMIT, "dry_run": dry_run,
            "max_world_episodes": max_world_episodes}
    start = _open(out, pins)
    if start:
        ck = torch.load(out / "latest.pt", weights_only=False)
        for k, m in mods.items():
            m.load_state_dict(ck[k])
        opt.load_state_dict(ck["optimizer"])
    cells = None
    base_rows: list[dict] = []
    for it in range(start, cfg.iterations):
        t0 = time.perf_counter()
        snap = out / "snapshot.pt"
        torch.save(_state(mods), snap)
        jobs = iteration_jobs(ts, train, cfg, it, str(snap), weights, max_world_episodes)
        trajs = im._pool(rollout_job, jobs, cfg.workers)
        dec, cri = build_batch(trajs, cfg)
        stats = _sync_steps(mods, opt, dec, cri, cfg, epoch_seed=cfg.seed * 1000 + it,
                            workers=cfg.workers)
        rec = {"iteration": it, "seconds": round(time.perf_counter() - t0, 1),
               "units": fmean(t["units"] for t in trajs),
               "deaths": sum(t["dead"] for t in trajs), "episodes": len(trajs),
               "decisions": len(dec), "cycles": len(cri),
               "terms": {k: fmean(t["terms"][k] for t in trajs) for k in trajs[0]["terms"]},
               "log_std": float(mods["extras"].log_std), **stats}
        if cfg.monitor_every and (it + 1) % cfg.monitor_every == 0:
            mdir = out / f"monitor_{it:03d}"
            mdir.mkdir(exist_ok=True)
            im.save_models(pm, om, mdir / "models.pt")
            rec["monitor"], cells, base_rows = monitor_read(
                ts, mon, atlas, str(mdir / "models.pt"), cfg, cells, base_rows,
                max_world_episodes)
        save_checkpoint(out / "latest.pt", mods, opt, it)
        with (out / "curve.jsonl").open("a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        log(f"[m6_mappo] it {it}: units {rec['units']:.1f} deaths {rec['deaths']} kl "
            f"{stats['approx_kl']:.4f} clip {stats['clip_frac']:.3f} value {stats['value']:.4f}"
            + (f" monitor g {rec['monitor']['g_tr']} F {rec['monitor']['F_tr']}"
               if "monitor" in rec else ""))
    im.save_models(pm, om, out / "models.pt")
    return 0


def monitor_read(ts: Any, mon: Sequence, atlas: str, models: str, cfg: MappoConfig,
                 cells: Any, base_rows: list[dict], max_world_episodes: int | None
                 ) -> tuple[dict, Any, list[dict]]:
    """The deployed GNN stack vs k5_rf5 + MWWk on the monitor world-episodes (Stage I's check)."""
    from ..eval.teacher_audit import load_atlas
    from .order_es_validate import monitor_cells
    from .producer_read import pilot_reading

    wes = mon[:max_world_episodes] if max_world_episodes else mon
    if cells is None:
        cells = monitor_cells(ts.training, load_atlas(atlas)["spreads"])
    if not base_rows:
        base_rows = im._pool(im.play_job, im._jobs(ts, wes, "monitor_base", im.MONITOR_REPLICATES,
                                                   cfg.horizon), cfg.workers)
    rows = im._pool(im.play_job, im._jobs(ts, wes, "monitor", im.MONITOR_REPLICATES, cfg.horizon,
                                          models), cfg.workers)
    r = pilot_reading(rows + base_rows, cells, 0)
    return ({"g_tr": r.get("g_tr"), "F_tr": r.get("F_tr"),
             "deaths": sum(x["dead"] == "true" for x in rows),
             "base_deaths": sum(x["dead"] == "true" for x in base_rows)}, cells, base_rows)


# --- L1: the potential weights (M5's noise-scaled rule) ---


def calibration_job(job: dict) -> dict:
    """k5_rf5 + MWWk on one keyed episode with the tracker: the episode's raw term totals."""
    from ..eval.baselines import producing_policy
    from ..eval.routing_audit import ordering_policy
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    env = im._episode_env(job)
    tr = PotentialTracker(PotentialWeights(**job["weights"]))
    tr.attach(env)
    run_episode(env, producing_policy(im.BASE_PRODUCER), ordering_policy(im.BASE_ORDERER),
                seed=job["episode"], action_seed=None)
    return {"world": job["world"], "episode": job["episode"], "replicate": job["replicate"],
            **tr.totals()}


def calibrate_weights(rows: Sequence[Mapping]) -> dict:
    """λ = σ_units / σ_rundown, μ = σ_units / σ_starving: pooled within-world-episode SDs of the
    episode totals across the base's re-keys (training rows only)."""
    from .order_es import _pooled_sd

    groups: dict[tuple, dict[str, list[float]]] = {}
    for r in rows:
        k = (r["world"], int(r["episode"]))
        for t in ("units", "rundown", "starving"):
            groups.setdefault(k, {}).setdefault(t, []).append(float(r[t]))
    sd = {t: _pooled_sd([v[t] for v in groups.values()]) for t in ("units", "rundown", "starving")}
    su = sd["units"][0]
    return {"lambda": su / sd["rundown"][0] if sd["rundown"][0] > 0 else None,
            "mu": su / sd["starving"][0] if sd["starving"][0] > 0 else None,
            "sigma": {t: v[0] for t, v in sd.items()}, "df": sd["units"][1],
            "world_episodes": len(groups), "episodes": len(rows),
            "base_units_mean": fmean(float(r["units"]) for r in rows)}


def run_calibrate(*, training_dir: str, out_dir: str, workers: int, horizon: int,
                  x: float, y: float, max_world_episodes: int | None, dry_run: bool) -> int:
    ts, train, _mon = im._training(training_dir)
    wes = train[:max_world_episodes] if max_world_episodes else train
    probe = PotentialWeights(0.0, 0.0, 0.0, x, y)
    jobs = [{**j, "weights": asdict(probe)} for r in CALIBRATION_REPLICATES
            for j in im._jobs(ts, wes, "calibrate", (r,), horizon)]
    rows = im._pool(calibration_job, jobs, workers)
    doc = {**calibrate_weights(rows), "x": x, "y": y, "dry_run": dry_run, "horizon": horizon,
           "replicates": list(CALIBRATION_REPLICATES), "source_commit": im.SOURCE_COMMIT,
           "nu_rule": "ν ≈ the base's mean episode units (a death costs about an episode)"}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    with (out / "rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[m6_mappo] calibration: lambda {doc['lambda']} mu {doc['mu']}", flush=True)
    return 0


def side_effect_xy() -> tuple[float, float]:
    from .order_es_worlds import X_RUNDOWN, Y_STARVATION

    return float(X_RUNDOWN), float(Y_STARVATION)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", required=True, choices=("calibrate", "train"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source-commit", default="")
    p.add_argument("--training-dir", required=True)
    p.add_argument("--atlas")
    p.add_argument("--init-models")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--horizon", type=int, default=pe.HORIZON)
    p.add_argument("--iterations", type=int)
    p.add_argument("--episodes-per-iter", type=int)
    p.add_argument("--monitor-every", type=int)
    p.add_argument("--max-world-episodes", type=int)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--nu", type=float, default=0.0)
    a = p.parse_args(argv)
    im.SOURCE_COMMIT = a.source_commit
    if not a.dry_run and len(a.source_commit) != 40:
        p.error("a real run needs a 40-hex --source-commit")
    x, y = side_effect_xy()
    if a.mode == "calibrate":
        return run_calibrate(training_dir=a.training_dir, out_dir=a.out_dir, workers=a.workers,
                             horizon=a.horizon, x=x, y=y,
                             max_world_episodes=a.max_world_episodes, dry_run=a.dry_run)
    over = {k: v for k, v in (("iterations", a.iterations),
                              ("episodes_per_iter", a.episodes_per_iter),
                              ("monitor_every", a.monitor_every)) if v is not None}
    cfg = MappoConfig(**{**asdict(MappoConfig()), "horizon": a.horizon, "workers": a.workers,
                         **over})
    return run_train(training_dir=a.training_dir, atlas=a.atlas, init_models=a.init_models,
                     out_dir=a.out_dir, cfg=cfg, weights=PotentialWeights(a.lam, a.mu, a.nu, x, y),
                     dry_run=a.dry_run, max_world_episodes=a.max_world_episodes)


if __name__ == "__main__":
    sys.exit(main())
