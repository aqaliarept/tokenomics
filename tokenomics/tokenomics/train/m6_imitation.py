"""M6 Stage I: imitation of k5_rf5 + MWWk by the fully learned GNN policies (add-m6-joint-mpg-gnn D4).

`python -m tokenomics.train.m6_imitation --mode {timing,collect,fit,check}`
- `timing` (L0, local, tiny): seconds per decision of the GNN policies against k5_rf5 / MWWk, and a horizon-H
  episode with the GNN stack, at 12 and 42 producers. Re-sizes sizing.md before any pod.
- `collect`: k5_rf5 + MWWk on M4's pinned training world-episodes (EXP-0058 `bar/`), excluding the 16 monitor
  world-episodes, at r0. It records every producer decision (features, k5_rf5's raw action as the label) and
  every order-slot decision (features, MWW's normalised weights over the eligible producers as a soft label).
  Each label is a deterministic function of the agent's view: tested.
- `fit`: behaviour cloning, with BCE on the 3 producer toggles, Huber on the replenish fractions, and KL on the
  orderer's slot distributions. v2 (EXP-0100, after DIAGNOSTICS 31): an in-model median / IQR normaliser over
  signed-log1p'd counts, fitted on the training samples and stored in the checkpoint; gradient clipping; a
  synchronous data-parallel fit (deterministic) instead of v1's Hogwild.
- `check` (the pre-registered imitation check, D4):
  - the GNN stack and k5_rf5 + MWWk on the 16 monitor world-episodes at replicates 0 / 1;
  - g_imit = the equal-weight mean of (imitated − bar) units, F_imit = M5's floors rule;
  - **`faithful`** if g_imit ≥ −F_imit and no extra deaths, else **`short`** (a stop: the user hears it before
    any RL);
  - reported: exact-command agreement, gate accuracy and replenish MAE (producer), with per toggle head and
    replenish row class (`producer_heads`), and KL and top-1 agreement (orderer), on the monitor episodes' labels;
  - reported only (v2): the role-swap arms, GNN producer + MWWk and k5_rf5 + GNN orderer, on the same keys.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from ..env.spaces import ACTION_LOGIT_BOUND
from . import m6_gnn as g
from . import producer_es as pe

BASE_PRODUCER = "entity_planner_buyahead_k5_rf5"
BASE_ORDERER = "mww_keyed"
IMIT_RULE = ("faithful if g_imit >= -F_imit (M5's floors rule on the 16 monitor world-episodes at "
             "replicates 0 / 1) and the imitated stack has no more deaths than k5_rf5 + MWWk; else "
             "short (stop: the user hears it before any RL)")
MONITOR_REPLICATES = (0, 1)
# the fit's pre-registered stopping rule and settings (v2, EXP-0100; v1 = EXP-0099's Hogwild fit).
# `batch` is per worker: a sync step averages `workers` shard gradients (global 64 x 16).
FIT = {"epochs": 20, "batch": 64, "lr": 3e-3, "seed": 0, "workers": 16, "mode": "auto",
       "clip_grad_norm": 1.0, "norm_decisions": 50_000, "sync_max_cost": 1.5}
MODE_RULE = ("auto: before the fit, one epoch of each mode on the full training samples (hogwild "
             "first, then sync; fresh models, the same seed); sync if its epoch costs <= "
             "sync_max_cost x hogwild's, else hogwild (with clipping); the mode and both times "
             "recorded in spec.json")
FIDELITY_METRICS = ("exact_command", "gate_accuracy", "replenish_mae_units", "orderer_kl",
                    "orderer_top1", "producer_heads")
# Reported-only role-swap arms (v2): each GNN actor beside the other base actor, same keys.
ARMS = {"monitor": ("gnn", "gnn"), "monitor_base": ("base", "base"),
        "swap_gnn_producer": ("gnn", "base"), "swap_gnn_orderer": ("base", "gnn")}
SWAP_ARMS = ("swap_gnn_producer", "swap_gnn_orderer")
TOGGLES = ("restore", "upgrade", "expand")
REGISTERED_ATLAS_HASH = "220886a7cf12aaac0cc0a48dc03c57f2dc2484bb2af6185a6160ec509c8a2a23"
PREREG_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v2.json"
SOURCE_COMMIT = ""  # set by `main` (--source-commit); recorded in every output
# v1 (m6_prereg_v1.json, 38db5c0c..., EXP-0099) stays frozen and unedited. v2's fit may reuse the
# samples v1's collect wrote (EXP-0099 artifacts/collect_sha256.txt): features and labels are
# unchanged (the normaliser lives in the model).
PREREG_V1_SHA256 = "38db5c0c96cfd2a7fa14eae1e3bf2832092738d1d13ce2fdcade25ac245c26f6"
EXP0099_SAMPLES = {
    "samples_train.pkl": "5c404d4e3fba082fde1762b221bb965360440bc421d98b38c7c817bb867dc10b",
    "samples_monitor.pkl": "0cbfa88646d25a99a4af895f314311cd4349feb4a62f734429f0ef2a68132abb"}
PREREG_SHA256: str | None = "2460cd07755930225837f996450f359e10480d495e3fc3e5be540d597a76d4a3"  # frozen before EXP-0100


# --- pre-registration (frozen before Stage I's data) ---


def registered_prereg() -> dict:
    return {
        "format": "m6-prereg-v2", "change": "add-m6-joint-mpg-gnn", "stage": "I (imitation), D4",
        "status": "frozen before EXP-0100's data; real collect / fit / check refuse any other file",
        "supersedes": {"m6_prereg_v1.json": PREREG_V1_SHA256,
                       "why": "EXP-0099 `short`; the orderer fit diverged on unscaled features "
                              "(DIAGNOSTICS 31)"},
        "decisions": "notes/user-decisions.md (2026-11-06): fully learned, from imitation; the "
                     "user hears a `short` reading before any RL",
        "base": {"producer": BASE_PRODUCER, "orderer": BASE_ORDERER},
        "data": {"training": {**pe.TRAINING_PINS, "world_episodes": "the 128 training world-"
                              "episodes minus the 16 monitor ones (112), replicate 0"},
                 "horizon": pe.HORIZON, "order_ttl": 2,
                 "labels": {"producer": "k5_rf5's raw action: 3 toggles (raw > 0) and per input "
                                        "row the replenish fraction raw / BOUND",
                            "orderer": "MWW's normalised makeable weights over the eligible "
                                       "producers per order slot; slots with one eligible "
                                       "producer dropped"},
                 "samples": {"reuse": "EXP-0099's collected samples (v1 collect), only at these "
                                      "SHA-256s; else a v2 collect",
                             "sha256": EXP0099_SAMPLES}},
        "model": {"hidden": g.HIDDEN, "rounds": g.ROUNDS,
                  "losses": "BCE (toggles) + Huber delta 0.1 (replenish fractions, masked) + KL "
                            "(orderer slots)",
                  "normaliser": {"format": g.NORM_FORMAT, "clip": g.NORM_CLIP,
                                 "transform": "signed log1p on the registered count columns",
                                 "orderer_log1p": {k: list(v) for k, v in g.ORDERER_LOG1P.items()},
                                 "producer_log1p": {k: list(v) for k, v in
                                                    g.PRODUCER_LOG1P.items()},
                                 "centre_scale": "median; IQR, IQR = 0 -> std, std = 0 -> 1",
                                 "fitted_on": "every k-th training decision per actor, k = "
                                              "ceil(n / norm_decisions); stored in the "
                                              "checkpoint"}},
        "fit": {**FIT, "stopping_rule": f"fixed {FIT['epochs']} epochs (no early stopping); the "
                                        "checkpoint SHA-256 is the pinned artifact",
                "sync": "synchronous data parallel: per step each worker's shard-mean gradient, "
                        "averaged in worker order, clipped to the global norm, one Adam step; "
                        "deterministic",
                "hogwild": "v1's fit (shared weights, a shard and an Adam per process) plus "
                           "per-step clipping; not bit-reproducible",
                "mode_rule": MODE_RULE},
        "check": {"rule": IMIT_RULE,
                  "monitor_set": "schedule.json `monitor` (16 world-episodes)",
                  "replicates": list(MONITOR_REPLICATES),
                  "comparison": "the GNN stack vs k5_rf5 + MWWk at the same DrawKeys",
                  "reported_only_arms": {a: {"producer": ARMS[a][0], "orderer": ARMS[a][1]}
                                         for a in SWAP_ARMS},
                  "swap_reading": "each swap arm - k5_rf5 + MWWk with the same floor rule; "
                                  "deaths; reported, never gating",
                  "floor": "M5's floors rule (producer_read.pilot_reading: max(the atlas-spread "
                           "floor at the read's counts, own 3 SE), the n = 1 rule); atlas spreads "
                           "from the registered atlas v2",
                  "floor_source": {"atlas_content_hash": REGISTERED_ATLAS_HASH}},
        "reported": {"fidelity_metrics": list(FIDELITY_METRICS),
                     "on": "the training and the monitor episodes' labels",
                     "also": "units of both stacks and deaths"}}


def freeze_prereg(path: Path = PREREG_PATH) -> tuple[str, str]:
    from ..eval.provenance import sha256_file

    text = json.dumps(registered_prereg(), indent=1, sort_keys=True) + "\n"
    if not path.exists():
        path.write_text(text)
        return "FROZEN", sha256_file(str(path))
    return ("MATCH" if json.loads(path.read_text()) == registered_prereg() else "MISMATCH"), \
        sha256_file(str(path))


def prereg_problems(path: Path = PREREG_PATH) -> list[str]:
    from ..eval.provenance import sha256_file

    if PREREG_SHA256 is None:
        return ["no pinned M6 pre-registration (m6_prereg_v2.json; `--mode freeze`)"]
    if not path.exists() or sha256_file(str(path)) != PREREG_SHA256:
        return [f"{path.name} does not match the pinned SHA-256 {PREREG_SHA256[:12]}"]
    if json.loads(path.read_text()) != registered_prereg():
        return ["the pre-registration differs from the code"]
    return []


def real_problems(*, horizon: int, max_world_episodes: int | None, ts: Any = None,
                  atlas: Any = None, epochs: int | None = None, workers: int | None = None,
                  seed: int | None = None) -> list[str]:
    p = prereg_problems()
    if horizon != pe.HORIZON:
        p.append(f"horizon {horizon} != {pe.HORIZON}")
    if max_world_episodes:
        p.append("--max-world-episodes is set")
    if ts is not None and ts.pins != pe.TRAINING_PINS:
        p.append("the training set is not M4's pinned set")
    if atlas is not None and atlas.get("content_hash") != REGISTERED_ATLAS_HASH:
        p.append("the atlas is not the registered atlas v2")
    for k, v in (("epochs", epochs), ("workers", workers), ("seed", seed)):
        if v is not None and v != FIT[k]:
            p.append(f"--{k} {v} != the registered {FIT[k]}")
    return p


def _refuse(problems: list[str]) -> None:
    if problems:
        raise ValueError("not a valid real run (" + "; ".join(problems) + "); pass --dry-run")


# --- labels (deterministic functions of the agent's view; tested) ---


def producer_label(env: Any, pid: str, action: np.ndarray, inputs: Sequence[Any]
                   ) -> dict[str, np.ndarray]:
    """k5_rf5's raw action as the label: 3 toggles (raw > 0 fires) and, per input row, the
    replenish fraction raw / BOUND in [0, 1] (rows the env does not index are masked)."""
    sp = env.spec_
    tog = (np.asarray(action[sp.num_inputs:sp.num_inputs + 3]) > 0).astype(np.float32)
    frac, mask = [], []
    for gp in inputs:
        if gp in sp.input_index:
            frac.append(float(np.clip(action[sp.input_index[gp]] / ACTION_LOGIT_BOUND, 0, 1)))
            mask.append(1.0)
        else:
            frac.append(0.0)
            mask.append(0.0)
    return {"toggles": tog, "replenish": np.asarray(frac, np.float32),
            "mask": np.asarray(mask, np.float32)}


def orderer_labels(env: Any, slots: Sequence, feats: dict[str, Any]) -> list[np.ndarray]:
    """Per slot, MWW's normalised weights over the eligible producers, in the features' column
    order (MWWk draws from exactly these)."""
    from ..eval.routing_diagnostics import makeable_weights

    pids = env.spec_.producer_ids
    out = []
    for (k, oid, q, el), cols in zip(slots, feats["elig"]):
        if not el or not cols:
            out.append(np.zeros(0, np.float32))
            continue
        w = dict(zip((pids[i] for i in el), makeable_weights(env, oid, el)))
        v = np.asarray([w[feats["pids"][c]] for c in cols], np.float64)
        out.append((v / v.sum()).astype(np.float32))
    return out


# --- recording episodes ---


def _episode_env(job: dict) -> Any:
    from ..eval.joint_stack import joint_stack_env
    from ..eval.routing_audit import job_config
    from ..eval.routing_diagnostics import DrawKey, set_draw_key

    env = joint_stack_env(job_config(job), job["horizon"])
    set_draw_key(env, DrawKey(str(job["world"]), int(job["episode"]), int(job["replicate"])))
    return env


def record_job(job: dict) -> dict:
    """Play k5_rf5 + MWWk on one keyed world-episode, recording every decision's features and
    label; returns {"producer": [...], "orderer": [...], "units", "dead"}."""
    from ..eval.baselines import producing_policy
    from ..eval.rotation_routers import _slots
    from ..eval.routing_audit import ordering_policy
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    env = _episode_env(job)
    base_p, base_o = producing_policy(BASE_PRODUCER), ordering_policy(BASE_ORDERER)
    prod_s: list = []
    ord_s: list = []

    def rec_p(e: Any, agent: str) -> np.ndarray:
        pid = e._role[agent][1]
        f = g.producer_features(e.system.producer_entity_view(pid), g.producer_context(e, pid))
        a = base_p(e, agent)
        prod_s.append((f, producer_label(e, pid, a, f["inputs"])))
        return a

    def rec_o(e: Any, agent: str) -> np.ndarray:
        aid = e._role[agent][1]
        slots = _slots(e, agent)
        if slots:
            f = g.orderer_features(e.system.product_stream_view(aid), slots, e.spec_.producer_ids)
            ys = orderer_labels(e, slots, f)
            if any(len(y) > 1 for y in ys):  # a single eligible producer carries no label signal
                ord_s.append((f, ys))
        return base_o(e, agent)

    run_episode(env, rec_p, rec_o, seed=job["episode"], action_seed=None)
    st = env.episode_stats
    return {"producer": prod_s, "orderer": ord_s, "units": float(st["consumer_completed_units"]),
            "dead": bool(st["dead"]), "job_id": job.get("job_id")}


def play_job(job: dict) -> dict:
    """One keyed episode of arm `job["kind"]` (ARMS: who plays producer / orderer)."""
    from ..eval.baselines import producing_policy
    from ..eval.routing_audit import ordering_policy
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    env = _episode_env(job)
    who_p, who_o = ARMS[job["kind"]]
    pm, om = load_models(job["models"]) if "gnn" in (who_p, who_o) else (None, None)
    p = g.gnn_producing_policy(pm) if who_p == "gnn" else producing_policy(BASE_PRODUCER)
    o = g.gnn_ordering_policy(om) if who_o == "gnn" else ordering_policy(BASE_ORDERER)
    run_episode(env, p, o, seed=job["episode"], action_seed=None)
    st = env.episode_stats
    return {"world": job["world"], "episode": str(job["episode"]),
            "replicate": str(job["replicate"]), "units": float(st["consumer_completed_units"]),
            "dead": "true" if st["dead"] else "false", "kind": job["kind"],
            "generation": str(job["generation"])}


def _jobs(ts: Any, wes: Sequence, kind: str, replicates: Sequence[int], horizon: int,
          models: str | None = None) -> list[dict]:
    return [{**pe.make_job(ts, kind=kind, producing=BASE_PRODUCER, theta=None, generation=0,
                           member=kind, world=w, episode=e, replicate=r, horizon=horizon),
             "models": models} for w, e in wes for r in replicates]


def _pool(fn: Any, jobs: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                             initializer=pe._init_worker) as ex:
        return list(ex.map(fn, jobs))


# --- fit (behaviour cloning) ---


def bc_losses(pm: g.ProducerGNN, om: g.OrdererGNN, prod: list, orde: list
              ) -> tuple[torch.Tensor, torch.Tensor]:
    lp = torch.zeros(())
    for f, y in prod:
        gates, rep = pm(f)
        lp = lp + torch.nn.functional.binary_cross_entropy_with_logits(
            gates, torch.from_numpy(y["toggles"]), reduction="sum")
        if rep.shape[0]:
            m = torch.from_numpy(y["mask"])
            lp = lp + (torch.nn.functional.huber_loss(rep, torch.from_numpy(y["replenish"]),
                                                      reduction="none", delta=0.1) * m).sum()
    lo = torch.zeros(())
    for f, ys in orde:
        for lg, y in zip(om(f), ys):
            if len(y):
                lo = lo - (torch.from_numpy(y) * torch.log_softmax(lg, 0)).sum()
    return lp / max(1, len(prod)), lo / max(1, len(orde))


def _epochs(pm: g.ProducerGNN, om: g.OrdererGNN, prod: list, orde: list, epochs: int, batch: int,
            lr: float, seed: int, log: Any, clip: float | None = None) -> list[dict]:
    torch.set_num_threads(1)
    params = [*pm.parameters(), *om.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    curve = []
    for ep in range(epochs):
        ip, io = rng.permutation(len(prod)), rng.permutation(len(orde))
        n = max(1, max(len(prod), len(orde)) // batch)
        tot = [0.0, 0.0]
        for b in range(n):
            bp = [prod[i] for i in ip[b * batch:(b + 1) * batch]]
            bo = [orde[i] for i in io[b * batch:(b + 1) * batch]]
            lp, lo = bc_losses(pm, om, bp, bo)
            opt.zero_grad()
            if (lp + lo).requires_grad:
                (lp + lo).backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(params, clip)
                opt.step()
            tot[0] += float(lp.detach())
            tot[1] += float(lo.detach())
        curve.append({"epoch": ep + 1, "producer_loss": tot[0] / n, "orderer_loss": tot[1] / n})
        log(f"[m6_imitation] epoch {ep + 1}: producer {tot[0] / n:.4f} orderer {tot[1] / n:.4f}")
    return curve


def _hogwild(args: tuple) -> list[dict]:
    pm, om, prod, orde, epochs, batch, lr, seed, clip = args
    return _epochs(pm, om, prod, orde, epochs, batch, lr, seed, lambda s: None, clip)


def _sync_worker(w: int, pm: g.ProducerGNN, om: g.OrdererGNN, prod: list, orde: list,
                 epochs: int, steps: int, batch: int, seed: int, gbuf: torch.Tensor,
                 lbuf: torch.Tensor, ready: Any, done: Any) -> None:
    """One shard of the synchronous fit: per step its shard-mean gradient into `gbuf[w]`."""
    torch.set_num_threads(1)
    params = [*pm.parameters(), *om.parameters()]
    for ep in range(epochs):
        rng = np.random.default_rng([seed, ep, w])
        ip, io = rng.permutation(len(prod)), rng.permutation(len(orde))
        for b in range(steps):
            ready.wait()  # the parent's step is applied to the shared weights
            for p in params:
                p.grad = None
            lp, lo = bc_losses(pm, om, [prod[i] for i in ip[b * batch:(b + 1) * batch]],
                               [orde[i] for i in io[b * batch:(b + 1) * batch]])
            if (lp + lo).requires_grad:
                (lp + lo).backward()
            gbuf[w] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)
                                  ).reshape(-1) for p in params])
            lbuf[w, 0], lbuf[w, 1] = float(lp.detach()), float(lo.detach())
            done.wait()


def _sync_fit(pm: g.ProducerGNN, om: g.OrdererGNN, prod: list, orde: list, epochs: int,
              batch: int, lr: float, seed: int, workers: int, clip: float | None, log: Any
              ) -> list[dict]:
    """Synchronous data-parallel fit (v2): shard w = samples[w::workers]; per step the workers'
    gradients are averaged in worker order, clipped, and one Adam step is applied to the shared
    weights. Deterministic given the seed (BLAS 1 thread, fixed reduction order)."""
    torch.set_num_threads(1)
    params = [*pm.parameters(), *om.parameters()]
    sizes = [p.numel() for p in params]
    pm.share_memory()
    om.share_memory()
    gbuf = torch.zeros(workers, sum(sizes)).share_memory_()
    lbuf = torch.zeros(workers, 2).share_memory_()
    ctx = torch.multiprocessing.get_context("fork")
    ready, done = ctx.Barrier(workers + 1, timeout=1800), ctx.Barrier(workers + 1, timeout=1800)
    steps = max(1, max(len(prod), len(orde)) // (batch * workers))
    procs = [ctx.Process(target=_sync_worker, args=(
        w, pm, om, prod[w::workers], orde[w::workers], epochs, steps, batch, seed, gbuf, lbuf,
        ready, done), daemon=True) for w in range(workers)]
    for pr in procs:
        pr.start()
    opt = torch.optim.Adam(params, lr=lr)
    curve = []
    try:
        for ep in range(epochs):
            tot = [0.0, 0.0]
            t0 = time.perf_counter()
            for _b in range(steps):
                ready.wait()
                done.wait()
                grad = gbuf[0].clone()
                for w in range(1, workers):
                    grad += gbuf[w]
                grad /= workers
                for p, chunk in zip(params, torch.split(grad, sizes)):
                    p.grad = chunk.view_as(p).clone()
                if clip:
                    torch.nn.utils.clip_grad_norm_(params, clip)
                opt.step()
                tot[0] += float(lbuf[:, 0].mean())
                tot[1] += float(lbuf[:, 1].mean())
            curve.append({"epoch": ep + 1, "producer_loss": tot[0] / steps,
                          "orderer_loss": tot[1] / steps,
                          "seconds": round(time.perf_counter() - t0, 3)})
            log(f"[m6_imitation] sync epoch {ep + 1}: producer {tot[0] / steps:.4f} orderer "
                f"{tot[1] / steps:.4f} ({curve[-1]['seconds']} s)")
    finally:
        for pr in procs:
            pr.join(timeout=60)
            if pr.is_alive():
                pr.terminate()
    bad = [pr.exitcode for pr in procs if pr.exitcode]
    if bad:
        raise RuntimeError(f"sync fit worker failed (exit codes {bad})")
    return curve


def _norm_feats(samples: list, cap: int) -> list[dict]:
    k = max(1, -(-len(samples) // cap))  # ceil(n / cap)
    return [f for f, _y in samples[::k]]


def fit(samples: dict, epochs: int, batch: int = 64, lr: float = 3e-3, seed: int = 0,
        log: Any = print, workers: int = 1, mode: str = "sync", clip: float | None = 1.0,
        norm_decisions: int = 50_000
        ) -> tuple[g.ProducerGNN, g.OrdererGNN, list[dict], dict]:
    """Behaviour cloning (v2). The normalisers are fitted first on every k-th training decision
    (DIAGNOSTICS 31). `workers` > 1: `mode` "sync" (deterministic data parallel, `_sync_fit`) or
    "hogwild" (shared weights, one shard and Adam per process; not bit-reproducible). Returns
    the models, the loss curve and {"normaliser": its record, "train_seconds": the training
    wall time without the normaliser fit}."""
    torch.manual_seed(seed)
    pm, om = g.ProducerGNN(), g.OrdererGNN()
    prod, orde = samples["producer"], samples["orderer"]
    info: dict[str, Any] = {"normaliser": {
        "producer": pm.norm.fit(_norm_feats(prod, norm_decisions)),
        "orderer": om.norm.fit(_norm_feats(orde, norm_decisions))}}
    t0 = time.perf_counter()
    if workers <= 1:
        curve = _epochs(pm, om, prod, orde, epochs, batch, lr, seed, log, clip)
    elif mode == "sync":
        curve = _sync_fit(pm, om, prod, orde, epochs, batch, lr, seed, workers, clip, log)
    else:
        curve = _hogwild_fit(pm, om, prod, orde, epochs, batch, lr, seed, workers, clip, log)
    info["train_seconds"] = round(time.perf_counter() - t0, 3)
    return pm, om, curve, info


def _hogwild_fit(pm: g.ProducerGNN, om: g.OrdererGNN, prod: list, orde: list, epochs: int,
                 batch: int, lr: float, seed: int, workers: int, clip: float | None, log: Any
                 ) -> list[dict]:
    pm.share_memory()
    om.share_memory()
    ctx = torch.multiprocessing.get_context("fork")
    shards = [(pm, om, prod[i::workers], orde[i::workers], epochs, batch, lr, seed + i, clip)
              for i in range(workers)]
    with ctx.Pool(workers) as pool:
        curves = pool.map(_hogwild, shards)
    curve = [{"epoch": e + 1,
              "producer_loss": fmean(c[e]["producer_loss"] for c in curves),
              "orderer_loss": fmean(c[e]["orderer_loss"] for c in curves)} for e in range(epochs)]
    log(f"[m6_imitation] hogwild x{workers}: final producer {curve[-1]['producer_loss']:.4f} "
        f"orderer {curve[-1]['orderer_loss']:.4f}")
    return curve


def choose_mode(samples: dict, workers: int, seed: int, log: Any = print) -> dict:
    """MODE_RULE: one epoch of each mode (hogwild, then sync), timed without the normaliser fit."""
    t = {}
    for m in ("hogwild", "sync"):
        t[m] = fit(samples, 1, batch=FIT["batch"], lr=FIT["lr"], seed=seed, log=lambda s: None,
                   workers=workers, mode=m, clip=FIT["clip_grad_norm"],
                   norm_decisions=FIT["norm_decisions"])[3]["train_seconds"]
    ratio = t["sync"] / max(t["hogwild"], 1e-9)
    mode = "sync" if ratio <= FIT["sync_max_cost"] else "hogwild"
    log(f"[m6_imitation] fit mode probe: hogwild {t['hogwild']:.1f} s, sync {t['sync']:.1f} s "
        f"(x{ratio:.2f}) -> {mode}")
    return {"rule": MODE_RULE, "epoch_seconds": t, "sync_over_hogwild": ratio, "mode": mode}


def _head_fidelity(conf: np.ndarray, rows: dict[str, list], split: dict[str, int], n: int) -> dict:
    """Per toggle head (restore / upgrade / expand) and replenish row class; which part breaks
    an exact command (v2 report)."""
    heads = {}
    for i, name in enumerate(TOGGLES):
        tp, fp, fn, tn = (int(x) for x in conf[i])
        heads[name] = {"label_rate": (tp + fn) / max(1, n), "pred_rate": (tp + fp) / max(1, n),
                       "accuracy": (tp + tn) / max(1, n), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    rep = {}
    for cls, du in rows.items():
        a = np.asarray(du, dtype=np.float64)
        rep[cls] = {"rows": int(a.size), "exact_share": float((a == 0).mean()) if a.size else None,
                    "within_1_unit_share": float((a <= 1).mean()) if a.size else None,
                    "mae_units": float(a.mean()) if a.size else None}
    return {"toggles": heads, "replenish_rows": rep,
            "commands": {k: v / max(1, n) for k, v in split.items()}}


def agreement(pm: g.ProducerGNN, om: g.OrdererGNN, prod: list, orde: list) -> dict:
    """Fidelity metrics on labelled decisions (reported, D4). Exact command = every toggle and
    every replenish row's rounded units equal the label's."""
    exact = gates_ok = n_gate = 0
    mae: list[float] = []
    conf = np.zeros((3, 4), dtype=np.int64)  # tp, fp, fn, tn per head
    rows: dict[str, list] = {"label_zero": [], "label_interior": [], "label_at_bound": []}
    split = {"exact": 0, "toggles_only_wrong": 0, "replenish_only_wrong": 0, "both_wrong": 0,
             "replenish_only_wrong_within_1_unit": 0}
    with torch.no_grad():
        for f, y in prod:
            gates, rep = pm(f)
            tg = (gates.numpy() > 0).astype(np.float32)
            yt = y["toggles"]
            gates_ok += int((tg == yt).sum())
            n_gate += 3
            conf[:, 0] += (tg == 1) & (yt == 1)
            conf[:, 1] += (tg == 1) & (yt == 0)
            conf[:, 2] += (tg == 0) & (yt == 1)
            conf[:, 3] += (tg == 0) & (yt == 0)
            mr = f["global"][0][9]  # log1p(max_replenish)
            units = np.expm1(mr)
            t_ok = bool((tg == yt).all())
            r_ok, r_near = True, True
            if rep.shape[0]:
                m = y["mask"] > 0
                du = np.abs(np.round(rep.numpy() * units) - np.round(y["replenish"] * units))[m]
                lab = y["replenish"][m]
                mae += du.tolist()
                for d, lv in zip(du.tolist(), lab.tolist()):
                    rows["label_zero" if lv == 0 else "label_at_bound" if lv >= 1
                         else "label_interior"].append(d)
                r_ok, r_near = bool((du == 0).all()), bool((du <= 1).all())
            ok = t_ok and r_ok
            exact += ok
            key = ("exact" if ok else "toggles_only_wrong" if r_ok else
                   "replenish_only_wrong" if t_ok else "both_wrong")
            split[key] += 1
            split["replenish_only_wrong_within_1_unit"] += int(t_ok and not r_ok and r_near)
        kl, top1, n_s = [], 0, 0
        for f, ys in orde:
            for lg, y in zip(om(f), ys):
                if len(y):
                    p = torch.softmax(lg, 0).numpy()
                    kl.append(float((y * (np.log(np.clip(y, 1e-9, 1)) - np.log(np.clip(p, 1e-9, 1))
                                          )).sum()))
                    top1 += int(np.argmax(p) == np.argmax(y))
                    n_s += 1
    return {"producer_decisions": len(prod), "exact_command": exact / max(1, len(prod)),
            "gate_accuracy": gates_ok / max(1, n_gate),
            "replenish_mae_units": fmean(mae) if mae else 0.0, "order_slots": n_s,
            "orderer_kl": fmean(kl) if kl else 0.0, "orderer_top1": top1 / max(1, n_s),
            "producer_heads": _head_fidelity(conf, rows, split, len(prod))}


def save_models(pm: g.ProducerGNN, om: g.OrdererGNN, path: Path) -> None:
    torch.save({"producer": pm.state_dict(), "orderer": om.state_dict(),
                "hidden": g.HIDDEN, "rounds": g.ROUNDS, "norm_format": g.NORM_FORMAT}, path)


def load_models(path: str | Path) -> tuple[g.ProducerGNN, g.OrdererGNN]:
    d = torch.load(path, weights_only=True)
    pm, om = g.ProducerGNN(d["hidden"], d["rounds"]), g.OrdererGNN(d["hidden"], d["rounds"])
    # a pre-normaliser checkpoint (EXP-0099) loads with the identity normaliser
    strict = d.get("norm_format") == g.NORM_FORMAT
    if not strict and "norm_format" in d:
        raise ValueError(f"unknown normaliser format {d['norm_format']}")
    pm.load_state_dict(d["producer"], strict=strict)
    om.load_state_dict(d["orderer"], strict=strict)
    if not strict:
        pm.norm.clip = om.norm.clip = None  # raw features, exactly as EXP-0099 played them
    return pm, om


# --- modes ---


def _training(training_dir: str) -> tuple[Any, list, list]:
    ts = pe.load_training(training_dir)
    mon = pe.monitor_set(ts)
    train = [we for we in ts.world_episodes if we not in set(mon)]
    return ts, train, mon


def run_collect(training_dir: str, out_dir: str, workers: int, horizon: int,
                max_world_episodes: int | None, dry_run: bool = False) -> int:
    ts, train, mon = _training(training_dir)
    if not dry_run:
        _refuse(real_problems(horizon=horizon, max_world_episodes=max_world_episodes, ts=ts))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = {"train": train[:max_world_episodes] if max_world_episodes else train,
            "monitor": mon[:max(1, max_world_episodes // 4)] if max_world_episodes else mon}
    for name, wes in sets.items():
        res = _pool(record_job, _jobs(ts, wes, f"imit_{name}", (0,), horizon), workers)
        samples = {"producer": [s for r in res for s in r["producer"]],
                   "orderer": [s for r in res for s in r["orderer"]]}
        with (out / f"samples_{name}.pkl").open("wb") as fh:
            pickle.dump(samples, fh)
        print(f"[m6_imitation] {name}: {len(wes)} world-episodes, {len(samples['producer'])} "
              f"producer and {len(samples['orderer'])} orderer decisions", flush=True)
    (out / "collect.json").write_text(json.dumps(
        {"training_pins": ts.pins, "horizon": horizon, "train": len(sets["train"]),
         "dry_run": dry_run, "prereg": PREREG_SHA256, "source_commit": SOURCE_COMMIT,
         "monitor": len(sets["monitor"]), "base": [BASE_PRODUCER, BASE_ORDERER]}, indent=1) + "\n")
    return 0


def sample_problems(collect_dir: Path, meta: dict) -> list[str]:
    """A real v2 fit reads a real v2 collect, or v1's (EXP-0099) samples at their pinned SHAs."""
    from ..eval.provenance import sha256_file

    if meta.get("dry_run"):
        return ["the collect run was a dry run"]
    if meta.get("prereg") == PREREG_SHA256:
        return []
    if meta.get("prereg") != PREREG_V1_SHA256:
        return ["the samples come from neither the v2 nor the v1 pre-registration"]
    return [f"{n} does not match EXP-0099's pinned SHA-256" for n, h in EXP0099_SAMPLES.items()
            if sha256_file(str(collect_dir / n)) != h]


def run_fit(collect_dir: str, out_dir: str, epochs: int, seed: int, workers: int = 1,
            dry_run: bool = False, mode: str = FIT["mode"]) -> int:
    c = Path(collect_dir)
    meta = json.loads((c / "collect.json").read_text())
    if not dry_run:
        _refuse(real_problems(horizon=meta["horizon"], max_world_episodes=None, epochs=epochs,
                              workers=workers, seed=seed)
                + ([] if mode == FIT["mode"] else [f"--fit-mode {mode} != {FIT['mode']}"])
                + sample_problems(c, meta))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (c / "samples_train.pkl").open("rb") as fh:
        tr = pickle.load(fh)
    with (c / "samples_monitor.pkl").open("rb") as fh:
        mo = pickle.load(fh)
    probe = choose_mode(tr, workers, seed) if mode == "auto" else None
    ran = probe["mode"] if probe else mode
    t0 = time.perf_counter()
    pm, om, curve, info = fit(tr, epochs, batch=FIT["batch"], lr=FIT["lr"], seed=seed,
                              workers=workers, mode=ran, clip=FIT["clip_grad_norm"],
                              norm_decisions=FIT["norm_decisions"])
    fit_s = time.perf_counter() - t0
    save_models(pm, om, out / "models.pt")
    from ..eval.provenance import sha256_file

    spec = {"fit_mode": ran, "mode_probe": probe, "requested_mode": mode, "fit": FIT,
            "epochs": epochs, "seed": seed, "workers": workers, "dry_run": dry_run,
            "prereg": PREREG_SHA256, "source_commit": SOURCE_COMMIT,
            "samples_from": {"prereg": meta.get("prereg"), "source_commit":
                             meta.get("source_commit")}}
    (out / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    doc = {"epochs": epochs, "seed": seed, "workers": workers, "mode": ran, "mode_probe": probe,
           "curve": curve, "fit_seconds": round(fit_s, 1),
           "train_seconds": info["train_seconds"], "dry_run": dry_run,
           "normaliser": info["normaliser"],
           "samples_from": {"prereg": meta.get("prereg"), "source_commit":
                            meta.get("source_commit")},
           "prereg": PREREG_SHA256, "models_sha256": sha256_file(str(out / "models.pt")),
           "source_commit": SOURCE_COMMIT,
           "train_agreement": agreement(pm, om, tr["producer"], tr["orderer"]),
           "monitor_agreement": agreement(pm, om, mo["producer"], mo["orderer"])}
    (out / "fit.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("train_agreement", "monitor_agreement")}), flush=True)
    return 0


def run_check(training_dir: str, atlas: str, models: str, out_dir: str, workers: int,
              horizon: int, max_world_episodes: int | None, dry_run: bool = False) -> int:
    """The pre-registered imitation check (D4); reported agreement comes from fit.json."""
    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas
    from .order_es_validate import monitor_cells
    from .producer_read import pilot_reading

    ts, _train, mon = _training(training_dir)
    if not dry_run:
        _refuse(real_problems(horizon=horizon, max_world_episodes=max_world_episodes, ts=ts,
                              atlas=load_atlas(atlas)))
    if max_world_episodes:
        mon = mon[:max_world_episodes]
    jobs = [j for arm in ARMS for j in _jobs(ts, mon, arm, MONITOR_REPLICATES, horizon, models)]
    rows = _pool(play_job, jobs, workers)
    cells = monitor_cells(ts.training, load_atlas(atlas)["spreads"])
    base = [x for x in rows if x["kind"] == "monitor_base"]
    r = pilot_reading([x for x in rows if x["kind"] == "monitor"] + base, cells, 0)
    deaths = {k: sum(x["dead"] == "true" for x in rows if x["kind"] == k) for k in ARMS}
    swaps = {}
    for arm in SWAP_ARMS:  # reported only (v2): the same floor rule against the base
        rs = pilot_reading([{**x, "kind": "monitor"} for x in rows if x["kind"] == arm] + base,
                           cells, 0)
        swaps[arm] = {"producer": ARMS[arm][0], "orderer": ARMS[arm][1], "g": rs.get("g_tr"),
                      "F": rs.get("F_tr"), "deaths": deaths[arm],
                      "within_floor": (rs.get("g_tr") is not None and rs.get("F_tr") is not None
                                       and rs["g_tr"] >= -rs["F_tr"])}
    g_imit, f_imit = r.get("g_tr"), r.get("F_tr")
    reading = ("faithful" if g_imit is not None and f_imit is not None and g_imit >= -f_imit
               and deaths["monitor"] <= deaths["monitor_base"] else "short")
    doc = {"rule": IMIT_RULE, "reading": reading, "g_imit": g_imit, "F_imit": f_imit,
           "dry_run": dry_run, "prereg": PREREG_SHA256, "models_sha256": sha256_file(models),
           "source_commit": SOURCE_COMMIT,
           "deaths": deaths, "world_episodes": len(mon), "horizon": horizon,
           "units_mean": {k: fmean(float(x["units"]) for x in rows if x["kind"] == k)
                          for k in ARMS},
           "swap_arms_reported_only": swaps}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "imitation_check.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    with (out / "episodes.csv").open("w") as fh:
        keys = ("kind", "world", "episode", "replicate", "units", "dead")
        fh.write(",".join(keys) + "\n")
        for x in sorted(rows, key=lambda x: tuple(str(x[k]) for k in keys[:4])):
            fh.write(",".join(str(x[k]) for k in keys) + "\n")
    print(f"[m6_imitation] imitation check: {reading} (g_imit {g_imit}, F_imit {f_imit}, "
          f"deaths {deaths}); swaps {json.dumps(swaps, default=str)}", flush=True)
    return 0


def run_timing(atlas: str, horizon: int, worlds: Sequence[str]) -> dict:
    """L0 timing probe: seconds per decision (GNN vs k5_rf5 / MWWk) and per horizon-H episode."""
    from ..eval.baselines import producing_policy
    from ..eval.decoder_check import MWWK, decoder_arm, select_views, world_entry
    from ..eval.roster import DEFAULT_ROSTER, load_roster, member_id
    from ..eval.routing_audit import build_jobs, ordering_policy
    from ..eval.runner import run_episode
    from ..eval.teacher_audit import load_atlas

    torch.set_num_threads(1)
    pm, om = g.ProducerGNN(), g.OrdererGNN()
    out = {}
    views = select_views(load_atlas(atlas), list(worlds))
    for view in views:
        arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
        arm = {**arm, "producing": BASE_PRODUCER,
               "subject": member_id(BASE_PRODUCER, arm["ordering"])}
        job = {**build_jobs([world_entry(view)], [arm], 1, horizon, None, {})[0], "order_ttl": 2}
        res = {}
        for name, (p, o) in {"base": (producing_policy(BASE_PRODUCER), ordering_policy(BASE_ORDERER)),
                             "gnn": (g.gnn_producing_policy(pm), g.gnn_ordering_policy(om))}.items():
            env = _episode_env(job)
            tp, to, n_p, n_o = [0.0], [0.0], [0], [0]

            def timed(pol: Any, acc: list, cnt: list) -> Any:
                def f(e: Any, a: str) -> np.ndarray:
                    t0 = time.perf_counter()
                    r_ = pol(e, a)
                    acc[0] += time.perf_counter() - t0
                    cnt[0] += 1
                    return r_
                return f

            t0 = time.perf_counter()
            run_episode(env, timed(p, tp, n_p), timed(o, to, n_o), seed=job["episode"])
            res[name] = {"episode_s": time.perf_counter() - t0,
                         "producer_ms_per_decision": 1000 * tp[0] / max(1, n_p[0]),
                         "orderer_ms_per_decision": 1000 * to[0] / max(1, n_o[0]),
                         "decisions": [n_p[0], n_o[0]]}
        out[f"{view.world if hasattr(view, 'world') else job['world']} ({job['size']})"] = res
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("timing", "collect", "fit", "check", "freeze"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source-commit", default="", help="recorded in every output (provenance)")
    p.add_argument("--training-dir")
    p.add_argument("--collect-dir")
    p.add_argument("--models")
    p.add_argument("--atlas")
    p.add_argument("--out-dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--horizon", type=int, default=pe.HORIZON)
    p.add_argument("--epochs", type=int, default=FIT["epochs"])
    p.add_argument("--fit-mode", default=FIT["mode"], choices=("auto", "sync", "hogwild"))
    p.add_argument("--seed", type=int, default=FIT["seed"])
    p.add_argument("--max-world-episodes", type=int)
    p.add_argument("--worlds", default="ws-atlas-v1-12-0111,ws-atlas-v1-42-0000")
    a = p.parse_args(argv)
    global SOURCE_COMMIT
    SOURCE_COMMIT = a.source_commit
    if not a.dry_run and a.mode in ("collect", "fit", "check") and len(SOURCE_COMMIT) != 40:
        p.error("a real run needs a 40-hex --source-commit")
    if a.mode == "freeze":
        status, sha = freeze_prereg()
        print(f"[m6_imitation] pre-registration {status} {sha}")
        return 0 if status in ("FROZEN", "MATCH") else 2
    if a.mode == "timing":
        print(json.dumps(run_timing(a.atlas, a.horizon, a.worlds.split(",")), indent=1))
        return 0
    if a.mode == "collect":
        return run_collect(a.training_dir, a.out_dir, a.workers, a.horizon, a.max_world_episodes,
                           a.dry_run)
    if a.mode == "fit":
        return run_fit(a.collect_dir, a.out_dir, a.epochs, a.seed, a.workers, a.dry_run,
                       a.fit_mode)
    return run_check(a.training_dir, a.atlas, a.models, a.out_dir, a.workers, a.horizon,
                     a.max_world_episodes, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
