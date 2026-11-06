"""M6 Stage I: imitation of k5_rf5 + MWWk by the fully learned GNN policies (add-m6-joint-mpg-gnn D4).

`python -m tokenomics.train.m6_imitation --mode {timing,collect,fit,check}`
- `timing` (L0, local, tiny): seconds per decision of the GNN policies against k5_rf5 / MWWk, and a horizon-H
  episode with the GNN stack, at 12 and 42 producers. Re-sizes sizing.md before any pod.
- `collect`: k5_rf5 + MWWk on M4's pinned training world-episodes (EXP-0058 `bar/`), excluding the 16 monitor
  world-episodes, at r0. It records every producer decision (features, k5_rf5's raw action as the label) and
  every order-slot decision (features, MWW's normalised weights over the eligible producers as a soft label).
  Each label is a deterministic function of the agent's view: tested.
- `fit`: behaviour cloning, with BCE on the 3 producer toggles, Huber on the replenish fractions, and KL on the
  orderer's slot distributions.
- `check` (the pre-registered imitation check, D4):
  - the GNN stack and k5_rf5 + MWWk on the 16 monitor world-episodes at replicates 0 / 1;
  - g_imit = the equal-weight mean of (imitated − bar) units, F_imit = M5's floors rule;
  - **`faithful`** if g_imit ≥ −F_imit and no extra deaths, else **`short`** (a stop: the user hears it before
    any RL);
  - reported: exact-command agreement, gate accuracy and replenish MAE (producer), and KL and top-1 agreement
    (orderer), on the monitor episodes' labels.
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
# the fit's pre-registered stopping rule and settings (EXP-0099)
FIT = {"epochs": 20, "batch": 64, "lr": 3e-3, "seed": 0, "workers": 16, "hogwild": True}
FIDELITY_METRICS = ("exact_command", "gate_accuracy", "replenish_mae_units", "orderer_kl",
                    "orderer_top1")
REGISTERED_ATLAS_HASH = "220886a7cf12aaac0cc0a48dc03c57f2dc2484bb2af6185a6160ec509c8a2a23"
PREREG_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v1.json"
SOURCE_COMMIT = ""  # set by `main` (--source-commit); recorded in every output
# frozen 2026-11-06 before EXP-0099; real collect / fit / check refuse any other file
PREREG_SHA256: str | None = "38db5c0c96cfd2a7fa14eae1e3bf2832092738d1d13ce2fdcade25ac245c26f6"


# --- pre-registration (frozen before Stage I's data) ---


def registered_prereg() -> dict:
    return {
        "format": "m6-prereg-v1", "change": "add-m6-joint-mpg-gnn", "stage": "I (imitation), D4",
        "status": "frozen before EXP-0099's data; real collect / fit / check refuse any other file",
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
                                       "producer dropped"}},
        "model": {"hidden": g.HIDDEN, "rounds": g.ROUNDS,
                  "losses": "BCE (toggles) + Huber delta 0.1 (replenish fractions, masked) + KL "
                            "(orderer slots)"},
        "fit": {**FIT, "stopping_rule": f"fixed {FIT['epochs']} epochs (no early stopping); the "
                                        "checkpoint SHA-256 is the pinned artifact"},
        "check": {"rule": IMIT_RULE,
                  "monitor_set": "schedule.json `monitor` (16 world-episodes)",
                  "replicates": list(MONITOR_REPLICATES),
                  "comparison": "the GNN stack vs k5_rf5 + MWWk at the same DrawKeys",
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
        return ["no pinned M6 pre-registration (m6_prereg_v1.json; `--mode freeze`)"]
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
    """One keyed episode of the GNN stack (job["models"] = path) or the base (None)."""
    from ..eval.baselines import producing_policy
    from ..eval.routing_audit import ordering_policy
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    env = _episode_env(job)
    if job.get("models"):
        pm, om = load_models(job["models"])
        p, o = g.gnn_producing_policy(pm), g.gnn_ordering_policy(om)
    else:
        p, o = producing_policy(BASE_PRODUCER), ordering_policy(BASE_ORDERER)
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
            lr: float, seed: int, log: Any) -> list[dict]:
    torch.set_num_threads(1)
    opt = torch.optim.Adam([*pm.parameters(), *om.parameters()], lr=lr)
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
            (lp + lo).backward()
            opt.step()
            tot[0] += float(lp.detach())
            tot[1] += float(lo.detach())
        curve.append({"epoch": ep + 1, "producer_loss": tot[0] / n, "orderer_loss": tot[1] / n})
        log(f"[m6_imitation] epoch {ep + 1}: producer {tot[0] / n:.4f} orderer {tot[1] / n:.4f}")
    return curve


def _hogwild(args: tuple) -> list[dict]:
    pm, om, prod, orde, epochs, batch, lr, seed = args
    return _epochs(pm, om, prod, orde, epochs, batch, lr, seed, lambda s: None)


def fit(samples: dict, epochs: int, batch: int = 64, lr: float = 3e-3, seed: int = 0,
        log: Any = print, workers: int = 1) -> tuple[g.ProducerGNN, g.OrdererGNN, list[dict]]:
    """Behaviour cloning. `workers` > 1: Hogwild (shared CPU weights, one shard per process, each
    with its own Adam), so a pod's cores are used; not bit-reproducible across runs."""
    torch.manual_seed(seed)
    pm, om = g.ProducerGNN(), g.OrdererGNN()
    prod, orde = samples["producer"], samples["orderer"]
    if workers <= 1:
        return pm, om, _epochs(pm, om, prod, orde, epochs, batch, lr, seed, log)
    pm.share_memory()
    om.share_memory()
    ctx = torch.multiprocessing.get_context("fork")
    shards = [(pm, om, prod[i::workers], orde[i::workers], epochs, batch, lr, seed + i)
              for i in range(workers)]
    with ctx.Pool(workers) as pool:
        curves = pool.map(_hogwild, shards)
    curve = [{"epoch": e + 1,
              "producer_loss": fmean(c[e]["producer_loss"] for c in curves),
              "orderer_loss": fmean(c[e]["orderer_loss"] for c in curves)} for e in range(epochs)]
    log(f"[m6_imitation] hogwild x{workers}: final producer {curve[-1]['producer_loss']:.4f} "
        f"orderer {curve[-1]['orderer_loss']:.4f}")
    return pm, om, curve


def agreement(pm: g.ProducerGNN, om: g.OrdererGNN, prod: list, orde: list) -> dict:
    """Fidelity metrics on labelled decisions (reported, D4)."""
    exact = gates_ok = n_gate = 0
    mae: list[float] = []
    with torch.no_grad():
        for f, y in prod:
            gates, rep = pm(f)
            tg = (gates.numpy() > 0).astype(np.float32)
            gates_ok += int((tg == y["toggles"]).sum())
            n_gate += 3
            mr = f["global"][0][9]  # log1p(max_replenish)
            units = np.expm1(mr)
            ok = bool((tg == y["toggles"]).all())
            if rep.shape[0]:
                du = np.abs(np.round(rep.numpy() * units) - np.round(y["replenish"] * units))
                du = du[y["mask"] > 0]
                mae += du.tolist()
                ok = ok and bool((du == 0).all())
            exact += ok
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
            "orderer_kl": fmean(kl) if kl else 0.0, "orderer_top1": top1 / max(1, n_s)}


def save_models(pm: g.ProducerGNN, om: g.OrdererGNN, path: Path) -> None:
    torch.save({"producer": pm.state_dict(), "orderer": om.state_dict(),
                "hidden": g.HIDDEN, "rounds": g.ROUNDS}, path)


def load_models(path: str | Path) -> tuple[g.ProducerGNN, g.OrdererGNN]:
    d = torch.load(path, weights_only=True)
    pm, om = g.ProducerGNN(d["hidden"], d["rounds"]), g.OrdererGNN(d["hidden"], d["rounds"])
    pm.load_state_dict(d["producer"])
    om.load_state_dict(d["orderer"])
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


def run_fit(collect_dir: str, out_dir: str, epochs: int, seed: int, workers: int = 1,
            dry_run: bool = False) -> int:
    c = Path(collect_dir)
    if not dry_run:
        meta = json.loads((c / "collect.json").read_text())
        _refuse(real_problems(horizon=meta["horizon"], max_world_episodes=None, epochs=epochs,
                              workers=workers, seed=seed)
                + ([] if not meta.get("dry_run") else ["the collect run was a dry run"]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (c / "samples_train.pkl").open("rb") as fh:
        tr = pickle.load(fh)
    with (c / "samples_monitor.pkl").open("rb") as fh:
        mo = pickle.load(fh)
    pm, om, curve = fit(tr, epochs, seed=seed, workers=workers)
    save_models(pm, om, out / "models.pt")
    from ..eval.provenance import sha256_file

    doc = {"epochs": epochs, "seed": seed, "workers": workers, "curve": curve, "dry_run": dry_run,
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
    jobs = (_jobs(ts, mon, "monitor", MONITOR_REPLICATES, horizon, models)
            + _jobs(ts, mon, "monitor_base", MONITOR_REPLICATES, horizon))
    rows = _pool(play_job, jobs, workers)
    cells = monitor_cells(ts.training, load_atlas(atlas)["spreads"])
    r = pilot_reading(rows, cells, 0)
    deaths = {k: sum(x["dead"] == "true" for x in rows if x["kind"] == k)
              for k in ("monitor", "monitor_base")}
    g_imit, f_imit = r.get("g_tr"), r.get("F_tr")
    reading = ("faithful" if g_imit is not None and f_imit is not None and g_imit >= -f_imit
               and deaths["monitor"] <= deaths["monitor_base"] else "short")
    doc = {"rule": IMIT_RULE, "reading": reading, "g_imit": g_imit, "F_imit": f_imit,
           "dry_run": dry_run, "prereg": PREREG_SHA256, "models_sha256": sha256_file(models),
           "source_commit": SOURCE_COMMIT,
           "deaths": deaths, "world_episodes": len(mon), "horizon": horizon,
           "units_mean": {k: fmean(float(x["units"]) for x in rows if x["kind"] == k)
                          for k in ("monitor", "monitor_base")}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "imitation_check.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"[m6_imitation] imitation check: {reading} (g_imit {g_imit}, F_imit {f_imit}, "
          f"deaths {deaths})", flush=True)
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
        return run_fit(a.collect_dir, a.out_dir, a.epochs, a.seed, a.workers, a.dry_run)
    return run_check(a.training_dir, a.atlas, a.models, a.out_dir, a.workers, a.horizon,
                     a.max_world_episodes, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
