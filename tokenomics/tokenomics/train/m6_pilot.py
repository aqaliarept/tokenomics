"""EXP-0113: the capped critic-free Stage 1 training pilot (pre-registration v10).

`python -m tokenomics.train.m6_pilot --mode {freeze,run} ...`

User decision 2026-11-06 (after EXP-0112: n80 = 48 seeds per half for E3 at 12, inside v9's 96; D0.3 at 12 per half
was underpowered by construction):
- **Learns:** the producer only, at 12 producers.
  - The orderer is frozen at EXP-0100's weights (keyed softmax draw).
  - Input buying is frozen at the start model's deterministic output.
  - The gates are tempered as in v9 (training rollouts only).
  - The start is EXP-0110's start model (EXP-0100 + zero-weight v4 inputs).
- **Estimator E3:**
  - G = 4 same-seed siblings, the per-cycle leave-one-out baseline on the return-to-go, γ = 0.99;
  - SEEDS_PER_ITER = 96 of M4's 112 training world-episodes per iteration (48 per half), sampled per iteration;
  - siblings play DrawKey replicate 1000 + 10 · iteration + i;
  - the normalised potential with EXP-0110's calibration.
- **Update:** PPO-clip on each producer decision's tempered-gate log-probability, plus β · KL(π_θ ‖ π_start)
  on the same states.
  - Workers own whole seeds, so each computes its advantages locally and keeps its decisions.
  - STEPS synchronous steps per iteration, averaging the workers' minibatch gradients; the per-agent Stepper
    (Adam, clip) with the per-step log.
- **Gate at iterations 5 and 10:** the greedy policy (the deployed decode) against the EXP-0100 stack on the 12
  monitor world-episodes × replicates 0 / 1 (paired keys), M5's floor rule; deaths, run-down and starvation shares
  (L5's clauses).
  - **Early stop at 5:** g < −F, more deaths, or a higher run-down share.
  - **Reading at 10:** `gain` (g > F, no side effect worse), `hold` (|g| ≤ F), `worse` (otherwise).
- **Checkpoint:** every iteration (resumable on the same out dir).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from ..env.potential import PotentialTracker, PotentialWeights
from ..env.spaces import ACTION_LOGIT_BOUND
from . import m6_gnn as g
from . import m6_imitation as im
from . import m6_stage0 as s0
from . import m6_stage0b as sb

PREREG_V10_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v10.json"
PREREG_V10_SHA256: str | None = "0123274c423807c57ef79f05edcdbb2830c3c76e116ad964fe349dfc2a7218db"  # frozen before EXP-0113 data
CFG = {"iterations": 10, "seeds_per_iter": 96, "siblings": 4, "gamma": 0.99, "clip": 0.2,
       "kl_coef": 0.04, "lr": 3e-4, "grad_clip": 1.0, "steps": 100, "batch": 256, "horizon": 100,
       "gate_iterations": [5, 10], "replicate_base": 1000, "seed": 0}
PILOT_RULE = (
    "gate at iterations 5 and 10: greedy (deployed) candidate vs the EXP-0100 stack on the 12 monitor "
    "world-episodes x replicates 0 / 1, M5's floors rule (g, F); early stop at 5 if g < -F, more deaths, or a "
    "higher run-down share; reading at 10: gain if g > F and deaths, run-down and starvation shares no worse; "
    "hold if |g| <= F; worse otherwise. Reported only: the training-return curve, KL to the start, the gate "
    "flip rates")


# --- rollouts with recorded decisions ---


class PilotPolicies(sb.ExplorePolicies):
    """v9's exploration (tempered gates; frozen replenish; frozen orderer), recording each producer
    decision's features, sampled gates, old log-probability and cycle for the PPO update."""

    def __init__(self, *a: Any) -> None:
        super().__init__(*a, score=False)
        self.decisions: list[dict] = []

    def producing(self, env: Any, agent: str) -> np.ndarray:
        pid = env._role[agent][1]
        f = g.producer_features(env.system.producer_entity_view(pid), g.producer_context(env, pid),
                                self.pm.features)
        with torch.no_grad():
            _g0, rep0 = self.pm0(f)
            gates, _ = self.pm.heads(f)
            tl = sb.tempered_logits(gates)
        bits = self.rng.random(3) < torch.sigmoid(tl).numpy()
        greedy = gates.numpy() > 0
        for i, k in enumerate(sb.GATES):
            self.flips[k][0] += int(bits[i] != greedy[i])
            self.flips[k][1] += 1
        lp = float(torch.distributions.Bernoulli(logits=tl).log_prob(
            torch.from_numpy(bits.astype(np.float32))).sum())
        self.decisions.append({"f": f, "bits": bits.astype(np.float32), "old_lp": lp,
                               "cycle": int(env.current_cycle)})
        sp = env.spec_
        act = np.zeros(sp.producing_action_dim, dtype=np.float32)
        for gp, frac in zip(f["inputs"], rep0.numpy()):
            if gp in sp.input_index:
                act[sp.input_index[gp]] = float(frac) * ACTION_LOGIT_BOUND
        act[sp.num_inputs:sp.num_inputs + 3] = np.where(bits, ACTION_LOGIT_BOUND,
                                                         -ACTION_LOGIT_BOUND)
        return act


def play_seed(job: dict, pm: g.ProducerGNN, pm0: g.ProducerGNN, om: g.OrdererGNN) -> dict:
    """The G siblings of one seed; E3 advantages per cycle; the decisions with their advantage."""
    from ..eval.runner import run_episode

    eps = []
    for i in range(CFG["siblings"]):
        jj = {**job, "replicate": CFG["replicate_base"] + 10 * job["iteration"] + i}
        env = im._episode_env(jj)
        tr = PotentialTracker(PotentialWeights(**job["weights"]))
        tr.attach(env)
        rng = np.random.default_rng([CFG["seed"], 13, job["iteration"],
                                     int(job["episode"]) % (1 << 31), i])
        pol = PilotPolicies(pm, pm0, om, om, rng)
        run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None)
        eps.append({"rewards": tr.rewards(float(job["unit_scale"])), "decisions": pol.decisions,
                    "flips": pol.flips, "units": tr.totals()["units"],
                    "dead": bool(env.episode_stats["dead"])})
    rtg = [s0._returns_to_go(e["rewards"], CFG["gamma"]) for e in eps]
    n = min(len(x) for x in rtg)
    m = np.stack([x[:n] for x in rtg])
    loo = (m.sum(0, keepdims=True) - m) / (CFG["siblings"] - 1)
    adv = m - loo
    out = []
    for i, e in enumerate(eps):
        for d in e["decisions"]:
            if d["cycle"] < n:
                out.append({**d, "adv": float(adv[i, d["cycle"]])})
    return {"decisions": out, "returns": [float(sum(e["rewards"])) for e in eps],
            "units": [e["units"] for e in eps], "dead": [e["dead"] for e in eps],
            "flips": {k: [sum(e["flips"][k][0] for e in eps), sum(e["flips"][k][1] for e in eps)]
                      for k in sb.GATES}}


def ppo_loss(pm: g.ProducerGNN, pm0: g.ProducerGNN, batch: Sequence[dict]
             ) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate on the tempered-gate log-probability, plus β · KL(π_θ ‖ π_start)."""
    ls = torch.nn.functional.logsigmoid
    pg, kl = torch.zeros(()), torch.zeros(())
    clipped = 0
    for d in batch:
        tl = sb.tempered_logits(pm.heads(d["f"])[0])
        with torch.no_grad():
            t0 = sb.tempered_logits(pm0.heads(d["f"])[0])
        lp = torch.distributions.Bernoulli(logits=tl).log_prob(torch.from_numpy(d["bits"])).sum()
        ratio = torch.exp(lp - d["old_lp"])
        a = d["adv"]
        pg = pg - torch.min(ratio * a, ratio.clamp(1 - CFG["clip"], 1 + CFG["clip"]) * a)
        p = torch.sigmoid(tl)
        kl = kl + (p * (ls(tl) - ls(t0)) + (1 - p) * (ls(-tl) - ls(-t0))).sum()
        clipped += int(abs(float(ratio.detach()) - 1) > CFG["clip"])
    n = max(1, len(batch))
    loss = pg / n + CFG["kl_coef"] * kl / n
    return loss, {"pg": float(pg.detach()) / n, "kl": float(kl.detach()) / n, "clip_frac": clipped / n}


def _worker(w: int, pm: g.ProducerGNN, start: str, jobs: list[dict], gbuf: torch.Tensor,
            sbuf: torch.Tensor, ready: Any, done: Any, q: Any, it: int) -> None:
    torch.set_num_threads(1)
    im.set_sharing()
    pm0, om = im.load_models(start)
    pm0.eval()
    om.eval()
    decs, info = [], []
    for j in jobs:
        r = play_seed(j, pm, pm0, om)
        decs += r["decisions"]
        info.append({k: r[k] for k in ("returns", "units", "dead", "flips")})
    q.put({"w": w, "info": info, "decisions": len(decs)})
    params = list(pm.parameters())
    rng = np.random.default_rng([CFG["seed"], 17, it, w])
    order = rng.permutation(len(decs)) if decs else np.zeros(0, int)
    for b in range(CFG["steps"]):
        ready.wait()
        for p in params:
            p.grad = None
        if len(decs):
            idx = order[(np.arange(b * CFG["batch"], (b + 1) * CFG["batch"])) % len(decs)]
            loss, st = ppo_loss(pm, pm0, [decs[i] for i in idx])
            loss.backward()
        else:
            st = {"pg": 0.0, "kl": 0.0, "clip_frac": 0.0}
        gbuf[w] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                             for p in params])
        sbuf[w] = torch.tensor([st["pg"], st["kl"], st["clip_frac"]])
        done.wait()


def iteration(pm: g.ProducerGNN, stepper: im.Stepper, start: str, jobs: list[dict], workers: int,
              it: int) -> dict:
    """One iteration: rollouts in the workers (whole seeds each), then STEPS synchronous PPO steps."""
    params = list(pm.parameters())
    sizes = [p.numel() for p in params]
    im.set_sharing()
    pm.share_memory()
    gbuf = torch.zeros(workers, sum(sizes)).share_memory_()
    sbuf = torch.zeros(workers, 3).share_memory_()
    ctx = torch.multiprocessing.get_context("fork")
    ready, done = ctx.Barrier(workers + 1, timeout=7200), ctx.Barrier(workers + 1, timeout=7200)
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(w, pm, start, jobs[w::workers], gbuf, sbuf, ready,
                                               done, q, it), daemon=True) for w in range(workers)]
    t0 = time.perf_counter()
    for pr in procs:
        pr.start()
    infos = [q.get(timeout=7200) for _ in range(workers)]
    t_roll = time.perf_counter() - t0
    tot = np.zeros(3)
    try:
        for _b in range(CFG["steps"]):
            ready.wait()
            done.wait()
            grad = gbuf[0].clone()
            for w in range(1, workers):
                grad += gbuf[w]
            grad /= workers
            for p, chunk in zip(params, torch.split(grad, sizes)):
                p.grad = chunk.view_as(p).clone()
            st = sbuf.mean(0).numpy()
            stepper.step({"pg": float(st[0]), "kl": float(st[1]), "clip_frac": float(st[2])}, it)
            tot += st
    finally:
        for pr in procs:
            pr.join(timeout=120)
            if pr.is_alive():
                pr.terminate()
    if any(pr.exitcode for pr in procs):
        raise RuntimeError("a pilot worker failed")
    seeds = [x for i in infos for x in i["info"]]
    flips = {k: sum(s["flips"][k][0] for s in seeds) / max(1, sum(s["flips"][k][1] for s in seeds))
             for k in sb.GATES}
    return {"iteration": it, "train_return": fmean(r for s in seeds for r in s["returns"]),
            "train_units": fmean(u for s in seeds for u in s["units"]),
            "deaths": sum(d for s in seeds for d in s["dead"]),
            "decisions": sum(i["decisions"] for i in infos), "flip_rate": flips,
            "pg": float(tot[0] / CFG["steps"]), "kl_to_start": float(tot[1] / CFG["steps"]),
            "clip_frac": float(tot[2] / CFG["steps"]), "rollout_seconds": round(t_roll, 1),
            "seconds": round(time.perf_counter() - t0, 1)}


def inline_iteration(pm: g.ProducerGNN, stepper: im.Stepper, start: str, jobs: list[dict],
                     it: int) -> dict:
    """workers = 1 (tests): the same computation in-process."""
    pm0, om = im.load_models(start)
    decs, seeds = [], []
    for j in jobs:
        r = play_seed(j, pm, pm0, om)
        decs += r["decisions"]
        seeds.append(r)
    rng = np.random.default_rng([CFG["seed"], 17, it, 0])
    order = rng.permutation(len(decs))
    tot = np.zeros(3)
    for b in range(CFG["steps"]):
        stepper.zero_grad()
        idx = order[np.arange(b * CFG["batch"], (b + 1) * CFG["batch"]) % len(decs)]
        loss, st = ppo_loss(pm, pm0, [decs[i] for i in idx])
        loss.backward()
        stepper.step(st, it)
        tot += [st["pg"], st["kl"], st["clip_frac"]]
    flips = {k: sum(s["flips"][k][0] for s in seeds) / max(1, sum(s["flips"][k][1] for s in seeds))
             for k in sb.GATES}
    return {"iteration": it, "train_return": fmean(x for s in seeds for x in s["returns"]),
            "train_units": fmean(x for s in seeds for x in s["units"]),
            "deaths": sum(d for s in seeds for d in s["dead"]), "decisions": len(decs),
            "flip_rate": flips, "pg": float(tot[0] / CFG["steps"]),
            "kl_to_start": float(tot[1] / CFG["steps"]), "clip_frac": float(tot[2] / CFG["steps"])}


# --- the gate ---


def gate(ts: Any, mon: Sequence, atlas: Mapping, cand: str, ref: str, workers: int, horizon: int,
         ref_rows: list | None) -> tuple[dict, list]:
    from .order_es_validate import monitor_cells, row_flags
    from .order_es_worlds import X_RUNDOWN
    from .producer_read import pilot_reading

    def rows_for(models: str) -> list:
        jobs = [{**j, "models": models, "x": float(X_RUNDOWN)}
                for j in im._jobs(ts, mon, "monitor", im.MONITOR_REPLICATES, horizon, models)]
        return im._pool(im.side_effect_job, jobs, workers)

    if ref_rows is None:
        ref_rows = rows_for(ref)
    cand_rows = rows_for(cand)
    cells = monitor_cells(ts.training, atlas["spreads"])
    r = pilot_reading([{**x, "kind": "monitor", "generation": "0"} for x in cand_rows]
                      + [{**x, "kind": "monitor_base"} for x in ref_rows], cells, 0)

    def flags(rows: list) -> dict:
        f = [row_flags(x) for x in rows]
        return {"deaths": sum(x[0] for x in f), "rundown_share": sum(x[1] for x in f) / len(f),
                "starvation_share": sum(x[2] for x in f) / len(f),
                "units": fmean(float(x["units"]) for x in rows)}

    fc, fr = flags(cand_rows), flags(ref_rows)
    return {"g": r.get("g_tr"), "F": r.get("F_tr"), "candidate": fc, "reference": fr}, ref_rows


def gate_subprocess(training_dir: str, atlas_path: str, cand: str, ref: str, workers: int,
                    horizon: int, out: Path, n_mon: int) -> dict:
    """The gate in a fresh process: a spawn pool started after the forked training workers in the
    same process hung (local smoke, 2026-11-06); a clean interpreter avoids the interplay. The
    reference rows are cached in `ref_rows.json`."""
    import subprocess

    res = out / f"gate_{Path(cand).stem}.json"
    cmd = [sys.executable, "-m", "tokenomics.train.m6_pilot", "--mode", "gate", "--training-dir",
           training_dir, "--atlas", atlas_path, "--candidate", cand, "--ref-models", ref,
           "--workers", str(workers), "--horizon", str(horizon), "--out-dir", str(out),
           "--gate-out", str(res), "--max-monitor", str(n_mon), "--dry-run"]
    subprocess.run(cmd, check=True)
    return json.loads(res.read_text())


def gate_main(training_dir: str, atlas_path: str, cand: str, ref: str, workers: int, horizon: int,
              out_dir: str, gate_out: str, n_mon: int) -> int:
    from ..eval.teacher_audit import load_atlas

    ts, _train, mon = im._training(training_dir)
    cache = Path(out_dir) / "ref_rows.json"
    ref_rows = json.loads(cache.read_text()) if cache.exists() else None
    gt, ref_rows = gate(ts, mon[:n_mon], load_atlas(atlas_path), cand, ref, workers, horizon,
                        ref_rows)
    cache.write_text(json.dumps(ref_rows))
    Path(gate_out).write_text(json.dumps(gt, default=float) + "\n")
    return 0


def gate_reading(gt: Mapping, final: bool) -> str:
    g_, f_ = gt["g"], gt["F"]
    c, r = gt["candidate"], gt["reference"]
    worse_side = (c["deaths"] > r["deaths"] or c["rundown_share"] > r["rundown_share"]
                  or c["starvation_share"] > r["starvation_share"])
    if not final:
        return "stop" if (g_ < -f_ or c["deaths"] > r["deaths"]
                          or c["rundown_share"] > r["rundown_share"]) else "continue"
    if g_ > f_ and not worse_side:
        return "gain"
    if abs(g_) <= f_:
        return "hold"
    return "worse"


# --- pre-registration v10 ---


def registered_prereg_v10() -> dict:
    return {
        "format": "m6-prereg-v10", "change": "add-m6-joint-mpg-gnn",
        "stage": "EXP-0113: the capped critic-free Stage 1 training pilot (producer only, 12 producers)",
        "status": "frozen before EXP-0113's data; the real run refuses any other file",
        "keeps": {"m6_prereg_v9.json": sb.PREREG_V9_SHA256, "note": "v1..v9 stay frozen"},
        "decision": "user 2026-11-06 (via the supervisor): a capped pilot; overrides v9's absent because the "
                    "power check found n80 = 48 seeds per half for E3 at 12 (inside 96) and D0.3 at 12 per "
                    "half was underpowered by construction",
        "start": {"models": "EXP-0110 start_models.pt (EXP-0100 + zero-weight v4 inputs)",
                  "reference_models_sha256": im.EXP0100_MODELS_SHA},
        "learns": "the producer only; the orderer frozen at EXP-0100's weights; replenish frozen at the "
                  "start's deterministic output; the gates tempered (v9 TEMPS) in training rollouts only",
        "temps": sb.TEMPS, "config": CFG,
        "estimator": "E3: per-cycle leave-one-out over the G siblings on the gamma-discounted return-to-go "
                     "of the normalised potential (EXP-0110 calibration), no std division",
        "seeds": "per iteration, SEEDS_PER_ITER of M4's 112 training world-episodes (monitor excluded), "
                 "sampled without replacement by numpy default_rng([seed, 11, iteration])",
        "update": "PPO-clip on the tempered-gate log-probability per producer decision + kl_coef * "
                  "KL(pi_theta || pi_start) on the same states; STEPS synchronous steps per iteration, "
                  "each the mean of the workers' minibatch gradients (BATCH decisions each, from their own "
                  "seeds); the per-agent Stepper (Adam, clip) with the step log",
        "gate": {"rule": PILOT_RULE, "iterations": CFG["gate_iterations"],
                 "set": "the 16 monitor world-episodes x replicates 0 / 1, horizon 100, greedy decode",
                 "floor": "M5's floors rule (producer_read.pilot_reading)",
                 "side_effects": "deaths; run-down share (r_end < X r_start); starvation share (fill < Y)"},
        "continuation": "only on gain (or hold with a rising curve, the user's call), a later decision"}


def freeze(path: Path = PREREG_V10_PATH) -> tuple[str, str]:
    from ..eval.provenance import sha256_file

    doc = registered_prereg_v10()
    if not path.exists():
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return "FROZEN", sha256_file(str(path))
    return ("MATCH" if json.loads(path.read_text()) == doc else "MISMATCH"), sha256_file(str(path))


def prereg_problems() -> list[str]:
    from ..eval.provenance import sha256_file

    if PREREG_V10_SHA256 is None:
        return ["no pinned m6_prereg_v10.json (`--mode freeze`)"]
    if not PREREG_V10_PATH.exists() or sha256_file(str(PREREG_V10_PATH)) != PREREG_V10_SHA256:
        return ["m6_prereg_v10.json does not match its pinned SHA-256"]
    if json.loads(PREREG_V10_PATH.read_text()) != registered_prereg_v10():
        return ["the pre-registration differs from the code"]
    return []


# --- the run ---


def run(training_dir: str, atlas_path: str, exp0110: str, ref_models: str, out_dir: str,
        workers: int, dry_run: bool = False) -> int:
    saved = dict(CFG)
    try:
        return _run(training_dir, atlas_path, exp0110, ref_models, out_dir, workers, dry_run)
    finally:
        CFG.clear()
        CFG.update(saved)  # a dry run shrinks steps / batch; the registered CFG is restored


def _run(training_dir: str, atlas_path: str, exp0110: str, ref_models: str, out_dir: str,
         workers: int, dry_run: bool) -> int:
    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas

    atlas = load_atlas(atlas_path)
    e10 = Path(exp0110)
    start = str(e10 / "start_models.pt")
    cal = json.loads((e10 / "calibration.json").read_text())
    if not dry_run:
        im._refuse(prereg_problems()
                   + ([] if sha256_file(ref_models) == im.EXP0100_MODELS_SHA else
                      ["--ref-models is not EXP-0100's"])
                   + ([] if atlas.get("content_hash") == im.REGISTERED_ATLAS_HASH else
                      ["not the registered atlas v2"]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts, train, mon = im._training(training_dir)
    cfg = dict(CFG)
    if dry_run:  # a true smoke: 2 iterations of 2 seeds, horizon 4, 3 steps, a 1-world gate at 1 and 2
        cfg.update({"iterations": 2, "seeds_per_iter": 2, "horizon": 4, "steps": 3, "batch": 8,
                    "gate_iterations": [1, 2]})
        mon = mon[:1]
        CFG.update({k: cfg[k] for k in ("steps", "batch")})
    c12 = cal["per_size"]["12"]
    weights = {"lam": c12["lambda"], "mu": c12["mu"], "nu": float(cfg["horizon"]), "x": cal["x"],
               "y": cal["y"]}
    pm, _om = im.load_models(start)
    stepper = im.Stepper({"producer": list(pm.parameters())}, cfg["lr"], cfg["grad_clip"], True,
                         out / "steps.jsonl")
    ck = out / "checkpoint.pt"
    curve_path = out / "curve.jsonl"
    first, gates = 1, {}
    if ck.exists():  # resume
        d = torch.load(ck, weights_only=False)
        pm.load_state_dict(d["producer"])
        stepper.load_state_dict(d["stepper"])
        first, gates = int(d["iteration"]) + 1, d["gates"]
    reading = None
    for it in range(first, cfg["iterations"] + 1):
        rng = np.random.default_rng([cfg["seed"], 11, it])
        pick = [train[i] for i in sorted(rng.choice(len(train), cfg["seeds_per_iter"],
                                                    replace=False))]
        jobs = []
        for w_, e_ in pick:
            j = im._jobs(ts, [(w_, e_)], "pilot", (0,), cfg["horizon"])[0]
            jobs.append({**j, "iteration": it, "weights": weights,
                         "unit_scale": cfg["horizon"] / cal["bar"][f"{w_}|{e_}"]})
        rec = (inline_iteration(pm, stepper, start, jobs, it) if workers <= 1 else
               iteration(pm, stepper, start, jobs, workers, it))
        if it in cfg["gate_iterations"]:
            cand = out / f"candidate_it{it:02d}.pt"
            _p0, om0 = im.load_models(start)
            im.save_models(pm, om0, cand)
            gt = gate_subprocess(training_dir, atlas_path, str(cand), ref_models, workers,
                                 cfg["horizon"], out, len(mon))
            final = it == cfg["gate_iterations"][-1]
            gt["reading"] = gate_reading(gt, final)
            gates[str(it)] = gt
            rec["gate"] = gt
        with curve_path.open("a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        torch.save({"producer": pm.state_dict(), "stepper": stepper.state_dict(), "iteration": it,
                    "gates": gates}, ck)
        print(f"[m6_pilot] it {it}: return {rec['train_return']:.2f} units {rec['train_units']:.0f} "
              f"deaths {rec['deaths']} kl {rec['kl_to_start']:.4f} clip {rec['clip_frac']:.3f} flips "
              f"{ {k: round(v, 3) for k, v in rec['flip_rate'].items()} }"
              + (f" gate {rec['gate']['reading']} g {rec['gate']['g']} F {rec['gate']['F']}"
                 if "gate" in rec else ""), flush=True)
        if "gate" in rec and rec["gate"]["reading"] in ("stop", "gain", "hold", "worse"):
            reading = rec["gate"]["reading"] if rec["gate"]["reading"] != "stop" else "stopped_early"
            break
    stepper.close()
    doc = {"rule": PILOT_RULE, "reading": reading, "gates": gates, "config": cfg,
           "prereg": PREREG_V10_SHA256, "dry_run": dry_run, "source_commit": im.SOURCE_COMMIT,
           "start_sha256": sha256_file(start)}
    (out / "reading.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[m6_pilot] reading {reading}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=("freeze", "run", "gate"), default="run")
    p.add_argument("--candidate")
    p.add_argument("--horizon", type=int, default=CFG["horizon"])
    p.add_argument("--gate-out")
    p.add_argument("--max-monitor", type=int)
    p.add_argument("--training-dir")
    p.add_argument("--atlas")
    p.add_argument("--exp0110")
    p.add_argument("--ref-models")
    p.add_argument("--out-dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source-commit", default="")
    a = p.parse_args(argv)
    im.SOURCE_COMMIT = a.source_commit
    if a.mode == "freeze":
        status, sha = freeze()
        print(f"[m6_pilot] pre-registration v10 {status} {sha}")
        return 0 if status in ("FROZEN", "MATCH") else 2
    if a.mode == "gate":  # the run's own subprocess (gate_subprocess); not a reading on its own
        torch.set_num_threads(1)
        return gate_main(a.training_dir, a.atlas, a.candidate, a.ref_models, a.workers, a.horizon,
                         a.out_dir, a.gate_out, a.max_monitor or 10 ** 9)
    if not a.dry_run and len(a.source_commit) != 40:
        p.error("a real run needs a 40-hex --source-commit")
    torch.set_num_threads(1)
    return run(a.training_dir, a.atlas, a.exp0110, a.ref_models, a.out_dir, a.workers, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
