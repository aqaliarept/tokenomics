"""EXP-0127 (task 4.4): the power check for on-policy RL of every producer head from F40 + v15 (pre-registration v16;
draft notes/4.4-4.5-draft.md section E). EXP-0112's method (`m6_stage0b`) with section A's exploration
(`m6_onpolicy`):
- **start:** the F40 + v15 composite (`m6_onpolicy.F40Model`); the orderer frozen (keyed draw, scored as in v9 for
  the analysis only); the 4.5 reward (λ / 4);
- **temperatures:** calibrated offline at the start of the run on the start's recorded decisions (EXP-0105 monitor
  samples for replenish; EXP-0126's held-out turn records for the fund / accept turns), recorded;
- **rollouts:** 24 seeds at 12 producers × G = 4 siblings with per-decision projected scores over the trainable
  producer parameters, and each head family's squared score;
- **gradient size:** ±ε along K = 4 random directions of the trainable parameters, ε at a per-direction mean KL
  (`m6_onpolicy.decision_kl`) of 0.01 on recorded start decisions;
- **reading (v9's rule):** `signal` if some estimator's producer cosine beats its null's q95 and its n80 ≤ 96 seeds
  per half; else `absent`. Reported: the head families' score shares, flip / swap rates, deaths, cost per episode.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from . import m6_gnn as g
from . import m6_imitation as im
from . import m6_onpolicy as op
from . import m6_power as pw
from . import m6_stage0 as s0
from . import m6_stage0b as sb

PREREG_V16_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v16.json"
PREREG_V16_SHA256: str | None = "6ad4e49fc3abd89501ad994ad5228be2df367068b66998dbce788e619c6a8221"  # frozen before EXP-0127 data (re-frozen if the review changes it)
K_DIRS = 4
KL_DECISIONS = 600
REPLENISH_SWAMP = 0.5  # replenish's share of the squared producer score above which it "swamps" the signal


def registered_prereg_v16() -> dict:
    return {
        "format": "m6-prereg-v16", "change": "add-m6-joint-mpg-gnn", "task": "4.4",
        "stage": "EXP-0127: power check for on-policy RL of every producer head (F40 + v15)",
        "start": "the F40 + v15 composite: EXP-0126 models.pt (F40 trunk, gates, v15 turn heads) + the EXP-0110 "
                 "start's replenish head on its own frozen trunk (only the replenish head trains); orderer frozen",
        "exploration": "gates Bernoulli(σ(logit / TEMPS)); replenish σ(z + σ_r ε); fund and accept-turn orders "
                       "Plackett–Luce at τ_f / τ_p; accept bits Bernoulli(σ(a / T_a)) with a from the pass in the "
                       "sampled order; targets: replenish mean |Δfrac| 0.02, adjacent-pair swaps 5%, bit flips 5%; "
                       "calibrated at the start of the run on recorded start decisions (fixed draws)",
        "reward": "Φ with EXP-0110's 12-producer calibration and λ × 0.25 (4.5's re-weighting)",
        "seeds": f"m6_stage0's {s0.SEEDS_12} 12-producer seeds × G = {s0.G_SIBLINGS} siblings, horizon 100",
        "gradient": f"±ε along K = {K_DIRS} random directions of the trainable producer parameters, ε at a "
                    f"per-direction mean KL of {pw.KL_TARGET} on {KL_DECISIONS} recorded start decisions",
        "reading": sb.V9_RULE + " (12 producers only; the orderer is frozen in 4.5 and reported)",
        "reported": "each head family's share of the squared projected score (gates / replenish / fund order / "
                    "accept order / accept bits), flip and swap rates, deaths, seconds per episode",
        "counterfactual": "the 'replenish greedy' reading: the rollouts' scores without replenish's part; the "
                          "gradient size from separate ±ε episodes along the same K directions with the replenish "
                          "slice zeroed and renormalised, ε recalibrated at KL 0.01 with σ_r = 0, those episodes "
                          "played at σ_r = 0; P = the trainable count without the replenish head",
        "consequence": "`signal` licenses 4.5. If replenish's share of the squared score exceeds "
                       f"{REPLENISH_SWAMP} it swamps the signal: 4.5 then plays replenish greedily, i.e. the "
                       "replenish head is exactly frozen at σ_r = 0 (supervisor 2026-11-07), reported; the "
                       "counterfactual reading licenses 4.5 in that case",
    }


def freeze() -> str:
    from ..eval.provenance import sha256_file

    PREREG_V16_PATH.write_text(json.dumps(registered_prereg_v16(), indent=1, sort_keys=True) + "\n")
    return sha256_file(str(PREREG_V16_PATH))


def prereg_problems() -> list[str]:
    from ..eval.provenance import sha256_file

    if PREREG_V16_SHA256 is None:
        return ["no pinned m6_prereg_v16.json (`--mode freeze`)"]
    if not PREREG_V16_PATH.exists() or sha256_file(str(PREREG_V16_PATH)) != PREREG_V16_SHA256:
        return ["m6_prereg_v16.json does not match its pinned SHA-256"]
    if json.loads(PREREG_V16_PATH.read_text()) != registered_prereg_v16():
        return ["the v16 pre-registration differs from the code"]
    return []


# --- perturbation along the trainable parameters ---


def replenish_mask(model: op.F40Model) -> torch.Tensor:
    """1 on the trainable entries that belong to the replenish head (`rep.replenish.*`), 0 elsewhere."""
    names = {id(p): n for n, p in model.named_parameters()}
    return torch.cat([torch.full((p.numel(),), float(names[id(p)].startswith("rep.replenish.")))
                      for p in model.trainable()])


def perturbed(model: op.F40Model, k: int, step: float, norep: bool = False) -> op.F40Model:
    """`step` · u_k on the trainable parameters; `norep` (the counterfactual of 4.5 playing replenish greedily):
    u_k with the replenish slice zeroed and renormalised."""
    m = copy.deepcopy(model)
    ps = m.trainable()
    n = sum(p.numel() for p in ps)
    u = pw.direction(n, "producer", k)
    if norep:
        u = u * (1 - replenish_mask(m))
        u = u / u.norm()
    with torch.no_grad():
        i = 0
        for p in ps:
            p.add_(step * u[i:i + p.numel()].view_as(p))
            i += p.numel()
    return m


def mean_kl(model: op.F40Model, k: int, eps: float, decs: Sequence[dict], temps: op.Temps,
            norep: bool = False) -> float:
    m1 = perturbed(model, k, eps, norep)
    with torch.no_grad():
        kls = []
        for d in decs:
            _l0, p0 = op.decision_logp(model, d, temps)
            _l1, p1 = op.decision_logp(m1, d, temps)
            kls.append(float(op.decision_kl(p0, p1, d, temps)))
    return fmean(kls)


# --- jobs ---

_MODELS: dict[tuple, Any] = {}


def _models(job: dict) -> tuple:
    key = (job["turn_models"], job["replenish_models"])
    if key not in _MODELS:
        m, om = op.load_start(*key)
        m0, _ = op.load_start(*key)
        _MODELS[key] = (m, m0, om)
    return _MODELS[key]


def _env_tracker(job: dict) -> tuple[Any, Any]:
    from ..env.potential import PotentialTracker, PotentialWeights

    env = op.turn_env(job)
    tr = PotentialTracker(PotentialWeights(**job["weights"]))
    tr.attach(env)
    return env, tr


def rollout_job(job: dict) -> dict:
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    m, m0, om = _models(job)
    temps = op.temps_from(job["temps"])
    env, tr = _env_tracker(job)
    rng = np.random.default_rng([s0.PROJ_SEED, 44, int(job["episode"]) % (1 << 31), int(job["sibling"])])
    pol = op.OnPolicy(m, m0, om, rng, temps, score=True)
    t0 = time.perf_counter()
    run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None,
                fund_policy=pol.fund, accept_policy=pol.accept)
    secs = time.perf_counter() - t0
    path = Path(job["npz_dir"]) / f"{job['size']}_{job['world']}_{job['episode']}_{job['sibling']}.npz"
    arrays = {}
    for role, r in pol.rec.items():
        arrays[f"{role}_cycle"] = np.asarray(r["cycle"], np.int32)
        arrays[f"{role}_agent"] = np.asarray(r["agent"])
        arrays[f"{role}_emb"] = np.stack(r["emb"]) if r["emb"] else np.zeros((0, 1), np.float32)
        arrays[f"{role}_score"] = np.stack(r["score"]) if r["score"] else np.zeros((0, s0.PROJ_K), np.float32)
    fam = np.asarray(pol.rec["producer"]["family_sq"], np.float64).reshape(-1, len(op.FAMILIES))
    np.savez(path, **arrays)
    # the counterfactual without replenish's score (as if replenish were played greedily): a second npz
    nr = pol.rec["producer"]["score_norep"]
    path_nr = path.with_name(path.stem + "_norep.npz")
    np.savez(path_nr, **{**arrays, "producer_score": np.stack(nr) if nr else arrays["producer_score"]})
    return {"seed": f"{job['world']}|{job['episode']}", "size": job["size"], "sibling": job["sibling"],
            "rewards": tr.rewards(float(job["unit_scale"])), "units": tr.totals()["units"],
            "dead": bool(env.episode_stats["dead"]), "seconds": secs, "npz": str(path), "npz_norep": str(path_nr),
            "horizon": job["horizon"],
            "flips": {k: pol.counts[f"flip_{k}"] for k in sb.GATES}, "counts": pol.counts,
            "family_sq": dict(zip(op.FAMILIES, fam.sum(0).tolist())),
            "decisions": {r: len(v["cycle"]) for r, v in pol.rec.items()}}


def perturb_job(job: dict) -> dict:
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    m, m0, om = _models(job)
    mp_ = perturbed(m, job["k"], job["sign"] * job["eps"], bool(job.get("norep")))
    temps = op.temps_from(job["temps"])
    env, tr = _env_tracker(job)
    rng = np.random.default_rng([s0.PROJ_SEED, 44, int(job["episode"]) % (1 << 31), 0])
    pol = op.OnPolicy(mp_, m0, om, rng, temps)
    run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None,
                fund_policy=pol.fund, accept_policy=pol.accept)
    return {**{k: job[k] for k in ("size", "role", "k", "sign", "world", "episode", "eps")},
            "ret": float(sum(tr.rewards(float(job["unit_scale"])))), "type": "perturb",
            "norep": bool(job.get("norep"))}


def any_job(job: dict) -> dict:
    return rollout_job(job) if job["type"] == "rollout" else perturb_job(job)


def start_decisions(turn_models: str, replenish_models: str, seeds: Sequence[dict], temps: op.Temps,
                    n: int) -> list[dict]:
    """Decisions of the start on a few seeds (sibling 0), every k-th kept: the KL calibration set."""
    from ..eval.runner import run_episode

    m, om = op.load_start(turn_models, replenish_models)
    decs: list[dict] = []
    for j in seeds:
        env, _tr = _env_tracker(j)
        pol = op.OnPolicy(m, m, om, np.random.default_rng([s0.PROJ_SEED, 4404]), temps)
        run_episode(env, pol.producing, pol.ordering, seed=j["episode"], action_seed=None,
                    fund_policy=pol.fund, accept_policy=pol.accept)
        decs += pol.decisions
    step = max(1, len(decs) // n)
    return decs[::step][:n]


def eps_for(model: op.F40Model, k: int, decs: Sequence[dict], temps: op.Temps, norep: bool = False) -> dict:
    lo, hi = 1e-5, 1e3
    for _ in range(24):
        mid = math.sqrt(lo * hi)
        if mean_kl(model, k, mid, decs, temps, norep) > pw.KL_TARGET:
            hi = mid
        else:
            lo = mid
    e = math.sqrt(lo * hi)
    return {"k": k, "eps": e, "kl": mean_kl(model, k, e, decs, temps, norep), "at_bound": hi >= 1e3}


def eps_job(job: dict) -> dict:
    """One direction's ε search in a pool worker (the parent's serial search left 15 of 16 cores idle)."""
    torch.set_num_threads(1)
    m, _m0, _om = _models(job)
    out = eps_for(m, job["k"], job["decs"], op.temps_from(job["temps"]), job["norep"])
    return {**out, "norep": job["norep"]}


def run(training_dir: str, atlas_path: str, turn_models: str, replenish_models: str, exp0110: str,
        monitor_pkl: str, turn_records: str, out_dir: str, workers: int, dry_run: bool = False,
        time_only: bool = False) -> int:
    import pickle

    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas
    from . import m6_turn_imitation as ti

    atlas = load_atlas(atlas_path)
    if not (dry_run or time_only):
        im._refuse(prereg_problems()
                   + ([] if atlas.get("content_hash") == im.REGISTERED_ATLAS_HASH else ["not the registered atlas v2"]))
    out = Path(out_dir)
    (out / "npz").mkdir(parents=True, exist_ok=True)
    cal = json.loads((Path(exp0110) / "calibration.json").read_text())
    _ts, j12, _j42 = s0.world_episodes(training_dir, atlas)
    seeds = [x for x in s0.stage0_seeds(j12, []) if x["size"] == 12]
    kdirs = K_DIRS
    if dry_run:
        seeds, kdirs = [{**x, "horizon": 6} for x in seeds[:2]], 1
    # temperatures (offline, recorded)
    m, _om = op.load_start(turn_models, replenish_models)
    with open(monitor_pkl, "rb") as fh:
        mo = pickle.load(fh)
    prod_feats = [f if f["capacity"].shape[1] == g.CAPACITY_DIM[g.FEATURES_V4] else g.v3_to_v4(f)
                  for f, _y in mo["producer"][::32][:60 if dry_run else 600]]
    del mo
    recs = ti.load_records(Path(turn_records), 600 if dry_run else 6000)
    calib = op.calibrate(m, prod_feats, recs)
    temps = op.temps_from(calib)
    (out / "calibration.json").write_text(json.dumps(calib, indent=1) + "\n")
    print(f"[stage0c] temps {calib['temps']} achieved {calib['achieved']}", flush=True)
    c = cal["per_size"]["12"]
    weights = {"lam": c["lambda"] * op.CFG["lambda_scale"], "mu": c["mu"], "x": cal["x"], "y": cal["y"]}
    base_jobs = []
    for j in seeds:
        scale = j["horizon"] / cal["bar"][f"{j['world']}|{j['episode']}"]
        base_jobs.append({**j, "replicate": s0.STAGE0_REPLICATE_BASE, "turn_models": turn_models,
                          "replenish_models": replenish_models, "weights": {**weights, "nu": float(j["horizon"])},
                          "unit_scale": scale, "temps": calib})
    if time_only:  # one real seed: G scored siblings + one ± perturbation pair, timed (the 4.4 sizing)
        t0 = time.perf_counter()
        rows = [rollout_job({**base_jobs[0], "replicate": s0.STAGE0_REPLICATE_BASE + i, "sibling": i,
                             "type": "rollout", "npz_dir": str(out / "npz")}) for i in range(s0.G_SIBLINGS)]
        t_roll = time.perf_counter() - t0
        t1 = time.perf_counter()
        for sign in (1, -1):
            perturb_job({**base_jobs[0], "type": "perturb", "role": "producer", "k": 0, "sign": sign, "eps": 1e-3})
        t_pert = time.perf_counter() - t1
        doc = {"time_only": True, "scored_sibling_seconds": t_roll / s0.G_SIBLINGS,
               "perturb_episode_seconds": t_pert / 2, "decisions": rows[0]["decisions"], "calibration": calib}
        (out / "timing.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
        print(f"[stage0c] timing {doc['scored_sibling_seconds']:.1f} s per scored sibling, "
              f"{doc['perturb_episode_seconds']:.1f} s per perturbed episode", flush=True)
        return 0
    decs = start_decisions(turn_models, replenish_models, base_jobs[:2], temps, KL_DECISIONS)
    # the "replenish greedy" counterfactual: directions without the replenish slice, ε recalibrated, the
    # perturbed episodes played (and their KL measured) at σ_r = 0
    calib0 = {**calib, "temps": {**calib["temps"], "replenish_sigma": 0.0}}
    ejobs = [{"turn_models": turn_models, "replenish_models": replenish_models, "k": k, "norep": nr,
              "decs": decs, "temps": calib0 if nr else calib} for k in range(kdirs) for nr in (False, True)]
    found = pw._pool(eps_job, ejobs, workers, out / "eps_rows.jsonl")
    eps = {str(e["k"]): e for e in found if not e["norep"]}
    eps0 = {str(e["k"]): e for e in found if e["norep"]}
    print(f"[stage0c] eps {json.dumps({k: round(v['eps'], 5) for k, v in eps.items()})} "
          f"norep {json.dumps({k: round(v['eps'], 5) for k, v in eps0.items()})}", flush=True)
    jobs = []
    for b in base_jobs:
        for i in range(s0.G_SIBLINGS):
            jobs.append({**b, "replicate": s0.STAGE0_REPLICATE_BASE + i, "sibling": i, "type": "rollout",
                         "npz_dir": str(out / "npz")})
        for k in range(kdirs):
            for sign in (1, -1):
                jobs.append({**b, "type": "perturb", "role": "producer", "k": k, "sign": sign,
                             "eps": eps[str(k)]["eps"]})
                jobs.append({**b, "type": "perturb", "role": "producer", "k": k, "sign": sign, "norep": True,
                             "eps": eps0[str(k)]["eps"], "temps": calib0})
    jobs.sort(key=lambda j: j["type"] != "rollout")
    rows = pw._pool(any_job, jobs, workers, out / "episode_rows.jsonl")
    meta = [r for r in rows if r.get("type") != "perturb"]
    prows = [r for r in rows if r.get("type") == "perturb" and not r.get("norep")]
    prows0 = [r for r in rows if r.get("type") == "perturb" and r.get("norep")]
    n_params = {"producer": sum(p.numel() for p in m.trainable()),
                "orderer": sum(p.numel() for p in _om.parameters())}
    n_params0 = {**n_params, "producer": n_params["producer"] - int(replenish_mask(m).sum())}
    doc = sb.analyse(meta, prows, n_params)
    fam_tot = {k: sum(r["family_sq"][k] for r in meta) for k in op.FAMILIES}
    s = sum(fam_tot.values()) or 1.0
    # pre-registered counterfactual: the same reading with replenish's score removed (used for 4.5's choice if
    # replenish's share of the squared score exceeds REPLENISH_SWAMP)
    nr = sb.analyse([{**r, "npz": r["npz_norep"]} for r in meta], prows0, n_params0)
    doc["without_replenish_score"] = {k: nr[k] for k in ("reading", "estimators", "seeds_needed")}
    doc["replenish_swamps"] = fam_tot["replenish"] / s > REPLENISH_SWAMP
    counts = {k: [sum(r["counts"][k][0] for r in meta), sum(r["counts"][k][1] for r in meta)]
              for k in meta[0]["counts"]} if meta else {}
    doc.update({"family_score_share": {k: v / s for k, v in fam_tot.items()},
                "rates": {k: (v[0] / v[1] if v[1] else None) for k, v in counts.items()},
                "calibration": calib, "eps": eps, "eps_norep": eps0, "dry_run": dry_run, "prereg": PREREG_V16_SHA256,
                "source_commit": im.SOURCE_COMMIT, "turn_models_sha256": sha256_file(turn_models),
                "replenish_models_sha256": sha256_file(replenish_models), "jobs": len(jobs), "rows": len(rows),
                "seconds_per_scored_episode": fmean(r["seconds"] for r in meta) if meta else None})
    (out / "reading.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[stage0c] reading {doc['reading']} estimators {doc['estimators']} shares {doc['family_score_share']}",
          flush=True)
    return 0 if len(rows) == len(jobs) else 2


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("freeze", "run", "time"), required=True)
    ap.add_argument("--training-dir")
    ap.add_argument("--atlas")
    ap.add_argument("--turn-models", help="EXP-0126 models.pt (F40 trunk + v15 heads)")
    ap.add_argument("--replenish-models", help="EXP-0110 start_models.pt")
    ap.add_argument("--exp0110", help="EXP-0110 artifacts (calibration.json)")
    ap.add_argument("--monitor-pkl", help="EXP-0105 samples_monitor.pkl")
    ap.add_argument("--turn-records", help="EXP-0126 heldout_records.npz")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir")
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "freeze":
        print(freeze())
        return 0
    im.SOURCE_COMMIT = a.source_commit
    return run(a.training_dir, a.atlas, a.turn_models, a.replenish_models, a.exp0110, a.monitor_pkl,
               a.turn_records, a.out_dir, a.workers, a.dry_run, time_only=a.mode == "time")


if __name__ == "__main__":
    sys.exit(main())
