"""EXP-0111: is EXP-0110's `no_signal` an absent gradient or an underpowered test? (DIAGNOSTICS 35)

`python -m tokenomics.train.m6_power --rows ROWS --exp0110 DIR --training-dir T --atlas A
  --monitor-samples M --out-dir D [--workers N]`

Pre-registered in openspec/changes/add-m6-joint-mpg-gnn/notes/exp0111-prereg.md (SHA pinned below), before any
computation:
1. **Noise model:** per size, role and estimator, the per-seed gradient estimates from EXP-0110's rows. Their
   centred values are resampled with replacement, which keeps the noise's low-dimensional structure.
2. **Planted-signal sweep:** the smallest signal, as a multiple of the per-seed noise RMS, that a simulated D0.3
   detects with ≥ 80% probability at 12, 24 and 48 seeds per half.
3. **True gradient size:** paired ±εu policy perturbations per role on sibling 0's seeds and draws (common random
   numbers), at a PPO-sized ε (mean per-decision KL 0.01 on the monitor samples).
4. **n80:** the seeds per half at which a gradient of the measured size is detected with ≥ 80% probability.
5. **Mechanism:** the producer score split into its gate and replenish terms, the orderer's score, and the gate
   flip rate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from ..env.potential import PotentialTracker, PotentialWeights
from . import m6_gnn as g
from . import m6_imitation as im
from . import m6_mappo as mp
from . import m6_stage0 as s0
from . import producer_es as pe

NOTE = Path(__file__).resolve().parents[3] / "openspec/changes/add-m6-joint-mpg-gnn/notes/exp0111-prereg.md"
NOTE_SHA256 = "43b02d2332c31edceffbbd2fc71e7b09830e307dd28e36d950c57e18e7665e05"
ROWS_SHA256 = "eb4bfa3906cec2c114e1"  # prefix of EXP-0110's stage0_rows.jsonl SHA-256
NS = (12, 24, 48, 96, 192, 384)
SWEEP_NS = (12, 24, 48)
DRAWS = 2000
KL_TARGET = 0.01
K_DIRS = {12: 4, 42: 3}
SEEDS_42 = 4
MECH_SEEDS = {12: 4, 42: 1}
ROLES = ("producer", "orderer")
ESTIMATORS = ("E1", "E2", "E3")


# --- 1 / 2 / 4: the noise model and the simulated D0.3 ---


def per_seed_estimates(seeds: Mapping[str, list[dict]], est: str, gamma: float
                       ) -> dict[str, np.ndarray]:
    """(1 / G) Σ_i Σ_t A_t s_i,t per seed, per role (projected)."""
    adv = s0.advantages(seeds, est, gamma)
    keys = sorted(seeds)
    return {r: np.stack([s0.gradient(seeds, adv, r, [k]) / s0.G_SIBLINGS for k in keys])
            for r in ROLES}


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(na * nb, 1e-300)


def detection(x: np.ndarray, mu_norm: float, n: int, rng: np.random.Generator,
              draws: int = DRAWS) -> tuple[float, float]:
    """(power, null q95): halves of n seeds, each seed = μ + a resampled centred estimate; μ =
    mu_norm along a random unit direction (shared by both halves of a draw)."""
    res = x - x.mean(0, keepdims=True)
    k = x.shape[1]

    m = len(res)

    def halves(mu: np.ndarray) -> np.ndarray:  # resampling as counts: memory-flat in n
        c1 = rng.multinomial(n, [1.0 / m] * m, size=draws) / n
        c2 = rng.multinomial(n, [1.0 / m] * m, size=draws) / n
        return _cos_rows(c1 @ res + mu, c2 @ res + mu)

    null = halves(np.zeros((draws, k)))
    q95 = float(np.quantile(null, 0.95))
    u = rng.normal(size=(draws, k))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return float((halves(mu_norm * u) > q95).mean()), q95


def noise_rms(x: np.ndarray) -> float:
    res = x - x.mean(0, keepdims=True)
    return float(np.sqrt((res ** 2).sum(1).mean()))


def planted_sweep(x: np.ndarray, rng: np.random.Generator) -> dict:
    """Per n: the smallest m (μ = m · noise RMS) detected with power >= 0.8."""
    rms = noise_rms(x)
    grid = np.geomspace(0.02, 5.0, 25)
    out = {}
    for n in SWEEP_NS:
        hit = None
        for m in grid:
            if detection(x, float(m) * rms, n, rng, 500)[0] >= 0.8:
                hit = float(m)
                break
        out[str(n)] = hit
    return {"noise_rms": rms, "min_detectable_multiple": out}


def n80(x: np.ndarray, mu_norm: float, rng: np.random.Generator) -> tuple[int | None, dict]:
    powers = {}
    for n in NS:
        p, _q = detection(x, mu_norm, n, rng)
        powers[str(n)] = p
        if p >= 0.8:
            return n, powers
    return None, powers


# --- 3: the true gradient size (paired policy perturbations) ---


def role_params(pm: g.ProducerGNN, om: g.OrdererGNN, role: str) -> list[torch.Tensor]:
    return list((pm if role == "producer" else om).parameters())


def direction(n: int, role: str, k: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(7000 + 100 * ROLES.index(role) + k)
    u = torch.randn(n, generator=gen)
    return u / u.norm()


def perturbed(models: str | tuple, role: str, k: int, step: float
              ) -> tuple[g.ProducerGNN, g.OrdererGNN]:
    """The start models (a path, or a loaded (pm, om) pair, copied) moved by step · u_k."""
    import copy

    pm, om = im.load_models(models) if isinstance(models, str) else copy.deepcopy(models)
    ps = role_params(pm, om, role)
    n = sum(p.numel() for p in ps)
    u = direction(n, role, k)
    with torch.no_grad():
        i = 0
        for p in ps:
            m = p.numel()
            p.add_(step * u[i:i + m].view_as(p))
            i += m
    return pm, om


def mean_kl(base: tuple, role: str, k: int, eps: float, prod: list, orde: list) -> float:
    """Mean per-decision KL(π_θ ‖ π_θ+εu) of one role on recorded decisions (the RL policy:
    Bernoulli gates + Gaussian replenish; categorical slots)."""
    pm0, om0 = base
    pm1, om1 = perturbed(base, role, k, eps)
    sd = math.exp(s0.INIT_LOG_STD)
    kls = []
    with torch.no_grad():
        if role == "producer":
            ls = torch.nn.functional.logsigmoid
            for f, y in prod:
                f4 = g.v3_to_v4(f) if f["capacity"].shape[1] == 17 else f
                ga, la = pm0.heads(f4)
                gb, lb = pm1.heads(f4)
                pa = torch.sigmoid(ga)
                kl = (pa * (ls(ga) - ls(gb)) + (1 - pa) * (ls(-ga) - ls(-gb))).sum()
                if la.shape[0]:
                    m = torch.from_numpy(y["mask"]) > 0
                    kl = kl + ((la - lb)[m] ** 2).sum() / (2 * sd * sd)
                kls.append(float(kl))
        else:
            for f, _y in orde:
                tot = 0.0
                for a, b in zip(om0(f), om1(f)):
                    if a.shape[0] > 1:
                        la, lb = torch.log_softmax(a, 0), torch.log_softmax(b, 0)
                        tot += float((la.exp() * (la - lb)).sum())
                kls.append(tot)
    return fmean(kls) if kls else 0.0


def calibrate_eps(models: str, role: str, k: int, prod: list, orde: list) -> dict:
    """Per direction k: ε with mean per-decision KL = KL_TARGET (bisection in log ε over
    [1e-5, 1e4]; a random direction moves the orderer little, so the bound is wide)."""
    base = im.load_models(models)
    lo, hi = 1e-5, 1e4
    for _ in range(26):  # ε to within a factor of ≈ 1.002
        mid = math.sqrt(lo * hi)
        if mean_kl(base, role, k, mid, prod, orde) > KL_TARGET:
            hi = mid
        else:
            lo = mid
    eps = math.sqrt(lo * hi)
    return {"eps": eps, "kl": mean_kl(base, role, k, eps, prod, orde), "at_bound": hi >= 1e4}


class _Plain(mp.RecordingPolicies):
    """The RL policies' sampling without recording (memory flat)."""

    def producing(self, env: Any, agent: str) -> np.ndarray:
        act = super().producing(env, agent)
        self.decisions.clear()
        self.cycle_inputs.clear()
        return act

    def ordering(self, env: Any, agent: str) -> np.ndarray:
        act = super().ordering(env, agent)
        self.decisions.clear()
        self.cycle_inputs.clear()
        return act


def perturb_job(job: dict) -> dict:
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    pm, om = perturbed(job["models"], job["role"], job["k"], job["sign"] * job["eps"])
    ex = mp.ActorExtras(s0.INIT_LOG_STD)
    env = im._episode_env(job)
    tr = PotentialTracker(PotentialWeights(**job["weights"]))
    tr.attach(env)
    rng = np.random.default_rng([s0.PROJ_SEED, int(job["episode"]) % (1 << 31), 0])  # sibling 0
    pol = _Plain(pm, om, ex, rng)
    run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None)
    return {**{k: job[k] for k in ("size", "role", "k", "sign", "world", "episode", "eps")},
            "ret": float(sum(tr.rewards(float(job["unit_scale"]))))}


def gradient_size(rows: Sequence[dict], n_params: Mapping[str, int], rng: np.random.Generator
                  ) -> dict:
    """Per role and size: D_k per direction (mean over seeds), |∇J|² = P (mean D² − mean SE²),
    and a 95% interval from a bootstrap over seeds and directions."""
    out: dict[str, Any] = {}
    for size in (12, 42):
        for role in ROLES:
            rs = [r for r in rows if r["size"] == size and r["role"] == role]
            if not rs:
                continue
            ks = sorted({r["k"] for r in rs})
            eps_k = {k: next(r["eps"] for r in rs if r["k"] == k) for k in ks}
            eps = fmean(eps_k.values())
            d = {}
            for k in ks:
                pairs: dict[str, dict[int, float]] = {}
                for r in rs:
                    if r["k"] == k:
                        pairs.setdefault(f"{r['world']}|{r['episode']}", {})[r["sign"]] = r["ret"]
                d[k] = np.asarray([(v[1] - v[-1]) / (2 * eps_k[k]) for v in pairs.values()
                                   if 1 in v and -1 in v])
            P = n_params[role]

            def est(dd: Mapping[int, np.ndarray]) -> float:
                return P * fmean(float(v.mean() ** 2 - v.var(ddof=1) / len(v)) for v in dd.values())

            g2 = est(d)
            boots = []
            for _ in range(1000):
                kk = rng.choice(ks, size=len(ks))
                boots.append(est({j: d[k][rng.integers(0, len(d[k]), len(d[k]))]
                                  for j, k in enumerate(kk)}))
            lo, hi = np.quantile(boots, [0.025, 0.975])
            out[f"{size}/{role}"] = {
                "eps": eps, "params": P, "D_by_direction": {str(k): float(v.mean()) for k, v in d.items()},
                "D_se": {str(k): float(v.std(ddof=1) / math.sqrt(len(v))) for k, v in d.items()},
                "grad_norm_sq": g2, "grad_norm_sq_ci95": [float(lo), float(hi)],
                "grad_norm": math.sqrt(max(g2, 0.0)),
                "improvement_per_update": eps * math.sqrt(max(g2, 0.0))}
    return out


# --- 5: mechanism ---


class _HeadScores(s0.ScorePolicies):
    """Per cycle: the squared norms of the producer gate-term and replenish-term scores and the
    orderer score (full parameter space), and the gate flips."""

    def __init__(self, *a: Any) -> None:
        super().__init__(*a)
        self.acc = {"gate": None, "replenish": None, "orderer": None}
        self.norms: list[dict[str, float]] = []
        self.flips = [0, 0]

    def _flush(self) -> None:
        if self.cycle is None:
            return
        rec = {}
        for term, ps in (("gate", self.params["producer"]), ("replenish", self.params["producer"]),
                         ("orderer", self.params["orderer"])):
            a = self.acc[term]
            if a is None or not a.requires_grad:
                rec[term] = 0.0
                continue
            gr = torch.autograd.grad(a, ps, allow_unused=True, retain_graph=True)  # shared graph
            rec[term] = float(sum((x ** 2).sum() for x in gr if x is not None))
        self.norms.append(rec)
        self.acc = {"gate": None, "replenish": None, "orderer": None}
        self.cycle_inputs = {}

    def _add2(self, env: Any, term: str, lp: torch.Tensor) -> None:
        c = int(env.current_cycle)
        if c != self.cycle:
            self._flush()
            self.cycle = c
        self.acc[term] = lp if self.acc[term] is None else self.acc[term] + lp

    def producing(self, env: Any, agent: str) -> np.ndarray:
        act = mp.RecordingPolicies.producing(self, env, agent)
        d = self.decisions.pop()
        gd, rd = mp.producer_dist(self.pm, self.ex, d["f"])
        self._add2(env, "gate", gd.log_prob(torch.from_numpy(d["bits"].astype(np.float32))).sum())
        if d["mask"].any():
            lp = rd.log_prob(torch.from_numpy(d["z"].astype(np.float32)))[torch.from_numpy(
                d["mask"])].sum()
            self._add2(env, "replenish", lp)
        with torch.no_grad():
            argmax = (gd.logits > 0).numpy()
        self.flips[0] += int((argmax != d["bits"]).sum())
        self.flips[1] += 3
        return act

    def ordering(self, env: Any, agent: str) -> np.ndarray:
        n = len(self.decisions)
        act = mp.RecordingPolicies.ordering(self, env, agent)
        if len(self.decisions) > n:
            d = self.decisions.pop()
            lp, _ = mp.orderer_logp(self.om, d["f"], d["picks"])
            self._add2(env, "orderer", lp)
        return act


def eps_job(job: dict) -> dict:
    torch.set_num_threads(1)
    return {"role": job["role"], "k": job["k"],
            **calibrate_eps(job["models"], job["role"], job["k"], job["prod"], job["orde"])}


def any_job(job: dict) -> dict:
    """The pool's dispatcher: perturbation and mechanism episodes share one pool."""
    return {**(mechanism_job(job) if job["type"] == "mechanism" else perturb_job(job)),
            "type": job["type"]}


def mechanism_job(job: dict) -> dict:
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    pm, om = im.load_models(job["models"])
    ex = mp.ActorExtras(s0.INIT_LOG_STD)
    env = im._episode_env(job)
    rng = np.random.default_rng([s0.PROJ_SEED, int(job["episode"]) % (1 << 31), 0])
    pol = _HeadScores(pm, om, ex, rng)
    run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None)
    pol._flush()
    tot = {t: sum(r[t] for r in pol.norms) for t in ("gate", "replenish", "orderer")}
    return {"size": job["size"], "world": job["world"], "episode": job["episode"], **tot,
            "gate_flip_rate": pol.flips[0] / max(1, pol.flips[1])}


# --- the run ---


def _pool(fn: Any, jobs: Sequence[dict], workers: int, path: Path) -> list[dict]:
    rows = []
    with path.open("w") as fh:
        if workers <= 1:
            for j in jobs:
                rows.append(fn(j))
                fh.write(json.dumps(rows[-1]) + "\n")
                fh.flush()
            return rows
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                                 initializer=pe._init_worker) as ex:
            for fut in as_completed([ex.submit(fn, j) for j in jobs]):
                rows.append(fut.result())
                fh.write(json.dumps(rows[-1]) + "\n")
                fh.flush()
    return rows


def run(rows_path: str, exp0110: str, training_dir: str, atlas: str, monitor_pkl: str,
        out_dir: str, workers: int, dry_run: bool = False, sub: int | None = None,
        note: str | None = None) -> int:
    import pickle

    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas

    if not dry_run:
        p = []
        if sha256_file(note or str(NOTE)) != NOTE_SHA256:  # the pod gets a staged copy
            p.append("the pre-registration note is not the pinned one")
        if not sha256_file(rows_path).startswith(ROWS_SHA256):
            p.append("the rows are not EXP-0110's")
        im._refuse(p)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    e10 = Path(exp0110)
    reading10 = json.loads((e10 / "reading.json").read_text())
    cal = json.loads((e10 / "calibration.json").read_text())
    models = str(e10 / "start_models.pt")
    rng = np.random.default_rng(20261107)
    rows = [json.loads(line) for line in open(rows_path)]
    # 1 / 2: the noise model per size, role and estimator
    noise: dict[str, Any] = {}
    xs: dict[tuple, np.ndarray] = {}
    for size in (12, 42):
        seeds = s0._by_seed([r for r in rows if r["size"] == size])
        gam = reading10["per_size"][str(size)]["gamma"]
        for est in ESTIMATORS:
            pe_ = per_seed_estimates(seeds, est, gam)
            for role in ROLES:
                xs[(size, est, role)] = pe_[role]
                noise[f"{size}/{est}/{role}"] = {
                    "seeds": len(pe_[role]), **planted_sweep(pe_[role], rng),
                    "mean_norm": float(np.linalg.norm(pe_[role].mean(0)))}
    del rows
    print(f"[m6_power] planted sweep done", flush=True)
    # 3: ε per role, then the paired perturbations
    with open(monitor_pkl, "rb") as fh:
        mo = pickle.load(fh)
    prod, orde = mo["producer"][::32][:sub or 600], mo["orderer"][::80][:sub or 600]
    del mo
    ejobs = [{"models": models, "role": role, "k": k, "prod": prod, "orde": orde}
             for role in ROLES for k in range(1 if dry_run else max(K_DIRS.values()))]
    eps: dict[str, dict] = {r: {} for r in ROLES}
    for e in _pool(eps_job, ejobs, workers, out / "eps_rows.jsonl"):
        eps[e["role"]][str(e["k"])] = {x: e[x] for x in ("eps", "kl", "at_bound")}
    print(f"[m6_power] eps {eps}", flush=True)
    ts, j12, j42 = s0.world_episodes(training_dir, load_atlas(atlas))
    seeds = s0.stage0_seeds(j12, j42)
    seeds12 = [j for j in seeds if j["size"] == 12]
    seeds42 = [j for j in seeds if j["size"] == 42][:SEEDS_42]
    kdirs = dict(K_DIRS)
    if dry_run:  # a true smoke: 1 seed per size, horizon 4, one direction
        seeds12 = [{**j, "horizon": 4} for j in seeds12[:1]]
        seeds42 = [{**j, "horizon": 4} for j in seeds42[:1]]
        kdirs = {12: 1, 42: 1}
    jobs = []
    for j in seeds42 + seeds12:
        c = cal["per_size"][str(j["size"])]
        w = {"lam": c["lambda"], "mu": c["mu"], "nu": float(j["horizon"]), "x": cal["x"],
             "y": cal["y"]}
        scale = j["horizon"] / cal["bar"][f"{j['world']}|{j['episode']}"]
        for role in ROLES:
            for k in range(kdirs[j["size"]]):
                for sign in (1, -1):
                    jobs.append({**j, "replicate": s0.STAGE0_REPLICATE_BASE, "models": models,
                                 "type": "perturb", "role": role, "k": k, "sign": sign,
                                 "eps": eps[role][str(k)]["eps"],
                                 "weights": w, "unit_scale": scale})
    for j in seeds42[:MECH_SEEDS[42]] + seeds12[:MECH_SEEDS[12]]:
        jobs.append({**j, "replicate": s0.STAGE0_REPLICATE_BASE, "models": models,
                     "type": "mechanism"})
    jobs.sort(key=lambda j: (-j["size"], j["type"] != "mechanism"))  # the longest first
    allrows = _pool(any_job, jobs, workers, out / "episode_rows.jsonl")
    prows = [r for r in allrows if r["type"] == "perturb"]
    mrows = [r for r in allrows if r["type"] == "mechanism"]
    pm0, om0 = im.load_models(models)
    n_params = {r: sum(p.numel() for p in role_params(pm0, om0, r)) for r in ROLES}
    grad = gradient_size(prows, n_params, rng)
    print(f"[m6_power] gradient size {json.dumps(grad)[:600]}", flush=True)
    # 4: seeds needed for the measured size, per estimator; the best estimator per role
    need: dict[str, Any] = {}
    for size in (12, 42):
        for role in ROLES:
            gs = grad.get(f"{size}/{role}")
            if gs is None:
                continue
            per = {}
            for est in ESTIMATORS:
                n, powers = n80(xs[(size, est, role)], gs["grad_norm"], rng)
                per[est] = {"n80": n, "power": powers}
            best = min(per, key=lambda e: per[e]["n80"] or 10 ** 9)
            need[f"{size}/{role}"] = {"per_estimator": per, "best": best, "n80": per[best]["n80"]}
    # 5: mechanism (its episodes ran in the shared pool above)
    mech = {}
    for size in (12, 42):
        rs = [r for r in mrows if r["size"] == size]
        if rs:
            tot = {t: sum(r[t] for r in rs) for t in ("gate", "replenish", "orderer")}
            pt = tot["gate"] + tot["replenish"]
            mech[str(size)] = {"producer_gate_share": tot["gate"] / pt if pt else None,
                               "producer_replenish_share": tot["replenish"] / pt if pt else None,
                               "score_sq": tot, "gate_flip_rate": fmean(r["gate_flip_rate"]
                                                                       for r in rs)}
    # reading
    ok = all(grad[f"12/{r}"]["grad_norm_sq_ci95"][0] > 0 and need[f"12/{r}"]["n80"] is not None
             for r in ROLES)
    reading = "underpowered" if ok else "absent"
    doc = {"reading": reading, "note_sha256": NOTE_SHA256, "noise": noise, "eps": eps,
           "gradient": grad, "seeds_needed": need, "mechanism": mech,
           "stage0_seconds_per_episode": {s: reading10["per_size"][s]["D0_5_seconds_per_episode"]
                                          for s in reading10["per_size"]},
           "source_commit": im.SOURCE_COMMIT, "dry_run": dry_run}
    (out / "power.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[m6_power] reading {reading} n80 {{ {', '.join(f'{k}: {v['n80']}' for k, v in need.items())} }}",
          flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rows", required=True)
    p.add_argument("--exp0110", required=True)
    p.add_argument("--training-dir", required=True)
    p.add_argument("--atlas", required=True)
    p.add_argument("--monitor-samples", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sub", type=int)
    p.add_argument("--note", help="the pre-registration note (a staged copy on the pod)")
    p.add_argument("--source-commit", default="")
    a = p.parse_args(argv)
    im.SOURCE_COMMIT = a.source_commit
    torch.set_num_threads(1)
    return run(a.rows, a.exp0110, a.training_dir, a.atlas, a.monitor_samples, a.out_dir,
               a.workers, a.dry_run, a.sub, a.note)


if __name__ == "__main__":
    sys.exit(main())
