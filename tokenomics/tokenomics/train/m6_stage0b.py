"""EXP-0112: critic-free Stage 0 with reshaped exploration, plus the power check (pre-registration v9).

`python -m tokenomics.train.m6_stage0b --mode {freeze,run} ...`

User decision 2026-11-06, after EXP-0111 `absent` (DIAGNOSTICS 35: 99.4–99.7% of the producer score was
replenish noise, and the investment gates flipped about 1%):
- **Input buying:** frozen at the start model's deterministic replenish output (a frozen copy). There is no
  noise and it is outside the RL gradient, so the RL producer acts only through its three gates.
- **Gates:** restore / upgrade / expand are sampled as Bernoulli(sigmoid(logit / T_g)). T_g (TEMPS) was set
  offline so the greedy flip rate (sampled ≠ logit > 0) is 5% per gate on the 12-producer monitor samples
  (EXP-0105's recorded start-policy decisions). Training rollouts only; the deployed greedy policy is
  unchanged.
- **Orderer:** as in v8 (keyed softmax draw).
- **E4:** an advantage A_d = G_t − V(x_d) per decision d of agent-role r at cycle t. G_t is the team
  return-to-go (γ from D0.2); x_d is the frozen start actor's pooled embedding of the agent's own view, plus t / H.
  V is ridge regression (α = RIDGE_ALPHA), separate from the actor, cross-fitted on the split halves (V fitted on
  one half advantages the other). Its held-out explained variance is reported. E4 joins E1–E3 in D0.3 against a
  permutation null (decision advantages moved between seeds, aligned by sibling and decision index).
- **The power check** (as EXP-0111): the true gradient size per role from paired ±ε perturbations at a
  per-direction mean KL of 0.01 under this exploration policy, and n80 for each estimator.
- **Reading (V9_RULE):** `signal` if at 12 some estimator's producer cosine beats its null's 95th percentile and
  its n80 (at the measured |∇J|) is ≤ 96 seeds per half; else `absent`, with the mechanism (flip rates, score
  shares, E4's explained variance). 42 is reported (D0.5 cost, gradient).
"""

from __future__ import annotations

import argparse
import json
import math
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
from . import m6_mappo as mp
from . import m6_power as pw
from . import m6_stage0 as s0

PREREG_V9_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v9.json"
PREREG_V9_SHA256: str | None = "660a7597ab19b0cdffe3e962fe552afc975a51e1cb1c6184ad9406a059fedc45"  # frozen before EXP-0112 data
TEMPS = {"restore": 3.4522, "upgrade": 2.5629, "expand": 2.4004}  # 5% greedy flips (offline)
GATES = ("restore", "upgrade", "expand")
RIDGE_ALPHA = 1.0
N80_MAX = 96
ESTIMATORS4 = ("E1", "E2", "E3", "E4")
V9_RULE = (
    "signal iff at 12 producers some estimator (E1, E2, E3, E4) has a producer split-half cosine above its "
    "permutation null's 95th percentile AND its n80 at the measured |grad J| (point estimate, paired +-eps "
    "perturbations at a per-direction mean KL 0.01) is <= 96 seeds per half; else absent, with the mechanism "
    "(per-gate flip rates, score shares, E4's held-out explained variance). 42: D0.5 cost and the gradient, "
    "reported only")


def tempered_logits(gates: torch.Tensor) -> torch.Tensor:
    return gates / torch.tensor([TEMPS[k] for k in GATES])


# --- the exploration policy (training rollouts only) ---


class ExplorePolicies:
    """The RL policies of v9. The producer acts through tempered gates (actor `pm`); replenish is the
    frozen start's deterministic output (`pm0`). The orderer is the keyed softmax draw (`om`). With
    `score`, each decision's projected score (∇ log π of the sampled action, the actor's parameters),
    the frozen start's pooled embedding of the agent's view, its cycle and agent are recorded."""

    def __init__(self, pm: g.ProducerGNN, pm0: g.ProducerGNN, om: g.OrdererGNN, om0: g.OrdererGNN,
                 rng: np.random.Generator, score: bool) -> None:
        self.pm, self.pm0, self.om, self.om0, self.rng, self.score = pm, pm0, om, om0, rng, score
        self.rec: dict[str, dict[str, list]] = {r: {"cycle": [], "agent": [], "emb": [], "score": []}
                                                for r in ("producer", "orderer")}
        self.flips = {k: [0, 0] for k in GATES}
        self.params = {"producer": list(pm.parameters()), "orderer": list(om.parameters())}

    def _project(self, role: str, lp: torch.Tensor) -> np.ndarray:
        ps = self.params[role]
        gr = torch.autograd.grad(lp, ps, allow_unused=True)
        flat = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1)
                          for x, p in zip(gr, ps)])
        return (s0.projection(role, flat.numel()) @ flat).numpy().astype(np.float32)

    def producing(self, env: Any, agent: str) -> np.ndarray:
        pid = env._role[agent][1]
        f = g.producer_features(env.system.producer_entity_view(pid), g.producer_context(env, pid),
                                self.pm.features)
        with torch.no_grad():
            _g0, rep0 = self.pm0(f)
            emb = self.pm0.embed(f).numpy() if self.score else None
        ctx = torch.enable_grad() if self.score else torch.no_grad()
        with ctx:
            gates, _logit = self.pm.heads(f)
            tl = tempered_logits(gates)
        p = torch.sigmoid(tl).detach().numpy()
        bits = self.rng.random(3) < p
        greedy = gates.detach().numpy() > 0
        for i, k in enumerate(GATES):
            self.flips[k][0] += int(bits[i] != greedy[i])
            self.flips[k][1] += 1
        if self.score:
            lp = torch.distributions.Bernoulli(logits=tl).log_prob(
                torch.from_numpy(bits.astype(np.float32))).sum()
            self._store(env, "producer", pid, emb, self._project("producer", lp))
        sp = env.spec_
        act = np.zeros(sp.producing_action_dim, dtype=np.float32)
        for gp, frac in zip(f["inputs"], rep0.numpy()):
            if gp in sp.input_index:
                act[sp.input_index[gp]] = float(frac) * ACTION_LOGIT_BOUND
        act[sp.num_inputs:sp.num_inputs + 3] = np.where(bits, ACTION_LOGIT_BOUND,
                                                         -ACTION_LOGIT_BOUND)
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
        ctx = torch.enable_grad() if self.score else torch.no_grad()
        with ctx:
            logits = self.om(f)
        col = {p: i for i, p in enumerate(env.spec_.producer_ids)}
        picks = []
        for (k, _oid, q, el), lg, cols in zip(slots, logits, f["elig"]):
            if el and len(cols):
                w = torch.softmax(lg.detach(), 0).numpy().astype(np.float64)
                pick = inverse_cdf(w, keyed_uniform(key, agent, cycle, k))
                mat[k, el] = _whole(el, el.index(col[f["pids"][cols[pick]]]), q)
                picks.append(pick)
            else:
                picks.append(-1)
        if self.score:
            lp = torch.zeros(())
            for lg, kk in zip(logits, picks):
                if kk >= 0 and lg.shape[0] > 1:
                    lp = lp + torch.log_softmax(lg, 0)[kk]
            if lp.requires_grad:
                with torch.no_grad():
                    emb = self.om0.embed(f).numpy()
                self._store(env, "orderer", aid, emb, self._project("orderer", lp))
        return _flat(mat)

    def _store(self, env: Any, role: str, agent: Any, emb: np.ndarray, sc: np.ndarray) -> None:
        r = self.rec[role]
        r["cycle"].append(int(env.current_cycle))
        r["agent"].append(str(agent))
        r["emb"].append(emb.astype(np.float32))
        r["score"].append(sc)


def _models(path: str) -> tuple:
    pm, om = im.load_models(path)
    pm0, om0 = im.load_models(path)
    for m in (pm, om, pm0, om0):
        m.eval()
    return pm, om, pm0, om0


def _env_tracker(job: dict) -> tuple[Any, PotentialTracker]:
    env = im._episode_env(job)
    tr = PotentialTracker(PotentialWeights(**job["weights"]))
    tr.attach(env)
    return env, tr


def rollout_job(job: dict) -> dict:
    """One sibling episode with scoring; the per-decision arrays go to an npz on the volume."""
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    pm, om, pm0, om0 = _models(job["models"])
    env, tr = _env_tracker(job)
    rng = np.random.default_rng([s0.PROJ_SEED, 9, int(job["episode"]) % (1 << 31),
                                 int(job["sibling"])])
    pol = ExplorePolicies(pm, pm0, om, om0, rng, score=True)
    t0 = time.perf_counter()
    run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None)
    secs = time.perf_counter() - t0
    path = Path(job["npz_dir"]) / f"{job['size']}_{job['world']}_{job['episode']}_{job['sibling']}.npz"
    arrays = {}
    for role, r in pol.rec.items():
        arrays[f"{role}_cycle"] = np.asarray(r["cycle"], np.int32)
        arrays[f"{role}_agent"] = np.asarray(r["agent"])
        arrays[f"{role}_emb"] = (np.stack(r["emb"]) if r["emb"] else
                                 np.zeros((0, 1), np.float32))
        arrays[f"{role}_score"] = (np.stack(r["score"]) if r["score"] else
                                   np.zeros((0, s0.PROJ_K), np.float32))
    np.savez(path, **arrays)
    return {"seed": f"{job['world']}|{job['episode']}", "size": job["size"],
            "sibling": job["sibling"], "rewards": tr.rewards(float(job["unit_scale"])),
            "units": tr.totals()["units"], "dead": bool(env.episode_stats["dead"]),
            "seconds": secs, "npz": str(path), "horizon": job["horizon"],
            "flips": pol.flips, "decisions": {r: len(v["cycle"]) for r, v in pol.rec.items()}}


def perturb_job(job: dict) -> dict:
    """Sibling 0's draws, the actor moved by ±ε u_k on one role; no scoring."""
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    pm, om, pm0, om0 = _models(job["models"])
    pmp, omp = pw.perturbed((pm, om), job["role"], job["k"], job["sign"] * job["eps"])
    env, tr = _env_tracker(job)
    rng = np.random.default_rng([s0.PROJ_SEED, 9, int(job["episode"]) % (1 << 31), 0])
    pol = ExplorePolicies(pmp, pm0, omp, om0, rng, score=False)
    run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None)
    return {**{k: job[k] for k in ("size", "role", "k", "sign", "world", "episode", "eps")},
            "ret": float(sum(tr.rewards(float(job["unit_scale"])))), "type": "perturb"}


def tempered_kl(base: tuple, role: str, k: int, eps: float, prod: list, orde: list) -> float:
    """Mean per-decision KL(π_θ ‖ π_θ+εu) under v9's policy: tempered gates (producer; replenish is
    frozen, so it carries no KL); categorical slots (orderer)."""
    if role == "orderer":
        return pw.mean_kl(base, role, k, eps, prod, orde)
    pm0, _om0 = base
    pm1, _om1 = pw.perturbed(base, role, k, eps)
    ls = torch.nn.functional.logsigmoid
    kls = []
    with torch.no_grad():
        for f, _y in prod:
            f4 = g.v3_to_v4(f) if f["capacity"].shape[1] == 17 else f
            a = tempered_logits(pm0.heads(f4)[0])
            b = tempered_logits(pm1.heads(f4)[0])
            pa = torch.sigmoid(a)
            kls.append(float((pa * (ls(a) - ls(b)) + (1 - pa) * (ls(-a) - ls(-b))).sum()))
    return fmean(kls)


def eps_job(job: dict) -> dict:
    torch.set_num_threads(1)
    base = im.load_models(job["models"])
    lo, hi = 1e-5, 1e4
    for _ in range(26):
        mid = math.sqrt(lo * hi)
        if tempered_kl(base, job["role"], job["k"], mid, job["prod"], job["orde"]) > pw.KL_TARGET:
            hi = mid
        else:
            lo = mid
    e = math.sqrt(lo * hi)
    return {"role": job["role"], "k": job["k"], "eps": e, "at_bound": hi >= 1e4,
            "kl": tempered_kl(base, job["role"], job["k"], e, job["prod"], job["orde"])}


def any_job(job: dict) -> dict:
    return rollout_job(job) if job["type"] == "rollout" else perturb_job(job)


# --- analysis ---


def _load(meta: Sequence[dict]) -> dict[str, dict]:
    """Per row: the per-decision arrays and the per-cycle aggregated scores per role."""
    out = {}
    for m in meta:
        z = np.load(m["npz"])
        t_n = len(m["rewards"])
        row = {**m, "dec": {}, "scores": {}}
        for role in ("producer", "orderer"):
            cyc, sc = z[f"{role}_cycle"], z[f"{role}_score"]
            agg = np.zeros((t_n, s0.PROJ_K), np.float32)
            ok = cyc < t_n
            np.add.at(agg, cyc[ok], sc[ok])
            row["scores"][role] = agg
            row["dec"][role] = {"cycle": cyc, "emb": z[f"{role}_emb"], "score": sc}
        out[f"{m['seed']}#{m['sibling']}"] = row
    return out


def _features(d: Mapping, horizon: int) -> np.ndarray:
    return np.concatenate([d["emb"], (d["cycle"][:, None] / horizon).astype(np.float32),
                           np.ones((len(d["cycle"]), 1), np.float32)], 1)


def _targets(row: Mapping, d: Mapping, gamma: float) -> np.ndarray:
    rtg = s0._returns_to_go(row["rewards"], gamma)
    c = np.minimum(d["cycle"], len(rtg) - 1)
    return rtg[c]


def ridge_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = x.T @ x + RIDGE_ALPHA * np.eye(x.shape[1])
    return np.linalg.solve(a, x.T @ y)


def e4_parts(seeds: Mapping[str, list[dict]], role: str, gamma: float, horizon: int
             ) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict]:
    """Cross-fitted E4: per seed and sibling, the decision advantages and scores; V is fitted on the
    other half's decisions. Also V's held-out explained variance per half."""
    keys = sorted(seeds)
    halves = (keys[0::2], keys[1::2])
    adv: dict[str, list[np.ndarray]] = {}
    scr: dict[str, list[np.ndarray]] = {}
    ev = {}
    for hi, (fit_keys, use_keys) in enumerate(((halves[1], halves[0]), (halves[0], halves[1]))):
        xs, ys = [], []
        for k in fit_keys:
            for r in seeds[k]:
                d = r["dec"][role]
                if len(d["cycle"]):
                    xs.append(_features(d, horizon))
                    ys.append(_targets(r, d, gamma))
        if not xs:
            continue
        w = ridge_fit(np.concatenate(xs), np.concatenate(ys))
        yh, yt = [], []
        for k in use_keys:
            adv[k], scr[k] = [], []
            for r in seeds[k]:
                d = r["dec"][role]
                if not len(d["cycle"]):
                    adv[k].append(np.zeros(0))
                    scr[k].append(np.zeros((0, s0.PROJ_K)))
                    continue
                v = _features(d, horizon) @ w
                tgt = _targets(r, d, gamma)
                adv[k].append(tgt - v)
                scr[k].append(d["score"])
                yh.append(v)
                yt.append(tgt)
        if yt:
            y, p = np.concatenate(yt), np.concatenate(yh)
            ev[f"half{hi}"] = float(1 - np.var(y - p) / np.var(y)) if np.var(y) > 0 else None
    return adv, scr, ev


def _g4(adv: Mapping[str, list[np.ndarray]], scr: Mapping[str, list[np.ndarray]],
        keys: Sequence[str], assign: Mapping[str, str] | None = None) -> np.ndarray:
    out = np.zeros(s0.PROJ_K)
    for k in keys:
        src = assign[k] if assign else k
        for a, s in zip(adv[src], scr[k]):
            n = min(len(a), len(s))
            if n:
                out += a[:n] @ s[:n]
    return out


def e4_split(seeds: Mapping[str, list[dict]], role: str, gamma: float, horizon: int,
             rng: np.random.Generator) -> dict:
    adv, scr, ev = e4_parts(seeds, role, gamma, horizon)
    keys = sorted(adv)
    h1, h2 = [k for k in sorted(seeds)[0::2] if k in adv], [k for k in sorted(seeds)[1::2] if k in adv]
    c = s0._cos(_g4(adv, scr, h1), _g4(adv, scr, h2))
    null = []
    for _ in range(s0.NULL_DRAWS):
        perm = dict(zip(keys, rng.permutation(keys)))
        x = s0._cos(_g4(adv, scr, h1, perm), _g4(adv, scr, h2, perm))
        if x is not None:
            null.append(x)
    q95 = float(np.quantile(null, 0.95)) if null else None
    per_seed = np.stack([_g4(adv, scr, [k]) / s0.G_SIBLINGS for k in keys]) if keys else None
    return {"cosine": c, "null_q95": q95, "passes": c is not None and q95 is not None and c > q95,
            "explained_variance": ev}, per_seed


def analyse(meta: Sequence[dict], prows: Sequence[dict], n_params: Mapping[str, int]) -> dict:
    rng = np.random.default_rng(20261112)
    rows = _load(meta)
    per: dict[str, Any] = {}
    for size in (12, 42):
        rs = [r for r in rows.values() if r["size"] == size]
        if not rs:
            continue
        seeds = s0._by_seed(rs)
        d2 = s0.drift_horizon(rs)
        gam = s0.gamma_from(d2["t_star"])
        h = int(rs[0]["horizon"])
        d3 = s0.split_half(rs, gam)
        xs = {}
        for est in ("E1", "E2", "E3"):
            pe_ = pw.per_seed_estimates(seeds, est, gam)
            for role in ("producer", "orderer"):
                xs[(est, role)] = pe_[role]
        for role in ("producer", "orderer"):
            res, x4 = e4_split(seeds, role, gam, h, rng)
            d3.setdefault("E4", {})[role] = res
            xs[("E4", role)] = x4
        flips = {k: sum(r["flips"][k][0] for r in rs) / max(1, sum(r["flips"][k][1] for r in rs))
                 for k in GATES}
        share = {}
        for role in ("producer", "orderer"):
            share[role] = float(sum((r["dec"][role]["score"].astype(np.float64) ** 2).sum()
                                    for r in rs))
        per[str(size)] = {"seeds": len(seeds), "episodes": len(rs), "deaths": sum(r["dead"] for r in rs),
                          "D0_1_scenario_share": s0.scenario_share(rs), "D0_2": d2, "gamma": gam,
                          "D0_3": d3, "D0_5_seconds_per_episode": fmean(r["seconds"] for r in rs),
                          "flip_rate": flips, "projected_score_sq": share,
                          "units_mean": fmean(r["units"] for r in rs)}
        per[str(size)]["_xs"] = xs
    grad = pw.gradient_size(prows, n_params, rng)
    need: dict[str, Any] = {}
    for size in (12, 42):
        if str(size) not in per:
            continue
        for role in ("producer", "orderer"):
            gs = grad.get(f"{size}/{role}")
            if gs is None:
                continue
            res = {}
            for est in ESTIMATORS4:
                x = per[str(size)]["_xs"].get((est, role))
                if x is None or len(x) < 2:
                    res[est] = {"n80": None}
                    continue
                n, powers = pw.n80(x, gs["grad_norm"], rng)
                res[est] = {"n80": n, "power": powers, "noise_rms": pw.noise_rms(x)}
            need[f"{size}/{role}"] = res
    for v in per.values():
        v.pop("_xs", None)
    d12 = per.get("12", {}).get("D0_3", {})
    winners = [e for e in ESTIMATORS4 if d12.get(e, {}).get("producer", {}).get("passes")
               and (need.get("12/producer", {}).get(e, {}).get("n80") or 10 ** 9) <= N80_MAX]
    return {"rule": V9_RULE, "reading": "signal" if winners else "absent", "estimators": winners,
            "per_size": per, "gradient": grad, "seeds_needed": need, "temps": TEMPS}


# --- pre-registration v9 ---


def registered_prereg_v9() -> dict:
    return {
        "format": "m6-prereg-v9", "change": "add-m6-joint-mpg-gnn",
        "stage": "EXP-0112: Stage 0 + power check with reshaped exploration",
        "status": "frozen before EXP-0112's data; the real run refuses any other file",
        "keeps": {"m6_prereg_v8.json": s0.PREREG_V8_SHA256, "note": "v1..v8 stay frozen"},
        "decision": "user 2026-11-06 (via the supervisor), after EXP-0111 absent: reshape exploration "
                    "and re-test",
        "start": {"models_sha256": im.EXP0100_MODELS_SHA, "extension": "as v8 (zero-weight v4 inputs)"},
        "policy": {"replenish": "frozen at the start model's deterministic output; no noise; outside "
                                "the RL gradient",
                   "gates": {"distribution": "Bernoulli(sigmoid(logit / T_g))", "temps": TEMPS,
                             "rule": "T_g set offline so the greedy flip rate (sampled != logit > 0) "
                                     "is 5% per gate on EXP-0105's 12-producer monitor samples (the "
                                     "start policy); the deployed greedy policy is unchanged",
                             "reported": "flip rate per gate and size in the rollouts"},
                   "orderer": "as v8 (keyed softmax draw, temperature 1)"},
        "worlds_seeds_calibration": "as v8: the same 24 + 8 Stage 0 seeds, G = 4 siblings, EXP-0110's "
                                    "calibration (lambda, mu, nu, unit scales) reused",
        "estimators": {
            "E1_E3": "as v8 (per-cycle aggregated scores)",
            "E4": {"advantage": "G_t (team return-to-go, gamma from D0.2) - V(x_d)",
                   "x": "the frozen start actor's pooled embedding of the agent's own view (producer "
                        "4 x hidden, orderer 3 x hidden), t / H, 1",
                   "V": f"ridge, alpha {RIDGE_ALPHA}, per role and size, separate from the actor",
                   "cross_fit": "V fitted on one split half advantages the other",
                   "reported": "held-out explained variance",
                   "null": "decision advantages moved between seeds (aligned by sibling and decision "
                           "index), 200 draws, 95th percentile"}},
        "power": "as EXP-0111 (notes/exp0111-prereg.md) under this policy: 12 24 seeds x 4 directions, 42 "
                 "4 x 3; eps per direction at mean per-decision KL 0.01 (tempered gates; categorical "
                 "slots); n80 per estimator at the measured |grad J|",
        "rule": V9_RULE}


def freeze(path: Path = PREREG_V9_PATH) -> tuple[str, str]:
    from ..eval.provenance import sha256_file

    doc = registered_prereg_v9()
    if not path.exists():
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return "FROZEN", sha256_file(str(path))
    return ("MATCH" if json.loads(path.read_text()) == doc else "MISMATCH"), sha256_file(str(path))


def prereg_problems() -> list[str]:
    from ..eval.provenance import sha256_file

    if PREREG_V9_SHA256 is None:
        return ["no pinned m6_prereg_v9.json (`--mode freeze`)"]
    if not PREREG_V9_PATH.exists() or sha256_file(str(PREREG_V9_PATH)) != PREREG_V9_SHA256:
        return ["m6_prereg_v9.json does not match its pinned SHA-256"]
    if json.loads(PREREG_V9_PATH.read_text()) != registered_prereg_v9():
        return ["the pre-registration differs from the code"]
    return []


# --- the run ---


def run(training_dir: str, atlas_path: str, exp0110: str, monitor_pkl: str, out_dir: str,
        workers: int, dry_run: bool = False) -> int:
    import pickle

    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas

    atlas = load_atlas(atlas_path)
    e10 = Path(exp0110)
    models = str(e10 / "start_models.pt")
    if not dry_run:
        im._refuse(prereg_problems()
                   + ([] if atlas.get("content_hash") == im.REGISTERED_ATLAS_HASH else
                      ["not the registered atlas v2"]))
    out = Path(out_dir)
    (out / "npz").mkdir(parents=True, exist_ok=True)
    cal = json.loads((e10 / "calibration.json").read_text())
    _ts, j12, j42 = s0.world_episodes(training_dir, atlas)
    seeds = s0.stage0_seeds(j12, j42)
    kdirs = dict(pw.K_DIRS)
    if dry_run:  # a true smoke: 2 seeds at 12 and 1 at 42, horizon 4, one direction
        seeds = [{**j, "horizon": 4} for j in ([x for x in seeds if x["size"] == 12][:2]
                                               + [x for x in seeds if x["size"] == 42][:1])]
        kdirs = {12: 1, 42: 1}
    with open(monitor_pkl, "rb") as fh:
        mo = pickle.load(fh)
    prod, orde = mo["producer"][::32][:60 if dry_run else 600], mo["orderer"][::80][:60 if dry_run else 600]
    del mo
    ejobs = [{"models": models, "role": r, "k": k, "prod": prod, "orde": orde}
             for r in ("producer", "orderer") for k in range(max(kdirs.values()))]
    eps: dict[str, dict] = {"producer": {}, "orderer": {}}
    for e in pw._pool(eps_job, ejobs, workers, out / "eps_rows.jsonl"):
        eps[e["role"]][str(e["k"])] = e
    print(f"[m6_stage0b] eps {json.dumps({r: {k: round(v['eps'], 4) for k, v in d.items()} for r, d in eps.items()})}",
          flush=True)
    jobs = []
    for j in seeds:
        c = cal["per_size"][str(j["size"])]
        w = {"lam": c["lambda"], "mu": c["mu"], "nu": float(j["horizon"]), "x": cal["x"], "y": cal["y"]}
        scale = j["horizon"] / cal["bar"][f"{j['world']}|{j['episode']}"]
        base = {**j, "replicate": s0.STAGE0_REPLICATE_BASE, "models": models, "weights": w,
                "unit_scale": scale}
        for i in range(s0.G_SIBLINGS):
            jobs.append({**base, "replicate": s0.STAGE0_REPLICATE_BASE + i, "sibling": i,
                         "type": "rollout", "npz_dir": str(out / "npz")})
        perturb_seeds = j["size"] == 12 or j in seeds[-pw.SEEDS_42:] or dry_run
        if perturb_seeds:
            for role in ("producer", "orderer"):
                for k in range(kdirs[j["size"]]):
                    for sign in (1, -1):
                        jobs.append({**base, "type": "perturb", "role": role, "k": k,
                                     "sign": sign, "eps": eps[role][str(k)]["eps"]})
    jobs.sort(key=lambda j: (-j["size"], j["type"] != "rollout"))
    rows = pw._pool(any_job, jobs, workers, out / "episode_rows.jsonl")
    meta = [r for r in rows if r.get("type") != "perturb"]
    prows = [r for r in rows if r.get("type") == "perturb"]
    pm, om = im.load_models(models)
    n_params = {"producer": sum(p.numel() for p in pm.parameters()),
                "orderer": sum(p.numel() for p in om.parameters())}
    doc = {**analyse(meta, prows, n_params), "eps": eps, "dry_run": dry_run,
           "prereg": PREREG_V9_SHA256, "source_commit": im.SOURCE_COMMIT,
           "start_models_sha256": sha256_file(models), "jobs": len(jobs), "rows": len(rows)}
    (out / "reading.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[m6_stage0b] reading {doc['reading']} estimators {doc['estimators']}", flush=True)
    return 0 if len(rows) == len(jobs) else 2


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=("freeze", "run"), default="run")
    p.add_argument("--training-dir")
    p.add_argument("--atlas")
    p.add_argument("--exp0110", help="EXP-0110's artifacts (start_models.pt, calibration.json)")
    p.add_argument("--monitor-samples")
    p.add_argument("--out-dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source-commit", default="")
    a = p.parse_args(argv)
    im.SOURCE_COMMIT = a.source_commit
    if a.mode == "freeze":
        status, sha = freeze()
        print(f"[m6_stage0b] pre-registration v9 {status} {sha}")
        return 0 if status in ("FROZEN", "MATCH") else 2
    if not a.dry_run and len(a.source_commit) != 40:
        p.error("a real run needs a 40-hex --source-commit")
    torch.set_num_threads(1)
    return run(a.training_dir, a.atlas, a.exp0110, a.monitor_samples, a.out_dir, a.workers,
               a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
