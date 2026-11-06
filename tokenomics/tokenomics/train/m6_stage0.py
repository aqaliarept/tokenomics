"""M6 critic-free RL, Stage 0: the diagnostic before any training (pre-registration v8).

`python -m tokenomics.train.m6_stage0 --mode {freeze,start,run} ...`

Supervisor decision 2026-11-06 (after EXP-0105 … EXP-0109). The design is docs/research/critic-free-team-learning.md,
"Recommendation", with D0.4 dropped.
- **Start policy:** the EXP-0100 producer and orderer. The producer's capacity encoder gains the v4 inputs (the
  ceiling margin and its sign) as two columns with zero weights (`extend_to_v4`), so the start policy is
  bit-identical to EXP-0100 (tested) while RL can learn to use the ceiling.
- **Worlds:** the training split only. At 12 producers, M4's training world-episodes (monitor excluded), horizon
  100. At 42, the atlas train-split worlds (`m6_imitation.COLLECT_42_WORLDS`; never held-out, so never
  42-0125), horizon 200.
- **Reward:** the normalised team potential (`env/potential.py`). s = H / B_we per world-episode; ν = H; λ and μ
  per size by M5's noise-scaled rule, all from `calibrate` on k5_rf5 + MWWk re-keys r1..r4 of the same
  world-episodes (training data only).
- **Rollouts:** G = 4 siblings per world-seed, differing only in their policy draws (the orderer's DrawKey
  replicate and the producer's sampling stream). For every cycle and role, the run records the potential reward
  and the policy-gradient score Σ ∇ log π of the role's decisions in that cycle, projected onto a fixed random
  k = 256 subspace (Johnson–Lindenstrauss: cosines kept to about ±0.06), so every estimator below is a linear
  combination formed afterwards.
- **Diagnostics:**
  - **D0.1, scenario share:** 1 − within-seed variance / total variance of the episode return, per size.
  - **D0.2, decorrelation horizon:** per cycle t, the median over sibling pairs of the across-seed correlation
    of Φ_t; t* = the first t where it drops below 0.5. It sets γ by 1 / (1 − γ) ≈ t*, clipped to [0.9, 0.99].
  - **D0.3, gradient reliability:** the split-half cosine (seeds split in two) of the policy gradient per role
    and size, for E1 (unpaired batch-mean baseline, episode return), E2 (same-seed leave-one-out, episode
    return) and E3 (same-seed leave-one-out, per-cycle return-to-go at D0.2's γ). Each is compared with a
    permutation null (advantages shuffled across seeds, 200 draws, 95th percentile).
  - **D0.5, cost:** wall seconds per 42-producer GNN episode at horizon 200.
- **Reading (STAGE0_RULE):** Stage 1 uses the estimator with the highest D0.3 cosine at 12, and runs only if
  that cosine exceeds its null's 95th percentile at 12 for both roles. If it fails at both sizes: `no_signal`.
  If it fails at 12 only: `signal_42_only` (to the supervisor). D0.1, D0.2 and D0.5 are reported; they set γ and
  the budget.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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
from . import producer_es as pe

PREREG_V8_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v8.json"
PREREG_V8_SHA256: str | None = "d55e757ce6d2f900bb1b847b035915bdc461fb843d9b1783ffccbb17d528ade6"  # frozen before Stage 0's data
G_SIBLINGS = 4
PROJ_K = 256
PROJ_SEED = 20261106
NULL_DRAWS = 200
SEEDS_12 = 24  # every k-th of the 112 training world-episodes
SEEDS_42 = 8  # the first episode of every k-th train-split 42 world
STAGE0_REPLICATE_BASE = 100  # sibling i plays DrawKey replicate 100 + i (calibration: 1..4; monitor: 0, 1)
CAL_42_EPISODES = 2  # the collect stage's two episodes per 42 world (stage m6-collect-42)
CAL_REPLICATES = (1, 2, 3, 4)
INIT_LOG_STD = -1.5
STAGE0_RULE = (
    "D0.3: per role (producer, orderer) and size, the split-half cosine of the policy gradient for E1 "
    "(unpaired batch-mean baseline, episode return), E2 (same-seed leave-one-out, episode return), E3 "
    "(same-seed leave-one-out, per-cycle return-to-go at gamma from D0.2), each against a permutation null "
    "(advantages shuffled across seeds, 200 draws, 95th percentile). Stage 1 uses the estimator with the "
    "highest mean cosine over the two roles at 12, and runs only if that estimator's cosine exceeds its null's "
    "95th percentile at 12 for both roles (reading `signal`). Else `signal_42_only` if it does at 42 for both "
    "roles, else `no_signal` (stop, record). D0.1, D0.2 (gamma = 1 - 1/t*, clipped to [0.9, 0.99]) and D0.5 are "
    "reported and set gamma and the Stage 1 budget")


# --- the start policy: EXP-0100 + the v4 inputs at zero weight ---


def extend_to_v4(pm: g.ProducerGNN) -> g.ProducerGNN:
    """A v4-feature producer whose margin and sign columns enter through their own zero-weight
    layer (`ProducerGNN(capacity_extra=2)`; the normaliser passes them at centre 0, scale 1): its
    outputs equal `pm`'s, bit for bit."""
    if pm.features != g.FEATURES_V2:
        raise ValueError("extend_to_v4 takes a v2 (EXP-0100) producer")
    n = g.CAPACITY_DIM[g.FEATURES_V2]
    new = g.ProducerGNN(pm.hidden, pm.rounds, g.FEATURES_V4,
                        capacity_extra=g.CAPACITY_DIM[g.FEATURES_V4] - n)
    sd = new.state_dict()
    old = pm.state_dict()
    sd["cap_extra.weight"] = torch.zeros_like(sd["cap_extra.weight"])
    for k, v in old.items():
        if k in ("norm.c_capacity", "norm.s_capacity"):
            t = torch.zeros_like(sd[k]) if k.endswith("c_capacity") else torch.ones_like(sd[k])
            t[:n] = v
            sd[k] = t
        else:
            sd[k] = v
    new.load_state_dict(sd)
    new.norm.clip = pm.norm.clip
    return new


def make_start(prev: str, out: Path) -> str:
    pm, om = im.load_models(prev)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save_models(extend_to_v4(pm), om, out)
    return str(out)


# --- worlds and seeds (training split only) ---


def world_episodes(training_dir: str, atlas: Mapping) -> tuple[Any, list[dict], list[dict]]:
    """(training set, the 12 calibration world-episode jobs, the 42 ones), replicate unset."""
    from ..eval.teacher_audit import world_view
    from .producer_read import entry_for

    ts, train, _mon = im._training(training_dir)
    j12 = [{**j, "size": 12} for j in im._jobs(ts, train, "stage0", (0,), pe.HORIZON)]
    views = {world_view(e)["world"]: world_view(e) for e in atlas["worlds"]}
    j42 = []
    for w in im.COLLECT_42_WORLDS:
        if views[w]["split"] != "train":
            raise ValueError(f"{w} is not a train-split world")
        e = entry_for(views[w], CAL_42_EPISODES, im.COLLECT_42["stage"])
        for ep in e.episode_seeds:
            j42.append({"world": e.id, "size": e.size, "regime": e.regime,
                        "world_seed": e.world_seed, "generator": e.generator, "episode": int(ep),
                        "replicate": 0, "horizon": im.COLLECT_42["horizon"], "kind": "stage0"})
    return ts, j12, j42


def stage0_seeds(j12: list[dict], j42: list[dict]) -> list[dict]:
    """SEEDS_12 evenly spaced 12-producer world-episodes; SEEDS_42 = the first episode of every
    k-th 42-producer world."""
    k12 = max(1, len(j12) // SEEDS_12)
    first42 = [j for i, j in enumerate(j42) if i % CAL_42_EPISODES == 0]
    k42 = max(1, len(first42) // SEEDS_42)
    return j12[::k12][:SEEDS_12] + first42[::k42][:SEEDS_42]


# --- calibration (training data only; per size) ---


def calibrate(rows: Sequence[Mapping]) -> dict:
    """Per size: B_we = the mean episode units per world-episode (s = H / B_we); u = units · s;
    λ = σ_u / σ_rundown and μ = σ_u / σ_starving, pooled within-world-episode SDs over r1..r4; ν = H."""
    from .order_es import _pooled_sd

    out: dict[str, Any] = {"bar": {}, "per_size": {}}
    for size in sorted({int(r["size"]) for r in rows}):
        rs = [r for r in rows if int(r["size"]) == size]
        h = int(rs[0]["horizon"])
        groups: dict[tuple, dict[str, list[float]]] = {}
        for r in rs:
            k = (r["world"], int(r["episode"]))
            for t in ("units", "rundown", "starving"):
                groups.setdefault(k, {}).setdefault(t, []).append(float(r[t]))
        bar = {k: fmean(v["units"]) for k, v in groups.items()}
        su, df = _pooled_sd([[u * h / bar[k] for u in v["units"]] for k, v in groups.items()])
        sr, _ = _pooled_sd([v["rundown"] for v in groups.values()])
        ss, _ = _pooled_sd([v["starving"] for v in groups.values()])
        out["per_size"][str(size)] = {
            "lambda": su / sr if sr > 0 else 0.0, "mu": su / ss if ss > 0 else 0.0,
            "nu": float(h), "horizon": h, "sigma": {"units_normalised": su, "rundown": sr,
                                                    "starving": ss},
            "df": df, "world_episodes": len(groups), "episodes": len(rs)}
        out["bar"].update({f"{w}|{e}": b for (w, e), b in bar.items()})
    return out


def calibration_job(job: dict) -> dict:
    row = mp.calibration_job(job)
    return {**row, "size": job["size"], "horizon": job["horizon"]}


# --- Stage 0 rollouts: rewards and projected per-cycle scores ---

_PROJ: dict[str, torch.Tensor] = {}


def projection(role: str, n: int) -> torch.Tensor:
    """The fixed k x n random projection of one role's parameter vector (same in every process)."""
    key = f"{role}:{n}"
    if key not in _PROJ:
        gen = torch.Generator().manual_seed(PROJ_SEED + (0 if role == "producer" else 1))
        _PROJ[key] = torch.randn(PROJ_K, n, generator=gen) / math.sqrt(PROJ_K)
    return _PROJ[key]


class ScorePolicies(mp.RecordingPolicies):
    """The RL policies' sampling (`m6_mappo.RecordingPolicies`), plus, per cycle and role, the
    projected score Σ ∇ log π(sampled action) of that cycle's decisions."""

    def __init__(self, pm: g.ProducerGNN, om: g.OrdererGNN, ex: mp.ActorExtras,
                 rng: np.random.Generator) -> None:
        super().__init__(pm, om, ex, rng)
        self.params = {"producer": [*pm.parameters(), *ex.parameters()],
                       "orderer": list(om.parameters())}
        self.acc: dict[str, Any] = {"producer": None, "orderer": None}
        self.cycle: int | None = None
        self.scores: dict[int, dict[str, np.ndarray]] = {}

    def _flush(self) -> None:
        if self.cycle is None:
            return
        out = {}
        for role, ps in self.params.items():
            n = sum(p.numel() for p in ps)
            a = self.acc[role]
            if a is None or not a.requires_grad:
                out[role] = np.zeros(PROJ_K, dtype=np.float32)
                continue
            gr = torch.autograd.grad(a, ps, allow_unused=True)
            flat = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1)
                              for x, p in zip(gr, ps)])
            out[role] = (projection(role, n) @ flat).numpy().astype(np.float32)
        self.scores[self.cycle] = out
        self.acc = {"producer": None, "orderer": None}
        self.cycle_inputs = {}  # the critic's inputs are not used here: free them

    def _add(self, env: Any, role: str, lp: torch.Tensor) -> None:
        c = int(env.current_cycle)
        if c != self.cycle:
            self._flush()
            self.cycle = c
        self.acc[role] = lp if self.acc[role] is None else self.acc[role] + lp

    def producing(self, env: Any, agent: str) -> np.ndarray:
        act = super().producing(env, agent)
        d = self.decisions.pop()
        lp, _ = mp.producer_logp(self.pm, self.ex, d["f"], d["bits"], d["z"], d["mask"])
        self._add(env, "producer", lp)
        return act

    def ordering(self, env: Any, agent: str) -> np.ndarray:
        n = len(self.decisions)
        act = super().ordering(env, agent)
        if len(self.decisions) > n:
            d = self.decisions.pop()
            lp, _ = mp.orderer_logp(self.om, d["f"], d["picks"])
            self._add(env, "orderer", lp)
        return act

    def finish(self) -> None:
        self._flush()


def stage0_job(job: dict) -> dict:
    """One sibling episode: per-cycle normalised rewards and projected scores per role."""
    from ..eval.runner import run_episode

    torch.set_num_threads(1)
    pm, om = im.load_models(job["models"])
    ex = mp.ActorExtras(INIT_LOG_STD)
    for m in (pm, om):
        m.eval()
    env = im._episode_env(job)
    w = job["weights"]
    tracker = PotentialTracker(PotentialWeights(**w))
    tracker.attach(env)
    rng = np.random.default_rng([PROJ_SEED, int(job["episode"]) % (1 << 31), int(job["sibling"])])
    pol = ScorePolicies(pm, om, ex, rng)
    t0 = time.perf_counter()
    run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None)
    pol.finish()
    secs = time.perf_counter() - t0
    rewards = tracker.rewards(float(job["unit_scale"]))
    cycles = sorted(pol.scores)
    return {"seed": f"{job['world']}|{job['episode']}", "size": job["size"],
            "sibling": job["sibling"], "rewards": rewards, "units": tracker.totals()["units"],
            "dead": bool(env.episode_stats["dead"]), "seconds": secs,
            "scores": {r: np.stack([pol.scores[c][r] for c in cycles]).tolist()
                       if cycles else [] for r in ("producer", "orderer")},
            "score_cycles": len(cycles)}


# --- the diagnostics (pure, on the rows) ---


def _returns_to_go(r: Sequence[float], gamma: float) -> np.ndarray:
    out, acc = np.zeros(len(r)), 0.0
    for t in reversed(range(len(r))):
        acc = r[t] + gamma * acc
        out[t] = acc
    return out


def _by_seed(rows: Sequence[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["seed"], []).append(r)
    return {k: sorted(v, key=lambda r: r["sibling"]) for k, v in out.items()
            if len(v) == G_SIBLINGS}


def scenario_share(rows: Sequence[dict]) -> float | None:
    """D0.1: 1 − within-seed variance / total variance of the episode return."""
    seeds = _by_seed(rows)
    allr = [sum(r["rewards"]) for v in seeds.values() for r in v]
    if len(allr) < 2:
        return None
    within = fmean(float(np.var([sum(r["rewards"]) for r in v], ddof=1)) for v in seeds.values())
    total = float(np.var(allr, ddof=1))
    return 1.0 - within / total if total > 0 else None


def drift_horizon(rows: Sequence[dict]) -> dict:
    """D0.2: per cycle t, the median over sibling pairs of the across-seed correlation of Φ_t."""
    seeds = list(_by_seed(rows).values())
    if len(seeds) < 3:
        return {"t_star": None, "median_corr": []}
    t_max = min(len(r["rewards"]) for v in seeds for r in v)
    med = []
    for t in range(t_max):
        cs = []
        for i in range(G_SIBLINGS):
            for j in range(i + 1, G_SIBLINGS):
                a = [v[i]["rewards"][t] for v in seeds]
                b = [v[j]["rewards"][t] for v in seeds]
                if np.std(a) > 0 and np.std(b) > 0:
                    cs.append(float(np.corrcoef(a, b)[0, 1]))
        med.append(float(np.median(cs)) if cs else None)
    t_star = next((t for t, m in enumerate(med) if m is not None and m < 0.5), None)
    return {"t_star": t_star, "median_corr": med}


def gamma_from(t_star: int | None) -> float:
    """1 / (1 − γ) ≈ t*, clipped to [0.9, 0.99]; no decorrelation within the horizon -> 0.99."""
    if t_star is None:
        return 0.99
    return float(min(0.99, max(0.9, 1.0 - 1.0 / max(t_star, 1))))


def advantages(seeds: Mapping[str, list[dict]], estimator: str, gamma: float
               ) -> dict[str, list[np.ndarray]]:
    """Per seed, per sibling, the per-cycle advantage vector A_t (E1 / E2 constant in t)."""
    ep = {k: [sum(r["rewards"]) for r in v] for k, v in seeds.items()}
    out = {}
    if estimator == "E1":
        mu = fmean(x for v in ep.values() for x in v)
        for k, v in seeds.items():
            out[k] = [np.full(len(r["rewards"]), ep[k][i] - mu) for i, r in enumerate(v)]
    elif estimator == "E2":
        for k, v in seeds.items():
            s = sum(ep[k])
            out[k] = [np.full(len(r["rewards"]), ep[k][i] - (s - ep[k][i]) / (G_SIBLINGS - 1))
                      for i, r in enumerate(v)]
    else:  # E3: per-cycle leave-one-out on the return-to-go
        for k, v in seeds.items():
            rtg = [_returns_to_go(r["rewards"], gamma) for r in v]
            n = min(len(x) for x in rtg)
            m = np.stack([x[:n] for x in rtg])
            loo = (m.sum(0, keepdims=True) - m) / (G_SIBLINGS - 1)
            out[k] = [np.concatenate([m[i] - loo[i], np.zeros(len(rtg[i]) - n)])
                      for i in range(G_SIBLINGS)]
    return out


def gradient(seeds: Mapping[str, list[dict]], adv: Mapping[str, list[np.ndarray]], role: str,
             keys: Sequence[str]) -> np.ndarray:
    """Σ over the seeds' siblings and cycles of A_t · (projected score of cycle t)."""
    gsum = np.zeros(PROJ_K)
    for k in keys:
        for r, a in zip(seeds[k], adv[k]):
            s = np.asarray(r["scores"][role])
            if s.size:
                n = min(len(s), len(a))
                gsum += (a[:n, None] * s[:n]).sum(0)
    return gsum


def _cos(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else None


def split_half(rows: Sequence[dict], gamma: float, seed: int = 0) -> dict:
    """D0.3 per estimator and role: the split-half cosine and its permutation null."""
    seeds = _by_seed(rows)
    keys = sorted(seeds)
    h1, h2 = keys[0::2], keys[1::2]
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for est in ("E1", "E2", "E3"):
        adv = advantages(seeds, est, gamma)
        out[est] = {}
        for role in ("producer", "orderer"):
            c = _cos(gradient(seeds, adv, role, h1), gradient(seeds, adv, role, h2))
            null = []
            for _ in range(NULL_DRAWS):
                perm = rng.permutation(keys)
                shuffled = {k: adv[p] for k, p in zip(keys, perm)}
                x = _cos(gradient(seeds, shuffled, role, h1), gradient(seeds, shuffled, role, h2))
                if x is not None:
                    null.append(x)
            q95 = float(np.quantile(null, 0.95)) if null else None
            out[est][role] = {"cosine": c, "null_q95": q95,
                              "passes": c is not None and q95 is not None and c > q95}
    return out


def stage0_reading(rows: Sequence[dict]) -> dict:
    per: dict[str, Any] = {}
    for size in (12, 42):
        rs = [r for r in rows if r["size"] == size]
        if not rs:
            continue
        d2 = drift_horizon(rs)
        gam = gamma_from(d2["t_star"])
        per[str(size)] = {"seeds": len(_by_seed(rs)), "episodes": len(rs),
                          "deaths": sum(r["dead"] for r in rs),
                          "D0_1_scenario_share": scenario_share(rs), "D0_2": d2, "gamma": gam,
                          "D0_3": split_half(rs, gam),
                          "D0_5_seconds_per_episode": fmean(r["seconds"] for r in rs)}

    def ok(size: str, est: str) -> bool:
        return size in per and all(per[size]["D0_3"][est][r]["passes"]
                                   for r in ("producer", "orderer"))

    best = None
    if "12" in per:
        best = max(("E1", "E2", "E3"), key=lambda e: fmean(
            (per["12"]["D0_3"][e][r]["cosine"] or -1.0) for r in ("producer", "orderer")))
    reading = ("signal" if best and ok("12", best) else
               "signal_42_only" if any(ok("42", e) for e in ("E1", "E2", "E3")) else "no_signal")
    return {"rule": STAGE0_RULE, "reading": reading, "estimator": best, "per_size": per}


# --- pre-registration v8 ---


def registered_prereg_v8() -> dict:
    return {
        "format": "m6-prereg-v8", "change": "add-m6-joint-mpg-gnn",
        "stage": "critic-free RL: Stage 0 (diagnostic) and the Stage 1 design",
        "status": "frozen before Stage 0's data; the real run refuses any other file",
        "keeps": {"m6_prereg_v7.json": im.PREREG_V7_SHA256, "note": "v1..v7 stay frozen"},
        "decisions": ["user 2026-11-06: the RL stage is critic-free (docs/research/"
                      "critic-free-team-learning.md)",
                      "supervisor 2026-11-06: stop tuning imitation; Stage 0 from the EXP-0100 stack "
                      "with the v4 inputs at zero weight; D0.4 dropped"],
        "start": {"models_sha256": im.EXP0100_MODELS_SHA,
                  "extension": "producer capacity encoder + 2 zero-weight input columns (margin, "
                               "sign); bit-identical outputs (tests/test_m6_stage0.py)",
                  "replenish_log_std": INIT_LOG_STD},
        "worlds": {"12": "M4's training world-episodes (monitor excluded), horizon 100",
                   "42": {"worlds": list(im.COLLECT_42_WORLDS), "episodes": CAL_42_EPISODES,
                          "stage": im.COLLECT_42["stage"], "horizon": im.COLLECT_42["horizon"]},
                   "excluded": "every held-out and validation world (42-0125 included)"},
        "calibration": {"policy": "k5_rf5 + MWWk", "replicates": list(CAL_REPLICATES),
                        "rule": "per size: s = H / B_we; lambda / mu = sigma(normalised units) / "
                                "sigma(term), pooled within world-episode; nu = H"},
        "stage0": {"siblings": G_SIBLINGS, "replicate_base": STAGE0_REPLICATE_BASE,
                   "seeds": {"12": SEEDS_12, "42": SEEDS_42},
                   "projection": {"k": PROJ_K, "seed": PROJ_SEED,
                                  "note": "Johnson-Lindenstrauss; cosines to about +-0.06"},
                   "null_draws": NULL_DRAWS, "rule": STAGE0_RULE},
        "stage1_design": {
            "learner": "critic-free: G = 4 same-seed siblings; Stage 0's estimator (leave-one-out "
                       "baseline, no std division); PPO-clip 0.2; KL penalty to the start policy, "
                       "coefficient 0.04; one Stepper per agent (Adam, clip 1.0) with per-step logs; "
                       "1 epoch per iteration",
            "data": "fresh training-split seeds every iteration; zero-spread groups dropped and "
                    "counted",
            "gate": "every 15 iterations: candidate vs the frozen EXP-0100 stack, one-sided paired "
                    "t-test alpha 0.05 on fresh train-split seeds, no extra deaths, run-down share no "
                    "worse",
            "pinned_before_stage1": "learning rates, iterations and seeds per iteration (sized from "
                                    "D0.5), in a further pre-registration"}}


def freeze(path: Path = PREREG_V8_PATH) -> tuple[str, str]:
    from ..eval.provenance import sha256_file

    doc = registered_prereg_v8()
    if not path.exists():
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return "FROZEN", sha256_file(str(path))
    return ("MATCH" if json.loads(path.read_text()) == doc else "MISMATCH"), sha256_file(str(path))


def prereg_problems() -> list[str]:
    from ..eval.provenance import sha256_file

    if PREREG_V8_SHA256 is None:
        return ["no pinned m6_prereg_v8.json (`--mode freeze`)"]
    if not PREREG_V8_PATH.exists() or sha256_file(str(PREREG_V8_PATH)) != PREREG_V8_SHA256:
        return ["m6_prereg_v8.json does not match its pinned SHA-256"]
    if json.loads(PREREG_V8_PATH.read_text()) != registered_prereg_v8():
        return ["the pre-registration differs from the code"]
    return []


# --- the run ---


def _stream(fn: Any, jobs: Sequence[dict], workers: int, path: Path) -> list[dict]:
    rows = []
    with path.open("w") as fh:
        if workers <= 1:
            for j in jobs:
                rows.append(fn(j))
                fh.write(json.dumps(rows[-1]) + "\n")
                fh.flush()
        else:
            with ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                                     initializer=pe._init_worker) as ex:
                for fut in as_completed([ex.submit(fn, j) for j in jobs]):
                    rows.append(fut.result())
                    fh.write(json.dumps(rows[-1]) + "\n")
                    fh.flush()
    return rows


def run(training_dir: str, atlas_path: str, prev: str, out_dir: str, workers: int,
        dry_run: bool = False, max_seeds: int | None = None, horizon: int | None = None) -> int:
    from ..eval.provenance import sha256_file
    from ..eval.teacher_audit import load_atlas
    from .order_es_worlds import X_RUNDOWN, Y_STARVATION

    atlas = load_atlas(atlas_path)
    if not dry_run:
        im._refuse(prereg_problems()
                   + ([] if sha256_file(prev) == im.EXP0100_MODELS_SHA else
                      ["--prev-models is not EXP-0100's"])
                   + ([] if atlas.get("content_hash") == im.REGISTERED_ATLAS_HASH else
                      ["not the registered atlas v2"])
                   + ([] if max_seeds is None and horizon is None else
                      ["--max-seeds / --horizon is set"]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = make_start(prev, out / "start_models.pt")
    ts, j12, j42 = world_episodes(training_dir, atlas)
    seeds = stage0_seeds(j12, j42)
    if max_seeds:
        seeds = [j for j in seeds if j["size"] == 12][:max_seeds] + \
            [j for j in seeds if j["size"] == 42][:max(1, max_seeds // 3)]
        keep = {(j["world"], j["episode"]) for j in seeds}
        j12 = [j for j in j12 if (j["world"], j["episode"]) in keep]
        j42 = [j for j in j42 if (j["world"], j["episode"]) in keep]
    if horizon:
        j12 = [{**j, "horizon": horizon} for j in j12]
        j42 = [{**j, "horizon": horizon} for j in j42]
        seeds = [{**j, "horizon": horizon} for j in seeds]
    # calibration: k5_rf5 + MWWk re-keys on the training world-episodes (both sizes)
    probe = asdict(PotentialWeights(0.0, 0.0, 0.0, float(X_RUNDOWN), float(Y_STARVATION)))
    cjobs = [{**j, "replicate": r, "weights": probe} for r in CAL_REPLICATES for j in j42 + j12]
    crows = _stream(calibration_job, cjobs, workers, out / "calibration_rows.jsonl")
    cal = calibrate(crows)
    cal.update({"x": float(X_RUNDOWN), "y": float(Y_STARVATION), "dry_run": dry_run})
    (out / "calibration.json").write_text(json.dumps(cal, indent=1) + "\n")
    print(f"[m6_stage0] calibration {json.dumps(cal['per_size'])}", flush=True)
    sjobs = []
    for j in seeds:
        c = cal["per_size"][str(j["size"])]
        w = asdict(PotentialWeights(c["lambda"], c["mu"], float(j["horizon"]), cal["x"], cal["y"]))
        scale = j["horizon"] / cal["bar"][f"{j['world']}|{j['episode']}"]
        for i in range(G_SIBLINGS):
            sjobs.append({**j, "replicate": STAGE0_REPLICATE_BASE + i, "sibling": i,
                          "models": start, "weights": w, "unit_scale": scale})
    sjobs.sort(key=lambda j: -j["size"])  # the 42 episodes are the longest: first
    rows = _stream(stage0_job, sjobs, workers, out / "stage0_rows.jsonl")
    doc = {**stage0_reading(rows), "dry_run": dry_run, "prereg": PREREG_V8_SHA256,
           "start_models_sha256": sha256_file(start), "source_commit": im.SOURCE_COMMIT,
           "jobs": len(sjobs), "rows": len(rows)}
    (out / "reading.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[m6_stage0] reading {doc['reading']} estimator {doc['estimator']}", flush=True)
    return 0 if len(rows) == len(sjobs) else 2


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=("freeze", "run"), default="run")
    p.add_argument("--training-dir")
    p.add_argument("--atlas")
    p.add_argument("--prev-models")
    p.add_argument("--out-dir")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-seeds", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source-commit", default="")
    a = p.parse_args(argv)
    im.SOURCE_COMMIT = a.source_commit
    if a.mode == "freeze":
        status, sha = freeze()
        print(f"[m6_stage0] pre-registration v8 {status} {sha}")
        return 0 if status in ("FROZEN", "MATCH") else 2
    if not a.dry_run and len(a.source_commit) != 40:
        p.error("a real run needs a 40-hex --source-commit")
    torch.set_num_threads(1)
    return run(a.training_dir, a.atlas, a.prev_models, a.out_dir, a.workers, a.dry_run,
               a.max_seeds, a.horizon)


if __name__ == "__main__":
    sys.exit(main())
