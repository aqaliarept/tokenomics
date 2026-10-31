"""Step-2 ES loop and gradient probe: `python -m tokenomics.train.order_es`
(add-m4-orderer-redesign design D6 / D7, task 5.2; learned-orderer "Outcome training by bounded
ES", "Relative disqualification").

`--mode train --arm A1|A2|A2-R2 --opt-seed s --generations G` trains the per-order residual (D5)
by antithetic OpenAI-ES:
- n pairs (12) of members theta +- sigma * eps_i, plus the center, every generation; sigma is
  the arm's per-parameter vector (`ArmSpec.sigma`), eps_i ~ N(0, 1) from the recorded noise
  stream `NOISE_STREAM` keyed by (opt seed, generation), so a resume redraws the same members.
- Common random numbers: every member of generation g plays the same minibatch (schedule.json's
  `minibatches[opt_seed][g - 1]`) under the same keys DrawKey(world, episode, 0); keys never
  carry the member.
- Fitness per world-episode U / B_we - h * H / D_we (h = 0.1), B_we and D_we from the bar table;
  a member's fitness is its minibatch mean. It only ranks members.
- Relative disqualification against A0 on the same world-episode and keys (the bar's A0 row):
  death A0 survives, run-down r_end < X r_start A0 does not show, or a consumer below Y that A0
  does not starve (per consumer). A member with a disqualifying world-episode, or one whose
  policy raised `NonFiniteChoice` (logged, never a contract failure), takes the worst rank.
- Centered ranks over the 2n members (average ranks on ties, the worst group shared), the
  estimate g_j = sum_i (r(+i) - r(-i)) eps_ij / (2 n sigma_j), and Adam (lr 0.02) ascending it,
  with weight decay 0.005 coupled into the gradient as OpenAI-ES's l2coeff.
- Every `MONITOR_EVERY` generations (and at generation 0) the center plays the 16 monitor
  world-episodes at replicates 0 and 1; A0 plays them once per run dir at both replicates (its
  replicate-0 units must reproduce the bar's A0 units, else the run is invalid).
- A checkpoint every generation (`checkpoints/gen-NNNN.json`: theta, Adam state, the training
  pins, the per-generation and monitor history, the composite), so a hard stop loses at most one
  generation; rows land in the CSVs as each job finishes, and a resume replays only missing jobs.
- `--resume-from <checkpoint>` continues another run dir's checkpoint (the main run continues the
  pilot) only when its training pins equal this run's.

`--mode probe --arms A1,A2` is D7's gradient-signal probe: each arm at its registered initial
theta (Delta = 0), n pairs, each member scored on the two disjoint probe minibatches; per arm the
cosine between the two minibatches' ES estimates and the paired (CRN) / unpaired variance ratio of
antithetic differences; the reading `signal` (A1 cosine >= 0.3), `weak` (>= 0.1, minibatch 24
next), `no_signal`; A2 cosine < 0.1 with A1 >= 0.1 -> A2 uses R = 2.

Jobs: one per member x world-episode, longest first (the bar's per-world A0 `cost_s`), process pool
with BLAS and torch at one thread. A real run refuses a pre-registration mismatch
(`es_prereg_m4_v1.json`, frozen by task 5.3), a training world other than 12 producers, any world
above 42, a dry-run bar, filters, another horizon than 100, or no 40-hex source commit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from ..env.spaces import CHOICE_V1, decode_pins
from ..eval.contract import (DIGEST, ContractMonitor, ContractRunError, append_failure,
                             ensure_failures_file, failure_from_dict, is_failure, job_failure,
                             read_failures, run_failure)
from ..eval.gate import DigestMismatch, row_from_stats, row_to_dict, CSV_COLUMNS
from ..eval.gate_cli import CONTRACT_INVALID, INVALID, Job, _write_spec
from ..eval.provenance import resolve_verified_commit, sha256_file, source_fingerprint
from ..eval.registry import WorldEntry
from ..eval.routing_audit import (PRODUCER, TRACE_PREFIX, TRACE_SCALARS, _init_worker, _is_commit,
                                  job_config)
from ..eval.routing_diagnostics import DrawKey, set_draw_key
from ..eval.routing_trace import RoutingTracer
from ..eval.teacher_audit import ROLE_CHECKPOINTS
from ..eval.world_space import point_from_record
from . import order_es_worlds as ow
from .per_order_scorer import (AGENT, ARCHITECTURE_VERSION, ARMS, B, DECODER_VERSION,
                               NORMALISER_VERSION, SIGMA, U_INIT_SEED, U_INIT_STD, ArmSpec,
                               FeatureNormaliser, NonFiniteChoice, input_transforms,
                               theta_sha256)

DRIVER = "tokenomics.train.order_es"
TRAIN, PROBE = "train", "probe"

# --- registered values (design D6 / D7; frozen in es_prereg_m4_v1.json by task 5.3) ---
PAIRS = 12
H_HOLDING = 0.1  # h in U / B_we - h * H / D_we
ADAM = {"lr": 0.02, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
WEIGHT_DECAY = 0.005
GENERATIONS = {"pilot": 20, "main": 40}
MONITOR_EVERY = 10
WEAK_MINIBATCH = 24  # D7 `weak`: the pilot and main runs re-sized to minibatch 24
NOISE_STREAM = "m4-es-noise-v1"
TRAINABLE_ARMS = ("A1", "A2", "A2-R2")
PROBE_ARMS = ("A1", "A2")
SIGNAL, WEAK, NO_SIGNAL = "signal", "weak", "no_signal"
SIGNAL_COSINE, WEAK_COSINE = 0.3, 0.1
A2_RANK2_BELOW = 0.1  # A2 cosine < 0.1 while A1 >= 0.1 -> A2 uses R = 2
HISTORY_ALPHA = 0.3  # env/agent_history.AgentHistory's default (= domain telemetry's EMA alpha)
MAX_READ_SIZE = 42  # spec "Ladder sizes stay at 12 and 42": no override
TRAINING_SIZE = 12

PREREG_FORMAT = "m4-es-prereg-v1"
PREREG_PATH = Path(__file__).resolve().parent / "registry" / "es_prereg_m4_v1.json"
DRAFT, FROZEN = "draft", "frozen"
CHECKPOINT_FORMAT = "m4-es-checkpoint-v1"
CENTER, A0_LABEL = "c", "A0"
MEMBER, MONITOR, PROBE_JOB = "member", "monitor", "probe"
NONFINITE_CHECK = "nonfinite_choice"
A0_REPRODUCTION = "a0_reproduction"

ES_TEXT = {
    "algorithm": "antithetic OpenAI-ES: members theta +- sigma * eps_i (eps_i ~ N(0, I), "
                 "i < pairs) plus the center every generation; per-parameter sigma",
    "rank_shaping": "centered ranks over the 2 * pairs members (rank / (2 pairs - 1) - 0.5, "
                    "average ranks on ties); disqualified and non-finite members share the "
                    "worst rank; the center is not ranked",
    "estimate": "g_j = sum_i (r(+i) - r(-i)) * eps_ij / (2 * pairs * sigma_j)",
    "update": "Adam ascending g - weight_decay * theta (L2 coupled into the gradient, "
              "OpenAI-ES l2coeff)",
    "noise": "eps of generation g: numpy default_rng(int(sha256('|'.join([noise_stream, "
             "'train', opt_seed, g]))[:16], 16)).standard_normal((pairs, n_params))",
}
FITNESS_TEXT = {
    "formula": "f = mean over the minibatch of U / B_we - h * H / D_we",
    "U": "consumer units (consumer_completed_units)",
    "B_we": "max over RR, MWW (R = 2 mean), TR and the roster v2 deterministic members (bar.csv)",
    "H": "holding_units_per_cycle", "D_we": "mean scheduled consumer demand per cycle (bar.csv)",
}
DQ_TEXT = ("a member's world-episode disqualifies when it dies and A0 does not, when r_end < "
           "x * r_start and A0's does not, or when a consumer's fill < y and A0 does not starve "
           "that consumer (per consumer), A0 on the same world-episode and keys (the bar's A0 "
           "row); one disqualifying world-episode -> the member takes the worst rank")
PROBE_TEXT = {
    "theta": "each arm's registered initial theta (A1: 0; A2: w = 0, theta_tau = 0, u ~ N(0, "
             "0.1) seed 0, v = 0), so Delta = 0 for both",
    "estimate": "per minibatch, the training estimate (centered ranks over that minibatch's "
                "member fitness); noise from noise_stream with ('probe', arm)",
    "cosine": "cosine between the two minibatches' estimates; an all-zero estimate leaves it "
              "undefined, read as no_signal",
    "variance_ratio": "var of paired (same minibatch) f(+i) - f(-i) over var of unpaired "
                      "f_1(+i) - f_2(-i), f_2(+i) - f_1(-i) (ddof 1, raw fitness, non-finite "
                      "members excluded); reported",
}


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(obj: object) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def _json_norm(obj: object) -> object:
    return json.loads(json.dumps(obj, sort_keys=True))


def _log(msg: str) -> None:
    print(msg, flush=True)


# --- pre-registration ---

STATIC_CLIP_STOP = ("a real ES run (probe or train) refuses a bar whose normaliser.json flags "
                    "static_psi_clip_42 (a 42-producer static count pinned at the clip); the "
                    "run stops and goes to the supervisor")


def bar_registration() -> dict[str, Any]:
    """D6's training-world and bar settings as the bar run pins them (`order_es_worlds`): the
    `worlds` section of the pre-registration, compared with the staged bar's `spec.json` pins in
    a real run (review round 1)."""
    from ..eval.roster import ROSTER_V2, load_roster

    roster = load_roster(ROSTER_V2)
    table = ow.bar_arms(roster)
    return _json_norm({
        "selection": {"target": ow.TARGET_WORLDS, "min_feasible": ow.MIN_FEASIBLE,
                      "max_replaced_share": ow.MAX_REPLACED_SHARE,
                      "rho_loaded": ow.RHO_LOADED, "saturation_fill": ow.SATURATION_FILL,
                      "fresh_atlas": ow.FRESH_ATLAS, "fresh_pool": ow.FRESH_POOL,
                      "train_stage": ow.TRAIN_STAGE,
                      "episodes_per_world": ow.EPISODES_PER_WORLD,
                      "atlas_worlds": ow.ATLAS_WORLDS},
        "arms": [a["arm"] for a in table], "replicates": ow.arm_replicates(table),
        "decode_by_arm": {a["arm"]: a["decode"] for a in table},
        "guards": {"x_rundown": ow.X_RUNDOWN, "y_starvation": ow.Y_STARVATION,
                   "side_effect_params": ow.SIDE_EFFECT_PARAMS.version},
        "schedule": {"stream": ow.SCHEDULE_STREAM, "monitor": ow.MONITOR_WORLD_EPISODES,
                     "monitor_replicates": ow.MONITOR_REPLICATES, "minibatch": ow.MINIBATCH,
                     "generations": ow.GENERATIONS, "opt_seeds": list(ow.OPT_SEEDS),
                     "probe_minibatch": ow.PROBE_MINIBATCH},
        "horizon": ow.HORIZON, "producer": PRODUCER, "roster_version": roster.version,
        "normaliser": ow.NORMALISER_PREREG,
    })


def bar_pin_problems(pins: Mapping[str, Any]) -> list[str]:
    """Where a bar run's `spec.json` pins differ from the registered D6 settings."""
    reg = bar_registration()
    got = _json_norm({k: pins.get(k) for k in reg if k != "replicates"})
    got["replicates"] = (ow.arm_replicates(pins["arm_table"]) if pins.get("arm_table")
                         else None)
    diff = _diff_paths(got, reg)
    return [f"the bar run's pins differ from the registered worlds / bar settings at "
            f"{', '.join(diff[:6])}"] if diff else []


def registered_sections() -> dict[str, Any]:
    """The sections of es_prereg_m4_v1.json this module enforces, built from its constants: a
    real run refuses a file whose sections differ (spec "Unregistered change refused")."""
    return _json_norm({
        "es": {**ES_TEXT, "pairs": PAIRS, "center_every_generation": True,
               "adam": ADAM, "weight_decay": {"value": WEIGHT_DECAY, "kind": "l2_coupled"},
               "opt_seeds": list(ow.OPT_SEEDS), "first_opt_seed": ow.OPT_SEEDS[0],
               "generations": GENERATIONS, "minibatch": ow.MINIBATCH,
               "weak_minibatch": WEAK_MINIBATCH, "noise_stream": NOISE_STREAM,
               "schedule_stream": ow.SCHEDULE_STREAM, "train_replicate": ow.TRAIN_REPLICATE,
               "fitness": {**FITNESS_TEXT, "h": H_HOLDING},
               "disqualification": {"rule": DQ_TEXT, "x_rundown": ow.X_RUNDOWN,
                                    "y_starvation": ow.Y_STARVATION,
                                    "side_effect_params": ow.SIDE_EFFECT_PARAMS.version},
               "nonfinite": "a member whose policy raises NonFiniteChoice takes the worst rank "
                            "and is logged; never a contract failure",
               "monitor": {"world_episodes": ow.MONITOR_WORLD_EPISODES,
                           "replicates": ow.MONITOR_REPLICATES, "every": MONITOR_EVERY,
                           "arms": ["center", "A0"], "at_generation_0": True},
               "horizon": ow.HORIZON, "training_size": TRAINING_SIZE,
               "max_read_size": MAX_READ_SIZE, "producer": PRODUCER,
               "history_mode": AGENT, "history_alpha": HISTORY_ALPHA,
               "static_clip_42_stop": STATIC_CLIP_STOP},
        "worlds": bar_registration(),
        "arms": {name: arm.to_dict() for name, arm in ARMS.items()},
        "sigma": {"w": SIGMA, "theta_tau": SIGMA, "u": SIGMA, "v": SIGMA},
        "policy": {"bound": B, "u_init": {"seed": U_INIT_SEED, "std": U_INIT_STD},
                   "decoder_version": DECODER_VERSION,
                   "architecture_version": ARCHITECTURE_VERSION, "ordering_decode": CHOICE_V1,
                   "base_sha256": ROLE_CHECKPOINTS["orderer"],
                   "input_transforms": input_transforms()},
        "normaliser": ow.NORMALISER_PREREG,
        "probe": {**PROBE_TEXT, "arms": list(PROBE_ARMS), "pairs": PAIRS, "minibatches": 2,
                  "minibatch": ow.PROBE_MINIBATCH,
                  "readings": {SIGNAL: SIGNAL_COSINE, WEAK: WEAK_COSINE},
                  "weak_next": {"minibatch": WEAK_MINIBATCH},
                  "a2_rank2_below": A2_RANK2_BELOW},
    })


# required paths of a pre-registration file (task 5.2 schema; 5.3 adds its reading sections)
PREREG_REQUIRED: tuple[tuple[str, ...], ...] = (
    ("format",), ("status",), ("arms",), ("sigma",), ("worlds", "selection"),
    ("worlds", "arms"), ("es", "pairs"), ("es", "minibatch"),
    ("es", "generations"), ("es", "adam", "lr"), ("es", "weight_decay"), ("es", "opt_seeds"),
    ("es", "fitness", "h"), ("es", "disqualification"), ("es", "monitor"),
    ("normaliser", "version"), ("normaliser", "fit_rows"), ("normaliser", "zero_spread_rule"),
    ("normaliser", "clip"), ("normaliser", "transforms"), ("policy", "decoder_version"),
    ("policy", "ordering_decode"), ("probe", "readings"), ("readings",),
)


def prereg_schema_problems(doc: Any) -> list[str]:
    if not isinstance(doc, dict):
        return ["pre-registration is not a JSON object"]
    out = []
    for path in PREREG_REQUIRED:
        node: Any = doc
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
        if node is None:
            out.append(f"pre-registration lacks {'.'.join(path)}")
    if doc.get("format") != PREREG_FORMAT:
        out.append(f"pre-registration format {doc.get('format')!r} != {PREREG_FORMAT!r}")
    if doc.get("status") not in (DRAFT, FROZEN):
        out.append(f"pre-registration status {doc.get('status')!r} not in {(DRAFT, FROZEN)}")
    return out


def _diff_paths(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b))
                for p in _diff_paths(a.get(k), b.get(k), f"{path}.{k}" if path else str(k))]
    return [] if a == b else [path]


def prereg_problems(doc: Any, *, mode: str, arm: str | None = None,
                    arms: Sequence[str] = (), opt_seed: int | None = None,
                    generations: int | None = None, minibatch: int | None = None,
                    pairs: int = PAIRS, monitor_every: int = MONITOR_EVERY) -> list[str]:
    """Why a real run may not start under `doc`: schema, a draft file, any enforced section
    differing from this module's registered values, or a run choice outside them (arm, opt seed,
    generations, minibatch, pairs, monitor cadence)."""
    out = prereg_schema_problems(doc)
    if out:
        return out
    if doc["status"] != FROZEN:
        out.append(f"pre-registration status is {doc['status']!r}, a real run needs {FROZEN!r}")
    reg = registered_sections()
    for sec, want in reg.items():
        diff = _diff_paths(_json_norm(doc.get(sec)), want)
        if diff:
            out.append(f"pre-registration differs from the registered values at "
                       f"{', '.join(f'{sec}.{d}' if d else sec for d in diff[:6])}")
    es = reg["es"]
    if pairs != es["pairs"]:
        out.append(f"pairs {pairs} != registered {es['pairs']}")
    if mode == TRAIN:
        if arm not in TRAINABLE_ARMS:
            out.append(f"arm {arm!r} is not a registered trainable arm {TRAINABLE_ARMS}")
        if opt_seed not in es["opt_seeds"]:
            out.append(f"opt seed {opt_seed} not in the registered {es['opt_seeds']}")
        if generations not in es["generations"].values():
            out.append(f"generations {generations} not in the registered "
                       f"{sorted(es['generations'].values())}")
        if minibatch not in (es["minibatch"], es["weak_minibatch"]):
            out.append(f"minibatch {minibatch} not in the registered "
                       f"{(es['minibatch'], es['weak_minibatch'])}")
        if monitor_every != es["monitor"]["every"]:
            out.append(f"monitor every {monitor_every} != registered {es['monitor']['every']}")
    else:
        if list(arms) != reg["probe"]["arms"]:
            out.append(f"probe arms {list(arms)} != registered {reg['probe']['arms']}")
        if minibatch != reg["probe"]["minibatch"]:
            out.append(f"probe minibatch {minibatch} != registered {reg['probe']['minibatch']}")
    return out


def draft_prereg() -> dict[str, Any]:
    """The DRAFT file task 5.3 finalises: the enforced sections plus placeholders for 5.3's
    validation readings."""
    return {"format": PREREG_FORMAT, "status": DRAFT, "change": "add-m4-orderer-redesign",
            "note": "DRAFT written by task 5.2 from order_es.registered_sections(); task 5.3 "
                    "fills `readings` (D7 / D9: floors, n = 1 rule, power rule, reading table, "
                    "seed table, labels), adds the normaliser fitting-code hash, sets status "
                    "'frozen' and writes train/registry/es_prereg_m4_v1.json",
            **registered_sections(),
            "readings": {"status": "TODO task 5.3"}}


def load_prereg(path: str | Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a JSON object")
    return doc


# --- pure ES ---


def centered_ranks(fitness: Sequence[float], worst: Sequence[bool]) -> np.ndarray:
    """Centered ranks in [-0.5, 0.5] over all members (average ranks on ties); `worst` members
    share the lowest rank group whatever their fitness. Finite fitness is required elsewhere."""
    f = np.asarray(fitness, dtype=np.float64)
    w = np.asarray(worst, dtype=bool)
    n = len(f)
    if n < 2:
        raise ValueError("centered ranks need at least two members")
    if not np.all(np.isfinite(f[~w])):
        raise ValueError("a ranked member has non-finite fitness")
    x = np.where(w, -np.inf, f)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ranks = (start + (counts - 1) / 2.0)[inv]
    return ranks / (n - 1) - 0.5


def es_estimate(eps: np.ndarray, r_plus: np.ndarray, r_minus: np.ndarray,
                sigma: np.ndarray) -> np.ndarray:
    """g_j = sum_i (r(+i) - r(-i)) eps_ij / (2 n sigma_j): the antithetic estimate on ranks."""
    n = eps.shape[0]
    return (np.asarray(r_plus) - np.asarray(r_minus)) @ eps / (2.0 * n * np.asarray(sigma))


@dataclass
class AdamState:
    m: np.ndarray
    v: np.ndarray
    t: int = 0

    @classmethod
    def zeros(cls, n: int) -> "AdamState":
        return cls(np.zeros(n), np.zeros(n), 0)

    def to_dict(self) -> dict[str, Any]:
        return {"m": [float(x) for x in self.m], "v": [float(x) for x in self.v], "t": self.t}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AdamState":
        return cls(np.asarray(d["m"], np.float64), np.asarray(d["v"], np.float64), int(d["t"]))


def adam_ascend(theta: np.ndarray, grad: np.ndarray, state: AdamState, *,
                lr: float = ADAM["lr"], beta1: float = ADAM["beta1"],
                beta2: float = ADAM["beta2"], eps: float = ADAM["eps"],
                weight_decay: float = WEIGHT_DECAY) -> tuple[np.ndarray, AdamState]:
    """One Adam step ascending `grad`, with L2 weight decay coupled into the gradient (OpenAI-ES
    minimises -g + l2coeff * theta)."""
    g = -np.asarray(grad, np.float64) + weight_decay * theta
    t = state.t + 1
    m = beta1 * state.m + (1 - beta1) * g
    v = beta2 * state.v + (1 - beta2) * g * g
    step = lr * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + eps)
    return theta - step, AdamState(m, v, t)


def noise(opt_seed: int, generation: int, pairs: int, n_params: int,
          purpose: str = "train") -> np.ndarray:
    seed = int(hashlib.sha256("|".join(str(p) for p in (NOISE_STREAM, purpose, opt_seed,
                                                        generation)).encode()
                              ).hexdigest()[:16], 16)
    return np.random.default_rng(seed).standard_normal((pairs, n_params))


def probe_noise(arm: str, pairs: int, n_params: int) -> np.ndarray:
    return noise(0, 0, pairs, n_params, purpose=f"probe|{arm}")


def member_labels(pairs: int) -> list[tuple[str, int, int]]:
    """(label, sign, pair) of the 2n ranked members, plus first ones then minus ones."""
    return ([(f"+{i:02d}", 1, i) for i in range(pairs)]
            + [(f"-{i:02d}", -1, i) for i in range(pairs)])


def member_thetas(theta: np.ndarray, eps: np.ndarray, sigma: np.ndarray
                  ) -> dict[str, np.ndarray]:
    """The center and theta +- sigma * eps_i, keyed by label."""
    out = {CENTER: np.array(theta, np.float64, copy=True)}
    for label, sign, i in member_labels(eps.shape[0]):
        out[label] = theta + sign * np.asarray(sigma) * eps[i]
    return out


@dataclass(frozen=True)
class Ranked:
    """The ranks of one generation (or one probe minibatch) and its estimate."""

    fitness: dict[str, float | None]
    worst: dict[str, bool]
    ranks: dict[str, float]
    estimate: np.ndarray


def rank_and_estimate(eps: np.ndarray, sigma: np.ndarray, fitness: Mapping[str, float | None],
                      worst: Mapping[str, bool]) -> Ranked:
    """Centered ranks over the 2n members (a None fitness = non-finite, worst) and the
    antithetic estimate."""
    labels = member_labels(eps.shape[0])
    bad = [bool(worst[lab]) or fitness[lab] is None for lab, _, _ in labels]
    f = [0.0 if fitness[lab] is None else float(fitness[lab]) for lab, _, _ in labels]
    r = centered_ranks(f, bad)
    ranks = {lab: float(r[k]) for k, (lab, _, _) in enumerate(labels)}
    n = eps.shape[0]
    est = es_estimate(eps, r[:n], r[n:], sigma)
    return Ranked(dict(fitness), {lab: b for (lab, _, _), b in zip(labels, bad)}, ranks, est)


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def variance_ratio(f1: Mapping[str, float | None], f2: Mapping[str, float | None],
                   pairs: int) -> float | None:
    """var(paired f_k(+i) - f_k(-i)) / var(unpaired f_1(+i) - f_2(-i), f_2(+i) - f_1(-i))."""
    paired, unpaired = [], []
    for i in range(pairs):
        p, m = f"+{i:02d}", f"-{i:02d}"
        for a, b, out in ((f1[p], f1[m], paired), (f2[p], f2[m], paired),
                          (f1[p], f2[m], unpaired), (f2[p], f1[m], unpaired)):
            if a is not None and b is not None:
                out.append(a - b)
    if len(paired) < 2 or len(unpaired) < 2:
        return None
    vu = float(np.var(unpaired, ddof=1))
    return None if vu == 0.0 else float(np.var(paired, ddof=1)) / vu


def probe_reading(cos: Mapping[str, float | None]) -> dict[str, Any]:
    """D7: `signal` A1 cosine >= 0.3, `weak` 0.1 <= A1 < 0.3, else `no_signal` (an undefined
    cosine too); A2 cosine < 0.1 (or undefined) while A1 >= 0.1 -> A2 uses R = 2."""
    a1 = cos.get("A1")
    reading = (NO_SIGNAL if a1 is None or a1 < WEAK_COSINE
               else WEAK if a1 < SIGNAL_COSINE else SIGNAL)
    a2 = cos.get("A2", cos.get("A2-R2"))
    rank2 = reading != NO_SIGNAL and "A2" in cos and (a2 is None or a2 < A2_RANK2_BELOW)
    nxt = {SIGNAL: "pilot as registered",
           WEAK: f"minibatch {WEAK_MINIBATCH}; pilot and main runs re-sized and launch-fit-checked",
           NO_SIGNAL: "diagnose, stop"}[reading]
    return {"reading": reading, "next": nxt, "a1_cosine": a1, "a2_cosine": a2,
            "a2_rank": 2 if rank2 else ARMS["A2"].rank,
            "a2_arm": "A2-R2" if rank2 else "A2",
            "minibatch": WEAK_MINIBATCH if reading == WEAK else ow.MINIBATCH}


# --- fitness and disqualification ---


@dataclass(frozen=True)
class BarRow:
    world: str
    episode: int
    bar: float
    demand_per_cycle: float
    a0_units: float
    a0_dead: bool
    a0_rundown: bool
    a0_starved: frozenset[str]
    demand_digest: str
    shock_digest: str


def _bool(s: str) -> bool:
    if s not in ("true", "false", "True", "False"):
        raise ValueError(f"not a boolean cell: {s!r}")
    return s in ("true", "True")


def read_bar(path: Path) -> dict[tuple[str, int], BarRow]:
    out = {}
    with path.open(newline="") as fh:
        for d in csv.DictReader(fh):
            r = BarRow(d["world"], int(d["episode"]), float(d["bar"]), float(d["demand_per_cycle"]),
                       float(d["a0_units"]), _bool(d["a0_dead"]), _bool(d["a0_rundown"]),
                       frozenset(json.loads(d["a0_starved_consumers"])), d["demand_digest"],
                       d["shock_digest"])
            if not (r.bar > 0 and r.demand_per_cycle > 0):
                raise ValueError(f"bar row {r.world} ep {r.episode}: B_we and D_we must be > 0")
            out[(r.world, r.episode)] = r
    return out


def _opt(s: str) -> float | None:
    return None if s in ("", "None") else float(s)


def disqualifying(row: Mapping[str, str], bar: BarRow) -> list[str]:
    """The relative guards (D6) a member violates on one world-episode and A0 does not."""
    out = []
    if _bool(row["dead"]) and not bar.a0_dead:
        out.append("death")
    rs, re_ = _opt(row["r_start"]), _opt(row["r_end"])
    if rs is not None and re_ is not None and re_ < ow.X_RUNDOWN * rs and not bar.a0_rundown:
        out.append("rundown")
    fills = json.loads(row["consumer_fill_rate"])
    starved = sorted(c for c, v in fills.items()
                     if v is not None and v < ow.Y_STARVATION and c not in bar.a0_starved)
    if starved:
        out.append("starvation:" + ",".join(starved))
    return out


def we_fitness(row: Mapping[str, str], bar: BarRow, h: float = H_HOLDING) -> float:
    holding = _opt(row["holding_units_per_cycle"]) or 0.0
    return float(row["units"]) / bar.bar - h * holding / bar.demand_per_cycle


@dataclass(frozen=True)
class MemberScore:
    fitness: float | None  # None: non-finite (the policy raised NonFiniteChoice)
    disqualified: bool
    reasons: tuple[str, ...]
    nonfinite: bool


def score_member(rows: Sequence[Mapping[str, str]], bar: Mapping[tuple[str, int], BarRow],
                 h: float = H_HOLDING) -> MemberScore:
    """Minibatch-mean fitness and the relative disqualification of one member."""
    if any(_bool(r["nonfinite"]) for r in rows):
        return MemberScore(None, False, ("nonfinite",), True)
    reasons, fs = [], []
    for r in rows:
        b = bar[(r["world"], int(r["episode"]))]
        fs.append(we_fitness(r, b, h))
        reasons += [f"{r['world']}:{r['episode']}:{x}" for x in disqualifying(r, b)]
    return MemberScore(fmean(fs), bool(reasons), tuple(reasons), False)


# --- inputs ---


@dataclass
class Inputs:
    """The staged bar run (EXP-0058's `bar/`): bar table, training worlds, schedule, normaliser,
    each checked against its own hash."""

    bar_dir: Path
    bar_spec: dict
    bar: dict[tuple[str, int], BarRow]
    training: dict
    schedule: dict
    normaliser: FeatureNormaliser
    normaliser_doc: dict
    entries: dict[str, WorldEntry]
    costs: dict[str, float]
    world_episodes: list[tuple[str, int]]
    hashes: dict[str, str] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return int(self.training["horizon"])

    @property
    def dry_run(self) -> bool:
        return bool(self.training["dry_run"])


def size_problems(sizes: Mapping[str, int]) -> list[str]:
    """Training at 12 producers only; nothing above 42 (no override)."""
    out = [f"world {w} has {s} producers (> {MAX_READ_SIZE}, refused without override)"
           for w, s in sorted(sizes.items()) if s > MAX_READ_SIZE]
    out += [f"training world {w} has {s} producers (training size is {TRAINING_SIZE})"
            for w, s in sorted(sizes.items()) if s != TRAINING_SIZE and s <= MAX_READ_SIZE]
    return out


def _fmt_we(items: Sequence[Mapping[str, Any]]) -> list[tuple[str, int]]:
    return [(str(d["world"]), int(d["episode"])) for d in items]


def load_inputs(bar_dir: str | Path) -> Inputs:
    bd = Path(bar_dir)
    spec = json.loads((bd / "spec.json").read_text())
    if spec.get("status") != ow.COMPLETE:
        raise ValueError(f"{bd} is not a complete bar run (status {spec.get('status')!r})")
    training = json.loads((bd / "training_worlds.json").read_text())
    if ow.content_hash(training) != training.get("content_hash"):
        raise ValueError("training_worlds.json does not match its content hash")
    schedule = json.loads((bd / "schedule.json").read_text())
    if ow.content_hash(schedule) != schedule.get("content_hash"):
        raise ValueError("schedule.json does not match its content hash")
    ndoc = json.loads((bd / "normaliser.json").read_text())
    nm = FeatureNormaliser.from_dict(ndoc["normaliser"])  # refuses another version / transform
    if nm.sha256 != ndoc["sha256"]:
        raise ValueError("normaliser.json does not match its sha256")
    if ndoc.get("training_worlds_sha256") != training["content_hash"]:
        raise ValueError("normaliser.json was not fitted on these training worlds")
    problems = size_problems({w["world"]: int(w["record"]["size"]) for w in training["worlds"]})
    if problems or int(training["size"]) != TRAINING_SIZE:
        raise ValueError("training sizes refused: " + "; ".join(
            problems or [f"training_worlds.json size {training['size']}"]))
    entries, costs = {}, {}
    for w in training["worlds"]:
        point = replace(point_from_record(w["record"]), shock_window=tuple(w["shock_window"]))
        if point.world_id != w["world"]:
            raise ValueError(f"record of {w['world']} describes {point.world_id}")
        c = ow.Candidate(int(w[ow.ORDER_INDEX]), str(w["source"]), point,
                         tuple(int(e) for e in w["episodes"]), float(w["rho"]))
        entries[c.world] = c.entry()
        cs = w.get("cost_s") or {}
        costs[c.world] = float(cs.get(ow.A0, max(cs.values(), default=1.0)))
    we = [(w["world"], int(ep)) for w in training["worlds"] for ep in w["episodes"]]
    bar = read_bar(bd / "bar.csv")
    if set(bar) != set(we):
        raise ValueError("bar.csv rows differ from the training world-episodes")
    # the schedule must be the recorded stream over these world-episodes
    if _fmt_we(schedule["monitor"]["world_episodes"]) != ow.monitor_set(we):
        raise ValueError("schedule.json monitor set differs from the recorded stream")
    return Inputs(bd, spec, bar, training, schedule, nm, ndoc, entries, costs, we, {
        "bar_csv_sha256": sha256_file(str(bd / "bar.csv")),
        "training_worlds_sha256": training["content_hash"],
        "schedule_sha256": schedule["content_hash"], "normaliser_sha256": nm.sha256,
        "normaliser_fit_rows_sha256": str(ndoc.get("fit_rows_sha256")),
        "bar_spec_pins_sha256": _sha(spec.get("pins"))})


def minibatch_for(inp: Inputs, opt_seed: int, generation: int, size: int) -> list[tuple[str, int]]:
    """Generation g's CRN minibatch: schedule.json's entry (checked against the recorded stream)
    when the schedule holds it, else the stream itself (the `weak` size or more generations)."""
    want = ow.minibatch(inp.world_episodes, opt_seed, generation, size)
    mbs = inp.schedule["minibatches"].get(str(opt_seed), [])
    if size == int(inp.schedule["minibatch_size"]) and generation <= len(mbs):
        got = _fmt_we(mbs[generation - 1])
        if got != want:
            raise ValueError(f"schedule.json minibatch {opt_seed}/{generation} differs from the "
                             f"recorded stream")
        return got
    return want


def probe_batches(inp: Inputs) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    p = inp.schedule["probe"]
    got = [_fmt_we(m) for m in p["minibatches"]]
    want = ow.probe_minibatches(inp.world_episodes, int(p["size"]))
    if (got[0], got[1]) != want:
        raise ValueError("schedule.json probe minibatches differ from the recorded stream")
    if set(got[0]) & set(got[1]):
        raise ValueError("probe minibatches overlap")
    return got[0], got[1]


# --- jobs and worker ---


def job_id(kind: str, generation: int | str, member: str, world: str, episode: int,
           replicate: int, arm: str = "") -> str:
    return f"{kind}:{arm}:g{generation}:{member}:{world}:{episode}:r{replicate}"


def make_job(inp: Inputs, ckpt: Mapping[str, str], *, kind: str, arm: str, theta: np.ndarray,
             generation: int | str, member: str, world: str, episode: int, replicate: int,
             sign: int = 0, pair: int = -1, batch: int = -1) -> Job:
    e = inp.entries[world]
    t = [float(x) for x in np.asarray(theta, np.float64)]
    return {"kind": kind, "arm": arm, "theta": t, "theta_sha256": theta_sha256(np.asarray(t)),
            "generation": str(generation), "member": member, "sign": sign, "pair": pair,
            "batch": batch, "world": world, "size": e.size, "regime": e.regime,
            "world_seed": e.world_seed, "generator": e.generator, "episode": int(episode),
            "replicate": int(replicate), "horizon": inp.horizon, "producing": PRODUCER,
            "subject": f"es/{arm}/{member}", "action_seed": None,
            "checkpoints": {"orderer": dict(ckpt)},
            "normaliser": inp.normaliser.to_dict(), "normaliser_sha256": inp.normaliser.sha256,
            "cost": inp.costs[world],
            "job_id": job_id(kind, generation, member, world, episode, replicate, arm)}


def generation_jobs(inp: Inputs, ckpt: Mapping[str, str], arm: str,
                    thetas: Mapping[str, np.ndarray], minibatch: Sequence[tuple[str, int]],
                    generation: int) -> list[Job]:
    """One job per member x minibatch world-episode, all at the training replicate's key (CRN:
    every member plays the same world-episodes and keys; the key never names the member)."""
    return [make_job(inp, ckpt, kind=MEMBER, arm=arm, theta=th, generation=generation,
                     member=lab, world=w, episode=e, replicate=ow.TRAIN_REPLICATE,
                     sign=0 if lab == CENTER else (1 if lab[0] == "+" else -1),
                     pair=-1 if lab == CENTER else int(lab[1:]))
            for lab, th in thetas.items() for w, e in minibatch]


def longest_first(jobs: Sequence[Job]) -> list[Job]:
    """By the world's measured A0 episode seconds (bar `cost_s`), longest first; stable."""
    return sorted(jobs, key=lambda j: -float(j["cost"]))


ES_COLUMNS = ("job_id", "kind", "arm", "generation", "member", "sign", "pair", "batch",
              "theta_sha256", "ordering_decode", "nonfinite", "nonfinite_message",
              "episode_seconds", "job_seconds")
ROW_COLUMNS = CSV_COLUMNS + tuple(TRACE_PREFIX + k for k in TRACE_SCALARS) + ES_COLUMNS

_NORMALISERS: dict[str, FeatureNormaliser] = {}


def _normaliser(job: Job) -> FeatureNormaliser:
    sha = job["normaliser_sha256"]
    if sha not in _NORMALISERS:
        nm = FeatureNormaliser.from_dict(job["normaliser"])
        if nm.sha256 != sha:
            raise ValueError("job normaliser does not match its sha256")
        _NORMALISERS[sha] = nm
    return _NORMALISERS[sha]


def run_es_job(job: Job) -> dict[str, str]:
    """One member's traced, keyed episode under choice-v1 -> its row. `NonFiniteChoice` is a
    row with `nonfinite` true (the member ranks worst); any other exception is the job's
    contract-failure row."""
    from ..eval.baselines import PRODUCING_BASELINES
    from ..eval.runner import run_episode
    from .per_order_scorer import PerOrderPolicy, per_order_env

    started = time.perf_counter()
    es = {"job_id": job["job_id"], "kind": job["kind"], "arm": job["arm"],
          "generation": job["generation"], "member": job["member"], "sign": str(job["sign"]),
          "pair": str(job["pair"]), "batch": str(job["batch"]),
          "theta_sha256": job["theta_sha256"], "ordering_decode": CHOICE_V1}
    try:
        env = per_order_env(job_config(job), job["horizon"])
        set_draw_key(env, DrawKey(str(job["world"]), int(job["episode"]), int(job["replicate"])))
        tracer = RoutingTracer()  # no forks here: the only overflow-order counter (4.3 review)
        tracer.attach(env)
        monitor = ContractMonitor(env)
        arm = ARMS[job["arm"]]
        policy = PerOrderPolicy(ow._base_scorer(job), arm, np.asarray(job["theta"], np.float64),
                                _normaliser(job) if arm.residual else None, AGENT)
        if theta_sha256(policy.theta) != job["theta_sha256"]:
            raise AssertionError("job theta does not match its sha256")
        t0 = time.perf_counter()
        try:
            run_episode(env, PRODUCING_BASELINES[job["producing"]], policy, seed=job["episode"],
                        action_seed=job["action_seed"], monitor=monitor)
        except NonFiniteChoice as exc:
            row = {c: "" for c in ROW_COLUMNS}
            row.update({"world": job["world"], "size": str(job["size"]),
                        "regime": job["regime"], "episode": str(job["episode"]),
                        "subject": job["subject"], "replicate": str(job["replicate"]), **es,
                        "nonfinite": "true", "nonfinite_message": str(exc)[:300],
                        "episode_seconds": repr(time.perf_counter() - t0),
                        "job_seconds": repr(time.perf_counter() - started)})
            return row
        secs = time.perf_counter() - t0
        stats = env.episode_stats
        monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=job["horizon"])
        if env.ordering_decode != CHOICE_V1:
            raise AssertionError(f"ES job ran under {env.ordering_decode}")
        row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"],
                                         job["episode"], job["subject"], job["replicate"],
                                         stats))
        row.update({TRACE_PREFIX + k: repr(float(v)) for k, v in tracer.summary().items()})
        row.update({**es, "nonfinite": "false", "nonfinite_message": "",
                    "episode_seconds": repr(secs),
                    "job_seconds": repr(time.perf_counter() - started)})
        return {c: row.get(c, "") for c in ROW_COLUMNS}
    except Exception as exc:  # noqa: BLE001 - recorded as the job's failure, never fatal
        return job_failure(job, exc).to_dict()


# --- executor ---


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _append(path: Path, row: Mapping[str, str], columns: Sequence[str]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in columns})
        fh.flush()
        os.fsync(fh.fileno())


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: "" if r.get(c) is None else r.get(c) for c in columns})
    os.replace(tmp, path)


def _check_digests(out: Path, row: Mapping[str, str], bar: Mapping[tuple[str, int], BarRow]
                   ) -> None:
    """Every member, generation and monitor row of a training world-episode sees the bar's
    demand and shock digests (CRN): a mismatch is a run row and raises."""
    if row.get("nonfinite") == "true":
        return
    b = bar[(row["world"], int(row["episode"]))]
    for what, want in (("demand_digest", b.demand_digest), ("shock_digest", b.shock_digest)):
        if row[what] != want:
            exc = DigestMismatch(f"{row['job_id']} saw {what} {row[what]}, the bar's A0 saw {want}")
            append_failure(out, run_failure(DIGEST, exc))
            raise exc


def play(out: Path, rows_file: str, jobs: Sequence[Job], workers: int,
         run_job_fn: Callable[[Job], dict[str, str]], bar: Mapping[tuple[str, int], BarRow],
         log: Callable[[str], None]) -> tuple[dict[str, dict[str, str]], int]:
    """Run every job whose row is not yet in `rows_file` (longest first), appending rows as they
    land; returns (rows of these jobs by job id, new contract failures)."""
    path = out / rows_file
    want = {j["job_id"]: j for j in jobs}
    have = {d["job_id"]: d for d in read_csv(path) if d["job_id"] in want}
    for jid, d in have.items():  # a row of an earlier session: same member theta
        if d["theta_sha256"] != want[jid]["theta_sha256"]:
            raise ValueError(f"{rows_file} row {jid} has another theta than this run's member")
    todo = longest_first([j for j in jobs if j["job_id"] not in have])
    failures = 0

    def finish(d: dict[str, str]) -> None:
        nonlocal failures
        if is_failure(d):
            failures += 1
            append_failure(out, failure_from_dict(d))
            log(f"[order_es] CONTRACT FAILURE {d['world']} ep={d['episode']} {d['subject']} "
                f"r{d['replicate']}: {d['check']} {d['error_type']}")
            return
        _check_digests(out, d, bar)
        _append(path, d, ROW_COLUMNS)
        have[d["job_id"]] = d
        if d["nonfinite"] == "true":
            log(f"[order_es] NONFINITE {d['job_id']}: {d['nonfinite_message']} (ranked worst)")

    if workers <= 1 or len(todo) <= 1:
        for j in todo:
            finish(run_job_fn(j))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                 initializer=_init_worker) as pool:
            futures = [pool.submit(run_job_fn, j) for j in todo]
            try:
                for fut in as_completed(futures):
                    finish(fut.result())
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
    return have, failures


# --- checkpoints and pins ---


def checkpoint_path(out: Path, generation: int) -> Path:
    return out / "checkpoints" / f"gen-{generation:04d}.json"


def latest_checkpoint(out: Path) -> Path | None:
    found = sorted((out / "checkpoints").glob("gen-*.json")) if (out / "checkpoints").exists() \
        else []
    return found[-1] if found else None


def save_checkpoint(path: Path, doc: dict) -> None:
    doc = dict(doc)
    doc["content_hash"] = _sha({k: v for k, v in doc.items() if k != "content_hash"})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_checkpoint(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        found = latest_checkpoint(p)
        if found is None:
            raise ValueError(f"{p} holds no checkpoints/gen-*.json")
        p = found
    doc = json.loads(p.read_text())
    if doc.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{p} is not a {CHECKPOINT_FORMAT} checkpoint")
    if _sha({k: v for k, v in doc.items() if k != "content_hash"}) != doc.get("content_hash"):
        raise ValueError(f"{p} does not match its content hash")
    if theta_sha256(np.asarray(doc["theta"])) != doc["theta_sha256"]:
        raise ValueError(f"{p} theta does not match its sha256")
    return doc


def init_theta(path: str | Path, arm: ArmSpec) -> np.ndarray:
    """A composite checkpoint's theta (dry runs only: a real run starts from the registered
    initial theta or a `--resume-from` checkpoint)."""
    from .per_order_scorer import CHECKPOINT_FORMAT as COMPOSITE

    doc = json.loads(Path(path).read_text())
    if doc.get("format") != COMPOSITE or doc.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(f"{path} is not a {COMPOSITE} {ARCHITECTURE_VERSION} composite")
    if doc["arm"]["name"] != arm.name:
        raise ValueError(f"{path} holds arm {doc['arm']['name']}, not {arm.name}")
    theta = np.asarray(doc["theta"], np.float64)
    if theta_sha256(theta) != doc["theta_sha256"] or theta.shape != (arm.n_params,):
        raise ValueError(f"{path} theta does not match its sha256 / the arm's size")
    return theta


def _pin_diff(a: Mapping, b: Mapping) -> str:
    return ", ".join(_diff_paths(_json_norm(a), _json_norm(b))[:8])


def _open_run(out: Path, pins: dict, fresh: dict, workers: int) -> tuple[dict, bool]:
    """A new spec, or (same pins) the recorded one with the resume appended; anything else is
    refused before a write. Only this driver's files are looked at, so a bar run below the out
    dir (EXP-0058's `bar/`) is left alone."""
    from ..eval.gate_cli import _open_spec

    spec_path = out / "spec.json"
    old = json.loads(spec_path.read_text()) if spec_path.exists() else None
    mine = [f for f in ("members.csv", "monitor.csv", "probe_episodes.csv") if (out / f).exists()]
    if old is None and (mine or (out / "checkpoints").exists()):
        raise ValueError(f"{out} holds ES rows or checkpoints without spec.json; refusing")
    if old is not None and old.get("driver") != DRIVER:
        raise ValueError(f"{out}/spec.json belongs to {old.get('driver')!r}, not {DRIVER}")
    if old is not None and old.get("pins") != _json_norm(pins):
        raise ValueError(f"{out} holds a run with different pins; refusing to resume "
                         f"(differs at {_pin_diff(old.get('pins') or {}, pins)})")
    spec = _open_spec(out, old, workers, {**fresh, "pins": _json_norm(pins)})
    ensure_failures_file(out)
    return spec, old is not None


# --- train ---


@dataclass
class TrainState:
    generation: int
    theta: np.ndarray
    adam: AdamState
    history: list[dict]
    monitor: list[dict]


def _member_rows(rows: Mapping[str, dict[str, str]], member: str
                 ) -> list[dict[str, str]]:
    return [d for d in rows.values() if d["member"] == member]


def monitor_summary(generation: int, center: Sequence[Mapping[str, str]],
                    a0_mean: float, history: Sequence[Mapping]) -> dict:
    units = [float(d["units"]) for d in center if d["nonfinite"] == "false"]
    mean = fmean(units) if len(units) == len(center) and units else None
    prev = {int(m["generation"]): m["center_units"] for m in history}
    back = prev.get(generation - MONITOR_EVERY)
    return {"generation": generation, "center_units": mean, "a0_units": a0_mean,
            "g_tr_raw": None if mean is None else mean - a0_mean,
            "rows": len(center), "nonfinite": len(center) - len(units),
            f"improvement_{MONITOR_EVERY}": None if mean is None or back is None
            else mean - back}


GENERATION_COLUMNS = ("generation", "minibatch_sha256", "center_fitness", "member_mean",
                      "member_max", "member_min", "disqualified", "nonfinite", "estimate_norm",
                      "step_norm", "theta_sha256", "theta_norm", "theta_tau", "tau", "seconds",
                      "jobs")
MONITOR_COLUMNS = ("generation", "center_units", "a0_units", "g_tr_raw",
                   f"improvement_{MONITOR_EVERY}", "rows", "nonfinite")
FITNESS_COLUMNS = ("generation", "member", "sign", "pair", "fitness", "disqualified",
                   "nonfinite", "rank", "reasons")


def real_run_problems(*, inp: Inputs, prereg: dict | None, prereg_issues: list[str],
                      source_commit: str, orderer_sha: str, filters: Mapping[str, Any]
                      ) -> list[str]:
    out = [msg for bad, msg in (
        (inp.dry_run or inp.bar_spec.get("kind") != "bar",
         "the bar run is a dry run (or not a bar run)"),
        (inp.horizon != ow.HORIZON, f"horizon {inp.horizon} != {ow.HORIZON}"),
        (not _is_commit(source_commit),
         "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
        (orderer_sha != ROLE_CHECKPOINTS["orderer"],
         f"orderer checkpoint {orderer_sha[:12]} is not the registered "
         f"{ROLE_CHECKPOINTS['orderer'][:12]}"),
        (len(inp.training["worlds"]) < ow.MIN_FEASIBLE,
         f"{len(inp.training['worlds'])} training worlds (< {ow.MIN_FEASIBLE})"),
        (prereg is None, "no pre-registration file (--prereg)"),
        (inp.normaliser_doc.get("prereg") != _json_norm(ow.NORMALISER_PREREG),
         "normaliser.json's rule block differs from the registered normaliser rule"),
        (bool((inp.normaliser_doc.get("flags") or {}).get("static_psi_clip_42")),
         "the bar's normaliser pins 42-producer static counts at the clip "
         f"({(inp.normaliser_doc.get('flags') or {}).get('static_psi_clip_42')}); "
         "stop, to the supervisor"),
    ) if bad]
    out += bar_pin_problems(inp.bar_spec.get("pins") or {})
    out += prereg_issues
    out += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    return out


def _checked_orderer(out: Path, path: str) -> dict[str, str]:
    from ..eval.teacher_audit import checked_orderer

    return checked_orderer(out, path, _log)


def training_pins(*, inp: Inputs, arm: ArmSpec, opt_seed: int, pairs: int, minibatch: int,
                  monitor_every: int, orderer_sha: str, prereg_sha: str | None,
                  probe_sha: str | None, init_sha: str, per_order: Mapping) -> dict:
    """Everything the optimisation trajectory depends on (a `--resume-from` must match them):
    not the generation target, the workers or the commit text."""
    return _json_norm({
        "driver": DRIVER, "mode": TRAIN, "arm": arm.to_dict(), "opt_seed": opt_seed,
        "pairs": pairs, "minibatch": minibatch, "monitor_every": monitor_every,
        "horizon": inp.horizon, "dry_run": inp.dry_run,
        "hyper": {"adam": ADAM, "weight_decay": WEIGHT_DECAY, "h": H_HOLDING,
                  "sigma": theta_sha256(arm.sigma()), "noise_stream": NOISE_STREAM,
                  "x_rundown": ow.X_RUNDOWN, "y_starvation": ow.Y_STARVATION},
        "inputs": {**inp.hashes, "orderer_sha256": orderer_sha, "prereg_sha256": prereg_sha,
                   "probe_sha256": probe_sha, "init_theta_sha256": init_sha},
        **decode_pins(CHOICE_V1), "per_order": per_order, "producer": PRODUCER,
        "source_tree_sha256": source_fingerprint()["tree_sha256"]})


def _probe_doc_problems(doc: Mapping[str, Any], arm: str, minibatch: int) -> list[str]:
    r = doc.get("reading", {})
    out = []
    if r.get("reading") not in (SIGNAL, WEAK):
        out.append(f"probe reading {r.get('reading')!r}: training needs signal or weak")
    elif minibatch != r.get("minibatch"):
        out.append(f"minibatch {minibatch} != the probe's {r.get('minibatch')}")
    if arm in ("A2", "A2-R2") and arm != r.get("a2_arm"):
        out.append(f"arm {arm} but the probe sets {r.get('a2_arm')}")
    if doc.get("status") != "complete" or doc.get("dry_run"):
        out.append("probe.json is not a complete real probe")
    return out


def run_train(*, bar_dir: str | Path, out_dir: str | Path, orderer_ckpt: str, arm: str,
              opt_seed: int, generations: int, workers: int = 1, prereg_path: str | None = None,
              probe_path: str | None = None, resume_from: str | None = None,
              init: str | None = None, pairs: int | None = None, minibatch: int | None = None,
              monitor_every: int | None = None, source_commit: str = "",
              dry_run: bool = False,
              run_job_fn: Callable[[Job], dict[str, str]] | None = None,
              log: Callable[[str], None] = _log) -> int:
    """Train (or resume) one arm x opt seed to `generations`; exit 0, or CONTRACT_INVALID."""
    from .per_order_scorer import (BaseScorer, PerOrderPolicy, composite_dict, per_order_pins,
                                   save_composite)

    out = Path(out_dir)
    run_job_fn = run_job_fn or run_es_job
    if out.resolve() == Path(bar_dir).resolve():
        raise ValueError("--out-dir must differ from --bar-dir")
    if arm not in TRAINABLE_ARMS:
        raise ValueError(f"--arm {arm!r}: trainable arms are {TRAINABLE_ARMS}")
    spec_arm = ARMS[arm]
    inp = load_inputs(bar_dir)
    pin = _checked_orderer(out, orderer_ckpt)
    filters = {"pairs": pairs, "monitor_every": monitor_every, "init": init}
    n_pairs = PAIRS if pairs is None else pairs
    every = MONITOR_EVERY if monitor_every is None else monitor_every
    size = int(inp.schedule["minibatch_size"]) if minibatch is None else minibatch
    prereg = load_prereg(prereg_path) if prereg_path else None
    probe_doc = json.loads(Path(probe_path).read_text()) if probe_path else None
    if not dry_run:
        issues = [] if prereg is None else prereg_problems(
            prereg, mode=TRAIN, arm=arm, opt_seed=opt_seed, generations=generations,
            minibatch=size, pairs=n_pairs, monitor_every=every)
        issues += (["no probe reading (--probe-reading, EXP-0058 probe.json)"]
                   if probe_doc is None else _probe_doc_problems(probe_doc, arm, size))
        problems = real_run_problems(inp=inp, prereg=prereg, prereg_issues=issues,
                                     source_commit=source_commit, orderer_sha=pin["sha256"],
                                     filters=filters)
        if problems:
            raise ValueError("not a valid ES training run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    elif prereg is not None and prereg_schema_problems(prereg):
        raise ValueError("pre-registration: " + "; ".join(prereg_schema_problems(prereg)))
    base = BaseScorer.load(pin["path"])
    theta0 = spec_arm.initial_theta()
    if init is not None:  # dry runs only: start from a composite's theta
        theta0 = init_theta(init, spec_arm)
    sigma = spec_arm.sigma()
    policy0 = PerOrderPolicy(base, spec_arm, theta0, inp.normaliser, AGENT)
    tpins = training_pins(
        inp=inp, arm=spec_arm, opt_seed=opt_seed, pairs=n_pairs, minibatch=size,
        monitor_every=every, orderer_sha=pin["sha256"],
        prereg_sha=sha256_file(prereg_path) if prereg_path else None,
        probe_sha=sha256_file(probe_path) if probe_path else None,
        init_sha=theta_sha256(theta0),
        per_order={**per_order_pins(policy0)["per_order"], "theta_sha256": None})
    resume_doc = load_checkpoint(resume_from) if resume_from else None
    if resume_doc is not None:
        if resume_doc["training_pins"] != tpins:
            raise ValueError("--resume-from checkpoint has other training pins; refusing "
                             f"(differs at {_pin_diff(resume_doc['training_pins'], tpins)})")
        if int(resume_doc["generation"]) >= generations:
            raise ValueError(f"--resume-from is at generation {resume_doc['generation']}, "
                             f">= --generations {generations}")
    pins = {"training": tpins, "training_sha256": _sha(tpins), "generations": generations,
            "resume_from": None if resume_doc is None else {
                "generation": int(resume_doc["generation"]),
                "content_hash": resume_doc["content_hash"]}}
    source = source_fingerprint(source_commit or None)
    spec, resumed = _open_run(out, pins, {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else TRAIN,
        "acceptance_read": False, "inputs": {"orderer": pin, "bar_dir": str(inp.bar_dir),
                                             "prereg": prereg_path, "probe": probe_path,
                                             "resume_from": resume_from, "init": init},
        "source": source, "decode_pins": decode_pins(CHOICE_V1),
        "pre_registered": {"design": "add-m4-orderer-redesign D6 / D7 (task 5.2)",
                           "sections": registered_sections()}}, workers)

    # state: this dir's latest checkpoint, else the resume source, else generation 0
    own = latest_checkpoint(out)
    if own is not None:
        doc = load_checkpoint(own)
        if doc["training_pins"] != tpins:
            raise AssertionError("own checkpoint pins differ from the run's")
    else:
        doc = resume_doc
    if doc is None:
        state = TrainState(0, theta0.copy(), AdamState.zeros(spec_arm.n_params), [], [])
    else:
        state = TrainState(int(doc["generation"]), np.asarray(doc["theta"], np.float64),
                           AdamState.from_dict(doc["adam"]), list(doc["history"]),
                           list(doc["monitor"]))
    log(f"[order_es] {'DRY RUN ' if dry_run else ''}train {arm} opt seed {opt_seed}: "
        f"from generation {state.generation} to {generations}, {n_pairs} pairs, minibatch "
        f"{size}, {spec_arm.n_params} params" + (" (resumed)" if resumed or doc else ""))
    monitor_we = _fmt_we(inp.schedule["monitor"]["world_episodes"])

    def checkpoint() -> None:
        pol = PerOrderPolicy(base, spec_arm, state.theta, inp.normaliser, AGENT)
        save_checkpoint(checkpoint_path(out, state.generation), {
            "format": CHECKPOINT_FORMAT, "generation": state.generation, "arm": arm,
            "opt_seed": opt_seed, "theta": [float(x) for x in state.theta],
            "theta_sha256": theta_sha256(state.theta), "adam": state.adam.to_dict(),
            "training_pins": tpins, "history": state.history, "monitor": state.monitor,
            "composite": composite_dict(pol)})
        write_csv(out / "generations.csv", GENERATION_COLUMNS, state.history)
        write_csv(out / "monitor_summary.csv", MONITOR_COLUMNS, state.monitor)

    def invalid(reason: str) -> int:
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "reason": reason,
                          "contract_failures": len(read_failures(out)),
                          "generation": state.generation}
        _write_spec(out, spec)
        log(f"[order_es] INVALID at generation {state.generation}: {reason}")
        return CONTRACT_INVALID

    def run_monitor() -> int | None:
        g = state.generation
        if any(int(m["generation"]) == g for m in state.monitor):
            return None
        jobs = [make_job(inp, pin, kind=MONITOR, arm=arm, theta=state.theta, generation=g,
                         member=CENTER, world=w, episode=e, replicate=r)
                for w, e in monitor_we for r in range(ow.MONITOR_REPLICATES)]
        a0_have = next((m["a0_units"] for m in state.monitor), None)
        a0_jobs = [] if a0_have is not None else [
            make_job(inp, pin, kind=MONITOR, arm="A0", theta=np.zeros(0), generation="a0",
                     member=A0_LABEL, world=w, episode=e, replicate=r)
            for w, e in monitor_we for r in range(ow.MONITOR_REPLICATES)]
        try:
            rows, failures = play(out, "monitor.csv", jobs + a0_jobs, workers, run_job_fn,
                                  inp.bar, log)
        except DigestMismatch as exc:
            return invalid(str(exc))
        if failures or len(rows) != len(jobs) + len(a0_jobs):
            return invalid(f"monitor generation {g}: {failures} contract failure(s)")
        if a0_have is None:
            a0 = [rows[j["job_id"]] for j in a0_jobs]
            for d in a0:
                if int(d["replicate"]) == ow.TRAIN_REPLICATE and float(d["units"]) != \
                        inp.bar[(d["world"], int(d["episode"]))].a0_units:
                    exc = ValueError(f"A0 on {d['world']} ep {d['episode']} played "
                                     f"{d['units']} units, the bar's A0 "
                                     f"{inp.bar[(d['world'], int(d['episode']))].a0_units}")
                    append_failure(out, run_failure(A0_REPRODUCTION, exc))
                    return invalid(str(exc))
            a0_have = fmean(float(d["units"]) for d in a0)
        summ = monitor_summary(g, [rows[j["job_id"]] for j in jobs], a0_have, state.monitor)
        state.monitor.append(summ)
        log(f"[order_es] monitor g={g}: center {summ['center_units']} A0 {a0_have:.2f} "
            f"g_tr_raw {summ['g_tr_raw']} improvement_{MONITOR_EVERY} "
            f"{summ[f'improvement_{MONITOR_EVERY}']}")
        checkpoint()
        return None

    if read_failures(out):  # a failure of an earlier session in this dir (gate convention)
        return invalid("contract_failures.csv holds rows from an earlier session")
    if state.generation % every == 0:
        code = run_monitor()
        if code is not None:
            return code
    while state.generation < generations:
        g = state.generation + 1
        t0 = time.perf_counter()
        mb = minibatch_for(inp, opt_seed, g, size)
        eps = noise(opt_seed, g, n_pairs, spec_arm.n_params)
        thetas = member_thetas(state.theta, eps, sigma)
        jobs = generation_jobs(inp, pin, arm, thetas, mb, g)
        try:
            rows, failures = play(out, "members.csv", jobs, workers, run_job_fn, inp.bar, log)
        except DigestMismatch as exc:
            return invalid(str(exc))
        if failures or len(rows) != len(jobs):
            return invalid(f"generation {g}: {failures} contract failure(s), "
                           f"{len(jobs) - len(rows)} missing row(s)")
        scores = {lab: score_member(_member_rows(rows, lab), inp.bar) for lab in thetas}
        ranked = rank_and_estimate(eps, sigma, {k: s.fitness for k, s in scores.items()},
                                   {k: s.disqualified for k, s in scores.items()})
        new_theta, new_adam = adam_ascend(state.theta, ranked.estimate, state.adam)
        fit_rows = [{"generation": g, "member": lab, "sign": 0 if lab == CENTER else
                     (1 if lab[0] == "+" else -1), "pair": "" if lab == CENTER else int(lab[1:]),
                     "fitness": s.fitness, "disqualified": s.disqualified,
                     "nonfinite": s.nonfinite, "rank": ranked.ranks.get(lab),
                     "reasons": ";".join(s.reasons)} for lab, s in scores.items()]
        kept = [d for d in read_csv(out / "member_fitness.csv") if int(d["generation"]) < g]
        write_csv(out / "member_fitness.csv", FITNESS_COLUMNS, kept + [
            {k: "" if v is None else repr(v) if isinstance(v, float) else str(v)
             for k, v in r.items()} for r in fit_rows])  # idempotent on a replayed generation
        fm = [s.fitness for lab, s in scores.items() if lab != CENTER and s.fitness is not None]
        tt = spec_arm.split(new_theta)[1]
        summary = {"generation": g, "minibatch_sha256": _sha(mb),
                   "center_fitness": scores[CENTER].fitness,
                   "member_mean": fmean(fm) if fm else None, "member_max": max(fm, default=None),
                   "member_min": min(fm, default=None),
                   "disqualified": sum(s.disqualified for k, s in scores.items() if k != CENTER),
                   "nonfinite": sum(s.nonfinite for k, s in scores.items() if k != CENTER),
                   "estimate_norm": float(np.linalg.norm(ranked.estimate)),
                   "step_norm": float(np.linalg.norm(new_theta - state.theta)),
                   "theta_sha256": theta_sha256(new_theta),
                   "theta_norm": float(np.linalg.norm(new_theta)), "theta_tau": tt,
                   "tau": math.exp(tt), "seconds": time.perf_counter() - t0, "jobs": len(jobs)}
        state.generation, state.theta, state.adam = g, new_theta, new_adam
        state.history.append(summary)
        checkpoint()
        log(f"[order_es] g={g}/{generations} {arm} s{opt_seed}: center f "
            f"{summary['center_fitness']} members mean {summary['member_mean']} max "
            f"{summary['member_max']} dq {summary['disqualified']} nonfinite "
            f"{summary['nonfinite']} |g| {summary['estimate_norm']:.4g} |dtheta| "
            f"{summary['step_norm']:.4g} tau {summary['tau']:.4f} ({summary['seconds']:.1f} s)")
        if g % every == 0:
            code = run_monitor()
            if code is not None:
                return code
    final = PerOrderPolicy(base, spec_arm, state.theta, inp.normaliser, AGENT)
    save_composite(final, out / "candidate.json")
    spec["status"] = "complete"
    spec["result"] = {"status": "complete", "generation": state.generation,
                      "theta_sha256": theta_sha256(state.theta),
                      "candidate_sha256": sha256_file(str(out / "candidate.json")),
                      "contract_failures": 0, "monitor": state.monitor[-1:]}
    _write_spec(out, spec)
    write_result(out, f"train {arm} opt seed {opt_seed}, generations {state.generation}",
                 dry_run)
    log(f"[order_es] {'DRY RUN ' if dry_run else ''}complete: {arm} s{opt_seed} generation "
        f"{state.generation}, theta {theta_sha256(state.theta)[:12]}")
    return 0


# --- probe ---


PROBE_FITNESS_COLUMNS = ("arm", "batch", "member", "sign", "pair", "fitness", "disqualified",
                         "nonfinite", "rank", "reasons")


def probe_from_rows(fitness: Sequence[Mapping[str, str]], arms: Sequence[str], pairs: int
                    ) -> dict[str, Any]:
    """Recompute every arm's estimates, cosine, variance ratio and the reading from the recorded
    per-member fitness rows (`probe_fitness.csv`) and the recorded noise stream."""
    per_arm: dict[str, Any] = {}
    for arm in arms:
        spec_arm = ARMS[arm]
        eps = probe_noise(arm, pairs, spec_arm.n_params)
        sigma = spec_arm.sigma()
        fits: list[dict[str, float | None]] = []
        ests = []
        for b in (0, 1):
            rows = {d["member"]: d for d in fitness if d["arm"] == arm and int(d["batch"]) == b}
            f = {m: None if rows[m]["fitness"] == "" else float(rows[m]["fitness"])
                 for m in rows}
            w = {m: rows[m]["disqualified"] == "True" for m in rows}
            ests.append(rank_and_estimate(eps, sigma, f, w).estimate)
            fits.append(f)
        per_arm[arm] = {"cosine": cosine(ests[0], ests[1]),
                        "variance_ratio": variance_ratio(fits[0], fits[1], pairs),
                        "estimate_norms": [float(np.linalg.norm(e)) for e in ests],
                        "disqualified": sum(d["disqualified"] == "True" for d in fitness
                                            if d["arm"] == arm),
                        "nonfinite": sum(d["nonfinite"] == "True" for d in fitness
                                         if d["arm"] == arm)}
    return {"arms": per_arm,
            "reading": probe_reading({a: v["cosine"] for a, v in per_arm.items()})}


def run_probe(*, bar_dir: str | Path, out_dir: str | Path, orderer_ckpt: str,
              arms: Sequence[str] = PROBE_ARMS, workers: int = 1,
              prereg_path: str | None = None, pairs: int | None = None,
              source_commit: str = "", dry_run: bool = False,
              run_job_fn: Callable[[Job], dict[str, str]] | None = None,
              log: Callable[[str], None] = _log) -> int:
    """D7's gradient-signal probe into `out_dir` (which may hold the bar run as `bar/`)."""
    out = Path(out_dir)
    run_job_fn = run_job_fn or run_es_job
    if out.resolve() == Path(bar_dir).resolve():
        raise ValueError("--out-dir must differ from --bar-dir (it may contain it)")
    for a in arms:
        if a not in ARMS or not ARMS[a].residual:
            raise ValueError(f"probe arm {a!r} is not a residual arm")
    inp = load_inputs(bar_dir)
    pin = _checked_orderer(out, orderer_ckpt)
    n_pairs = PAIRS if pairs is None else pairs
    mb = probe_batches(inp)
    prereg = load_prereg(prereg_path) if prereg_path else None
    if not dry_run:
        issues = [] if prereg is None else prereg_problems(
            prereg, mode=PROBE, arms=arms, minibatch=len(mb[0]), pairs=n_pairs)
        problems = real_run_problems(inp=inp, prereg=prereg, prereg_issues=issues,
                                     source_commit=source_commit, orderer_sha=pin["sha256"],
                                     filters={"pairs": pairs})
        if problems:
            raise ValueError("not a valid ES probe run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    pins = _json_norm({
        "driver": DRIVER, "mode": PROBE, "arms": list(arms), "pairs": n_pairs,
        "minibatches": [list(map(list, m)) for m in mb], "horizon": inp.horizon,
        "dry_run": dry_run, "hyper": {"h": H_HOLDING, "noise_stream": NOISE_STREAM,
                                      "sigma": {a: theta_sha256(ARMS[a].sigma()) for a in arms}},
        "inputs": {**inp.hashes, "orderer_sha256": pin["sha256"],
                   "prereg_sha256": sha256_file(prereg_path) if prereg_path else None},
        **decode_pins(CHOICE_V1), "producer": PRODUCER,
        "policy": {"decoder_version": DECODER_VERSION,
                   "architecture_version": ARCHITECTURE_VERSION,
                   "normaliser_version": NORMALISER_VERSION},
        "source_tree_sha256": source_fingerprint()["tree_sha256"]})
    spec, resumed = _open_run(out, pins, {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else PROBE,
        "acceptance_read": False, "inputs": {"orderer": pin, "bar_dir": str(inp.bar_dir),
                                             "prereg": prereg_path},
        "source": source_fingerprint(source_commit or None),
        "pre_registered": {"design": "add-m4-orderer-redesign D7 gradient-signal probe",
                           "probe": registered_sections()["probe"]}}, workers)
    log(f"[order_es] {'DRY RUN ' if dry_run else ''}probe {','.join(arms)}: {n_pairs} pairs x "
        f"2 minibatches of {len(mb[0])}" + (" (resumed)" if resumed else ""))
    jobs = []
    for a in arms:
        eps = probe_noise(a, n_pairs, ARMS[a].n_params)
        thetas = member_thetas(ARMS[a].initial_theta(), eps, ARMS[a].sigma())
        for lab, th in thetas.items():
            if lab == CENTER:
                continue
            for b, batch in enumerate(mb):
                for w, e in batch:
                    jobs.append(make_job(inp, pin, kind=PROBE_JOB, arm=a, theta=th,
                                         generation="probe", member=lab, world=w, episode=e,
                                         replicate=ow.TRAIN_REPLICATE,
                                         sign=1 if lab[0] == "+" else -1, pair=int(lab[1:]),
                                         batch=b))
    try:
        rows, failures = play(out, "probe_episodes.csv", jobs, workers, run_job_fn, inp.bar,
                              log)
    except DigestMismatch as exc:
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "reason": str(exc)}
        _write_spec(out, spec)
        return CONTRACT_INVALID
    if failures or len(rows) != len(jobs) or read_failures(out):
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(read_failures(out)),
                          "missing_rows": len(jobs) - len(rows)}
        _write_spec(out, spec)
        log("[order_es] probe INVALID (contract failures or missing rows)")
        return CONTRACT_INVALID
    fit_rows = []
    for a in arms:
        eps = probe_noise(a, n_pairs, ARMS[a].n_params)
        for b in (0, 1):
            mine = {jid: d for jid, d in rows.items()
                    if d["arm"] == a and int(d["batch"]) == b}
            scores = {lab: score_member(_member_rows(mine, lab), inp.bar)
                      for lab, _, _ in member_labels(n_pairs)}
            ranked = rank_and_estimate(eps, ARMS[a].sigma(),
                                       {k: s.fitness for k, s in scores.items()},
                                       {k: s.disqualified for k, s in scores.items()})
            for lab, sign, i in member_labels(n_pairs):
                s = scores[lab]
                fit_rows.append({"arm": a, "batch": b, "member": lab, "sign": sign, "pair": i,
                                 "fitness": "" if s.fitness is None else repr(s.fitness),
                                 "disqualified": str(s.disqualified),
                                 "nonfinite": str(s.nonfinite), "rank": repr(ranked.ranks[lab]),
                                 "reasons": ";".join(s.reasons)})
    write_csv(out / "probe_fitness.csv", PROBE_FITNESS_COLUMNS, fit_rows)
    result = probe_from_rows(read_csv(out / "probe_fitness.csv"), arms, n_pairs)
    doc = {"status": "complete", "dry_run": dry_run, "pairs": n_pairs,
           "minibatches": [list(map(list, m)) for m in mb], **result,
           "noise": {a: {"stream": NOISE_STREAM, "purpose": f"probe|{a}"} for a in arms},
           "pre_registered": registered_sections()["probe"],
           "note": "dry run: not the EXP-0058 reading" if dry_run else ""}
    (out / "probe.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    spec["status"] = "complete"
    spec["result"] = {"status": "complete", "reading": result["reading"],
                      "probe_sha256": sha256_file(str(out / "probe.json")),
                      "contract_failures": 0}
    _write_spec(out, spec)
    write_result(out, f"gradient-signal probe {','.join(arms)}: {result['reading']['reading']}",
                 dry_run)
    log(f"[order_es] {'DRY RUN ' if dry_run else ''}probe: " + ", ".join(
        f"{a} cosine {v['cosine']} variance ratio {v['variance_ratio']}"
        for a, v in result["arms"].items()) + f" -> {result['reading']['reading']}")
    return 0


SKELETON_END = "Conclusions\n\nTODO.\n"


def write_result(out: Path, what: str, dry_run: bool) -> None:
    path = out / "RESULT.md"
    if path.exists() and not path.read_text().endswith(SKELETON_END):
        return
    banner = ("> **Dry run: not an ES result.** Filters or a dry-run bar were used.\n\n"
              if dry_run else "")
    path.write_text(f"# {out.name}\n\n{banner}Step-2 ES (add-m4-orderer-redesign D6 / D7, task "
                    f"5.2), driver `{DRIVER}`: {what}.\n\n## Conclusions\n\nTODO.\n")


# --- CLI ---


def _opt_int(s: str | None) -> int | None:
    return None if s is None else int(s)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=(TRAIN, PROBE), default=TRAIN)
    p.add_argument("--bar-dir", required=True,
                   help="a complete order_es_worlds bar run (bar.csv, training_worlds.json, "
                        "schedule.json, normaliser.json, spec.json)")
    p.add_argument("--orderer-ckpt", required=True, help="EXP-0012 orderer_scorer.pt (base)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arm", help="train: A1, A2 or A2-R2")
    p.add_argument("--arms", default=",".join(PROBE_ARMS), help="probe: comma-separated")
    p.add_argument("--opt-seed", type=int, default=0)
    p.add_argument("--generations", type=int)
    p.add_argument("--resume-from", help="a checkpoint (or run dir) whose pins match")
    p.add_argument("--prereg", default=str(PREREG_PATH) if PREREG_PATH.exists() else None,
                   help="es_prereg_m4_v1.json (frozen by task 5.3)")
    p.add_argument("--probe-reading", help="train: the probe's probe.json (EXP-0058)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--source-commit", default="")
    p.add_argument("--verify-source-commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--pairs", help=f"dry run: (real {PAIRS})")
    p.add_argument("--minibatch", help=f"train: {ow.MINIBATCH} or the weak {WEAK_MINIBATCH}")
    p.add_argument("--monitor-every", help=f"dry run: (real {MONITOR_EVERY})")
    p.add_argument("--init", help="dry run: start from a composite checkpoint's theta")
    a = p.parse_args(argv)
    commit = resolve_verified_commit() if a.verify_source_commit else a.source_commit
    try:
        if a.mode == PROBE:
            return run_probe(bar_dir=a.bar_dir, out_dir=a.out_dir, orderer_ckpt=a.orderer_ckpt,
                             arms=[x for x in a.arms.split(",") if x], workers=a.workers,
                             prereg_path=a.prereg, pairs=_opt_int(a.pairs),
                             source_commit=commit, dry_run=a.dry_run)
        if not a.arm or a.generations is None:
            p.error("--mode train needs --arm and --generations")
        return run_train(bar_dir=a.bar_dir, out_dir=a.out_dir, orderer_ckpt=a.orderer_ckpt,
                         arm=a.arm, opt_seed=a.opt_seed, generations=a.generations,
                         workers=a.workers, prereg_path=a.prereg, probe_path=a.probe_reading,
                         resume_from=a.resume_from, init=a.init, pairs=_opt_int(a.pairs),
                         minibatch=_opt_int(a.minibatch),
                         monitor_every=_opt_int(a.monitor_every), source_commit=commit,
                         dry_run=a.dry_run)
    except (ContractRunError, DigestMismatch):  # recorded in contract_failures.csv
        return CONTRACT_INVALID


if __name__ == "__main__":
    sys.exit(main())
