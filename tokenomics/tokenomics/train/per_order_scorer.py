"""Per-order residual orderer (add-m4-orderer-redesign design D5, task 4.3; ADR-0058 /
ADR-0059).

Step 2 of the M4 ordering ladder: every order goes whole to one eligible producer, drawn from
softmax(s / tau) with the decoder check's keyed uniform (design D3), where

    s_i(o)  = b_i + Delta_theta(i, o)
    b_i     = the frozen EXP-0012 scorer's score (one forward per stream per cycle)
    phi_i   = normalised [shared row (31) | agent-local history (9) + has-completed (1)
                          | own-turn assigned (2)]
    psi_o   = normalised [product (9) | quantity, tokens / quantity, TTL remaining, kind]
    Delta   = B * tanh((w . phi_i + sum_r (u_r . phi_i)(v_r . psi_o)) / B),  B = 2.0
    tau     = exp(theta_tau)

Why a bounded residual over a frozen base: theta = 0 is exactly the decoder check's TS
(EXP-0057), so ES (design D6) starts from the measured policy and every arm nests it. The
agent id only keys the draw (ADR-0058): one portable model, agents differ only through their
own history, their own orders and their sampling streams. The draw key carries no ES member,
so the antithetic members of one generation see identical uniforms (D6 common random
numbers).

Choices are emitted as `choice-v1` one-hot shares (ADR-0059): no split action exists. The
policy refuses a `logits-v1` env, and refuses to emit a non-finite score or share (it never
hands NaN to decode).

Latency before any completion (task 4.3 supervisor addition 1): the history reads 0.0 there,
which a linear weight would read as "fastest". phi carries a has-completed indicator beside
the 9 history features, so a linear term can tell "no completion yet" from "completes in the
routing cycle". An imputation was rejected on data: on the fixture world 83 % of completed
latencies are 0, so the fitted median is 0.0, the default itself. This makes phi 43 wide:
A1 44 parameters, A2 (R = 4) 44 + 4 (43 + 13) = 268 (design D5, as amended by task 4.3).

Size-scaled counts (supervisor decision 3, task 5.2): `eligible_producer_count` is
log1p-transformed before the normaliser, in the policy and in the fit rows alike, so a
12-producer fit does not pin 42-producer products at the clip (`LOG1P_PSI`; the normaliser and
the architecture versions are v2, so v1 composites and normaliser files are refused).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..domain.order import ORDER_TTL
from ..domain.system import order_kind
from ..env.agent_history import FEATURES as HISTORY_FEATURES
from ..env.agent_history import attach_agent_history, env_agent_id
from ..env.spaces import CHOICE_V1, decode_pins
from ..eval.routing_diagnostics import (_slots, draw_key, inverse_cdf, keyed_uniform,
                                        makeable_weights, softmax_weights)
from .orderer_scorer import (_ORDERER_PRODUCER_ROW_FEATURES, _ORDERER_PRODUCT_FEATURES, _BOUND,
                             _producer_row_features, _product_features)

if TYPE_CHECKING:
    from ..domain.config import Configuration
    from ..env import TokenomicsAECEnv
    from .orderer_scorer import OrdererScoreModel

DECODER_VERSION = "per-order-choice-v1"
ARCHITECTURE_VERSION = "orderer-per-order-residual-v2"  # v2: log1p inputs (task 5.2)
CHECKPOINT_FORMAT = "per-order-composite-v1"
NORMALISER_VERSION = "median-iqr-log1p-v2"  # v2: fitted on the log1p-transformed inputs
BASE_DECODER_VERSION = "allocate_suborders-v1"
BASE_ARCHITECTURE_VERSION = "orderer-producer-row-scorer-v1"

B = 2.0  # residual bound, logits (D5)
SIGMA = 0.05  # per-parameter ES scale for w, u, v and theta_tau (D5)
U_INIT_STD = 0.1
U_INIT_SEED = 0  # fixed seed of A2's u init (D5); recorded in every checkpoint
CLIP = 5.0  # normalised features are clipped to +-CLIP (bounds one coordinate's pull)

HAS_COMPLETED = "has_completed"  # 1 once the pair has a completed sub-order (task 4.3)
OWN_TURN_FEATURES = ("assigned_units", "assigned_orders")
ORDER_FEATURES = ("quantity", "tokens_per_unit", "ttl_remaining", "kind")
PHI_FEATURES: tuple[str, ...] = (tuple(_ORDERER_PRODUCER_ROW_FEATURES) + tuple(HISTORY_FEATURES)
                                 + (HAS_COMPLETED,) + OWN_TURN_FEATURES)
PSI_FEATURES: tuple[str, ...] = tuple(_ORDERER_PRODUCT_FEATURES) + ORDER_FEATURES
N_PHI = len(PHI_FEATURES)  # 31 + 10 + 2 = 43
N_PSI = len(PSI_FEATURES)  # 9 + 4 = 13
# Inputs whose range is a function of the producer count, log1p-transformed before the
# normaliser (fit rows included; supervisor decision 3, task 5.2). Only E, the producers
# declaring the order's product, qualifies: at 12 producers E is 2 or 4 by construction, at 42
# also 7 or 14, so a 12-producer fit pins it at the clip on every 42-producer E = 14 product.
# The other config counts (depth, recipe inputs, capacity types) have the same values at 12 and
# 42 on the atlas worlds, and the unit volumes (load, demand, history EMAs, own-turn units) are
# demand-driven, a range the 12-producer training worlds' demand levers already span.
LOG1P_PHI: tuple[str, ...] = ()
LOG1P_PSI: tuple[str, ...] = ("eligible_producer_count",)
# Registered scale floor of each transformed count (review round 1): the spread between E's two
# 12-producer values, log1p(4) - log1p(2). The fit runs on decision rows, whose E = 2 share is not
# fixed; outside ~[0.25, 0.75] the IQR is 0 and the std (or a ~0 std) would pin E = 14 at the
# clip. With the floor, |z| <= (log1p(14) - log1p(1)) / floor = 3.9 for any median >= log1p(1).
SCALE_FLOORS: dict[str, float] = {"psi.eligible_producer_count": math.log1p(4.0) - math.log1p(2.0)}
_LOG1P_PHI_COLS = np.asarray([PHI_FEATURES.index(n) for n in LOG1P_PHI], dtype=np.intp)
_LOG1P_PSI_COLS = np.asarray([PSI_FEATURES.index(n) for n in LOG1P_PSI], dtype=np.intp)
_SHARED = slice(0, len(_ORDERER_PRODUCER_ROW_FEATURES))
_HIST = slice(_SHARED.stop, _SHARED.stop + len(HISTORY_FEATURES) + 1)  # history + indicator
LATENCY_COL = _HIST.start + HISTORY_FEATURES.index("completion_latency")
HAS_COMPLETED_COL = _HIST.stop - 1

AGENT = "agent"
SHARED_ONLY = "shared_only"  # D9 ablation: history = mean over the product's agents
HISTORY_MODES = (AGENT, SHARED_ONLY)
# The frozen base the residual corrects (task 7.6, user decision "Re-base on MWW"): `ts` = the
# EXP-0012 trained scores, drawn as TS (D5, pre-registration v1 / v2); `mww` = MWW's keyed
# whole-order choice (MWWk: weights = makeable units). The residual is a logit add on the
# base's log-weights, p ~ exp((log w + Delta) / tau) = w ** (1 / tau) * exp(Delta / tau),
# computed multiplicatively so theta = 0 gives MWWk's weights bit for bit (A0 == MWWk).
BASE_TS, BASE_MWW = "ts", "mww"
BASE_KINDS = (BASE_TS, BASE_MWW)


# --- arms and theta layout ---


@dataclass(frozen=True)
class ArmSpec:
    """A0: no residual (theta empty, tau = 1). A1: w (43) + theta_tau. A2: A1 + rank-R
    bilinear u_r (43), v_r (13). Layout: [w | theta_tau | U (R x 43, row-major) | V (R x 13)],
    so A1's theta is a prefix of A2's."""

    name: str
    residual: bool
    rank: int = 0

    @property
    def n_params(self) -> int:
        return 0 if not self.residual else N_PHI + 1 + self.rank * (N_PHI + N_PSI)

    def split(self, theta: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """(w, theta_tau, U (R, 43), V (R, 13)) views of a flat theta."""
        t = np.asarray(theta, dtype=np.float64)
        if t.shape != (self.n_params,):
            raise ValueError(f"arm {self.name}: theta has shape {t.shape}, "
                             f"expected ({self.n_params},)")
        if not self.residual:
            return np.zeros(N_PHI), 0.0, np.zeros((0, N_PHI)), np.zeros((0, N_PSI))
        w, tt = t[:N_PHI], float(t[N_PHI])
        o = N_PHI + 1
        u = t[o:o + self.rank * N_PHI].reshape(self.rank, N_PHI)
        o += self.rank * N_PHI
        v = t[o:o + self.rank * N_PSI].reshape(self.rank, N_PSI)
        return w, tt, u, v

    def sigma(self) -> np.ndarray:
        """Per-parameter ES scale sigma_j (D5): 0.05 on w, u, v (normalised feature units) and
        on theta_tau (log tau). Stored with the arm in every checkpoint."""
        return np.full(self.n_params, SIGMA, dtype=np.float64)

    def initial_theta(self) -> np.ndarray:
        """w = 0, theta_tau = 0, u ~ N(0, 0.1) from U_INIT_SEED, v = 0: Delta starts at 0."""
        theta = np.zeros(self.n_params, dtype=np.float64)
        if self.residual and self.rank:
            rng = np.random.default_rng(U_INIT_SEED)
            o = N_PHI + 1
            theta[o:o + self.rank * N_PHI] = rng.normal(0.0, U_INIT_STD, self.rank * N_PHI)
        return theta

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "residual": self.residual, "rank": self.rank,
                "n_params": self.n_params}

ARMS: dict[str, ArmSpec] = {
    "A0": ArmSpec("A0", residual=False),
    "A1": ArmSpec("A1", residual=True, rank=0),
    "A2": ArmSpec("A2", residual=True, rank=4),
    "A2-R2": ArmSpec("A2-R2", residual=True, rank=2),  # D7: R = 2 on a low-signal probe
}


# --- input transform (before the normaliser) ---


def transform_phi(phi_raw: np.ndarray) -> np.ndarray:
    """log1p on `LOG1P_PHI` (none at present); a copy, the input is never written."""
    x = np.array(phi_raw, dtype=np.float64, copy=True)
    if _LOG1P_PHI_COLS.size:
        x[..., _LOG1P_PHI_COLS] = np.log1p(x[..., _LOG1P_PHI_COLS])
    return x


def transform_psi(psi_raw: np.ndarray) -> np.ndarray:
    """log1p on `LOG1P_PSI` (`eligible_producer_count`); a copy, the input is never written."""
    x = np.array(psi_raw, dtype=np.float64, copy=True)
    if _LOG1P_PSI_COLS.size:
        x[..., _LOG1P_PSI_COLS] = np.log1p(x[..., _LOG1P_PSI_COLS])
    return x


def input_transforms() -> dict[str, str]:
    """feature -> transform, as recorded in every normaliser (hashed) and pre-registration."""
    return {**{f"phi.{n}": "log1p" for n in LOG1P_PHI},
            **{f"psi.{n}": "log1p" for n in LOG1P_PSI}}


def apply_scale_floors(phi_scale: Sequence[float], psi_scale: Sequence[float]
                       ) -> tuple[list[float], list[float]]:
    """max(fitted scale, registered floor) on the floored columns (`SCALE_FLOORS`)."""
    ps, qs = [float(x) for x in phi_scale], [float(x) for x in psi_scale]
    for key, floor in SCALE_FLOORS.items():
        block, name = key.split(".", 1)
        target, names = (ps, PHI_FEATURES) if block == "phi" else (qs, PSI_FEATURES)
        j = names.index(name)
        target[j] = max(target[j], floor)
    return ps, qs


# --- normaliser ---


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class FeatureNormaliser:
    """Fixed per-feature median / scale for phi (43) and psi (13), fitted once (`fit`) and then
    frozen (D5); `apply_*` are pure. Scale = IQR; IQR = 0 -> the column's standard deviation;
    that 0 too -> 1.0 (the zero-spread rule). Normalised values are clipped to +-clip. Hashed
    (`sha256`) over its canonical JSON. Its inputs are the transformed rows (`transform_phi` /
    `transform_psi`, recorded as `transforms`): the policy and the fit rows both transform
    before normalising."""

    phi_median: tuple[float, ...]
    phi_scale: tuple[float, ...]
    psi_median: tuple[float, ...]
    psi_scale: tuple[float, ...]
    clip: float = CLIP
    n_rows: int = 0
    version: str = NORMALISER_VERSION
    phi_features: tuple[str, ...] = PHI_FEATURES
    psi_features: tuple[str, ...] = PSI_FEATURES
    transforms: tuple[tuple[str, str], ...] = tuple(sorted(input_transforms().items()))

    def __post_init__(self) -> None:
        if (len(self.phi_median), len(self.phi_scale)) != (N_PHI, N_PHI) or \
                (len(self.psi_median), len(self.psi_scale)) != (N_PSI, N_PSI):
            raise ValueError("normaliser dimensions do not match phi (43) / psi (13)")
        vals = self.phi_median + self.phi_scale + self.psi_median + self.psi_scale + (self.clip,)
        if not all(math.isfinite(x) for x in vals) or min(self.phi_scale + self.psi_scale) <= 0:
            raise ValueError("normaliser medians / scales must be finite, scales > 0")
        if dict(self.transforms) != input_transforms():
            raise ValueError("normaliser input transforms differ from this module's")

    @staticmethod
    def _column_stats(x: np.ndarray) -> tuple[list[float], list[float]]:
        med, scale = [], []
        for j in range(x.shape[1]):
            col = x[:, j]
            q25, q50, q75 = np.percentile(col, [25.0, 50.0, 75.0])
            iqr = float(q75 - q25)
            # all-equal values have zero spread exactly: np.std of a constant that is not a
            # binary fraction (e.g. log1p(2)) is ~1e-16, which would become the scale
            sd = 0.0 if np.ptp(col) == 0 else float(np.std(col))
            med.append(float(q50))
            scale.append(iqr if iqr > 0 else (sd if sd > 0 else 1.0))
        return med, scale

    @classmethod
    def fit(cls, rows: "RawRows", clip: float = CLIP) -> "FeatureNormaliser":
        """Fit on raw rows (`RawRows`: phi per (decision, eligible producer), psi per
        decision). Task 5.1 fits it on A0's bar-precompute rows (notes/4.3.md)."""
        phi, psi = rows.arrays()
        if not len(phi) or not len(psi):
            raise ValueError("normaliser fit needs at least one phi and one psi row")
        pm, ps = cls._column_stats(phi)
        qm, qs = cls._column_stats(psi)
        ps, qs = apply_scale_floors(ps, qs)
        return cls(tuple(pm), tuple(ps), tuple(qm), tuple(qs), float(clip), int(len(psi)))

    @classmethod
    def identity(cls) -> "FeatureNormaliser":
        """Median 0, scale 1 (tests; never a deployed normaliser)."""
        return cls((0.0,) * N_PHI, (1.0,) * N_PHI, (0.0,) * N_PSI, (1.0,) * N_PSI)

    def apply_phi(self, phi_raw: np.ndarray) -> np.ndarray:
        z = (np.asarray(phi_raw, dtype=np.float64) - np.asarray(self.phi_median)) / np.asarray(
            self.phi_scale)
        return np.clip(z, -self.clip, self.clip)

    def apply_psi(self, psi_raw: np.ndarray) -> np.ndarray:
        z = (np.asarray(psi_raw, dtype=np.float64) - np.asarray(self.psi_median)) / np.asarray(
            self.psi_scale)
        return np.clip(z, -self.clip, self.clip)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "phi_features": list(self.phi_features),
                "psi_features": list(self.psi_features), "phi_median": list(self.phi_median),
                "phi_scale": list(self.phi_scale), "psi_median": list(self.psi_median),
                "psi_scale": list(self.psi_scale), "clip": self.clip, "n_rows": self.n_rows,
                "transforms": dict(self.transforms), "scale_floors": dict(SCALE_FLOORS)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureNormaliser":
        if d.get("version") != NORMALISER_VERSION:
            raise ValueError(f"normaliser version {d.get('version')!r} != {NORMALISER_VERSION}")
        if tuple(d["phi_features"]) != PHI_FEATURES or tuple(d["psi_features"]) != PSI_FEATURES:
            raise ValueError("normaliser feature names differ from this module's phi / psi")
        if d.get("transforms") != input_transforms():
            raise ValueError(f"normaliser input transforms {d.get('transforms')!r} differ from "
                             f"this module's {input_transforms()!r}")
        if d.get("scale_floors") != SCALE_FLOORS:
            raise ValueError(f"normaliser scale floors {d.get('scale_floors')!r} differ from "
                             f"this module's {SCALE_FLOORS!r}")
        ps, qs = list(map(float, d["phi_scale"])), list(map(float, d["psi_scale"]))
        if apply_scale_floors(ps, qs) != (ps, qs):
            raise ValueError("normaliser scale below its registered floor (not this fit's)")
        return cls(tuple(map(float, d["phi_median"])), tuple(ps),
                   tuple(map(float, d["psi_median"])), tuple(qs),
                   float(d["clip"]), int(d["n_rows"]))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict())).hexdigest()


@dataclass
class RawRows:
    """Unnormalised inputs collected while playing, the normaliser's fit rows: one phi row per
    (decision, eligible producer), one psi row per decision, after `transform_phi` /
    `transform_psi` (the normaliser's input space)."""

    phi: list[np.ndarray] = field(default_factory=list)
    psi: list[np.ndarray] = field(default_factory=list)

    def add(self, phi: np.ndarray, psi: np.ndarray) -> None:
        self.phi.append(np.array(phi, dtype=np.float64).reshape(-1, N_PHI))
        self.psi.append(np.array(psi, dtype=np.float64).reshape(1, N_PSI))

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.psi:
            return np.zeros((0, N_PHI)), np.zeros((0, N_PSI))
        return np.vstack(self.phi), np.vstack(self.psi)


# --- raw inputs ---


def shared_rows(view: Any, pids: Sequence[str]) -> np.ndarray:
    """(P, 31) producer rows, sanitised exactly as the base scorer's input (NaN -> 0,
    +-inf -> +-10): the rows every agent's base score reads."""
    rows = [_producer_row_features(view.producers[p], view.producer_products[p]) for p in pids]
    return np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=_BOUND,
                         neginf=-_BOUND)


def history_rows(env: "TokenomicsAECEnv", aid: str, pids: Sequence[str]) -> np.ndarray:
    """(P, 10): the agent-local history's 9 features and the has-completed indicator (1 once
    the pair has a completed sub-order this episode; its latency is defined only then)."""
    h = env.agent_history  # type: ignore[attr-defined]
    done = [[1.0 if h.totals(aid, p)["completed"] > 0 else 0.0] for p in pids]
    return np.concatenate([h.rows(aid, pids), np.asarray(done, dtype=np.float64)], axis=1)


def order_features(env: "TokenomicsAECEnv", oid: str, info: Any) -> list[float]:
    """[quantity, tokens / quantity, TTL remaining, kind]: the order as the agent routes it
    (ADR-0058 item 1). kind = 1 for a consumer order, 0 for a producer (replenishment /
    investment) order."""
    order = env.system.orders[oid]
    q = int(info.quantity)
    return [float(q), float(info.tokens) / max(1, q),
            float(max(0, ORDER_TTL - order.cycle_counter)),
            1.0 if order_kind(order) == "consumer" else 0.0]


# --- policy ---


class NonFiniteChoice(ValueError):
    """A non-finite score, temperature or share reached the draw: refused before decode
    (carry-forward: never emit NaN, never hang)."""


@dataclass
class Decision:
    """One routed order's inputs and draw (tests, row collection)."""

    slot: int
    order_id: str
    quantity: int
    eligible: list[int]
    b: np.ndarray
    phi: np.ndarray | None
    psi: np.ndarray | None
    own_turn: np.ndarray
    delta: np.ndarray
    u: float
    chosen: int  # index into spec.producer_ids


@dataclass(frozen=True)
class BaseScorer:
    """The frozen EXP-0012 scorer as the policy's base component, with its checkpoint's
    SHA-256 (`allocate_suborders-v1` versioned; loads unchanged, never trained here)."""

    model: "OrdererScoreModel"
    sha256: str
    path: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "BaseScorer":
        from ..eval.provenance import sha256_file
        from .orderer_scorer import load_scorer

        model = load_scorer(str(path))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, sha256_file(str(path)), str(path))


class PerOrderPolicy:
    """The D5 policy as an ordering `Policy` (callable (env, agent) -> flat choice-v1 action).

    Needs a `choice-v1` env with agent-local history attached (`per_order_env`) and a draw key
    (`set_draw_key`). Stateless across calls: everything it reads lives on the env."""

    def __init__(self, base: BaseScorer, arm: ArmSpec | str, theta: np.ndarray | None = None,
                 normaliser: FeatureNormaliser | None = None, history_mode: str = AGENT) -> None:
        from .orderer_scorer import scorer_ordering_policy

        self.base = base
        self.arm = ARMS[arm] if isinstance(arm, str) else arm
        theta = self.arm.initial_theta() if theta is None else np.asarray(theta, np.float64)
        self.w, self.theta_tau, self.u, self.v = self.arm.split(theta)
        self.theta = np.array(theta, dtype=np.float64, copy=True)
        if history_mode not in HISTORY_MODES:
            raise ValueError(f"history_mode {history_mode!r} not in {HISTORY_MODES}")
        if normaliser is None and self.arm.residual:
            raise ValueError(f"arm {self.arm.name} ({history_mode}) needs a fitted normaliser")
        self.normaliser = normaliser
        self.history_mode = history_mode
        self._base_policy = scorer_ordering_policy(base.model)
        self.sink: RawRows | None = None  # set to collect normaliser fit rows
        self.record: list[Decision] | None = None  # set to record every decision

    # the draw's inputs, never reading other agents' current-cycle routing (ADR-0058)
    def _history(self, env: "TokenomicsAECEnv", aid: str, pids: Sequence[str]) -> np.ndarray:
        if self.history_mode == AGENT:
            return history_rows(env, aid, pids)
        # shared-only (D9): the feature-wise mean over every stream agent of the product,
        # never-routed ones included (their rows are the prior), so identical across the
        # product's agents; never_routed / has_completed become fractions
        product = env.system.stream_products[aid]
        agents = sorted(a for a, x in env.system.stream_products.items() if x == product)
        return np.mean([history_rows(env, a, pids) for a in agents], axis=0)

    def decide(self, env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        if getattr(env, "ordering_decode", None) != CHOICE_V1:
            raise ValueError(f"{DECODER_VERSION} policy needs ordering_decode={CHOICE_V1!r}, "
                             f"env has {getattr(env, 'ordering_decode', None)!r}")
        if not hasattr(env, "agent_history"):
            raise ValueError("per-order policy needs attach_agent_history(env) (D5)")
        with np.errstate(all="ignore"):
            tau = float(np.exp(self.theta_tau)) if self.arm.residual else 1.0
        if not (math.isfinite(tau) and tau > 0):
            raise NonFiniteChoice(f"temperature exp({self.theta_tau}) = {tau}")
        sp = env.spec_
        key = draw_key(env)
        action = np.zeros((sp.k_stream, sp.num_producers), dtype=np.float32)
        slots = _slots(env, agent)
        if not any(elig for *_, elig in slots):
            return action.flatten()  # nothing to route: no base forward (draws unchanged)
        cycle = env.system.cycle_counter
        aid = env_agent_id(env, agent)
        pids = list(sp.producer_ids)
        view = env.system.product_stream_view(aid)
        if sorted(view.producers) != pids:
            raise ValueError("stream view producers differ from spec.producer_ids")
        # one base forward per stream per cycle, exactly the scores TS draws from
        b = np.asarray(self._base_policy(env, agent), dtype=np.float64).reshape(
            sp.k_stream, sp.num_producers)[0]
        need_inputs = self.arm.residual or self.sink is not None or self.record is not None
        own = np.zeros((len(pids), 2), dtype=np.float64)
        if need_inputs:
            shared = shared_rows(view, pids)
            hist = self._history(env, aid, pids)
            product = np.asarray(_product_features(view.product_stats), dtype=np.float64)
            incoming = env.system.ordering_agents[aid].incoming
        for k, oid, q, elig in slots:
            if not elig:
                continue
            phi = psi = None
            delta = np.zeros(len(elig))
            if need_inputs:
                raw = transform_phi(np.concatenate([shared, hist, own], axis=1)[elig])
                psi_raw = transform_psi(np.concatenate([product,
                                                        order_features(env, oid, incoming[oid])]))
                if self.sink is not None:
                    self.sink.add(raw, psi_raw)
                if self.normaliser is not None:
                    phi = self.normaliser.apply_phi(raw)
                    psi = self.normaliser.apply_psi(psi_raw)
                if self.arm.residual:
                    assert phi is not None and psi is not None
                    with np.errstate(all="ignore"):
                        z = phi @ self.w + (phi @ self.u.T) @ (self.v @ psi)
                        delta = B * np.tanh(z / B)
            with np.errstate(all="ignore"):
                s = (b[elig] + delta) / tau
                weights = softmax_weights(s)
            if not (np.all(np.isfinite(s)) and np.all(np.isfinite(weights))
                    and weights.sum() > 0):
                raise NonFiniteChoice(f"{agent} cycle {cycle} slot {k}: non-finite score / "
                                      f"share (theta has non-finite effect)")
            u = keyed_uniform(key, agent, cycle, k)
            chosen = elig[inverse_cdf(weights, u)]
            action[k, chosen] = 1.0  # whole order: one-hot share
            if self.record is not None:
                self.record.append(Decision(k, oid, q, list(elig), b[elig].copy(), phi, psi,
                                            own[elig].copy(), delta, u, chosen))
            own[chosen, 0] += q
            own[chosen, 1] += 1.0
        return action.flatten()

    __call__ = decide


class MWWBasePolicy(PerOrderPolicy):
    """The re-based policy (task 7.6, user decision "Re-base on MWW"; pre-registration v3): the
    frozen base is MWW's keyed whole-order choice (MWWk) instead of TS. A subclass, so
    `PerOrderPolicy`'s source (inside the v1 / v2 normaliser fitting-code hash) is unchanged;
    v3 pins this class's source on its own (`mww_base_code_sha256`). The residual is a logit
    add on MWW's log-weights, p ~ w ** (1 / tau) * exp((Delta - max Delta) / tau): at theta = 0
    the weights are MWWk's bit for bit (A0 == MWWk), and a producer MWW gives weight 0 is never
    drawn while another eligible producer is makeable."""

    base_kind = BASE_MWW

    def __init__(self, base: BaseScorer, arm: ArmSpec | str, theta: np.ndarray | None = None,
                 normaliser: FeatureNormaliser | None = None, history_mode: str = AGENT) -> None:
        super().__init__(base, arm, theta, normaliser, history_mode)
        self._base_policy = None  # no scorer forward: the base weights are MWW's, per slot

    def decide(self, env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        if getattr(env, "ordering_decode", None) != CHOICE_V1:
            raise ValueError(f"{DECODER_VERSION} policy needs ordering_decode={CHOICE_V1!r}, "
                             f"env has {getattr(env, 'ordering_decode', None)!r}")
        if not hasattr(env, "agent_history"):
            raise ValueError("per-order policy needs attach_agent_history(env) (D5)")
        with np.errstate(all="ignore"):
            tau = float(np.exp(self.theta_tau)) if self.arm.residual else 1.0
        if not (math.isfinite(tau) and tau > 0):
            raise NonFiniteChoice(f"temperature exp({self.theta_tau}) = {tau}")
        sp = env.spec_
        key = draw_key(env)
        action = np.zeros((sp.k_stream, sp.num_producers), dtype=np.float32)
        slots = _slots(env, agent)
        if not any(elig for *_, elig in slots):
            return action.flatten()  # nothing to route: no base forward (draws unchanged)
        cycle = env.system.cycle_counter
        aid = env_agent_id(env, agent)
        pids = list(sp.producer_ids)
        view = env.system.product_stream_view(aid)
        if sorted(view.producers) != pids:
            raise ValueError("stream view producers differ from spec.producer_ids")
        need_inputs = self.arm.residual or self.sink is not None or self.record is not None
        own = np.zeros((len(pids), 2), dtype=np.float64)
        if need_inputs:
            shared = shared_rows(view, pids)
            hist = self._history(env, aid, pids)
            product = np.asarray(_product_features(view.product_stats), dtype=np.float64)
            incoming = env.system.ordering_agents[aid].incoming
        for k, oid, q, elig in slots:
            if not elig:
                continue
            phi = psi = None
            delta = np.zeros(len(elig))
            if need_inputs:
                raw = transform_phi(np.concatenate([shared, hist, own], axis=1)[elig])
                psi_raw = transform_psi(np.concatenate([product,
                                                        order_features(env, oid, incoming[oid])]))
                if self.sink is not None:
                    self.sink.add(raw, psi_raw)
                if self.normaliser is not None:
                    phi = self.normaliser.apply_phi(raw)
                    psi = self.normaliser.apply_psi(psi_raw)
                if self.arm.residual:
                    assert phi is not None and psi is not None
                    with np.errstate(all="ignore"):
                        z = phi @ self.w + (phi @ self.u.T) @ (self.v @ psi)
                        delta = B * np.tanh(z / B)
            with np.errstate(all="ignore"):  # MWWk's weights, per slot, as MWWk reads them
                bw = makeable_weights(env, oid, elig)
                base_k = np.log(bw)
                s = (delta - delta.max()) / tau
                weights = bw ** (1.0 / tau) * np.exp(s)
            if not (np.all(np.isfinite(s)) and np.all(np.isfinite(weights))
                    and weights.sum() > 0):
                raise NonFiniteChoice(f"{agent} cycle {cycle} slot {k}: non-finite score / "
                                      f"share (theta has non-finite effect)")
            u = keyed_uniform(key, agent, cycle, k)
            chosen = elig[inverse_cdf(weights, u)]
            action[k, chosen] = 1.0  # whole order: one-hot share
            if self.record is not None:
                self.record.append(Decision(k, oid, q, list(elig), np.array(base_k), phi, psi,
                                            own[elig].copy(), delta, u, chosen))
            own[chosen, 0] += q
            own[chosen, 1] += 1.0
        return action.flatten()

    __call__ = decide


def make_policy(base: BaseScorer, arm: ArmSpec | str, theta: np.ndarray | None = None,
                normaliser: FeatureNormaliser | None = None, history_mode: str = AGENT,
                base_kind: str = BASE_TS) -> PerOrderPolicy:
    """The per-order policy on its frozen base: `ts` = `PerOrderPolicy` (v1 / v2), `mww` =
    `MWWBasePolicy` (v3, task 7.6)."""
    if base_kind not in BASE_KINDS:
        raise ValueError(f"base_kind {base_kind!r} not in {BASE_KINDS}")
    cls = MWWBasePolicy if base_kind == BASE_MWW else PerOrderPolicy
    return cls(base, arm, theta, normaliser, history_mode)


def base_kind_of(policy: PerOrderPolicy) -> str:
    return getattr(policy, "base_kind", BASE_TS)


# --- env helper and pins (carry-forward: choice-v1 drivers pass the decode and pin it) ---


def per_order_env(config: "Configuration", horizon: int) -> "TokenomicsAECEnv":
    """The joint-stack env under `choice-v1`, agent-local history attached."""
    from ..eval.joint_stack import joint_stack_env

    env = joint_stack_env(config, horizon, ordering_decode=CHOICE_V1)
    attach_agent_history(env)
    return env


def per_order_pins(policy: PerOrderPolicy) -> dict[str, Any]:
    """Pins a driver merges into its `spec.json` (resume compares them): the decode pin plus
    the composite's identity."""
    return {**decode_pins(CHOICE_V1), "per_order": {
        "decoder_version": DECODER_VERSION, "architecture_version": ARCHITECTURE_VERSION,
        "arm": policy.arm.name, "theta_sha256": theta_sha256(policy.theta),
        "base_sha256": policy.base.sha256, "history_mode": policy.history_mode,
        "normaliser_sha256": policy.normaliser.sha256 if policy.normaliser else None,
        **({"base_kind": BASE_MWW} if base_kind_of(policy) == BASE_MWW else {})}}


# --- composite checkpoint ---


def theta_sha256(theta: np.ndarray) -> str:
    return hashlib.sha256(_canonical([float(x) for x in np.asarray(theta, np.float64)])
                          ).hexdigest()


def composite_dict(policy: PerOrderPolicy) -> dict[str, Any]:
    nm = policy.normaliser
    return {
        "format": CHECKPOINT_FORMAT, "decoder_version": DECODER_VERSION,
        "architecture_version": ARCHITECTURE_VERSION, "ordering_decode": CHOICE_V1,
        "arm": policy.arm.to_dict(), "bound": B, "u_init": {"seed": U_INIT_SEED,
                                                             "std": U_INIT_STD},
        "theta": [float(x) for x in policy.theta], "theta_sha256": theta_sha256(policy.theta),
        "sigma": [float(x) for x in policy.arm.sigma()],
        "base": {"sha256": policy.base.sha256, "decoder_version": BASE_DECODER_VERSION,
                 "architecture_version": BASE_ARCHITECTURE_VERSION,
                 **({"kind": BASE_MWW} if base_kind_of(policy) == BASE_MWW else {})},
        "normaliser": None if nm is None else nm.to_dict(),
        "normaliser_sha256": None if nm is None else nm.sha256,
    }


def save_composite(policy: PerOrderPolicy, path: str | Path) -> None:
    Path(path).write_text(json.dumps(composite_dict(policy), indent=1, sort_keys=True,
                                     allow_nan=False) + "\n")


def load_composite(path: str | Path, base_path: str | Path, env: "TokenomicsAECEnv",
                   history_mode: str = AGENT) -> PerOrderPolicy:
    """Load a composite into `env`'s decode. Refused (naming the decoder version) in a
    `logits-v1` env, and refused for anything that is not a composite of this version, a base
    whose SHA-256 differs, a normaliser whose hash differs, or theta of the wrong size."""
    decode = getattr(env, "ordering_decode", None)
    if decode != CHOICE_V1:
        raise ValueError(f"composite checkpoint {DECODER_VERSION} refused in an "
                         f"ordering_decode={decode!r} env (needs {CHOICE_V1!r}, ADR-0059)")
    try:
        ckpt = json.loads(Path(path).read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint ({exc.__class__.__name__}"
                         f"; a legacy scorer is not a deployed {CHOICE_V1} policy)") from None
    if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    for k, want in (("decoder_version", DECODER_VERSION),
                    ("architecture_version", ARCHITECTURE_VERSION),
                    ("ordering_decode", CHOICE_V1)):
        if ckpt.get(k) != want:
            raise ValueError(f"composite {k} {ckpt.get(k)!r} != {want!r}")
    base = BaseScorer.load(base_path)
    if base.sha256 != ckpt["base"]["sha256"]:
        raise ValueError(f"base {base_path} has sha256 {base.sha256}, composite pins "
                         f"{ckpt['base']['sha256']}")
    arm = ArmSpec(str(ckpt["arm"]["name"]), bool(ckpt["arm"]["residual"]),
                  int(ckpt["arm"]["rank"]))
    if ARMS.get(arm.name) != arm or ckpt["arm"].get("n_params") != arm.n_params:
        raise ValueError(f"composite arm {ckpt['arm']} is not a registered arm ({sorted(ARMS)})")
    for k, want in (("bound", B), ("u_init", {"seed": U_INIT_SEED, "std": U_INIT_STD}),
                    ("sigma", [float(x) for x in arm.sigma()])):
        if ckpt.get(k) != want:
            raise ValueError(f"composite {k} differs from this module's ({want!r})")
    theta = np.asarray(ckpt["theta"], dtype=np.float64)
    if theta_sha256(theta) != ckpt["theta_sha256"]:
        raise ValueError("composite theta does not match its theta_sha256")
    nm = None
    if ckpt["normaliser"] is not None:
        nm = FeatureNormaliser.from_dict(ckpt["normaliser"])
        if nm.sha256 != ckpt["normaliser_sha256"]:
            raise ValueError("composite normaliser does not match its normaliser_sha256")
    kind = ckpt["base"].get("kind", BASE_TS)
    if kind not in BASE_KINDS:
        raise ValueError(f"composite base kind {kind!r} not in {BASE_KINDS}")
    return make_policy(base, arm, theta, nm, history_mode, kind)
