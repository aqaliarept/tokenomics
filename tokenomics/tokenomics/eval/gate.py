"""Hindsight-baseline and verdict arithmetic of the measurement gate (design D6,
add-measurement-gate-harness; gate-harness "Hindsight-baseline arithmetic", "Gate report contents",
"Verdict is reproducible from rows"; agent-acceptance margin, no-losing-world, zero-death and
side-effect clauses).

Rows first, arithmetic pure: the driver (task 8.1) writes one `EpisodeRow` per (world, episode,
subject, replicate); everything the verdict says is a pure function of those rows plus the frozen
roster, side-effect parameters and registry version. The CSV round-trip is exact (floats are
written with `repr`), so `--mode recompute` rebuilds a report equal to the recorded one.

Decisions this module fixes (ARCHITECTURE.md §7.2):
- Only candidate episodes are judged. Roster rows feed the hindsight baseline and nothing else;
  a baseline's death or starvation is not a gate failure.
- Thresholds are inclusive: a ratio, `r_end` or fill rate exactly at its bar passes.
- Margins are judged per (economy size, regime): "train small, play large" requires every margin
  at each size, so a 12-producer surplus cannot hide a 120-producer shortfall. Every gate size
  (`roster.GATE_SIZES`, 12 and 120) must have both a calm and a shocked group; a group with no
  gating world FAILS its margin (no evidence is not a pass), so a report missing a size rejects.
- The input is the gate set only. A row at any other size (e.g. the 1000+ producer "future scale
  target", which agent-acceptance says SHALL NOT gate) is refused, not silently judged: such
  observations are reported outside this verdict.
- A zero-baseline world has no finite ratio. It is excluded from every clause (margins,
  no-losing-world, death and side-effect checks) and listed in the verdict as a non-gating
  observation; screening (8.2) should already have replaced it.
- Run-down with `r_start` or `r_end` None (unmeasurable) fails closed; `r_start == 0` passes
  arithmetically (`r_end >= X * 0`). A consumer whose fill rate is None ordered nothing and is not
  judged by starvation (side-effect-guards "Consumer that ordered nothing").
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from statistics import fmean
from typing import Collection, Iterable, Mapping, Sequence

from .roster import GATE_SIZES, Roster, content_hash
from .seeding import replicate_mean

CALM = "calm"
SHOCKED = "shocked"
REGIMES = (CALM, SHOCKED)

# Acceptance bars (openspec/specs/agent-acceptance). X and Y are NOT here: they are frozen
# calibration outputs (design D4) and arrive through `SideEffectParams`.
CALM_MARGIN = 1.05
SHOCKED_MARGIN = 1.10
MARGINS = {CALM: CALM_MARGIN, SHOCKED: SHOCKED_MARGIN}
NO_LOSING_FLOOR = 1.0

CANDIDATE = "candidate"  # subject id of candidate rows; every other subject is a roster member id

# Clause names, in report order.
CALM_CLAUSE = "calm_margin"
SHOCKED_CLAUSE = "shocked_margin"
NO_LOSING_CLAUSE = "no_losing_world"
ZERO_DEATH_CLAUSE = "zero_death"
RUN_DOWN_CLAUSE = "run_down"
STARVATION_CLAUSE = "starvation"
CLAUSES = (
    CALM_CLAUSE, SHOCKED_CLAUSE, NO_LOSING_CLAUSE, ZERO_DEATH_CLAUSE, RUN_DOWN_CLAUSE,
    STARVATION_CLAUSE,
)
MARGIN_CLAUSE = {CALM: CALM_CLAUSE, SHOCKED: SHOCKED_CLAUSE}

ACCEPT = "accept"
REJECT = "reject"

# Monitored side-effect metrics (design D3): reported per world, never gate.
MONITORED = (
    "inversion_rate",
    "min_capacity_ratio",
    "min_spare_fraction",
    "investment_rejection_rate",
    "consumer_rejection_rate",
    "holding_units",
    "holding_units_per_cycle",
)


class ZeroBaselineWorld(ValueError):
    """A world's hindsight baseline is 0, so it has no finite ratio (agent-acceptance "Gate worlds
    have a non-zero hindsight baseline")."""


class DigestMismatch(ValueError):
    """Two subjects saw different demand on the same (world, episode) (gate-harness "Identical
    demand across observation contracts")."""


# --- side-effect parameters ---


@dataclass(frozen=True)
class SideEffectParams:
    """Frozen X (run-down factor) and Y (starvation floor) with the version of the parameter file
    they came from (design D4; frozen by 8.2/10.1, never hard-coded here)."""

    x: float
    y: float
    version: str

    def __post_init__(self) -> None:
        for name in ("x", "y"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"side-effect parameter {name} must be finite and >= 0, got {v!r}")
            object.__setattr__(self, name, float(v))
        if not self.version:
            raise ValueError("side-effect parameters need a version")


# Side-effect parameter file (design D4; frozen by 8.2/10.1 as side_effect_params_v1.json). Same
# versioning as roster/registry: the version is a sha256 content hash over every field but the
# hash, re-checked on load, so a report naming the version names exactly its X and Y.
DEFAULT_PARAMS = Path(__file__).parent / "registry" / "side_effect_params_v1.json"


def side_effect_params_payload(
    x: float, y: float, q: float, *, name: str = "side-effect-params", rule: str = "",
) -> dict:
    """The file payload (no hash). `q` and `rule` record how X and Y were calibrated (D4)."""
    return {"name": name, "x": float(x), "y": float(y), "q": float(q), "rule": rule}


def side_effect_params_json(x: float, y: float, q: float, **kw: str) -> str:
    d = side_effect_params_payload(x, y, q, **kw)
    d["content_hash"] = content_hash(d)
    return json.dumps(d, indent=2, sort_keys=True) + "\n"


def params_from_dict(d: Mapping[str, object]) -> SideEffectParams:
    """Parse a parameter file, refusing a recorded hash that does not match its contents."""
    d = dict(d)
    recorded = d.pop("content_hash", None)
    if recorded is None:
        raise ValueError("side-effect parameter file records no content_hash; it has no version")
    if set(d) != {"name", "x", "y", "q", "rule"}:
        raise ValueError(f"side-effect parameter file fields {sorted(d)} != name, x, y, q, rule")
    if content_hash(d) != recorded:
        raise ValueError(f"side-effect parameter hash {content_hash(d)} != recorded {recorded}: "
                         "edited without a new version")
    return SideEffectParams(x=d["x"], y=d["y"], version=str(recorded))  # type: ignore[arg-type]


def load_side_effect_params(path: str | Path = DEFAULT_PARAMS) -> SideEffectParams:
    return params_from_dict(json.loads(Path(path).read_text()))


# --- rows ---

FillRates = tuple[tuple[str, float | None], ...]


def _opt_float(v: float | None, what: str) -> float | None:
    """None stays None (the env's "no denominator"); anything else must be a finite number. The
    env never emits NaN or inf (side_effects.py: zero denominators give None), so a non-finite
    value is corrupt data. It is refused on ingestion: a NaN compares False with every bar, so a
    `<` check would silently pass it, and NaN/inf would also leak into the report JSON."""
    if v is None:
        return None
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{what} is not finite: {v!r}")
    return f


@dataclass(frozen=True)
class EpisodeRow:
    """One subject's run of one episode (design D6). `subject` is `CANDIDATE` or a roster member id
    (`roster.member_id`); `replicate` is 0 for a deterministic subject. `units` is
    `consumer_completed_units`, the primary metric. `regime` and `size` are the world's registry
    fields, carried on every row so the CSV alone determines the margin groups.

    Side-effect fields mirror `env/side_effects.py`; `consumer_fill_rate` is stored as sorted
    (consumer, rate) pairs so the row is hashable and its serialization deterministic."""

    world: str
    size: int
    regime: str
    episode: int
    subject: str
    replicate: int
    units: float
    dead: bool
    demand_digest: str
    r_start: float | None
    r_end: float | None
    consumer_fill_rate: FillRates
    inversion_rate: float | None = None
    min_capacity_ratio: float | None = None
    min_spare_fraction: float | None = None
    investment_rejection_rate: float | None = None
    consumer_rejection_rate: float | None = None
    holding_units: float | None = None
    holding_units_per_cycle: float | None = None

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"row {self.world}/{self.episode}: unknown regime {self.regime!r}")
        if not isinstance(self.dead, bool):
            raise ValueError(f"row {self.world}/{self.episode}: dead must be a bool")
        if self.replicate < 0:
            raise ValueError(f"row {self.world}/{self.episode}: negative replicate")
        units = float(self.units)
        if not math.isfinite(units) or units < 0:
            raise ValueError(f"row {self.world}/{self.episode}/{self.subject}: units {self.units!r}")
        object.__setattr__(self, "units", units)
        where = f"row {self.world}/{self.episode}/{self.subject}/{self.replicate}"
        for name in ("r_start", "r_end", *MONITORED):
            object.__setattr__(self, name, _opt_float(getattr(self, name), f"{where} {name}"))
        rates = self.consumer_fill_rate
        if isinstance(rates, Mapping):
            rates = rates.items()
        object.__setattr__(
            self, "consumer_fill_rate",
            tuple(sorted(
                (str(c), _opt_float(v, f"{where} fill rate of {c}")) for c, v in rates
            )),
        )

    @property
    def fill_rates(self) -> dict[str, float | None]:
        return dict(self.consumer_fill_rate)


def row_from_stats(
    world: str, size: int, regime: str, episode: int, subject: str, replicate: int,
    stats: Mapping[str, object],
) -> EpisodeRow:
    """Build a row from a finished episode's `episode_stats` (keys of env/aec_env.py and
    env/side_effects.py), so the stats → row mapping lives next to the arithmetic that reads it."""
    return EpisodeRow(
        world=world, size=size, regime=regime, episode=episode, subject=subject,
        replicate=replicate,
        units=stats["consumer_completed_units"],  # type: ignore[arg-type]
        dead=bool(stats["dead"]),
        demand_digest=str(stats["demand_digest"]),
        r_start=stats["r_start"],  # type: ignore[arg-type]
        r_end=stats["r_end"],  # type: ignore[arg-type]
        consumer_fill_rate=stats["consumer_fill_rate"],  # type: ignore[arg-type]
        **{k: stats[k] for k in MONITORED},  # type: ignore[arg-type]
    )


# CSV columns, in file order (the `episodes.csv` header of task 8.1).
CSV_COLUMNS = tuple(f.name for f in fields(EpisodeRow))
_INT_COLUMNS = ("size", "episode", "replicate")
_STR_COLUMNS = ("world", "regime", "subject", "demand_digest")


def _cell(name: str, v: object) -> str:
    """Exact text of one value: `repr` for floats (shortest round-trip form), "" for None, JSON
    (which also writes floats with repr) for the fill-rate pairs."""
    if name == "consumer_fill_rate":
        return json.dumps(dict(v), sort_keys=True)  # type: ignore[call-overload]
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def row_to_dict(row: EpisodeRow) -> dict[str, str]:
    """A row as CSV-ready strings, keyed by `CSV_COLUMNS`."""
    return {name: _cell(name, getattr(row, name)) for name in CSV_COLUMNS}


def row_from_dict(d: Mapping[str, str]) -> EpisodeRow:
    """Inverse of `row_to_dict`; refuses missing columns rather than defaulting them."""
    missing = [c for c in CSV_COLUMNS if c not in d]
    if missing:
        raise ValueError(f"episode row lacks columns {missing}")
    kw: dict[str, object] = {}
    for name in CSV_COLUMNS:
        s = d[name]
        if name in _STR_COLUMNS:
            kw[name] = s
        elif name in _INT_COLUMNS:
            kw[name] = int(s)
        elif name == "dead":
            if s not in ("true", "false"):
                raise ValueError(f"dead must be 'true' or 'false', got {s!r}")
            kw[name] = s == "true"
        elif name == "consumer_fill_rate":
            kw[name] = json.loads(s)
        else:
            kw[name] = None if s == "" else float(s)
    return EpisodeRow(**kw)  # type: ignore[arg-type]


def rows_to_csv(rows: Iterable[EpisodeRow]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(row_to_dict(r))
    return buf.getvalue()


def rows_from_csv(text: str) -> list[EpisodeRow]:
    return [row_from_dict(d) for d in csv.DictReader(io.StringIO(text))]


# --- hindsight-baseline arithmetic (gate-harness "Hindsight-baseline arithmetic") ---


def episode_baseline(member_values: Mapping[str, float]) -> float:
    """Per-episode hindsight baseline: the max over roster members of their per-episode value (a
    random member's value is already its replicate mean, `seeding.replicate_mean`)."""
    if not member_values:
        raise ValueError("episode baseline needs at least one roster member value")
    return max(member_values.values())


def world_baseline(episode_baselines: Sequence[float]) -> float:
    """World hindsight baseline: the mean of its per-episode baselines. Not the max over members of
    their means, which is lower whenever the best member changes from episode to episode."""
    if not episode_baselines:
        raise ValueError("world baseline needs at least one episode")
    return fmean(episode_baselines)


def world_ratio(candidate_mean: float, baseline: float) -> float:
    """Candidate mean units over the world hindsight baseline on the same episodes. Refuses a zero
    baseline instead of returning inf/nan (agent-acceptance "Gate worlds have a non-zero hindsight
    baseline"); the caller excludes the world."""
    if baseline == 0:
        raise ZeroBaselineWorld("world hindsight baseline is 0: no finite ratio")
    return candidate_mean / baseline


def regime_ratio(ratios: Sequence[float]) -> float:
    """Equal-weight mean of a regime's world ratios: a world with more episodes weighs no more."""
    if not ratios:
        raise ValueError("regime ratio needs at least one gating world")
    return fmean(ratios)


# --- report and verdict ---


@dataclass(frozen=True)
class Failure:
    """One reason a clause failed. Scope fields not meaningful for the clause are None: a margin
    names (size, regime); no-losing-world a world; death/run-down/starvation an episode and
    replicate; starvation also the consumer."""

    clause: str
    detail: str
    value: float | None
    threshold: float
    size: int | None = None
    regime: str | None = None
    world: str | None = None
    episode: int | None = None
    replicate: int | None = None
    consumer: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class ClauseVerdict:
    clause: str
    passed: bool
    failures: tuple[Failure, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "clause": self.clause,
            "passed": self.passed,
            "failures": [f.to_dict() for f in self.failures],
        }


@dataclass(frozen=True)
class WorldResult:
    """Per-world report line (gate-harness "Gate report contents"). `ratio` is None and
    `gating` False for a zero-baseline world. Hard-check counts are over candidate rows."""

    world: str
    size: int
    regime: str
    episodes: tuple[int, ...]
    candidate_mean: float
    baseline: float
    ratio: float | None
    gating: bool
    deaths: int
    run_down_failures: int
    starvation_failures: int
    monitored: tuple[tuple[str, float | None], ...]

    def to_dict(self) -> dict[str, object]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["episodes"] = list(self.episodes)
        d["monitored"] = dict(self.monitored)
        return d


@dataclass(frozen=True)
class RegimeResult:
    size: int
    regime: str
    worlds: tuple[str, ...]
    mean_ratio: float | None
    margin: float
    margin_passed: bool
    no_losing_passed: bool

    def to_dict(self) -> dict[str, object]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["worlds"] = list(self.worlds)
        return d


@dataclass(frozen=True)
class GateReport:
    """Everything the verdict says, with the frozen versions it was computed under. Equal reports
    serialize to identical JSON (`to_json`), which is what "Recompute from the CSV" compares."""

    roster_version: str
    registry_version: str
    params_version: str
    x: float
    y: float
    worlds: tuple[WorldResult, ...]
    regimes: tuple[RegimeResult, ...]
    clauses: tuple[ClauseVerdict, ...]
    non_gating_worlds: tuple[str, ...]
    overall: str
    failing_clauses: tuple[str, ...] = field(default=())

    def clause(self, name: str) -> ClauseVerdict:
        for c in self.clauses:
            if c.clause == name:
                return c
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {
            "roster_version": self.roster_version,
            "registry_version": self.registry_version,
            "params_version": self.params_version,
            "x": self.x,
            "y": self.y,
            "worlds": [w.to_dict() for w in self.worlds],
            "regimes": [r.to_dict() for r in self.regimes],
            "clauses": [c.to_dict() for c in self.clauses],
            "non_gating_worlds": list(self.non_gating_worlds),
            "overall": self.overall,
            "failing_clauses": list(self.failing_clauses),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _check_rows(
    rows: Sequence[EpisodeRow], roster: Roster, gate_sizes: Collection[int],
) -> dict[str, dict[int, list[EpisodeRow]]]:
    """Group rows by world → episode and refuse anything the arithmetic cannot honestly use:
    rows outside the gate sizes, unknown subjects, duplicate or missing replicates, a missing member or candidate on an episode
    (a max over a subset would understate the bar), inconsistent world metadata, and demand-digest
    mismatches."""
    members = set(roster.member_ids)
    expected = {mid: set(range(roster.replicates_for(roster.member(mid)))) for mid in members}
    by_world: dict[str, dict[int, list[EpisodeRow]]] = {}
    meta: dict[str, tuple[int, str]] = {}
    for r in rows:
        if r.size not in gate_sizes:
            raise ValueError(f"world {r.world}: size {r.size} is not a gate size "
                             f"{sorted(gate_sizes)}; non-gate scales are reported outside the verdict")
        if r.subject != CANDIDATE and r.subject not in members:
            raise ValueError(f"row subject {r.subject!r} is neither the candidate nor a member of "
                             f"roster {roster.version}")
        if meta.setdefault(r.world, (r.size, r.regime)) != (r.size, r.regime):
            raise ValueError(f"world {r.world}: rows disagree on (size, regime)")
        by_world.setdefault(r.world, {}).setdefault(r.episode, []).append(r)
    if not by_world:
        raise ValueError("no episode rows")
    for world, eps in by_world.items():
        for ep, ep_rows in eps.items():
            reps: dict[str, list[int]] = {}
            for r in ep_rows:
                reps.setdefault(r.subject, []).append(r.replicate)
            for subj, got in reps.items():
                if len(got) != len(set(got)):
                    raise ValueError(f"{world}/{ep}: duplicate replicate rows for {subj}")
            missing = sorted(members - set(reps))
            if missing:
                raise ValueError(f"{world}/{ep}: no rows for roster members {missing}")
            if CANDIDATE not in reps:
                raise ValueError(f"{world}/{ep}: no candidate rows")
            for mid in members:
                if set(reps[mid]) != expected[mid]:
                    raise ValueError(f"{world}/{ep}: {mid} has replicates {sorted(reps[mid])}, "
                                     f"roster wants {sorted(expected[mid])}")
            if reps[CANDIDATE] and set(reps[CANDIDATE]) != set(range(len(reps[CANDIDATE]))):
                raise ValueError(f"{world}/{ep}: candidate replicates {sorted(reps[CANDIDATE])} "
                                 "are not 0..n-1")
            ref = next(r for r in ep_rows if r.subject == CANDIDATE).demand_digest
            for r in ep_rows:
                if r.demand_digest != ref:
                    raise DigestMismatch(
                        f"world {world} episode {ep}: {r.subject} replicate {r.replicate} saw "
                        f"demand digest {r.demand_digest}, candidate saw {ref}"
                    )
    return by_world


def _subject_value(rows: Sequence[EpisodeRow], subject: str) -> float:
    return replicate_mean([r.units for r in sorted(rows, key=lambda r: r.replicate)
                           if r.subject == subject])


def _monitored_means(rows: Sequence[EpisodeRow]) -> tuple[tuple[str, float | None], ...]:
    """Pass-through: mean of each monitored metric over the candidate's rows, skipping None (a
    metric with no denominator on that episode); None when no row has it."""
    out = []
    for name in MONITORED:
        vals = [getattr(r, name) for r in rows if getattr(r, name) is not None]
        out.append((name, fmean(vals) if vals else None))
    return tuple(out)


def _episode_failures(r: EpisodeRow, params: SideEffectParams) -> list[Failure]:
    """Death, run-down and starvation failures of one candidate row."""
    scope = {"size": r.size, "regime": r.regime, "world": r.world, "episode": r.episode,
             "replicate": r.replicate}
    out: list[Failure] = []
    if r.dead:
        out.append(Failure(ZERO_DEATH_CLAUSE, "episode ended in system death", 1.0, 0.0, **scope))
    if r.r_start is None or r.r_end is None:
        out.append(Failure(RUN_DOWN_CLAUSE, "run-down unmeasurable (r_start or r_end is None)",
                           None, params.x, **scope))
    elif r.r_end < params.x * r.r_start:
        out.append(Failure(
            RUN_DOWN_CLAUSE, f"r_end {r.r_end!r} < X·r_start {params.x * r.r_start!r}",
            r.r_end / r.r_start if r.r_start else None, params.x, **scope))
    for consumer, rate in r.consumer_fill_rate:
        if rate is not None and rate < params.y:
            out.append(Failure(STARVATION_CLAUSE, f"consumer {consumer} fill rate {rate!r} < Y",
                               rate, params.y, consumer=consumer, **scope))
    return out


def compute_report(
    rows: Sequence[EpisodeRow], roster: Roster, params: SideEffectParams, registry_version: str,
    gate_sizes: Collection[int] = GATE_SIZES,
) -> GateReport:
    """The gate report and verdict, a pure function of the rows and the frozen versions (D6).
    `gate_sizes` defaults to the acceptance sizes; only fixtures narrow it."""
    if not gate_sizes:
        raise ValueError("no gate sizes")
    by_world = _check_rows(rows, roster, gate_sizes)
    worlds: list[WorldResult] = []
    failures: dict[str, list[Failure]] = {c: [] for c in CLAUSES}
    for world in sorted(by_world):
        eps = by_world[world]
        episodes = tuple(sorted(eps))
        any_row = eps[episodes[0]][0]
        size, regime = any_row.size, any_row.regime
        ep_baselines = [
            episode_baseline({mid: _subject_value(eps[ep], mid) for mid in roster.member_ids})
            for ep in episodes
        ]
        baseline = world_baseline(ep_baselines)
        cand_mean = fmean([_subject_value(eps[ep], CANDIDATE) for ep in episodes])
        cand_rows = [r for ep in episodes
                     for r in sorted(eps[ep], key=lambda r: r.replicate) if r.subject == CANDIDATE]
        ep_fail = [f for r in cand_rows for f in _episode_failures(r, params)]
        gating = baseline != 0
        ratio = world_ratio(cand_mean, baseline) if gating else None
        if gating:
            for f in ep_fail:
                failures[f.clause].append(f)
            if ratio < NO_LOSING_FLOOR:  # type: ignore[operator]
                failures[NO_LOSING_CLAUSE].append(Failure(
                    NO_LOSING_CLAUSE, f"world ratio {ratio!r} < {NO_LOSING_FLOOR!r}", ratio,
                    NO_LOSING_FLOOR, size=size, regime=regime, world=world))
        worlds.append(WorldResult(
            world=world, size=size, regime=regime, episodes=episodes, candidate_mean=cand_mean,
            baseline=baseline, ratio=ratio, gating=gating,
            deaths=sum(1 for f in ep_fail if f.clause == ZERO_DEATH_CLAUSE),
            run_down_failures=sum(1 for f in ep_fail if f.clause == RUN_DOWN_CLAUSE),
            starvation_failures=len({(f.episode, f.replicate) for f in ep_fail
                                     if f.clause == STARVATION_CLAUSE}),
            monitored=_monitored_means(cand_rows),
        ))

    regimes: list[RegimeResult] = []
    for size in sorted(gate_sizes):
        for regime in REGIMES:
            group = [w for w in worlds if w.size == size and w.regime == regime and w.gating]
            margin = MARGINS[regime]
            mean = regime_ratio([w.ratio for w in group]) if group else None  # type: ignore[misc]
            margin_ok = mean is not None and mean >= margin
            if not margin_ok:
                detail = (f"mean ratio {mean!r} < {margin!r}" if mean is not None
                          else "no gating worlds in this regime at this size")
                failures[MARGIN_CLAUSE[regime]].append(Failure(
                    MARGIN_CLAUSE[regime], detail, mean, margin, size=size, regime=regime))
            regimes.append(RegimeResult(
                size=size, regime=regime, worlds=tuple(w.world for w in group), mean_ratio=mean,
                margin=margin, margin_passed=margin_ok,
                no_losing_passed=all(w.ratio >= NO_LOSING_FLOOR for w in group),  # type: ignore[operator]
            ))

    clauses = tuple(ClauseVerdict(c, not failures[c], tuple(failures[c])) for c in CLAUSES)
    failing = tuple(c.clause for c in clauses if not c.passed)
    return GateReport(
        roster_version=roster.version, registry_version=registry_version,
        params_version=params.version, x=params.x, y=params.y,
        worlds=tuple(worlds), regimes=tuple(regimes), clauses=clauses,
        non_gating_worlds=tuple(w.world for w in worlds if not w.gating),
        overall=REJECT if failing else ACCEPT, failing_clauses=failing,
    )


def recompute_from_csv(
    text: str, roster: Roster, params: SideEffectParams, registry_version: str,
    gate_sizes: Collection[int] = GATE_SIZES,
) -> GateReport:
    """`--mode recompute` core (D6): the report rebuilt from `episodes.csv` text alone."""
    return compute_report(rows_from_csv(text), roster, params, registry_version, gate_sizes)


if __name__ == "__main__":
    # `python -m tokenomics.eval.gate`: the driver lives in gate_cli so this module stays pure.
    import sys

    from .gate_cli import main

    sys.exit(main())
