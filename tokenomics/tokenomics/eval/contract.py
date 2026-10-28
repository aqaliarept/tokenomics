"""Contract-failure accounting for gate and contract runs (add-shocks-i-drift design D10;
gate-harness "Contract failures are counted").

A contract failure is a broken run, not a bad result: a checkpoint that does not load, an
episode that raises, an observation or action outside its space, an episode that ends neither
terminated nor truncated, a digest mismatch between paired runs. Death and low value are not
failures; they are what the verdict judges.

- `ContractMonitor` hooks the shared step loop (`runner.run_episode(monitor=...)`). It checks
  every observation `env.last()` returns and every policy action before the env decodes it for
  full space membership: an ndarray of the space's shape, a dtype that casts safely to the
  space's, finite, inside the bounds. Decoding would otherwise clip or rescale a bad action
  silently. It also **latches** each agent's outcome the first time `env.last()` reports
  `terminated` or `truncated`, because PettingZoo's dead-step handling deletes the flags on
  `step(None)`; checking `env.terminations` after the loop would see nothing.
- `ContractMonitor.verify` runs after the loop: every `possible_agents` agent has a latched
  outcome, latched terminations agree with `dead` and latched truncations with
  `cycles == horizon`. A truncation is never inferred from the cycle count.
- `Failure` rows (scope `job` or `run`) go to an append-only `contract_failures.csv`. A run with
  any row is invalid, whatever happens on resume.
"""

from __future__ import annotations

import csv
import math
import os
import traceback
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

JOB = "job"
RUN = "run"
SCOPES = (JOB, RUN)

# check names
EXCEPTION = "exception"
OBSERVATION = "observation_space"
ACTION = "action_space"
TERMINAL = "terminal_outcome"
CHECKPOINT = "checkpoint"
DIGEST = "digest_pairing"

FAILURES_FILE = "contract_failures.csv"
TERMINATED = "terminated"
TRUNCATED = "truncated"


class ContractViolation(Exception):
    """A check failed inside an episode; `check` names it."""

    def __init__(self, check: str, message: str) -> None:
        super().__init__(message)
        self.check = check


@dataclass(frozen=True)
class Failure:
    """One `contract_failures.csv` row. Job fields are None on a `run` row."""

    scope: str
    check: str
    error_type: str
    message: str
    world: str | None = None
    episode: int | None = None
    subject: str | None = None
    replicate: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"contract failure scope must be one of {SCOPES}, got {self.scope!r}")

    def to_dict(self) -> dict[str, str]:
        return {f.name: "" if getattr(self, f.name) is None else str(getattr(self, f.name))
                for f in fields(self)}


FAILURE_COLUMNS = tuple(f.name for f in fields(Failure))


def job_failure(job: Mapping[str, Any], exc: BaseException) -> Failure:
    """The `job` row for an exception (a `ContractViolation` keeps its check name)."""
    check = exc.check if isinstance(exc, ContractViolation) else EXCEPTION
    message = str(exc) if isinstance(exc, ContractViolation) else (
        f"{exc}\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:])}")
    return Failure(scope=JOB, check=check, error_type=type(exc).__name__, message=message.strip(),
                   world=job["world"], episode=job["episode"], subject=job["subject"],
                   replicate=job["replicate"])


def run_failure(check: str, exc: BaseException) -> Failure:
    return Failure(scope=RUN, check=check, error_type=type(exc).__name__, message=str(exc))


def append_failure(out: Path, failure: Failure) -> None:
    """Append one row (header on a new file) and fsync: the record survives a crash."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / FAILURES_FILE
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FAILURE_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(failure.to_dict())
        fh.flush()
        os.fsync(fh.fileno())


def ensure_failures_file(out: Path) -> None:
    """A header-only file, so a run with no failure shows a count of 0 rather than no record."""
    path = out / FAILURES_FILE
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FAILURE_COLUMNS, lineterminator="\n").writeheader()


def read_failures(out: Path) -> list[dict[str, str]]:
    path = out / FAILURES_FILE
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# --- per-step checks ---


def space_problem(x: Any, space: spaces.Space) -> str | None:
    """Why `x` is not a member of `space`, or None. Box: ndarray, shape, safe dtype cast,
    finiteness, bounds. Other spaces: `space.contains`. This full check runs only on a failure
    or for spaces `_BoxCheck` cannot take; the step loop uses the cheap `_BoxCheck`."""
    if not isinstance(space, spaces.Box):
        return None if space.contains(x) else f"{x!r} not in {space}"
    if not isinstance(x, np.ndarray):
        return f"not an ndarray ({type(x).__name__})"
    if x.shape != space.shape:
        return f"shape {x.shape} != {space.shape}"
    if not np.can_cast(x.dtype, space.dtype):
        return f"dtype {x.dtype} does not cast safely to {space.dtype}"
    if x.dtype.kind in "fc" and not np.isfinite(x).all():
        return f"{int(x.size - np.isfinite(x).sum())} non-finite value(s)"
    lo, hi = x < space.low, x > space.high
    if lo.any() or hi.any():
        i = int(np.flatnonzero(lo | hi)[0])
        return (f"{int(lo.sum() + hi.sum())} value(s) outside the bounds, e.g. index {i}: "
                f"{x.flat[i]!r} not in [{space.low.flat[i]!r}, {space.high.flat[i]!r}]")
    return None


class _BoxCheck:
    """The per-step membership test of one Box space with uniform bounds (every gate space):
    type, shape and dtype by identity, then one `min` and one `max` reduction. A NaN makes
    `min` NaN and an inf lands in `min`/`max`, so finiteness and bounds come from two scalars;
    no temporaries. Anything it does not accept goes to `space_problem` for the message."""

    def __init__(self, space: spaces.Box) -> None:
        self.space = space
        self.shape = space.shape
        self.dtype = space.dtype
        self.lo = float(space.low.flat[0])
        self.hi = float(space.high.flat[0])

    def ok(self, x: Any) -> bool:
        if type(x) is not np.ndarray or x.shape != self.shape or (
                x.dtype != self.dtype and not np.can_cast(x.dtype, self.dtype)):
            return False
        if x.size == 0:
            return True
        mn, mx = float(x.min()), float(x.max())
        return (math.isfinite(mn) and math.isfinite(mx) and mn >= self.lo and mx <= self.hi)


def _fast_check(space: spaces.Space) -> _BoxCheck | None:
    if isinstance(space, spaces.Box) and space.low.size and space.dtype.kind == "f" and (
            (space.low == space.low.flat[0]).all() and (space.high == space.high.flat[0]).all()):
        return _BoxCheck(space)
    return None


class ContractMonitor:
    """The `runner.EpisodeMonitor` of a gate job (module docstring)."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.outcomes: dict[str, set[str]] = {}
        self._checks: dict[int, _BoxCheck | None] = {}

    def _check(self, x: Any, space: spaces.Space, check: str, what: str) -> None:
        key = id(space)
        if key not in self._checks:
            self._checks[key] = _fast_check(space)
        fast = self._checks[key]
        if fast is not None and fast.ok(x):
            return
        problem = space_problem(x, space)
        if problem is not None:
            raise ContractViolation(check, f"{what}: {problem}")

    def on_last(self, agent: str, observation: Any, terminated: bool, truncated: bool) -> None:
        self._check(observation, self.env.observation_space(agent), OBSERVATION,
                    f"observation of {agent}")
        if (terminated or truncated) and agent not in self.outcomes:
            self.outcomes[agent] = ({TERMINATED} if terminated else set()) | (
                {TRUNCATED} if truncated else set())

    def on_action(self, agent: str, action: Any) -> None:
        self._check(action, self.env.action_space(agent), ACTION, f"action of {agent}")

    def verify(self, dead: bool, cycles: int, horizon: int) -> None:
        """After the loop: every agent latched, and the latches agree with the stats."""
        agents = list(self.env.possible_agents)
        missing = [a for a in agents if a not in self.outcomes]
        if missing:
            raise ContractViolation(TERMINAL, f"no terminal outcome for {len(missing)} of "
                                              f"{len(agents)} agents (e.g. {missing[:3]}); "
                                              f"cycles {cycles}, horizon {horizon}, dead {dead}")
        for want, flag in ((dead, TERMINATED), (cycles == horizon, TRUNCATED)):
            wrong = [a for a in agents if (flag in self.outcomes[a]) != want]
            if wrong:
                raise ContractViolation(TERMINAL, (
                    f"{len(wrong)} agents (e.g. {wrong[:3]}) latched "
                    f"{'no ' if want else ''}{flag} but dead={dead}, cycles {cycles}, "
                    f"horizon {horizon}"))


class ContractRunError(RuntimeError):
    """The run recorded contract failures and ends invalid (non-zero exit)."""


def is_failure(d: Mapping[str, str]) -> bool:
    """A job result is a failure row (`Failure.to_dict`) rather than an episode row."""
    return "scope" in d and "check" in d


def failure_from_dict(d: Mapping[str, str]) -> Failure:
    def opt_int(s: str) -> int | None:
        return None if s == "" else int(s)

    return Failure(scope=d["scope"], check=d["check"], error_type=d["error_type"],
                   message=d["message"], world=d["world"] or None,
                   episode=opt_int(d["episode"]), subject=d["subject"] or None,
                   replicate=opt_int(d["replicate"]))
