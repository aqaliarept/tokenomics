"""Resilience rows: per-shock dip and recovery against the shock-free twin (ADR-0055;
add-shocks-i-drift design D6; stochastic-environment "Resilience is reported").

The shock-free twin of a shocked episode is the same world config with its shocks removed
(drift kept), run with the same episode seed. Every demand and shock stream is independent,
so the twin sees identical demand; the harness checks that through equal demand digests.
Resilience is a robustness report of the candidate, not an acceptance clause.

Inputs are per-played-cycle consumer units completed (`consumer_fulfilled_units_trace`;
cycle c is index c-1) for the shocked episode and its twin, plus the shocked episode's
schedule. H is the number of cycles the shocked episode played (shorter on early death).
For a shock starting at t with last active cycle e = t + duration - 1, window w and
tolerance tau:
- recovery cycle r: the first c in [e+1, H-w+1] whose window sum shocked[c..c+w-1] is at
  least (1 - tau) times the twin's sum over the same window;
- `recovery_cycles` = r - (e + 1), so 0 means recovered the cycle the shock ended;
- `dip_units` = sum over c in [t, r-1] of twin(c) - shocked(c), signed (negative when the
  shocked run did better);
- `dip_fraction` = dip_units over the twin's units on the same span (None when that is 0);
- no such r, or a shocked episode that ended in system death (`shocked_dead`, passed in
  explicitly from the episode's `dead` stat, never inferred from trace lengths): the row is
  unrecovered, `recovery_cycles` is None and the dip runs to the last played cycle H. A
  death censors every row, even one whose windows had already caught up with the twin;
- `overlapped`: another shock's active span [t', e'] meets the row's span [t, r-1] (or
  [t, H] when unrecovered).
A shock that starts after the last played cycle never took effect and gets no row.
The comparison is exact: tau is read as its decimal with `Fraction(str(tau))`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from ..domain.config import Configuration
from ..domain.shocks import ShockEvent


@dataclass(frozen=True)
class ResilienceParams:
    window: int = 5
    tolerance: float = 0.05

    def __post_init__(self) -> None:
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 1:
            raise ValueError(f"resilience window must be an integer >= 1, got {self.window!r}")
        if not 0.0 <= self.tolerance < 1.0:
            raise ValueError(f"resilience tolerance must be in [0, 1), got {self.tolerance!r}")


# design D7 values; the shocks-i profile file (task 5.1) carries them for gate runs
DEFAULT_PARAMS = ResilienceParams()


@dataclass(frozen=True)
class ResilienceRow:
    kind: str
    producer: str | None
    capacity_types: tuple[str, ...]
    start: int
    end: int  # last active cycle
    magnitude: float
    recovered: bool
    recovery_cycle: int | None
    recovery_cycles: int | None
    dip_units: int
    dip_fraction: float | None
    overlapped: bool


def shock_free_twin(config: Configuration) -> Configuration:
    """The same world without shocks (D6). Demand, drift included, is untouched."""
    return replace(config, shocks=None)


def _active_end(e: ShockEvent) -> int:
    return e.start_cycle + e.duration - 1


def resilience_rows(
    shocked: Sequence[int], twin: Sequence[int], schedule: Sequence[ShockEvent],
    params: ResilienceParams = DEFAULT_PARAMS, *, shocked_dead: bool,
) -> list[ResilienceRow]:
    """One row per shock that took effect, in schedule order (module docstring).
    `shocked_dead` is the shocked episode's `dead` stat; the twin's outcome does not
    matter (only its units are compared)."""
    h = len(shocked)
    if len(twin) < h:
        raise ValueError(f"twin trace has {len(twin)} cycles, the shocked episode played {h}")
    w, keep = params.window, 1 - Fraction(str(params.tolerance))

    def unit(trace: Sequence[int], c: int) -> int:
        return trace[c - 1]

    def window_ok(c: int) -> bool:
        s = sum(unit(shocked, x) for x in range(c, c + w))
        t = sum(unit(twin, x) for x in range(c, c + w))
        return s >= keep * t

    rows: list[ResilienceRow] = []
    for idx, e in enumerate(schedule):
        t0, end = e.start_cycle, _active_end(e)
        if t0 > h:
            continue  # never took effect: the episode ended first
        r = None if shocked_dead else next(
            (c for c in range(end + 1, h - w + 2) if window_ok(c)), None)
        span_end = h if r is None else r - 1
        dip = sum(unit(twin, c) - unit(shocked, c) for c in range(t0, span_end + 1))
        twin_units = sum(unit(twin, c) for c in range(t0, span_end + 1))
        overlapped = any(
            j != idx and o.start_cycle <= span_end and _active_end(o) >= t0
            for j, o in enumerate(schedule)
        )
        rows.append(ResilienceRow(
            kind=e.kind.value, producer=e.target.producer, capacity_types=e.target.capacity_types,
            start=t0, end=end, magnitude=e.magnitude, recovered=r is not None,
            recovery_cycle=r, recovery_cycles=None if r is None else r - (end + 1),
            dip_units=dip, dip_fraction=None if twin_units == 0 else dip / twin_units,
            overlapped=overlapped,
        ))
    return rows
