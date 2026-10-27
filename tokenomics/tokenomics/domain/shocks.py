"""Shock model: pure configuration data for supply shocks (ADR-0055; add-shocks-i-drift
design D1; stochastic-environment "Shock kinds", "Shock delivery order").

The schema covers all five kinds, but only capacity hits and producer outages are built
(M2). Validation rejects every other kind with an error that names it, so a config cannot
schedule a shock the simulation would silently ignore. The domain never draws a schedule:
the env generates one from a `ShockGenerator` and the episode seed and injects it (design
D3), and the domain replays it. This module therefore holds data and checks only, with no
RNG.

Kind table (design D1):

| Kind            | Target                                | Duration | Magnitude    | M2       |
|-----------------|---------------------------------------|----------|--------------|----------|
| capacity_hit    | producer + >= 1 capacity types it owns | 1        | m in (0, 1)  | built    |
| producer_outage | producer                              | >= 1     | 1.0          | built    |
| link_cut        | producer + a product it declares      | >= 1     | 1.0          | rejected |
| demand_surge    | consumer                              | >= 1     | factor > 1   | rejected |
| fund_shock      | the investment fund                   | >= 1     | m in (0, 1]  | rejected |

Every event starts at cycle >= 2, so the state observed right after reset is never shocked.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .common import CapacityType, ConsumerId, Product, ProducerId


class ShockKind(str, Enum):
    CAPACITY_HIT = "capacity_hit"
    PRODUCER_OUTAGE = "producer_outage"
    LINK_CUT = "link_cut"
    DEMAND_SURGE = "demand_surge"
    FUND_SHOCK = "fund_shock"


IMPLEMENTED_SHOCK_KINDS: frozenset[ShockKind] = frozenset(
    {ShockKind.CAPACITY_HIT, ShockKind.PRODUCER_OUTAGE}
)
MIN_START_CYCLE = 2  # cycle 1 is ready at reset; a shock there would be observed at reset


@dataclass(frozen=True)
class ShockTarget:
    """What a shock hits. Each kind uses a subset of the fields (module kind table); the
    unused fields must stay empty."""

    producer: ProducerId | None = None
    capacity_types: tuple[CapacityType, ...] = ()
    product: Product | None = None
    consumer: ConsumerId | None = None


@dataclass(frozen=True)
class ShockEvent:
    kind: ShockKind
    target: ShockTarget
    start_cycle: int
    duration: int
    magnitude: float


@dataclass(frozen=True)
class ShockGenerator:
    """Seeded generation of `count` events of one kind (design D3). Ranges are inclusive.
    `targets` lists the producers it may hit; empty = every producer. The env draws the
    events; nothing here is random."""

    kind: ShockKind
    count: int
    start_window: tuple[int, int]
    duration: tuple[int, int]
    magnitude: tuple[float, float]
    targets: tuple[ProducerId, ...] = ()


@dataclass(frozen=True)
class ShockModel:
    """A config's shocks: explicit events plus at most one generator per kind."""

    events: tuple[ShockEvent, ...] = ()
    generators: tuple[ShockGenerator, ...] = ()


@dataclass(frozen=True)
class ProducerFacts:
    """What validation needs to know about one producer."""

    capacity_types: frozenset[CapacityType]
    produces: frozenset[Product]


# --- validation ---


def _unsupported(kind: ShockKind, where: str) -> ValueError:
    built = ", ".join(sorted(k.value for k in IMPLEMENTED_SHOCK_KINDS))
    return ValueError(f"{where}: unsupported shock kind {kind.value!r} (not implemented yet; "
                      f"implemented: {built})")


def _check_kind(kind: object, where: str) -> ShockKind:
    if not isinstance(kind, ShockKind):
        raise ValueError(f"{where}: kind must be a ShockKind, got {kind!r}")
    if kind not in IMPLEMENTED_SHOCK_KINDS:
        raise _unsupported(kind, where)
    return kind


def _check_int(value: object, name: str, where: str) -> int:
    """Cycles, durations and counts are discrete: a fractional onset would never equal the
    integer cycle counter and the shock would silently vanish. bool is excluded."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}: {name} must be an integer, got {value!r}")
    return value


def _check_number(value: object, name: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where}: {name} must be a finite number, got {value!r}")
    return float(value)


def _check_pair(value: object, name: str, where: str) -> tuple[object, object]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{where}: {name} must be a (lo, hi) pair, got {value!r}")
    return value[0], value[1]


def _check_producer(pid: ProducerId | None, producers: Mapping[ProducerId, ProducerFacts],
                    where: str) -> ProducerFacts:
    if pid is None:
        raise ValueError(f"{where}: target names no producer")
    if pid not in producers:
        raise ValueError(f"{where}: target producer {pid!r} does not exist")
    return producers[pid]


def _check_event(e: ShockEvent, producers: Mapping[ProducerId, ProducerFacts],
                 where: str) -> None:
    kind = _check_kind(e.kind, where)
    _check_int(e.start_cycle, "start cycle", where)
    _check_int(e.duration, "duration", where)
    if e.start_cycle < MIN_START_CYCLE:
        raise ValueError(f"{where}: start cycle must be >= {MIN_START_CYCLE}, got {e.start_cycle}")
    if isinstance(e.magnitude, bool) or not isinstance(e.magnitude, (int, float)) \
            or not math.isfinite(e.magnitude):
        raise ValueError(f"{where}: magnitude must be finite, got {e.magnitude!r}")
    t = e.target
    if t.product is not None or t.consumer is not None:
        raise ValueError(f"{where}: a {kind.value} target names only a producer"
                         + (" and capacity types" if kind is ShockKind.CAPACITY_HIT else ""))
    facts = _check_producer(t.producer, producers, where)
    if kind is ShockKind.CAPACITY_HIT:
        if not t.capacity_types:
            raise ValueError(f"{where}: a capacity hit names at least one capacity type")
        if len(set(t.capacity_types)) != len(t.capacity_types):
            raise ValueError(f"{where}: duplicate capacity types {list(t.capacity_types)}")
        unowned = sorted(set(t.capacity_types) - facts.capacity_types)
        if unowned:
            raise ValueError(f"{where}: target producer {t.producer!r} does not own capacity "
                             f"type(s) {unowned}")
        if e.duration != 1:
            raise ValueError(f"{where}: a capacity hit has duration 1, got {e.duration}")
        if not 0.0 < e.magnitude < 1.0:
            raise ValueError(f"{where}: capacity-hit magnitude must be in (0, 1), got {e.magnitude}")
    else:  # producer outage
        if t.capacity_types:
            raise ValueError(f"{where}: an outage target names no capacity types")
        if e.duration < 1:
            raise ValueError(f"{where}: duration must be >= 1, got {e.duration}")
        if e.magnitude != 1.0:
            raise ValueError(f"{where}: outage magnitude must be 1.0, got {e.magnitude}")


def _check_int_range(value: object, name: str, where: str) -> tuple[int, int]:
    lo, hi = _check_pair(value, name, where)
    lo, hi = _check_int(lo, f"{name} lo", where), _check_int(hi, f"{name} hi", where)
    if lo > hi:
        raise ValueError(f"{where}: {name} must be a range lo <= hi, got [{lo}, {hi}]")
    return lo, hi


def _check_number_range(value: object, name: str, where: str) -> tuple[float, float]:
    lo, hi = _check_pair(value, name, where)
    lo, hi = _check_number(lo, f"{name} lo", where), _check_number(hi, f"{name} hi", where)
    if lo > hi:
        raise ValueError(f"{where}: {name} must be a range lo <= hi, got [{lo}, {hi}]")
    return lo, hi


def _check_generator(g: ShockGenerator, producers: Mapping[ProducerId, ProducerFacts],
                     where: str) -> None:
    kind = _check_kind(g.kind, where)
    if _check_int(g.count, "count", where) < 0:
        raise ValueError(f"{where}: count must be >= 0, got {g.count}")
    s_lo, _ = _check_int_range(g.start_window, "start window", where)
    d_lo, d_hi = _check_int_range(g.duration, "duration", where)
    m_lo, m_hi = _check_number_range(g.magnitude, "magnitude", where)
    if not isinstance(g.targets, tuple):
        raise ValueError(f"{where}: targets must be a tuple of producer ids, got {g.targets!r}")
    if s_lo < MIN_START_CYCLE:
        raise ValueError(f"{where}: start window must begin at cycle >= {MIN_START_CYCLE}, "
                         f"got {s_lo}")
    if kind is ShockKind.CAPACITY_HIT:
        if (d_lo, d_hi) != (1, 1):
            raise ValueError(f"{where}: a capacity hit has duration 1, got [{d_lo}, {d_hi}]")
        if not (0.0 < m_lo and m_hi < 1.0):
            raise ValueError(f"{where}: capacity-hit magnitude must lie in (0, 1), "
                             f"got [{m_lo}, {m_hi}]")
    else:  # producer outage
        if d_lo < 1:
            raise ValueError(f"{where}: duration must be >= 1, got [{d_lo}, {d_hi}]")
        if (m_lo, m_hi) != (1.0, 1.0):
            raise ValueError(f"{where}: outage magnitude must be 1.0, got [{m_lo}, {m_hi}]")
    if len(set(g.targets)) != len(g.targets):
        raise ValueError(f"{where}: duplicate target producers {list(g.targets)}")
    for pid in g.targets:
        _check_producer(pid, producers, where)


def validate_shock_model(model: ShockModel, producers: Mapping[ProducerId, ProducerFacts]) -> None:
    """Raise ValueError naming the first bad entry (design D1 kind table). The kind check
    runs first, so an unbuilt kind is always reported as unsupported."""
    for i, e in enumerate(model.events):
        kind = getattr(e.kind, "value", e.kind)
        _check_event(e, producers, f"shock event #{i} ({kind})")
    seen: set[ShockKind] = set()
    for i, g in enumerate(model.generators):
        kind = getattr(g.kind, "value", g.kind)
        where = f"shock generator #{i} ({kind})"
        _check_generator(g, producers, where)
        if g.kind in seen:
            raise ValueError(f"{where}: at most one generator per kind")
        seen.add(g.kind)


# --- JSON form (camelCase, like the rest of the config schema) ---


def _kind_from(value: object, where: str) -> ShockKind:
    try:
        return ShockKind(value)
    except ValueError:
        known = ", ".join(k.value for k in ShockKind)
        raise ValueError(f"{where}: unknown shock kind {value!r}; known: {known}") from None


def _json_number(value: object) -> object:
    """JSON integers become floats (magnitude 1 == 1.0); anything else is passed through for
    validation to reject with an entry-naming error."""
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _pair(value: object, where: str, name: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{where}: {name} must be a [lo, hi] pair, got {value!r}")
    return (value[0], value[1])


def shock_model_from_dict(data: Mapping) -> ShockModel:
    events: list[ShockEvent] = []
    for i, e in enumerate(data.get("events", ())):
        where = f"shock event #{i}"
        t = e.get("target", {})
        events.append(ShockEvent(
            kind=_kind_from(e.get("kind"), where),
            target=ShockTarget(
                producer=t.get("producer"),
                capacity_types=tuple(t.get("capacityTypes", ())),
                product=t.get("product"),
                consumer=t.get("consumer"),
            ),
            start_cycle=e["startCycle"],
            duration=e["duration"],
            magnitude=_json_number(e["magnitude"]),
        ))
    generators: list[ShockGenerator] = []
    for i, g in enumerate(data.get("generators", ())):
        where = f"shock generator #{i}"
        m_lo, m_hi = _pair(g["magnitude"], where, "magnitude")
        generators.append(ShockGenerator(
            kind=_kind_from(g.get("kind"), where),
            count=g["count"],
            start_window=_pair(g["startWindow"], where, "startWindow"),
            duration=_pair(g["duration"], where, "duration"),
            magnitude=(_json_number(m_lo), _json_number(m_hi)),
            targets=tuple(g.get("targets", ())),
        ))
    return ShockModel(events=tuple(events), generators=tuple(generators))


def _target_to_dict(t: ShockTarget) -> dict:
    d: dict = {}
    if t.producer is not None:
        d["producer"] = t.producer
    if t.capacity_types:
        d["capacityTypes"] = list(t.capacity_types)
    if t.product is not None:
        d["product"] = t.product
    if t.consumer is not None:
        d["consumer"] = t.consumer
    return d


def shock_model_to_dict(model: ShockModel) -> dict:
    """Inverse of `shock_model_from_dict`."""
    return {
        "events": [
            {"kind": e.kind.value, "target": _target_to_dict(e.target),
             "startCycle": e.start_cycle, "duration": e.duration, "magnitude": e.magnitude}
            for e in model.events
        ],
        "generators": [
            {"kind": g.kind.value, "count": g.count, "startWindow": list(g.start_window),
             "duration": list(g.duration), "magnitude": list(g.magnitude),
             "targets": list(g.targets)}
            for g in model.generators
        ],
    }
