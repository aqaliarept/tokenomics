"""Seeded shock-schedule generation (ADR-0055; add-shocks-i-drift design D3;
stochastic-environment "Seeded event schedule", "Events are independent of agent actions").

The env draws an episode's shock schedule at `reset`, before any action, from the world
configuration and the episode seed only, and injects it into the domain, which replays it
and draws nothing.

Streams. Each shock kind draws from its own generator, seeded by
`sha256("tokenomics|shock:<kind>|<seed>")`, so enabling a kind never moves another kind's
schedule. The legacy demand stream (`np.random.default_rng(seed)` in
`TokenomicsAECEnv._build_consumers`) is never touched, so every existing demand draw is
preserved. Future kinds (and the drift stream) take their own stream names.

Draw order per event, fixed: start cycle (uniform on the window), duration (uniform on its
range), magnitude (uniform on its range, rounded to 4 decimals and kept inside the range),
target producer (uniform over the generator's targets, default every producer in config
order), then for a capacity hit one owned capacity type (uniform over the sorted owned
types).

Horizon. Generator windows live in the config, not in the horizon. The env merges explicit
and generated events and then drops events starting after the horizon, so the schedule of
a shorter horizon is a prefix of a longer one.

Digest. `shock_digest` is the sha256 hex of a canonical compact JSON of the injected
schedule (the empty schedule has its own digest). The gate compares it across subjects, as
it does `demand_digest`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np

from ..domain.config import Configuration
from ..domain.shocks import (ShockEvent, ShockGenerator, ShockKind, ShockModel, ShockTarget,
                             shock_model_to_dict)

MAGNITUDE_DECIMALS = 4


def derived_rng(seed: int, stream: str) -> np.random.Generator:
    """A generator for one named stream of one episode, independent of every other stream
    and of the legacy demand stream. Stable across processes (no `hash()`)."""
    digest = hashlib.sha256(f"tokenomics|{stream}|{seed}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def shock_stream(kind: ShockKind) -> str:
    return f"shock:{kind.value}"


def _magnitude(rng: np.random.Generator, lo: float, hi: float) -> float:
    m = round(float(rng.uniform(lo, hi)), MAGNITUDE_DECIMALS)
    return min(max(m, lo), hi)


def generate_events(gen: ShockGenerator, config: Configuration, seed: int) -> list[ShockEvent]:
    """`gen.count` events of one kind, drawn on that kind's stream (module docstring)."""
    rng = derived_rng(seed, shock_stream(gen.kind))
    owned = {p.id: sorted(p.capacities) for p in config.producer_configs}
    targets = gen.targets or tuple(p.id for p in config.producer_configs)
    (s_lo, s_hi), (d_lo, d_hi), (m_lo, m_hi) = gen.start_window, gen.duration, gen.magnitude
    events: list[ShockEvent] = []
    for _ in range(gen.count):
        start = int(rng.integers(s_lo, s_hi + 1))
        duration = int(rng.integers(d_lo, d_hi + 1))
        magnitude = _magnitude(rng, m_lo, m_hi)
        producer = targets[int(rng.integers(len(targets)))]
        if gen.kind is ShockKind.CAPACITY_HIT:
            types = owned[producer]
            target = ShockTarget(producer=producer,
                                 capacity_types=(types[int(rng.integers(len(types)))],))
        else:
            target = ShockTarget(producer=producer)
        events.append(ShockEvent(gen.kind, target, start, duration, magnitude))
    return events


def _sort_key(e: ShockEvent) -> tuple:
    return (e.start_cycle, e.kind.value, e.target.producer or "", e.target.capacity_types,
            e.duration, e.magnitude)


def build_schedule(config: Configuration, seed: int | None, horizon: int) -> tuple[ShockEvent, ...]:
    """The episode's shock schedule: explicit events plus every generator's draws, sorted by
    (start, kind, target, ...) and cut at the horizon. Empty when the config has no shocks.
    A generator needs an integer seed; an unseeded episode would not be reproducible."""
    model = config.shocks
    if model is None:
        return ()
    events = list(model.events)
    if model.generators:
        if seed is None:
            raise ValueError("a shock generator needs an integer episode seed; "
                             "reset(seed=None) would not be reproducible")
        for gen in model.generators:
            events.extend(generate_events(gen, config, int(seed)))
    events.sort(key=_sort_key)
    return tuple(e for e in events if e.start_cycle <= horizon)


def canonical_events(schedule: Sequence[ShockEvent]) -> list[dict]:
    """The schedule's events in the config JSON form, with numbers normalized so equal
    events encode identically: cycles and durations as int, magnitude as float (a direct
    `magnitude=1` and the loader's `1.0` compare equal and must digest equal)."""
    events = shock_model_to_dict(ShockModel(events=tuple(schedule)))["events"]
    for e in events:
        e["startCycle"], e["duration"] = int(e["startCycle"]), int(e["duration"])
        e["magnitude"] = float(e["magnitude"])
    return events


def shock_digest(schedule: Sequence[ShockEvent]) -> str:
    """sha256 hex of the canonical compact JSON (sorted keys, no spaces) of the schedule."""
    payload = json.dumps(canonical_events(schedule), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
