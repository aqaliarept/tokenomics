"""Named, versioned shock profiles for gate worlds (ADR-0055; add-shocks-i-drift design D7).

`eval/registry/shock_profiles_v1.json` holds the pre-registered profiles; its content hash
(the `eval/roster.py` pattern) is its version, and an edited file without a new hash is
refused. Registry v2 shocked worlds name a profile, and the registry records the file hash.

A profile resolves into a world config with `apply_shock_profile`:
- one seeded generator per kind, with every producer as a target and the count resolved
  per world as round_half_up(P * rate), rate kept as an exact fraction, so the shock
  pressure per producer is the same at every size (shocks-i: hits 2 at 12, 7 at 42;
  outages 1 at 12, 4 at 42);
- demand drift with the profile's j, only when the world has B9-randomized consumers;
- the resilience window and tolerance travel with the profile for the report.
A changed value is a new profile version, never an in-place edit after the v2 freeze.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from ..domain.config import Configuration, DemandDrift
from ..domain.shocks import ShockGenerator, ShockKind, ShockModel
from .resilience import ResilienceParams
from .roster import content_hash

DEFAULT_SHOCK_PROFILES = Path(__file__).parent / "registry" / "shock_profiles_v1.json"
SHOCKS_I = "shocks-i"


class ShockProfileHashMismatch(ValueError):
    """The profile file's recorded content hash does not match its contents."""


@dataclass(frozen=True)
class KindProfile:
    kind: ShockKind
    rate: tuple[int, int]  # events per producer per episode, as numerator / denominator
    start_window: tuple[int, int]
    duration: tuple[int, int]
    magnitude: tuple[float, float]

    def count(self, producers: int) -> int:
        """round_half_up(producers * num / den), exact."""
        num, den = self.rate
        return (2 * producers * num + den) // (2 * den)

    def generator(self, producers: int) -> ShockGenerator:
        return ShockGenerator(self.kind, count=self.count(producers),
                              start_window=self.start_window, duration=self.duration,
                              magnitude=self.magnitude)


@dataclass(frozen=True)
class ShockProfile:
    name: str
    kinds: tuple[KindProfile, ...]
    drift_j: float
    resilience: ResilienceParams


@dataclass(frozen=True)
class ShockProfiles:
    name: str
    profiles: Mapping[str, ShockProfile]
    version: str  # content hash of the file

    def get(self, name: str) -> ShockProfile:
        if name not in self.profiles:
            raise KeyError(f"unknown shock profile {name!r}; known: {sorted(self.profiles)}")
        return self.profiles[name]


def _pair(v: object, what: str) -> tuple:
    if not isinstance(v, list) or len(v) != 2:
        raise ValueError(f"{what} must be a [lo, hi] pair, got {v!r}")
    return (v[0], v[1])


def _profile_from_dict(name: str, d: Mapping) -> ShockProfile:
    kinds = []
    for kind_name in sorted(d["kinds"]):
        k = d["kinds"][kind_name]
        num, den = _pair(k["rate"], f"{name}.{kind_name}.rate")
        if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (num, den)) \
                or den == 0:
            raise ValueError(f"{name}.{kind_name}.rate must be non-negative ints with den > 0")
        m_lo, m_hi = _pair(k["magnitude"], f"{name}.{kind_name}.magnitude")
        kinds.append(KindProfile(
            kind=ShockKind(kind_name), rate=(num, den),
            start_window=_pair(k["start_window"], f"{name}.{kind_name}.start_window"),
            duration=_pair(k["duration"], f"{name}.{kind_name}.duration"),
            magnitude=(float(m_lo), float(m_hi)),
        ))
    r = d["resilience"]
    return ShockProfile(name=name, kinds=tuple(kinds), drift_j=float(d["drift_j"]),
                        resilience=ResilienceParams(window=r["window"],
                                                    tolerance=float(r["tolerance"])))


def shock_profiles_from_dict(d: Mapping) -> ShockProfiles:
    d = dict(d)
    recorded = d.pop("content_hash", None)
    if recorded is None:
        raise ShockProfileHashMismatch("shock profile file records no content_hash")
    actual = content_hash(d)
    if actual != recorded:
        raise ShockProfileHashMismatch(
            f"shock profile content hash {actual} != recorded {recorded}: edited without a "
            "new version")
    profiles = {n: _profile_from_dict(n, p) for n, p in d["profiles"].items()}
    return ShockProfiles(name=d["name"], profiles=profiles, version=recorded)


def load_shock_profiles(path: str | Path = DEFAULT_SHOCK_PROFILES) -> ShockProfiles:
    return shock_profiles_from_dict(json.loads(Path(path).read_text()))


def load_shock_profile(name: str, path: str | Path = DEFAULT_SHOCK_PROFILES) -> ShockProfile:
    return load_shock_profiles(path).get(name)


def apply_shock_profile(cfg: Configuration, profile: ShockProfile, producers: int) -> Configuration:
    """The world `cfg` with the profile's generators (counts resolved for `producers`) and,
    when it has B9-randomized consumers, the profile's demand drift. Validated."""
    if producers != len(cfg.producer_configs):
        raise ValueError(f"profile resolved for {producers} producers, world has "
                         f"{len(cfg.producer_configs)}")
    generators = tuple(k.generator(producers) for k in profile.kinds)
    drift = (DemandDrift(profile.drift_j) if any(c.is_randomized for c in cfg.consumers)
             else None)
    out = replace(cfg, shocks=ShockModel(generators=generators), demand_drift=drift)
    out.validate()
    return out


def _write_v1(path: str | Path = DEFAULT_SHOCK_PROFILES) -> str:
    """Regenerate shock_profiles_v1.json from the D7 values (pre-registered, add-shocks-i-drift
    design D7). Returns the content hash."""
    payload = {
        "name": "shock-profiles-v1",
        "profiles": {
            SHOCKS_I: {
                "kinds": {
                    "capacity_hit": {"rate": [1, 6], "start_window": [21, 90],
                                     "duration": [1, 1], "magnitude": [0.3, 0.6]},
                    "producer_outage": {"rate": [1, 12], "start_window": [21, 90],
                                        "duration": [3, 6], "magnitude": [1.0, 1.0]},
                },
                "drift_j": 0.34,
                "resilience": {"window": 5, "tolerance": 0.05},
            },
        },
    }
    payload["content_hash"] = content_hash(payload)
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return payload["content_hash"]


if __name__ == "__main__":
    print(_write_v1())
