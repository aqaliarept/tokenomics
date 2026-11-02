"""World registry: the versioned list of gate and development worlds (design D7,
add-measurement-gate-harness; gate-harness "World registry", "Regimes", "Zero-baseline screening",
"Sealed worlds need a pre-registration").

Each entry is a task-archive generator ref (`dense_family` / `sparse_wide_family`, task 6.1) plus
its split, regime, size, family, world seed, episode seeds and status. The version is a sha256
content hash over every field but the hash itself (the `eval/roster.py` pattern): an edited file
without a new hash is refused on load, so a report naming the version names exactly its worlds.

Seed scheme (`build_worlds_v1`, the draft generator):
- World seeds are allocated sequentially from the split's range (development 1000-1999, sealed
  5000-5999) in a fixed order (family in `SEED_ORDER`, size, regime, gate worlds then reserves),
  so every world seed is unique in the registry and the two splits are disjoint by construction.
  Only the M1 gate families (`GATE_FAMILIES`: sparse-wide, task 10.4) are emitted; dense keeps
  its seed block reserved, so the sparse-wide worlds are the ones EXP-0044 attempt 1 screened.
- Episode seed j of world `id` is `int(sha256(f"episode|{id}|{j}")[:8], 16) % 2**31`: different
  across worlds (episodes are not correlated between worlds), stable across processes, and
  checked on load to be unique registry-wide and disjoint from every world seed.

Statuses: `gate` (in the gate set), `reserve` (ordered by `reserve_position` within its
(split, family, size, regime) group, used only by zero-baseline replacement) and `replaced`
(a zero-baseline world taken out by screening). Screening is pure over roster-only episode rows
and records its outcome in the registry, so a screened registry has a new version. Screening is by
split and records the splits it covered: M1 (tasks 8.2/10.1) screens development only; the sealed
split gets its own roster-only screen in M5, before any candidate touches a sealed world, and a
report on a split requires that split to be screened. The development headroom check (D10) is
recorded in the same `ScreeningRecord` against the pre-registered `saturation_ceiling`.

Sealed worlds are gated: `check_sealed_access` refuses them without a pre-registration manifest
whose checkpoint hashes (via `eval/provenance.py`) and roster / registry / parameter versions match,
and appends every authorized sealed evaluation to `experiments/SEALED_ACCESS.log`. Roster-only
screening is not a candidate evaluation and needs no manifest.
"""

from __future__ import annotations

import argparse
import copy
import csv
import functools
import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

from ..domain.config import Configuration
from .families import (CALM, DENSE, FAMILIES, NEUTRAL_CAPACITY_SCALE, REGIMES, SHOCKED,
                       SPARSE_WIDE, check_capacity_scale)
from .gate import CANDIDATE, EpisodeRow, Headroom, episode_baseline, world_baseline
from .provenance import assert_inputs_match, sha256_file
from .roster import GATE_SIZES, Roster, check_screened_on, content_hash, name_version_tag
from .seeding import replicate_mean
from .task_archive import REPRODUCERS

REGISTRY_V1 = Path(__file__).parent / "registry" / "worlds_v1.json"
REGISTRY_V2 = Path(__file__).parent / "registry" / "worlds_v2.json"
# The default is the frozen v2 set since task 10.1 (add-shocks-i-drift D8); v1 stays selectable
# by path. Both packaged files are screens: the drafts come from build_worlds_v1/v2.
DEFAULT_REGISTRY = REGISTRY_V2
V2_SHOCK_PROFILE = "shocks-i"  # add-shocks-i-drift D8: the profile every v2 shocked world names
SEALED_ACCESS_LOG = Path(__file__).resolve().parents[3] / "experiments" / "SEALED_ACCESS.log"

DEVELOPMENT = "development"
SEALED = "sealed"
SPLITS = (DEVELOPMENT, SEALED)
GATE = "gate"
RESERVE = "reserve"
REPLACED = "replaced"
# add-gate-v3 D3 step 4: a family x size that failed development headroom, in both splits (v3+)
EXCLUDED = "excluded"
STATUSES = (GATE, RESERVE, REPLACED, EXCLUDED)

# Region families (add-gate-v3 D2): world-space draws, from registry v3 on.
WS_LOADED = "ws_loaded"
WS_WIDE = "ws_wide"
REGION_FAMILIES = (WS_LOADED, WS_WIDE)
WORLD_SPACE_GENERATOR = "world_space"
FAMILY_GENERATORS = {DENSE: "dense_family", SPARSE_WIDE: "sparse_wide_family",
                     WS_LOADED: WORLD_SPACE_GENERATOR, WS_WIDE: WORLD_SPACE_GENERATOR}
# Seed allocation order of the draft generator: every defined family keeps its seed block, so a
# family's worlds never move when another family is (not) emitted.
SEED_ORDER = (DENSE, SPARSE_WIDE)
# Families the M1 gate admits (task 10.4, user decision 2026-10-27). Dense stays buildable but
# non-gating: EXP-0044 found dense-12 saturated (median best fill 0.954 > 0.95) and the 10.3
# probe found capacity is not binding there (the shortfall is mostly warm-up; DIAGNOSTICS 18).
GATE_FAMILIES = (SPARSE_WIDE,)
GATE_FAMILIES_V3 = (SPARSE_WIDE, WS_LOADED, WS_WIDE)  # add-gate-v3 D2 / ADR-0060 §2
# The sizes each gate family is declared at (gate-harness "Scale families"); coverage checks
# each family at these from v3 on.
FAMILY_SIZES = {SPARSE_WIDE: GATE_SIZES, WS_LOADED: (12,), WS_WIDE: (42,)}
SIZES = GATE_SIZES  # the registry holds worlds at exactly the gate sizes
CALM_PER_SHOCKED = 2  # gate-harness "Regimes": calm >= 2x shocked at each size

# D7 defaults: split -> (seed range, {regime: (worlds, episode seeds per world)}).
SPLIT_SEED_RANGES = {DEVELOPMENT: (1000, 1999), SEALED: (5000, 5999)}
WORLD_COUNTS = {
    DEVELOPMENT: {CALM: (8, 3), SHOCKED: (4, 10)},
    SEALED: {CALM: (16, 3), SHOCKED: (8, 10)},
}
RESERVES_PER_GROUP = 4  # per (split, family, size, regime)
DEFAULT_SATURATION_CEILING = 0.95  # D10: headroom ceiling on the hindsight-baseline fill rate
SPARSE_FANOUT = 2
CALM_SCHEDULE_LENGTH = 7

# Region draws (add-gate-v3 D2): the choice levels per family (ws_loaded alternates them by slot
# index within each (split, regime) group, gate worlds first), the Latin-hypercube pool per
# (split, size), the measured-rho acceptance band (lo, hi], and the counts per split.
REGION_CHOICES = {WS_LOADED: ("dense", "sparse_f4"), WS_WIDE: ("sparse_f14",)}
REGION_POOL = 128
RHO_BAND = (0.28, 3.1)
REGION_WORLD_COUNTS = {
    DEVELOPMENT: {CALM: (4, 3), SHOCKED: (2, 10)},
    SEALED: {CALM: (8, 3), SHOCKED: (4, 10)},
}
# Seeds region worlds must avoid besides the registries v1 / v2 (gate-harness "Region worlds
# are fresh draws ..."): M4's training-draw stream (`train.order_es_worlds.FRESH_ATLAS`, pinned
# by a test without importing the training module), its first 4,096 indices at each size.
M4_TRAIN_STREAM = "m4-es-train-v1"
M4_RESERVED_INDICES = 4096


class RegistryHashMismatch(ValueError):
    """The registry's recorded content hash does not match its contents."""


class SealedAccessError(PermissionError):
    """A sealed world was requested without a matching pre-registration."""


# --- entries ---


@dataclass(frozen=True)
class WorldEntry:
    """One registry world. `generator` is a task-archive generator ref
    (`{"generator": name, "params": {...}}`) that `config()` replays."""

    id: str
    split: str
    family: str
    size: int
    regime: str
    world_seed: int
    episode_seeds: tuple[int, ...]
    generator: dict
    status: str = GATE
    reserve_position: int | None = None
    # Region entries only (add-gate-v3 D2): {"pool": n, "record": world_space.world_record}, the
    # draw's hypercube pool size and the world record. None (and omitted from the payload, so v1 /
    # v2 hashes are unchanged) on legacy entries.
    region: dict | None = None

    @property
    def group(self) -> tuple[str, str, int, str]:
        """Reserve group: replacements stay within (split, family, size, regime)."""
        return (self.split, self.family, self.size, self.regime)

    def config(self) -> Configuration:
        """Replay the world through the vetted task-archive reproducer."""
        name = self.generator["generator"]
        if name not in REPRODUCERS:
            raise KeyError(f"{self.id}: unknown generator {name!r}")
        cfg = REPRODUCERS[name](self.world_seed, **self.generator.get("params", {}))
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return {
            "id": self.id, "split": self.split, "family": self.family, "size": self.size,
            "regime": self.regime, "world_seed": self.world_seed,
            "episode_seeds": list(self.episode_seeds), "generator": copy.deepcopy(self.generator),
            "status": self.status, "reserve_position": self.reserve_position,
        } | ({} if self.region is None else {"region": copy.deepcopy(self.region)})

    @classmethod
    def from_dict(cls, d: dict) -> WorldEntry:
        return cls(
            id=d["id"], split=d["split"], family=d["family"], size=d["size"],
            regime=d["regime"], world_seed=d["world_seed"],
            episode_seeds=tuple(d["episode_seeds"]), generator=copy.deepcopy(d["generator"]),
            status=d["status"], reserve_position=d["reserve_position"],
            region=copy.deepcopy(d.get("region")),
        )


@dataclass(frozen=True)
class Replacement:
    replaced: str
    by: str
    skipped: tuple[str, ...]  # reserves consulted before `by` that were themselves zero

    def to_dict(self) -> dict:
        return {"replaced": self.replaced, "by": self.by, "skipped": list(self.skipped)}


@dataclass(frozen=True)
class Exclusion:
    """A family × size the v3 development screen did not admit (add-gate-v3 D3 step 4), with the
    headroom result that excluded it. Every entry of it, in both splits, has status `excluded`."""

    family: str
    size: int
    headroom: Headroom

    def to_dict(self) -> dict:
        return {"family": self.family, "size": self.size, "headroom": self.headroom.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> Exclusion:
        return cls(d["family"], d["size"], Headroom(**d["headroom"]))


@dataclass(frozen=True)
class ScreeningRecord:
    """Zero-baseline screening outcome (gate-harness "Zero-baseline screening"): the roster it ran,
    the splits it covered, the world baseline means it saw (gate and consulted reserve worlds), the
    replacements, and (once recorded) the D10 headroom check per family × size."""

    roster_version: str
    baseline_means: tuple[tuple[str, float], ...]
    replacements: tuple[Replacement, ...]
    splits: tuple[str, ...] = (DEVELOPMENT,)
    headroom: tuple[Headroom, ...] | None = None
    # v3+ (add-gate-v3 D3 step 4); None and omitted from the payload otherwise (hash unchanged).
    exclusions: tuple[Exclusion, ...] | None = None

    def to_dict(self) -> dict:
        return {
            "roster_version": self.roster_version,
            "baseline_means": {k: v for k, v in self.baseline_means},
            "replacements": [r.to_dict() for r in self.replacements],
            "splits": list(self.splits),
            "headroom": None if self.headroom is None else [h.to_dict() for h in self.headroom],
        } | ({} if self.exclusions is None
             else {"exclusions": [x.to_dict() for x in self.exclusions]})

    def excluded(self) -> set[tuple[str, int]]:
        """The excluded (family, size) pairs."""
        return {(x.family, x.size) for x in self.exclusions or ()}

    @classmethod
    def from_dict(cls, d: dict) -> ScreeningRecord:
        return cls(
            roster_version=d["roster_version"],
            baseline_means=tuple(sorted((k, float(v)) for k, v in d["baseline_means"].items())),
            replacements=tuple(Replacement(r["replaced"], r["by"], tuple(r["skipped"]))
                               for r in d["replacements"]),
            splits=tuple(d["splits"]),
            headroom=None if d["headroom"] is None else tuple(Headroom(**h) for h in d["headroom"]),
            exclusions=(None if d.get("exclusions") is None
                        else tuple(Exclusion.from_dict(x) for x in d["exclusions"])),
        )


@dataclass(frozen=True)
class Registry:
    """A versioned world registry. Entries are kept sorted by id so the hash identifies the world
    set, not a file's line order; `version` is the content hash."""

    name: str
    seed_ranges: tuple[tuple[str, int, int], ...]
    entries: tuple[WorldEntry, ...]
    screening: ScreeningRecord | None = None
    saturation_ceiling: float = DEFAULT_SATURATION_CEILING
    # Content hash of the shock-profile file the entries' `shock_profile` names resolve in
    # (add-shocks-i-drift D8). None for v1, which names no profile; omitted from the payload
    # then, so the v1 hash is unchanged.
    shock_profiles: str | None = None
    # Hash of the registry draft a v2+ screen read, recorded by that screen (add-shocks-i-drift
    # D8/D10). It pairs the frozen registry with the roster and parameters of the same screen;
    # the frozen registry's own hash differs, since screening writes its record into the file.
    # None on v1 and on unscreened drafts; omitted from the payload then (hash unchanged).
    screened_on: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(sorted(self.entries, key=lambda e: e.id)))
        object.__setattr__(self, "seed_ranges", tuple(sorted(self.seed_ranges)))
        self.validate()

    def validate(self) -> None:
        """gate-harness "World registry" and "Regimes": unique ids; every field well-formed and
        consistent with its generator ref; world seeds inside their split's range, unique per
        (family, size) and disjoint across splits per family; episode seeds non-empty, unique
        registry-wide and disjoint from world seeds; at each (split, size), gate calm worlds >= 2x
        gate shocked worlds; reserve positions contiguous per group.

        From v3 (add-gate-v3 D2, D3 step 4): region families (`ws_loaded` at 12, `ws_wide` at
        42) carry their draw and world record and are checked by set disjointness instead of the
        seed range (`_check_region`); the legacy dense family is refused; `excluded` entries and
        recorded exclusions must agree (`_check_exclusions`). A v1 / v2 registry holding a region
        family, an `excluded` entry or a recorded exclusion is refused."""
        if not 0 < self.saturation_ceiling <= 1:
            raise ValueError(f"saturation ceiling must be in (0, 1], got {self.saturation_ceiling}")
        if self.screened_on is not None:
            if self.screening is None or self.version_tag < 2:
                raise ValueError(f"registry {self.name!r}: screened_on is recorded only on a "
                                 f"screened v2+ registry (v{self.version_tag})")
            check_screened_on(self.screened_on, f"registry {self.name!r}")
        if self.screening is not None and (not self.screening.splits
                                           or not set(self.screening.splits) <= set(SPLITS)):
            raise ValueError(f"screened splits {self.screening.splits} not a subset of {SPLITS}")
        for h in (self.screening.headroom or ()) if self.screening is not None else ():
            _check_headroom(h, self.saturation_ceiling)
        ranges = {s: (lo, hi) for s, lo, hi in self.seed_ranges}
        if set(ranges) != set(SPLITS):
            raise ValueError(f"seed ranges must cover exactly {SPLITS}, got {sorted(ranges)}")
        (dlo, dhi), (slo, shi) = ranges[DEVELOPMENT], ranges[SEALED]
        if not (dhi < slo or shi < dlo):
            raise ValueError("development and sealed seed ranges overlap")
        ids = [e.id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate world ids: {sorted({i for i in ids if ids.count(i) > 1})}")
        seen_seed: dict[tuple[str, int, int], str] = {}
        split_of: dict[tuple[str, int], str] = {}
        episode_owner: dict[int, str] = {}
        world_seeds = {e.world_seed for e in self.entries}
        tag = self.version_tag
        self._check_region_seeds()  # before the per-entry checks, so the clash names its source
        for e in self.entries:
            if e.split not in SPLITS:
                raise ValueError(f"{e.id}: unknown split {e.split!r}")
            if e.family not in FAMILY_GENERATORS:
                raise ValueError(f"{e.id}: unknown family {e.family!r}")
            if e.regime not in REGIMES:
                raise ValueError(f"{e.id}: world declares no valid regime ({e.regime!r})")
            if e.status not in STATUSES:
                raise ValueError(f"{e.id}: unknown status {e.status!r}")
            if tag < 3 and (e.family in REGION_FAMILIES or e.status == EXCLUDED
                            or e.region is not None):
                raise ValueError(f"{e.id}: region families, `excluded` entries and region "
                                 f"records exist from registry v3 on (this is v{tag})")
            if tag >= 3 and e.family == DENSE:
                raise ValueError(f"{e.id}: the legacy dense family is outside every gate "
                                 "registry from v3 on")
            if e.family in REGION_FAMILIES:
                self._check_region_entry(e)
            else:
                if e.region is not None:
                    raise ValueError(f"{e.id}: a {e.family} entry carries no region record")
                other = split_of.setdefault((e.family, e.world_seed), e.split)
                if other != e.split:
                    raise ValueError(f"family {e.family} world seed {e.world_seed} is listed "
                                     f"under both {other} and {e.split}")
                lo, hi = ranges[e.split]
                if not lo <= e.world_seed <= hi:
                    raise ValueError(f"{e.id}: world seed {e.world_seed} outside {e.split} "
                                     f"range [{lo}, {hi}]")
            key = (e.family, e.size, e.world_seed)
            if key in seen_seed:
                raise ValueError(f"{e.id} and {seen_seed[key]} are the same world {key}")
            seen_seed[key] = e.id
            if e.family not in REGION_FAMILIES:
                self._check_generator(e)
            if not e.episode_seeds:
                raise ValueError(f"{e.id}: no episode seeds")
            for s in e.episode_seeds:
                if s in world_seeds:
                    raise ValueError(f"{e.id}: episode seed {s} is also a world seed")
                if s in episode_owner:
                    raise ValueError(f"episode seed {s} shared by {episode_owner[s]} and {e.id}")
                episode_owner[s] = e.id
        self._check_reserves()
        self._check_regime_balance()
        self._check_shock_profiles()
        self._check_exclusions()

    @property
    def version_tag(self) -> int:
        """The registry generation from its name (`worlds-v2`, also a derived scope such as
        `worlds-v2-dry-run`, -> 2); 1 when the name carries none."""
        return name_version_tag(self.name)

    @staticmethod
    def _check_region_entry(e: WorldEntry) -> None:
        """A region entry (add-gate-v3 D2): its family's size; a `world_space` generator ref with
        params exactly {producers, levers, shock_window} plus `shock_profile` on shocked worlds;
        the family's choice; calm = shocks off + cyclic, shocked = `shocks-i` (or the test-only
        `shocks-i-smoke` with its window) with its drift at the neutral shock levers; a region
        record whose world record names this world (stream `gate_v3_stream(split)`, seed,
        levers, window, generator version), whose levers are row `index` of `region_sample` at
        the recorded pool, and whose measured rho lies in `RHO_BAND`. The config hash is replayed
        on load (`check_replay`)."""
        from .world_space import (CYCLIC, GENERATOR_VERSION, Levers, check_levers,
                                  check_shock_profile, gate_v3_stream)

        want_size = FAMILY_SIZES[e.family][0]
        if e.size != want_size:
            raise ValueError(f"{e.id}: a {e.family} world has {want_size} producers, the entry "
                             f"declares {e.size}")
        g = e.generator
        if not isinstance(g, dict) or set(g) != {"generator", "params"} \
                or g.get("generator") != WORLD_SPACE_GENERATOR:
            raise ValueError(f"{e.id}: a region entry needs a {WORLD_SPACE_GENERATOR!r} generator "
                             f"ref, got {g!r}")
        params = g["params"]
        keys = {"producers", "levers", "shock_window"} | (
            {"shock_profile"} if e.regime == SHOCKED else set())
        if not isinstance(params, dict) or set(params) != keys:
            raise ValueError(f"{e.id}: region generator params "
                             f"{sorted(params) if isinstance(params, dict) else params!r} != "
                             f"{sorted(keys)}")
        if params["producers"] != e.size or isinstance(params["producers"], bool):
            raise ValueError(f"{e.id}: generator producers {params['producers']!r} != size {e.size}")
        w = params["shock_window"]
        if not (isinstance(w, list) and len(w) == 2
                and all(isinstance(x, int) and not isinstance(x, bool) for x in w)):
            raise ValueError(f"{e.id}: shock_window must be a [start, end] pair of ints, got {w!r}")
        if not isinstance(params["levers"], dict):
            raise ValueError(f"{e.id}: levers must be an object")
        levers = check_levers(e.size, Levers.from_dict(params["levers"]))
        if levers.to_dict() != params["levers"]:
            raise ValueError(f"{e.id}: levers are not in canonical form (every lever written)")
        if levers.choice not in REGION_CHOICES[e.family]:
            raise ValueError(f"{e.id}: {e.family} choice is one of {REGION_CHOICES[e.family]}, "
                             f"got {levers.choice!r}")
        if e.regime == CALM:
            if levers.shocks or levers.pattern != CYCLIC:
                raise ValueError(f"{e.id}: a calm region world has shocks off and the {CYCLIC} "
                                 f"pattern, got shocks={levers.shocks}, {levers.pattern!r}")
        else:
            name = params["shock_profile"]
            if not isinstance(name, str):
                raise ValueError(f"{e.id}: shock_profile must be a profile name")
            check_shock_profile(name, levers=levers, shock_window=w)
        region = e.region
        if not isinstance(region, dict) or set(region) != {"pool", "record"}:
            raise ValueError(f"{e.id}: a region entry records {{pool, record}}, got {region!r}")
        pool, rec = region["pool"], region["record"]
        if isinstance(pool, bool) or not isinstance(pool, int) or pool < 1:
            raise ValueError(f"{e.id}: region pool must be a positive int, got {pool!r}")
        stream = gate_v3_stream(e.split)
        want = {"atlas": stream, "size": e.size, "seed": e.world_seed,
                "generator_version": GENERATOR_VERSION, "levers": params["levers"],
                "shock_window": w}
        got = {k: rec.get(k) for k in want} if isinstance(rec, dict) else None
        if got != want:
            raise ValueError(f"{e.id}: world record {got} does not describe this entry {want}")
        index = rec.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < pool:
            raise ValueError(f"{e.id}: record index {index!r} outside the pool of {pool}")
        drawn = _region_draw(stream, e.size, pool, levers.choice, levers.shocks,
                             levers.pattern)[index]
        if drawn.seed != e.world_seed or drawn.levers != levers:
            raise ValueError(f"{e.id}: levers / seed are not draw {index} of {stream} "
                             f"(size {e.size}, pool {pool})")
        rho = (rec.get("coordinates") or {}).get("rho")
        lo, hi = RHO_BAND
        if not isinstance(rho, (int, float)) or not lo < rho <= hi:
            raise ValueError(f"{e.id}: recorded rho {rho!r} outside ({lo}, {hi}]")

    def _check_region_seeds(self) -> None:
        """Region world seeds by explicit sets (gate-harness "World registry", "Region worlds are
        fresh draws ..."): disjoint across splits (naming both entries), and no region world or
        episode seed equals a seed of the registries v1 / v2, the atlas-v1 sample or its EXP-0050
        anchors (with their episode seeds), or the first 4,096 `m4-es-train-v1` draws."""
        region = [e for e in self.entries if e.family in REGION_FAMILIES]
        if not region:
            return
        by_seed: dict[int, WorldEntry] = {}
        for e in region:
            other = by_seed.setdefault(e.world_seed, e)
            if other is not e and other.split != e.split:
                raise ValueError(f"region world seed {e.world_seed} is under both splits: "
                                 f"{other.id} ({other.split}) and {e.id} ({e.split})")
        reserved = region_reserved_seeds()
        for e in region:
            for what, seed in [("world", e.world_seed)] + [("episode", s) for s in e.episode_seeds]:
                if seed in reserved:
                    raise ValueError(f"{e.id}: {what} seed {seed} is reserved by "
                                     f"{reserved[seed]}")

    def _check_exclusions(self) -> None:
        """add-gate-v3 D3 step 4: exclusions exist from v3 on and only on a screened registry with
        a headroom record; each names a family × size whose recorded headroom failed, with that
        result; every entry of an excluded family × size (both splits, gate and reserve) has
        status `excluded`, and every `excluded` entry belongs to a recorded exclusion."""
        sc = self.screening
        exclusions = sc.exclusions if sc is not None else None
        flagged = {(e.family, e.size) for e in self.entries if e.status == EXCLUDED}
        if exclusions is None:
            if flagged:
                raise ValueError(f"entries of {sorted(flagged)} are excluded but the registry "
                                 "records no exclusion")
            return
        if self.version_tag < 3:
            raise ValueError(f"registry {self.name!r} records exclusions; they exist from v3 on")
        if sc.headroom is None:
            raise ValueError("exclusions are recorded with the headroom check that decided them")
        heads = {(h.family, h.size): h for h in sc.headroom}
        pairs = [(x.family, x.size) for x in exclusions]
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"duplicate exclusions {pairs}")
        for x in exclusions:
            h = heads.get((x.family, x.size))
            if h is None or h.passed or h != x.headroom:
                raise ValueError(f"exclusion {x.family}/{x.size} does not match a failed headroom "
                                 f"result (recorded {h}, exclusion {x.headroom})")
        for e in self.entries:
            if ((e.family, e.size) in set(pairs)) != (e.status == EXCLUDED):
                raise ValueError(f"{e.id}: status {e.status!r} disagrees with the exclusions "
                                 f"{sorted(pairs)}")

    def _check_shock_profiles(self) -> None:
        """gate-harness "Regimes" from v2 on: every shocked entry (gate and reserve, since a
        reserve can replace a gate world) names a shock profile, and the registry records the
        hash of the profile file it resolves in. Calm entries never name one (_check_generator)."""
        named = [e for e in self.entries if "shock_profile" in e.generator["params"]]
        if named and self.shock_profiles is None:
            raise ValueError(f"{named[0].id}: names a shock profile but the registry records no "
                             "shock_profiles hash")
        if self.version_tag >= 2:
            for e in self.entries:
                if e.regime == SHOCKED and "shock_profile" not in e.generator["params"]:
                    raise ValueError(f"{e.id}: a registry v{self.version_tag} shocked world "
                                     "names no shock profile")

    @staticmethod
    def _check_generator(e: WorldEntry) -> None:
        """Schema of the generator ref (cheap, on every construction): exactly {generator,
        params}; the family's generator; params keys from the builder's signature, int-valued
        except `capacity_scale`; `fanout` only (and always) for sparse, `schedule_length` only
        (and always) for calm, an optional non-neutral `capacity_scale` (a number in (0, 2],
        task 10.2), and producers/regime equal to the entry's. Value constraints the builder enforces (fanout
        range, producers a multiple of 3, schedule_length >= 1) are checked by replay on load
        (`check_replay`)."""
        g = e.generator
        if not isinstance(g, dict) or set(g) != {"generator", "params"}:
            raise ValueError(f"{e.id}: generator ref must be exactly {{generator, params}}, got {g!r}")
        want = FAMILY_GENERATORS[e.family]
        if g.get("generator") != want:
            raise ValueError(f"{e.id}: generator {g.get('generator')!r} != {want!r} for {e.family}")
        params = g["params"]
        if not isinstance(params, dict):
            raise ValueError(f"{e.id}: generator params must be an object")
        keys = {"producers", "regime"}
        if e.family == SPARSE_WIDE:
            keys.add("fanout")
        if e.regime == CALM:
            keys.add("schedule_length")
        if "shock_profile" in params:
            if e.regime != SHOCKED:
                raise ValueError(f"{e.id}: a {e.regime} world names shock profile "
                                 f"{params['shock_profile']!r}; calm worlds carry no shocks")
            if not isinstance(params["shock_profile"], str) or not params["shock_profile"]:
                raise ValueError(f"{e.id}: shock_profile must be a profile name")
            keys.add("shock_profile")
        if "capacity_scale" in params:
            # optional, and only in canonical form: absent means neutral (task 10.2)
            v = params["capacity_scale"]
            if check_capacity_scale(v) == NEUTRAL_CAPACITY_SCALE:
                raise ValueError(f"{e.id}: a neutral capacity_scale is written by omission")
            keys.add("capacity_scale")
        if set(params) != keys:
            raise ValueError(f"{e.id}: generator params {sorted(params)} != {sorted(keys)}")
        for k in keys - {"regime", "capacity_scale", "shock_profile"}:
            v = params[k]
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{e.id}: generator param {k} must be an int, got {v!r}")
        if params.get("producers") != e.size:
            raise ValueError(f"{e.id}: generator producers {params.get('producers')} != size {e.size}")
        if params.get("regime") != e.regime:
            raise ValueError(f"{e.id}: generator regime {params.get('regime')!r} != {e.regime!r}")

    def _check_reserves(self) -> None:
        by_group: dict[tuple[str, str, int, str], list[int]] = {}
        for e in self.entries:
            if e.status == RESERVE and e.reserve_position is None:
                raise ValueError(f"{e.id}: a reserve needs a reserve_position")
            if self.screening is None and e.status == REPLACED:
                raise ValueError(f"{e.id}: only screening replaces a world")
            if e.status == GATE and e.reserve_position is not None and self.screening is None:
                raise ValueError(f"{e.id}: an unscreened gate world has no reserve position")
            if e.reserve_position is not None:
                by_group.setdefault(e.group, []).append(e.reserve_position)
        for group, pos in by_group.items():
            if sorted(pos) != list(range(len(pos))):
                raise ValueError(f"reserve positions of {group} are {sorted(pos)}, not 0..n-1")

    def _check_regime_balance(self) -> None:
        counts: dict[tuple[str, int], dict[str, int]] = {}
        for e in self.gate_worlds():
            c = counts.setdefault((e.split, e.size), {CALM: 0, SHOCKED: 0})
            c[e.regime] += 1
        for (split, size), c in sorted(counts.items()):
            if c[CALM] < CALM_PER_SHOCKED * c[SHOCKED]:
                raise ValueError(f"{split} size {size}: {c[CALM]} calm worlds < "
                                 f"{CALM_PER_SHOCKED}x {c[SHOCKED]} shocked")

    def check_replay(self) -> None:
        """Replay every entry's generator ref into a validated Configuration (gate-harness "World
        registry": a replayable generator reference). ~0.5 s for the draft; run on load."""
        for e in self.entries:
            try:
                cfg = e.config()
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{e.id}: generator ref does not replay: {exc}") from exc
            if len(cfg.producer_configs) != e.size:
                raise ValueError(f"{e.id}: replays {len(cfg.producer_configs)} producers, "
                                 f"declares size {e.size}")
            if e.regime == CALM and (cfg.shocks is not None or cfg.demand_drift is not None):
                raise ValueError(f"{e.id}: a calm world replays with shocks or demand drift")
            if e.region is not None:
                from .routing_audit import config_digest

                digest = config_digest(cfg)
                if digest != e.region["record"]["config_sha256"]:
                    raise ValueError(f"{e.id}: replayed config sha256 {digest} != recorded "
                                     f"{e.region['record']['config_sha256']}")
        if self.shock_profiles is not None:
            from .shock_profiles import load_shock_profiles

            current = load_shock_profiles().version
            if current != self.shock_profiles:
                raise ValueError(f"registry records shock profiles {self.shock_profiles}, the "
                                 f"packaged profile file is {current}")

    def coverage_problems(self, sizes: Collection[int] = GATE_SIZES,
                          gate_families: Collection[str] | None = None) -> list[str]:
        """D10 admits a family only on development evidence at both sizes: every family with gate
        worlds in any split (a sealed-only family included) must have development gate worlds at
        every gate size. Only the M1 gate families may appear (task 10.4: dense is non-gating),
        and every one of them must be present.

        From v3 (add-gate-v3 D2, D3 step 4) the gate families are `GATE_FAMILIES_V3`, each is
        required at its declared sizes (`FAMILY_SIZES`), and a recorded exclusion is a family ×
        size the registry does not declare. v1 / v2 output is unchanged."""
        v3 = self.version_tag >= 3
        if gate_families is None:
            gate_families = GATE_FAMILIES_V3 if v3 else GATE_FAMILIES
        have = {(e.family, e.size) for e in self.gate_worlds(DEVELOPMENT)}
        present = {e.family for e in self.entries}
        families = sorted(present | set(gate_families))
        where = f"registry v{self.version_tag}" if v3 else "M1"
        problems = [f"family {f} is not a gate family in {where} (gate families: "
                    f"{', '.join(gate_families)})" for f in sorted(present - set(gate_families))]
        if not v3:
            return problems + [f"no development gate worlds for {f}/{z}"
                               for f in families for z in sorted(sizes) if (f, z) not in have]
        excluded = self.screening.excluded() if self.screening is not None else set()
        return problems + [f"no development gate worlds for {f}/{z}"
                           for f in families for z in sorted(sizes)
                           if z in FAMILY_SIZES.get(f, GATE_SIZES)
                           and (f, z) not in have and (f, z) not in excluded]

    def admission_problems(self, split: str, sizes: Collection[int] = GATE_SIZES) -> list[str]:
        """Why a real report on `split` may not run on this registry (empty = admitted): the split
        must be screened, every family must have development gate worlds at every gate size, and
        the development headroom check (D10, the family admission) must be recorded for every
        family × size and pass."""
        sc = self.screening
        if sc is None or split not in sc.splits:
            return [f"registry {self.version[:12]} has not screened the {split} split"]
        if DEVELOPMENT not in sc.splits:
            return [f"registry {self.version[:12]} has not screened the development split"]
        if sc.headroom is None:
            return [f"registry {self.version[:12]} records no development headroom check"]
        groups = {(e.family, e.size) for e in self.gate_worlds(DEVELOPMENT)}
        got = {(h.family, h.size): h for h in sc.headroom}
        problems = self.coverage_problems(sizes)
        problems += [f"no headroom result for {g}" for g in sorted(groups - set(got))]
        excluded = sc.excluded()  # v3+: a recorded exclusion is not a failure (D3 step 4)
        problems += [f"headroom failed for {h.family}/{h.size}: median fill {h.median_fill!r} "
                     f">= ceiling {h.ceiling!r}" for h in sc.headroom
                     if not h.passed and (h.family, h.size) not in excluded]
        return problems

    def gate_worlds(self, split: str | None = None) -> tuple[WorldEntry, ...]:
        return tuple(e for e in self.entries
                     if e.status == GATE and (split is None or e.split == split))

    def entry(self, world_id: str) -> WorldEntry:
        for e in self.entries:
            if e.id == world_id:
                return e
        raise KeyError(world_id)

    def payload(self) -> dict:
        return {
            "name": self.name,
            "seed_ranges": {s: [lo, hi] for s, lo, hi in self.seed_ranges},
            "entries": [e.to_dict() for e in self.entries],
            "screening": None if self.screening is None else self.screening.to_dict(),
            "saturation_ceiling": self.saturation_ceiling,
        } | ({} if self.shock_profiles is None else {"shock_profiles": self.shock_profiles}) | (
            {} if self.screened_on is None else {"screened_on": self.screened_on})

    @property
    def version(self) -> str:
        return content_hash(self.payload())

    def to_json(self) -> str:
        d = self.payload()
        d["content_hash"] = self.version
        return json.dumps(d, indent=1, sort_keys=True) + "\n"


def registry_from_dict(d: dict) -> Registry:
    """Parse a registry file, refusing a recorded hash that does not match its contents."""
    d = dict(d)
    recorded = d.pop("content_hash", None)
    if recorded is None:
        raise RegistryHashMismatch("registry file records no content_hash; it has no version")
    actual = content_hash(d)
    if actual != recorded:
        raise RegistryHashMismatch(
            f"registry content hash {actual} != recorded {recorded}: edited without a new version")
    reg = Registry(
        name=d["name"],
        seed_ranges=tuple((s, lo, hi) for s, (lo, hi) in d["seed_ranges"].items()),
        entries=tuple(WorldEntry.from_dict(e) for e in d["entries"]),
        screening=None if d["screening"] is None else ScreeningRecord.from_dict(d["screening"]),
        saturation_ceiling=d["saturation_ceiling"],
        shock_profiles=d.get("shock_profiles"),
        screened_on=d.get("screened_on"),
    )
    if reg.version != recorded:
        raise RegistryHashMismatch(
            f"registry {recorded} is not in canonical form (re-derived hash {reg.version})")
    if reg.screening is not None and reg.version_tag >= 2 and reg.screened_on is None:
        raise ValueError(f"screened registry {reg.name!r} ({recorded[:12]}) records no "
                         "screened_on: a frozen v2+ registry must name the draft its screen read")
    reg.check_replay()
    return reg


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> Registry:
    return registry_from_dict(json.loads(Path(path).read_text()))


def save_registry(registry: Registry, path: str | Path) -> None:
    Path(path).write_text(registry.to_json())


# --- the draft generator ---


def episode_seed(world_id: str, j: int) -> int:
    """Episode seed j of a world (module docstring)."""
    return int(hashlib.sha256(f"episode|{world_id}|{j}".encode()).hexdigest()[:8], 16) % 2**31


def _generator_ref(family: str, size: int, regime: str,
                   capacity_scale: float | None = None, shock_profile: str | None = None) -> dict:
    """The family's generator ref. The capacity scale (default: the family's recorded one) is
    written only when it is not neutral, so the ref alone fixes the world and an unscaled
    registry keeps its bytes (task 10.2)."""
    params: dict = {"producers": size, "regime": regime}
    if family == SPARSE_WIDE:
        params["fanout"] = SPARSE_FANOUT
    if regime == CALM:
        params["schedule_length"] = CALM_SCHEDULE_LENGTH
    scale = FAMILIES[family].capacity_scale if capacity_scale is None else capacity_scale
    if check_capacity_scale(scale) != NEUTRAL_CAPACITY_SCALE:
        params["capacity_scale"] = scale
    if shock_profile is not None:
        params["shock_profile"] = shock_profile
    return {"generator": FAMILY_GENERATORS[family], "params": params}


def build_worlds_v1(families: Collection[str] = GATE_FAMILIES) -> Registry:
    """Draft registry v1 with the D7 defaults: per split × family × size, the calm and shocked
    gate worlds plus `RESERVES_PER_GROUP` reserves per regime, seeds per the module docstring.
    Seeds are allocated over every family in `SEED_ORDER` and only `families` are emitted, so a
    sparse-wide world is the same world whether or not dense is in the registry (task 10.4)."""
    unknown = sorted(set(families) - set(SEED_ORDER))
    if unknown:
        raise ValueError(f"unknown families {unknown}; known: {list(SEED_ORDER)}")
    entries: list[WorldEntry] = []
    for split in SPLITS:
        lo, hi = SPLIT_SEED_RANGES[split]
        nxt = lo
        for family in SEED_ORDER:
            for size in SIZES:
                for regime in REGIMES:
                    worlds, episodes = WORLD_COUNTS[split][regime]
                    slots = [(GATE, i, None) for i in range(worlds)]
                    slots += [(RESERVE, k, k) for k in range(RESERVES_PER_GROUP)]
                    for status, i, pos in slots:
                        if family not in families:
                            nxt += 1  # the seed block stays reserved
                            continue
                        tag = f"{i:02d}" if status == GATE else f"r{i}"
                        wid = f"{split[:3]}-{family}-{size}-{regime}-{tag}"
                        entries.append(WorldEntry(
                            id=wid, split=split, family=family, size=size, regime=regime,
                            world_seed=nxt,
                            episode_seeds=tuple(episode_seed(wid, j) for j in range(episodes)),
                            generator=_generator_ref(family, size, regime),
                            status=status, reserve_position=pos,
                        ))
                        nxt += 1
        if nxt - 1 > hi:
            raise ValueError(f"{split} needs seeds up to {nxt - 1}, beyond its range end {hi}")
    return Registry(
        name="worlds-v1",
        seed_ranges=tuple((s, *SPLIT_SEED_RANGES[s]) for s in SPLITS),
        entries=tuple(entries),
    )


def build_worlds_v2(families: Collection[str] = GATE_FAMILIES) -> Registry:
    """Draft registry v2 (add-shocks-i-drift D8): the v1 draft's worlds, ids, world seeds,
    episode seeds and reserve order, with every shocked entry (gate and reserve) naming the
    "shocks-i" profile and the registry recording the profile file's hash. Calm entries are
    copied unchanged. Unscreened: the v2 screen (EXP-0047) freezes it."""
    from .shock_profiles import load_shock_profiles

    v1 = build_worlds_v1(families)
    entries = []
    for e in v1.entries:
        if e.regime == SHOCKED:
            g = copy.deepcopy(e.generator)
            g["params"]["shock_profile"] = V2_SHOCK_PROFILE
            e = replace(e, generator=g)
        entries.append(e)
    return Registry(name="worlds-v2", seed_ranges=v1.seed_ranges, entries=tuple(entries),
                    saturation_ceiling=v1.saturation_ceiling,
                    shock_profiles=load_shock_profiles().version)


# --- registry v3: region draws (add-gate-v3 D2) ---


@functools.lru_cache(maxsize=64)
def _region_draw(stream: str, size: int, pool: int, choice: str, shocks: bool,
                 pattern: str) -> tuple:
    """`world_space.region_sample` at the neutral shock levers (cached: validation re-derives
    every region entry's draw)."""
    from .world_space import region_sample

    return tuple(region_sample(stream, size, pool, choice=choice, shocks=shocks,
                               pattern=pattern))


def _registry_file_seeds(path: Path) -> dict[int, str]:
    """World and episode seeds of a packaged registry file, read without replay; the recorded
    content hash must match."""
    d = json.loads(path.read_text())
    recorded = d.pop("content_hash", None)
    if recorded is None or content_hash(d) != recorded:
        raise RegistryHashMismatch(f"{path}: content hash does not match its contents")
    out: dict[int, str] = {}
    for e in d["entries"]:
        out.setdefault(e["world_seed"], f"{d['name']} world {e['id']}")
        for s in e["episode_seeds"]:
            out.setdefault(s, f"{d['name']} episode of {e['id']}")
    return out


@functools.lru_cache(maxsize=1)
def region_reserved_seeds() -> dict[int, str]:
    """Every seed a region world or episode must avoid, mapped to who holds it (gate-harness
    "Region worlds are fresh draws disjoint from atlas and M4 worlds"; tasks.md 2.2): the world
    and episode seeds of the packaged registries v1 / v2; the atlas-v1 sample (the worlds of atlas
    v1 / v2, every split) at 12 and 42 with its screen and confirm episode seeds; the EXP-0050
    fanout grid (the anchors' source) with its episode seeds; the first `M4_RESERVED_INDICES`
    world seeds of `M4_TRAIN_STREAM` at 12 and 42."""
    from .atlas import ATLAS, CONFIRM_EPISODES, CONFIRM_STAGE, SCREEN_EPISODES, SCREEN_WORLDS
    from .routing_audit import fanout_worlds
    from .world_space import episode_seeds, world_id, world_seed

    out: dict[int, str] = {}
    for path in (REGISTRY_V1, REGISTRY_V2):
        for s, who in _registry_file_seeds(path).items():
            out.setdefault(s, who)
    for size, n in sorted(SCREEN_WORLDS.items()):
        for i in range(n):
            wid = world_id(ATLAS, size, i)
            out.setdefault(world_seed(ATLAS, size, i), f"atlas world {wid}")
            for s in (episode_seeds(wid, SCREEN_EPISODES)
                      + episode_seeds(wid, CONFIRM_EPISODES, CONFIRM_STAGE)):
                out.setdefault(s, f"atlas episode of {wid}")
    for e in fanout_worlds():
        out.setdefault(e.world_seed, f"EXP-0050 world {e.id}")
        for s in e.episode_seeds:
            out.setdefault(s, f"EXP-0050 episode of {e.id}")
    for size in SIZES:
        for i in range(M4_RESERVED_INDICES):
            out.setdefault(world_seed(M4_TRAIN_STREAM, size, i),
                           f"M4 draw {world_id(M4_TRAIN_STREAM, size, i)}")
    return out


def _rho(cfg: Configuration) -> float:
    """`world_space.measure`'s rho, without the demand replay."""
    from .families import demand_units_per_cycle, top_layer_throughput

    return demand_units_per_cycle(cfg) / top_layer_throughput(cfg)


def region_entries(split: str, family: str, pool: int = REGION_POOL,
                   counts: Mapping[str, Mapping[str, tuple[int, int]]] = REGION_WORLD_COUNTS,
                   ) -> tuple[list[WorldEntry], list[dict]]:
    """One region family's entries in one split, and its acceptance rows (D2 "Acceptance filter").

    Slots in a fixed order: regimes in `REGIMES` order, each gate worlds then
    `RESERVES_PER_GROUP` reserves. `ws_loaded` alternates dense / sparse_f4 by slot index
    within its (split, regime) group (even = dense); `ws_wide` is sparse_f14. Draws come from
    `region_sample(gate_v3_stream(split), size, pool)` in index order, each index tried once by
    the slot that reaches it; a slot takes the first draw whose world (built with the slot's
    choice and regime) has rho in `RHO_BAND`. Rejections are kept with their rho. A pool too small
    for every slot is refused, never shrunk."""
    from .shock_profiles import SHOCKS_I
    from .world_space import CYCLIC, DRIFT, gate_v3_stream, world_record

    if family not in REGION_FAMILIES:
        raise ValueError(f"unknown region family {family!r}; known: {REGION_FAMILIES}")
    size = FAMILY_SIZES[family][0]
    stream = gate_v3_stream(split)
    lo, hi = RHO_BAND
    entries: list[WorldEntry] = []
    rows: list[dict] = []
    k = 0
    for regime in REGIMES:
        shocks, pattern = (False, CYCLIC) if regime == CALM else (True, DRIFT)
        worlds, episodes = counts[split][regime]
        slots = [(GATE, i, None) for i in range(worlds)]
        slots += [(RESERVE, j, j) for j in range(RESERVES_PER_GROUP)]
        for s, (status, i, pos) in enumerate(slots):
            choices = REGION_CHOICES[family]
            choice = choices[s % len(choices)]
            tag = f"{i:02d}" if status == GATE else f"r{i}"
            wid = f"{split[:3]}-{family}-{size}-{regime}-{tag}"
            while True:
                if k >= pool:
                    raise ValueError(f"{stream} size {size}: draw pool of {pool} exhausted at "
                                     f"{wid}; the draft refuses to shrink a group")
                point = _region_draw(stream, size, pool, choice, shocks, pattern)[k]
                rho = _rho(point.config())
                ok = lo < rho <= hi
                rows.append({"split": split, "family": family, "size": size, "regime": regime,
                             "slot": tag, "choice": choice, "draw_index": k, "seed": point.seed,
                             "rho": rho, "accepted": ok, "world": wid if ok else ""})
                k += 1
                if ok:
                    break
            record = world_record(point)
            if record["coordinates"]["rho"] != rho:
                raise AssertionError(f"{wid}: record rho {record['coordinates']['rho']} != {rho}")
            params: dict = {"producers": size, "levers": point.levers.to_dict(),
                            "shock_window": list(point.shock_window)}
            if regime == SHOCKED:
                params["shock_profile"] = SHOCKS_I
            entries.append(WorldEntry(
                id=wid, split=split, family=family, size=size, regime=regime,
                world_seed=point.seed,
                episode_seeds=tuple(episode_seed(wid, j) for j in range(episodes)),
                generator={"generator": WORLD_SPACE_GENERATOR, "params": params},
                status=status, reserve_position=pos,
                region={"pool": pool, "record": record}))
    return entries, rows


def draft_worlds_v3(pool: int = REGION_POOL) -> tuple[Registry, list[dict]]:
    """Draft registry v3 and its acceptance rows (add-gate-v3 D2): every v2 draft entry
    unchanged (the frozen v2 set has no replacements, so ids, seeds, episode seeds, generator refs,
    statuses and reserve positions equal the frozen v2's), plus `ws_loaded` (12) and `ws_wide`
    (42) per split with the D2 counts and 4 reserves per regime. Unscreened."""
    v2 = build_worlds_v2()
    entries = list(v2.entries)
    rows: list[dict] = []
    for split in SPLITS:
        for family in REGION_FAMILIES:
            got, r = region_entries(split, family, pool)
            entries += got
            rows += r
    reg = Registry(name="worlds-v3", seed_ranges=v2.seed_ranges, entries=tuple(entries),
                   saturation_ceiling=v2.saturation_ceiling, shock_profiles=v2.shock_profiles)
    return reg, rows


def build_worlds_v3() -> Registry:
    """Draft registry v3 (`draft_worlds_v3` without the acceptance rows)."""
    return draft_worlds_v3()[0]


ACCEPTANCE_COLUMNS = ("split", "family", "size", "regime", "slot", "choice", "draw_index", "seed",
                      "rho", "accepted", "world")


def write_acceptance(rows: Sequence[Mapping], path: str | Path) -> None:
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ACCEPTANCE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def smoke_variant(e: WorldEntry) -> WorldEntry:
    """The horizon-12 dry-run form of a shocked region entry (add-gate-v3 2.2, the 2.1 review's
    carry-forward): `shock_profile` "shocks-i-smoke" with every start window at
    `SMOKE_START_WINDOW`, and its own world record. `gate_cli.test_only_problem` refuses it in a
    real run (`names_test_only_profile`)."""
    from .shock_profiles import SHOCKS_I_SMOKE, SMOKE_START_WINDOW
    from .world_space import point_from_record, world_record

    if e.family not in REGION_FAMILIES or e.regime != SHOCKED or e.region is None:
        raise ValueError(f"{e.id}: only a shocked region entry has a region smoke form")
    point = replace(point_from_record(e.region["record"]), shock_window=tuple(SMOKE_START_WINDOW))
    g = copy.deepcopy(e.generator)
    g["params"]["shock_profile"] = SHOCKS_I_SMOKE
    g["params"]["shock_window"] = list(SMOKE_START_WINDOW)
    return replace(e, generator=g, region={"pool": e.region["pool"], "record": world_record(point)})


# --- zero-baseline screening (gate-harness "Zero-baseline screening") ---


def world_baseline_means(
    registry: Registry, roster: Roster, rows: Iterable[EpisodeRow],
) -> dict[str, float]:
    """Hindsight-baseline mean per world from roster-only rows (gate arithmetic: per-episode max
    over members of their replicate means, averaged over the world's registered episodes).
    Refuses candidate rows, non-members, unknown worlds, rows whose size/regime/episode disagree
    with the registry, and incomplete member/replicate coverage: the replacement must not depend
    on any candidate result, and a max over a subset would understate the bar."""
    members = set(roster.member_ids)
    by_world: dict[str, dict[int, dict[str, dict[int, float]]]] = {}
    for r in rows:
        if r.subject == CANDIDATE:
            raise ValueError(f"candidate row on {r.world}/{r.episode}: screening is roster-only")
        if r.subject not in members:
            raise ValueError(f"row subject {r.subject!r} is not a member of roster {roster.version}")
        e = registry.entry(r.world)
        if (r.size, r.regime) != (e.size, e.regime):
            raise ValueError(f"{r.world}: row (size, regime) {(r.size, r.regime)} != registry "
                             f"{(e.size, e.regime)}")
        if r.episode not in e.episode_seeds:
            raise ValueError(f"{r.world}: episode {r.episode} is not a registered episode seed")
        reps = by_world.setdefault(r.world, {}).setdefault(r.episode, {}).setdefault(r.subject, {})
        if r.replicate in reps:
            raise ValueError(f"{r.world}/{r.episode}: duplicate replicate {r.replicate} for {r.subject}")
        reps[r.replicate] = r.units
    means: dict[str, float] = {}
    for world, eps in by_world.items():
        e = registry.entry(world)
        if set(eps) != set(e.episode_seeds):
            raise ValueError(f"{world}: rows cover episodes {sorted(eps)}, registry has "
                             f"{sorted(e.episode_seeds)}")
        per_episode = []
        for ep in e.episode_seeds:
            values: dict[str, float] = {}
            for mid in roster.member_ids:
                got = eps[ep].get(mid, {})
                want = set(range(roster.replicates_for(roster.member(mid))))
                if set(got) != want:
                    raise ValueError(f"{world}/{ep}: {mid} has replicates {sorted(got)}, roster "
                                     f"wants {sorted(want)}")
                values[mid] = replicate_mean([got[i] for i in sorted(got)])
            per_episode.append(episode_baseline(values))
        means[world] = world_baseline(per_episode)
    return means


@dataclass(frozen=True)
class ReplacementPlan:
    """What zero-baseline screening decides from the baseline means known so far
    (`plan_replacements`). `consulted` are the reserves examined, in order; `pending` are
    reserves the fixed order will certainly consult but whose rows are not in yet. The plan is
    final when `pending` is empty."""

    replacements: tuple[Replacement, ...]
    consulted: tuple[str, ...]
    pending: tuple[str, ...]


def plan_replacements(
    registry: Registry, means: Mapping[str, float], splits: Collection[str] = (DEVELOPMENT,),
) -> ReplacementPlan:
    """The single replacement rule behind both the eager and the on-demand screen (task 8.6).

    Gate worlds of `splits` are visited in id order. Each one with a zero hindsight baseline
    takes reserves from its (split, family, size, regime) group in reserve-position order,
    skipping zero reserves, until a reserve has a non-zero mean. A reserve with no mean yet goes
    to `pending` and leaves its slot unresolved. Later slots of the same group keep popping the
    queue. That is exact: every unresolved slot will consume at least one more reserve, so the
    next reserve in order is consulted whatever the pending one turns out to be. On-demand
    screening therefore runs exactly the reserves the eager screen consults. Raises when a group
    runs out of reserves even if every pending reserve proves usable."""
    targets = [e for e in registry.gate_worlds() if e.split in splits]
    missing = [e.id for e in targets if e.id not in means]
    if missing:
        raise ValueError(f"no baseline rows for gate worlds {missing}")
    queues: dict[tuple[str, str, int, str], list[WorldEntry]] = {}
    for e in registry.entries:
        if e.status == RESERVE:
            queues.setdefault(e.group, []).append(e)
    for q in queues.values():
        q.sort(key=lambda e: e.reserve_position)  # type: ignore[arg-type, return-value]
    replacements: list[Replacement] = []
    consulted: list[str] = []
    pending: list[str] = []
    for e in targets:
        if means[e.id] != 0:
            continue
        skipped: list[str] = []
        queue = queues.get(e.group, [])
        while True:
            if not queue:
                raise ValueError(f"{e.id}: zero hindsight baseline and no usable reserve left in "
                                 f"{e.group}")
            r = queue.pop(0)
            if r.id not in means:
                pending.append(r.id)
                break
            consulted.append(r.id)
            if means[r.id] != 0:
                replacements.append(Replacement(e.id, r.id, tuple(skipped)))
                break
            skipped.append(r.id)
    return ReplacementPlan(tuple(replacements), tuple(consulted), tuple(pending))


def screen_zero_baselines(
    registry: Registry, roster: Roster, rows: Sequence[EpisodeRow],
    splits: Collection[str] = (DEVELOPMENT,),
) -> Registry:
    """Replace every zero-baseline gate world by the next unused reserve of its (split, family,
    size, regime) group, in reserve-position order, skipping reserves whose own baseline is zero
    (`plan_replacements`). Pure over roster-only rows, so the outcome is the same whichever
    candidate is evaluated later. Returns a new registry (new version) carrying the
    `ScreeningRecord`.

    Screens only the gate worlds of `splits` (development by default, module docstring) and
    refuses rows of worlds in other splits. Needs a baseline mean for every such gate world and for
    every reserve it consults; refuses an already-screened registry and raises when a group runs
    out of reserves. The record keeps the means of the gate worlds and the consulted reserves
    only, so rows of reserves never consulted (an eager screen runs them all) do not change the
    outcome: eager and on-demand screens give the same registry."""
    if registry.screening is not None:
        raise ValueError(f"registry {registry.version} is already screened")
    splits = tuple(sorted(set(splits)))
    if not splits or not set(splits) <= set(SPLITS):
        raise ValueError(f"splits {splits} must be a non-empty subset of {SPLITS}")
    means = world_baseline_means(registry, roster, rows)
    outside = sorted(w for w in means if registry.entry(w).split not in splits)
    if outside:
        raise ValueError(f"rows for worlds outside the screened splits {splits}: {outside[:3]}")
    plan = plan_replacements(registry, means, splits)
    if plan.pending:
        raise ValueError(f"reserve {plan.pending[0]} was consulted but has no baseline rows")
    updated = {e.id: e for e in registry.entries}
    for rep in plan.replacements:
        updated[rep.replaced] = replace(updated[rep.replaced], status=REPLACED)
        for sk in rep.skipped:
            updated[sk] = replace(updated[sk], status=REPLACED)
        updated[rep.by] = replace(updated[rep.by], status=GATE)
    kept = {e.id for e in registry.gate_worlds() if e.split in splits} | set(plan.consulted)
    record = ScreeningRecord(
        roster_version=roster.version,
        baseline_means=tuple(sorted((w, m) for w, m in means.items() if w in kept)),
        replacements=plan.replacements,
        splits=splits,
    )
    return Registry(name=registry.name, seed_ranges=registry.seed_ranges,
                    entries=tuple(updated.values()), screening=record,
                    saturation_ceiling=registry.saturation_ceiling,
                    shock_profiles=registry.shock_profiles)


def _check_headroom(h: Headroom, ceiling: float) -> None:
    """A headroom result must use the registry's ceiling and its verdict must follow from its own
    numbers (pass iff median < ceiling), so a hand-edited pass flag is refused."""
    if h.ceiling != ceiling:
        raise ValueError(f"headroom {h.family}/{h.size} used ceiling {h.ceiling}, registry "
                         f"pre-registers {ceiling}")
    if h.passed != (h.median_fill < h.ceiling) or h.worlds < 1:
        raise ValueError(f"headroom {h.family}/{h.size} is inconsistent: passed={h.passed}, "
                         f"median {h.median_fill!r}, ceiling {h.ceiling!r}, worlds {h.worlds}")


def record_headroom(registry: Registry, results: Sequence[Headroom]) -> Registry:
    """Record the D10 headroom check in a screened registry (a new version). Every result must use
    the registry's pre-registered ceiling; the results are recorded whether or not they pass, and
    the caller decides what a failure stops."""
    if registry.screening is None:
        raise ValueError("headroom is recorded on a screened registry")
    if registry.screening.headroom is not None:
        raise ValueError(f"registry {registry.version} already records a headroom check")
    if not results:
        raise ValueError("no headroom results")
    for h in results:
        _check_headroom(h, registry.saturation_ceiling)
    return replace(registry, screening=replace(registry.screening, headroom=tuple(results)))


def record_exclusions(registry: Registry) -> Registry:
    """add-gate-v3 D3 step 4: exclude every family × size whose recorded headroom failed, in both
    splits (gate worlds and reserves get status `excluded`), and record each exclusion with its
    headroom result (a new version). v3+ only; needs the headroom record."""
    sc = registry.screening
    if registry.version_tag < 3:
        raise ValueError(f"registry {registry.name!r}: exclusions exist from v3 on")
    if sc is None or sc.headroom is None:
        raise ValueError("exclusions follow a recorded headroom check")
    if sc.exclusions is not None:
        raise ValueError(f"registry {registry.version} already records exclusions")
    failed = [h for h in sc.headroom if not h.passed]
    pairs = {(h.family, h.size) for h in failed}
    entries = tuple(replace(e, status=EXCLUDED) if (e.family, e.size) in pairs else e
                    for e in registry.entries)
    exclusions = tuple(Exclusion(h.family, h.size, h) for h in sorted(
        failed, key=lambda h: (h.family, h.size)))
    return replace(registry, entries=entries, screening=replace(sc, exclusions=exclusions))


# --- sealed access (gate-harness "Sealed worlds need a pre-registration") ---


@dataclass(frozen=True)
class Preregistration:
    """A frozen pre-registration manifest: candidate checkpoint SHA-256 per role plus the roster,
    registry and side-effect parameter versions the sealed run must use."""

    checkpoints: tuple[tuple[str, str], ...]
    roster_version: str
    registry_version: str
    params_version: str
    sha256: str = field(default="", compare=False)  # of the manifest file, for the access log

    def __post_init__(self) -> None:
        """A manifest that pins no candidate checkpoint binds nothing, so it is refused however it
        was built (file or direct construction); every pin must be a 64-hex SHA-256."""
        if not self.checkpoints:
            raise ValueError("pre-registration names no checkpoints")
        for role, sha in self.checkpoints:
            if not role or not isinstance(sha, str) or len(sha) != 64 or any(
                    c not in "0123456789abcdef" for c in sha):
                raise ValueError(f"pre-registration pin for {role!r} is not a sha256: {sha!r}")
        for name in ("roster_version", "registry_version", "params_version"):
            if not getattr(self, name):
                raise ValueError(f"pre-registration has no {name}")

    def to_dict(self) -> dict:
        return {
            "checkpoints": dict(self.checkpoints),
            "roster_version": self.roster_version,
            "registry_version": self.registry_version,
            "params_version": self.params_version,
        }


def load_preregistration(path: str | Path) -> Preregistration:
    d = json.loads(Path(path).read_text())
    return Preregistration(
        checkpoints=tuple(sorted((d.get("checkpoints") or {}).items())),
        roster_version=d["roster_version"], registry_version=d["registry_version"],
        params_version=d["params_version"], sha256=sha256_file(str(path)),
    )


def check_sealed_access(
    registry: Registry,
    world_ids: Collection[str],
    *,
    preregistration: Preregistration | None,
    checkpoints: Mapping[str, str | Path],
    roster_version: str,
    params_version: str,
    log_path: str | Path = SEALED_ACCESS_LOG,
    now: datetime | None = None,
) -> dict | None:
    """Gate a run before any episode. Development-only requests pass untouched (None). A request
    touching a sealed world needs a manifest whose roster / registry / parameter versions equal
    the run's and whose checkpoint hashes equal the SHA-256 of the files the run will load (every
    role, no extra roles); anything else raises `SealedAccessError` and logs nothing. On success the
    access is appended to `log_path` and the record returned."""
    for wid in world_ids:
        registry.entry(wid)  # unknown ids fail loudly
    sealed = sorted(w for w in world_ids if registry.entry(w).split == SEALED)
    if not sealed:
        return None
    if preregistration is None:
        raise SealedAccessError(f"sealed worlds {sealed} requested without a pre-registration manifest")
    p = preregistration
    if not p.checkpoints or not checkpoints:  # also guards objects built around __post_init__
        raise SealedAccessError("a sealed run must pin at least one candidate checkpoint")
    for what, got, want in (("registry", registry.version, p.registry_version),
                            ("roster", roster_version, p.roster_version),
                            ("parameter", params_version, p.params_version)):
        if got != want:
            raise SealedAccessError(f"{what} version {got} != pre-registered {want}")
    want_ckpt = dict(p.checkpoints)
    if set(checkpoints) != set(want_ckpt):
        raise SealedAccessError(f"checkpoint roles {sorted(checkpoints)} != pre-registered "
                                f"{sorted(want_ckpt)}")
    manifest = {role: {"path": str(path), "sha256": sha256_file(str(path))}
                for role, path in checkpoints.items()}
    try:
        assert_inputs_match(manifest, want_ckpt)
    except ValueError as exc:
        raise SealedAccessError(str(exc)) from exc
    record = {
        "time": (now or datetime.now(timezone.utc)).isoformat(),
        "worlds": sealed,
        "registry_version": registry.version,
        "roster_version": roster_version,
        "params_version": params_version,
        "checkpoints": {role: m["sha256"] for role, m in sorted(manifest.items())},
        "preregistration_sha256": p.sha256,
    }
    append_sealed_access(record, log_path)
    return record


def append_sealed_access(record: dict, log_path: str | Path = SEALED_ACCESS_LOG) -> None:
    """Append one JSON line to the sealed access log (append-only; never rewritten)."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="world registry tools (design D7)")
    ap.add_argument("--write-draft", action="store_true",
                    help="regenerate a draft registry (v1: build_worlds_v1, v2: build_worlds_v2)")
    ap.add_argument("--draft-version", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--out", default=None, help="default: worlds_v1.json / worlds_v2.json; "
                    "--write-draft needs it (v3: also writes acceptance.csv beside it)")
    args = ap.parse_args()
    if args.write_draft and args.out is None:
        ap.error("the packaged worlds_v1.json / worlds_v2.json are frozen screens; "
                 "pass --out for a draft")
    out = args.out or str(REGISTRY_V2 if args.draft_version == 2 else REGISTRY_V1)
    if not args.write_draft:
        reg = load_registry(out)
        print(f"{out}: {len(reg.entries)} entries, version {reg.version}")
        return
    args.out = out
    if args.draft_version == 3:
        if Path(out).resolve().parent == REGISTRY_V2.resolve().parent:
            ap.error("the v3 draft goes to a Drafts path, never the packaged registry directory")
        reg, rows = draft_worlds_v3()
        save_registry(reg, out)
        write_acceptance(rows, Path(out).parent / "acceptance.csv")
        load_registry(out)  # the written draft reloads (hash check and replay)
    else:
        reg = build_worlds_v1() if args.draft_version == 1 else build_worlds_v2()
        save_registry(reg, args.out)
    print(f"wrote {args.out}: {len(reg.entries)} entries, version {reg.version}")


if __name__ == "__main__":
    main()
