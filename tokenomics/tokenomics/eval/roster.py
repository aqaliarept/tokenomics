"""Gate baseline roster: the frozen set of baseline stacks the hindsight baseline is the per-episode
max over (design D5, add-measurement-gate-harness; gate-harness "Frozen baseline roster",
"Roster pruning uses development evidence only").

A roster file names its members (producing × ordering baseline, each with its env contract and
random flag), R (replicates per random member) and an optional pruning record. Its version is a
sha256 content hash over every field but the hash itself, recorded in the file and re-checked on
load: an edited member list without a new hash is refused, so a gate report that names the version
names exactly the stacks its hindsight baseline was the max over. The gate only gets rosters
through `load_roster`.

Pruning is pure over per-episode development rows, so the rule is auditable from the rows alone,
and it refuses anything that could let the candidate shape the bar: non-development rows, rows of
non-members (candidate rows), a frozen roster.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Collection, Iterable

from ..domain.config import Configuration
from ..env import TokenomicsAECEnv
from .baselines import (
    EXTRA_PRODUCING,
    ORDERING_BASELINES,
    PRODUCING_BASELINES,
    STOCHASTIC_BASELINES,
    Policy,
    producing_policy,
)
from .joint_stack import joint_stack_env

ROSTER_V1 = Path(__file__).parent / "registry" / "roster_v1.json"
ROSTER_V2 = Path(__file__).parent / "registry" / "roster_v2.json"
DEFAULT_ROSTER = ROSTER_V2  # the frozen v2 roster since task 10.1 (add-shocks-i-drift D8)

# Ordering baselines each draft roster crosses with every producing baseline. Explicit, so a
# baseline added to `ORDERING_BASELINES` later never changes an existing version's draft.
# v1 (D5, add-measurement-gate-harness): the hand-written routers plus `blend_0.5`. v2
# (add-shocks-i-drift D8/D9): v1 plus the shock-reactive router.
ROSTER_V1_ORDERING = ("even_split", "proportional", "obs_price", "blend_0.5", "random")
SHOCK_REACTIVE = "shock_reactive"
ROSTER_V2_ORDERING = ROSTER_V1_ORDERING + (SHOCK_REACTIVE,)
# v3 (add-gate-v3 D1): v2 plus the five routers promoted from `routing_diagnostics`. Pruning on
# the v3 development screen, not a hand rule, decides which of them stay.
PROMOTED_V3 = ("round_robin", "even_split_rotating", "random_whole", "makeable_weighted_whole",
               "makeable_split")
ROSTER_V3_ORDERING = ROSTER_V2_ORDERING + PROMOTED_V3
# Producers the v3 draft adds beyond `PRODUCING_BASELINES` (add-gate-v3 task 1.3, user decision
# "draft now, re-screen later"; the roster holds the strongest known policy): M4's buy-ahead
# producer at k* = 4 (EXP-0076 / EXP-0077). They live in `baselines.EXTRA_PRODUCING`, so the v1 /
# v2 drafts (which cross `PRODUCING_BASELINES` only) and every frozen file stay byte-identical.
V3_EXTRA_PRODUCING = ("entity_planner_buyahead_k4",)
# The pre-screen pilot's roster (add-gate-v3 D3 step 0): `entity_planner` alone, the only
# producer that has ever set the per-episode best. A dry-run screen input, never a gate roster.
PILOT_PRODUCING = "entity_planner"
PILOT_V3_NAME = "roster-v3-pilot-ep (pilot, not a gate roster)"

# Env contracts (D5). JOINT_STACK = `joint_stack_env`, the ordering mechanics the candidate runs
# under. LEGACY = the default flat-observation env the `obs_*` baselines decode.
JOINT_STACK = "joint_stack"
LEGACY = "legacy"
CONTRACTS = (JOINT_STACK, LEGACY)

RANDOM_POLICY = "random"


def is_stochastic(producing: str, ordering: str) -> bool:
    """A member runs R replicates when either policy draws from the action-space RNG
    (`STOCHASTIC_BASELINES`, add-gate-v3 D1). For every v1 / v2 member this is `random` in the
    pair, so their stored `random` flags (and hashes) are unchanged."""
    return producing in STOCHASTIC_BASELINES or ordering in STOCHASTIC_BASELINES

DEFAULT_REPLICATES = 5  # R, tasks.md 10.1 sizing
DEVELOPMENT = "development"
# Economy sizes the gate measures (producers; design "Economy sizes": 12 and 42, user decision
# 2026-10-27). About 120 producers is a future, non-gating target.
GATE_SIZES = (12, 42)

NEVER_ARGMAX_RULE = "never-argmax-v1"
# add-gate-v3 D3 step 5: the two-wave screen's rule.
NEVER_ARGMAX_TWO_WAVE_RULE = "never-argmax-two-wave-v1"
NEVER_ARGMAX_TWO_WAVE_TEXT = (
    "Wave 1 runs every member on the wave-1 development worlds; wave 2 runs W2 = the members "
    "that are the per-episode argmax on at least one admitted wave-1 episode, plus every member "
    "whose producer is entity_planner, on the wave-2 worlds. A member is kept iff it is in W2 "
    "and is the per-episode argmax on at least one admitted episode of either wave, the argmax "
    "taken over the members run on that episode (every member tied at the max counts). Members "
    "outside W2 are dropped as not carried to wave 2. Excluded family x size episodes are not "
    "evidence."
)
WAVE2_PRODUCING = PILOT_PRODUCING  # every entity_planner member runs in wave 2 (D3 step 2)
NEVER_ARGMAX_TEXT = (
    "Drop a member that is never the per-episode argmax on any development episode at any "
    "measured economy size. Every member attaining an episode's max value counts as its argmax "
    "(ties never prune a co-winner). The evidence must cover every member on every episode."
)


def name_version_tag(name: str) -> int:
    """The generation a registry, roster or parameter file name carries (`worlds-v2`,
    `roster-v2`, `side-effect-params-v2`, also a derived name such as `worlds-v2-dry-run` -> 2);
    1 when the name carries none (add-shocks-i-drift D10 "Versioned screen outputs")."""
    m = re.search(r"-v(\d+)(?:-|$)", name)
    return int(m.group(1)) if m else 1


def check_screened_on(value: object, what: str) -> None:
    """A recorded `screened_on` must be a registry content hash (sha256, 64 lowercase hex): an
    empty or malformed key could otherwise compare equal across files and pass for one screen's
    provenance (add-shocks-i-drift D10)."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{what}: screened_on must be a sha256 hex digest, got {value!r}")


class RosterHashMismatch(ValueError):
    """The roster's recorded content hash does not match its contents (edited without a new
    version); the gate must not compute a hindsight baseline from it."""


def member_id(producing: str, ordering: str) -> str:
    """Stable member key used in episode rows."""
    return f"{producing}/{ordering}"


def contract_for(producing: str, ordering: str) -> str:
    """Env contract a stack runs under (D5).

    `obs_*` baselines decode the legacy flat observation, so a stack with either policy `obs_*`
    runs LEGACY; every other stack runs JOINT_STACK. `random` samples `env.action_space`, which
    exists under both contracts, so it never forces one: `random/random` runs JOINT_STACK (the
    candidate's mechanics, where the bar matters) and `random` paired with an `obs_*` baseline
    follows that baseline to LEGACY."""
    if producing.startswith("obs_") or ordering.startswith("obs_"):
        return LEGACY
    return JOINT_STACK


@dataclass(frozen=True)
class RosterMember:
    """One baseline stack. `random` = either policy samples actions (`is_stochastic`), so the
    member runs R replicates and its per-episode value is the replicate mean (D5; add-gate-v3
    D1 widens it beyond `random`, the stored field keeps its name)."""

    producing: str
    ordering: str
    contract: str
    random: bool

    @property
    def id(self) -> str:
        return member_id(self.producing, self.ordering)

    def to_dict(self) -> dict:
        return {
            "producing": self.producing,
            "ordering": self.ordering,
            "contract": self.contract,
            "random": self.random,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RosterMember:
        return cls(
            producing=d["producing"], ordering=d["ordering"],
            contract=d["contract"], random=d["random"],
        )


@dataclass(frozen=True)
class EpisodeKey:
    """One development episode the pruning rule was applied to."""

    world: str
    size: int
    episode: int


@dataclass(frozen=True)
class WaveRecord:
    """One member wave of a two-wave screen (add-gate-v3 D3 step 5): the members it ran, the
    development episodes it contributed as evidence, and a digest of those rows."""

    members: tuple[str, ...]
    episodes: tuple[EpisodeKey, ...]
    rows_digest: str

    def to_dict(self) -> dict:
        return {"members": list(self.members),
                "episodes": [{"world": e.world, "size": e.size, "episode": e.episode}
                             for e in self.episodes],
                "rows_digest": self.rows_digest}

    @classmethod
    def from_dict(cls, d: dict) -> WaveRecord:
        return cls(members=tuple(d["members"]),
                   episodes=tuple(EpisodeKey(e["world"], e["size"], e["episode"])
                                  for e in d["episodes"]),
                   rows_digest=d["rows_digest"])


@dataclass(frozen=True)
class PruningRecord:
    """What pruning did and on which evidence (gate-harness "Never-winning stack dropped"): the
    rule, the roster it was applied to, the development episodes, a digest of the exact rows, and
    the dropped members."""

    rule: str
    rule_text: str
    parent_version: str
    episodes: tuple[EpisodeKey, ...]
    rows_digest: str
    dropped: tuple[str, ...]
    # Two-wave rule only (add-gate-v3 D3 step 5): each wave's record and the members not carried
    # to wave 2. None, and left out of the payload, otherwise (v1 / v2 hashes unchanged).
    waves: tuple[WaveRecord, ...] | None = None
    not_carried: tuple[str, ...] | None = None

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "rule_text": self.rule_text,
            "parent_version": self.parent_version,
            "episodes": [
                {"world": e.world, "size": e.size, "episode": e.episode} for e in self.episodes
            ],
            "rows_digest": self.rows_digest,
            "dropped": list(self.dropped),
        } | ({} if self.waves is None else {
            "waves": [w.to_dict() for w in self.waves],
            "not_carried": list(self.not_carried or ())})

    @classmethod
    def from_dict(cls, d: dict) -> PruningRecord:
        return cls(
            rule=d["rule"], rule_text=d["rule_text"], parent_version=d["parent_version"],
            episodes=tuple(EpisodeKey(e["world"], e["size"], e["episode"]) for e in d["episodes"]),
            rows_digest=d["rows_digest"], dropped=tuple(d["dropped"]),
            waves=(None if d.get("waves") is None
                   else tuple(WaveRecord.from_dict(w) for w in d["waves"])),
            not_carried=None if d.get("not_carried") is None else tuple(d["not_carried"]),
        )


@dataclass(frozen=True)
class Roster:
    """A versioned roster. `version` is the content hash (`content_hash`); members are kept sorted
    by id so the hash identifies the member set, not a file's line order."""

    name: str
    replicates: int
    members: tuple[RosterMember, ...]
    frozen: bool = False
    pruning: PruningRecord | None = None
    # Hash of the draft registry the v2+ screen that froze this roster read (add-shocks-i-drift
    # D8/D10): the key that pairs a frozen registry, roster and parameter set. None on v1 and on
    # unfrozen drafts; omitted from the payload then, so their hashes are unchanged.
    screened_on: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(sorted(self.members, key=lambda m: m.id)))
        self.validate()

    @property
    def version_tag(self) -> int:
        return name_version_tag(self.name)

    def validate(self) -> None:
        """Structural checks: R >= 1, unique members, known policies, and contract/random flags
        that match the policies (a hand-edited flag would run a member under the wrong env or
        with the wrong replicate count)."""
        if not isinstance(self.replicates, int) or isinstance(self.replicates, bool) or self.replicates < 1:
            raise ValueError(f"roster R must be >= 1, got {self.replicates}")
        if not self.members:
            raise ValueError("roster has no members")
        if self.screened_on is not None:
            if not (self.frozen and self.version_tag >= 2):
                raise ValueError(f"roster {self.name!r}: screened_on is recorded only on a frozen "
                                 f"v2+ roster (frozen={self.frozen}, v{self.version_tag})")
            check_screened_on(self.screened_on, f"roster {self.name!r}")
        ids = [m.id for m in self.members]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate roster members: {sorted({i for i in ids if ids.count(i) > 1})}")
        for m in self.members:
            if m.producing not in PRODUCING_BASELINES and m.producing not in EXTRA_PRODUCING:
                raise ValueError(f"{m.id}: unknown producing baseline {m.producing!r}")
            if m.ordering not in ORDERING_BASELINES:
                raise ValueError(f"{m.id}: unknown ordering baseline {m.ordering!r}")
            if m.contract != contract_for(m.producing, m.ordering):
                raise ValueError(
                    f"{m.id}: contract {m.contract!r} != {contract_for(m.producing, m.ordering)!r}"
                )
            if m.random != is_stochastic(m.producing, m.ordering):
                raise ValueError(f"{m.id}: random flag {m.random} does not match its policies")

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(m.id for m in self.members)

    def member(self, mid: str) -> RosterMember:
        for m in self.members:
            if m.id == mid:
                return m
        raise KeyError(mid)

    def replicates_for(self, m: RosterMember) -> int:
        """R for a random member, 1 for a deterministic one (D5)."""
        return self.replicates if m.random else 1

    def payload(self) -> dict:
        """Every field but the hash: what `content_hash` covers and what the file stores."""
        return {
            "name": self.name,
            "frozen": self.frozen,
            "replicates": self.replicates,
            "members": [m.to_dict() for m in self.members],
            "pruning": None if self.pruning is None else self.pruning.to_dict(),
        } | ({} if self.screened_on is None else {"screened_on": self.screened_on})

    @property
    def version(self) -> str:
        return content_hash(self.payload())

    def to_json(self) -> str:
        d = self.payload()
        d["content_hash"] = self.version
        return json.dumps(d, indent=2, sort_keys=True) + "\n"


def _canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(payload: dict) -> str:
    """sha256 hex over the canonical JSON of a roster payload (no `content_hash` field)."""
    if "content_hash" in payload:
        raise ValueError("content_hash must not cover itself")
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def roster_from_dict(d: dict) -> Roster:
    """Parse a roster file's JSON object, refusing a recorded hash that does not match its
    contents (gate-harness "Roster edited without a new version")."""
    d = dict(d)
    recorded = d.pop("content_hash", None)
    if recorded is None:
        raise RosterHashMismatch("roster file records no content_hash; it has no version")
    actual = content_hash(d)
    if actual != recorded:
        raise RosterHashMismatch(
            f"roster content hash {actual} != recorded {recorded}: edited without a new version"
        )
    roster = Roster(
        name=d["name"],
        replicates=d["replicates"],
        members=tuple(RosterMember.from_dict(m) for m in d["members"]),
        frozen=d["frozen"],
        pruning=None if d["pruning"] is None else PruningRecord.from_dict(d["pruning"]),
        screened_on=d.get("screened_on"),
    )
    # Re-derive from the parsed object: catches a file whose member list is not in canonical
    # (sorted) order, since the recorded hash was taken over the file's own order.
    if roster.version != recorded:
        raise RosterHashMismatch(
            f"roster {recorded} is not in canonical form (re-derived hash {roster.version})"
        )
    if roster.frozen and roster.version_tag >= 2 and roster.screened_on is None:
        raise ValueError(f"frozen roster {roster.name!r} ({recorded[:12]}) records no screened_on: "
                         "a frozen v2+ roster must name the registry draft its screen read")
    return roster


def load_roster(path: str | Path = DEFAULT_ROSTER) -> Roster:
    return roster_from_dict(json.loads(Path(path).read_text()))


def save_roster(roster: Roster, path: str | Path) -> None:
    Path(path).write_text(roster.to_json())


def _cross(ordering: Iterable[str],
           producing: Iterable[str] = tuple(PRODUCING_BASELINES)) -> tuple[RosterMember, ...]:
    """Every producing baseline (or each of `producing`) paired with each of `ordering`."""
    ordering = tuple(ordering)
    return tuple(
        RosterMember(p, o, contract_for(p, o), is_stochastic(p, o))
        for p in producing
        for o in ordering
    )


def build_roster_v1(replicates: int = DEFAULT_REPLICATES) -> Roster:
    """Draft roster v1 (D5): every producing baseline × every v1 ordering baseline, `blend_0.5`
    included."""
    return Roster(name="roster-v1", replicates=replicates, members=_cross(ROSTER_V1_ORDERING))


def build_roster_v2(replicates: int = DEFAULT_REPLICATES) -> Roster:
    """Draft roster v2 (add-shocks-i-drift D8): roster v1's draft plus every producing baseline
    paired with `shock_reactive` (D9), unfrozen and unpruned. `contract_for` sends
    `obs_stock_aware/shock_reactive` to LEGACY; the router reads only state both contracts have.
    The baseline-only screen on registry v2 prunes and freezes it."""
    return Roster(name="roster-v2", replicates=replicates, members=_cross(ROSTER_V2_ORDERING))


def build_roster_v3(replicates: int = DEFAULT_REPLICATES) -> Roster:
    """Draft roster v3 (add-gate-v3 D1): every producing baseline (8) plus `V3_EXTRA_PRODUCING`
    (task 1.3: `entity_planner_buyahead_k4`) × the v2 ordering baselines and the five promoted
    routers (11) = 99 members, 35 stochastic (239 runs per world-episode at R = 5), unfrozen and
    unpruned. `obs_stock_aware` × a promoted router runs LEGACY
    (`contract_for`); the routers read `env.system` and emit the raw action both contracts
    decode. The two-wave v3 development screen (D3) prunes and freezes it."""
    return Roster(name="roster-v3", replicates=replicates,
                  members=_cross(ROSTER_V3_ORDERING)
                  + _cross(ROSTER_V3_ORDERING, producing=V3_EXTRA_PRODUCING))


def build_pilot_roster_v3(replicates: int = DEFAULT_REPLICATES) -> Roster:
    """The pre-screen pilot's roster (add-gate-v3 D3 step 0): `entity_planner` × the 11 v3
    ordering baselines (23 runs per world-episode), unfrozen. Its name says it is a pilot, not
    a gate roster: its rows never prune or calibrate anything."""
    return Roster(name=PILOT_V3_NAME, replicates=replicates,
                  members=_cross(ROSTER_V3_ORDERING, producing=(PILOT_PRODUCING,)))


def member_policies(m: RosterMember) -> tuple[Policy, Policy]:
    """(producing, ordering) policies of a member."""
    return producing_policy(m.producing), ORDERING_BASELINES[m.ordering]


def member_env(m: RosterMember, config: Configuration, horizon: int) -> TokenomicsAECEnv:
    """The env a member runs in, per its contract. Pairing across contracts holds because demand
    depends only on config + episode seed (D5)."""
    if m.contract == JOINT_STACK:
        return joint_stack_env(config, horizon)
    return TokenomicsAECEnv(config=config, horizon=horizon)


# --- pruning ---


@dataclass(frozen=True)
class DevRow:
    """One member's value on one episode. For a random member `value` is already the replicate
    mean (D5); one row per (world, size, episode, member)."""

    world: str
    split: str
    size: int
    episode: int
    member: str
    value: float


def rows_digest(rows: Iterable[DevRow]) -> str:
    """sha256 over the canonical, order-independent JSON of the evidence rows."""
    canon = sorted(
        (r.world, r.split, r.size, r.episode, r.member, float(r.value)) for r in rows
    )
    return hashlib.sha256(_canonical(canon).encode()).hexdigest()


def argmax_winners(rows: Iterable[DevRow]) -> set[str]:
    """Members that attain the per-episode max on at least one episode. Every member tied at the
    max counts, so a tie never prunes a co-winner."""
    by_ep: dict[EpisodeKey, dict[str, float]] = {}
    for r in rows:
        by_ep.setdefault(EpisodeKey(r.world, r.size, r.episode), {})[r.member] = r.value
    winners: set[str] = set()
    for values in by_ep.values():
        best = max(values.values())
        winners.update(mid for mid, v in values.items() if v == best)
    return winners


def prune_roster(
    roster: Roster, rows: Collection[DevRow], sizes: Collection[int] = GATE_SIZES
) -> Roster:
    """Apply the never-argmax rule to baseline-only development rows (D5) and return a new,
    unfrozen roster carrying the pruning record, hence a new version.

    Refuses, rather than filters, anything outside the rule's evidence: a frozen or already
    pruned roster, non-development rows, rows of non-members (a candidate's rows must never touch
    membership), non-finite values, duplicate rows, and an episode on which some member has no
    row (a missing member could neither win nor be fairly dropped). The evidence must cover every
    size in `sizes` ("at either economy size") and no other."""
    if roster.frozen:
        raise ValueError(f"roster {roster.version} is frozen; pruning happens only before freezing")
    if roster.pruning is not None:
        raise ValueError(f"roster {roster.version} is already pruned")
    if not rows:
        raise ValueError("no development rows to prune on")
    members = set(roster.member_ids)
    seen: set[tuple[EpisodeKey, str]] = set()
    by_ep: dict[EpisodeKey, set[str]] = {}
    for r in rows:
        if r.split != DEVELOPMENT:
            raise ValueError(f"pruning row on split {r.split!r}: only development evidence counts")
        if r.member not in members:
            raise ValueError(f"pruning row for non-member {r.member!r} (candidate rows not allowed)")
        if not math.isfinite(r.value):
            raise ValueError(f"non-finite value for {r.member} on {r.world}/{r.size}/{r.episode}")
        key = EpisodeKey(r.world, r.size, r.episode)
        if (key, r.member) in seen:
            raise ValueError(f"duplicate row for {r.member} on {key}")
        seen.add((key, r.member))
        by_ep.setdefault(key, set()).add(r.member)
    for key, present in by_ep.items():
        if present != members:
            raise ValueError(f"episode {key} lacks rows for {sorted(members - present)}")
    got_sizes = {k.size for k in by_ep}
    if got_sizes != set(sizes):
        raise ValueError(f"evidence covers sizes {sorted(got_sizes)}, rule needs {sorted(sizes)}")

    winners = argmax_winners(rows)
    dropped = tuple(mid for mid in roster.member_ids if mid not in winners)
    record = PruningRecord(
        rule=NEVER_ARGMAX_RULE,
        rule_text=NEVER_ARGMAX_TEXT,
        parent_version=roster.version,
        episodes=tuple(sorted(by_ep, key=lambda k: (k.size, k.world, k.episode))),
        rows_digest=rows_digest(rows),
        dropped=dropped,
    )
    kept = tuple(m for m in roster.members if m.id in winners)
    return replace(roster, members=kept, frozen=False, pruning=record)


def _check_wave_rows(rows: Collection[DevRow], members: set[str], what: str
                     ) -> dict[EpisodeKey, set[str]]:
    """The episodes of one wave's evidence, refusing what `prune_roster` refuses: non-development
    rows, non-members, non-finite values, duplicates and an episode missing a member."""
    seen: set[tuple[EpisodeKey, str]] = set()
    by_ep: dict[EpisodeKey, set[str]] = {}
    for r in rows:
        if r.split != DEVELOPMENT:
            raise ValueError(f"{what}: row on split {r.split!r}: only development evidence counts")
        if r.member not in members:
            raise ValueError(f"{what}: row for {r.member!r}, not a member of this wave")
        if not math.isfinite(r.value):
            raise ValueError(f"{what}: non-finite value for {r.member} on {r.world}/{r.episode}")
        key = EpisodeKey(r.world, r.size, r.episode)
        if (key, r.member) in seen:
            raise ValueError(f"{what}: duplicate row for {r.member} on {key}")
        seen.add((key, r.member))
        by_ep.setdefault(key, set()).add(r.member)
    for key, present in by_ep.items():
        if present != members:
            raise ValueError(f"{what}: episode {key} lacks rows for {sorted(members - present)}")
    return by_ep


def wave2_members(roster: Roster, wave1_rows: Collection[DevRow]) -> tuple[str, ...]:
    """W2 (add-gate-v3 D3 step 2): the members that are the per-episode argmax on at least one
    wave-1 episode (every tied member counts), plus every `entity_planner` member, in roster
    order. `wave1_rows` must already exclude excluded family x size episodes."""
    _check_wave_rows(wave1_rows, set(roster.member_ids), "wave 1")
    winners = argmax_winners(wave1_rows)
    return tuple(m.id for m in roster.members
                 if m.id in winners or m.producing == WAVE2_PRODUCING)


def prune_two_wave(roster: Roster, wave1_rows: Collection[DevRow],
                   wave2_rows: Collection[DevRow], w2: Collection[str]) -> Roster:
    """`never-argmax-two-wave-v1` (add-gate-v3 D3 step 5; gate-harness "Roster pruning uses
    development evidence only" with its pre-registered member waves): keep a member iff it is in
    W2 and is the per-episode argmax on at least one episode of either wave, each episode's argmax
    taken over the members run on it. Returns an unfrozen roster carrying the record (both waves'
    members, episodes and row digests, and the members not carried to wave 2).

    Refuses a frozen or already pruned roster, a W2 that is not `wave2_members`' plan, wave rows
    that do not cover their wave's members on every episode, and empty wave-1 evidence. Wave 2 may
    be empty (no wave-2 world, or its family x size excluded)."""
    if roster.frozen:
        raise ValueError(f"roster {roster.version} is frozen; pruning happens only before freezing")
    if roster.pruning is not None:
        raise ValueError(f"roster {roster.version} is already pruned")
    if not wave1_rows:
        raise ValueError("no wave-1 development rows to prune on")
    w2 = tuple(w2)
    if w2 != wave2_members(roster, wave1_rows):
        raise ValueError(f"W2 {list(w2)} is not the planned wave-2 member set")
    ep1 = _check_wave_rows(wave1_rows, set(roster.member_ids), "wave 1")
    ep2 = _check_wave_rows(wave2_rows, set(w2), "wave 2") if wave2_rows else {}
    if set(ep1) & set(ep2):
        raise ValueError(f"episodes in both waves: {sorted(set(ep1) & set(ep2))[:3]}")
    winners = argmax_winners(wave1_rows) | argmax_winners(wave2_rows)
    kept_ids = [mid for mid in roster.member_ids if mid in w2 and mid in winners]
    if not kept_ids:
        raise ValueError("two-wave pruning would keep no member")
    def order(k: EpisodeKey) -> tuple[int, str, int]:
        return (k.size, k.world, k.episode)

    record = PruningRecord(
        rule=NEVER_ARGMAX_TWO_WAVE_RULE,
        rule_text=NEVER_ARGMAX_TWO_WAVE_TEXT,
        parent_version=roster.version,
        episodes=tuple(sorted([*ep1, *ep2], key=order)),
        rows_digest=rows_digest([*wave1_rows, *wave2_rows]),
        dropped=tuple(mid for mid in roster.member_ids if mid not in kept_ids),
        waves=(WaveRecord(roster.member_ids, tuple(sorted(ep1, key=order)),
                          rows_digest(wave1_rows)),
               WaveRecord(w2, tuple(sorted(ep2, key=order)), rows_digest(wave2_rows))),
        not_carried=tuple(mid for mid in roster.member_ids if mid not in w2),
    )
    kept = tuple(m for m in roster.members if m.id in set(kept_ids))
    return replace(roster, members=kept, frozen=False, pruning=record)


def freeze_roster(roster: Roster) -> Roster:
    """Mark a roster frozen (a new version): after this, pruning refuses it."""
    return replace(roster, frozen=True)
