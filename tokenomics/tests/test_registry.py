"""World registry (design D7, add-measurement-gate-harness; gate-harness "World registry",
"Regimes", "Zero-baseline screening", "Sealed worlds need a pre-registration")."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tokenomics.eval.families import CALM, DENSE, SHOCKED, SPARSE_WIDE
from tokenomics.eval.gate import CANDIDATE, EpisodeRow
from tokenomics.eval.registry import (
    DEFAULT_REGISTRY,
    DEVELOPMENT,
    GATE,
    REPLACED,
    RESERVE,
    SEALED,
    SEALED_ACCESS_LOG,
    Preregistration,
    Registry,
    RegistryHashMismatch,
    SealedAccessError,
    WorldEntry,
    build_worlds_v1,
    check_sealed_access,
    load_preregistration,
    load_registry,
    registry_from_dict,
    screen_zero_baselines,
    world_baseline_means,
)
from tokenomics.eval.roster import Roster, RosterMember, contract_for, member_id
from tokenomics.eval.task_archive import TaskRecord, replay

A = member_id("always_restore", "even_split")
B = member_id("restore_when_low", "proportional")
ROSTER = Roster(name="t", replicates=2, members=tuple(
    RosterMember(p, o, contract_for(p, o), False)
    for p, o in (("always_restore", "even_split"), ("restore_when_low", "proportional"))))


@pytest.fixture(scope="module")
def draft() -> Registry:
    return load_registry()


def _entries_dict(reg: Registry) -> dict:
    d = reg.payload()
    d["content_hash"] = reg.version
    return d


# --- the draft ---


def test_draft_is_reproduced_byte_for_byte(draft: Registry) -> None:
    assert build_worlds_v1().to_json() == DEFAULT_REGISTRY.read_text()
    assert draft == build_worlds_v1()


def test_draft_has_d7_defaults(draft: Registry) -> None:
    for split, calm, shocked in ((DEVELOPMENT, (8, 3), (4, 10)), (SEALED, (16, 3), (8, 10))):
        for family in (DENSE, SPARSE_WIDE):
            for size in (12, 42):
                group = [e for e in draft.gate_worlds(split)
                         if (e.family, e.size) == (family, size)]
                for regime, (n, eps) in ((CALM, calm), (SHOCKED, shocked)):
                    worlds = [e for e in group if e.regime == regime]
                    assert len(worlds) == n
                    assert {len(e.episode_seeds) for e in worlds} == {eps}
                    reserves = sorted((e for e in draft.entries if e.status == RESERVE
                                       and e.group == (split, family, size, regime)),
                                      key=lambda e: e.reserve_position)
                    assert [e.reserve_position for e in reserves] == [0, 1, 2, 3]
    dev = {e.world_seed for e in draft.entries if e.split == DEVELOPMENT}
    sealed = {e.world_seed for e in draft.entries if e.split == SEALED}
    assert min(dev) >= 1000 and max(dev) <= 1999 and min(sealed) >= 5000 and max(sealed) <= 5999


def test_every_draft_world_replays_to_a_valid_config(draft: Registry) -> None:
    """Every entry's generator ref replays through task_archive into a valid Configuration whose
    regime is what the entry declares: calm = every consumer on a fixed schedule; shocked = B9
    seeded random demand, no schedule (gate-harness "Regimes")."""
    for e in draft.entries:
        rec = TaskRecord(task_id=e.id, task_family="gate", generator_version="v1",
                         serializer_version="v1", world_seed=e.world_seed,
                         episode_seed=e.episode_seeds[0], world_spec=e.generator)
        cfg = replay(rec)
        assert len(cfg.producer_configs) == e.size
        if e.regime == CALM:
            assert all(c.schedule for c in cfg.consumers), e.id
        else:
            assert all(not c.schedule and c.is_randomized for c in cfg.consumers), e.id


def test_episode_seeds_unique_and_disjoint_from_world_seeds(draft: Registry) -> None:
    eps = [s for e in draft.entries for s in e.episode_seeds]
    assert len(eps) == len(set(eps))
    assert not set(eps) & {e.world_seed for e in draft.entries}


# --- World registry ---


def test_entry_gives_every_required_field(draft: Registry) -> None:
    e = draft.entry("dev-sparse_wide-42-shocked-00")
    assert (e.family, e.size, e.regime, e.split) == (SPARSE_WIDE, 42, SHOCKED, DEVELOPMENT)
    assert e.generator == {"generator": "sparse_wide_family",
                           "params": {"producers": 42, "regime": SHOCKED, "fanout": 2}}
    assert e.world_seed and e.episode_seeds


def test_overlapping_seeds_rejected(draft: Registry) -> None:
    """gate-harness "Overlapping seeds rejected": the same family and world seed under both splits
    fails validation."""
    dev = draft.entry("dev-dense-12-calm-00")
    sealed_twin = replace(draft.entry("sea-dense-42-calm-00"), world_seed=dev.world_seed)
    entries = tuple(sealed_twin if e.id == sealed_twin.id else e for e in draft.entries)
    with pytest.raises(ValueError, match="dense world seed 1000 is listed under both "
                                         "development and sealed"):
        Registry(name="x", seed_ranges=draft.seed_ranges, entries=entries)
    # a sealed seed in the development range that no dense world uses: caught by the range rule
    stray = replace(draft.entry("sea-dense-42-calm-00"), world_seed=1999)
    with pytest.raises(ValueError, match="outside sealed range"):
        Registry(name="x", seed_ranges=draft.seed_ranges,
                 entries=tuple(stray if e.id == stray.id else e for e in draft.entries))


def test_overlapping_split_ranges_rejected(draft: Registry) -> None:
    with pytest.raises(ValueError, match="ranges overlap"):
        Registry(name="x", seed_ranges=(("development", 1000, 5500), ("sealed", 5000, 5999)),
                 entries=draft.entries)


def test_duplicate_ids_and_shared_episode_seeds_rejected(draft: Registry) -> None:
    e = draft.entries[0]
    with pytest.raises(ValueError, match="duplicate world ids"):
        Registry(name="x", seed_ranges=draft.seed_ranges,
                 entries=draft.entries + (replace(e, world_seed=1999),))
    other = draft.entries[1]
    clash = replace(other, episode_seeds=(e.episode_seeds[0],))
    with pytest.raises(ValueError, match="shared by"):
        Registry(name="x", seed_ranges=draft.seed_ranges,
                 entries=tuple(clash if x.id == other.id else x for x in draft.entries))


def test_generator_ref_must_match_entry(draft: Registry) -> None:
    e = draft.entry("dev-dense-12-calm-00")
    bad = replace(e, generator={"generator": "dense_family",
                                "params": {"producers": 12, "regime": SHOCKED,
                                           "schedule_length": 7}})
    with pytest.raises(ValueError, match="generator regime"):
        Registry(name="x", seed_ranges=draft.seed_ranges,
                 entries=tuple(bad if x.id == e.id else x for x in draft.entries))


def _with_params(draft: Registry, wid: str, **change: object) -> dict:
    """A re-hashed file whose entry `wid` has edited generator params (None deletes a key)."""
    d = _entries_dict(draft)
    for e in d["entries"]:
        if e["id"] == wid:
            for k, v in change.items():
                if v is None:
                    e["generator"]["params"].pop(k)
                else:
                    e["generator"]["params"][k] = v
    del d["content_hash"]
    from tokenomics.eval.roster import content_hash
    d["content_hash"] = content_hash(d)
    return d


@pytest.mark.parametrize("wid,change,match", [
    ("dev-dense-12-calm-00", {"schedule_length": 0}, "does not replay"),
    ("dev-sparse_wide-12-shocked-00", {"fanout": 9}, "does not replay"),
    ("dev-dense-12-calm-00", {"bogus": 1}, "generator params"),
    ("dev-sparse_wide-12-shocked-00", {"fanout": None}, "generator params"),
    ("dev-dense-12-shocked-00", {"schedule_length": 7}, "generator params"),
    ("dev-dense-12-calm-00", {"schedule_length": "7"}, "must be an int"),
    ("dev-dense-12-calm-00", {"producers": 13}, "generator producers"),
])
def test_malformed_generator_ref_refused_on_load(draft: Registry, wid: str, change: dict,
                                                 match: str) -> None:
    """A re-hashed registry whose generator ref cannot replay (or breaks the ref schema) is refused
    on load, not at first use."""
    with pytest.raises(ValueError, match=match):
        registry_from_dict(_with_params(draft, wid, **change))


def test_edited_registry_refused(draft: Registry) -> None:
    d = _entries_dict(draft)
    d["entries"][0]["episode_seeds"] = [7]
    with pytest.raises(RegistryHashMismatch, match="edited without a new version"):
        registry_from_dict(d)
    d = _entries_dict(draft)
    del d["content_hash"]
    with pytest.raises(RegistryHashMismatch):
        registry_from_dict(d)


# --- Regimes ---


def _world(i: int, regime: str, size: int = 42, split: str = DEVELOPMENT) -> WorldEntry:
    wid = f"w-{regime}-{i}"
    params = {"producers": size, "regime": regime}
    if regime == CALM:
        params["schedule_length"] = 7
    return WorldEntry(id=wid, split=split, family=DENSE, size=size, regime=regime,
                      world_seed=1000 + i + (0 if regime == CALM else 500),
                      episode_seeds=(100_000 + i * 10 + (0 if regime == CALM else 5),),
                      generator={"generator": "dense_family", "params": params})


RANGES = (("development", 1000, 1999), ("sealed", 5000, 5999))


def test_too_few_calm_worlds() -> None:
    """gate-harness "Too few calm worlds": 10 shocked and 15 calm at 42 fails validation."""
    entries = [_world(i, SHOCKED) for i in range(10)] + [_world(i, CALM) for i in range(15)]
    with pytest.raises(ValueError, match="15 calm worlds < 2x 10 shocked"):
        Registry(name="x", seed_ranges=RANGES, entries=tuple(entries))
    ok = [_world(i, SHOCKED) for i in range(10)] + [_world(i, CALM) for i in range(20)]
    Registry(name="x", seed_ranges=RANGES, entries=tuple(ok))  # exactly 2x passes


def test_world_without_valid_regime_rejected() -> None:
    bad = replace(_world(0, CALM), regime="stormy")
    with pytest.raises(ValueError, match="no valid regime"):
        Registry(name="x", seed_ranges=RANGES, entries=(bad,))


def test_calm_demand_ignores_episode_seed_shocked_does_not(draft: Registry) -> None:
    """A calm registry world has deterministic demand; a shocked one is seeded (B9)."""
    from tokenomics.env import TokenomicsAECEnv

    def digest(e: WorldEntry, seed: int) -> str:
        env = TokenomicsAECEnv(config=e.config(), horizon=12)
        env.reset(seed=seed)
        return env.episode_stats["demand_digest"]

    calm = draft.entry("dev-dense-12-calm-00")
    shocked = draft.entry("dev-dense-12-shocked-00")
    assert digest(calm, calm.episode_seeds[0]) == digest(calm, calm.episode_seeds[1])
    assert digest(shocked, shocked.episode_seeds[0]) != digest(shocked, shocked.episode_seeds[1])


# --- Zero-baseline screening ---


def _rows(reg: Registry, means: dict[str, tuple[float, float]], *,
          subject_override: str | None = None) -> list[EpisodeRow]:
    """Roster rows: on every episode of each listed world, member A delivers `a` and B `b`
    (one replicate each: ROSTER has no random member)."""
    rows = []
    for wid, (a, b) in means.items():
        e = reg.entry(wid)
        for ep in e.episode_seeds:
            for mid, units in ((A, a), (B, b)):
                rows.append(EpisodeRow(
                    world=wid, size=e.size, regime=e.regime, episode=ep,
                    subject=subject_override or mid, replicate=0, units=units, dead=False,
                    demand_digest="d", r_start=1.0, r_end=1.0, consumer_fill_rate={}))
    return rows


def _all_gate(reg: Registry, value: float = 5.0) -> dict[str, tuple[float, float]]:
    """Every development gate world (M1 screens development only)."""
    return {e.id: (value, 1.0) for e in reg.gate_worlds(DEVELOPMENT)}


def test_baseline_means_are_per_episode_max_means(draft: Registry) -> None:
    wid = "dev-dense-12-calm-00"
    e = draft.entry(wid)
    rows = []
    for ep, (a, b) in zip(e.episode_seeds, [(10, 6), (4, 8), (0, 0)], strict=True):
        for mid, u in ((A, a), (B, b)):
            rows.append(EpisodeRow(world=wid, size=12, regime=CALM, episode=ep, subject=mid,
                                   replicate=0, units=u, dead=False, demand_digest="d",
                                   r_start=1.0, r_end=1.0, consumer_fill_rate={}))
    assert world_baseline_means(draft, ROSTER, rows) == {wid: 6.0}  # (10 + 8 + 0) / 3


def test_zero_world_replaced_by_next_reserve(draft: Registry) -> None:
    """A zero-baseline gate world is replaced by its group's next reserve with a non-zero
    baseline, in reserve order; a zero reserve is skipped; the outcome is recorded and the
    registry gets a new version."""
    means = _all_gate(draft)
    zero = "dev-sparse_wide-42-calm-03"
    means[zero] = (0.0, 0.0)
    means["dev-sparse_wide-42-calm-r0"] = (0.0, 0.0)
    means["dev-sparse_wide-42-calm-r1"] = (2.0, 3.0)
    screened = screen_zero_baselines(draft, ROSTER, _rows(draft, means))
    assert screened.version != draft.version
    assert screened.entry(zero).status == REPLACED
    assert screened.entry("dev-sparse_wide-42-calm-r0").status == REPLACED
    assert screened.entry("dev-sparse_wide-42-calm-r1").status == GATE
    assert screened.entry("dev-sparse_wide-42-calm-r2").status == RESERVE
    (rep,) = screened.screening.replacements
    assert (rep.replaced, rep.by, rep.skipped) == (
        zero, "dev-sparse_wide-42-calm-r1", ("dev-sparse_wide-42-calm-r0",))
    assert screened.screening.roster_version == ROSTER.version
    assert len(screened.gate_worlds()) == len(draft.gate_worlds())
    # the screened registry round-trips through its file form with the outcome recorded
    assert registry_from_dict(json.loads(screened.to_json())) == screened
    with pytest.raises(ValueError, match="already screened"):
        screen_zero_baselines(screened, ROSTER, _rows(screened, means))


def test_replacement_is_candidate_independent(draft: Registry) -> None:
    """gate-harness "Replacement is candidate-independent": screening takes roster rows only, so
    any candidate row is refused and the result depends on nothing a candidate produced."""
    means = _all_gate(draft)
    means["dev-dense-12-shocked-00"] = (0.0, 0.0)
    means["dev-dense-12-shocked-r0"] = (1.0, 0.0)
    rows = _rows(draft, means)
    a = screen_zero_baselines(draft, ROSTER, rows)
    assert a == screen_zero_baselines(draft, ROSTER, list(reversed(rows)))
    e = draft.entry("dev-dense-12-shocked-00")
    cand = replace(rows[0], subject=CANDIDATE, world=e.id, episode=e.episode_seeds[0],
                   size=e.size, regime=e.regime, units=99.0)
    with pytest.raises(ValueError, match="screening is roster-only"):
        screen_zero_baselines(draft, ROSTER, rows + [cand])
    with pytest.raises(ValueError, match="not a member"):
        screen_zero_baselines(draft, ROSTER, rows + [replace(rows[0], subject="noop/random")])


def test_screening_refuses_incomplete_evidence(draft: Registry) -> None:
    means = _all_gate(draft)
    del means["dev-dense-42-calm-07"]
    with pytest.raises(ValueError, match="no baseline rows for gate worlds"):
        screen_zero_baselines(draft, ROSTER, _rows(draft, means))
    means = _all_gate(draft)
    rows = [r for r in _rows(draft, means) if not (r.world == "dev-dense-12-calm-00"
                                                   and r.subject == B)]
    with pytest.raises(ValueError, match="has replicates"):
        screen_zero_baselines(draft, ROSTER, rows)
    means = _all_gate(draft)
    means["dev-dense-12-calm-00"] = (0.0, 0.0)  # needs r0, which has no rows
    with pytest.raises(ValueError, match="consulted but has no baseline rows"):
        screen_zero_baselines(draft, ROSTER, _rows(draft, means))


def test_screening_raises_when_reserves_run_out(draft: Registry) -> None:
    means = _all_gate(draft)
    group = [f"dev-dense-12-shocked-0{i}" for i in range(4)]
    for wid in group:
        means[wid] = (0.0, 0.0)
    for k in range(4):
        means[f"dev-dense-12-shocked-r{k}"] = (0.0 if k == 3 else 1.0, 0.0)
    with pytest.raises(ValueError, match="no usable reserve left"):
        screen_zero_baselines(draft, ROSTER, _rows(draft, means))


# --- Sealed worlds need a pre-registration ---


def _ckpts(tmp_path: Path) -> dict[str, Path]:
    out = {}
    for role, body in (("orderer", b"orderer-weights"), ("producer", b"producer-weights")):
        p = tmp_path / f"{role}.pt"
        p.write_bytes(body)
        out[role] = p
    return out


def _manifest(tmp_path: Path, reg: Registry, ckpts: dict[str, Path], **override: object) -> Path:
    from tokenomics.eval.provenance import sha256_file

    d = {"checkpoints": {r: sha256_file(str(p)) for r, p in ckpts.items()},
         "roster_version": "roster-v", "registry_version": reg.version,
         "params_version": "params-v", **override}
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(d))
    return path


SEALED_WORLD = "sea-dense-12-calm-00"
NOW = datetime(2026, 10, 26, tzinfo=timezone.utc)


def _check(reg: Registry, tmp_path: Path, prereg: Preregistration | None,
           ckpts: dict[str, Path], worlds: tuple[str, ...] = (SEALED_WORLD,)) -> dict | None:
    return check_sealed_access(reg, worlds, preregistration=prereg, checkpoints=ckpts,
                               roster_version="roster-v", params_version="params-v",
                               log_path=tmp_path / "SEALED_ACCESS.log", now=NOW)


def test_log_default_is_the_experiments_file() -> None:
    assert SEALED_ACCESS_LOG == Path("/srv/research/tokenomics/experiments/SEALED_ACCESS.log")


def test_sealed_run_without_manifest_refused(draft: Registry, tmp_path: Path) -> None:
    with pytest.raises(SealedAccessError, match="without a pre-registration manifest"):
        _check(draft, tmp_path, None, _ckpts(tmp_path))
    assert not (tmp_path / "SEALED_ACCESS.log").exists()


def test_development_run_needs_no_manifest(draft: Registry, tmp_path: Path) -> None:
    assert _check(draft, tmp_path, None, {}, worlds=("dev-dense-12-calm-00",)) is None
    assert not (tmp_path / "SEALED_ACCESS.log").exists()


def test_checkpoint_differs_from_manifest_refused(draft: Registry, tmp_path: Path) -> None:
    ckpts = _ckpts(tmp_path)
    prereg = load_preregistration(_manifest(tmp_path, draft, ckpts))
    ckpts["producer"].write_bytes(b"retrained-producer")
    with pytest.raises(SealedAccessError, match="'producer' sha256"):
        _check(draft, tmp_path, prereg, ckpts)
    assert not (tmp_path / "SEALED_ACCESS.log").exists()  # nothing recorded


@pytest.mark.parametrize("field,value,match", [
    ("registry_version", "other", "registry version"),
    ("roster_version", "other", "roster version"),
    ("params_version", "other", "parameter version"),
])
def test_version_mismatch_refused(draft: Registry, tmp_path: Path, field: str, value: str,
                                  match: str) -> None:
    ckpts = _ckpts(tmp_path)
    prereg = load_preregistration(_manifest(tmp_path, draft, ckpts, **{field: value}))
    with pytest.raises(SealedAccessError, match=match):
        _check(draft, tmp_path, prereg, ckpts)


def test_checkpoint_roles_must_match(draft: Registry, tmp_path: Path) -> None:
    ckpts = _ckpts(tmp_path)
    prereg = load_preregistration(_manifest(tmp_path, draft, ckpts))
    with pytest.raises(SealedAccessError, match="checkpoint roles"):
        _check(draft, tmp_path, prereg, {"orderer": ckpts["orderer"]})


def test_empty_checkpoint_pins_refused(draft: Registry, tmp_path: Path) -> None:
    """A manifest that binds no candidate checkpoint is refused on every path (file, direct
    construction, and an object that bypassed validation) and nothing is logged."""
    with pytest.raises(ValueError, match="names no checkpoints"):
        Preregistration(checkpoints=(), roster_version="roster-v",
                        registry_version=draft.version, params_version="params-v")
    with pytest.raises(ValueError, match="names no checkpoints"):
        load_preregistration(_manifest(tmp_path, draft, {}))
    with pytest.raises(ValueError, match="not a sha256"):
        Preregistration(checkpoints=(("orderer", "abc"),), roster_version="roster-v",
                        registry_version=draft.version, params_version="params-v")
    ckpts = _ckpts(tmp_path)
    prereg = load_preregistration(_manifest(tmp_path, draft, ckpts))
    hollow = object.__new__(Preregistration)
    for k, v in (("checkpoints", ()), ("roster_version", "roster-v"),
                 ("registry_version", draft.version), ("params_version", "params-v"),
                 ("sha256", "")):
        object.__setattr__(hollow, k, v)
    for pre, given in ((hollow, {}), (prereg, {})):
        with pytest.raises(SealedAccessError, match="at least one candidate checkpoint"):
            _check(draft, tmp_path, pre, given)
    assert not (tmp_path / "SEALED_ACCESS.log").exists()


def test_sealed_access_appended_to_log(draft: Registry, tmp_path: Path) -> None:
    ckpts = _ckpts(tmp_path)
    manifest = _manifest(tmp_path, draft, ckpts)
    prereg = load_preregistration(manifest)
    rec = _check(draft, tmp_path, prereg, ckpts,
                 worlds=(SEALED_WORLD, "dev-dense-12-calm-00", "sea-sparse_wide-42-shocked-01"))
    rec2 = _check(draft, tmp_path, prereg, ckpts)
    lines = (tmp_path / "SEALED_ACCESS.log").read_text().splitlines()
    assert [json.loads(x) for x in lines] == [rec, rec2]
    assert rec["worlds"] == sorted([SEALED_WORLD, "sea-sparse_wide-42-shocked-01"])
    assert rec["registry_version"] == draft.version
    assert rec["checkpoints"] == dict(prereg.checkpoints)
    assert rec["time"] == NOW.isoformat() and rec["preregistration_sha256"] == prereg.sha256


def test_unknown_world_refused(draft: Registry, tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        _check(draft, tmp_path, None, {}, worlds=("nope",))


# --- screening by split, headroom record (D7, D10) ---


def test_screening_covers_development_only_by_default(draft: Registry) -> None:
    screened = screen_zero_baselines(draft, ROSTER, _rows(draft, _all_gate(draft)))
    assert screened.screening.splits == (DEVELOPMENT,)
    assert screened.entry("sea-dense-12-calm-00").status == GATE  # sealed untouched
    sealed = {e.id: (5.0, 1.0) for e in draft.gate_worlds(SEALED)}
    with pytest.raises(ValueError, match="outside the screened splits"):
        screen_zero_baselines(draft, ROSTER, _rows(draft, {**_all_gate(draft), **sealed}))
    both = screen_zero_baselines(draft, ROSTER, _rows(draft, {**_all_gate(draft), **sealed}),
                                 splits=(DEVELOPMENT, SEALED))
    assert both.screening.splits == (DEVELOPMENT, SEALED)


def test_draft_preregisters_saturation_ceiling(draft: Registry) -> None:
    assert draft.saturation_ceiling == 0.95
    assert json.loads(DEFAULT_REGISTRY.read_text())["saturation_ceiling"] == 0.95
    with pytest.raises(ValueError, match="saturation ceiling"):
        replace(draft, saturation_ceiling=1.5)


def test_headroom_recorded_in_screened_registry(draft: Registry) -> None:
    from tokenomics.eval.gate import Headroom
    from tokenomics.eval.registry import record_headroom

    screened = screen_zero_baselines(draft, ROSTER, _rows(draft, _all_gate(draft)))
    results = [Headroom(DENSE, 12, 8, 0.6, 0.95, True), Headroom(SPARSE_WIDE, 42, 8, 0.97, 0.95,
                                                                 False)]
    with pytest.raises(ValueError, match="headroom is recorded on a screened registry"):
        record_headroom(draft, results)
    rec = record_headroom(screened, results)
    assert rec.version != screened.version
    assert registry_from_dict(json.loads(rec.to_json())) == rec
    assert rec.screening.headroom == tuple(results)
    with pytest.raises(ValueError, match="already records"):
        record_headroom(rec, results)
    with pytest.raises(ValueError, match="pre-registers 0.95"):
        record_headroom(screened, [Headroom(DENSE, 12, 8, 0.6, 0.9, True)])
