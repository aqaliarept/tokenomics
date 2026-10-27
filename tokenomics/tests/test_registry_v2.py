"""Registry v2 and the shocks-i profile (ADR-0055; add-shocks-i-drift design D7, D8;
gate-harness "Regimes", "Versioned registry for shocked worlds")."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tokenomics.domain.shocks import ShockKind
from tokenomics.env.shock_schedule import build_schedule
from tokenomics.eval.families import (CALM, FAMILIES, SHOCKED, SPARSE_WIDE, build_family_config,
                                      family_demand_ratio)
from tokenomics.eval.registry import (DEFAULT_REGISTRY, REGISTRY_V2, Registry, WorldEntry,
                                      build_worlds_v1, build_worlds_v2, load_registry,
                                      registry_from_dict)
from tokenomics.eval.roster import load_roster
from tokenomics.eval.scale import config_to_dict
from tokenomics.eval.shock_profiles import (DEFAULT_SHOCK_PROFILES, SHOCKS_I,
                                            ShockProfileHashMismatch, _write_v1,
                                            load_shock_profile, shock_profiles_from_dict)

REGISTRY_DIR = DEFAULT_REGISTRY.parent
V1_HASHES = {  # EXP-0044 attempt-2 frozen files (task 11.1 of add-measurement-gate-harness)
    "worlds_v1.json": "566946fe0c0363ec888f9a58bfe7bd480bcc948e049c93e9923088e2cab911e3",
    "roster_v1.json": "c2832dbb44d26601f279fc04c0c483c543c5a5fcdc8cb2231090fc03f76a3731",
    "side_effect_params_v1.json": "e6651b2ebe541b18ade0a8904f27b0a08844e2a4f8ee0e231ca062514472f2c1",
}


@pytest.fixture(scope="module")
def v2() -> Registry:
    return load_registry(REGISTRY_V2)


@pytest.fixture(scope="module")
def v1() -> Registry:
    return load_registry(DEFAULT_REGISTRY)


# --- v1 untouched ---


def test_v1_untouched(v1: Registry) -> None:
    """Scenario "v1 untouched": the three frozen v1 files keep their recorded hashes and every
    v1 world replays exactly as before (no profile, no shocks, no drift)."""
    assert v1.version == V1_HASHES["worlds_v1.json"]
    assert load_roster(REGISTRY_DIR / "roster_v1.json").version == V1_HASHES["roster_v1.json"]
    params = json.loads((REGISTRY_DIR / "side_effect_params_v1.json").read_text())
    assert params["content_hash"] == V1_HASHES["side_effect_params_v1.json"]
    for name, h in V1_HASHES.items():
        assert json.loads((REGISTRY_DIR / name).read_text())["content_hash"] == h
    assert v1.shock_profiles is None and "shock_profiles" not in v1.payload()
    assert v1.version_tag == 1
    for e in v1.entries:
        assert "shock_profile" not in e.generator["params"]
        cfg = e.config()
        assert cfg.shocks is None and cfg.demand_drift is None


# --- v2 draft ---


def test_v2_draft_reproduces_byte_for_byte(tmp_path: Path) -> None:
    assert build_worlds_v2().to_json() == REGISTRY_V2.read_text()
    out = tmp_path / "profiles.json"
    _write_v1(out)
    assert out.read_bytes() == DEFAULT_SHOCK_PROFILES.read_bytes()


def test_calm_worlds_carried_over(v1: Registry, v2: Registry) -> None:
    """Scenario "Calm worlds carried over": same ids, world seeds, episode seeds and replayed
    configs as v1, for every v2 calm entry; shocked entries keep their v1 seeds too."""
    draft_v1 = {e.id: e for e in build_worlds_v1().entries}
    frozen_v1 = {e.id: e for e in v1.entries}
    assert set(e.id for e in v2.entries) == set(draft_v1) == set(frozen_v1)
    for e in v2.entries:
        d, f = draft_v1[e.id], frozen_v1[e.id]
        assert (e.world_seed, e.episode_seeds) == (d.world_seed, d.episode_seeds) == (
            f.world_seed, f.episode_seeds)
        if e.regime == CALM:
            assert e == d
            assert e.generator == f.generator
            assert config_to_dict(e.config()) == config_to_dict(f.config())
        else:
            params = dict(e.generator["params"])
            assert params.pop("shock_profile") == SHOCKS_I
            assert params == d.generator["params"]


def test_v2_shocked_worlds_carry_shocks_and_drift(v2: Registry) -> None:
    profile = load_shock_profile(SHOCKS_I)
    assert v2.version_tag == 2 and v2.shock_profiles is not None
    for e in v2.entries:
        cfg = e.config()
        if e.regime == CALM:
            assert cfg.shocks is None and cfg.demand_drift is None
            continue
        assert cfg.demand_drift is not None and cfg.demand_drift.j == profile.drift_j
        counts = {g.kind: g.count for g in cfg.shocks.generators}
        assert counts == ({ShockKind.CAPACITY_HIT: 2, ShockKind.PRODUCER_OUTAGE: 1}
                          if e.size == 12 else
                          {ShockKind.CAPACITY_HIT: 7, ShockKind.PRODUCER_OUTAGE: 4})
        assert all(g.targets == () for g in cfg.shocks.generators)


def test_resolved_counts_and_schedule() -> None:
    """Counts are round_half_up(P * rate): hits 2 / outages 1 at 12, 7 / 4 at 42; a schedule
    at horizon 100 has exactly those events inside cycles 21-90."""
    for producers, (hits, outages) in ((12, (2, 1)), (42, (7, 4))):
        cfg = build_family_config(1003, family=SPARSE_WIDE, producers=producers, regime=SHOCKED,
                                  shock_profile=SHOCKS_I)
        schedule = build_schedule(cfg, 7, 100)
        kinds = [e.kind for e in schedule]
        assert kinds.count(ShockKind.CAPACITY_HIT) == hits
        assert kinds.count(ShockKind.PRODUCER_OUTAGE) == outages
        assert all(21 <= e.start_cycle <= 90 for e in schedule)
        assert build_schedule(cfg, 7, 12) == ()  # the window starts after cycle 20


@pytest.mark.parametrize("producers", [12, 42])
def test_profile_changes_nothing_but_shocks_and_drift(producers: int) -> None:
    plain = build_family_config(1007, family=SPARSE_WIDE, producers=producers, regime=SHOCKED)
    shocked = build_family_config(1007, family=SPARSE_WIDE, producers=producers, regime=SHOCKED,
                                  shock_profile=SHOCKS_I)
    assert replace(shocked, shocks=None, demand_drift=None) == plain


def test_families_ratio_with_the_profile() -> None:
    """The 12-vs-42 family ratio contract holds with the profile on (it only adds shocks and
    drift, whose stationary demand marginal equals B9's), and equals the profile-free ratio."""
    fam = FAMILIES[SPARSE_WIDE]
    r12 = family_demand_ratio(SPARSE_WIDE, 12, SHOCKED, shock_profile=SHOCKS_I)
    r42 = family_demand_ratio(SPARSE_WIDE, 42, SHOCKED, shock_profile=SHOCKS_I)
    assert abs(r42 / r12 - 1) <= fam.ratio_tolerance
    assert r12 == family_demand_ratio(SPARSE_WIDE, 12, SHOCKED)
    assert r42 == family_demand_ratio(SPARSE_WIDE, 42, SHOCKED)


# --- Regimes ---


def _with_entry(reg: Registry, world_id: str, **params_patch: object) -> Registry:
    entries = []
    for e in reg.entries:
        if e.id == world_id:
            g = copy.deepcopy(e.generator)
            for k, v in params_patch.items():
                if v is None:
                    g["params"].pop(k, None)
                else:
                    g["params"][k] = v
            e = replace(e, generator=g)
        entries.append(e)
    return replace(reg, entries=tuple(entries))


def test_calm_world_with_shocks(v2: Registry) -> None:
    """Scenario "Calm world with shocks": a calm entry naming a profile fails, naming it."""
    calm = next(e for e in v2.entries if e.regime == CALM)
    with pytest.raises(ValueError, match=rf"{calm.id}: a calm world names shock profile"):
        _with_entry(v2, calm.id, shock_profile=SHOCKS_I)
    with pytest.raises(ValueError, match=r"a calm world carries no shock profile"):
        build_family_config(1000, family=SPARSE_WIDE, producers=12, regime=CALM,
                            shock_profile=SHOCKS_I)


def test_shocked_v2_world_without_shocks(v2: Registry) -> None:
    """Scenario "Shocked v2 world without shocks": a v2 shocked entry without a profile fails."""
    shocked = next(e for e in v2.entries if e.regime == SHOCKED and e.status == "gate")
    with pytest.raises(ValueError, match=rf"{shocked.id}: a registry v2 shocked world names no "
                                         "shock profile"):
        _with_entry(v2, shocked.id, shock_profile=None)
    reserve = next(e for e in v2.entries if e.regime == SHOCKED and e.status == "reserve")
    with pytest.raises(ValueError, match=rf"{reserve.id}: a registry v2 shocked world"):
        _with_entry(v2, reserve.id, shock_profile=None)


def test_profile_needs_a_recorded_hash(v2: Registry) -> None:
    with pytest.raises(ValueError, match=r"names a shock profile but the registry records no"):
        replace(v2, shock_profiles=None)


def test_derived_scope_keeps_the_version_tag(v2: Registry) -> None:
    scope = replace(v2, name="worlds-v2-dry-run")
    assert scope.version_tag == 2
    assert replace(v2, name="worlds-v12").version_tag == 12


# --- profile file versioning ---


def test_edited_profile_file_refused() -> None:
    d = json.loads(DEFAULT_SHOCK_PROFILES.read_text())
    d["profiles"][SHOCKS_I]["drift_j"] = 0.5
    with pytest.raises(ShockProfileHashMismatch, match=r"edited without a new version"):
        shock_profiles_from_dict(d)


def test_registry_bound_to_another_profile_file_refused() -> None:
    d = json.loads(REGISTRY_V2.read_text())
    d["shock_profiles"] = "0" * 64
    d.pop("content_hash")
    from tokenomics.eval.roster import content_hash

    d["content_hash"] = content_hash(d)
    with pytest.raises(ValueError, match=r"the packaged profile file is"):
        registry_from_dict(d)


def test_entry_naming_an_unknown_profile_does_not_replay(v2: Registry) -> None:
    shocked = next(e for e in v2.entries if e.regime == SHOCKED)
    bad = _with_entry(v2, shocked.id, shock_profile="no-such-profile")
    with pytest.raises(ValueError, match=r"does not replay.*unknown shock profile"):
        bad.check_replay()


def test_world_entry_type_is_unchanged() -> None:
    assert {f for f in WorldEntry.__dataclass_fields__} == {
        "id", "split", "family", "size", "regime", "world_seed", "episode_seeds", "generator",
        "status", "reserve_position"}


@pytest.mark.parametrize("kind, producers, count", [
    (ShockKind.PRODUCER_OUTAGE, 5, 0), (ShockKind.PRODUCER_OUTAGE, 6, 1),
    (ShockKind.PRODUCER_OUTAGE, 7, 1), (ShockKind.PRODUCER_OUTAGE, 12, 1),
    (ShockKind.PRODUCER_OUTAGE, 18, 2), (ShockKind.PRODUCER_OUTAGE, 30, 3),
    (ShockKind.PRODUCER_OUTAGE, 42, 4),
    (ShockKind.CAPACITY_HIT, 2, 0), (ShockKind.CAPACITY_HIT, 3, 1), (ShockKind.CAPACITY_HIT, 4, 1),
    (ShockKind.CAPACITY_HIT, 12, 2), (ShockKind.CAPACITY_HIT, 15, 3), (ShockKind.CAPACITY_HIT, 42, 7),
])
def test_counts_round_half_up(kind: ShockKind, producers: int, count: int) -> None:
    """round_half_up, not banker's rounding: ties 0.5 -> 1 (P6 outages, P3 hits) and 2.5 -> 3
    (P30 outages, P15 hits), where round() would give 0 and 2."""
    k = next(x for x in load_shock_profile(SHOCKS_I).kinds if x.kind is kind)
    assert k.count(producers) == count


def test_apply_shock_profile_adds_drift_only_to_randomized_worlds() -> None:
    """Drift attaches only when the world has B9-randomized consumers: a fixed-demand world
    (phase1) and a world whose consumers replay explicit cyclic schedules (a calm family
    world, even though its B9 fields look randomized) get the shocks and no drift; nothing else
    in either config moves."""
    from tokenomics.domain.config import load_config
    from tokenomics.eval.shock_profiles import apply_shock_profile

    profile = load_shock_profile(SHOCKS_I)
    phase1 = load_config(REGISTRY_DIR.parents[1] / "config" / "phase1.json")
    calm = build_family_config(1002, family=SPARSE_WIDE, producers=12, regime=CALM)
    assert all(c.products and c.quantity_max > c.quantity_min and c.schedule
               for c in calm.consumers)
    randomized = build_family_config(1002, family=SPARSE_WIDE, producers=12, regime=SHOCKED)
    for cfg, want_drift in ((phase1, None), (calm, None), (randomized, profile.drift_j)):
        out = apply_shock_profile(cfg, profile, len(cfg.producer_configs))
        assert out.shocks is not None and len(out.shocks.generators) == 2
        assert (None if out.demand_drift is None else out.demand_drift.j) == want_drift
        assert replace(out, shocks=None, demand_drift=None) == cfg
    with pytest.raises(ValueError, match=r"resolved for 5 producers, world has 4"):
        apply_shock_profile(phase1, profile, 5)


def test_decoded_shocks_i_matches_the_pre_registered_d7_values() -> None:
    """Literal D7 values (add-shocks-i-drift design D7), checked on the decoded profile and on
    a world's resolved generators, so a decoder or file change that moves any applied
    parameter fails here."""
    p = load_shock_profile(SHOCKS_I)
    decoded = {k.kind: (k.rate, k.start_window, k.duration, k.magnitude) for k in p.kinds}
    assert decoded == {
        ShockKind.CAPACITY_HIT: ((1, 6), (21, 90), (1, 1), (0.3, 0.6)),
        ShockKind.PRODUCER_OUTAGE: ((1, 12), (21, 90), (3, 6), (1.0, 1.0)),
    }
    assert p.drift_j == 0.34
    assert (p.resilience.window, p.resilience.tolerance) == (5, 0.05)
    cfg = build_family_config(1001, family=SPARSE_WIDE, producers=42, regime=SHOCKED,
                              shock_profile=SHOCKS_I)
    resolved = {g.kind: (g.count, g.start_window, g.duration, g.magnitude, g.targets)
                for g in cfg.shocks.generators}
    assert resolved == {
        ShockKind.CAPACITY_HIT: (7, (21, 90), (1, 1), (0.3, 0.6), ()),
        ShockKind.PRODUCER_OUTAGE: (4, (21, 90), (3, 6), (1.0, 1.0), ()),
    }
    assert cfg.demand_drift.j == 0.34
