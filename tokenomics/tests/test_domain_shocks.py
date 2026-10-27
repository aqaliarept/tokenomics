"""Shock model schema and validation (ADR-0055; add-shocks-i-drift design D1, D5;
stochastic-environment "Shock kinds", "Shock delivery order").

Only the config layer is under test here: kinds, targets, start cycles, the rejection of
unbuilt kinds, the JSON round-trip, and the fingerprint of configs without shocks (pinned at
its pre-M1 values, restored by this change). Mechanics (capacity hits, outages) are tested where they land (tasks 2.1, 2.2).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tokenomics.domain.config import Configuration, load_config
from tokenomics.domain.shocks import (
    IMPLEMENTED_SHOCK_KINDS,
    ShockEvent,
    ShockGenerator,
    ShockKind,
    ShockModel,
    ShockTarget,
    shock_model_from_dict,
    shock_model_to_dict,
)
from tokenomics.env.spaces import config_fingerprint
from tokenomics.eval.families import build_family_config
from tokenomics.eval.scale import build_scale_config, config_to_dict

PKG = Path(__file__).resolve().parents[1] / "tokenomics"
PHASE1 = PKG / "config" / "phase1.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# phase1: p1..p4 each own one type, capacity-1..capacity-4; consumer c1.
HIT = ShockEvent(ShockKind.CAPACITY_HIT, ShockTarget(producer="p1", capacity_types=("capacity-1",)),
                 start_cycle=5, duration=1, magnitude=0.4)
OUTAGE = ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer="p2"),
                    start_cycle=7, duration=3, magnitude=1.0)
HIT_GEN = ShockGenerator(ShockKind.CAPACITY_HIT, count=2, start_window=(21, 90), duration=(1, 1),
                         magnitude=(0.3, 0.6))
OUTAGE_GEN = ShockGenerator(ShockKind.PRODUCER_OUTAGE, count=1, start_window=(21, 90),
                            duration=(3, 6), magnitude=(1.0, 1.0), targets=("p3", "p4"))


def _phase1() -> Configuration:
    return load_config(PHASE1)


def _with(*events: ShockEvent, generators: tuple[ShockGenerator, ...] = ()) -> Configuration:
    return replace(_phase1(), shocks=ShockModel(events=tuple(events), generators=generators))


# --- Shock kinds ---


def test_exactly_five_kinds_two_implemented() -> None:
    assert {k.value for k in ShockKind} == {
        "capacity_hit", "producer_outage", "link_cut", "demand_surge", "fund_shock"}
    assert IMPLEMENTED_SHOCK_KINDS == {ShockKind.CAPACITY_HIT, ShockKind.PRODUCER_OUTAGE}


def test_schedule_lists_a_shock() -> None:
    """Scenario "Schedule lists a shock": every entry names one of the five kinds with its
    target, start cycle, duration and magnitude."""
    cfg = _with(HIT, OUTAGE)
    cfg.validate()
    assert cfg.shocks is not None
    for e in cfg.shocks.events:
        assert isinstance(e.kind, ShockKind)
        assert isinstance(e.target, ShockTarget) and e.target.producer is not None
        assert isinstance(e.start_cycle, int) and isinstance(e.duration, int)
        assert isinstance(e.magnitude, float)


@pytest.mark.parametrize("event, match", [
    (replace(HIT, target=ShockTarget(producer="p1", capacity_types=("capacity-2",))),
     r"shock event #0 \(capacity_hit\).*does not own capacity type\(s\) \['capacity-2'\]"),
    (replace(HIT, target=ShockTarget(producer="nope", capacity_types=("capacity-1",))),
     r"shock event #0 \(capacity_hit\).*producer 'nope' does not exist"),
    (replace(OUTAGE, target=ShockTarget(producer="ghost")),
     r"shock event #0 \(producer_outage\).*producer 'ghost' does not exist"),
    (replace(HIT, target=ShockTarget(producer="p1")), r"at least one capacity type"),
    (replace(HIT, target=ShockTarget(producer="p1", capacity_types=("capacity-1",), consumer="c1")),
     r"names only a producer"),
    (replace(OUTAGE, target=ShockTarget(producer="p2", capacity_types=("capacity-2",))),
     r"outage target names no capacity types"),
])
def test_target_fits_the_kind(event: ShockEvent, match: str) -> None:
    """Scenario "Target fits the kind": an unowned capacity type or a missing producer fails
    validation, naming the entry and the bad target."""
    with pytest.raises(ValueError, match=match):
        _with(event).validate()


@pytest.mark.parametrize("event, match", [
    (replace(HIT, duration=2), r"capacity hit has duration 1"),
    (replace(HIT, magnitude=1.0), r"magnitude must be in \(0, 1\)"),
    (replace(HIT, magnitude=0.0), r"magnitude must be in \(0, 1\)"),
    (replace(HIT, magnitude=float("nan")), r"magnitude must be finite"),
    (replace(OUTAGE, duration=0), r"duration must be >= 1"),
    (replace(OUTAGE, magnitude=0.5), r"outage magnitude must be 1.0"),
])
def test_kind_rules_on_duration_and_magnitude(event: ShockEvent, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _with(event).validate()


@pytest.mark.parametrize("event", [replace(HIT, start_cycle=1), replace(OUTAGE, start_cycle=1),
                                   replace(OUTAGE, start_cycle=0)])
def test_shocks_start_after_the_first_cycle(event: ShockEvent) -> None:
    """Scenario "Shocks start after the first cycle"."""
    with pytest.raises(ValueError, match=r"start cycle must be >= 2"):
        _with(event).validate()


def test_generator_window_starts_after_the_first_cycle() -> None:
    with pytest.raises(ValueError, match=r"start window must begin at cycle >= 2"):
        _with(generators=(replace(HIT_GEN, start_window=(1, 10)),)).validate()


@pytest.mark.parametrize("event, match", [
    (replace(HIT, start_cycle=2.5), r"shock event #0 \(capacity_hit\): start cycle must be an integer"),
    (replace(HIT, start_cycle=float("nan")), r"start cycle must be an integer"),
    (replace(HIT, start_cycle=True), r"start cycle must be an integer"),
    (replace(OUTAGE, duration=1.5), r"shock event #0 \(producer_outage\): duration must be an integer"),
    (replace(OUTAGE, magnitude="1.0"), r"magnitude must be finite"),
])
def test_event_discrete_fields_are_integers(event: ShockEvent, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _with(event).validate()


@pytest.mark.parametrize("gen, match", [
    (replace(OUTAGE_GEN, count=1.5), r"shock generator #0 \(producer_outage\): count must be an integer"),
    (replace(OUTAGE_GEN, count=float("nan")), r"count must be an integer"),
    (replace(OUTAGE_GEN, count=True), r"count must be an integer"),
    (replace(OUTAGE_GEN, start_window=(2.5, 5)), r"start window lo must be an integer"),
    (replace(OUTAGE_GEN, start_window=(21, 90.0)), r"start window hi must be an integer"),
    (replace(OUTAGE_GEN, duration=(1.5, 3.5)), r"duration lo must be an integer"),
    (replace(OUTAGE_GEN, duration=(3, float("inf"))), r"duration hi must be an integer"),
    (replace(OUTAGE_GEN, start_window=(21, 50, 90)), r"start window must be a \(lo, hi\) pair"),
    (replace(OUTAGE_GEN, magnitude=(1.0,)), r"magnitude must be a \(lo, hi\) pair"),
    (replace(HIT_GEN, magnitude=(0.3, float("nan"))), r"magnitude hi must be a finite number"),
])
def test_generator_discrete_fields_are_integers(gen: ShockGenerator, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _with(generators=(gen,)).validate()


@pytest.mark.parametrize("patch, match", [
    ({"events": [{"kind": "producer_outage", "target": {"producer": "p2"}, "startCycle": 2.5,
                  "duration": 2, "magnitude": 1.0}]}, r"start cycle must be an integer"),
    ({"events": [{"kind": "producer_outage", "target": {"producer": "p2"}, "startCycle": 3,
                  "duration": 2.0, "magnitude": 1}]}, r"duration must be an integer"),
    ({"generators": [{"kind": "producer_outage", "count": 1.5, "startWindow": [21, 90],
                      "duration": [3, 6], "magnitude": [1, 1]}]}, r"count must be an integer"),
    ({"generators": [{"kind": "producer_outage", "count": 1, "startWindow": [21.5, 90],
                      "duration": [3, 6], "magnitude": [1, 1]}]}, r"start window lo must be an integer"),
    ({"generators": [{"kind": "fund_shock", "count": 1.5, "startWindow": [2.5, 9],
                      "duration": [1, 3], "magnitude": [0.5, 0.5]}]},
     r"unsupported shock kind 'fund_shock'"),
])
def test_json_fractional_fields_rejected(tmp_path: Path, patch: dict, match: str) -> None:
    """Standard JSON fractional values are caught at load; the kind check still comes first."""
    data = json.loads(PHASE1.read_text())
    data["shocks"] = patch
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_valid_boundaries_and_integer_magnitudes_accepted(tmp_path: Path) -> None:
    """Boundary cases stay valid: start cycle 2, a point window, count 0, duration 1, and
    JSON integer outage magnitudes (1 == 1.0)."""
    _with(replace(HIT, start_cycle=2), replace(OUTAGE, start_cycle=2, duration=1),
          generators=(replace(HIT_GEN, count=0, start_window=(2, 2)),
                      replace(OUTAGE_GEN, start_window=(2, 2), duration=(1, 1)))).validate()
    data = json.loads(PHASE1.read_text())
    data["shocks"] = {
        "events": [{"kind": "producer_outage", "target": {"producer": "p2"}, "startCycle": 2,
                    "duration": 1, "magnitude": 1}],
        "generators": [{"kind": "producer_outage", "count": 0, "startWindow": [2, 2],
                        "duration": [1, 1], "magnitude": [1, 1]}],
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(data))
    cfg = load_config(path)
    assert cfg.shocks is not None and cfg.shocks.events[0].magnitude == 1.0


# --- Shock delivery order ---


@pytest.mark.parametrize("event", [
    ShockEvent(ShockKind.FUND_SHOCK, ShockTarget(), start_cycle=5, duration=2, magnitude=0.5),
    ShockEvent(ShockKind.LINK_CUT, ShockTarget(producer="p1", product=10), start_cycle=5,
               duration=2, magnitude=1.0),
    ShockEvent(ShockKind.DEMAND_SURGE, ShockTarget(consumer="c1"), start_cycle=5, duration=2,
               magnitude=3.0),
])
def test_unimplemented_kind_scheduled(event: ShockEvent) -> None:
    """Scenario "Unimplemented kind scheduled": the error names the unsupported kind."""
    with pytest.raises(ValueError, match=rf"unsupported shock kind '{event.kind.value}'"):
        _with(event).validate()


@pytest.mark.parametrize("kind", [ShockKind.LINK_CUT, ShockKind.DEMAND_SURGE, ShockKind.FUND_SHOCK])
def test_unimplemented_kind_in_a_generator(kind: ShockKind) -> None:
    """Scenario "Unimplemented kind in a generator"."""
    gen = ShockGenerator(kind, count=1, start_window=(5, 9), duration=(1, 3), magnitude=(0.5, 0.5))
    with pytest.raises(ValueError, match=rf"unsupported shock kind '{kind.value}'"):
        _with(generators=(gen,)).validate()


def test_unimplemented_kind_rejected_from_json(tmp_path: Path) -> None:
    data = json.loads(PHASE1.read_text())
    data["shocks"] = {"events": [{"kind": "fund_shock", "target": {}, "startCycle": 4,
                                  "duration": 2, "magnitude": 0.5}]}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r"unsupported shock kind 'fund_shock'"):
        load_config(path)


def test_unknown_kind_string_rejected() -> None:
    with pytest.raises(ValueError, match=r"unknown shock kind 'meteor'"):
        shock_model_from_dict({"events": [{"kind": "meteor", "target": {}, "startCycle": 4,
                                           "duration": 1, "magnitude": 0.5}]})


def test_one_generator_per_kind() -> None:
    with pytest.raises(ValueError, match=r"at most one generator per kind"):
        _with(generators=(HIT_GEN, replace(HIT_GEN, count=1))).validate()


@pytest.mark.parametrize("gen, match", [
    (replace(HIT_GEN, duration=(1, 2)), r"capacity hit has duration 1"),
    (replace(HIT_GEN, magnitude=(0.3, 1.0)), r"magnitude must lie in \(0, 1\)"),
    (replace(OUTAGE_GEN, magnitude=(0.5, 1.0)), r"outage magnitude must be 1.0"),
    (replace(OUTAGE_GEN, start_window=(30, 20)), r"start window must be a range lo <= hi"),
    (replace(OUTAGE_GEN, targets=("p3", "zz")), r"producer 'zz' does not exist"),
    (replace(OUTAGE_GEN, targets=("p3", "p3")), r"duplicate target producers"),
    (replace(OUTAGE_GEN, count=-1), r"count must be >= 0"),
])
def test_generator_rules(gen: ShockGenerator, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _with(generators=(gen,)).validate()


# --- JSON round-trip ---


def test_json_round_trip(tmp_path: Path) -> None:
    cfg = _with(HIT, OUTAGE, generators=(HIT_GEN, OUTAGE_GEN))
    cfg.validate()
    d = config_to_dict(cfg)
    assert d["shocks"] == shock_model_to_dict(cfg.shocks)
    path = tmp_path / "shocked.json"
    path.write_text(json.dumps(d))
    loaded = load_config(path)
    assert loaded.shocks == cfg.shocks
    assert config_to_dict(loaded) == d


def test_no_shocks_key_when_unset() -> None:
    """The key is written only when set, so legacy configs serialize unchanged."""
    d = config_to_dict(_phase1())
    assert "shocks" not in d
    assert _phase1().shocks is None


# --- fingerprint (design D5) ---

# Pre-M1 config_fingerprint values, computed at 55539bf^ (the commit before
# add-measurement-gate-harness task 2.2 added `ConsumerConfig.schedule`, whose hashed empty
# default had moved every fingerprint). Families did not exist then: their values are pinned as
# of this change (calm worlds carry a non-empty schedule, so theirs never moved).
BASE_FINGERPRINTS = {
    "config/phase1.json": "7b51b2b62645e2c6",
    "fixtures/competition.json": "7d6e306c8319af5a",
    "fixtures/complex.json": "b11d506933783989",
    "fixtures/composite.json": "467d3d83cf5a1ae6",
    "fixtures/random_demand.json": "82e483017a3ad2ad",
    "scale-0": "cecd9e044397e926",
    "scale-1": "381159e13f081051",
    "scale-2": "0021837a6b94d23b",
    "scale-3": "61b406c5edd3872c",
    "sparse_wide-12-calm": "61bc931beeae007a",
    "sparse_wide-12-shocked": "841dabd71e8912ae",
    "sparse_wide-42-calm": "2208e31edf172540",
    "sparse_wide-42-shocked": "f6b6dc4e6fa741a9",
    "dense-12-calm": "7a5f298beafc47bb",
    "dense-12-shocked": "8ca7d8d36122412c",
    "dense-42-calm": "cf52882c62fc403d",
    "dense-42-shocked": "f29abc2ba0b580c0",
}
# The EXP-0006B champion's stored fingerprint (tests/test_champion_regression.py): enriched v2
# orderer obs on build_scale_config(0, depth=3).
CHAMPION_FINGERPRINT = "548120bd72921234"


def _base_config(name: str) -> Configuration:
    if name.startswith("config/"):
        return load_config(PKG / name)
    if name.startswith("fixtures/"):
        return load_config(FIXTURES / name.split("/", 1)[1])
    if name.startswith("scale-"):
        return build_scale_config(int(name.split("-")[1]))
    family, producers, regime = name.split("-")
    return build_family_config(1000, family=family, producers=int(producers), regime=regime)


def test_packaged_config_set_is_covered() -> None:
    """Every packaged/fixture JSON config is pinned, so a new one cannot slip past."""
    pinned = {n for n in BASE_FINGERPRINTS if n.startswith(("config/", "fixtures/"))}
    found = {f"config/{p.name}" for p in (PKG / "config").glob("*.json")}
    found |= {f"fixtures/{p.name}" for p in FIXTURES.glob("*.json")}
    assert found == pinned


@pytest.mark.parametrize("name", sorted(BASE_FINGERPRINTS))
def test_fingerprint_unchanged_without_shocks(name: str) -> None:
    assert config_fingerprint(_base_config(name)) == BASE_FINGERPRINTS[name]


def test_champion_fingerprint_restored() -> None:
    cfg = build_scale_config(0, depth=3)
    assert config_fingerprint(cfg, orderer_obs_version=2) == CHAMPION_FINGERPRINT


def test_non_empty_schedule_still_changes_the_fingerprint() -> None:
    plain = _phase1()
    c1 = plain.consumers[0]
    scheduled = replace(plain, consumers=[replace(c1, schedule=((c1.product, 1),))])
    other = replace(plain, consumers=[replace(c1, schedule=((c1.product, 2),))])
    fps = {config_fingerprint(plain), config_fingerprint(scheduled), config_fingerprint(other)}
    assert len(fps) == 3


def test_shocked_config_has_its_own_fingerprint() -> None:
    plain = _phase1()
    shocked = _with(HIT)
    assert config_fingerprint(shocked) != config_fingerprint(plain)
    assert config_fingerprint(_with(HIT)) == config_fingerprint(shocked)
