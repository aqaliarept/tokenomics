"""Registry v3 (add-gate-v3 task 2.2, design D2, D3 step 4; ADR-0060 §2; gate-harness "World
registry", "Scale families", "Development headroom check", "Registry v3 keeps v2 entries",
"Region families in registry v3", "Region regimes", "Region entries carry their world record",
"Region worlds are fresh draws disjoint from atlas and M4 worlds")."""

from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tokenomics.eval.families import CALM, SHOCKED, SPARSE_WIDE
from tokenomics.eval.gate import Headroom
from tokenomics.eval.registry import (
    DEVELOPMENT, EXCLUDED, GATE, M4_TRAIN_STREAM, REGION_CHOICES, REGION_FAMILIES,
    REGION_WORLD_COUNTS, REGISTRY_V1, REGISTRY_V2, RESERVE, RESERVES_PER_GROUP, RHO_BAND, SEALED,
    WS_LOADED, WS_WIDE, Exclusion, Registry, ScreeningRecord, WorldEntry, build_worlds_v2,
    draft_worlds_v3, load_registry, record_exclusions, record_headroom, region_entries,
    region_reserved_seeds, registry_from_dict, save_registry, smoke_variant)
from tokenomics.eval.roster import content_hash
from tokenomics.eval.routing_audit import config_digest
from tokenomics.eval.shock_profiles import (SHOCKS_I, SHOCKS_I_SMOKE, SMOKE_START_WINDOW,
                                            apply_shock_profile, load_shock_profile,
                                            names_test_only_profile, smoke_profile)

REPO = Path(__file__).resolve().parents[2]
DRAFTS = REPO / "openspec" / "changes" / "add-gate-v3" / "drafts"
PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def drafted() -> tuple[Registry, list[dict]]:
    return draft_worlds_v3()


@pytest.fixture(scope="module")
def v3(drafted: tuple[Registry, list[dict]]) -> Registry:
    return drafted[0]


def _region(reg: Registry) -> list[WorldEntry]:
    return [e for e in reg.entries if e.family in REGION_FAMILIES]


def _kinds_key(gens: object) -> list:
    return sorted(gens, key=lambda g: g.kind.value)  # type: ignore[attr-defined]


# --- contents ---


def test_v2_entries_carried_over(v3: Registry) -> None:
    """Every frozen v2 entry is in v3 with the same id, seeds, generator ref, status and reserve
    position, and replays the same configuration (gate-harness "Registry v3 keeps v2 entries")."""
    frozen = load_registry(REGISTRY_V2)
    for e in frozen.entries:
        got = v3.entry(e.id)
        assert got.to_dict() == e.to_dict()
        assert config_digest(got.config()) == config_digest(e.config())
    legacy = [e for e in v3.entries if e.family not in REGION_FAMILIES]
    assert len(legacy) == len(frozen.entries)
    assert v3.shock_profiles == frozen.shock_profiles and v3.name == "worlds-v3"


def test_region_group_sizes(v3: Registry) -> None:
    """D2 counts per split and family: calm 4 x 3 / shocked 2 x 10 development, 8 / 4 sealed, 4
    reserves per regime; ws_loaded at 12 alternates dense / sparse_f4 by slot, ws_wide is f14 at
    42."""
    for split in (DEVELOPMENT, SEALED):
        for family in REGION_FAMILIES:
            for regime in (CALM, SHOCKED):
                group = sorted((e for e in _region(v3) if (e.split, e.family, e.regime)
                                == (split, family, regime)),
                               key=lambda e: (e.status != GATE, e.id))
                worlds, episodes = REGION_WORLD_COUNTS[split][regime]
                gate = [e for e in group if e.status == GATE]
                res = sorted((e for e in group if e.status == RESERVE),
                             key=lambda e: e.reserve_position)
                assert len(gate) == worlds and len(res) == RESERVES_PER_GROUP
                assert [e.reserve_position for e in res] == list(range(RESERVES_PER_GROUP))
                assert all(len(e.episode_seeds) == episodes for e in group)
                slots = gate + res
                for s, e in enumerate(slots):
                    choice = e.generator["params"]["levers"]["choice"]
                    want = REGION_CHOICES[family][s % len(REGION_CHOICES[family])]
                    assert choice == want, e.id
                    assert e.size == (12 if family == WS_LOADED else 42)
    dev_wide = [e for e in v3.gate_worlds(DEVELOPMENT) if e.family == WS_WIDE]
    assert sorted((e.regime, len(e.episode_seeds)) for e in dev_wide) == (
        [(CALM, 3)] * 4 + [(SHOCKED, 10)] * 2)


def test_region_regimes(v3: Registry) -> None:
    """Calm region worlds: cyclic deterministic demand, no shocks, no drift. Shocked: the
    registered shocks-i generators and its drift exactly, named in the generator ref."""
    profile = load_shock_profile(SHOCKS_I)
    for e in _region(v3):
        cfg = e.config()
        lv = e.generator["params"]["levers"]
        if e.regime == CALM:
            assert cfg.shocks is None and cfg.demand_drift is None
            assert cfg.demand_pattern is None and lv["pattern"] == "cyclic"
            assert all(c.schedule for c in cfg.consumers)
            assert "shock_profile" not in e.generator["params"]
        else:
            want = apply_shock_profile(cfg, profile, e.size)
            assert _kinds_key(cfg.shocks.generators) == _kinds_key(want.shocks.generators)
            assert cfg.demand_drift == want.demand_drift is not None
            assert e.generator["params"]["shock_profile"] == SHOCKS_I


def test_region_draws_in_index_order(drafted: tuple[Registry, list[dict]]) -> None:
    """Each (split, family) pool is used in index order, each index once; accepted draws have
    rho in (0.28, 3.1] and became the slot's world; rejected ones are recorded with a rho outside
    the band; the entry's record and draw agree."""
    reg, rows = drafted
    lo, hi = RHO_BAND
    by_pool: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_pool.setdefault((r["split"], r["family"]), []).append(r)
    assert set(by_pool) == {(s, f) for s in (DEVELOPMENT, SEALED) for f in REGION_FAMILIES}
    rejected = 0
    for (split, family), rs in by_pool.items():
        assert [r["draw_index"] for r in rs] == list(range(len(rs)))
        for r in rs:
            assert r["accepted"] == (lo < r["rho"] <= hi)
            rejected += not r["accepted"]
            if r["accepted"]:
                e = reg.entry(r["world"])
                rec = e.region["record"]
                assert (rec["index"], rec["seed"], e.world_seed) == (
                    r["draw_index"], r["seed"], r["seed"])
                assert rec["coordinates"]["rho"] == r["rho"]
        assert sum(r["accepted"] for r in rs) == len(
            [e for e in _region(reg) if (e.split, e.family) == (split, family)])
    assert rejected > 0


def test_pool_exhausted_is_refused() -> None:
    with pytest.raises(ValueError, match="exhausted"):
        region_entries(DEVELOPMENT, WS_LOADED, pool=10)


def test_hashed_region_seeds_load(v3: Registry, tmp_path: Path) -> None:
    """Region seeds lie outside the legacy ranges and load; the saved draft reloads with its hash
    and replays."""
    assert all(not 1000 <= e.world_seed <= 5999 for e in _region(v3))
    path = tmp_path / "worlds_v3_draft.json"
    save_registry(v3, path)
    assert load_registry(path).version == v3.version


# --- seeds ---


def test_m4_and_atlas_names_pinned() -> None:
    """The reserved set uses M4's real stream (`order_es_worlds.FRESH_ATLAS`, read without
    importing the training module) and the atlas's constants."""
    from tokenomics.eval.atlas import ATLAS, SCREEN_WORLDS

    src = PACKAGE / "tokenomics" / "train" / "order_es_worlds.py"
    names = {t.id: n.value for n in ast.parse(src.read_text()).body if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Name)}
    assert ast.literal_eval(names["FRESH_ATLAS"]) == M4_TRAIN_STREAM == "m4-es-train-v1"
    assert ATLAS == "atlas-v1" and SCREEN_WORLDS == {12: 512, 42: 128}


def test_region_seeds_disjoint_from_every_reserved_source(v3: Registry) -> None:
    """No region world or episode seed is a registry v1 / v2, atlas-v1 sample or anchor, or first
    4,096 m4-es-train-v1 seed, and the reserved set holds every source."""
    from tokenomics.eval.routing_audit import fanout_worlds
    from tokenomics.eval.world_space import episode_seeds, world_id, world_seed

    reserved = region_reserved_seeds()
    for path in (REGISTRY_V1, REGISTRY_V2):
        for e in load_registry(path).entries:
            assert e.world_seed in reserved and set(e.episode_seeds) <= set(reserved)
    for size, n in ((12, 512), (42, 128)):
        for i in range(n):
            wid = world_id("atlas-v1", size, i)
            assert world_seed("atlas-v1", size, i) in reserved
            assert set(episode_seeds(wid, 1) + episode_seeds(wid, 2, "confirm")) <= set(reserved)
    for size in (12, 42):
        assert {world_seed(M4_TRAIN_STREAM, size, i) for i in range(4096)} <= set(reserved)
    for e in fanout_worlds():  # EXP-0050's grid, the anchors' source
        assert e.world_seed in reserved and set(e.episode_seeds) <= set(reserved)
    for e in _region(v3):
        assert e.world_seed not in reserved and not set(e.episode_seeds) & set(reserved)
    dev = {e.world_seed for e in _region(v3) if e.split == DEVELOPMENT}
    assert dev and not dev & {e.world_seed for e in _region(v3) if e.split == SEALED}


def test_atlas_world_offered_is_refused(v3: Registry) -> None:
    """An entry with an atlas world's seed (atlas v2 train world 12-0111) is refused naming the
    entry and the atlas world."""
    from tokenomics.eval.world_space import world_seed

    e = next(e for e in _region(v3) if e.family == WS_LOADED)
    bad = replace(e, world_seed=world_seed("atlas-v1", 12, 111))
    entries = tuple(bad if x.id == e.id else x for x in v3.entries)
    with pytest.raises(ValueError, match=rf"{e.id}.*ws-atlas-v1-12-0111"):
        replace(v3, entries=entries)


def test_colliding_stream_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A draft whose development stream is atlas-v1's is refused naming the atlas world; one whose
    splits share a stream is refused naming a development and a sealed entry."""
    import tokenomics.eval.world_space as ws

    real = ws.gate_v3_stream
    monkeypatch.setattr(ws, "gate_v3_stream",
                        lambda split: "atlas-v1" if split == DEVELOPMENT else real(split))
    with pytest.raises(ValueError, match=r"reserved by atlas world ws-atlas-v1-12-"):
        draft_worlds_v3()
    monkeypatch.setattr(ws, "gate_v3_stream", lambda split: real(DEVELOPMENT))
    with pytest.raises(ValueError, match=r"under both splits: dev-.* \(development\) and sea-"):
        draft_worlds_v3()


# --- validation ---


def test_record_mismatch_refused_on_load(v3: Registry, tmp_path: Path) -> None:
    """A region entry whose replayed config hash differs from its record is refused naming it."""
    d = json.loads(v3.to_json())
    d.pop("content_hash")
    target = next(x for x in d["entries"] if x["family"] == WS_WIDE)
    target["region"]["record"]["config_sha256"] = "0" * 64
    d["content_hash"] = content_hash(d)
    with pytest.raises(ValueError, match=rf"{target['id']}: replayed config sha256"):
        registry_from_dict(d)


def _swap(reg: Registry, old: WorldEntry, new: WorldEntry) -> tuple[WorldEntry, ...]:
    return tuple(new if x.id == old.id else x for x in reg.entries)


@pytest.mark.parametrize("edit, match", [
    (lambda e: {**e.generator["params"]["levers"], "k_c": 1.0}, "not draw"),
    (lambda e: {**e.generator["params"]["levers"], "choice": "sparse_f2"}, "choice is one of"),
    (lambda e: {**e.generator["params"]["levers"], "pattern": "iid"}, "calm region world"),
])
def test_tampered_region_entry_refused(v3: Registry, edit: object, match: str) -> None:
    e = next(e for e in _region(v3) if e.family == WS_LOADED and e.regime == CALM)
    g = {"generator": e.generator["generator"],
         "params": {**e.generator["params"], "levers": edit(e)}}  # type: ignore[operator]
    rec = {**e.region["record"], "levers": g["params"]["levers"]}
    bad = replace(e, generator=g, region={**e.region, "record": rec})
    with pytest.raises(ValueError, match=match):
        replace(v3, entries=_swap(v3, e, bad))


def test_region_family_size_refused(v3: Registry) -> None:
    e = next(e for e in _region(v3) if e.family == WS_WIDE)
    with pytest.raises(ValueError, match=rf"{e.id}: a ws_wide world has 42 producers"):
        replace(v3, entries=_swap(v3, e, replace(e, size=12)))


def test_region_family_refused_before_v3(v3: Registry) -> None:
    """A v2 (or v1) registry holding a region entry is refused."""
    v2 = build_worlds_v2()
    e = _region(v3)[0]
    for name in ("worlds-v2", "worlds-v1"):
        with pytest.raises(ValueError, match="from registry v3 on"):
            replace(v2, name=name, entries=v2.entries + (e,))


def test_dense_refused_in_v3(v3: Registry) -> None:
    from tokenomics.eval.registry import build_worlds_v1

    dense = next(e for e in build_worlds_v1(("dense", "sparse_wide")).entries
                 if e.family == "dense")
    with pytest.raises(ValueError, match="legacy dense family"):
        replace(v3, entries=v3.entries + (dense,))


def test_shocked_region_entry_must_name_a_profile(v3: Registry) -> None:
    e = next(e for e in _region(v3) if e.regime == SHOCKED)
    params = {k: v for k, v in e.generator["params"].items() if k != "shock_profile"}
    with pytest.raises(ValueError, match="region generator params"):
        replace(v3, entries=_swap(v3, e, replace(e, generator={**e.generator,
                                                                "params": params})))


# --- smoke form (carry-forward from the 2.1 review, supervisor-approved) ---


def test_smoke_variant_replays_and_is_test_only(v3: Registry) -> None:
    """A shocked region entry's smoke form names shocks-i-smoke with the smoke window; a
    registry holding it validates and replays; its generators equal `smoke_profile(shocks-i)`'s
    with shocks-i's drift; the test-only guard flags it for a real run."""
    from tokenomics.eval.gate_cli import test_only_problem

    e = next(e for e in _region(v3) if e.regime == SHOCKED and e.family == WS_WIDE)
    sm = smoke_variant(e)
    from tokenomics.eval.world_space import point_from_record, world_record

    full = world_record(replace(point_from_record(e.region["record"]),
                                shock_window=tuple(SMOKE_START_WINDOW)))
    assert sm.region["record"] == full  # the cheap record equals a full re-measure
    assert sm.generator["params"]["shock_profile"] == SHOCKS_I_SMOKE
    assert sm.generator["params"]["shock_window"] == list(SMOKE_START_WINDOW)
    reg = replace(v3, entries=_swap(v3, e, sm))
    reg.check_replay()
    cfg = reg.entry(e.id).config()
    want = apply_shock_profile(cfg, smoke_profile(load_shock_profile(SHOCKS_I)), e.size)
    assert _kinds_key(cfg.shocks.generators) == _kinds_key(want.shocks.generators)
    assert all(tuple(g.start_window) == tuple(SMOKE_START_WINDOW) for g in cfg.shocks.generators)
    assert cfg.demand_drift is not None and cfg.demand_drift == apply_shock_profile(
        cfg, load_shock_profile(SHOCKS_I), e.size).demand_drift
    assert names_test_only_profile(sm.generator)
    problem = test_only_problem([reg.entry(e.id)])
    assert problem is not None and e.id in problem and "--dry-run" in problem
    assert test_only_problem(v3.entries) is None
    with pytest.raises(ValueError, match="only a shocked region entry"):
        smoke_variant(next(x for x in _region(v3) if x.regime == CALM))


@pytest.mark.parametrize("name, window, match", [
    (SHOCKS_I_SMOKE, [21, 90], "needs shock_window"),   # smoke name, real window
    (SHOCKS_I, list(SMOKE_START_WINDOW), "needs shock_window"),  # real name, smoke window
    ("shocks-ii", [21, 90], "shocks-i' only"),
    ("shocks-ii", list(SMOKE_START_WINDOW), "shocks-i' only"),
])
def test_smoke_name_and_window_must_agree(v3: Registry, name: str, window: list,
                                          match: str) -> None:
    from tokenomics.eval.task_archive import REPRODUCERS

    e = next(e for e in _region(v3) if e.regime == SHOCKED)
    params = {**e.generator["params"], "shock_profile": name, "shock_window": window}
    with pytest.raises(ValueError, match=match):
        REPRODUCERS["world_space"](e.world_seed, **params)
    g = {"generator": e.generator["generator"], "params": params}
    with pytest.raises(ValueError, match=match):
        replace(v3, entries=_swap(v3, e, replace(e, generator=g)))


# --- exclusions (D3 step 4) ---


def _screened(reg: Registry, failed: tuple[str, int] | None) -> Registry:
    """A synthetic screened v3 set: both splits screened, headroom recorded per family x size,
    `failed` above the ceiling."""
    record = ScreeningRecord(roster_version="r" * 64, baseline_means=(), replacements=(),
                             splits=(DEVELOPMENT, SEALED))
    screened = replace(reg, screening=record, screened_on=reg.version)
    heads = [Headroom(f, z, 4, 0.96 if (f, z) == failed else 0.5, 0.95, (f, z) != failed)
             for f, z in ((SPARSE_WIDE, 12), (SPARSE_WIDE, 42), (WS_LOADED, 12), (WS_WIDE, 42))]
    return record_headroom(screened, heads)


def test_recorded_exclusion_passes_admission(v3: Registry, tmp_path: Path) -> None:
    """sparse_wide/12 failed and excluded: every one of its entries (both splits, gate and
    reserve) is `excluded`, development and sealed admission and coverage pass, and the set
    saves and reloads."""
    reg = record_exclusions(_screened(v3, (SPARSE_WIDE, 12)))
    sw12 = [e for e in reg.entries if (e.family, e.size) == (SPARSE_WIDE, 12)]
    assert sw12 and all(e.status == EXCLUDED for e in sw12)
    assert {e.split for e in sw12} == {DEVELOPMENT, SEALED}
    assert all(e.status != EXCLUDED for e in reg.entries if (e.family, e.size) != (SPARSE_WIDE, 12))
    assert reg.screening.exclusions == (Exclusion(SPARSE_WIDE, 12, next(
        h for h in reg.screening.headroom if not h.passed)),)
    assert reg.coverage_problems() == []
    assert reg.admission_problems(DEVELOPMENT) == [] and reg.admission_problems(SEALED) == []
    assert not any(e.family == SPARSE_WIDE and e.size == 12 for e in reg.gate_worlds())
    path = tmp_path / "frozen_v3.json"
    save_registry(reg, path)
    assert load_registry(path).version == reg.version


def test_failed_headroom_without_exclusion_refused(v3: Registry) -> None:
    reg = _screened(v3, (SPARSE_WIDE, 12))
    assert reg.admission_problems(DEVELOPMENT) == [
        "headroom failed for sparse_wide/12: median fill 0.96 >= ceiling 0.95"]
    assert reg.admission_problems(SEALED) == reg.admission_problems(DEVELOPMENT)


def test_inconsistent_exclusions_refused(v3: Registry) -> None:
    reg = record_exclusions(_screened(v3, (SPARSE_WIDE, 12)))
    one = next(e for e in reg.entries if e.status == EXCLUDED)
    with pytest.raises(ValueError, match="disagrees with the exclusions"):
        replace(reg, entries=_swap(reg, one, replace(one, status=GATE)))
    passed = next(h for h in reg.screening.headroom if h.passed)
    bad = replace(reg.screening, exclusions=(Exclusion(passed.family, passed.size, passed),))
    with pytest.raises(ValueError, match="does not match a failed headroom"):
        replace(reg, screening=bad)
    with pytest.raises(ValueError, match="records no exclusion"):
        replace(reg, screening=replace(reg.screening, exclusions=None))
    with pytest.raises(ValueError, match="exist from v3 on"):
        record_exclusions(replace(load_registry(REGISTRY_V2)))


def test_exclusions_refused_before_v3() -> None:
    """An `excluded` entry or a recorded exclusion in a v1 / v2 registry is refused."""
    v2 = load_registry(REGISTRY_V2)
    e = v2.entries[0]
    with pytest.raises(ValueError, match="from registry v3 on"):
        replace(v2, entries=_swap(v2, e, replace(e, status=EXCLUDED)))
    h = v2.screening.headroom[0]
    failed = replace(h, median_fill=0.96, passed=False)
    sc = replace(v2.screening, headroom=(failed,) + v2.screening.headroom[1:],
                 exclusions=(Exclusion(h.family, h.size, failed),))
    with pytest.raises(ValueError, match="exist from v3 on"):
        replace(v2, screening=sc)
    v1 = load_registry(REGISTRY_V1)
    with pytest.raises(ValueError, match="from registry v3 on"):
        replace(v1, entries=_swap(v1, v1.entries[0], replace(v1.entries[0], status=EXCLUDED)))


# --- v1 / v2 admission and coverage unchanged ---

# Output at base commit a217883, from the same variants (/tmp/gv3_22_variants.py run there).
BASE_OUTPUT = {
    "v1": {
        "admission": {
            "development": [],
            "sealed": [
                "registry 566946fe0c03 has not screened the sealed split"
            ]
        },
        "coverage": []
    },
    "v2": {
        "admission": {
            "development": [],
            "sealed": [
                "registry 7b09f536e988 has not screened the sealed split"
            ]
        },
        "coverage": []
    },
    "v2-both-splits": {
        "admission": {
            "development": [],
            "sealed": []
        },
        "coverage": []
    },
    "v2-draft": {
        "admission": {
            "development": [
                "registry 3032147f1f17 has not screened the development split"
            ],
            "sealed": [
                "registry 3032147f1f17 has not screened the sealed split"
            ]
        },
        "coverage": []
    },
    "v2-failed-42": {
        "admission": {
            "development": [
                "headroom failed for sparse_wide/42: median fill 0.96 >= ceiling 0.95"
            ],
            "sealed": [
                "registry 1d0b531d909f has not screened the sealed split"
            ]
        },
        "coverage": []
    },
    "v2-no-dev-42": {
        "admission": {
            "development": [
                "no development gate worlds for sparse_wide/42"
            ],
            "sealed": [
                "registry a1cd59ceb1df has not screened the sealed split"
            ]
        },
        "coverage": [
            "no development gate worlds for sparse_wide/42"
        ]
    },
    "v2-no-headroom": {
        "admission": {
            "development": [
                "registry 7cd364e9bd68 records no development headroom check"
            ],
            "sealed": [
                "registry 7cd364e9bd68 has not screened the sealed split"
            ]
        },
        "coverage": []
    }
}


def _variants() -> list[tuple[str, Registry]]:
    v1, v2 = load_registry(REGISTRY_V1), load_registry(REGISTRY_V2)
    sc = v2.screening
    h = tuple(replace(x, median_fill=0.96, passed=False) if x.size == 42 else x
              for x in sc.headroom)
    return [
        ("v1", v1), ("v2", v2),
        ("v2-both-splits", replace(v2, screening=replace(sc, splits=(DEVELOPMENT, SEALED)))),
        ("v2-failed-42", replace(v2, screening=replace(sc, headroom=h))),
        ("v2-no-headroom", replace(v2, screening=replace(sc, headroom=None))),
        ("v2-no-dev-42", replace(v2, entries=tuple(
            e for e in v2.entries if not (e.split == DEVELOPMENT and e.size == 42)))),
        ("v2-draft", build_worlds_v2()),
    ]


def test_v1_v2_admission_and_coverage_unchanged() -> None:
    got = {name: {"coverage": r.coverage_problems(),
                  "admission": {s: r.admission_problems(s) for s in (DEVELOPMENT, SEALED)}}
           for name, r in _variants()}
    assert got == BASE_OUTPUT


def test_frozen_v1_v2_versions_unchanged_and_default_stays_v2() -> None:
    """gate-harness "Packaging a gate version does not move driver defaults"."""
    from tokenomics.eval.registry import DEFAULT_REGISTRY

    assert load_registry(REGISTRY_V1).version.startswith("566946fe0c03")
    assert load_registry(REGISTRY_V2).version.startswith("7b09f536e988")
    assert DEFAULT_REGISTRY == REGISTRY_V2


# --- draft output ---


def test_write_draft_cli(tmp_path: Path) -> None:
    """`--write-draft --draft-version 3` writes the draft and acceptance.csv beside it, and refuses
    the packaged registry directory."""
    out = tmp_path / "worlds_v3_draft.json"
    cmd = [sys.executable, "-m", "tokenomics.eval.registry", "--write-draft", "--draft-version",
           "3", "--out"]
    done = subprocess.run(cmd + [str(out)], capture_output=True, text=True, cwd=PACKAGE)
    assert done.returncode == 0, done.stderr
    reg = load_registry(out)
    assert reg.version == draft_worlds_v3()[0].version
    with (tmp_path / "acceptance.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows and set(rows[0]) >= {"draw_index", "slot", "rho", "accepted"}
    bad = subprocess.run(cmd + [str(REGISTRY_V2.parent / "worlds_v3_draft.json")],
                         capture_output=True, text=True, cwd=PACKAGE)
    assert bad.returncode != 0 and "Drafts path" in bad.stderr
    assert not (REGISTRY_V2.parent / "worlds_v3_draft.json").exists()


def test_drafts_on_disk_are_the_builder(v3: Registry) -> None:
    if not DRAFTS.exists():  # the openspec tree is not shipped to pods
        pytest.skip("Drafts dir not present")
    assert (DRAFTS / "worlds_v3_draft.json").read_text() == v3.to_json()
    assert load_registry(DRAFTS / "worlds_v3_draft.json").version == v3.version
