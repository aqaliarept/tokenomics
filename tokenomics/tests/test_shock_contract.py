"""M2 exit driver `eval/shock_contract.py` (add-shocks-i-drift task 6.1, design D11, D7, D10;
project-mission "Champion runs on a shocked ruleset unretrained").

The smoke loads both stacks (portable: EXP-0035 producer + EXP-0012 orderer; flat: the
`champion.json` checkpoints) and plays one shocked episode and its shock-free twin per stack at
horizon 12 under the test-only `shocks-i-smoke` profile. It skips cleanly when a git-lfs
artifact is only a pointer. The refusals and the checkpoint failure rows need no artifact.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Callable

import pytest

pytest.importorskip("torch")

from tokenomics.eval.contract import CHECKPOINT, FAILURE_COLUMNS, RUN, read_failures
from tokenomics.eval.gate import CANDIDATE, TWIN, rows_from_csv
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.env.shock_schedule import build_schedule
from tokenomics.eval import shock_contract
from tokenomics.eval.shock_contract import (
    DEFAULT_CHAMPION,
    FLAT,
    PORTABLE,
    expected_keys,
    flat_grid,
    load_champion,
    STACKS,
    _config,
    _policies,
    build_jobs,
    flat_grid_problem,
    main,
    portable_worlds,
    real_run_problems,
)
from tokenomics.eval.shock_profiles import SHOCKS_I, SHOCKS_I_SMOKE

REPO = Path(__file__).resolve().parents[2]
PORTABLE_PROD = (REPO / "experiments" / "EXP-0035-distill-epochs" / "replay" / "artifacts"
                 / "producer_candidate.pt")
PORTABLE_ORDERER = (REPO / "experiments" / "EXP-0012-portable-orderer-scorer" / "artifacts"
                    / "orderer_scorer.pt")
HORIZON = 12
COMMIT = "0123456789abcdef0123456789abcdef01234567"
PORTABLE_WORLD = "dev-sparse_wide-12-shocked-00"
FLAT_WORLD = "flat-scale-d3-s6"


def _lfs_pulled(path: Path) -> bool:
    """True only for real content: an un-pulled git-lfs file is a small text pointer."""
    if not path.exists():
        return False
    with path.open("rb") as f:
        return not f.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")


def _champion_ckpts() -> list[Path]:
    champ = load_champion(DEFAULT_CHAMPION)
    return [Path(champ["checkpoint"]), Path(champ["producer_checkpoint"])]


ARTIFACTS = [PORTABLE_PROD, PORTABLE_ORDERER, *_champion_ckpts()]
requires_artifacts = pytest.mark.skipif(
    not all(_lfs_pulled(p) for p in ARTIFACTS),
    reason="a portable or champion.json checkpoint is absent (git-lfs not pulled)")


def _smoke_argv(out: Path, *extra: str) -> list[str]:
    return ["--out-dir", str(out), "--prod-ckpt", str(PORTABLE_PROD),
            "--orderer-ckpt", str(PORTABLE_ORDERER), "--shock-profile", SHOCKS_I_SMOKE,
            "--horizon", str(HORIZON), "--sizes", "12", "--worlds-per-size", "1",
            "--episodes-per-world", "1", *extra]


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not all(_lfs_pulled(p) for p in ARTIFACTS):
        pytest.skip("a portable or champion.json checkpoint is absent (git-lfs not pulled)")
    out = tmp_path_factory.mktemp("EXP-shock-contract")
    # one process: a torch import per spawned worker thrashes a small box (no swap)
    assert main(_smoke_argv(out, "--dry-run")) == 0
    return out


def _rows(out: Path) -> list:
    return rows_from_csv((out / "episodes.csv").read_text())


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# --- the smoke: both stacks, one shocked episode + twin each ---


@requires_artifacts
def test_smoke_writes_the_exp_layout_and_the_contract_holds(smoke_dir: Path) -> None:
    for name in ("spec.json", "episodes.csv", "resilience.csv", "summary.csv",
                 "contract_failures.csv", "RESULT.md"):
        assert (smoke_dir / name).is_file(), name
    assert read_failures(smoke_dir) == []
    assert (smoke_dir / "contract_failures.csv").read_text().strip() == ",".join(FAILURE_COLUMNS)
    spec = json.loads((smoke_dir / "spec.json").read_text())
    assert spec["driver"] == "tokenomics.eval.shock_contract"
    assert spec["kind"] == "dry-run" and spec["status"] == "complete"
    assert spec["result"]["exit_holds"] is True
    assert spec["result"]["acceptance_read"] is False
    assert spec["result"]["contract_failures"] == 0
    assert spec["result"]["episodes"] == 2 and spec["result"]["twins"] == 2
    pins = spec["pins"]
    assert pins["stacks"] == [PORTABLE, FLAT]
    assert pins["shock_profile"] == SHOCKS_I_SMOKE and pins["horizon"] == HORIZON
    assert set(pins["grid"]) == {PORTABLE_WORLD, FLAT_WORLD}
    assert set(pins["checkpoints"]) == {"portable_producer", "portable_orderer", "champion",
                                        "flat_producer", "flat_orderer"}
    assert "DRY RUN — not an acceptance read" in (smoke_dir / "RESULT.md").read_text()
    keys = {(r.world, r.subject) for r in _rows(smoke_dir)}
    assert keys == {(w, s) for w in (PORTABLE_WORLD, FLAT_WORLD) for s in (CANDIDATE, TWIN)}
    summary = {d["stack"]: d for d in _csv(smoke_dir / "summary.csv")}
    assert set(summary) == {PORTABLE, FLAT}
    assert all(d["episodes"] == "1" and d["twins"] == "1" for d in summary.values())


@requires_artifacts
@pytest.mark.parametrize("world", [PORTABLE_WORLD, FLAT_WORLD])
def test_smoke_schedule_fires_both_kinds_inside_the_played_cycles(smoke_dir: Path,
                                                                  world: str) -> None:
    """D7 smoke assertions on each stack's shocked episode: a non-empty schedule holding both
    kinds, a capacity-hit onset and an outage cycle inside the played cycles; the twin has no
    shock and the same demand."""
    rows = {r.subject: r for r in _rows(smoke_dir) if r.world == world}
    cand, twin = rows[CANDIDATE], rows[TWIN]
    assert cand.shock_schedule is not None and cand.fulfilled_trace is not None
    schedule = json.loads(cand.shock_schedule)
    played = len(cand.fulfilled_trace)
    assert schedule and cand.shock_events == len(schedule)
    assert {e["kind"] for e in schedule} == {"capacity_hit", "producer_outage"}
    assert any(e["kind"] == "capacity_hit" and 1 <= e["startCycle"] <= played for e in schedule)
    assert any(e["kind"] == "producer_outage" and e["startCycle"] <= played
               and e["startCycle"] + e["duration"] - 1 >= 1 for e in schedule)
    assert twin.shock_events == 0 and json.loads(twin.shock_schedule) == []
    assert twin.demand_digest == cand.demand_digest


@requires_artifacts
@pytest.mark.parametrize("world", [PORTABLE_WORLD, FLAT_WORLD])
def test_smoke_resilience_has_a_row_per_shock(smoke_dir: Path, world: str) -> None:
    cand = next(r for r in _rows(smoke_dir) if r.world == world and r.subject == CANDIDATE)
    schedule = json.loads(cand.shock_schedule)
    res = [d for d in _csv(smoke_dir / "resilience.csv") if d["world"] == world]
    assert [(d["kind"], d["producer"], int(d["start"])) for d in res] == [
        (e["kind"], e["target"]["producer"], e["startCycle"]) for e in schedule]


@requires_artifacts
def test_rerun_on_a_complete_dir_resumes_without_running_jobs(smoke_dir: Path,
                                                              tmp_path: Path) -> None:
    out = tmp_path / "EXP-resume"
    shutil.copytree(smoke_dir, out)
    before = (out / "episodes.csv").read_text()
    assert main(_smoke_argv(out, "--dry-run")) == 0
    assert (out / "episodes.csv").read_text() == before
    assert len(json.loads((out / "spec.json").read_text())["resumes"]) == 1


# --- worker caches serve one identity each (a process may serve several runs) ---


def _flat_job(profile: str, profiles_version: str = "profiles-hash") -> dict:
    grid = flat_grid(load_champion(DEFAULT_CHAMPION), profile, sizes=None, worlds_per_size=1,
                     episodes_per_world=1)
    return build_jobs(grid, {}, HORIZON, profile, {FLAT: {}}, profiles_version)[0]


def test_config_cache_is_keyed_by_the_build_not_the_world_id(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shock_contract, "_CONFIGS", {})
    smoke, real = _flat_job(SHOCKS_I_SMOKE), _flat_job(SHOCKS_I)
    assert (smoke["world"], smoke["episode"]) == (real["world"], real["episode"])
    a = _config(smoke)
    assert _config(dict(smoke)) is a
    assert build_schedule(a, smoke["episode"], HORIZON)
    assert build_schedule(_config(real), real["episode"], HORIZON) == ()  # onsets from cycle 21
    assert _config(_flat_job(SHOCKS_I_SMOKE, "other-profiles-hash")) is not a


def test_policy_cache_is_keyed_by_checkpoint_paths_and_contents(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake(ck: dict) -> tuple[object, object]:
        calls.append(dict(ck))
        return object(), object()

    monkeypatch.setattr(shock_contract, "_POLICIES", {})
    monkeypatch.setattr(shock_contract, "_portable_policies", fake)

    def job(path: str, sha: str) -> dict:
        return {"stack": PORTABLE,
                "checkpoints": {"producer": path, "orderer": "o.pt", "sha256": [sha, "o"]}}

    first = _policies(job("first.pt", "a"))
    assert _policies(job("first.pt", "a")) is first
    second = _policies(job("second.pt", "b"))
    changed = _policies(job("first.pt", "c"))  # same path, new contents
    assert len({id(first), id(second), id(changed)}) == 3
    assert [c["producer"] for c in calls] == ["first.pt", "second.pt", "first.pt"]


@requires_artifacts
def test_a_second_run_in_the_same_process_uses_its_own_profile(smoke_dir: Path,
                                                               tmp_path: Path) -> None:
    """After the smoke-profile run in this process, a run on the registered profile at horizon
    12 plays no shock (its onsets start at cycle 21): no cached smoke world is reused."""
    out = tmp_path / "EXP-second"
    assert main(["--out-dir", str(out), "--dry-run", "--stacks", FLAT, "--horizon", str(HORIZON),
                 "--worlds-per-size", "1", "--episodes-per-world", "1"]) == 0
    rows = _rows(out)
    assert {r.world for r in rows} == {FLAT_WORLD}
    assert all(r.shock_events == 0 for r in rows)
    smoke_flat = [r for r in _rows(smoke_dir) if r.world == FLAT_WORLD]
    assert {r.episode for r in rows} == {r.episode for r in smoke_flat}


# --- refusals (before any write) ---


def test_recorded_run_with_the_registered_inputs_is_admitted() -> None:
    registry = load_registry(REGISTRY_V2)
    assert real_run_problems(
        stacks=STACKS, profile=SHOCKS_I, horizon=100, source_commit=COMMIT, registry=registry,
        worlds=portable_worlds(registry, profile=SHOCKS_I), filtered=False,
        champion=load_champion(DEFAULT_CHAMPION)) == []


def _drop_42(entries: tuple) -> tuple:
    return tuple(e for e in entries if not (e.regime == "shocked" and e.size == 42))


def _subset(entries: tuple) -> tuple:
    """The review repro: two calm 12 worlds plus one shocked 12 world cut to one episode."""
    calm = [e for e in entries if e.split == "development" and e.regime == "calm"
            and e.size == 12][:2]
    shocked = next(e for e in entries if e.id == PORTABLE_WORLD)
    return (*calm, dataclasses.replace(shocked, episode_seeds=shocked.episode_seeds[:1]))


def _reseeded(entries: tuple) -> tuple:
    return tuple(dataclasses.replace(e, episode_seeds=(1, *e.episode_seeds[1:]))
                 if e.id == PORTABLE_WORLD else e for e in entries)


@pytest.mark.parametrize("alter", [_drop_42, _subset, _reseeded])
def test_registry_that_changes_the_portable_grid_is_refused(
        alter: Callable[[tuple], tuple]) -> None:
    registry = load_registry(REGISTRY_V2)
    altered = dataclasses.replace(registry, entries=alter(registry.entries))
    problems = real_run_problems(
        stacks=STACKS, profile=SHOCKS_I, horizon=100, source_commit=COMMIT, registry=altered,
        worlds=portable_worlds(altered, profile=SHOCKS_I), filtered=False,
        champion=load_champion(DEFAULT_CHAMPION))
    assert len(problems) == 1
    assert problems[0].startswith("portable worlds differ from the registry v2 draft's")


@pytest.mark.parametrize("change", [{"held_out_topology_seeds": [6]}, {"depth": 4},
                                    {"held_out_topology_seeds": [6, 7, 8, 10]}])
def test_champion_manifest_that_changes_the_flat_grid_is_refused(
        change: dict, tmp_path: Path) -> None:
    manifest = json.loads(DEFAULT_CHAMPION.read_text()) | change
    for k in ("checkpoint", "producer_checkpoint"):
        manifest[k] = str((DEFAULT_CHAMPION.parent / manifest[k]).resolve())
    path = tmp_path / "champion.json"
    path.write_text(json.dumps(manifest))
    assert flat_grid_problem(load_champion(path)).startswith("champion manifest declares")
    out = tmp_path / "EXP-refused"
    with pytest.raises(ValueError, match="champion manifest declares"):
        main(["--out-dir", str(out), "--prod-ckpt", str(PORTABLE_PROD), "--orderer-ckpt",
              str(PORTABLE_ORDERER), "--champion", str(path), "--source-commit", COMMIT])
    assert not out.exists()


def test_smoke_profile_is_refused_without_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "EXP-refused"
    with pytest.raises(ValueError, match="test-only and runs only with --dry-run"):
        main(["--out-dir", str(out), "--prod-ckpt", str(PORTABLE_PROD),
              "--orderer-ckpt", str(PORTABLE_ORDERER), "--shock-profile", SHOCKS_I_SMOKE,
              "--source-commit", COMMIT])
    assert not out.exists()


def test_filters_and_horizon_are_refused_without_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "EXP-refused"
    with pytest.raises(ValueError, match="filters are set.*horizon 12 != 100"):
        main(["--out-dir", str(out), "--prod-ckpt", str(PORTABLE_PROD),
              "--orderer-ckpt", str(PORTABLE_ORDERER), "--horizon", "12",
              "--episodes-per-world", "1", "--source-commit", COMMIT])
    with pytest.raises(SystemExit):
        main(["--out-dir", str(out), "--stacks", FLAT, "--source-commit", COMMIT])
    assert not out.exists()


# --- checkpoint failures are contract failures ---


def test_broken_checkpoint_paths_are_recorded_as_contract_failures(tmp_path: Path) -> None:
    """A missing portable checkpoint and a missing champion manifest: one `run` row per stack,
    the run ends invalid (exit 2) before any job."""
    out = tmp_path / "EXP-broken"
    status = main(["--out-dir", str(out), "--dry-run", "--prod-ckpt",
                   str(tmp_path / "missing_producer.pt"), "--orderer-ckpt",
                   str(PORTABLE_ORDERER), "--champion", str(tmp_path / "missing.json"),
                   "--shock-profile", SHOCKS_I_SMOKE, "--horizon", str(HORIZON)])
    assert status == 2
    failures = read_failures(out)
    assert [(f["scope"], f["check"]) for f in failures] == [(RUN, CHECKPOINT)] * 2
    assert failures[0]["message"].startswith("portable stack: FileNotFoundError")
    assert failures[1]["message"].startswith("flat stack: FileNotFoundError")
    spec = json.loads((out / "spec.json").read_text())
    assert spec["status"] == "invalid" and spec["result"]["exit_holds"] is False
    assert not (out / "episodes.csv").exists()
    assert "INVALID" in (out / "RESULT.md").read_text()


def test_unloadable_checkpoint_is_a_contract_failure(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    out = tmp_path / "EXP-unloadable"
    status = main(["--out-dir", str(out), "--dry-run", "--stacks", PORTABLE,
                   "--prod-ckpt", str(bad), "--orderer-ckpt", str(bad),
                   "--shock-profile", SHOCKS_I_SMOKE, "--horizon", str(HORIZON)])
    assert status == 2
    failures = read_failures(out)
    assert len(failures) == 1 and failures[0]["check"] == CHECKPOINT
    assert failures[0]["message"].startswith("portable stack: ")


# --- the recorded run's grid (D11 counts) ---


def test_recorded_grid_is_80_plus_40_episodes_each_with_a_twin() -> None:
    worlds = portable_worlds(load_registry(REGISTRY_V2), profile=SHOCKS_I)
    assert sorted({e.size for e in worlds}) == [12, 42]
    assert len(worlds) == 8 and sum(len(e.episode_seeds) for e in worlds) == 80
    assert all(e.generator["params"]["shock_profile"] == SHOCKS_I for e in worlds)
    grid = flat_grid(load_champion(DEFAULT_CHAMPION), SHOCKS_I, sizes=None,
                     worlds_per_size=None, episodes_per_world=None)
    assert sorted(grid) == [f"flat-scale-d3-s{t}" for t in (6, 7, 8, 9)]
    assert all(g["size"] == 12 and len(set(g["episodes"])) == 10 for g in grid.values())
    portable = {e.id: {"episodes": list(e.episode_seeds)} for e in worlds}
    assert len(expected_keys(portable | grid)) == 2 * (80 + 40)
