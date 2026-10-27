"""Gate driver end to end (design D6, D11, add-measurement-gate-harness; gate-harness "Identical
demand across observation contracts", "Verdict is reproducible from rows", "Provenance binding",
"Sealed worlds need a pre-registration"). Fixture: 2 calm 12-producer worlds, horizon 12, 1
episode each, a 3-member roster (one random member with R = 2, one legacy-contract member)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

from tokenomics.eval.gate import (
    CANDIDATE,
    DigestMismatch,
    load_side_effect_params,
    params_from_dict,
    side_effect_params_json,
)
from tokenomics.eval.gate_cli import (
    build_jobs,
    main,
    read_rows,
    recompute,
    run_job,
    run_report,
    select_worlds,
)
from tokenomics.eval.registry import (
    Registry,
    SealedAccessError,
    SEED_ORDER,
    build_worlds_v1,
    load_registry,
    save_registry,
)
from tokenomics.eval.roster import Roster, RosterMember, contract_for, load_roster, save_roster

WORLDS = ("dev-dense-12-calm-00", "dev-sparse_wide-12-calm-00")
MEMBERS = (("entity_planner", "proportional"), ("random", "even_split"),
           ("obs_stock_aware", "obs_price"))
SUBJECT = "baseline:entity_planner/blend_0.5"


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("inputs")
    full = build_worlds_v1(SEED_ORDER)
    reg = Registry(name="fixture", seed_ranges=full.seed_ranges,
                   entries=tuple(full.entry(w) for w in WORLDS))
    save_registry(reg, d / "registry.json")
    roster = Roster(name="fixture-roster", replicates=2, members=tuple(
        RosterMember(p, o, contract_for(p, o), "random" in (p, o)) for p, o in MEMBERS))
    save_roster(roster, d / "roster.json")
    (d / "params.json").write_text(side_effect_params_json(0.5, 0.0, 0.05, name="fixture"))
    return {"registry": d / "registry.json", "roster": d / "roster.json",
            "params": d / "params.json"}


def _argv(inputs: dict[str, Path], out: Path, *extra: str) -> list[str]:
    return ["--mode", "report", "--out-dir", str(out), "--subject", SUBJECT,
            "--registry", str(inputs["registry"]), "--roster", str(inputs["roster"]),
            "--params", str(inputs["params"]), "--horizon", "12", "--episodes-per-world", "1",
            "--dry-run", *extra]


@pytest.fixture(scope="module")
def report_dir(inputs: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("EXP-gate-fixture")
    assert main(_argv(inputs, out, "--workers", "2", "--source-commit", "a" * 40)) == 0
    return out


def test_report_writes_exp_layout(report_dir: Path) -> None:
    for name in ("spec.json", "episodes.csv", "worlds.csv", "regimes.csv", "verdict.json",
                 "RESULT.md"):
        assert (report_dir / name).exists(), name
    rows = read_rows(report_dir / "episodes.csv")
    # per world-episode: candidate 1 + entity_planner 1 + random 2 replicates + obs 1 = 5
    assert len(rows) == 2 * 5
    assert {r.world for r in rows} == set(WORLDS)
    assert sum(r.subject == CANDIDATE for r in rows) == 2
    with (report_dir / "worlds.csv").open() as fh:
        worlds = list(csv.DictReader(fh))
    assert [w["world"] for w in worlds] == sorted(WORLDS)
    verdict = json.loads((report_dir / "verdict.json").read_text())
    assert verdict["overall"] in ("accept", "reject")


def test_spec_records_provenance(report_dir: Path, inputs: dict[str, Path]) -> None:
    spec = json.loads((report_dir / "spec.json").read_text())
    assert spec["status"] == "complete"
    assert spec["source"]["commit"] == "a" * 40 and spec["source"]["tree_sha256"]
    assert spec["environment"]["python"]
    pins = spec["pins"]
    assert pins["registry_version"] == load_registry(inputs["registry"]).version
    assert pins["roster_version"] == load_roster(inputs["roster"]).version
    assert pins["params_version"] == load_side_effect_params(inputs["params"]).version
    assert pins["checkpoints"] == {}  # baseline subject: no policy checkpoints


def test_random_member_replicates_use_hash_seeds(report_dir: Path) -> None:
    rows = read_rows(report_dir / "episodes.csv")
    rnd = [r for r in rows if r.subject == "random/even_split"]
    assert sorted(r.replicate for r in rnd) == [0, 0, 1, 1]
    jobs = build_jobs(select_worlds(load_registry(), "development", sizes=[12], max_worlds=1),
                      load_roster(), SUBJECT, episodes_per_world=1)
    seeds = {(j["subject"], j["replicate"]): j["action_seed"] for j in jobs}
    assert seeds[("random/even_split", 0)] != seeds[("random/even_split", 1)]
    assert seeds[("always_restore/even_split", 0)] is None  # deterministic: default seeding


def test_recompute_matches_report(report_dir: Path) -> None:
    report, ok = recompute(report_dir)
    assert ok
    assert report.to_json() == (report_dir / "verdict.json").read_text()


def test_documented_recompute_command_runs(report_dir: Path) -> None:
    """The CLAUDE.md command, as written (`python -m tokenomics.eval.gate --mode recompute
    --out-dir <EXP dir>`)."""
    cmd = [sys.executable, "-m", "tokenomics.eval.gate", "--mode", "recompute",
           "--out-dir", str(report_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert "MATCH" in res.stdout and "MISMATCH" not in res.stdout


def test_recompute_detects_tampered_digest(report_dir: Path, tmp_path: Path) -> None:
    """An injected digest mismatch in the CSV fails recompute, naming world, episode, member."""
    for name in ("spec.json", "verdict.json"):
        (tmp_path / name).write_text((report_dir / name).read_text())
    with (report_dir / "episodes.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    victim = next(r for r in rows if r["subject"] == "obs_stock_aware/obs_price")
    victim["demand_digest"] = "0" * 64
    with (tmp_path / "episodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(DigestMismatch, match=rf"world {victim['world']} episode "
                                             rf"{victim['episode']}: obs_stock_aware/obs_price"):
        recompute(tmp_path)


def test_report_digest_mismatch_raises_early(inputs: dict[str, Path], tmp_path: Path) -> None:
    """A member whose episode saw different demand stops the report run with an error naming
    world, episode and member (gate-harness "Digest mismatch")."""
    def tampered(job: dict) -> dict:
        row = run_job(job)
        if job["subject"] == "obs_stock_aware/obs_price":
            row["demand_digest"] = "f" * 64
        return row

    with pytest.raises(DigestMismatch, match=r"world dev-dense-12-calm-00 episode \d+: "
                                             r"obs_stock_aware/obs_price replicate 0"):
        _run(inputs, tmp_path, run_job_fn=tampered, max_worlds=1)


def _run(inputs: dict[str, Path], out: Path, **kw: object) -> tuple:
    params_text = Path(inputs["params"]).read_text()
    defaults = dict(
        out_dir=out, subject=SUBJECT, split="development", horizon=12, workers=1,
        registry=load_registry(inputs["registry"]), roster=load_roster(inputs["roster"]),
        params=params_from_dict(json.loads(params_text)),
        params_payload=json.loads(params_text), episodes_per_world=1, dry_run=True,
        source_commit="a" * 40, log=lambda s: None)  # the commit report_dir was made with
    defaults.update(kw)
    return run_report(**defaults)  # type: ignore[arg-type]


def test_resume_skips_done_rows(report_dir: Path, inputs: dict[str, Path], tmp_path: Path) -> None:
    """Keep the header + 6 complete rows and a torn half line; the rerun drops the torn line, runs
    only the 4 missing jobs and reproduces the original verdict byte for byte."""
    lines = (report_dir / "episodes.csv").read_text().splitlines(keepends=True)
    (tmp_path / "episodes.csv").write_text("".join(lines[:7]) + lines[7][:20])
    (tmp_path / "spec.json").write_text((report_dir / "spec.json").read_text())
    ran: list[dict] = []

    def counting(job: dict) -> dict:
        ran.append(job)
        return run_job(job)

    report, counts = _run(inputs, tmp_path, run_job_fn=counting)
    assert counts == {"jobs": 10, "skipped": 6, "run": 4} and len(ran) == 4
    assert report.to_json() == (report_dir / "verdict.json").read_text()
    assert len(read_rows(tmp_path / "episodes.csv")) == 10
    # a compatible resume keeps the original provenance and records itself
    old, new = (json.loads((d / "spec.json").read_text()) for d in (report_dir, tmp_path))
    for k in ("source", "environment", "started", "date", "pins"):
        assert new[k] == old[k], k
    assert len(new["resumes"]) == 1 and new["status"] == "complete"


def test_resume_refuses_different_pins(report_dir: Path, inputs: dict[str, Path],
                                       tmp_path: Path) -> None:
    (tmp_path / "spec.json").write_text((report_dir / "spec.json").read_text())
    with pytest.raises(ValueError, match="different pins"):
        _run(inputs, tmp_path, horizon=11)


def test_sealed_split_needs_preregistration(inputs: dict[str, Path], tmp_path: Path) -> None:
    full = build_worlds_v1(SEED_ORDER)
    reg = Registry(name="fixture-sealed", seed_ranges=full.seed_ranges,
                   entries=(full.entry("sea-dense-12-calm-00"),))
    log = tmp_path / "SEALED_ACCESS.log"
    with pytest.raises(SealedAccessError, match="without a pre-registration manifest"):
        _run(inputs, tmp_path / "out", split="sealed", registry=reg, access_log=log)
    assert not log.exists() and not (tmp_path / "out").exists()  # refused before anything ran


def test_portable_subject_needs_checkpoints(inputs: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs --prod-ckpt and --orderer-ckpt"):
        _run(inputs, tmp_path, subject="portable")


def test_params_file_is_versioned(inputs: dict[str, Path]) -> None:
    d = json.loads(Path(inputs["params"]).read_text())
    p = params_from_dict(d)
    assert (p.x, p.y, p.version) == (0.5, 0.0, d["content_hash"])
    d["y"] = 0.2
    with pytest.raises(ValueError, match="edited without a new version"):
        params_from_dict(d)


def _child_blas(_: int) -> tuple[dict[str, str], int | None]:
    from tokenomics.eval.gate_cli import BLAS_ENV, blas_threads

    return {k: os.environ.get(k, "") for k in BLAS_ENV}, blas_threads()


def test_blas_pinned_in_spawned_workers_despite_inherited_settings(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A shell exporting OPENBLAS_NUM_THREADS=32 (etc.) still yields one-thread workers: the
    parent forces the env before spawning and each worker's OpenBLAS reports 1 thread."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from tokenomics.eval.gate_cli import BLAS_ENV, _worker_init, blas_threads, pin_blas

    for k in BLAS_ENV:
        monkeypatch.setenv(k, "32")
    pin_blas()
    assert all(os.environ[k] == "1" for k in BLAS_ENV)
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"),
                             initializer=_worker_init, initargs=(12, None, None)) as pool:
        env, threads = pool.submit(_child_blas, 0).result()
    assert set(env.values()) == {"1"}
    if threads is not None:
        assert threads == 1
    if blas_threads() is not None:
        assert blas_threads() == 1  # the in-process (--workers 1) path is pinned too


# --- resume / recompute integrity (pinned job grid, provenance) ---


def _copy_run(src: Path, dst: Path, *, rows: list[dict] | None = None) -> None:
    for name in ("spec.json", "verdict.json"):
        (dst / name).write_text((src / name).read_text())
    if rows is None:
        (dst / "episodes.csv").write_text((src / "episodes.csv").read_text())
        return
    with (dst / "episodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _csv_rows(d: Path) -> list[dict]:
    with (d / "episodes.csv").open() as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("drop,match", [
    (lambda r: r["world"] == "dev-sparse_wide-12-calm-00", "incomplete"),  # a whole world
    (lambda r: r["subject"] == CANDIDATE and r["world"] == "dev-dense-12-calm-00", "incomplete"),
    (lambda r: r["subject"] == "random/even_split" and r["replicate"] == "1", "incomplete"),
])
def test_recompute_refuses_missing_rows(report_dir: Path, tmp_path: Path,
                                       drop: Callable[[dict], bool], match: str) -> None:
    rows = [r for r in _csv_rows(report_dir) if not drop(r)]
    _copy_run(report_dir, tmp_path, rows=rows)
    with pytest.raises(ValueError, match=match):
        recompute(tmp_path)


def test_recompute_refuses_extra_and_duplicate_rows(report_dir: Path, tmp_path: Path) -> None:
    rows = _csv_rows(report_dir)
    _copy_run(report_dir, tmp_path, rows=rows + [dict(rows[0], world="dev-dense-12-calm-01")])
    with pytest.raises(ValueError, match="not in the pinned job grid"):
        recompute(tmp_path)
    _copy_run(report_dir, tmp_path, rows=rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate row"):
        recompute(tmp_path)


def test_resume_refuses_rows_outside_grid(report_dir: Path, inputs: dict[str, Path],
                                          tmp_path: Path) -> None:
    rows = _csv_rows(report_dir)
    _copy_run(report_dir, tmp_path, rows=rows[:3] + [dict(rows[0], episode="7")])
    before = (tmp_path / "spec.json").read_text()
    with pytest.raises(ValueError, match="not in the pinned job grid"):
        _run(inputs, tmp_path)
    assert (tmp_path / "spec.json").read_text() == before


def test_orphan_csv_refused(report_dir: Path, inputs: dict[str, Path], tmp_path: Path) -> None:
    (tmp_path / "episodes.csv").write_text((report_dir / "episodes.csv").read_text())
    with pytest.raises(ValueError, match="orphan rows"):
        _run(inputs, tmp_path)
    assert not (tmp_path / "spec.json").exists()


def test_resume_refuses_changed_source(report_dir: Path, inputs: dict[str, Path], tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows made by one source tree are never mixed with rows from another: a resume whose source
    fingerprint differs is refused before spec.json or episodes.csv change."""
    import tokenomics.eval.gate_cli as gc

    _copy_run(report_dir, tmp_path)
    before = {n: (tmp_path / n).read_text() for n in ("spec.json", "episodes.csv")}
    monkeypatch.setattr(gc, "source_fingerprint",
                        lambda commit=None: {"tree_sha256": "0" * 64, "n_files": 1,
                                             "commit": "a" * 40})
    with pytest.raises(ValueError, match="different pins"):
        _run(inputs, tmp_path, source_commit="a" * 40)
    assert {n: (tmp_path / n).read_text() for n in before} == before


# --- dry run vs real report ---


def test_real_report_requires_frozen_inputs_and_commit(inputs: dict[str, Path],
                                                       tmp_path: Path) -> None:
    out = tmp_path / "real"
    with pytest.raises(ValueError, match=r"roster .* is not frozen.*registry .* has not screened the development split"
                                         r".*no full 40-hex source commit.*filters are set"):
        _run(inputs, out, dry_run=False, source_commit="")
    assert not out.exists()


def test_real_report_accepts_frozen_screened_full_commit(inputs: dict[str, Path],
                                                         tmp_path: Path) -> None:
    """The guard passes with a frozen roster, a screened registry and a commit (stops at the
    first job via an exploding runner: no episode is needed to show the guard let it through)."""
    from tokenomics.eval.gate import EpisodeRow
    from tokenomics.eval.registry import screen_zero_baselines
    from tokenomics.eval.roster import freeze_roster

    def boom_unused(job: dict) -> dict:
        raise AssertionError("a refused report must not run any job")

    reg = build_worlds_v1()  # the M1 registry: sparse-wide at both gate sizes (task 10.4)
    roster = freeze_roster(load_roster(inputs["roster"]))
    def rows_for(registry: Registry) -> list[EpisodeRow]:
        return [EpisodeRow(world=e.id, size=e.size, regime=e.regime, episode=ep, subject=m.id,
                           replicate=r, units=1.0, dead=False, demand_digest="d", r_start=1.0,
                           r_end=1.0, consumer_fill_rate={})
                for e in registry.gate_worlds("development") for ep in e.episode_seeds
                for m in roster.members for r in range(roster.replicates_for(m))]

    rows = rows_for(reg)
    from tokenomics.eval.families import DENSE, SPARSE_WIDE
    from tokenomics.eval.gate import Headroom
    from tokenomics.eval.registry import record_headroom

    unrecorded = screen_zero_baselines(reg, roster, rows)
    ok = [Headroom(SPARSE_WIDE, z, 8, 0.5, 0.95, True) for z in (12, 42)]
    screened = record_headroom(unrecorded, ok)
    fixture = load_registry(inputs["registry"])  # 12 producers only
    fixture_screened = record_headroom(
        screen_zero_baselines(fixture, roster, rows_for(fixture)),
        [Headroom(DENSE, 12, 1, 0.5, 0.95, True), Headroom(SPARSE_WIDE, 12, 1, 0.5, 0.95, True)])
    for bad, why in ((unrecorded, "records no development headroom check"),
                     (record_headroom(unrecorded, ok[:-1] + [Headroom(SPARSE_WIDE, 42, 8, 0.97,
                                                                      0.95, False)]),
                      "headroom failed"),
                     (record_headroom(unrecorded, ok[:-1]), "no headroom result for"),
                     (fixture_screened, "no development gate worlds for dense/42")):
        with pytest.raises(ValueError, match=why):
            _run(inputs, tmp_path / "refused", dry_run=False, registry=bad, roster=roster,
                 episodes_per_world=None, source_commit="b" * 40, run_job_fn=boom_unused)
    assert not (tmp_path / "refused").exists()

    class Boom(Exception):
        pass

    def boom(job: dict) -> dict:
        raise Boom

    with pytest.raises(Boom):
        _run(inputs, tmp_path, dry_run=False, registry=screened, roster=roster,
             episodes_per_world=None, source_commit="b" * 40, run_job_fn=boom)
    spec = json.loads((tmp_path / "spec.json").read_text())
    assert spec["kind"] == "report" and spec["pins"]["dry_run"] is False


def test_dry_run_is_marked(report_dir: Path) -> None:
    spec = json.loads((report_dir / "spec.json").read_text())
    assert spec["kind"] == "dry-run" and spec["pins"]["dry_run"] is True
    assert "DRY RUN" in (report_dir / "RESULT.md").read_text()


def test_cli_roster_filters_need_dry_run(inputs: dict[str, Path], tmp_path: Path) -> None:
    argv = [a for a in _argv(inputs, tmp_path / "x") if a != "--dry-run"]
    with pytest.raises(SystemExit):
        main(argv + ["--max-members", "1"])


def test_portable_subject_pins_checkpoint_hashes(inputs: dict[str, Path], tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """The portable path pins the SHA-256 of both checkpoints (provenance) and a resume with a
    changed checkpoint is refused. Fake checkpoint files + a stub worker that plays the candidate
    with a baseline: no torch, no model."""
    import tokenomics.eval.gate_cli as gc
    from tokenomics.eval.provenance import sha256_file

    prod, order = tmp_path / "producer.pt", tmp_path / "orderer.pt"
    prod.write_bytes(b"fake-producer")
    order.write_bytes(b"fake-orderer")
    monkeypatch.setattr(gc, "_worker_init", lambda h, p, o: gc._W.update(horizon=h, configs={}))

    def stub(job: dict) -> dict:
        if job["producing"] == "portable":
            job = dict(job, producing="entity_planner", ordering="proportional")
        return run_job(job)

    out = tmp_path / "out"
    _run(inputs, out, subject="portable", prod_ckpt=str(prod), orderer_ckpt=str(order),
         run_job_fn=stub, max_worlds=1)
    spec = json.loads((out / "spec.json").read_text())
    want = {"producer": sha256_file(str(prod)), "orderer": sha256_file(str(order))}
    assert spec["pins"]["checkpoints"] == want
    assert {k: v["sha256"] for k, v in spec["inputs"].items()} == want
    assert spec["inputs"]["producer"]["path"] == str(prod)
    assert recompute(out)[1]
    order.write_bytes(b"retrained-orderer")
    with pytest.raises(ValueError, match="different pins"):
        _run(inputs, out, subject="portable", prod_ckpt=str(prod), orderer_ckpt=str(order),
             run_job_fn=stub, max_worlds=1)
