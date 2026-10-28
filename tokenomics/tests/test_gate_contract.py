"""Contract-failure accounting (add-shocks-i-drift task 5.3, design D10; gate-harness "Contract
failures are counted": "Episode raises", "Action outside its space", "Episode without a terminal
outcome", "Failed run stays invalid", "Death is not a contract failure").

Fixtures stay at horizon 12, one episode, in-process (workers 1, so monkeypatched policies and
encoders reach the jobs). The world is `dev-sparse_wide-12-calm-01` at a reduced capacity scale:
at 0.1 `noop` production dies at cycle 10 (an early death), at 0.5 exactly at cycle 12 (death
and truncation together); `entity_planner` survives both."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import tokenomics.env.aec_env as aec_env
from tokenomics.env import TokenomicsAECEnv
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.contract import (
    ACTION,
    CHECKPOINT,
    EXCEPTION,
    FAILURE_COLUMNS,
    OBSERVATION,
    TERMINAL,
    TERMINATED,
    TRUNCATED,
    ContractMonitor,
    ContractRunError,
    ContractViolation,
    space_problem,
)
from tokenomics.eval.gate import DigestMismatch, load_side_effect_params, side_effect_params_json
from tokenomics.eval.gate_cli import (
    main,
    read_rows,
    recompute_details,
    run_probe,
    run_report,
    run_screen,
    scaled_entry,
)
from tokenomics.eval.registry import SEED_ORDER, Registry, build_worlds_v1, load_registry, save_registry
from tokenomics.eval.roster import Roster, RosterMember, contract_for, load_roster, save_roster
from tokenomics.eval.runner import run_episode

HORIZON = 12
WORLD = "dev-sparse_wide-12-calm-01"
BROKEN = "obs_price"  # the ordering baseline a test breaks; only one roster member uses it
MEMBERS = (("entity_planner", "even_split"), ("noop", BROKEN))
SUBJECT = "baseline:entity_planner/proportional"


def _entry(scale: float):  # -> WorldEntry
    return scaled_entry(build_worlds_v1(SEED_ORDER).entry(WORLD), scale)


def _env(scale: float = 0.1, horizon: int = HORIZON) -> TokenomicsAECEnv:
    return TokenomicsAECEnv(config=_entry(scale).config(), horizon=horizon)


def _monitored(env: TokenomicsAECEnv, prod: str = "entity_planner", order: str = "even_split",
               **kw: object) -> ContractMonitor:
    mon = ContractMonitor(env)
    run_episode(env, PRODUCING_BASELINES[prod], ORDERING_BASELINES[order], seed=7,
                action_seed=1, monitor=mon, **kw)  # type: ignore[arg-type]
    return mon


def _verify(mon: ContractMonitor, env: TokenomicsAECEnv) -> None:
    mon.verify(dead=bool(env.episode_stats["dead"]), cycles=env.episode_stats["cycles"],
               horizon=env.horizon)


# --- valid endings latch, and record nothing ---


@pytest.mark.parametrize("scale, prod, cycles, dead, latched", [
    (0.1, "entity_planner", 12, False, {TRUNCATED}),          # horizon end
    (0.1, "noop", 10, True, {TERMINATED}),                    # early death
    (0.5, "noop", 12, True, {TERMINATED, TRUNCATED}),         # death on the last cycle
])
def test_valid_endings_latch_every_agent(scale: float, prod: str, cycles: int, dead: bool,
                                         latched: set[str]) -> None:
    env = _env(scale)
    mon = _monitored(env, prod)
    assert (env.episode_stats["cycles"], env.episode_stats["dead"]) == (cycles, dead)
    assert set(mon.outcomes) == set(env.possible_agents)
    assert all(v == latched for v in mon.outcomes.values())
    _verify(mon, env)  # death is not a contract failure


# --- "Episode without a terminal outcome" ---


def test_loop_cut_by_max_iter_has_no_terminal_outcome() -> None:
    env = _env()
    mon = _monitored(env, max_iter=40)
    assert env.episode_stats["cycles"] < HORIZON
    with pytest.raises(ContractViolation, match="no terminal outcome") as ei:
        _verify(mon, env)
    assert ei.value.check == TERMINAL


class _FlagsForcedUnset(TokenomicsAECEnv):
    """Never reports truncation: the flags a horizon end sets are cleared at once."""

    def step(self, action: object) -> None:
        super().step(action)  # type: ignore[arg-type]
        self.truncations = {a: False for a in self.agents}


def test_horizon_reached_with_flags_unset_is_not_inferred() -> None:
    """cycles == horizon, yet no agent was ever reported truncated: a failure, because the
    checker never infers a truncation from the cycle count."""
    env = _FlagsForcedUnset(config=_entry(0.1).config(), horizon=HORIZON)
    mon = _monitored(env, max_iter=HORIZON * len(env.possible_agents))
    assert env.episode_stats["cycles"] == HORIZON and not env.episode_stats["dead"]
    with pytest.raises(ContractViolation, match="no terminal outcome"):
        _verify(mon, env)


class _StubEnv:
    possible_agents = ["a", "b"]


@pytest.mark.parametrize("outcomes, dead, cycles, match", [
    ({"a": {TRUNCATED}, "b": {TRUNCATED}}, False, 11, "latched truncated but"),
    ({"a": {TERMINATED}, "b": {TERMINATED}}, False, 12, "latched terminated but"),
    ({"a": {TRUNCATED}, "b": {TRUNCATED}}, True, 12, "latched no terminated but"),
    ({"a": {TERMINATED}, "b": {TERMINATED}}, True, 12, "latched no truncated but"),
    ({"a": {TRUNCATED}}, False, 12, "no terminal outcome for 1 of 2"),
])
def test_latches_must_agree_with_the_stats(outcomes: dict, dead: bool, cycles: int,
                                           match: str) -> None:
    mon = ContractMonitor(_StubEnv())
    mon.outcomes = outcomes
    with pytest.raises(ContractViolation, match=match):
        mon.verify(dead=dead, cycles=cycles, horizon=12)


# --- "Action outside its space" and observations ---


def _bad(base: Callable, edit: Callable[[np.ndarray, object], np.ndarray]) -> Callable:
    def policy(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        return edit(base(env, agent), env.action_space(agent))
    return policy


def _nan(a: np.ndarray, space: object) -> np.ndarray:
    """NaN in the last slot (a producer's investment toggle, an orderer's last logit): the env
    decodes it without raising (a NaN in the first slot would raise inside decode)."""
    a = a.copy()
    a[-1] = np.nan
    return a


def _above(a: np.ndarray, space: object) -> np.ndarray:
    a = a.copy()
    a[-1] = space.high.flat[-1] + 1.0  # type: ignore[attr-defined]
    return a


def _float64(a: np.ndarray, space: object) -> np.ndarray:
    return a.astype(np.float64)


@pytest.mark.parametrize("role, edit, match", [
    ("prod", _nan, "non-finite"),             # a NaN toggle reads as "off", silently
    ("order", _nan, "non-finite"),
    ("prod", _above, "outside the bounds"),   # the env clips the quantity
    ("order", _above, "outside the bounds"),
    ("prod", _float64, "does not cast safely"),
])
def test_action_outside_its_space_is_caught_although_the_env_accepts_it(
        role: str, edit: Callable, match: str) -> None:
    prod, order = PRODUCING_BASELINES["entity_planner"], ORDERING_BASELINES["even_split"]
    if role == "prod":
        prod = _bad(prod, edit)
    else:
        order = _bad(order, edit)
    env = _env()  # without the monitor the env decodes the action and the episode completes
    stats = run_episode(env, prod, order, seed=7, action_seed=1)
    assert stats["cycles"] == HORIZON
    env = _env()
    with pytest.raises(ContractViolation, match=match) as ei:
        run_episode(env, prod, order, seed=7, action_seed=1, monitor=ContractMonitor(env))
    assert ei.value.check == ACTION


def test_observation_outside_its_space(monkeypatch: pytest.MonkeyPatch) -> None:
    real = aec_env.encode_producing

    def encoder(*args: object, **kw: object) -> np.ndarray:
        obs = real(*args, **kw)
        obs[0] = 1e9  # above OBS_CLIP
        return obs

    monkeypatch.setattr(aec_env, "encode_producing", encoder)
    env = _env()
    with pytest.raises(ContractViolation, match="outside the bounds") as ei:
        _monitored(env)
    assert ei.value.check == OBSERVATION


@pytest.mark.parametrize("x, match", [
    ([0.0, 1.0], "not an ndarray"), (np.zeros(3, np.float32), "shape"),
    (np.array([0.0, np.inf], np.float32), "non-finite"),
    (np.array([0.0, -1.0], np.float32), "outside the bounds"),
])
def test_space_problem_names_the_defect(x: object, match: str) -> None:
    from gymnasium import spaces

    problem = space_problem(x, spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32))
    assert problem is not None and match in problem
    mon = ContractMonitor(type("E", (), {"action_space": lambda self, a: spaces.Box(
        0.0, 1.0, shape=(2,), dtype=np.float32)})())
    with pytest.raises(ContractViolation, match=match):
        mon.on_action("x", x)
    mon.on_action("x", np.array([0.0, 1.0], np.float32))  # bounds are inclusive


def test_infinite_value_in_an_unbounded_space() -> None:
    from gymnasium import spaces

    box = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
    mon = ContractMonitor(type("E", (), {"action_space": lambda self, a: box})())
    with pytest.raises(ContractViolation, match="non-finite"):
        mon.on_action("x", np.array([np.inf, 0.0], np.float32))
    mon.on_action("x", np.array([-1e30, 1e30], np.float32))


def test_outcome_is_latched_the_first_time() -> None:
    from gymnasium import spaces

    box = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
    mon = ContractMonitor(type("E", (), {"observation_space": lambda self, a: box})())
    obs = np.zeros(1, np.float32)
    mon.on_last("a", obs, False, False)
    assert mon.outcomes == {}
    mon.on_last("a", obs, False, True)
    mon.on_last("a", obs, True, True)  # later flags never rewrite the latch
    assert mon.outcomes == {"a": {TRUNCATED}}


# --- gate runs ---


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("inputs")
    full = build_worlds_v1(SEED_ORDER)
    save_registry(Registry(name="fixture", seed_ranges=full.seed_ranges,
                           entries=(_entry(0.1),)), d / "registry.json")
    save_roster(Roster(name="fixture-roster", replicates=1, members=tuple(
        RosterMember(p, o, contract_for(p, o), False) for p, o in MEMBERS)), d / "roster.json")
    (d / "params.json").write_text(side_effect_params_json(0.5, 0.0, 0.05, name="fixture"))
    return {"registry": d / "registry.json", "roster": d / "roster.json",
            "params": d / "params.json"}


def _argv(inputs: dict[str, Path], out: Path, subject: str = SUBJECT, *extra: str) -> list[str]:
    return ["--mode", "report", "--out-dir", str(out), "--subject", subject,
            "--registry", str(inputs["registry"]), "--roster", str(inputs["roster"]),
            "--params", str(inputs["params"]), "--horizon", str(HORIZON),
            "--episodes-per-world", "1", "--workers", "1", "--dry-run", *extra]


def _failures(out: Path) -> list[dict[str, str]]:
    with (out / "contract_failures.csv").open() as fh:
        return list(csv.DictReader(fh))


def _raises(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
    raise RuntimeError("policy blew up")


_nan_order = _bad(ORDERING_BASELINES["even_split"], _nan)  # decoded silently without the check


@pytest.mark.parametrize("broken, check, error_type", [
    (_raises, EXCEPTION, "RuntimeError"), (_nan_order, ACTION, "ContractViolation"),
])
def test_failed_job_is_recorded_others_finish_and_the_run_stays_invalid(
        inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        broken: Callable, check: str, error_type: str) -> None:
    out = tmp_path / "run"
    with monkeypatch.context() as m:
        m.setitem(ORDERING_BASELINES, BROKEN, broken)
        assert main(_argv(inputs, out)) == 2
    (f,) = _failures(out)
    assert tuple(f) == FAILURE_COLUMNS
    member = f"noop/{BROKEN}"
    assert (f["scope"], f["check"], f["error_type"], f["world"], f["subject"],
            f["replicate"]) == ("job", check, error_type, WORLD, member, "0")
    assert int(f["episode"]) == _entry(0.1).episode_seeds[0]
    rows = read_rows(out / "episodes.csv")
    assert sorted(r.subject for r in rows) == ["candidate", "entity_planner/even_split"]
    spec = json.loads((out / "spec.json").read_text())
    verdict = json.loads((out / "verdict.json").read_text())
    assert spec["status"] == verdict["status"] == "invalid"
    assert (verdict["contract_failures"], verdict["missing_rows"]) == (1, 1)
    assert not (out / "worlds.csv").exists() and "INVALID" in (out / "RESULT.md").read_text()
    r = recompute_details(out)
    assert (r.status, r.report, r.contract_failures, r.missing_rows) == ("invalid", None, 1, 1)

    # "Failed run stays invalid": the policy is fixed, the resume runs the failed job, which
    # now succeeds; the failure row is kept and the run is still invalid.
    assert main(_argv(inputs, out)) == 2
    assert len(read_rows(out / "episodes.csv")) == 3
    assert _failures(out) == [f]
    r = recompute_details(out)
    assert (r.status, r.contract_failures, r.missing_rows) == ("invalid", 1, 0)
    assert main(["--mode", "recompute", "--out-dir", str(out)]) == 2
    assert json.loads((out / "spec.json").read_text())["status"] == "invalid"


def test_episode_without_terminal_outcome_in_a_gate_run(inputs: dict[str, Path], tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Every job's step loop is cut by `max_iter`: each job records a terminal-outcome failure
    (checked in the job, after the loop) and the run is invalid."""
    import tokenomics.eval.runner as runner

    real = runner.run_episode
    monkeypatch.setattr(runner, "run_episode",
                        lambda *a, **kw: real(*a, **{**kw, "max_iter": 40}))
    out = tmp_path / "cut"
    assert main(_argv(inputs, out)) == 2
    fails = _failures(out)
    assert sorted(f["subject"] for f in fails) == sorted(
        ["candidate", "entity_planner/even_split", f"noop/{BROKEN}"])
    assert {(f["scope"], f["check"]) for f in fails} == {("job", TERMINAL)}
    assert not (out / "episodes.csv").exists() or read_rows(out / "episodes.csv") == []


def test_death_is_not_a_contract_failure(inputs: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "dies"
    assert main(_argv(inputs, out, "baseline:noop/even_split")) == 0
    cand = next(r for r in read_rows(out / "episodes.csv") if r.subject == "candidate")
    assert cand.dead
    assert _failures(out) == []  # header-only file: zero failures, recorded
    assert (out / "contract_failures.csv").read_text().strip() == ",".join(FAILURE_COLUMNS)
    verdict = json.loads((out / "verdict.json").read_text())
    assert "zero_death" in verdict["failing_clauses"]
    assert json.loads((out / "spec.json").read_text())["result"]["contract_failures"] == 0
    assert recompute_details(out).ok


def test_missing_checkpoint_is_a_run_failure(inputs: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "ckpt"
    missing = str(tmp_path / "absent.pt")
    argv = _argv(inputs, out, "portable", "--prod-ckpt", missing, "--orderer-ckpt", missing)
    assert main(argv) == 2
    (f,) = _failures(out)
    assert (f["scope"], f["check"], f["error_type"], f["world"], f["episode"]) == \
        ("run", CHECKPOINT, "FileNotFoundError", "", "")
    assert not (out / "episodes.csv").exists()
    spec = json.loads((out / "spec.json").read_text())
    verdict = json.loads((out / "verdict.json").read_text())
    assert spec["status"] == verdict["status"] == "invalid"
    assert verdict["contract_failures"] == 1 and "pins" not in spec
    assert spec["inputs"]["producer"]["path"] == missing
    assert recompute_details(out).status == "invalid"
    with pytest.raises(ContractRunError, match="checkpoint failed"):
        run_report(out_dir=tmp_path / "ckpt2", subject="portable", split="development",
                   horizon=HORIZON, workers=1, registry=load_registry(inputs["registry"]),
                   roster=load_roster(inputs["roster"]),
                   params=load_side_effect_params(inputs["params"]), params_payload={},
                   prod_ckpt=missing, orderer_ckpt=missing, episodes_per_world=1,
                   dry_run=True)


def test_retry_after_the_checkpoint_is_restored_runs_and_stays_invalid(
        inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that failed before any job (missing checkpoint) leaves a pin-less invalid spec;
    a retry once the file exists pins and runs every job, and the recorded failure keeps
    the run invalid."""
    out = tmp_path / "retry"
    prod, order = tmp_path / "producer.pt", tmp_path / "orderer.pt"
    kw = _portable_kw(inputs, out, prod, order, monkeypatch)
    with pytest.raises(ContractRunError, match="checkpoint failed"):
        run_report(**kw)  # type: ignore[arg-type]
    assert "pins" not in json.loads((out / "spec.json").read_text())
    prod.write_bytes(b"fake-producer")
    order.write_bytes(b"fake-orderer")
    report, counts = run_report(**kw)  # type: ignore[arg-type]
    assert report is None and counts["run"] == 3 == len(read_rows(out / "episodes.csv"))
    spec = json.loads((out / "spec.json").read_text())
    assert spec["status"] == "invalid" and spec["pins"]["subject"] == "portable"
    assert [(f["scope"], f["check"]) for f in _failures(out)] == [("run", CHECKPOINT)]
    r = recompute_details(out)
    assert (r.status, r.contract_failures, r.missing_rows) == ("invalid", 1, 0)


def _portable_kw(inputs: dict[str, Path], out: Path, prod: Path, order: Path,
                 monkeypatch: pytest.MonkeyPatch) -> dict:
    """run_report kwargs for a portable dry run whose candidate plays a baseline: loaders
    stubbed (fake checkpoint bytes), no torch model."""
    import tokenomics.eval.gate_cli as gc
    import tokenomics.train.orderer_scorer as osc
    import tokenomics.train.producer_scorer as psc

    monkeypatch.setattr(osc, "load_scorer", lambda path: None)
    monkeypatch.setattr(psc, "load_producer_score_model", lambda path: None)
    monkeypatch.setattr(gc, "_worker_init", lambda h, p, o: gc._W.update(horizon=h, configs={}))

    def stub(job: dict) -> dict:
        if job["producing"] == "portable":
            job = dict(job, producing="entity_planner", ordering="proportional")
        return gc.run_job(job)

    return dict(out_dir=out, subject="portable", split="development", horizon=HORIZON,
                workers=1, registry=load_registry(inputs["registry"]),
                roster=load_roster(inputs["roster"]),
                params=load_side_effect_params(inputs["params"]), params_payload={},
                prod_ckpt=str(prod), orderer_ckpt=str(order), episodes_per_world=1,
                dry_run=True, run_job_fn=stub, log=lambda _: None)


def test_checkpoint_lost_after_a_complete_run_invalidates_it(
        inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete portable report rerun after its checkpoint disappeared: the run row is
    appended, and spec.json / verdict.json turn invalid, keeping the recorded pins."""
    prod, order = tmp_path / "producer.pt", tmp_path / "orderer.pt"
    prod.write_bytes(b"fake-producer")
    order.write_bytes(b"fake-orderer")
    out = tmp_path / "out"
    kw = _portable_kw(inputs, out, prod, order, monkeypatch)
    report, _ = run_report(**kw)  # type: ignore[arg-type]
    assert report is not None
    before = json.loads((out / "spec.json").read_text())
    assert before["status"] == "complete" and "overall" in json.loads(
        (out / "verdict.json").read_text())
    prod.unlink()
    with pytest.raises(ContractRunError, match="checkpoint failed"):
        run_report(**kw)  # type: ignore[arg-type]
    spec = json.loads((out / "spec.json").read_text())
    verdict = json.loads((out / "verdict.json").read_text())
    assert spec["status"] == verdict["status"] == "invalid"
    assert spec["pins"] == before["pins"] and spec["source"] == before["source"]
    assert [(f["scope"], f["check"]) for f in _failures(out)] == [("run", CHECKPOINT)]
    assert recompute_details(out).status == "invalid"


@pytest.mark.parametrize("broken", ["producer", "orderer"])
def test_unloadable_checkpoint_is_a_run_failure(inputs: dict[str, Path], tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch,
                                                broken: str) -> None:
    """A checkpoint that hashes but does not load fails in the parent, before any job. The
    other checkpoint's loader is stubbed, so each load is checked on its own."""
    pytest.importorskip("torch")
    import tokenomics.train.orderer_scorer as osc
    import tokenomics.train.producer_scorer as psc

    if broken == "producer":
        monkeypatch.setattr(osc, "load_scorer", lambda path: None)
    else:
        monkeypatch.setattr(psc, "load_producer_score_model", lambda path: None)
    bad = tmp_path / "garbage.pt"
    bad.write_bytes(b"not a checkpoint")
    out = tmp_path / "unloadable"
    assert main(_argv(inputs, out, "portable", "--prod-ckpt", str(bad),
                      "--orderer-ckpt", str(bad))) == 2
    (f,) = _failures(out)
    assert (f["scope"], f["check"]) == ("run", CHECKPOINT)
    assert f["error_type"] != "FileNotFoundError"
    assert not (out / "episodes.csv").exists()
    assert json.loads((out / "verdict.json").read_text())["status"] == "invalid"


def _tamper_member(job: dict) -> dict[str, str]:
    """Real rows, except the broken member's demand digest: a pairing failure."""
    from tokenomics.eval.gate_cli import run_job

    d = run_job(job)
    return {**d, "demand_digest": "f" * 64} if job["subject"] == f"noop/{BROKEN}" else d


@pytest.mark.parametrize("mode", ["screen", "probe"])
def test_screen_and_probe_record_a_digest_mismatch_as_invalid(
        inputs: dict[str, Path], tmp_path: Path, mode: str) -> None:
    out = tmp_path / mode
    common = dict(horizon=HORIZON, workers=1, registry=load_registry(inputs["registry"]),
                  roster=load_roster(inputs["roster"]), episodes_per_world=1,
                  run_job_fn=_tamper_member, log=lambda _: None)
    with pytest.raises(DigestMismatch, match=f"noop/{BROKEN} replicate 0 saw demand digest"):
        if mode == "screen":
            run_screen(out_dir=out, reserves="none", dry_run=True, **common)  # type: ignore[arg-type]
        else:
            run_probe(out_dir=out, scales=[1.0], families=["sparse_wide"], **common)  # type: ignore[arg-type]
    run_dir = out if mode == "screen" else out / "scale-1"
    assert json.loads((run_dir / "spec.json").read_text())["status"] == "invalid"
    assert [(f["scope"], f["check"]) for f in _failures(run_dir)] == [("run", "digest_pairing")]


def test_cli_exits_2_on_a_screen_digest_mismatch(inputs: dict[str, Path], tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    import tokenomics.eval.gate_cli as gc

    def mismatch(**kw: object) -> None:
        raise DigestMismatch("world w episode 1: m replicate 0 saw demand digest x")

    monkeypatch.setattr(gc, "run_screen", mismatch)
    assert main(["--mode", "screen", "--out-dir", str(tmp_path / "s"), "--registry",
                 str(inputs["registry"]), "--roster", str(inputs["roster"]), "--dry-run"]) == 2


def test_screen_with_a_failed_job_is_invalid(inputs: dict[str, Path], tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "screen"
    monkeypatch.setitem(ORDERING_BASELINES, BROKEN, _raises)
    with pytest.raises(ContractRunError, match="1 contract failure"):
        run_screen(out_dir=out, horizon=HORIZON, workers=1,
                   registry=load_registry(inputs["registry"]),
                   roster=load_roster(inputs["roster"]), reserves="none",
                   episodes_per_world=1, dry_run=True, log=lambda _: None)
    assert json.loads((out / "spec.json").read_text())["status"] == "invalid"
    assert not (out / "frozen").exists() and not (out / "screening.json").exists()
