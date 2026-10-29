"""PoC-B deployed-path cost driver (task 10.1, design D7): slope + flag arithmetic, the
42-producer cap, stage timers that add up and change nothing, and a dry run's layout."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from tokenomics.env import aec_env
from tokenomics.eval import deployed_cost as dc
from tokenomics.domain.config import Configuration
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from tokenomics.eval.families import CALM, SPARSE_WIDE, build_family_config
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.runner import run_episode

REPO = Path(__file__).resolve().parents[2]
CKPTS = {
    "producer": REPO / "experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt",
    "orderer": REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"}


def _lfs_present(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs_ckpts = pytest.mark.skipif(not all(_lfs_present(p) for p in CKPTS.values()),
                                 reason="git-lfs checkpoints absent")


def _world(producers: int = 12, fanout: int = 2) -> Configuration:
    return build_family_config(1000, family=SPARSE_WIDE, producers=producers, regime=CALM,
                               fanout=fanout)


# --- slope and flag ---


def test_slope_is_one_on_linear_and_two_on_quadratic() -> None:
    xs = [2.0, 4.0, 7.0, 14.0]
    assert dc.loglog_slope(xs, [3e-4 * x for x in xs]) == pytest.approx(1.0)
    assert dc.loglog_slope(xs, [5e-6 * x * x for x in xs]) == pytest.approx(2.0)
    assert dc.loglog_slope(xs, [7e-4] * 4) == pytest.approx(0.0)


def test_slope_refuses_degenerate_series() -> None:
    with pytest.raises(ValueError):
        dc.loglog_slope([1.0], [1.0])
    with pytest.raises(ValueError):
        dc.loglog_slope([2.0, 2.0], [1.0, 3.0])
    with pytest.raises(ValueError):
        dc.loglog_slope([1.0, 2.0], [0.0, 1.0])


def test_flag_only_above_threshold_on_fanout_and_consumer_axes() -> None:
    assert dc.superlinear("fanout", 1.11) and dc.superlinear("consumers_per_producer", 2.0)
    assert not dc.superlinear("fanout", 1.1) and not dc.superlinear("fanout", 1.0)
    assert not dc.superlinear("producers", 2.0)  # P axis: reported, never flagged


def _synthetic_cells(overrides: dict[str, Callable[[dc.Cell], float]]) -> dict[str, dict]:
    """Every cells.csv measure column at 1e-4 per cell, except `overrides` (column -> f(cell))."""
    out = {}
    for c in dc.grid():
        row = {col: 1e-4 for col in dc.CELL_COLUMNS if col.endswith("_s")}
        row.update({col: f(c) for col, f in overrides.items()})
        out[c.id] = row
    return out


def test_slope_rows_flag_superlinear_stage() -> None:
    cells = _synthetic_cells({"obs_per_decision_s": lambda c: 1e-6 * c.fanout ** 2,
                              "decode_per_decision_s":
                                  lambda c: 1e-5 * c.consumers_per_producer ** 1.5,
                              "domain_per_cycle_s": lambda c: 1e-6 * c.fanout ** 3})
    rows = {(r["axis"], r["measure"], r["stage"]): r for r in dc.slope_rows(cells)}
    assert rows[("fanout", "decision", dc.OBS)]["slope"] == pytest.approx(2.0)
    assert rows[("fanout", "decision", dc.OBS)]["flagged"] == "Superlinear stage"
    assert rows[("consumers_per_producer", "decision", dc.DECODE)]["slope"] == pytest.approx(1.5)
    assert rows[("consumers_per_producer", "decision", dc.DECODE)]["flagged"] == "Superlinear stage"
    assert rows[("consumers_per_producer", "decision", dc.OBS)]["flagged"] == ""  # E fixed at 4
    assert rows[("fanout", "decision", dc.FORWARD)]["slope"] == pytest.approx(0.0)
    assert not any(r["flagged"] for (axis, _, _), r in rows.items() if axis == "producers")
    # per cycle: reported, never flagged
    assert rows[("fanout", "cycle", dc.DOMAIN)]["slope"] == pytest.approx(3.0)
    assert rows[("fanout", "cycle", dc.DOMAIN)]["flagged"] == ""


def test_per_role_row_flags_what_pooling_dilutes() -> None:
    """Review concern 1: a producer stage at slope 1.5 per producer decision, pooled over a
    decision count dominated by orderers that grows ~E, shows well under 1.1 pooled; the
    per-role row still flags it."""
    def prod_obs(c: dc.Cell) -> float:
        return 1e-5 * c.fanout ** 1.5

    def pooled_obs(c: dc.Cell) -> float:  # 42 producer + ~50·E orderer decisions per cycle
        return prod_obs(c) * 42 / (42 + 50 * c.fanout)

    cells = _synthetic_cells({"prod_obs_per_decision_s": prod_obs,
                              "obs_per_decision_s": pooled_obs})
    rows = {(r["axis"], r["measure"], r["stage"]): r for r in dc.slope_rows(cells)}
    assert rows[("fanout", "decision", dc.OBS)]["slope"] < 1.1
    assert rows[("fanout", "decision", dc.OBS)]["flagged"] == ""
    assert rows[("fanout", "prod_decision", dc.OBS)]["slope"] == pytest.approx(1.5)
    assert rows[("fanout", "prod_decision", dc.OBS)]["flagged"] == "Superlinear stage"
    assert dc.superlinear("fanout", 1.5, "order_decision")
    assert not dc.superlinear("fanout", 1.5, "cycle")


def _timing_row(cell: str, seed: int, rep: int, scale: float) -> dict[str, Any]:
    """A synthetic timings.csv row: every stage `scale` s, 100 decisions (40 prod, 60
    order), 10 cycles; the role columns split each policy stage 40 / 60."""
    c = dc.parse_cell(cell)
    row: dict[str, Any] = {"cell": cell, "producers": c.producers, "fanout": c.fanout,
                           "consumer_multiplier": c.consumer_multiplier, "world_seed": seed,
                           "repeat": rep, "decisions": 100, "prod_decisions": 40,
                           "order_decisions": 60, "cycles": 10, "wall_s": 5 * scale}
    row.update({f"{s}_s": scale for s in dc.STAGES})
    for s in dc.POLICY_STAGES:
        row[f"prod_{s}_s"], row[f"order_{s}_s"] = 0.4 * scale, 0.6 * scale
    return row


def test_cell_table_is_median_over_repeats_then_mean_over_worlds() -> None:
    """Review concern 2: 3 worlds (mean 3 != median 2) x 3 repeats with an outlier repeat;
    then the slope."""
    rows = []
    for cell, base in (("P42-E2-k1", 1.0), ("P42-E4-k1", 4.0)):
        for seed, w in ((1000, 1.0), (1001, 2.0), (1002, 6.0)):
            for rep, outlier in ((0, 1.0), (1, 1.0), (2, 100.0)):
                rows.append(_timing_row(cell, seed, rep, base * w * outlier))
    table = dc.cell_table(rows)
    e2 = table["P42-E2-k1"]
    assert e2["worlds"] == 3
    assert e2["obs_per_decision_s"] == pytest.approx(3.0 / 100)  # outlier ignored, mean of 1 2 6
    assert e2["total_per_decision_s"] == pytest.approx(5 * 3.0 / 100)
    assert e2["obs_per_cycle_s"] == pytest.approx(3.0 / 10)
    assert e2["prod_obs_per_decision_s"] == pytest.approx(0.4 * 3.0 / 40)
    assert e2["order_total_per_decision_s"] == pytest.approx(3 * 0.6 * 3.0 / 60)
    assert e2["decisions"] == 100 and e2["cycles"] == 10
    rows_ = {(r["axis"], r["measure"], r["stage"]): r for r in dc.slope_rows(table)}
    assert rows_[("fanout", "decision", dc.OBS)]["slope"] == pytest.approx(2.0)  # 4x at 2x E
    assert rows_[("fanout", "decision", dc.OBS)]["points"] == 2


# --- grid and the 42-producer cap ---


def test_grid_is_the_d7_cells() -> None:
    ids = [c.id for c in dc.grid()]
    assert ids == ["P12-E2-k1", "P24-E2-k1", "P42-E2-k1", "P42-E4-k1", "P42-E7-k1",
                   "P42-E14-k1", "P42-E4-k2", "P42-E4-k4"]
    assert max(c.producers for c in dc.grid()) == 42
    assert [c.consumers for c in dc.axes()["consumers_per_producer"]] == [11, 21, 42]


@pytest.mark.parametrize("producers", [45, 120])
def test_larger_economy_refused_naming_the_size(producers: int) -> None:
    with pytest.raises(ValueError, match=f"Larger economy requested: {producers} producers"):
        dc.make_cell(producers, 2)
    with pytest.raises(ValueError, match=f"{producers} producers"):
        dc.parse_cell(f"P{producers}-E2-k1")


def test_cli_refuses_larger_economy_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Larger economy requested: 120 producers"):
        dc.main(["--prod-ckpt", "missing.pt", "--orderer-ckpt", "missing.pt", "--dry-run",
                 "--out-dir", str(tmp_path / "x"), "--cells", "P120-E2-k1"])
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize("flag, kwargs", [("--cells", {"cells": ["P12-E2-k1"]}),
                                           ("--seeds", {"seeds": 1}),
                                           ("--repeats", {"repeats": 1})])
def test_real_run_needs_commit_full_grid_and_pins(tmp_path: Path, flag: str,
                                                  kwargs: dict[str, Any]) -> None:
    for name in ("p.pt", "o.pt"):
        (tmp_path / name).write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="not a valid PoC-B run") as e:
        dc.run_deployed_cost(out_dir=tmp_path / "out", prod_ckpt=str(tmp_path / "p.pt"),
                             orderer_ckpt=str(tmp_path / "o.pt"), horizon=4, **kwargs)
    msg = str(e.value)
    assert "source commit" in msg and "horizon 4" in msg and f"{flag} is set" in msg
    assert "producer checkpoint" in msg and "orderer checkpoint" in msg
    assert not (tmp_path / "out").exists()


def test_workers_capped_at_vcpu_minus_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)))
    monkeypatch.setattr(dc, "CGROUP_ROOT", tmp_path / "no-cgroup")
    assert dc.max_workers() == 6
    with pytest.raises(ValueError, match=r"--workers must be in \[1, 6\]"):
        dc.run_deployed_cost(out_dir=tmp_path, prod_ckpt="p", orderer_ckpt="o", workers=7,
                             dry_run=True)


def test_vcpu_is_affinity_capped_by_cgroup_quota(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Review concern 3: a container reporting the host's 64 CPUs, an affinity mask of 16 and
    a cgroup v2 quota of 8 CPUs has 8 usable vCPU."""
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)))
    (tmp_path / "cpu.max").write_text("800000 100000\n")
    counts = dc.cpu_counts(tmp_path)
    assert counts == {"os_cpu_count": 64, "affinity": 16, "cgroup_cpu_quota": 8.0,
                      "cgroup_cpu_raw": "800000 100000", "vcpu": 8}
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert dc.cpu_counts(tmp_path)["vcpu"] == 16
    (tmp_path / "cpu.max").unlink()
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("250000\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert dc.cpu_counts(tmp_path)["vcpu"] == 2  # v1: 2.5 CPUs rounded down


# --- timers ---


def _baselines() -> tuple[Policy, Policy]:
    return PRODUCING_BASELINES["entity_planner"], ORDERING_BASELINES["proportional"]


def _untrained_portable() -> tuple[tuple[Policy, Policy], dict[str, Any]]:
    pytest.importorskip("torch")
    from tokenomics.train.orderer_scorer import OrdererScoreModel, scorer_ordering_policy
    from tokenomics.train.producer_scorer import ProducerScoreModel, scorer_producing_policy

    prod, order = ProducerScoreModel(), OrdererScoreModel()
    return ((scorer_producing_policy(prod), scorer_ordering_policy(order)),
            {dc.PROD: prod, dc.ORDER: order})


def _assert_uninstrumented(env: Any, models: dict[str, Any]) -> None:
    from tokenomics.env.decode import decode_ordering, decode_producing

    assert "observe" not in env.__dict__ and "reset" not in env.__dict__
    assert not set(dc._SYSTEM_STAGES) & set(env.system.__dict__)
    assert aec_env.decode_producing is decode_producing
    assert aec_env.decode_ordering is decode_ordering
    assert all("forward_view" not in m.__dict__ for m in models.values())


def test_timed_and_untimed_episodes_match_baselines() -> None:
    cfg = _world()
    prod, order = _baselines()
    plain = run_episode(joint_stack_env(cfg, 6), prod, order, 0)
    env = joint_stack_env(cfg, 6)
    timed, seconds, counts = dc.timed_episode(env, prod, order, 0)
    assert timed == plain
    assert seconds[f"{dc.FORWARD}_s"] == 0.0  # no model in a baseline stack
    assert counts["prod_decisions"] == 12 * 6 and counts["order_decisions"] > 0
    _assert_uninstrumented(env, {})  # the instrumentation is gone afterwards


def test_timed_and_untimed_episodes_match_portable() -> None:
    (prod, order), models = _untrained_portable()
    cfg = _world(12, 4)
    plain = run_episode(joint_stack_env(cfg, 6), prod, order, 0)
    env = joint_stack_env(cfg, 6)
    timed, seconds, _ = dc.timed_episode(env, prod, order, 0, models)
    assert timed == plain
    assert all(seconds[f"{s}_s"] > 0 for s in dc.STAGES)
    for s in dc.POLICY_STAGES:  # every policy-side second is charged to exactly one role
        assert seconds[f"prod_{s}_s"] > 0 and seconds[f"order_{s}_s"] > 0
        assert seconds[f"prod_{s}_s"] + seconds[f"order_{s}_s"] == pytest.approx(
            seconds[f"{s}_s"], rel=1e-9)
    _assert_uninstrumented(env, models)


def test_instrumentation_removed_when_a_policy_raises() -> None:
    """Review concern 2: restoration sits in `finally`."""
    (prod, order), models = _untrained_portable()
    calls = {"n": 0}

    def failing(env: Any, agent: str) -> Any:
        calls["n"] += 1
        if calls["n"] > 20:
            raise RuntimeError("boom")
        return order(env, agent)

    env = joint_stack_env(_world(), 6)
    with pytest.raises(RuntimeError, match="boom"):
        dc.timed_episode(env, prod, failing, 0, models)
    _assert_uninstrumented(env, models)


def test_instrumentation_removed_after_a_partial_install() -> None:
    """Review nit 4a: a model without `forward_view` fails the install; nothing leaks."""
    class NoForward:
        pass

    env = joint_stack_env(_world(), 4)
    env.reset(seed=0)
    with pytest.raises(AttributeError, match="forward_view"):
        with dc.instrumented(env, dc.StageTimer(), {dc.PROD: NoForward()}):
            pass
    _assert_uninstrumented(env, {})


def test_stage_timers_sum_to_episode_wall() -> None:
    (prod, order), models = _untrained_portable()
    for policies, ms in (((prod, order), models), (_baselines(), ())):
        _, seconds, _ = dc.timed_episode(joint_stack_env(_world(), 8), *policies, 0, ms)
        total = sum(seconds[f"{s}_s"] for s in dc.STAGES)
        assert abs(total - seconds["wall_s"]) <= 0.05 * seconds["wall_s"]
        assert min(seconds[f"{s}_s"] for s in dc.STAGES) >= 0.0


def test_timer_charges_nested_calls_to_the_innermost_stage() -> None:
    timer = dc.StageTimer()
    inner = timer.wrap(dc.FORWARD, lambda: time.sleep(0.02))
    outer = timer.wrap(dc.DECODE, lambda: (time.sleep(0.01), inner()))
    timer.start()
    outer()
    timer.stop()
    assert timer.totals[dc.FORWARD] >= 0.02
    assert 0.01 <= timer.totals[dc.DECODE] < 0.02
    assert inner() is None  # outside start/stop: pass-through, nothing charged


# --- dry run ---


@needs_ckpts
def test_dry_run_writes_the_layout(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    out = tmp_path / "EXP-dry"
    t0 = time.perf_counter()
    code = dc.main(["--prod-ckpt", str(CKPTS["producer"]), "--orderer-ckpt",
                    str(CKPTS["orderer"]), "--out-dir", str(out), "--dry-run",
                    "--cells", "P12-E2-k1", "--seeds", "1", "--repeats", "1", "--horizon", "4",
                    "--workers", "1"])
    assert code == 0 and time.perf_counter() - t0 < 60
    for name in ("spec.json", "timings.csv", "cells.csv", "slopes.csv", "RESULT.md"):
        assert (out / name).exists(), name
    spec = json.loads((out / "spec.json").read_text())
    assert spec["kind"] == "dry-run" and spec["pins"]["checkpoints"] == dc.CHECKPOINTS
    assert spec["host"]["vcpu"] and "python" in spec["host"]
    rows = list(csv.DictReader((out / "timings.csv").open()))
    assert len(rows) == 1 and rows[0]["cell"] == "P12-E2-k1" and rows[0]["cycles"] == "4"
    assert "DRY RUN" in (out / "RESULT.md").read_text()
    with pytest.raises(ValueError, match="already holds a run"):
        dc.main(["--prod-ckpt", str(CKPTS["producer"]), "--orderer-ckpt",
                 str(CKPTS["orderer"]), "--out-dir", str(out), "--dry-run",
                 "--cells", "P12-E2-k1", "--seeds", "1", "--repeats", "1", "--horizon", "4"])


def _sleeper(seconds: float) -> Callable[..., None]:
    def fake(*args: Any, **kwargs: Any) -> None:
        time.sleep(seconds)
    return fake


class _SlowModel:
    """A model stand-in whose `forward_view` only sleeps."""

    def forward_view(self, view: Any) -> None:
        time.sleep(0.02)


def test_every_wrapper_charges_its_own_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review round 2, concern 1: each role-carrying call lands in its own `by_role` cell
    (observe by the agent's role); `domain` and `harness` carry no role."""
    from tokenomics.env.aec_env import TokenomicsAECEnv

    env = joint_stack_env(_world(), 4)
    env.reset(seed=0)
    system_type = type(env.system)
    for name in dc._SYSTEM_STAGES:
        monkeypatch.setattr(system_type, name, _sleeper(0.02))
    monkeypatch.setattr(TokenomicsAECEnv, "observe", _sleeper(0.02))
    monkeypatch.setattr(aec_env, "decode_producing", _sleeper(0.02))
    monkeypatch.setattr(aec_env, "decode_ordering", _sleeper(0.02))
    prod_agent = next(a for a in env.possible_agents if env._role[a][0] == dc.PROD)
    order_agent = next(a for a in env.possible_agents if env._role[a][0] == dc.ORDER)
    timer = dc.StageTimer()
    models = {dc.PROD: _SlowModel(), dc.ORDER: _SlowModel()}
    calls: list[tuple[str, Callable[[], Any], str, str | None]] = [
        ("observe prod", lambda: env.observe(prod_agent), dc.OBS, dc.PROD),
        ("observe order", lambda: env.observe(order_agent), dc.OBS, dc.ORDER),
        ("producer_entity_view", lambda: env.system.producer_entity_view(), dc.OBS, dc.PROD),
        ("product_stream_view", lambda: env.system.product_stream_view(), dc.OBS, dc.ORDER),
        ("producing_agent_action", lambda: env.system.producing_agent_action(), dc.DECODE,
         dc.PROD),
        ("ordering_agent_action", lambda: env.system.ordering_agent_action(), dc.DECODE,
         dc.ORDER),
        ("decode_producing", lambda: aec_env.decode_producing(), dc.DECODE, dc.PROD),
        ("decode_ordering", lambda: aec_env.decode_ordering(), dc.DECODE, dc.ORDER),
        ("prod forward", lambda: models[dc.PROD].forward_view(None), dc.FORWARD, dc.PROD),
        ("order forward", lambda: models[dc.ORDER].forward_view(None), dc.FORWARD, dc.ORDER),
        ("start_ordering", lambda: env.system.start_ordering(), dc.DOMAIN, None),
        ("complete_cycle", lambda: env.system.complete_cycle(), dc.DOMAIN, None),
        # harness time is charged at the next timer event: flush with a role-free no-op
        ("harness", lambda: (time.sleep(0.02), flush()), dc.HARNESS, None),
    ]
    flush = timer.wrap(dc.DOMAIN, lambda: None)
    with dc.instrumented(env, timer, models):
        env.reset(seed=0)  # installs the System wrappers
        timer.start()
        for label, call, stage, role in calls:
            totals, by_role = dict(timer.totals), dict(timer.by_role)
            call()
            d_stage = {k: timer.totals[k] - totals[k] for k in totals}
            d_role = {k: timer.by_role[k] - by_role[k] for k in by_role}
            assert d_stage[stage] >= 0.02, label
            assert max(v for k, v in d_stage.items() if k != stage) < 0.01, label
            if role is None:
                assert max(d_role.values()) < 0.01, label
            else:
                assert d_role[(role, stage)] >= 0.02, label
                assert max(v for k, v in d_role.items() if k != (role, stage)) < 0.01, label
        timer.stop()


def test_run_job_maps_each_model_to_its_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review rounds 2-3, concern 1: `run_job` hands the worker's producer model the producer
    role and its orderer model the orderer role (identity check, no timing)."""
    prod_model, order_model = object(), object()
    seen: dict[str, Any] = {}

    def capture(env: Any, prod: Policy, order: Policy, seed: int,
                models: dict[str, Any] | None = None) -> tuple[dict, dict, dict]:
        seen.update(models or {})
        stats = {"cycles": 1, "dead": False, "consumer_completed_units": 0}
        seconds = {f"{s}_s": 0.0 for s in dc.STAGES} | {"wall_s": 0.0}
        return stats, seconds, {"prod_decisions": 1, "order_decisions": 1}

    import tokenomics.train.orderer_scorer as orderer_scorer
    import tokenomics.train.producer_scorer as producer_scorer

    monkeypatch.setattr(dc, "timed_episode", capture)
    monkeypatch.setattr(producer_scorer, "scorer_producing_policy", lambda m: m)
    monkeypatch.setattr(orderer_scorer, "scorer_ordering_policy", lambda m: m)
    monkeypatch.setitem(dc._W, "models", (prod_model, order_model))
    dc.run_job({"producers": 12, "fanout": 2, "consumer_multiplier": 1, "world_seed": 1000,
                "repeat": 0, "episode_seed": 0, "horizon": 2})
    assert seen[dc.PROD] is prod_model and seen[dc.ORDER] is order_model


def test_flag_setting_has_one_source() -> None:
    """Review round 2, concern 2: `superlinear` flags exactly the measures `MEASURES` marks."""
    for m, (_, _, _, flagged) in dc.MEASURES.items():
        assert dc.superlinear("fanout", 2.0, m) is flagged
    assert dc.FLAGGED_MEASURES == tuple(m for m, v in dc.MEASURES.items() if v[3])


def test_cgroup_quota_read_from_own_nested_cgroup(tmp_path: Path) -> None:
    """Review round 2, 3c: the v2 quota of this process's cgroup, not the root's."""
    nested = tmp_path / "kubepods" / "pod1"
    nested.mkdir(parents=True)
    (tmp_path / "cpu.max").write_text("max 100000\n")
    (nested / "cpu.max").write_text("400000 100000\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/kubepods/pod1\n")
    assert dc.cgroup_cpu_limit(tmp_path, proc) == (4.0, "400000 100000")
    proc.write_text("0::/elsewhere\n")  # no cpu.max there: fall back to the root
    assert dc.cgroup_cpu_limit(tmp_path, proc) == (None, "max 100000")


def test_cgroup_permission_error_reads_as_unknown(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    real = Path.read_text

    def denied(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name in ("cpu.max", "proc_cgroup"):
            raise PermissionError(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    assert dc.cgroup_cpu_limit(tmp_path, tmp_path / "proc_cgroup") == (None, None)
