"""M5 task 2.1: the producer residual on rf3 (design D2): theta = 0 is the base action for action at
12 and 42 producers; the parameter count does not depend on size; each correction moves its
decision; the policy reads only the producer's own view and history (ADR-0058); no episode state
carries over."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokenomics.eval import baselines as b
from tokenomics.train import producer_residual as pr

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parents[1]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
W12, W42 = "ws-atlas-v1-12-0111", "ws-atlas-v1-42-0013"


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs_atlas = pytest.mark.skipif(not _lfs(ATLAS_V2), reason="EXP-0052 atlas absent")


def _episode(policy, world: str, horizon: int, proxy: bool = False) -> list[bytes]:
    """The producing actions of one EXP-0057-path episode (MWWk r0); `proxy` hides every system
    attribute but the ones a producer policy may read."""
    from tokenomics.eval.decoder_check import (MWWK, decoder_arm, decoder_policies,
                                               job_draw_key, select_views, world_entry)
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster
    from tokenomics.eval.routing_audit import build_jobs, job_config
    from tokenomics.eval.routing_diagnostics import set_draw_key
    from tokenomics.eval.runner import run_episode
    from tokenomics.eval.teacher_audit import load_atlas

    view = select_views(load_atlas(ATLAS_V2), [world])[0]
    arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
    job = [j for j in build_jobs([world_entry(view)], [arm], 1, horizon, None, {})
           if j["replicate"] == 0][0]
    env = joint_stack_env(job_config(job), horizon)
    set_draw_key(env, job_draw_key(job))
    seen: list[bytes] = []

    def recording(e, agent):
        if proxy:
            e = _Restricted(e)
        a = policy(e, agent)
        seen.append(np.asarray(a).tobytes())
        return a

    run_episode(env, recording, decoder_policies(job)[1], seed=job["episode"],
                action_seed=job["action_seed"])
    return seen


class _System:
    """Exposes only what a producer policy may read: its own entity view, the cycle counter and
    the fund level (the base's existing affordability gate). Anything else raises."""

    ALLOWED = ("producer_entity_view", "cycle_counter", "investment_fund")

    def __init__(self, system: object) -> None:
        object.__setattr__(self, "_s", system)

    def __getattr__(self, name: str) -> object:
        if name not in self.ALLOWED:
            raise AttributeError(f"producer policy read system.{name}")
        return getattr(object.__getattribute__(self, "_s"), name)


_PROXIES: dict = {}


class _Restricted:
    def __init__(self, env: object) -> None:
        self._env = env
        s = env.system  # one proxy per system so weak-keyed state sees a stable key
        self.system = _PROXIES.setdefault(id(s), (s, _System(s)))[1]
        self._role, self.spec_ = env._role, env.spec_


@needs_atlas
@pytest.mark.parametrize("world,horizon", [(W12, 30), (W42, 15)])
def test_theta_zero_is_rf3_action_for_action(world: str, horizon: int) -> None:
    rf3 = b.EXTRA_PRODUCING[pr.BASE_PRODUCER]
    assert _episode(pr.producer_residual_policy(np.zeros(pr.THETA_DIM)), world, horizon) == \
        _episode(b.producing_entity_planner_buyahead_rf(4, blocked_first=False, keep_room=False),
                 world, horizon)
    assert rf3.rf_parts == (False, False, True)


@needs_atlas
def test_policy_reads_only_its_own_view_and_the_fund() -> None:
    theta = np.full(pr.THETA_DIM, 0.3)
    _episode(pr.producer_residual_policy(theta), W12, 10, proxy=True)  # raises on any other read


@needs_atlas
def test_no_state_carries_across_episodes() -> None:
    import gc

    theta = np.linspace(-0.4, 0.4, pr.THETA_DIM)
    shared = pr.producer_residual_policy(theta)
    first = _episode(shared, W12, 25)
    _episode(shared, W42, 10)
    assert _episode(shared, W12, 25) == first == _episode(pr.producer_residual_policy(theta),
                                                          W12, 25)
    gc.collect()
    assert len(shared.state) == 0


def test_parameter_count_is_size_independent() -> None:
    from tests.test_buyahead import _rf_view

    small, big = _rf_view(limit=6, reserved_gadgets=4), _rf_view(limit=60, reserved_gadgets=0)
    for v in (small, big):
        phi = pr.input_features(v, {})
        assert all(f.shape == (pr.N_IN,) for f in phi.values())
        assert pr.producer_features(v, {}).shape == (pr.N_P,)
    assert pr.THETA_DIM == 2 * pr.N_IN + pr.N_P == 23
    with pytest.raises(ValueError):
        pr.split_theta(np.zeros(pr.THETA_DIM + 1))


def _plan(view, theta, **kw):
    args = dict(max_replenish=100, replenish_cost_per_unit=None, funding_budget=None,
                inventory_expansion_increases=10, inventory_expansion_cost=0)
    args.update(kw)
    return pr.plan_residual(view, theta, {}, **args)


def _unit(block: str, feature: str, value: float) -> np.ndarray:
    theta = np.zeros(pr.THETA_DIM)
    off = {"k": 0, "b": pr.N_IN, "e": 2 * pr.N_IN}[block]
    names = pr.PRODUCER_FEATURES if block == "e" else pr.INPUT_FEATURES
    theta[off + names.index(feature)] = value
    return theta


def test_each_correction_moves_its_decision() -> None:
    from tests.test_buyahead import _rf_view

    view = _rf_view(limit=200, reserved_gadgets=0)  # raw use 16 / cycle, rare 0
    base = _plan(view, np.zeros(pr.THETA_DIM))
    assert base.replenishments == b.plan_reservations_first(
        view, ahead_cycles=4, max_replenish=100, replenish_cost_per_unit=None,
        funding_budget=None, kept_room=0, inventory_expansion_increases=10,
        inventory_expansion_cost=0, blocked_first=False, keep_room=False).replenishments
    assert base.replenishments == {"raw": 64, "rare": 3}  # max(3, ceil(4 x 16)) / floor 3
    bigger = _plan(view, _unit("k", "bias", float(np.log(2))))  # k = 8
    assert bigger.replenishments["raw"] == 100  # ceil(8 x 16) = 128, capped by max_replenish
    smaller = _plan(view, _unit("k", "bias", -float(np.log(4))))  # k = 1
    assert smaller.replenishments["raw"] == 16
    capped = _plan(view, _unit("k", "bias", 50.0))  # clip at 4 e^2
    assert capped.replenishments["raw"] == 100
    extra = _plan(view, _unit("b", "bias", 2.0))
    assert extra.replenishments == {"raw": 66, "rare": 5}
    assert not base.do_inventory_expansion
    assert _plan(view, _unit("e", "bias", 1.0)).do_inventory_expansion
    assert not _plan(view, _unit("e", "bias", -1.0)).do_inventory_expansion
    assert not _plan(view, _unit("e", "bias", 1.0), inventory_expansion_increases=0
                     ).do_inventory_expansion


def test_extra_buying_counts_in_a_surplus_state() -> None:
    """Review m5-2-1: with stock above the planner's target (raw 70 > 64, rare 10 > floor 3) the base
    buys nothing, and extra_g still adds extra_g units on top of that want."""
    from tests.test_buyahead import _rf_view

    view = _rf_view(limit=200, reserved_gadgets=0)
    rows = [b.replace(r, inventory={"raw": 70, "rare": 10}[r.input_product])
            for r in view.input_rows]
    view = b.replace(view, input_rows=rows)
    assert _plan(view, np.zeros(pr.THETA_DIM)).replenishments == {}
    assert _plan(view, _unit("b", "bias", 2.0)).replenishments == {"raw": 2, "rare": 2}
    assert _plan(view, _unit("b", "bias", 5.0)).replenishments == {"raw": 5, "rare": 5}
