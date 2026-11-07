"""4.4 / 4.5 (m6_onpolicy, m6_stage0c): Plackett–Luce, the per-decision log-probability, the greedy limit = the
deployed policy, the PPO ratio at the start, the F40 composite, calibration, perturbation and save / load."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenomics.train import m6_onpolicy as op  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
E = REPO / "experiments"
TURNS = E / "EXP-0126-m6-turn-imitation-v15/artifacts/models.pt"
START = E / "EXP-0110-m6-stage0/artifacts/start_models.pt"
IT40 = E / "EXP-0114-m6-pilot-continued/artifacts/candidate_it40.pt"
RECS = E / "EXP-0126-m6-turn-imitation-v15/artifacts/heldout_records.npz"
TRAINING = E / "EXP-0058-es-gradient-probe/bar"


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs = pytest.mark.skipif(not (_lfs(TURNS) and _lfs(START) and (TRAINING / "training_worlds.json").exists()),
                           reason="records absent")


def test_plackett_luce_is_a_distribution_and_the_sampler_follows_it() -> None:
    s = torch.tensor([1.0, 0.3, -0.5, 2.0])
    total = sum(math.exp(float(op.pl_logp(s, list(p), 0.7))) for p in itertools.permutations(range(4)))
    assert total == pytest.approx(1.0, abs=1e-6)
    rng = np.random.default_rng(0)
    top = np.bincount([op.pl_sample(s.numpy(), 0.7, rng)[0] for _ in range(20000)], minlength=4) / 20000
    assert np.allclose(top, torch.softmax(s / 0.7, 0).numpy(), atol=0.015)


def test_rank_scores_and_learnable_adjacency() -> None:
    order = [2, 0, 3, 1]
    assert list(np.argsort(-op.rank_scores(order, 4), kind="stable")) == order
    rep = np.array([True, True, False, True])
    adj = dict(op.learnable_adjacent(np.array([3.0, 2.0, 1.0, 0.0]), rep))
    assert adj == {0: 2, 1: 2, 2: 3}  # 0 and 1 (both replenishment) are never compared; 3 is last
    assert op.swaps([2, 0, 1, 3], list(adj.items())) == 2


def _setup(horizon: int = 6) -> tuple:
    from tokenomics.train import m6_imitation as im
    from tokenomics.train import m6_turn_imitation as ti

    model, om = op.load_start(str(TURNS), str(START))
    model0, _ = op.load_start(str(TURNS), str(START))
    ts, train, _mon = im._training(str(TRAINING))
    job = ti._jobs(ts, train[:1], "t", "x", horizon)[0]
    return model, model0, om, job


@needs
def test_recorded_log_probability_equals_the_recomputed_one_and_ppo_starts_at_ratio_one() -> None:
    from tokenomics.eval.runner import run_episode

    model, model0, om, job = _setup()
    temps = op.Temps(replenish_sigma=0.3, fund_tau=2.0, order_tau=1.0, accept_t=3.0)
    env = op.turn_env(job)
    pol = op.OnPolicy(model, model0, om, np.random.default_rng(1), temps)
    run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None,
                fund_policy=pol.fund, accept_policy=pol.accept)
    kinds = {d["kind"] for d in pol.decisions}
    assert kinds == {"prod", "fund", "acc"}
    with torch.no_grad():
        err = max(abs(float(op.decision_logp(model, d, temps)[0]) - d["old_lp"]) for d in pol.decisions)
    assert err < 1e-4
    batch = [{**d, "adv": 1.0} for d in pol.decisions[:80]]
    _loss, st = op.ppo_loss(model, model0, batch, temps)
    assert st["pg"] == pytest.approx(-1.0, abs=1e-4) and abs(st["kl"]) < 1e-4 and st["clip_frac"] == 0.0


@needs
def test_the_greedy_limit_is_the_deployed_policy() -> None:
    """At vanishing temperatures the training policy plays exactly the deployed one: the composite's greedy gates
    and replenish, and the turn heads' greedy fund order, accept order and accept bits."""
    from tokenomics.eval.runner import run_episode
    from tokenomics.train import m6_gnn as g

    model, model0, om, job = _setup(8)
    cold = op.Temps(gates=(1e-9, 1e-9, 1e-9), replenish_sigma=0.0, fund_tau=1e-9, order_tau=1e-9, accept_t=1e-9)
    env = op.turn_env(job)
    pol = op.OnPolicy(model, model0, om, np.random.default_rng(2), cold)
    a = run_episode(env, pol.producing, pol.ordering, seed=job["episode"], action_seed=None,
                    fund_policy=pol.fund, accept_policy=pol.accept)
    env2 = op.turn_env(job)
    fp, ap = g.gnn_turn_policies(model.pm)
    b = run_episode(env2, op.greedy_producing_policy(model), g.gnn_ordering_policy(om), seed=job["episode"],
                    action_seed=None, fund_policy=fp, accept_policy=ap)
    assert a == b


@needs
@pytest.mark.skipif(not _lfs(IT40), reason="records absent")
def test_the_composite_starts_as_f40() -> None:
    """The composite's deployed investment action = F40's (candidate_it40's greedy gates + the start's replenish,
    EXP-0116's start_rep)."""
    from tokenomics.train import m6_gap_probe as gp
    from tokenomics.train import m6_imitation as im

    model, _m0, _om, job = _setup(4)
    pm40, _ = im.load_models(str(IT40))
    pm0, _ = im.load_models(str(START))
    env = op.turn_env(job)
    env.reset(seed=job["episode"])
    f40 = gp._policy("start_rep", pm40, pm0, np.random.default_rng(0))
    mine = op.greedy_producing_policy(model)
    for agent in env.possible_agents:
        if env._role[agent][0] == "prod":
            assert np.allclose(f40(env, agent), mine(env, agent), atol=1e-6)


@needs
def test_only_the_trainable_parameters_move_and_the_model_round_trips(tmp_path: Path) -> None:
    from tokenomics.train import m6_stage0c as sc

    model, _m0, om, _job = _setup()
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert any(n.startswith("rep.trunk") or n.startswith("rep.enc") for n in frozen)
    assert all(not n.startswith("rep.replenish") for n in frozen) and "pm.replenish.weight" in frozen
    moved = sc.perturbed(model, 0, 0.5)
    for n, p in moved.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n])
    assert any(not torch.equal(p, q) for p, q in zip(moved.trainable(), model.trainable()))
    op.save_model(model, om, tmp_path / "m.pt")
    back, _ = op.load_model(tmp_path / "m.pt")
    assert all(torch.equal(v, back.state_dict()[k]) for k, v in model.state_dict().items()
               if not k.startswith("pm.replenish."))
    assert torch.isnan(back.pm.replenish.weight).all()  # the unused head is a sentinel
    from tokenomics.train import m6_imitation as im

    with pytest.raises(ValueError, match="m6_onpolicy.load_model"):
        im.load_models(tmp_path / "m.pt")
    norep = sc.perturbed(model, 0, 0.5, norep=True)
    for (n, p), q in zip(norep.named_parameters(), model.parameters()):
        if n.startswith("rep.replenish."):
            assert torch.equal(p, q)


@needs
@pytest.mark.skipif(not _lfs(RECS), reason="records absent")
def test_calibration_hits_its_targets() -> None:
    from tokenomics.train import m6_gnn as g
    from tokenomics.train import m6_turn_imitation as ti

    model, _m0, _om, job = _setup()
    env = op.turn_env(job)
    env.reset(seed=job["episode"])
    feats = []
    for agent in env.possible_agents:
        if env._role[agent][0] == "prod":
            pid = env._role[agent][1]
            feats.append(g.producer_features(env.system.producer_entity_view(pid), g.producer_context(env, pid),
                                             model.features))
    cal = op.calibrate(model, feats, ti.load_records(RECS, 400))
    ach = cal["achieved"]
    assert ach["replenish_mean_abs_dfrac"] == pytest.approx(0.02, rel=0.01)
    assert ach["fund_swap"] == pytest.approx(0.05, abs=0.01) and ach["order_swap"] == pytest.approx(0.05, abs=0.01)
    assert ach["accept_flip"] == pytest.approx(0.05, rel=0.01)


@needs
def test_one_iteration_times_on_real_data() -> None:
    from tokenomics.train import m6_turn_imitation as ti

    model, model0, om, job = _setup(4)
    temps = op.Temps(replenish_sigma=0.3, fund_tau=2.0, order_tau=1.0, accept_t=3.0)
    jobs = [{**job, "iteration": 1, "weights": {"lam": 0.18, "mu": 0.09, "nu": 4.0, "x": 0.75, "y": 0.73},
             "unit_scale": 1.0, "turn_models": str(TURNS), "replenish_models": str(START)}]
    t = op.time_iteration(model, model0, om, jobs, temps, workers=1, steps=2)
    assert t["decisions"] > 0 and t["seconds_per_step"] > 0 and t["episodes"] == op.CFG["siblings"]
    assert set(t["kinds"]) == {"prod", "fund", "acc"} and ti is not None


def test_plackett_luce_kl_is_exact_nonnegative_and_pulls_toward_the_start() -> None:
    torch.manual_seed(0)
    s0 = torch.randn(5)
    order = [3, 0, 4, 1, 2]
    assert float(op.pl_kl(s0.clone(), s0, order, 0.8)) == pytest.approx(0.0, abs=1e-7)
    for _ in range(20):
        s = (s0 + torch.randn(5)).requires_grad_(True)
        kl = op.pl_kl(s, s0, order, 0.8)
        assert float(kl) >= 0
        kl.backward()
        with torch.no_grad():
            closer = s - 0.05 * s.grad
        assert float(op.pl_kl(closer, s0, order, 0.8)) < float(kl)
    # 2 items: the single stage is the Bernoulli KL of the first pick
    a, b = torch.tensor([0.7, -0.2]), torch.tensor([0.1, 0.4])
    p, q = torch.softmax(a, 0), torch.softmax(b, 0)
    assert float(op.pl_kl(a, b, [0, 1], 1.0)) == pytest.approx(float((p * (p / q).log()).sum()), abs=1e-6)


@needs
def test_the_deployed_checkpoint_path_plays_the_greedy_training_path(tmp_path: Path) -> None:
    """Gates and validation deploy only via `load_model` + `deployed_policies`: after a save / load round trip that
    path plays the same episode as the training policy at vanishing temperatures."""
    from tokenomics.eval.runner import run_episode

    model, model0, om, job = _setup(8)
    op.save_model(model, om, tmp_path / "m.pt")
    back, bom = op.load_model(tmp_path / "m.pt")
    cold = op.Temps(gates=(1e-9, 1e-9, 1e-9), replenish_sigma=0.0, fund_tau=1e-9, order_tau=1e-9, accept_t=1e-9)
    pol = op.OnPolicy(model, model0, om, np.random.default_rng(2), cold)
    a = run_episode(op.turn_env(job), pol.producing, pol.ordering, seed=job["episode"], action_seed=None,
                    fund_policy=pol.fund, accept_policy=pol.accept)
    dp = op.deployed_policies(back, bom)
    b = run_episode(op.turn_env(job), dp["producing"], dp["ordering"], seed=job["episode"], action_seed=None,
                    fund_policy=dp["fund"], accept_policy=dp["accept"])
    assert a == b
    assert pol.counts["flip_restore"][1] > 0 and pol.counts["flip_restore"][0] == 0  # per-gate counters


@needs
def test_the_first_step_check_catches_rollouts_played_by_another_model() -> None:
    from tokenomics.eval.runner import run_episode
    from tokenomics.train import m6_stage0c as sc

    model, model0, om, job = _setup(4)
    temps = op.Temps(replenish_sigma=0.3, fund_tau=2.0, order_tau=1.0, accept_t=3.0)
    pol = op.OnPolicy(model, model0, om, np.random.default_rng(3), temps)
    run_episode(op.turn_env(job), pol.producing, pol.ordering, seed=job["episode"], action_seed=None,
                fund_policy=pol.fund, accept_policy=pol.accept)
    assert op.check_first_step(model, pol.decisions, temps) < 1e-3
    with pytest.raises(RuntimeError, match="not played by the current model"):
        op.check_first_step(sc.perturbed(model, 0, 2.0), pol.decisions, temps)
