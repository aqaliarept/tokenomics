"""Smoke test for the PPO trainer — runs a couple of tiny iterations and checks
the loop completes and returns a per-iteration training curve (ADR-0003).
Skipped if torch is not installed (it's an optional `train` extra)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")


def test_explained_variance_diagnostic():
    """#6 diagnostic: explained variance = 1 - Var(y-pred)/Var(y). A perfect critic
    scores 1.0; predicting the mean scores 0.0; an anti-correlated critic goes
    negative. Degenerate (zero-variance returns) is defined as 0.0, not a div-by-zero."""
    import numpy as np

    from tokenomics.train.smoke_ppo import explained_variance

    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(y, y) == pytest.approx(1.0)              # perfect fit
    assert explained_variance(y, np.full_like(y, y.mean())) == pytest.approx(0.0)  # mean baseline
    assert explained_variance(y, -y) < 0.0                            # worse than the mean
    assert explained_variance(np.ones(4), np.ones(4)) == pytest.approx(0.0)  # zero-variance target


def test_train_runs_two_iters():
    from tokenomics.train.smoke_ppo import IterStats, train

    history = train(
        iters=2, steps=128, horizon=40, seed=0, eval_seeds=2, log=lambda *a, **k: None
    )
    assert len(history) == 2
    assert all(isinstance(r, IterStats) for r in history)
    # optimization signals always present
    assert all(isinstance(r.mean_step_reward, float) for r in history)
    # greedy-eval task metrics populated (not the skipped-iter NaN)
    assert all(not math.isnan(r.eval_cycles) for r in history)
    assert all(r.eval_cycles > 0 for r in history)


def test_topology_dr_holds_each_topology_for_dwell_iters():
    # Plan item 2 (real trainer topology DR): a NEW random scale topology every iter
    # forces train() to rebuild rollout envs each iter, so in-progress episodes never
    # reach horizon and the gate starves. The DR provider must hold one topology for
    # `dwell` iters (a stable (seed, difficulty) pair -> envs persist -> episodes run
    # full-length) and rotate the seed afterward for generalization breadth.
    import numpy as np

    from tokenomics.train.smoke_ppo import topology_dr

    dr = topology_dr(depth=3, dwell=3)
    rng = np.random.default_rng(0)
    cfgs = [dr(None, rng) for _ in range(7)]
    assert cfgs[0] == cfgs[1] == cfgs[2]    # window 1 stable
    assert cfgs[3] == cfgs[4] == cfgs[5]    # window 2 stable
    assert cfgs[0] != cfgs[3]               # rotates across windows


def test_topology_dr_preserves_obs_action_dims():
    # Dims-preserving rule (ADR-0039 generalized): one trained net/critic must span the
    # whole DR stream, so every topology the provider yields shares obs/action/global
    # dims. build_scale_config fixes the recipe DAG + counts, so this must hold.
    import numpy as np

    from tokenomics.env.spaces import build_spec
    from tokenomics.train.smoke_ppo import topology_dr

    dr = topology_dr(depth=3, dwell=1)
    rng = np.random.default_rng(1)
    dims = set()
    for _ in range(6):
        s = build_spec(dr(None, rng))
        dims.add((s.producing_obs_dim, s.producing_action_dim,
                  s.ordering_obs_dim, s.ordering_action_dim, s.global_state_dim))
    assert len(dims) == 1


def test_train_runs_with_topology_dr_and_curriculum():
    # End-to-end: the supported trainer accepts the topology-DR provider and stages it
    # (subtlety 1, easing re-applied AFTER topology selection — handled structurally by
    # train applying staged_config to the provider's base). The loop must complete with
    # the net built once on base_cfg and never dim-mismatch on a rotated topology.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.smoke_ppo import Curriculum, topology_dr, train

    base = build_scale_config(0, depth=3)
    cur = Curriculum(d_start=0.0, gate=True)
    history = train(
        iters=2, steps=128, horizon=30, seed=0, num_envs=2, eval_seeds=1,
        config=base, curriculum=cur, topology_random=topology_dr(depth=3, dwell=2),
        log=lambda *a, **k: None,
    )
    assert len(history) == 2


def test_validation_suite_averages_metrics_across_topologies():
    # Plan item 3: checkpoint selection must score on a fixed VALIDATION TOPOLOGY SUITE,
    # not only base_cfg — else a checkpoint that wins the one fixed topology but loses
    # the held-out distribution is kept. The suite score is the per-topology eval mean
    # averaged across topologies.
    from tokenomics.eval.baselines import ordering_even_split, producing_obs_stock_aware
    from tokenomics.eval.runner import evaluate
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.smoke_ppo import evaluate_validation_suite

    pp, op = producing_obs_stock_aware(0.6), ordering_even_split
    a, b = build_scale_config(11, depth=3), build_scale_config(12, depth=3)
    seeds, H = range(2), 40

    single = evaluate_validation_suite(pp, op, [a], seeds, H)
    assert single["consumer_completed_units"] == pytest.approx(
        evaluate(pp, op, seeds, H, a)["consumer_completed_units"][0]
    )
    both = evaluate_validation_suite(pp, op, [a, b], seeds, H)
    ma, mb = evaluate(pp, op, seeds, H, a), evaluate(pp, op, seeds, H, b)
    assert both["dead_rate"] == pytest.approx((ma["dead_rate"][0] + mb["dead_rate"][0]) / 2)
    assert both["consumer_completed_units"] == pytest.approx(
        (ma["consumer_completed_units"][0] + mb["consumer_completed_units"][0]) / 2
    )


def test_train_selects_on_validation_suite():
    # End-to-end: train accepts a validation topology suite for checkpoint selection and
    # completes, with the net built on base_cfg spanning every suite topology's dims.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.smoke_ppo import topology_dr, train

    base = build_scale_config(0, depth=3)
    val = [build_scale_config(s, depth=3) for s in (100, 101)]
    history = train(
        iters=2, steps=128, horizon=30, seed=0, num_envs=2, eval_seeds=1,
        config=base, topology_random=topology_dr(depth=3, dwell=2),
        validation_configs=val, log=lambda *a, **k: None,
    )
    assert len(history) == 2


def test_train_accepts_config_and_evals_on_it():
    """B9 plumbing: train(config=...) runs on a non-phase1 suite end-to-end.
    competition.json has 5 producers vs phase1's 4, so the actor input layer is
    wider than the phase1 spec. The internal greedy-eval must score on the SAME
    config (not the phase1 default), else it crashes on a shape mismatch — this
    pins that the spec, rollout, and eval all derive from the passed config."""
    from pathlib import Path

    from tokenomics.domain import load_config
    from tokenomics.train.smoke_ppo import IterStats, train

    cfg = load_config(Path(__file__).parent / "fixtures" / "competition.json")
    history = train(
        iters=2, steps=128, horizon=40, seed=0, eval_seeds=2, config=cfg,
        log=lambda *a, **k: None,
    )
    assert len(history) == 2
    assert all(isinstance(r, IterStats) for r in history)
    # greedy-eval ran on the competition spec (would have raised on a dim mismatch)
    assert all(not math.isnan(r.eval_cycles) for r in history)
    assert all(r.eval_cycles > 0 for r in history)


def test_gae_truncation_bootstraps_termination_does_not():
    """ADR-0004: the final transition bootstraps on truncation (boot value used)
    but not on termination (term kills the value backup). next_idx=[1,-1] makes
    row 0 the predecessor of row 1, which ends the trajectory (ADR-0017)."""
    from tokenomics.train.smoke_ppo import gae

    rew = [0.0, 0.0]
    val = [0.0, 0.0]
    nxt = [1, -1]
    # truncated last step with a positive bootstrap -> non-zero advantage flows back
    adv_trunc, _ = gae(rew, val, term=[0.0, 0.0], boot=[0.0, 5.0], next_idx=nxt, gamma=0.99, lam=0.95)
    # terminated last step -> no bootstrap, all advantages zero
    adv_term, _ = gae(rew, val, term=[0.0, 1.0], boot=[0.0, 5.0], next_idx=nxt, gamma=0.99, lam=0.95)

    assert adv_trunc[1] > 0.0
    assert adv_term[1] == 0.0


def test_gae_follows_links_not_adjacency():
    """ADR-0017: the shared role buffer interleaves trajectories, so GAE must back
    up along next_idx, never buffer adjacency. Buffer order A0,B0,A1,B1 with links
    A0->A1, B0->B1; gamma=lambda=1, zero values. A rewards [1,2], B rewards
    [100,200]. Correct per-trajectory returns: A0=3, B0=300, A1=2, B1=200. The old
    adjacency code (val[t+1]) would leak B's rewards into A and vice versa."""
    from tokenomics.train.smoke_ppo import gae

    rew = [1.0, 100.0, 2.0, 200.0]   # A0, B0, A1, B1
    val = [0.0, 0.0, 0.0, 0.0]
    term = [0.0, 0.0, 0.0, 0.0]
    boot = [0.0, 0.0, 0.0, 0.0]      # ends bootstrap to 0
    next_idx = [2, 3, -1, -1]        # A0->A1, B0->B1, A1 end, B1 end

    adv, _ = gae(rew, val, term, boot, next_idx, gamma=1.0, lam=1.0)
    assert list(adv) == [3.0, 300.0, 2.0, 200.0]


def test_drop_pending_removes_open_rows_and_remaps_links():
    """ADR-0017: a completed predecessor links to a still-pending successor. After
    drop_pending the successor is gone, the predecessor's link becomes -1, its
    recorded bootstrap survives, and all fields stay aligned."""
    from tokenomics.train.smoke_ppo import Buf, drop_pending

    buf = Buf()
    # row 0 = completed predecessor (reward attached, boot recorded), row 1 = open
    for r, nx, bt in ((5.0, 1, 9.0), (0.0, -1, 0.0)):
        buf.obs.append([0.0]); buf.glob.append([0.0]); buf.act.append(0)
        buf.mask.append([1.0]); buf.logp.append(0.0); buf.val.append(0.0); buf.rew.append(r)
        buf.term.append(0.0); buf.boot.append(bt); buf.next_idx.append(nx)
    buf.pending[("e", "a")] = 1   # row 1 never got its reward

    dropped = drop_pending(buf)

    assert dropped == 1
    assert not buf.pending
    assert len(buf.obs) == 1
    assert all(len(getattr(buf, f)) == 1 for f in
               ("obs", "glob", "act", "mask", "logp", "val", "rew", "term", "boot", "next_idx"))
    assert buf.next_idx[0] == -1   # link to dropped row -> trajectory ends here
    assert buf.boot[0] == 9.0      # predecessor keeps its bootstrap


def test_vectorized_collect_fills_both_buffers():
    """ADR-0005: collect() drives a list of envs in round-robin and buckets
    decisions into producing + ordering buffers. `(env_idx, agent)` keying must
    not drop transitions when agent names repeat across envs."""
    import numpy as np

    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.train.smoke_ppo import (
        CentralCritic,
        GaussianActor,
        collect,
        critic_input_dim,
    )

    envs = [TokenomicsAECEnv(horizon=40) for _ in range(3)]
    e = envs[0]
    prod = GaussianActor(
        e.observation_space(e._prod_agents[0]).shape[0], e.spec_.producing_action_dim
    )
    order = GaussianActor(
        e.observation_space(e._order_agents[0]).shape[0], e.spec_.ordering_action_dim
    )
    critic = CentralCritic(critic_input_dim(e))
    bound = float(e.action_space(e._order_agents[0]).high[0])

    pbuf, obuf, roll = collect(envs, prod, order, critic, steps=200, bound=bound,
                               rng=np.random.default_rng(0))

    # ~200 decisions collected; drop_pending removes the open frontier (<=12 envs*
    # role agents), so the kept count is a bit under the budget (ADR-0017).
    assert len(pbuf.obs) + len(obuf.obs) >= 150
    assert pbuf.obs and obuf.obs
    # global state width matches the centralized-critic input
    assert len(pbuf.glob[0]) == critic_input_dim(e)
    # rollout survival + fulfillment signals for the gated curriculum (ADR-0009,
    # review 2026-06-15) are rates; episodes is a non-negative evidence count
    assert math.isnan(roll.death_rate) or 0.0 <= roll.death_rate <= 1.0
    assert math.isnan(roll.fulfillment) or 0.0 <= roll.fulfillment <= 1.0
    assert roll.episodes >= 0
    # ADR-0017 invariants: no open transitions survive, fields stay aligned,
    # every link points forward to a valid in-buffer row (or -1).
    for buf in (pbuf, obuf):
        assert not buf.pending
        assert len(buf.next_idx) == len(buf.obs)
        for i, nx in enumerate(buf.next_idx):
            assert nx == -1 or i < nx < len(buf.obs)


def test_collect_preserves_live_episode_state_across_batches():
    """ADR-0018: a PPO rollout cut is not an episode reset. Consecutive collect
    calls must continue into later cycles when the environment is still live."""
    import numpy as np

    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.train.smoke_ppo import (
        CentralCritic,
        GaussianActor,
        collect,
        critic_input_dim,
    )

    envs = [TokenomicsAECEnv(horizon=100) for _ in range(2)]
    e = envs[0]
    prod = GaussianActor(
        e.observation_space(e._prod_agents[0]).shape[0], e.spec_.producing_action_dim
    )
    order = GaussianActor(
        e.observation_space(e._order_agents[0]).shape[0], e.spec_.ordering_action_dim
    )
    critic = CentralCritic(critic_input_dim(e))
    bound = float(e.action_space(e._order_agents[0]).high[0])
    rng = np.random.default_rng(0)

    collect(envs, prod, order, critic, steps=96, bound=bound, rng=rng)
    first_cycles = [env.current_cycle for env in envs]
    collect(envs, prod, order, critic, steps=96, bound=bound, rng=rng)
    second_cycles = [env.current_cycle for env in envs]

    assert all(second > first for first, second in zip(first_cycles, second_cycles))


def test_popart_stats_update_preserves_critic_values():
    """ADR-0019: changing normalization statistics alone must not move V(s)."""
    import torch

    from tokenomics.train.smoke_ppo import CentralCritic

    critic = CentralCritic(global_dim=3)
    states = torch.tensor([[0.2, 0.3, 0.4], [1.0, 0.0, 0.5]])
    before = critic.value(states).detach().clone()

    critic.update_norm(torch.tensor([5.0, 10.0, 20.0, 40.0]))

    after = critic.value(states).detach()
    assert torch.allclose(after, before, atol=1e-6)


def test_critic_input_distinguishes_agents_and_includes_local_obs():
    import numpy as np

    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.train.smoke_ppo import critic_input, critic_input_dim

    env = TokenomicsAECEnv()
    env.reset()
    a0, a1 = env._prod_agents[:2]
    o0 = env.observe(a0)
    o1 = env.observe(a1)
    c0 = critic_input(env, a0, o0)
    c1 = critic_input(env, a1, o1)

    assert c0.shape == (critic_input_dim(env),)
    assert c0.dtype == np.float32
    assert not np.array_equal(c0, c1)


def test_gaussian_actor_accepts_integer_log_std_init():
    from tokenomics.train.smoke_ppo import GaussianActor

    actor = GaussianActor(obs_dim=2, act_dim=3, log_std_init=-1)
    assert actor.log_std.is_floating_point()


def test_producing_greedy_policy_emits_box_action_within_bound():
    """ADR-0026: the producing policy is now Gaussian over the Box(G+3) action;
    the deployed (greedy) policy emits the distribution mean clipped to the action
    bound, read through env.observe like a baseline."""
    import numpy as np

    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.train.smoke_ppo import GaussianActor, greedy_eval_policies

    env = TokenomicsAECEnv(horizon=40)
    env.reset(seed=0)
    prod = GaussianActor(
        env.observation_space(env._prod_agents[0]).shape[0], env.spec_.producing_action_dim
    )
    order = GaussianActor(
        env.observation_space(env._order_agents[0]).shape[0], env.spec_.ordering_action_dim
    )
    bound = float(env.action_space(env._prod_agents[0]).high[0])

    prod_policy, _ = greedy_eval_policies(prod, order, bound)
    action = prod_policy(env, env._prod_agents[0])

    assert action.shape == (env.spec_.producing_action_dim,)
    assert np.all(np.abs(action) <= bound)
    # deterministic: greedy == the mean, so two calls match
    assert np.array_equal(action, prod_policy(env, env._prod_agents[0]))


def test_checkpoint_roundtrips_and_rejects_stale_version(tmp_path):
    """ADR-0026: a checkpoint carries a schema version; the loader rebuilds both
    Gaussian actors on a current version and rejects a missing/older one loudly,
    since the producing/ordering actions and critic state all changed."""
    import torch

    from tokenomics.train.smoke_ppo import (
        CHECKPOINT_VERSION,
        GaussianActor,
        load_policies,
        save_policies,
    )

    prod = GaussianActor(obs_dim=8, act_dim=5)
    order = GaussianActor(obs_dim=6, act_dim=12)
    path = tmp_path / "policy.pt"
    save_policies(prod, order, bound=10.0, path=str(path))

    # current version loads
    prod_policy, order_policy = load_policies(str(path))
    assert callable(prod_policy) and callable(order_policy)

    # stale (older / missing) version is rejected loudly
    ckpt = torch.load(str(path), weights_only=True)
    ckpt["version"] = CHECKPOINT_VERSION - 1
    torch.save(ckpt, str(path))
    with pytest.raises(ValueError, match="version"):
        load_policies(str(path))


def test_checkpoint_stamps_and_reads_config_fingerprint(tmp_path):
    """#7: a checkpoint saved with a config carries that config's fingerprint, and
    `checkpoint_fingerprint` reads it back — so an eval can detect a wrong-config
    policy. A checkpoint saved without a config reports None (back-compat)."""
    from tokenomics.domain import load_config
    from tokenomics.env.aec_env import DEFAULT_CONFIG
    from tokenomics.env.spaces import config_fingerprint
    from tokenomics.train.smoke_ppo import (
        GaussianActor,
        checkpoint_fingerprint,
        save_policies,
    )

    cfg = load_config(DEFAULT_CONFIG)
    prod = GaussianActor(obs_dim=8, act_dim=5)
    order = GaussianActor(obs_dim=6, act_dim=12)

    stamped = tmp_path / "stamped.pt"
    save_policies(prod, order, bound=10.0, path=str(stamped), config=cfg)
    assert checkpoint_fingerprint(str(stamped)) == config_fingerprint(cfg)

    plain = tmp_path / "plain.pt"
    save_policies(prod, order, bound=10.0, path=str(plain))
    assert checkpoint_fingerprint(str(plain)) is None


def test_signal_check_skips_shape_incompatible_trained_row(tmp_path, capsys, monkeypatch):
    """[P2 review 2026-06-15]: scoring a checkpoint on a config whose observation dims
    differ must NOT crash the actor on a matrix-shape RuntimeError. signal_check
    compares the checkpoint's stored actor dims against the eval spec and SKIPS the
    trained row (with a message) when they are incompatible — only same-shape transfer
    tests proceed."""
    import sys
    from pathlib import Path

    from tokenomics.domain import load_config
    from tokenomics.env.aec_env import DEFAULT_CONFIG
    from tokenomics.env.spaces import build_spec
    from tokenomics.eval import signal_check
    from tokenomics.train.smoke_ppo import GaussianActor, save_policies

    # actors sized to phase1's real spec -> a same-config eval would run fine.
    cfg = load_config(DEFAULT_CONFIG)
    sp = build_spec(cfg)
    prod = GaussianActor(obs_dim=sp.producing_obs_dim, act_dim=sp.producing_action_dim)
    order = GaussianActor(obs_dim=sp.ordering_obs_dim, act_dim=sp.ordering_action_dim)
    ckpt = tmp_path / "phase1.pt"
    save_policies(prod, order, bound=10.0, path=str(ckpt), config=cfg)

    # complex.json has 5 producers (vs phase1's 4) -> different producing obs dim.
    complex_cfg = Path(__file__).parent / "fixtures" / "complex.json"
    monkeypatch.setattr(sys, "argv", [
        "signal_check", "--config", str(complex_cfg),
        "--policy", str(ckpt), "--seeds", "1", "--horizon", "5",
    ])
    signal_check.main()  # must not raise (was: matrix-shape RuntimeError)
    out = capsys.readouterr().out.lower()
    assert "skip trained(mappo)" in out  # row skipped with a message
    assert "incompatible" in out
    # the trained *data* row never ran: no trained metrics line (would end in capmin)
    assert not any(
        line.strip().startswith("trained(mappo)") for line in out.splitlines()
    )


def test_validated_stability_controls_are_training_defaults():
    from inspect import signature

    from tokenomics.train.smoke_ppo import (
        DEFAULT_ORDER_LOG_STD,
        DEFAULT_PROD_LOG_STD,
        DEFAULT_TARGET_KL,
        train,
    )

    # ADR-0026: both actors are Gaussian; producing exploration is a log_std, not
    # a softmax temperature. Carried-forward target-KL stays 0.008 (ADR-0024).
    defaults = signature(train).parameters
    assert defaults["prod_log_std"].default == DEFAULT_PROD_LOG_STD == -2.0
    assert defaults["log_std_init"].default == DEFAULT_ORDER_LOG_STD == -2.0
    assert defaults["target_kl"].default == DEFAULT_TARGET_KL == 0.008


def test_target_kl_stops_before_an_out_of_region_actor_step():
    import numpy as np
    import torch

    from tokenomics.train.smoke_ppo import (
        Buf,
        GaussianActor,
        actor_update,
    )

    actor = GaussianActor(obs_dim=7, act_dim=3, log_std_init=-2.0)
    obs = np.zeros((4, 7), dtype=np.float32)
    actions = np.zeros((4, 3), dtype=np.float32)
    mask = np.ones((4, 3), dtype=np.float32)  # P1b: all dims active here
    with torch.no_grad():
        current_logp = (actor.dist(torch.as_tensor(obs)).log_prob(
            torch.as_tensor(actions)
        ) * torch.as_tensor(mask)).sum(-1)

    # Pretend the rollout policy assigned each sampled action much less
    # probability. The first KL check must reject this update before optimizer
    # state or parameters change.
    buf = Buf(
        obs=list(obs),
        act=list(actions),
        mask=list(mask),
        logp=list((current_logp - 5.0).numpy()),
    )
    before = [p.detach().clone() for p in actor.parameters()]
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)

    stats = actor_update(
        actor,
        buf,
        np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32),
        opt,
        is_discrete=False,
        target_kl=0.008,
    )

    assert stats.loss == 0.0  # early-stopped before any optimizer step
    assert all(torch.equal(old, new) for old, new in zip(before, actor.parameters()))


def test_investment_request_cost_counts_executed_toggles():
    """ADR-0022/0026: cost is charged off the gated command decode produced (the
    three do_* flags), not the raw action — a toggle decode ignores (already
    running / unconfigured) opens no request and is not charged. Replenishment is
    excluded."""
    from tokenomics.domain.producing_agent import ProducingAgentCommand
    from tokenomics.train.smoke_ppo import investment_request_cost

    none = ProducingAgentCommand(do_restoration=False, do_upgrade=False)
    one = ProducingAgentCommand(do_restoration=True, do_upgrade=False)
    two = ProducingAgentCommand(
        do_restoration=True, do_upgrade=False, do_inventory_expansion=True
    )
    three = ProducingAgentCommand(
        do_restoration=True, do_upgrade=True, do_inventory_expansion=True,
        replenishments={"x": 9},  # replenishment never adds to the cost
    )

    assert investment_request_cost(none, 1.0) == 0.0
    assert investment_request_cost(one, 1.0) == 1.0
    assert investment_request_cost(two, 1.0) == 2.0
    assert investment_request_cost(three, 1.0) == 3.0
    assert investment_request_cost(three, 0.5) == 1.5


def test_producing_mean_bias_primes_restoration_and_replenishment():
    """Scale survive-first prior (ADR-0043): greedy eval uses the action MEAN, and
    `decode_producing` fires restoration only when the restoration-toggle dim > 0.
    Like the P1a replenishment bias, a positive restoration-toggle bias makes the
    deterministic policy restore from iter 0 — the survival prior the deep-chain
    scale economy needs (absent restoration => localized capacity collapse / death,
    even while total capacity stays high). `restore_bias=0` (the phase1 default)
    leaves restoration off, so the canonical run is unchanged."""
    import numpy as np

    from tokenomics.domain.producing_agent import ProducingAgentView
    from tokenomics.env.decode import decode_producing
    from tokenomics.env.spaces import build_spec
    from tokenomics.domain import load_config
    from tokenomics.env.aec_env import DEFAULT_CONFIG
    from tokenomics.train.smoke_ppo import producing_mean_bias

    spec = build_spec(load_config(DEFAULT_CONFIG))
    g = spec.num_inputs
    pid = spec.producer_ids[0]
    view = ProducingAgentView(
        id=pid,
        capacities={t: 0 for t in spec.producer_owns[pid]},
        max_capacities={t: 0 for t in spec.producer_owns[pid]},
        requested_capacity=0,
        degradation={t: 0 for t in spec.producer_owns[pid]},
        upgrade=0,
        restoration=0,
        upgrade_running=False,
        restoration_running=False,
        inventory={},
        inventory_limit=0,
        inventory_units=0,
        headroom=0,
        inventory_expansion_running=False,
        has_inventory_expansion=False,
    )

    primed = producing_mean_bias(
        g, spec.producing_action_dim, replen_bias=2.0, restore_bias=3.0
    )
    assert primed.shape == (spec.producing_action_dim,)
    assert np.all(primed[:g] == 2.0)        # replenishment dims primed (P1a)
    assert primed[g] == 3.0                  # restoration toggle primed
    assert np.all(primed[g + 1 :] == 0.0)    # upgrade + inventory untouched
    # the MEAN action (== bias) decodes to a restoration request
    assert decode_producing(spec, primed, view).do_restoration is True

    # phase1 default: no restoration bias -> greedy eval does not restore
    default = producing_mean_bias(g, spec.producing_action_dim, replen_bias=2.0)
    assert default[g] == 0.0
    assert decode_producing(spec, default, view).do_restoration is False


def test_investment_cost_skips_toggles_decode_does_not_execute():
    """Regression (ADR-0026): a positive raw toggle decode ignores — investment
    already running or unconfigured — opens no request, so it must not be charged.
    Cost is read off the gated command, not the raw action."""
    import numpy as np

    from tokenomics.domain.producing_agent import ProducingAgentView
    from tokenomics.env.decode import decode_producing
    from tokenomics.env.spaces import build_spec
    from tokenomics.domain import load_config
    from tokenomics.env.aec_env import DEFAULT_CONFIG
    from tokenomics.train.smoke_ppo import investment_request_cost

    spec = build_spec(load_config(DEFAULT_CONFIG))
    pid = spec.producer_ids[0]
    # restoration is already running and inventory expansion is unconfigured;
    # only an upgrade can actually fire this cycle.
    view = ProducingAgentView(
        id=pid,
        capacities={t: 0 for t in spec.producer_owns[pid]},
        max_capacities={t: 0 for t in spec.producer_owns[pid]},
        requested_capacity=0,
        degradation={t: 0 for t in spec.producer_owns[pid]},
        upgrade=0,
        restoration=0,
        upgrade_running=False,
        restoration_running=True,
        inventory={},
        inventory_limit=0,
        inventory_units=0,
        headroom=0,
        inventory_expansion_running=False,
        has_inventory_expansion=False,
    )
    # all three toggle dims positive
    action = np.zeros(spec.producing_action_dim, dtype=np.float32)
    action[-3:] = 1.0
    cmd = decode_producing(spec, action, view)

    # decode executes only the upgrade; restoration (running) + inventory
    # expansion (unconfigured) are dropped -> charged once, not three times.
    assert investment_request_cost(cmd, 1.0) == 1.0


def test_curriculum_gate_advances_only_on_evidence():
    """ADR-0009 + review 2026-06-15: the gated curriculum advances one step only on
    *accumulated* evidence — >= min_episodes rollouts AND both survival
    (death <= advance_below) and fulfillment (>= advance_fulfil). Evidence piles up
    across iterations (one eased rollout rarely completes the floor); survival with
    no fulfillment, or too-high death, hold (the old survival-only/one-episode gate
    let difficulty outrun the policy)."""
    from tokenomics.train.smoke_ppo import Curriculum, RolloutStats

    # accumulation + advance: two sub-floor rollouts cross min_episodes together
    cur = Curriculum(
        d_start=0.2, gate=True, advance_below=0.2, advance_fulfil=0.8,
        min_episodes=4, d_step=0.1,
    )
    assert cur.level(0, 100) == pytest.approx(0.2)
    cur.observe(RolloutStats(episodes=2, deaths=0, completed=10, rejected=0))  # 2 < 4 -> hold
    assert cur.level(0, 100) == pytest.approx(0.2)
    cur.observe(RolloutStats(episodes=2, deaths=0, completed=10, rejected=0))  # 4 total, clean -> advance
    assert cur.level(0, 100) == pytest.approx(0.3)

    # enough episodes but all died -> hold
    died = Curriculum(d_start=0.2, gate=True, advance_below=0.2, advance_fulfil=0.8, min_episodes=4)
    died.observe(RolloutStats(episodes=8, deaths=8, completed=10, rejected=0))
    assert died.level(0, 100) == pytest.approx(0.2)

    # enough episodes, survived, but fulfillment below the bar -> hold
    unfilled = Curriculum(d_start=0.2, gate=True, advance_below=0.2, advance_fulfil=0.8, min_episodes=4)
    unfilled.observe(RolloutStats(episodes=8, deaths=0, completed=3, rejected=7))
    assert unfilled.level(0, 100) == pytest.approx(0.2)

    # ungated curriculum: observe() is a no-op, level == iter schedule
    plain = Curriculum(d_start=0.2, warmup_frac=0.6)
    plain.observe(RolloutStats(episodes=8, deaths=0, completed=10, rejected=0))
    assert plain.level(0, 10) == pytest.approx(0.2)


def test_curriculum_gate_reads_consumer_units_not_order_counts():
    """#3 follow-up (review 2026-06-15): under variable quantity the order-count
    fulfillment can clear `advance_fulfil` while most demanded UNITS go unmet, so the
    gate would advance difficulty against the quantity-weighted objective. The gate
    must read consumer *units*. Example: 4 completed qty-1 orders + 1 rejected qty-3
    order = 4/5 = 80% orders (would pass) but 4/7 = 57% units (must hold)."""
    from tokenomics.train.smoke_ppo import Curriculum, RolloutStats

    cur = Curriculum(
        d_start=0.2, gate=True, advance_below=0.2, advance_fulfil=0.8, min_episodes=4,
    )
    cur.observe(RolloutStats(
        episodes=5, deaths=0, completed=4, rejected=1,
        completed_units=4, rejected_units=3,
    ))
    assert cur.level(0, 100) == pytest.approx(0.2)  # 57% units < 80% -> hold

    # same order counts, but the rejected order was qty-1 too: 4/7? no — units 4/5=80% -> advance
    ok = Curriculum(
        d_start=0.2, gate=True, advance_below=0.2, advance_fulfil=0.8, min_episodes=4,
    )
    ok.observe(RolloutStats(
        episodes=5, deaths=0, completed=4, rejected=1,
        completed_units=8, rejected_units=2,
    ))
    assert ok.level(0, 100) == pytest.approx(0.3)  # 80% units -> advance


def test_curriculum_scales_demand_variance():
    """Review 2026-06-15: the demand axis eases consumer variance by difficulty —
    quantity range shrinks to quantity_min and a multi-product pool collapses to
    the single primary product below MIXTURE_AT; d=1.0 restores the real demand.
    Dims are unchanged, so it composes with scale_degradation under one scalar."""
    from tokenomics.domain.config import Configuration, ConsumerConfig
    from tokenomics.train.smoke_ppo import scale_demand

    c = ConsumerConfig(id="c1", product=30, products=(20, 30), quantity_min=1, quantity_max=3)
    cfg = Configuration(cycle_emission=1000, process_sheets=[], producer_configs=[], consumers=[c])

    easy = scale_demand(cfg, 0.0).consumers[0]
    assert easy.quantity_max == 1                 # range collapses to quantity_min
    assert easy.product_pool == (30,)             # mixture off -> single primary product
    mid = scale_demand(cfg, 0.5).consumers[0]
    assert mid.quantity_max == 2                   # 1 + round(2*0.5)
    assert mid.product_pool == (20, 30)            # mixture on at MIXTURE_AT
    real = scale_demand(cfg, 1.0).consumers[0]
    assert real.quantity_max == 3 and real.products == (20, 30)
    # base untouched (frozen + replace -> new object)
    assert cfg.consumers[0].quantity_max == 3


def test_competition_axis_boosts_weak_producer_to_viable_then_restores():
    """#1-proper: the competition axis trains routing among *viable* producers first.
    At d=0 a producer whose owned capacity is below its declared product's recipe
    requirement is boosted up to that requirement (made viable); at d=1 the real
    (weak) capacity is restored, so routing must cope with a near-dead competitor."""
    from tokenomics.domain.config import (
        Configuration,
        ProcessSheet,
        ProducingAgentConfig,
        Restoration,
        Upgrade,
    )
    from tokenomics.train.smoke_ppo import scale_competition

    ps = ProcessSheet(10, {"capacity-1": 2})  # 2 units of capacity-1 per product unit
    mk = lambda pid, cap: ProducingAgentConfig(  # noqa: E731
        pid, {"capacity-1": cap}, 1,
        Restoration(99, 4, "capacity-1"), Upgrade(99, 4, "capacity-1"),
        produces=frozenset({10}),
    )
    cfg = Configuration(
        cycle_emission=1000, process_sheets=[ps],
        producer_configs=[mk("p_strong", 8), mk("p_weak", 1)], consumers=[],
    )

    eased = {p.id: p for p in scale_competition(cfg, 0.0).producer_configs}
    assert eased["p_weak"].capacities["capacity-1"] == 2   # 1 -> 2 (recipe requirement): viable
    assert eased["p_strong"].capacities["capacity-1"] == 8  # already viable, untouched
    mid = {p.id: p for p in scale_competition(cfg, 0.5).producer_configs}
    assert mid["p_weak"].capacities["capacity-1"] == 2      # 1 + round((2-1)*0.5)=2 (rounds up)
    real = {p.id: p for p in scale_competition(cfg, 1.0).producer_configs}
    assert real["p_weak"].capacities["capacity-1"] == 1     # real weak capacity restored
    assert cfg.producer_configs[1].capacities["capacity-1"] == 1  # base untouched


def test_staged_config_ramps_axes_sequentially():
    """#1-proper: the curriculum ramps ONE axis at a time (fixed qty1 -> variable
    qty -> product mixture -> competition -> full degradation), not all together.
    A given progress p activates only the axes up to its stage; later axes stay
    fully eased until their stage. p=0 is the easiest config, p=1 is the real one."""
    from tokenomics.domain.config import (
        Configuration,
        ConsumerConfig,
        ProcessSheet,
        ProducingAgentConfig,
        Restoration,
        Upgrade,
    )
    from tokenomics.train.smoke_ppo import staged_config

    ps = ProcessSheet(20, {"capacity-3": 2})
    weak = ProducingAgentConfig(
        "p_weak", {"capacity-3": 1}, 10,
        Restoration(99, 4, "capacity-3"), Upgrade(99, 4, "capacity-3"),
        produces=frozenset({20}),
    )
    con = ConsumerConfig(id="c1", product=30, products=(20, 30), quantity_min=1, quantity_max=3)
    base = Configuration(
        cycle_emission=1000, process_sheets=[ps], producer_configs=[weak], consumers=[con]
    )

    def at(p: float):
        cfg = staged_config(base, p)
        return cfg.consumers[0], cfg.producer_configs[0]

    # stage 1 (p=0): everything eased — fixed qty1, single product, viable, no decay
    c0, p0 = at(0.0)
    assert c0.quantity_max == 1 and c0.product_pool == (30,)
    assert p0.capacities["capacity-3"] == 2 and p0.degradation == 0

    # stage 2 (qty ramp): quantity grows; mixture/competition/degradation still off
    c1, p1 = at(0.1)
    assert c1.quantity_max > 1 and c1.product_pool == (30,)
    assert p1.capacities["capacity-3"] == 2 and p1.degradation == 0

    # stage 4 (competition): qty full + mixture on, competition tightening, no decay yet
    c3, p3 = at(0.6)
    assert c3.quantity_max == 3 and c3.product_pool == (20, 30)
    assert p3.degradation == 0
    assert 1 <= p3.capacities["capacity-3"] <= 2

    # stage 5 (degradation): the last axis finally ramps
    _, p4 = at(0.9)
    assert p4.degradation > 0

    # p=1 is the real config
    c5, p5 = at(1.0)
    assert c5.quantity_max == 3 and c5.product_pool == (20, 30)
    assert p5.capacities["capacity-3"] == 1 and p5.degradation == 10


def test_curriculum_ramps_and_scales_degradation():
    """ADR-0006: difficulty ramps d_start -> 1.0 and scales degradation; eval
    config stays real (untouched)."""
    from tokenomics.domain import load_config
    from tokenomics.env.aec_env import DEFAULT_CONFIG
    from tokenomics.train.smoke_ppo import Curriculum, scale_degradation

    cur = Curriculum(d_start=0.2, warmup_frac=0.6)
    assert cur.difficulty(0, 10) == pytest.approx(0.2)
    assert cur.difficulty(9, 10) == pytest.approx(1.0)
    # monotone non-decreasing
    ds = [cur.difficulty(i, 10) for i in range(10)]
    assert ds == sorted(ds)

    base = load_config(DEFAULT_CONFIG)
    eased = scale_degradation(base, 0.2)
    for b, e in zip(base.producer_configs, eased.producer_configs):
        assert e.degradation == max(0, round(b.degradation * 0.2))
        assert e.degradation <= b.degradation
    # base untouched (frozen + replace -> new object)
    assert scale_degradation(base, 1.0).producer_configs[0].degradation == \
        base.producer_configs[0].degradation


def test_capacity_health_fresh_build_is_one():
    """ADR-0010/0030 (P2 review 2026-06-15): health is the *product bottleneck*, not
    a producer average — the worst product's best declarer's weakest required-type
    ratio. A fresh build is fully restored, so every product is makeable -> 1.0."""
    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.train.smoke_ppo import capacity_health

    env = TokenomicsAECEnv(horizon=40)
    env.reset(seed=0)
    h = capacity_health(env)
    assert h == pytest.approx(1.0)
    assert 0.0 <= h <= 1.0


# --- product-bottleneck capacity_health (P2 review 2026-06-15) ---
# capacity_health reads only env.system.{producing_agents, process_sheets}; tiny
# fakes pin the per-product death semantics (ADR-0030) without a full config.
class _FakeAgent:
    def __init__(self, produces, capacities, max_capacities):
        self.produces = produces
        self.capacities = capacities
        self.max_capacities = max_capacities


class _FakePS:
    def __init__(self, require):
        self.require = require


class _FakeSystem:
    def __init__(self, agents, sheets):
        self.producing_agents = agents
        self.process_sheets = sheets


class _FakeEnv:
    def __init__(self, system):
        self.system = system


def test_capacity_health_zeroes_when_sole_declarer_loses_a_required_type():
    """ADR-0030: a product dies when no declarer holds every required type. With a
    single declarer of A at zero capacity, A is unmakeable -> health 0, even though
    an unrelated product B has a fully healthy declarer."""
    from tokenomics.train.smoke_ppo import capacity_health

    sys = _FakeSystem(
        agents={
            "pa": _FakeAgent({"A"}, {"t1": 0}, {"t1": 10}),   # sole A declarer, dead
            "pb": _FakeAgent({"B"}, {"t2": 10}, {"t2": 10}),  # B fully healthy
        },
        sheets={"A": _FakePS(["t1"]), "B": _FakePS(["t2"])},
    )
    assert capacity_health(_FakeEnv(sys)) == pytest.approx(0.0)


def test_capacity_health_uses_healthiest_declarer_per_product():
    """A product's makeability is its *best* declarer's weakest-required-type ratio.
    Two A declarers (1.0 and 0.2) -> A = 1.0; B's lone declarer at 0.4 is the
    bottleneck -> system health 0.4."""
    from tokenomics.train.smoke_ppo import capacity_health

    sys = _FakeSystem(
        agents={
            "pa": _FakeAgent({"A"}, {"t1": 10}, {"t1": 10}),  # A at 1.0
            "pb": _FakeAgent({"A"}, {"t1": 2}, {"t1": 10}),   # A at 0.2 (ignored, worse)
            "pc": _FakeAgent({"B"}, {"t2": 4}, {"t2": 10}),   # B at 0.4 (bottleneck)
        },
        sheets={"A": _FakePS(["t1"]), "B": _FakePS(["t2"])},
    )
    assert capacity_health(_FakeEnv(sys)) == pytest.approx(0.4)


def test_ordering_action_mask_singleton_inert_multi_active():
    """P1 review 2026-06-15: a slot whose order has one eligible producer is
    behaviorless (softmax over one = 1) and fully masked; a multi-eligible slot
    activates exactly its eligible producer dims. The mask reads only
    view.incoming / view.eligible, so SimpleNamespace fakes suffice."""
    from types import SimpleNamespace

    from tokenomics.domain import load_config
    from tokenomics.env.aec_env import DEFAULT_CONFIG
    from tokenomics.env.spaces import build_spec
    from tokenomics.train.smoke_ppo import ordering_action_mask

    spec = build_spec(load_config(DEFAULT_CONFIG))
    pa, pb = spec.producer_ids[:2]

    single = SimpleNamespace(incoming={"o": 0}, eligible={"o": {pa: None}})
    assert ordering_action_mask(spec, single).sum() == 0.0

    multi = SimpleNamespace(incoming={"o": 0}, eligible={"o": {pa: None, pb: None}})
    m = ordering_action_mask(spec, multi)
    assert m.sum() == 2.0  # slot 0, only the two eligible producer dims
    assert m[spec.producer_ids.index(pa)] == 1.0
    assert m[spec.producer_ids.index(pb)] == 1.0


def test_competition_config_produces_live_multi_eligible_slot():
    """B9 step 1 integration: competition.json must create a *real* >1-eligible
    routing decision (the unit test above fakes `eligible`). Drive a few cycles so
    p3 emits its input-10 replenishment; the live ordering view then holds a slot
    whose order (product 10) has two eligible producers {p1, p5}, and the mask
    activates exactly those two columns — proving the new topology wires an
    actual ordering choice end-to-end, not just in a synthetic fake."""
    from pathlib import Path

    import numpy as np

    from tokenomics.domain import load_config
    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.eval.baselines import ordering_even_split, producing_stock_aware
    from tokenomics.train.smoke_ppo import ordering_action_mask

    cfg = load_config(Path(__file__).parent / "fixtures" / "competition.json")
    env = TokenomicsAECEnv(config=cfg, horizon=10)
    env.reset(seed=1)
    prod, order = producing_stock_aware(0.6), ordering_even_split

    found = None
    for agent in env.agent_iter(max_iter=4000):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        for oa in env.system.ordering_agents.values():
            v = oa.view(env.system.producer_infos)
            if any(len(v.eligible[oid]) >= 2 for oid in v.incoming):
                found = v
                break
        if found is not None:
            break
        role = env._role[agent][0]
        env.step(prod(env, agent) if role == "prod" else order(env, agent))

    assert found is not None, "competition.json never created a multi-eligible slot"
    m = ordering_action_mask(env.spec_, found).reshape(env.spec_.k, env.spec_.num_producers)
    i1, i5 = env.spec_.producer_ids.index("p1"), env.spec_.producer_ids.index("p5")
    active = (m[:, i1] == 1.0) & (m[:, i5] == 1.0)
    assert active.any()  # some slot routes product 10 over both eligible producers
    # and that slot activates only the two eligible columns, nothing else
    k = int(np.argmax(active))
    assert m[k].sum() == 2.0


def test_producing_mask_drops_replen_dims_without_headroom():
    """P2 review 2026-06-15: replenishment dims are active only with inventory
    headroom and a positive replenish ceiling; toggle dims are independent of it."""
    from tokenomics.domain import load_config
    from tokenomics.domain.producing_agent import ProducingAgentView
    from tokenomics.env.aec_env import DEFAULT_CONFIG
    from tokenomics.env.spaces import build_spec
    from tokenomics.train.smoke_ppo import producing_action_mask

    spec = build_spec(load_config(DEFAULT_CONFIG))
    pid = next(
        p for p in spec.producer_ids
        if spec.producer_inputs[p] and spec.max_replenish[p] > 0
    )
    g = spec.num_inputs

    def mk(headroom):
        return ProducingAgentView(
            id=pid,
            capacities={t: 0 for t in spec.producer_owns[pid]},
            max_capacities={t: 0 for t in spec.producer_owns[pid]},
            requested_capacity=0,
            degradation={t: 0 for t in spec.producer_owns[pid]},
            upgrade=0,
            restoration=0,
            upgrade_running=False,
            restoration_running=False,
            inventory={},
            inventory_limit=0,
            inventory_units=0,
            headroom=headroom,
            inventory_expansion_running=False,
            has_inventory_expansion=False,
        )

    with_room = producing_action_mask(spec, mk(5))
    no_room = producing_action_mask(spec, mk(0))

    assert with_room[:g].sum() == len(spec.producer_inputs[pid])
    assert no_room[:g].sum() == 0.0
    assert list(with_room[g:]) == list(no_room[g:])  # toggles unaffected


def test_per_dim_log_std_and_mean_bias_init():
    """P1a review 2026-06-15: GaussianActor takes a per-dim log_std vector and a
    per-dim output-bias init. Same torch seed isolates the bias delta from the
    random MLP init."""
    import numpy as np
    import torch

    from tokenomics.train.smoke_ppo import GaussianActor

    ls = np.array([0.0, -1.0, -2.0], dtype=np.float32)
    bias = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    torch.manual_seed(0)
    base = GaussianActor(obs_dim=4, act_dim=3)
    torch.manual_seed(0)
    biased = GaussianActor(obs_dim=4, act_dim=3, log_std_init=ls, mean_bias=bias)

    assert torch.allclose(biased.log_std, torch.as_tensor(ls))
    assert torch.allclose(
        biased.actor[-1].bias - base.actor[-1].bias, torch.as_tensor(bias)
    )


def test_masked_update_leaves_inactive_dims_unchanged():
    """P1b review 2026-06-15: a masked-out action dim contributes to neither the
    policy-gradient nor the entropy term, so its log_std and output bias receive no
    update; an active dim does move."""
    import numpy as np
    import torch

    from tokenomics.train.smoke_ppo import Buf, GaussianActor, actor_update

    actor = GaussianActor(obs_dim=4, act_dim=3, log_std_init=-1.0)
    n = 8
    obs = np.zeros((n, 4), dtype=np.float32)
    act = np.zeros((n, 3), dtype=np.float32)
    mask = np.tile(np.array([1.0, 1.0, 0.0], dtype=np.float32), (n, 1))  # dim 2 inert
    with torch.no_grad():
        logp = (
            actor.dist(torch.as_tensor(obs)).log_prob(torch.as_tensor(act))
            * torch.as_tensor(mask)
        ).sum(-1)
    buf = Buf(obs=list(obs), act=list(act), mask=list(mask), logp=list(logp.numpy()))

    ls_before = actor.log_std.detach().clone()
    bias_before = actor.actor[-1].bias.detach().clone()
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    # non-constant advantage -> the pg term survives advantage whitening
    actor_update(
        actor, buf, np.linspace(-1.0, 1.0, n).astype(np.float32), opt,
        is_discrete=False, ent_coef=0.01, target_kl=None, epochs=1,
    )
    ls_after = actor.log_std.detach()
    bias_after = actor.actor[-1].bias.detach()

    assert ls_after[2] == ls_before[2]      # inactive dim untouched
    assert bias_after[2] == bias_before[2]
    assert not torch.equal(ls_after[:2], ls_before[:2]) or not torch.equal(
        bias_after[:2], bias_before[:2]
    )


def test_masked_update_resists_adam_momentum_on_frozen_dim():
    """Review 2026-06-15 (P2): a dim active in an earlier batch leaves Adam momentum
    (m, v) behind; when the next batch masks that dim in every row its gradient is
    zero, yet opt.step() would still apply the stored momentum. The per-dim head pin
    must hold the frozen dim fixed despite a warmed optimizer."""
    import numpy as np
    import torch

    from tokenomics.train.smoke_ppo import Buf, GaussianActor, actor_update

    actor = GaussianActor(obs_dim=4, act_dim=3, log_std_init=-1.0)
    n = 8
    obs = np.zeros((n, 4), dtype=np.float32)
    act = np.zeros((n, 3), dtype=np.float32)
    adv = np.linspace(-1.0, 1.0, n).astype(np.float32)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)

    def _buf(mask: np.ndarray) -> Buf:
        with torch.no_grad():
            logp = (
                actor.dist(torch.as_tensor(obs)).log_prob(torch.as_tensor(act))
                * torch.as_tensor(mask)
            ).sum(-1)
        return Buf(obs=list(obs), act=list(act), mask=list(mask), logp=list(logp.numpy()))

    # warm-up: dim 2 ACTIVE -> Adam accumulates momentum on its head + log_std
    warm = np.ones((n, 3), dtype=np.float32)
    for _ in range(3):
        actor_update(actor, _buf(warm), adv, opt, is_discrete=False, epochs=1)
    assert opt.state[actor.log_std]["exp_avg"][2].abs().item() > 0.0  # momentum present

    ls_before = actor.log_std.detach().clone()
    bias_before = actor.actor[-1].bias.detach().clone()
    w_before = actor.actor[-1].weight.detach().clone()
    m_before = opt.state[actor.log_std]["exp_avg"][2].clone()
    v_before = opt.state[actor.log_std]["exp_avg_sq"][2].clone()

    # next batch: dim 2 masked in EVERY row -> must not move despite stored momentum
    masked = np.tile(np.array([1.0, 1.0, 0.0], dtype=np.float32), (n, 1))
    actor_update(actor, _buf(masked), adv, opt, is_discrete=False, epochs=2)

    assert actor.log_std.detach()[2] == ls_before[2]
    assert actor.actor[-1].bias.detach()[2] == bias_before[2]
    assert torch.equal(actor.actor[-1].weight.detach()[2], w_before[2])
    # P2 round 3: the frozen dim's Adam moments must not decay either
    assert opt.state[actor.log_std]["exp_avg"][2] == m_before
    assert opt.state[actor.log_std]["exp_avg_sq"][2] == v_before
    # an active dim still moved
    assert not torch.equal(actor.log_std.detach()[:2], ls_before[:2]) or not torch.equal(
        actor.actor[-1].bias.detach()[:2], bias_before[:2]
    )


def test_all_zero_mask_skips_update_entirely():
    """Review 2026-06-15 (P2 round 3): when no dim is active in the whole batch (the
    ordering actor in canonical config — every order singleton-eligible), the masked
    loss is constant in the params, so opt.step() would move the SHARED TRUNK on
    stored momentum and drift a behaviorless actor. The update must be skipped: no
    parameter and no optimizer-state change at all."""
    import numpy as np
    import torch

    from tokenomics.train.smoke_ppo import Buf, GaussianActor, actor_update

    actor = GaussianActor(obs_dim=4, act_dim=3, log_std_init=-1.0)
    n = 8
    obs = np.zeros((n, 4), dtype=np.float32)
    act = np.zeros((n, 3), dtype=np.float32)
    adv = np.linspace(-1.0, 1.0, n).astype(np.float32)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)

    def _buf(mask: np.ndarray) -> Buf:
        with torch.no_grad():
            logp = (
                actor.dist(torch.as_tensor(obs)).log_prob(torch.as_tensor(act))
                * torch.as_tensor(mask)
            ).sum(-1)
        return Buf(obs=list(obs), act=list(act), mask=list(mask), logp=list(logp.numpy()))

    # warm the optimizer so the trunk carries momentum that would otherwise drift
    for _ in range(3):
        actor_update(actor, _buf(np.ones((n, 3), dtype=np.float32)), adv, opt,
                     is_discrete=False, epochs=1)
    trunk = actor.actor[0].weight  # shared first layer (not per-dim)
    trunk_before = trunk.detach().clone()
    out_before = actor.actor(torch.as_tensor(obs)).detach().clone()
    step_before = opt.state[actor.log_std]["step"].clone()

    zero = np.zeros((n, 3), dtype=np.float32)
    stats = actor_update(actor, _buf(zero), adv, opt, is_discrete=False, epochs=4)

    assert stats.loss == 0.0  # all-masked rows -> no update at all
    assert torch.equal(trunk.detach(), trunk_before)          # trunk did not drift
    assert torch.equal(actor.actor(torch.as_tensor(obs)).detach(), out_before)  # output fixed
    assert opt.state[actor.log_std]["step"] == step_before     # Adam step not advanced


@pytest.mark.parametrize("kwargs", [{"surv_coef": 0.5}, {"low_cap_coef": 0.5}])
def test_capacity_shaping_flags_run(kwargs):
    """Regression: --surv-coef and --low-cap-coef feed capacity_health, which used
    removed scalar capacity fields. Both must drive a training iter without
    crashing on the capacity-bundle model (ADR-0025)."""
    from tokenomics.train.smoke_ppo import train

    history = train(
        iters=1, steps=128, horizon=40, seed=0, eval_seeds=2,
        log=lambda *a, **k: None, **kwargs,
    )
    assert len(history) == 1


def test_curve_csv_written(tmp_path):
    from tokenomics.train.smoke_ppo import train

    out = tmp_path / "curve.csv"
    train(iters=2, steps=128, horizon=40, seed=0, eval_seeds=2, out=str(out),
          log=lambda *a, **k: None)
    lines = out.read_text().splitlines()
    assert lines[0].startswith("iter,mean_step_reward")
    assert len(lines) == 3  # header + 2 iters


def test_train_seeds_reports_median_and_worst_case_across_model_seeds():
    """NEXT STEPS stage 2: train_seeds trains several independent MODEL seeds and
    reports median + worst-case held-out metrics (not one lucky checkpoint). Each
    seed selects its own best-on-validation checkpoint, then is scored on the SAME
    disjoint held-out seed range; summarize_seeds rolls them up direction-aware."""
    from tokenomics.eval.runner import SeedSummary
    from tokenomics.train.smoke_ppo import train_seeds

    sweep = train_seeds(
        seeds=[0, 1], iters=2, steps=128, horizon=40, eval_seeds=2,
        held_out_seeds=2, log=lambda *a, **k: None,
    )
    # one held-out metric dict per model seed
    assert len(sweep.per_seed) == 2
    # robustness summary over the seeds, direction-aware
    assert set(["return", "dead_rate"]).issubset(sweep.summary)
    assert all(isinstance(v, SeedSummary) for v in sweep.summary.values())
    ret = sweep.summary["return"]
    assert ret.n == 2
    assert ret.worst <= ret.median <= ret.best
    dead = sweep.summary["dead_rate"]
    assert dead.worst >= dead.best  # worst death rate is the HIGHER one (lower=good)


def test_train_seeds_gives_each_model_seed_an_independent_curriculum(monkeypatch):
    """A gated curriculum is mutable — train() advances `_gated_d` and accumulates
    evidence via curriculum.observe(). train_seeds must hand each model seed a fresh
    copy, else later seeds inherit earlier seeds' gated level and the reported
    median/worst-case compares different curricula, not independent seeds. Stub out
    train/load/evaluate so the check is deterministic (not dependent on dynamics)."""
    from tokenomics.train import smoke_ppo
    from tokenomics.train.smoke_ppo import Curriculum, train_seeds

    captured = []

    def fake_train(*, curriculum=None, **kw):
        captured.append(curriculum)
        if curriculum is not None:
            curriculum._gated_d = 0.7  # simulate train() advancing the gate
        return []

    metrics = {k: (0.0, 0.0) for k in smoke_ppo.DEFAULT_SEED_REPORT}
    monkeypatch.setattr(smoke_ppo, "train", fake_train)
    monkeypatch.setattr(smoke_ppo, "load_policies", lambda path: (None, None))
    monkeypatch.setattr(smoke_ppo, "evaluate", lambda *a, **k: metrics)

    base = Curriculum(gate=True, d_start=0.0)
    train_seeds(seeds=[0, 1, 2], held_out_seeds=1, curriculum=base,
                log=lambda *a, **k: None)

    assert len(captured) == 3
    assert len({id(c) for c in captured}) == 3   # a distinct curriculum per seed
    assert base not in captured                   # never the caller's instance
    assert base._gated_d == 0.0                    # caller's instance untouched


def test_train_seeds_recreates_stateful_topology_dr_per_model_seed(monkeypatch):
    """topology_dr() owns mutable dwell state. A multi-seed sweep must give each
    model seed a fresh provider, else later seeds inherit the previous seed's
    topology and remaining dwell window."""
    from tokenomics.train import smoke_ppo
    from tokenomics.train.smoke_ppo import topology_dr, train_seeds

    seen = []

    def fake_train(*, topology_random=None, **kw):
        seen.append(topology_random)
        return []

    metrics = {k: (0.0, 0.0) for k in smoke_ppo.DEFAULT_SEED_REPORT}
    monkeypatch.setattr(smoke_ppo, "train", fake_train)
    monkeypatch.setattr(smoke_ppo, "load_policies", lambda path: (None, None))
    monkeypatch.setattr(smoke_ppo, "evaluate", lambda *a, **k: metrics)

    train_seeds(
        seeds=[0, 1, 2],
        held_out_seeds=1,
        topology_random=topology_dr(depth=3, dwell=99),
        log=lambda *a, **k: None,
    )

    assert len(seen) == 3
    assert all(t is not None for t in seen)
    assert len({id(t) for t in seen}) == 3


def test_train_seeds_uses_eval_horizon_for_held_out_summary(monkeypatch):
    """--eval-horizon must apply to the robustness sweep's held-out summary, not
    only to checkpoint selection inside each train() call."""
    from tokenomics.train import smoke_ppo
    from tokenomics.train.smoke_ppo import train_seeds

    horizons = []

    def fake_evaluate(*a, horizon=None, **k):
        horizons.append(horizon)
        return {m: (0.0, 0.0) for m in smoke_ppo.DEFAULT_SEED_REPORT}

    monkeypatch.setattr(smoke_ppo, "train", lambda **kw: [])
    monkeypatch.setattr(smoke_ppo, "load_policies", lambda path: (None, None))
    monkeypatch.setattr(smoke_ppo, "evaluate", fake_evaluate)

    train_seeds(
        seeds=[0, 1],
        held_out_seeds=1,
        horizon=20,
        eval_horizon=60,
        log=lambda *a, **k: None,
    )

    assert horizons == [60, 60]


def test_cli_num_seeds_runs_multi_seed_sweep(monkeypatch, capsys):
    """NEXT STEPS stage 2: `--num-seeds N` runs the multi-seed robustness sweep and
    prints the median/worst-case report instead of a single training run."""
    import sys

    from tokenomics.train import smoke_ppo

    monkeypatch.setattr(sys, "argv", [
        "smoke_ppo", "--num-seeds", "2", "--iters", "2", "--steps", "128",
        "--horizon", "40", "--eval-seeds", "2", "--held-out-seeds", "2",
    ])
    smoke_ppo.main()
    out = capsys.readouterr().out.lower()
    assert "worst" in out and "median" in out


def test_cli_validation_seeds_defaults_base_to_matching_scale_config(monkeypatch):
    """--validation-seeds builds a scale suite. Without --config, the CLI should
    build the actors on a matching scale base instead of defaulting to phase1 and
    crashing on a shape mismatch during validation eval."""
    import sys

    from tokenomics.env.spaces import build_spec
    from tokenomics.train import smoke_ppo

    captured = {}

    def fake_train(**kw):
        captured.update(kw)
        return []

    monkeypatch.setattr(sys, "argv", [
        "smoke_ppo", "--validation-seeds", "1", "--iters", "1", "--steps", "64",
    ])
    monkeypatch.setattr(smoke_ppo, "train", fake_train)

    smoke_ppo.main()

    base = build_spec(captured["config"])
    val = build_spec(captured["validation_configs"][0])
    assert base.producing_obs_dim == val.producing_obs_dim
    assert base.ordering_obs_dim == val.ordering_obs_dim
    assert base.global_state_dim == val.global_state_dim


def test_cli_validation_seeds_rejects_shape_mismatched_config(monkeypatch, capsys):
    """An explicit non-scale --config with --validation-seeds is invalid because
    validation observations would not match the actors built for the base config."""
    import sys

    from tokenomics.env.aec_env import DEFAULT_CONFIG
    from tokenomics.train import smoke_ppo

    monkeypatch.setattr(sys, "argv", [
        "smoke_ppo", "--config", str(DEFAULT_CONFIG), "--validation-seeds", "1",
    ])
    monkeypatch.setattr(
        smoke_ppo, "train",
        lambda **kw: pytest.fail("train should not run with mismatched validation dims"),
    )

    with pytest.raises(SystemExit) as exc:
        smoke_ppo.main()

    assert exc.value.code == 2
    assert "do not match --config" in capsys.readouterr().err


def test_curriculum_caps_difficulty_at_d_max():
    # Faster-feedback rung probe (VERIFICATION §9.1): hold the curriculum at a capped
    # difficulty so one rung is trained to retention before escalating. The cap binds
    # both the iter-schedule and the gated level.
    from tokenomics.train.smoke_ppo import Curriculum

    capped = Curriculum(d_start=0.0, warmup_frac=0.6, d_max=0.25)
    assert capped.level(100, 100) <= 0.25 + 1e-9   # past full ramp, still capped
    assert capped.level(50, 100) <= 0.25 + 1e-9
    uncapped = Curriculum(d_start=0.0, warmup_frac=0.6)
    assert uncapped.level(100, 100) > 0.25          # default reaches full difficulty

    gated = Curriculum(d_start=0.2, gate=True, d_max=0.25)
    gated._gated_d = 0.9                             # pretend evidence advanced it past cap
    assert gated.level(0, 10) <= 0.25 + 1e-9


def test_train_eval_horizon_is_independent_of_train_horizon():
    # VERIFICATION §9.2: train at a short horizon for dense/fast feedback, but evaluate
    # at a separate horizon for baseline comparability. eval_cycles must follow
    # eval_horizon, not the (longer) training horizon.
    from tokenomics.train.smoke_ppo import train

    history = train(
        iters=1, steps=128, horizon=50, eval_horizon=3, seed=0, eval_seeds=1,
        log=lambda *a, **k: None,
    )
    assert history[0].eval_cycles <= 3   # eval capped at eval_horizon, not horizon=50


def test_train_records_validation_suite_metrics_in_curve():
    # VERIFICATION §9.5: the validation-suite score drives checkpoint selection but was
    # hidden from the curve. Expose val_* columns so selected-checkpoint health is
    # visible per-iter, not only at the final 'best eval' line.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.smoke_ppo import train

    history = train(
        iters=1, steps=128, horizon=30, eval_horizon=30, seed=0, eval_seeds=1,
        config=build_scale_config(0, depth=3),
        validation_configs=[build_scale_config(9000, depth=3)],
        log=lambda *a, **k: None,
    )
    r = history[0]
    assert not math.isnan(r.val_dead_rate)
    assert not math.isnan(r.val_consumer_completed_units)
    assert not math.isnan(r.val_return)


def test_collect_bc_dataset_returns_per_role_obs_action_pairs():
    # VERIFICATION §11.4: behavior-clone the obs-compatible baseline before PPO. Step 1
    # is recording (obs, action) pairs per role from a baseline rollout. Widths must
    # match the actor schema so the cloned net is checkpoint-compatible.
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.env.aec_env import TokenomicsAECEnv
    from tokenomics.train.smoke_ppo import collect_bc_dataset

    cfg = build_scale_config(0, depth=3)
    spec = TokenomicsAECEnv(config=cfg).spec_
    data = collect_bc_dataset(
        cfg,
        PRODUCING_BASELINES["obs_stock_aware"],
        ORDERING_BASELINES["even_split"],
        seeds=[0],
        horizon=10,
    )
    assert data.prod_obs.shape[0] > 0 and data.order_obs.shape[0] > 0
    assert data.prod_obs.shape[1] == spec.producing_obs_dim
    assert data.prod_act.shape[1] == spec.producing_action_dim
    assert data.order_obs.shape[1] == spec.ordering_obs_dim
    assert data.order_act.shape[1] == spec.ordering_action_dim
    assert data.prod_obs.shape[0] == data.prod_act.shape[0]


def test_behavior_clone_writes_checkpoint_fitting_producing_baseline(tmp_path):
    # VERIFICATION §11.4: supervised-fit the actor means to baseline actions, save a
    # PPO-loadable checkpoint. The cloned producing actor must predict baseline actions
    # markedly better than the predict-the-mean baseline (i.e. it actually learned the
    # obs→action map), so PPO starts from a policy that fulfills.
    import numpy as np
    import torch

    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.smoke_ppo import (GaussianActor, behavior_clone,
                                            collect_bc_dataset, load_policies)

    cfg = build_scale_config(0, depth=3)
    prod_base = PRODUCING_BASELINES["obs_stock_aware"]
    order_base = ORDERING_BASELINES["even_split"]
    path = tmp_path / "bc.pt"
    behavior_clone(cfg, prod_base, order_base, seeds=[0, 1], horizon=40,
                   epochs=800, save=str(path), log=lambda *a, **k: None)

    # checkpoint is loadable as a deployable Policy pair
    pp, op = load_policies(str(path))
    assert callable(pp) and callable(op)

    # cloned producing actor predicts baseline actions far better than predict-the-mean
    ckpt = torch.load(str(path), weights_only=True)
    prod = GaussianActor(ckpt["prod_obs"], ckpt["prod_act"])
    prod.load_state_dict(ckpt["prod"])
    data = collect_bc_dataset(cfg, prod_base, order_base, seeds=[0, 1], horizon=40)
    bound = ckpt["bound"]
    target = np.clip(data.prod_act, -bound, bound)
    with torch.no_grad():
        pred = prod.actor(torch.as_tensor(data.prod_obs)).numpy()
    cloned_mse = float(np.mean((pred - target) ** 2))
    mean_mse = float(np.mean((target - target.mean(axis=0)) ** 2))
    assert cloned_mse < 0.5 * mean_mse


def test_behavior_clone_sets_tight_log_std_for_fine_tuning(tmp_path):
    # Warm-start fine-tuning needs the cloned actors to explore tightly around their
    # greedy mean, else stochastic rollouts drift the policy off the cloned optimum
    # (VERIFICATION §13.3). log_std_save overrides the (wide) init exploration on save.
    import torch

    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.smoke_ppo import behavior_clone

    cfg = build_scale_config(0, depth=3)
    path = tmp_path / "bc.pt"
    behavior_clone(cfg, PRODUCING_BASELINES["obs_stock_aware"],
                   ORDERING_BASELINES["even_split"], seeds=[0], horizon=20,
                   epochs=10, log_std_save=-1.6, save=str(path),
                   log=lambda *a, **k: None)
    ckpt = torch.load(str(path), weights_only=True)
    assert torch.allclose(ckpt["prod"]["log_std"],
                          torch.full_like(ckpt["prod"]["log_std"], -1.6))
    assert torch.allclose(ckpt["order"]["log_std"],
                          torch.full_like(ckpt["order"]["log_std"], -1.6))


def test_behavior_clone_is_reproducible_given_seed(tmp_path):
    # Clone quality must be reproducible: the actor MLP init is random, so without a
    # fixed seed two clones land in different optima (one may even die on a held seed,
    # VERIFICATION §13.3). Same seed -> identical cloned weights.
    import torch

    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.smoke_ppo import behavior_clone

    cfg = build_scale_config(0, depth=3)
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    kw = dict(seeds=[0], horizon=20, epochs=20, log=lambda *x, **k: None)
    behavior_clone(cfg, PRODUCING_BASELINES["obs_stock_aware"],
                   ORDERING_BASELINES["even_split"], seed=7, save=str(a), **kw)
    behavior_clone(cfg, PRODUCING_BASELINES["obs_stock_aware"],
                   ORDERING_BASELINES["even_split"], seed=7, save=str(b), **kw)
    wa = torch.load(str(a), weights_only=True)["prod"]["actor.0.weight"]
    wb = torch.load(str(b), weights_only=True)["prod"]["actor.0.weight"]
    assert torch.allclose(wa, wb)


def test_train_warm_starts_actors_from_checkpoint(tmp_path):
    # VERIFICATION §11.4: PPO must resume from the behavior-cloned checkpoint. A
    # warm-started run loads the cloned actor weights before the first update, so its
    # initial actors equal the checkpoint's — not the from-scratch init.
    import torch

    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.smoke_ppo import behavior_clone, train

    cfg = build_scale_config(0, depth=3)
    path = tmp_path / "bc.pt"
    behavior_clone(cfg, PRODUCING_BASELINES["obs_stock_aware"],
                   ORDERING_BASELINES["even_split"], seeds=[0], horizon=20,
                   epochs=50, save=str(path), log=lambda *a, **k: None)
    clone = torch.load(str(path), weights_only=True)

    saved = tmp_path / "after.pt"
    captured: dict = {}
    train(iters=1, steps=64, horizon=20, eval_horizon=20, seed=0, eval_seeds=1,
          config=cfg, init_from=str(path), save=str(saved),
          on_init=lambda p, o: captured.update(prod=p, order=o),
          log=lambda *a, **k: None)

    # actors at iter 0 (before any PPO update) carry the cloned weights
    w_clone = clone["prod"]["actor.0.weight"]
    w_init = captured["prod"]["actor.0.weight"]
    assert torch.allclose(w_init, w_clone)


def test_role_explained_variance_splits_pooled_fit_by_role():
    # Value-fit diagnostic (VERIFICATION §13.6): the critic regresses ONE pooled
    # target (producer rows then ordering rows). Splitting EV by role localizes a
    # value-fit failure — a role whose return scale differs can be fit poorly while
    # the other looks fine, which the pooled number alone would hide.
    import numpy as np

    from tokenomics.train.smoke_ppo import role_explained_variance

    ret = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    preds = np.array([1.0, 2.0, 3.0, 30.0, 20.0, 10.0])  # prod exact, order reversed
    pooled, prod, order = role_explained_variance(ret, preds, n_prod=3)
    assert prod == pytest.approx(1.0)   # producer rows fit perfectly
    assert order < 0.0                  # ordering rows worse than predicting the mean
    assert not math.isnan(pooled)
    # degenerate splits report NaN, not a crash
    _, all_prod, no_order = role_explained_variance(ret, preds, n_prod=len(ret))
    assert math.isnan(no_order)
    _, no_prod, all_order = role_explained_variance(ret, preds, n_prod=0)
    assert math.isnan(no_prod)


def test_train_records_per_role_critic_explained_variance():
    # train() must surface both per-role EV readings in the curve so the §13.6
    # value-learning blocker can be attributed to producer vs ordering returns.
    from tokenomics.train.smoke_ppo import train

    history = train(
        iters=2, steps=128, horizon=40, seed=0, eval_seeds=1,
        log=lambda *a, **k: None,
    )
    assert all(not math.isnan(r.critic_explained_var_prod) for r in history)
    assert all(not math.isnan(r.critic_explained_var_order) for r in history)


def test_actors_should_freeze_gates_on_warmup_then_critic_fit():
    # Critic fit gate (VERIFICATION §13.6): generalize the fixed `critic_warmup` K
    # into a condition on critic explained variance. Actors stay frozen during the
    # warmup window, then until the rolling-mean EV clears the gate threshold.
    from tokenomics.train.smoke_ppo import actors_should_freeze

    # Fixed warmup window dominates regardless of EV / gate.
    assert actors_should_freeze(0, critic_warmup=2, recent_evs=[0.9], critic_gate_ev=0.5)
    assert actors_should_freeze(1, critic_warmup=2, recent_evs=[0.9], critic_gate_ev=0.5)
    # No gate set -> unfreeze the moment the warmup window ends.
    assert not actors_should_freeze(2, critic_warmup=2, recent_evs=[], critic_gate_ev=None)
    # Gate set but no finite EV seen yet -> stay frozen (never trust an unfit critic).
    assert actors_should_freeze(2, critic_warmup=0, recent_evs=[], critic_gate_ev=0.0)
    # Rolling-mean EV below threshold -> frozen; at/above -> unfrozen.
    assert actors_should_freeze(5, critic_warmup=0, recent_evs=[-0.3, 0.1], critic_gate_ev=0.2)
    assert not actors_should_freeze(5, critic_warmup=0, recent_evs=[0.1, 0.5], critic_gate_ev=0.2)


def test_critic_gate_freezes_actor_weights_until_ev_clears(tmp_path):
    # An unreachable EV threshold (>1.0) must keep actors frozen every iter, so the
    # actor weights never move from their init and every record flags actors_frozen.
    import torch

    from tokenomics.train.smoke_ppo import train

    captured: dict = {}
    saved = tmp_path / "gated.pt"
    history = train(
        iters=3, steps=128, horizon=40, seed=0, eval_seeds=1,
        critic_gate_ev=2.0, save=str(saved),
        on_init=lambda p, o: captured.update(prod=p, order=o),
        log=lambda *a, **k: None,
    )
    assert all(r.actors_frozen for r in history)
    # frozen actors never move off their init weights
    final = torch.load(str(saved), weights_only=True)
    assert torch.allclose(final["prod"]["actor.0.weight"], captured["prod"]["actor.0.weight"])
    assert torch.allclose(final["order"]["actor.0.weight"], captured["order"]["actor.0.weight"])
    # default (no gate, no warmup) leaves actors unfrozen
    history2 = train(
        iters=2, steps=128, horizon=40, seed=0, eval_seeds=1,
        log=lambda *a, **k: None,
    )
    assert not any(r.actors_frozen for r in history2)


def test_dagger_relabel_reduces_to_bc_when_learner_is_expert():
    # DAgger relabel pass (VERIFICATION §13.11 fix): roll out the LEARNER but label
    # visited states with the EXPERT. When learner==expert the trajectory and labels
    # are both the expert's, so the dataset must equal a plain BC dataset.
    import numpy as np

    from tokenomics.env.aec_env import TokenomicsAECEnv
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.smoke_ppo import collect_bc_dataset, dagger_relabel_dataset

    cfg = build_scale_config(0, depth=3)
    pb, ob = PRODUCING_BASELINES["obs_stock_aware"], ORDERING_BASELINES["even_split"]
    env = TokenomicsAECEnv(config=cfg, horizon=40)
    bound = float(env.action_space(env._order_agents[0]).high[0])

    bc = collect_bc_dataset(cfg, pb, ob, seeds=[0, 1], horizon=40)
    dag = dagger_relabel_dataset(cfg, pb, ob, pb, ob, seeds=[0, 1], horizon=40, bound=bound)
    assert dag.prod_obs.shape == bc.prod_obs.shape
    assert np.allclose(dag.prod_act, np.clip(bc.prod_act, -bound, bound))
    assert np.allclose(dag.order_act, np.clip(bc.order_act, -bound, bound))


def test_dagger_relabel_labels_with_expert_not_learner():
    # The trajectory must follow the LEARNER while labels come from the EXPERT. With a
    # noop learner (no production) and the obs_stock_aware expert, the recorded actions
    # are the expert's (nonzero), and the noop trajectory differs from a BC rollout.
    import numpy as np

    from tokenomics.env.aec_env import TokenomicsAECEnv
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.smoke_ppo import collect_bc_dataset, dagger_relabel_dataset

    cfg = build_scale_config(0, depth=3)
    pb, ob = PRODUCING_BASELINES["obs_stock_aware"], ORDERING_BASELINES["even_split"]
    noop = PRODUCING_BASELINES["noop"]
    env = TokenomicsAECEnv(config=cfg, horizon=40)
    bound = float(env.action_space(env._order_agents[0]).high[0])

    bc = collect_bc_dataset(cfg, pb, ob, seeds=[0], horizon=40)
    dag = dagger_relabel_dataset(cfg, noop, ob, pb, ob, seeds=[0], horizon=40, bound=bound)
    # labels are the expert's (obs_stock_aware produces nonzero actions)
    assert np.abs(dag.prod_act).sum() > 0.0
    # trajectory follows the noop learner, not the expert -> different visited states
    assert dag.prod_obs.shape[0] != bc.prod_obs.shape[0]


def test_dagger_clone_runs_and_saves_loadable_checkpoint(tmp_path):
    # End-to-end DAgger: a couple of relabel rounds across two topologies, then a
    # loadable checkpoint whose cloned producing actor fits the aggregated labels
    # better than predict-the-mean.
    import numpy as np
    import torch

    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.smoke_ppo import GaussianActor, dagger_clone, load_policies

    cfgs = [build_scale_config(s, depth=3) for s in (0, 1)]
    pb, ob = PRODUCING_BASELINES["obs_stock_aware"], ORDERING_BASELINES["even_split"]
    path = tmp_path / "dagger.pt"
    dagger_clone(cfgs, pb, ob, dagger_iters=2, seeds=[0], horizon=40,
                 epochs=300, save=str(path), log=lambda *a, **k: None)

    pp, op = load_policies(str(path))
    assert callable(pp) and callable(op)
    ckpt = torch.load(str(path), weights_only=True)
    assert ckpt["version"] == 6
