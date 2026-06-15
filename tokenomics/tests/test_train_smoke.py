"""Smoke test for the PPO trainer — runs a couple of tiny iterations and checks
the loop completes and returns a per-iteration training curve (ADR-0003).
Skipped if torch is not installed (it's an optional `train` extra)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")


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
        buf.logp.append(0.0); buf.val.append(0.0); buf.rew.append(r)
        buf.term.append(0.0); buf.boot.append(bt); buf.next_idx.append(nx)
    buf.pending[("e", "a")] = 1   # row 1 never got its reward

    dropped = drop_pending(buf)

    assert dropped == 1
    assert not buf.pending
    assert len(buf.obs) == 1
    assert all(len(getattr(buf, f)) == 1 for f in
               ("obs", "glob", "act", "logp", "val", "rew", "term", "boot", "next_idx"))
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

    pbuf, obuf, death_rate = collect(envs, prod, order, critic, steps=200, bound=bound,
                                     rng=np.random.default_rng(0))

    # ~200 decisions collected; drop_pending removes the open frontier (<=12 envs*
    # role agents), so the kept count is a bit under the budget (ADR-0017).
    assert len(pbuf.obs) + len(obuf.obs) >= 150
    assert pbuf.obs and obuf.obs
    # global state width matches the centralized-critic input
    assert len(pbuf.glob[0]) == critic_input_dim(e)
    # rollout survival signal for the gated curriculum (ADR-0009) is a rate
    assert math.isnan(death_rate) or 0.0 <= death_rate <= 1.0
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
    with torch.no_grad():
        current_logp = actor.dist(torch.as_tensor(obs)).log_prob(
            torch.as_tensor(actions)
        ).sum(-1)

    # Pretend the rollout policy assigned each sampled action much less
    # probability. The first KL check must reject this update before optimizer
    # state or parameters change.
    buf = Buf(
        obs=list(obs),
        act=list(actions),
        logp=list((current_logp - 5.0).numpy()),
    )
    before = [p.detach().clone() for p in actor.parameters()]
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)

    loss = actor_update(
        actor,
        buf,
        np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32),
        opt,
        is_discrete=False,
        target_kl=0.008,
    )

    assert loss == 0.0
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


def test_curriculum_gate_holds_until_survival():
    """ADR-0009: gated curriculum holds difficulty until the rollout survives
    (death_rate <= advance_below), then steps up; ungated ignores observe()."""
    from tokenomics.train.smoke_ppo import Curriculum

    cur = Curriculum(d_start=0.2, gate=True, advance_below=0.2, d_step=0.1)
    assert cur.level(0, 100) == pytest.approx(0.2)
    assert cur.level(99, 100) == pytest.approx(0.2)
    cur.observe(1.0)  # all died -> hold
    assert cur.level(0, 100) == pytest.approx(0.2)
    cur.observe(float("nan"))  # no completed episode -> hold
    assert cur.level(0, 100) == pytest.approx(0.2)
    cur.observe(0.0)  # survived -> advance one step
    assert cur.level(0, 100) == pytest.approx(0.3)

    # ungated curriculum: observe() is a no-op, level == iter schedule
    plain = Curriculum(d_start=0.2, warmup_frac=0.6)
    plain.observe(0.0)
    assert plain.level(0, 10) == pytest.approx(0.2)


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


def test_capacity_health_from_capacity_bundle():
    """ADR-0010/0025: health = mean total-capacity/total-max over producers,
    summed across the capacity bundle. Fresh build is fully restored -> 1.0."""
    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.train.smoke_ppo import capacity_health

    env = TokenomicsAECEnv(horizon=40)
    env.reset(seed=0)
    h = capacity_health(env)
    assert h == pytest.approx(1.0)
    assert 0.0 <= h <= 1.0


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
