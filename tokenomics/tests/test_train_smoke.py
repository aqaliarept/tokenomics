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
    loss = actor_update(actor, _buf(zero), adv, opt, is_discrete=False, epochs=4)

    assert loss == 0.0
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
