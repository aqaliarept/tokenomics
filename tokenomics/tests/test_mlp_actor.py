"""Flat MLP actors: init, action masks, checkpoint IO, staged economies, and the
behaviour-cloning / DAgger fits that produced the champion producer (EXP-0002 → 0004).
Skipped if torch is not installed (it's an optional `train` extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_gaussian_actor_accepts_integer_log_std_init():
    from tokenomics.train.mlp_actor import GaussianActor

    actor = GaussianActor(obs_dim=2, act_dim=3, log_std_init=-1)
    assert actor.log_std.is_floating_point()


def test_producing_greedy_policy_emits_box_action_within_bound():
    """ADR-0026: the producing policy is now Gaussian over the Box(G+3) action;
    the deployed (greedy) policy emits the distribution mean clipped to the action
    bound, read through env.observe like a baseline."""
    import numpy as np

    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.train.mlp_actor import GaussianActor, greedy_eval_policies

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

    from tokenomics.train.mlp_actor import (
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
    from tokenomics.train.mlp_actor import (
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
    from tokenomics.train.mlp_actor import GaussianActor, save_policies

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
    from tokenomics.train.mlp_actor import producing_mean_bias

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


def test_curriculum_scales_demand_variance():
    """Review 2026-06-15: the demand axis eases consumer variance by difficulty —
    quantity range shrinks to quantity_min and a multi-product pool collapses to
    the single primary product below MIXTURE_AT; d=1.0 restores the real demand.
    Dims are unchanged, so it composes with scale_degradation under one scalar."""
    from tokenomics.domain.config import Configuration, ConsumerConfig
    from tokenomics.train.mlp_actor import scale_demand

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
    from tokenomics.train.mlp_actor import scale_competition

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
    from tokenomics.train.mlp_actor import staged_config

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


def test_scale_degradation_eases_and_leaves_base_untouched():
    """ADR-0006: the degradation axis scales each producer's degradation down; the
    base config stays real (untouched)."""
    from tokenomics.domain import load_config
    from tokenomics.env.aec_env import DEFAULT_CONFIG
    from tokenomics.train.mlp_actor import scale_degradation

    base = load_config(DEFAULT_CONFIG)
    eased = scale_degradation(base, 0.2)
    for b, e in zip(base.producer_configs, eased.producer_configs):
        assert e.degradation == max(0, round(b.degradation * 0.2))
        assert e.degradation <= b.degradation
    # base untouched (frozen + replace -> new object)
    assert scale_degradation(base, 1.0).producer_configs[0].degradation == \
        base.producer_configs[0].degradation


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


def test_ordering_action_mask_singleton_inert_multi_active():
    """P1 review 2026-06-15: a slot whose order has one eligible producer is
    behaviorless (softmax over one = 1) and fully masked; a multi-eligible slot
    activates exactly its eligible producer dims. The mask reads only
    view.incoming / view.eligible, so SimpleNamespace fakes suffice."""
    from types import SimpleNamespace

    from tokenomics.domain import load_config
    from tokenomics.env.aec_env import DEFAULT_CONFIG
    from tokenomics.env.spaces import build_spec
    from tokenomics.train.mlp_actor import ordering_action_mask

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
    from tokenomics.train.mlp_actor import ordering_action_mask

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
    from tokenomics.train.mlp_actor import producing_action_mask

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

    from tokenomics.train.mlp_actor import GaussianActor

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


def test_collect_bc_dataset_returns_per_role_obs_action_pairs():
    # VERIFICATION §11.4: behavior-clone the obs-compatible baseline. Step 1
    # is recording (obs, action) pairs per role from a baseline rollout. Widths must
    # match the actor schema so the cloned net is checkpoint-compatible.
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.env.aec_env import TokenomicsAECEnv
    from tokenomics.train.mlp_actor import collect_bc_dataset

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
    # loadable checkpoint. The cloned producing actor must predict baseline actions
    # markedly better than the predict-the-mean baseline (i.e. it actually learned the
    # obs→action map), so the clone starts from a policy that fulfills.
    import numpy as np
    import torch

    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.mlp_actor import (GaussianActor, behavior_clone,
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
    from tokenomics.train.mlp_actor import behavior_clone

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
    from tokenomics.train.mlp_actor import behavior_clone

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


def test_dagger_relabel_reduces_to_bc_when_learner_is_expert():
    # DAgger relabel pass (VERIFICATION §13.11 fix): roll out the LEARNER but label
    # visited states with the EXPERT. When learner==expert the trajectory and labels
    # are both the expert's, so the dataset must equal a plain BC dataset.
    import numpy as np

    from tokenomics.env.aec_env import TokenomicsAECEnv
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.mlp_actor import collect_bc_dataset, dagger_relabel_dataset

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
    from tokenomics.train.mlp_actor import collect_bc_dataset, dagger_relabel_dataset

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
    from tokenomics.train.mlp_actor import GaussianActor, dagger_clone, load_policies

    cfgs = [build_scale_config(s, depth=3) for s in (0, 1)]
    pb, ob = PRODUCING_BASELINES["obs_stock_aware"], ORDERING_BASELINES["even_split"]
    path = tmp_path / "dagger.pt"
    dagger_clone(cfgs, pb, ob, dagger_iters=2, seeds=[0], horizon=40,
                 epochs=300, save=str(path), log=lambda *a, **k: None)

    pp, op = load_policies(str(path))
    assert callable(pp) and callable(op)
    ckpt = torch.load(str(path), weights_only=True)
    assert ckpt["version"] == 6
