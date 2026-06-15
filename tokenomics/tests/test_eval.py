"""Eval harness + the signal-check invariant: a sensible policy must clearly
beat do-nothing, otherwise the env poses no learning problem."""

from __future__ import annotations

from tokenomics.env import TokenomicsAECEnv
from tokenomics.eval.baselines import (
    ORDERING_BASELINES,
    PRODUCING_BASELINES,
    ordering_even_split,
    producing_noop,
    producing_stock_aware,
)
from tokenomics.eval.runner import evaluate, run_episode


def test_run_episode_returns_stats():
    env = TokenomicsAECEnv(horizon=40)
    s = run_episode(env, producing_noop, ordering_even_split, seed=0)
    assert s["cycles"] > 0
    assert "consumer_completed" in s and "cap_min" in s


def test_signal_exists():
    seeds = range(5)
    noop = evaluate(PRODUCING_BASELINES["noop"], ORDERING_BASELINES["even_split"], seeds=seeds, horizon=100)
    # stock_aware (restore + replenish) is the only baseline that feeds composite
    # assembly, so it is the sensible policy the RL bar must clear (ADR-0025 B7).
    good = evaluate(producing_stock_aware(0.6), ordering_even_split, seeds=seeds, horizon=100)
    # do-nothing should die early; the sensible policy should survive much longer
    assert noop["dead_rate"][0] > 0.5
    assert good["cycles"][0] > 2 * noop["cycles"][0]
    # and complete materially more consumer demand
    assert good["consumer_completed"][0] > 1.5 * noop["consumer_completed"][0]


def test_all_baselines_run():
    for pname, pf in PRODUCING_BASELINES.items():
        for oname, of in ORDERING_BASELINES.items():
            env = TokenomicsAECEnv(horizon=20)
            run_episode(env, pf, of, seed=1)


# --- B7 finding: ordering_proportional weights by bundle throughput, in-bounds ---
def _two_eligible_env():
    import numpy as np
    from tokenomics.domain.config import (Configuration, ProcessSheet,
        ProducingAgentConfig, Restoration, Upgrade, ConsumerConfig)

    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[
            ProcessSheet(product=1, require={"capacity-1": 10}),
            ProcessSheet(product=2, require={"capacity-2": 1}),
        ],
        producer_configs=[
            # pa: lots of the REQUIRED type -> high throughput for product 1
            ProducingAgentConfig("pa", {"capacity-1": 100}, 5,
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1")),
            # pb: tiny required type but a HUGE irrelevant type -> high TOTAL capacity,
            #     zero throughput for product 1
            ProducingAgentConfig("pb", {"capacity-1": 5, "capacity-2": 100000}, 5,
                Restoration(2, 50, "capacity-2"), Upgrade(2, 40, "capacity-2")),
        ],
        consumers=[ConsumerConfig("c1", 1)],
    )
    env = TokenomicsAECEnv(config=cfg, horizon=10)
    env.reset(seed=0)
    while env._role[env.agent_selection][0] == "prod":
        env.step(np.zeros(env.spec_.producing_action_dim, dtype=np.float32))
    return env


def test_ordering_proportional_stays_in_action_bounds():
    from tokenomics.eval.baselines import ordering_proportional

    env = _two_eligible_env()
    action = ordering_proportional(env, "order_c1")
    assert env.action_space("order_c1").contains(action)


def test_ordering_proportional_weights_by_bundle_throughput():
    from tokenomics.eval.baselines import ordering_proportional

    env = _two_eligible_env()
    sp = env.spec_
    action = ordering_proportional(env, "order_c1").reshape(sp.k, sp.num_producers)
    pa, pb = sp.producer_ids.index("pa"), sp.producer_ids.index("pb")
    # pa can actually make product 1 (throughput 10) while pb cannot (throughput 0)
    # despite pb's huge total capacity -> pa must get the higher logit.
    assert action[0, pa] > action[0, pb]


def test_ordering_proportional_discounts_reserved_capacity():
    import numpy as np
    from tokenomics.domain.config import (Configuration, ProcessSheet,
        ProducingAgentConfig, Restoration, Upgrade, ConsumerConfig)
    from tokenomics.domain.producing_agent import Bid, _Reservation
    from tokenomics.eval.baselines import ordering_proportional

    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 10})],
        producer_configs=[
            ProducingAgentConfig("pa", {"capacity-1": 100}, 5,
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1")),
            ProducingAgentConfig("pb", {"capacity-1": 100}, 5,
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1")),
        ],
        consumers=[ConsumerConfig("c1", 1)],
    )
    env = TokenomicsAECEnv(config=cfg, horizon=10)
    env.reset(seed=0)
    while env._role[env.agent_selection][0] == "prod":
        env.step(np.zeros(env.spec_.producing_action_dim, dtype=np.float32))
    # pb is fully reserved (all 100 capacity-1 held by a waiting composite) -> it can
    # make 0 units this cycle despite identical raw capacity to pa.
    env.system.producing_agents["pb"].reservations.append(
        _Reservation(Bid({"capacity-1": 100}, 1, "x", product=20, inputs={10: 1}))
    )
    sp = env.spec_
    action = ordering_proportional(env, "order_c1").reshape(sp.k, sp.num_producers)
    pa, pb = sp.producer_ids.index("pa"), sp.producer_ids.index("pb")
    assert action[0, pa] > action[0, pb]
