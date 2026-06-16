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
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                produces=frozenset({1})),
            # pb: tiny required type but a HUGE irrelevant type -> declares product 1
            #     too (owns capacity-1) but its throughput for it is zero (5 < 10).
            ProducingAgentConfig("pb", {"capacity-1": 5, "capacity-2": 100000}, 5,
                Restoration(2, 50, "capacity-2"), Upgrade(2, 40, "capacity-2"),
                produces=frozenset({1, 2})),
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
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                produces=frozenset({1})),
            ProducingAgentConfig("pb", {"capacity-1": 100}, 5,
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                produces=frozenset({1})),
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


def test_ordering_proportional_separates_zero_from_one_unit_throughput():
    # log(max(throughput,1)) mapped throughput 0 and 1 to the SAME logit, so a
    # fully-reserved (0-unit) producer split the order evenly with a 1-unit one.
    # A producer that can make a unit must clearly outrank one that can make none.
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
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                produces=frozenset({1})),
            ProducingAgentConfig("pb", {"capacity-1": 100}, 5,
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                produces=frozenset({1})),
        ],
        consumers=[ConsumerConfig("c1", 1)],
    )
    env = TokenomicsAECEnv(config=cfg, horizon=10)
    env.reset(seed=0)
    while env._role[env.agent_selection][0] == "prod":
        env.step(np.zeros(env.spec_.producing_action_dim, dtype=np.float32))
    # pa: reserve 90 -> 10 avail -> throughput 1; pb: reserve 100 -> 0 avail -> throughput 0
    env.system.producing_agents["pa"].reservations.append(
        _Reservation(Bid({"capacity-1": 90}, 1, "ra", product=1, inputs={2: 1}))
    )
    env.system.producing_agents["pb"].reservations.append(
        _Reservation(Bid({"capacity-1": 100}, 1, "rb", product=1, inputs={2: 1}))
    )
    sp = env.spec_
    action = ordering_proportional(env, "order_c1").reshape(sp.k, sp.num_producers)
    pa, pb = sp.producer_ids.index("pa"), sp.producer_ids.index("pb")
    assert action[0, pa] > action[0, pb]
    e = np.exp([action[0, pa], action[0, pb]])
    assert e[1] / e.sum() < 0.05  # the 0-unit producer gets a negligible share


# --- B9 step 1: with >1 eligible producer, ordering routing changes outcomes ---
from pathlib import Path

import numpy as np

from tokenomics.domain import load_config
from tokenomics.env.spaces import ACTION_LOGIT_BOUND

COMPETITION = Path(__file__).parent / "fixtures" / "competition.json"


def _ordering_to(target_pid: str):
    """All-to-one ordering policy: route every sub-order to `target_pid` by giving
    it +BOUND and all others -BOUND. decode restricts each slot to its eligible
    producers, so a slot where the target is ineligible (a singleton product) just
    splits evenly among its lone eligible producer (a no-op)."""

    def policy(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        sp = env.spec_
        mat = np.full((sp.k, sp.num_producers), -ACTION_LOGIT_BOUND, dtype=np.float32)
        mat[:, sp.producer_ids.index(target_pid)] = ACTION_LOGIT_BOUND
        return mat.flatten()

    return policy


def test_routing_changes_consumer_fulfillment():
    # competition.json duplicates the producer of intermediate product 10 into a
    # strong (p1, capacity-1=120) and a weak (p5, capacity-1=1 < recipe's 2 -> makes
    # 0 units) eligible pair. p3 replenishes input 10; routing that replenishment
    # all-to-weak starves composite assembly, all-to-strong feeds it -> the ordering
    # choice, behaviorless under singleton topology, now moves consumer fulfillment.
    cfg = load_config(COMPETITION)
    strong = evaluate(producing_stock_aware(0.6), _ordering_to("p1"),
                      seeds=[1], horizon=30, config=cfg)
    weak = evaluate(producing_stock_aware(0.6), _ordering_to("p5"),
                    seeds=[1], horizon=30, config=cfg)
    assert strong["consumer_completed"][0] > weak["consumer_completed"][0]


def test_randomized_demand_episode_is_deterministic_per_seed():
    # B9 step 2: an episode under seeded product+quantity demand replays exactly for
    # a given seed (so baselines and trained policies face the same workload), and a
    # different seed yields a different workload (hence generally a different return).
    cfg = load_config(Path(__file__).parent / "fixtures" / "random_demand.json")
    from tokenomics.eval.baselines import ordering_even_split, producing_stock_aware

    a = evaluate(producing_stock_aware(0.6), ordering_even_split,
                 seeds=[3], horizon=30, config=cfg)
    b = evaluate(producing_stock_aware(0.6), ordering_even_split,
                 seeds=[3], horizon=30, config=cfg)
    c = evaluate(producing_stock_aware(0.6), ordering_even_split,
                 seeds=[4], horizon=30, config=cfg)
    assert a["return"] == b["return"]  # same seed -> identical replay
    assert a["consumer_completed"] == b["consumer_completed"]
    assert a["return"] != c["return"]  # different seed -> different demand workload


# --- NEXT STEPS stage 1: observation-compatible baselines ---
# White-box baselines read env.system (oracle); these act on ONLY the observation
# vector the RL policy sees, so they are the fair "strongest observation-compatible
# baseline" the release bar must be cleared against.
def test_obs_producing_restores_when_observed_capacity_low():
    from tokenomics.eval.baselines import decide_producing_obs

    env = TokenomicsAECEnv(horizon=20)
    env.reset(seed=0)
    pid = env.spec_.producer_ids[0]
    agent = f"prod_{pid}"
    g = env.spec_.num_inputs  # restoration toggle sits at index g (after replenish dims)

    # full capacity -> healthy -> no restoration
    act = decide_producing_obs(env.spec_, env.observe(agent), pid, threshold=0.6, stock_target=3)
    assert act[g + 0] <= 0

    # drive observed capacity well below threshold -> restoration fires
    p = env.system.producing_agents[pid]
    for t in p.capacities:
        p.capacities[t] = 0.1 * p.max_capacities[t]
    act_low = decide_producing_obs(env.spec_, env.observe(agent), pid, threshold=0.6, stock_target=3)
    assert act_low[g + 0] > 0


def test_obs_producing_replenishes_relevant_input_below_target():
    from tokenomics.eval.baselines import decide_producing_obs

    env = TokenomicsAECEnv(horizon=20)
    env.reset(seed=0)
    pid = next(p for p in env.spec_.producer_ids if env.spec_.producer_inputs[p])
    agent = f"prod_{pid}"
    inp = env.spec_.producer_inputs[pid][0]
    gi = env.spec_.input_index[inp]
    p = env.system.producing_agents[pid]

    # empty observed stock -> requests replenishment of that input
    p.inventory[inp] = 0
    act = decide_producing_obs(env.spec_, env.observe(agent), pid, threshold=0.6, stock_target=3)
    assert act[gi] > 0

    # observed stock at/above target -> no replenishment requested
    p.inventory[inp] = 10
    act2 = decide_producing_obs(env.spec_, env.observe(agent), pid, threshold=0.6, stock_target=3)
    assert act2[gi] <= 0


def test_obs_producing_action_in_bounds():
    from tokenomics.eval.baselines import decide_producing_obs

    env = TokenomicsAECEnv(horizon=20)
    env.reset(seed=0)
    for pid in env.spec_.producer_ids:
        agent = f"prod_{pid}"
        act = decide_producing_obs(env.spec_, env.observe(agent), pid, threshold=0.6, stock_target=3)
        assert env.action_space(agent).contains(act)


def test_obs_ordering_prefers_lower_cutoff_price_and_in_bounds():
    from tokenomics.eval.baselines import decide_ordering_obs

    env = TokenomicsAECEnv(horizon=20)
    env.reset(seed=0)
    sp = env.spec_
    order_agent = next(a for a, (r, _) in env._role.items() if r == "order")
    pa, pb = sp.producer_ids[0], sp.producer_ids[1]
    # cheaper (lower cut-off price) producer should attract more routing weight
    env.system.producing_agents[pa].cut_off_price = 1.0
    env.system.producing_agents[pb].cut_off_price = 5.0
    obs = env.observe(order_agent)
    action = decide_ordering_obs(sp, obs)
    assert env.action_space(order_agent).contains(action)
    mat = action.reshape(sp.k, sp.num_producers)
    ia, ib = sp.producer_ids.index(pa), sp.producer_ids.index(pb)
    assert mat[0, ia] > mat[0, ib]


def test_obs_ordering_does_not_prefer_unproven_producer():
    # A cut-off price of 0 means "no production yet" (undefined), NOT "cheapest".
    # On competition topologies the unproven producer is often the one that cannot
    # produce (capacity is hidden, ADR-0029), so routing toward it starves assembly.
    # A proven low-price producer must outrank an unproven (price-0) one.
    from tokenomics.eval.baselines import decide_ordering_obs

    env = TokenomicsAECEnv(horizon=20)
    env.reset(seed=0)
    sp = env.spec_
    order_agent = next(a for a, (r, _) in env._role.items() if r == "order")
    proven, unproven = sp.producer_ids[0], sp.producer_ids[1]
    env.system.producing_agents[proven].cut_off_price = 2.0
    # unproven keeps the default UNDEFINED cut-off price -> encodes to 0.0
    mat = decide_ordering_obs(sp, env.observe(order_agent)).reshape(sp.k, sp.num_producers)
    assert mat[0, sp.producer_ids.index(proven)] > mat[0, sp.producer_ids.index(unproven)]


def test_observation_compatible_baseline_beats_donothing():
    # The release bar is "beat the strongest OBSERVATION-COMPATIBLE baseline". This
    # is that baseline: obs_stock_aware acts on the obs vector alone (no env.system),
    # yet must still clearly beat do-nothing — survive much longer and fulfil more
    # consumer demand — to be a meaningful bar (mirrors test_signal_exists).
    seeds = range(5)
    noop = evaluate(PRODUCING_BASELINES["noop"], ORDERING_BASELINES["even_split"],
                    seeds=seeds, horizon=100)
    good = evaluate(PRODUCING_BASELINES["obs_stock_aware"], ORDERING_BASELINES["even_split"],
                    seeds=seeds, horizon=100)
    assert good["cycles"][0] > 2 * noop["cycles"][0]
    assert good["consumer_completed"][0] > 1.5 * noop["consumer_completed"][0]


# --- multi-seed robustness summary (NEXT STEPS stage 2) ---
# The ladder requires "median and worst-case results, not one successful checkpoint"
# across independent MODEL seeds. summarize_seeds is a pure function of the per-seed
# metric dicts (no torch, no training) + a per-metric direction, so the robustness
# report is unit-testable without running PPO.
def test_summarize_seeds_reports_median_and_direction_aware_worst_case():
    from tokenomics.eval.runner import summarize_seeds

    per_seed = [
        {"return": 1000.0, "dead_rate": 0.0},
        {"return": 1200.0, "dead_rate": 0.2},
        {"return": 800.0, "dead_rate": 1.0},
    ]
    summary = summarize_seeds(
        per_seed, directions={"return": "higher", "dead_rate": "lower"}
    )
    # median is the per-metric middle value regardless of direction
    assert summary["return"].median == 1000.0
    assert summary["dead_rate"].median == 0.2
    # worst-case respects direction: lowest return, but HIGHEST dead_rate
    assert summary["return"].worst == 800.0
    assert summary["dead_rate"].worst == 1.0
    # best is the opposite extreme
    assert summary["return"].best == 1200.0
    assert summary["dead_rate"].best == 0.0
    assert summary["return"].n == 3


def test_summarize_seeds_even_count_median_and_single_seed_degenerate():
    from tokenomics.eval.runner import summarize_seeds

    # even count: median averages the two middle values (statistics.median)
    even = summarize_seeds(
        [{"return": 10.0}, {"return": 20.0}, {"return": 30.0}, {"return": 40.0}],
        directions={"return": "higher"},
    )["return"]
    assert even.median == 25.0
    assert even.worst == 10.0 and even.best == 40.0

    # single seed: median == worst == best == the lone value (no spurious spread)
    one = summarize_seeds([{"return": 7.0}], directions={"return": "higher"})["return"]
    assert one.median == one.worst == one.best == one.mean == 7.0
    assert one.n == 1
