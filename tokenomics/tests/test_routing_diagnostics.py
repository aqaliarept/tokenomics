"""Diagnostic routers and the EXP-0048 driver (add-m3-development-audit task 1.2, design D2;
development-audit "Diagnostic routers stay outside the roster").

Fixture: a three-producer economy. p1 and p2 both make primal product 10 (fanout 2); p3 makes
composite 20 from one unit of 10, so its replenishment orders for 10 are producer orders over
the same two producers. Consumer c1 orders `quantity` units of 10 every cycle; c2 (optional)
orders one 20.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from tokenomics.domain.config import (ConsumerConfig, Configuration, ProcessSheet,
                                      ProducingAgentConfig, Restoration, Upgrade)
from tokenomics.domain.system import order_kind
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.allocator import allocate_suborders
from tokenomics.env.decode import decode_ordering
from tokenomics.eval.baselines import (ORDERING_BASELINES, PRODUCING_BASELINES, Policy,
                                       ordering_blend, ordering_even_split, random_action)
from tokenomics.eval.contract import ContractMonitor, read_failures
from tokenomics.eval.gate import PARAMS_V2, load_side_effect_params
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.eval.roster import JOINT_STACK, ROSTER_V2, load_roster
from tokenomics.eval.routing_audit import (EDGE_COLUMNS, build_jobs, edge_table,
                                           real_run_problems, router_table, run_audit,
                                           select_worlds)
from tokenomics.eval.routing_diagnostics import (DIAGNOSTIC_ROUTERS, ROUTER_KIND, STOCHASTIC,
                                                 ordering_random_scaled, split_logits)
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.routing_trace import CONSUMER, PRODUCER, RoutingTracer, order_class
from tokenomics.eval.runner import run_episode
from tokenomics.eval.seeding import replicate_seed
from tokenomics.eval.shock_profiles import SHOCKS_I_SMOKE

HORIZON = 8
PRODUCER_IDS = ("p1", "p2")


def _config(quantity: int = 3, *, composite: bool = False, degradation: int = 5,
            capacities: tuple[int, int] = (120, 120)) -> Configuration:
    sheets = [ProcessSheet(10, {"capacity-1": 2}),
              ProcessSheet(20, {"capacity-2": 2}, inputs={10: 1})]

    def prod(pid: str, cap: str, capacity: int, produces: int, **kw: object) -> ProducingAgentConfig:
        return ProducingAgentConfig(
            id=pid, capacities={cap: capacity}, degradation=degradation,
            restoration=Restoration(require=10, restores=60, type=cap, cost=40),
            upgrade=Upgrade(require=10, increases=40, type=cap, cost=40),
            produces=frozenset({produces}), **kw)  # type: ignore[arg-type]

    producers = [prod("p1", "capacity-1", capacities[0], 10),
                 prod("p2", "capacity-1", capacities[1], 10),
                 prod("p3", "capacity-2", 120, 20, inventory_limit=30, max_replenish=10,
                      replenish_cost_per_unit=2)]
    consumers = [ConsumerConfig(id="c1", product=10, schedule=((10, quantity),))]
    if composite:
        consumers.append(ConsumerConfig(id="c2", product=20, schedule=((20, 1),)))
    cfg = Configuration(1000, sheets, producers, consumers)
    cfg.validate()
    return cfg


def _shocked_world_config() -> Configuration:
    entry = select_worlds(load_registry(REGISTRY_V2), sizes=[12], regimes=["shocked"],
                          max_worlds=1, shock_profile=SHOCKS_I_SMOKE)[0]
    return entry.config()


Record = dict[str, object]


def _record(cfg: Configuration, router: Policy, *, action_seed: int | None = None,
            horizon: int = HORIZON, also: dict[str, Policy] | None = None,
            ) -> tuple[list[Record], dict]:
    """Run one joint-stack episode with `entity_planner` and `router`, under the contract
    monitor; record, per routed order and call, the decoded split of `router`'s action (the
    split the env applies) and of each policy in `also` evaluated on the same state."""
    env = joint_stack_env(cfg, horizon)
    log: list[Record] = []

    def order(e: TokenomicsAECEnv, agent: str) -> np.ndarray:
        _, aid = e._role[agent]
        view = e.system.ordering_agents[aid].view(e.system.producer_infos)
        others = {n: p(e, agent) for n, p in (also or {}).items()}  # before the router draws
        act = router(e, agent)
        cmd = decode_ordering(e.spec_, np.asarray(act), view)
        alt = {n: decode_ordering(e.spec_, np.asarray(a), view) for n, a in others.items()}
        mat = np.asarray(act).reshape(e.spec_.k_stream, e.spec_.num_producers)
        for slot, oid in enumerate(sorted(view.incoming)):
            oi = view.incoming[oid]
            log.append({
                "cycle": e.system.cycle_counter, "agent": agent, "oid": oid, "slot": slot,
                "q": oi.quantity, "tokens": oi.tokens, "eligible": set(view.eligible[oid]),
                "class": order_class(order_kind(e.system.orders[oid])),
                "subs": [(s.producer_id, s.quantity, s.tokens) for s in cmd.orders[oid]],
                "row": mat[slot].copy() if slot < e.spec_.k_stream else None,
                "alt": {n: [(s.producer_id, s.quantity, s.tokens) for s in c.orders[oid]]
                        for n, c in alt.items()},
                "alt_row": {n: np.asarray(a).reshape(mat.shape)[slot].copy()
                            for n, a in others.items() if slot < e.spec_.k_stream},
                "throughput": {pid: p.capacities.get("capacity-1", 0)
                               - p.reserved_capacity().get("capacity-1", 0)
                               for pid, p in e.system.producing_agents.items()
                               if pid in PRODUCER_IDS}})
        return act

    monitor = ContractMonitor(env)
    stats = run_episode(env, PRODUCING_BASELINES["entity_planner"], order, seed=0,
                        action_seed=action_seed, monitor=monitor)
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=horizon)
    return log, stats


def _c1(log: list[Record]) -> list[Record]:
    return [r for r in log if r["class"] == CONSUMER and str(r["agent"]).startswith("order_c1")]


# --- the exact-split helper ---


def _compositions(q: int, n: int):  # noqa: ANN202 - generator of int tuples
    if n == 1:
        yield (q,)
        return
    for i in range(q + 1):
        for rest in _compositions(q - i, n - 1):
            yield (i,) + rest


def test_split_logits_decode_to_the_exact_split() -> None:
    """Every split of q <= 7 over up to 5 producers, and whole / even splits over up to 14
    producers and q <= 20, decode to exactly the requested units."""
    targets = [u for n in range(1, 6) for q in range(1, 8) for u in _compositions(q, n)]
    for n in range(1, 15):
        for q in range(1, 21):
            targets.append(tuple(q if j == n - 1 else 0 for j in range(n)))
            base, rest = divmod(q, n)
            targets.append(tuple(base + (1 if j >= n - rest else 0) for j in range(n)))
    for units in targets:
        n, q = len(units), sum(units)
        ids = [f"p{j}" for j in range(n)]
        subs = allocate_suborders(ids, split_logits(units), np.ones(n, dtype=bool), q, 7 * q + 3)
        got = dict.fromkeys(ids, 0)
        for s in subs:
            got[s.producer_id] = s.quantity
        assert tuple(got.values()) == units


# --- every router ---


@pytest.mark.parametrize("name", sorted(DIAGNOSTIC_ROUTERS))
def test_router_conserves_and_routes_only_to_eligible(name: str) -> None:
    """Quantity and tokens conserved, only eligible producers, actions inside the action space
    (the contract monitor), on the fixture and on a shocked registry world."""
    for cfg, horizon in ((_config(3, composite=True), HORIZON), (_shocked_world_config(), 4)):
        log, _ = _record(cfg, DIAGNOSTIC_ROUTERS[name], action_seed=11, horizon=horizon)
        assert log
        for r in log:
            subs = r["subs"]
            assert sum(q for _, q, _ in subs) == r["q"]
            assert sum(t for _, _, t in subs) == r["tokens"]
            assert {p for p, _, _ in subs} <= r["eligible"]


def test_diagnostic_router_kinds() -> None:
    assert sorted(DIAGNOSTIC_ROUTERS) == sorted([
        "round_robin", "even_split_rotating", "proportional_load_aware", "proportional_whole",
        "random_scaled_0.5", "random_scaled_1", "random_scaled_2", "random_scaled_5",
        "random_fixed_episode", "blend_noise_0.15", "blend_noise_0.6", "random_consumer_only",
        "random_producer_only", "random_whole", "even_split_random_leftover"])
    assert sorted(n for n, k in ROUTER_KIND.items() if k != STOCHASTIC) == [
        "even_split_rotating", "proportional_load_aware", "proportional_whole", "round_robin"]


# --- deterministic routers ---


def test_round_robin_rotates_a_quantity_one_order() -> None:
    log, _ = _record(_config(1), DIAGNOSTIC_ROUTERS["round_robin"])
    first = {r["cycle"]: r["subs"] for r in _c1(log) if r["slot"] == 0}
    cycles = sorted(first)
    assert len(cycles) >= 4
    for subs in first.values():
        assert len(subs) == 1 and subs[0][1] == 1
    consecutive = [(a, b) for a, b in zip(cycles, cycles[1:]) if b == a + 1]
    assert len(consecutive) >= 3
    for a, b in consecutive:
        assert first[a][0][0] != first[b][0][0]


def test_even_split_rotating_differs_only_in_leftover_placement() -> None:
    log, _ = _record(_config(3), DIAGNOSTIC_ROUTERS["even_split_rotating"],
                     also={"even_split": ordering_even_split})
    leftover = set()
    for r in _c1(log):
        mine = sorted(q for _, q, _ in r["subs"])
        theirs = sorted(q for _, q, _ in r["alt"]["even_split"])
        assert mine == theirs == [1, 2]
        leftover.add(next(p for p, q, _ in r["subs"] if q == 2))
        assert next(p for p, q, _ in r["alt"]["even_split"] if q == 2) == "p1"  # lowest index
    assert leftover == {"p1", "p2"}  # the rotation moves the odd unit
    log2, _ = _record(_config(2), DIAGNOSTIC_ROUTERS["even_split_rotating"],
                      also={"even_split": ordering_even_split})
    for r in _c1(log2):  # no leftover: identical to even_split
        assert sorted(r["subs"]) == sorted(r["alt"]["even_split"])


def test_whole_routers_never_split() -> None:
    """One sub-order per order (quantity 3, fanout 2); the tracer's consumer split degree is
    1.0 for the whole routers and 2.0 for even_split on the same fixture."""
    for name in ("proportional_whole", "random_whole", "round_robin"):
        log, _ = _record(_config(3), DIAGNOSTIC_ROUTERS[name], action_seed=5)
        assert _c1(log) and all(len(r["subs"]) == 1 for r in _c1(log))
    for name, policy, degree in (
            ("proportional_whole", DIAGNOSTIC_ROUTERS["proportional_whole"], 1.0),
            ("random_whole", DIAGNOSTIC_ROUTERS["random_whole"], 1.0),
            ("even_split", ordering_even_split, 2.0)):
        env = joint_stack_env(_config(3), HORIZON)
        tracer = RoutingTracer()
        tracer.attach(env)
        run_episode(env, PRODUCING_BASELINES["entity_planner"], policy, seed=0, action_seed=5)
        assert tracer.summary()["split_degree_consumer"] == degree, name


def test_proportional_whole_constant_for_unchanged_throughputs() -> None:
    log, _ = _record(_config(3, degradation=0, capacities=(200, 120)),
                     DIAGNOSTIC_ROUTERS["proportional_whole"])
    by_throughput: dict[tuple, set[str]] = defaultdict(set)
    for r in _c1(log):
        by_throughput[tuple(sorted(r["throughput"].items()))].add(r["subs"][0][0])
    assert len(_c1(log)) >= HORIZON - 1
    assert all(len(v) == 1 for v in by_throughput.values())
    assert set().union(*by_throughput.values()) == {"p1"}  # the larger throughput


def test_proportional_load_aware_nets_bids_already_placed() -> None:
    """Two consumers order 3 units of 10 in one cycle from two 12-capacity producers. The
    consumer stream that routes second sees the first one's bids: its logit for a producer the
    first stream bid on drops below `proportional`'s (which nets only reservations)."""
    cfg = _config(3, capacities=(12, 12))
    cfg = Configuration(cfg.cycle_emission, cfg.process_sheets, cfg.producer_configs,
                        cfg.consumers + [ConsumerConfig(id="c3", product=10,
                                                        schedule=((10, 3),))])
    log, _ = _record(cfg, DIAGNOSTIC_ROUTERS["proportional_load_aware"],
                     also={"proportional": ORDERING_BASELINES["proportional"]})
    first_cycle = min(r["cycle"] for r in log)
    rows = [r for r in log if r["cycle"] == first_cycle and r["class"] == CONSUMER]
    first, second = rows[0], rows[1]
    assert first["agent"] != second["agent"] and first["subs"]
    for pid, q, _ in first["subs"]:
        i = PRODUCER_IDS.index(pid)  # producer index = position in the config
        assert q > 0 and second["row"][i] < second["alt_row"]["proportional"][i]


# --- stochastic routers ---


def _actions(cfg: Configuration, router: Policy, action_seed: int) -> list[np.ndarray]:
    log, _ = _record(cfg, router, action_seed=action_seed)
    return [r["row"] for r in log if r["row"] is not None]


def test_random_scaled_10_is_random() -> None:
    seed = replicate_seed("w", 1, 0)
    a = _actions(_config(3, composite=True), ordering_random_scaled(10), seed)
    b = _actions(_config(3, composite=True), random_action, seed)
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
    small = _actions(_config(3, composite=True), DIAGNOSTIC_ROUTERS["random_scaled_0.5"], seed)
    assert all(np.abs(x).max() <= 0.5 for x in small)


def test_random_fixed_episode_held_within_and_varies_across_replicates() -> None:
    def per_agent(seed: int) -> dict[str, list[np.ndarray]]:
        log, _ = _record(_config(3, composite=True), DIAGNOSTIC_ROUTERS["random_fixed_episode"],
                         action_seed=seed)
        out: dict[str, list[np.ndarray]] = defaultdict(list)
        for r in log:
            if r["slot"] == 0:
                out[str(r["agent"])].append(r["row"])
        return out

    a, b = per_agent(replicate_seed("w", 1, 0)), per_agent(replicate_seed("w", 1, 1))
    assert any(len(rows) > 1 for rows in a.values())
    for rows in a.values():
        assert all(np.array_equal(rows[0], x) for x in rows)
    assert any(not np.array_equal(a[k][0], b[k][0]) for k in a.keys() & b.keys())


@pytest.mark.parametrize("name,randomised", [("random_consumer_only", CONSUMER),
                                             ("random_producer_only", PRODUCER)])
def test_one_class_routers_randomise_only_their_class(name: str, randomised: str) -> None:
    log, _ = _record(_config(3, composite=True), DIAGNOSTIC_ROUTERS[name], action_seed=3,
                     also={"blend_0.5": ordering_blend(0.5)})
    seen = defaultdict(int)
    for r in log:
        if r["row"] is None:
            continue
        seen[r["class"]] += 1
        same = np.array_equal(r["row"], r["alt_row"]["blend_0.5"])
        assert same == (r["class"] != randomised), (r["class"], r["agent"])
    assert seen[CONSUMER] and seen[PRODUCER]


def test_random_whole_draws_its_producer() -> None:
    """Fanout 2: within an episode both producers receive whole orders, and the sequence of
    receivers differs across replicate seeds (not a fixed lowest-index choice)."""
    def receivers(seed: int) -> list[str]:
        log, _ = _record(_config(3), DIAGNOSTIC_ROUTERS["random_whole"], action_seed=seed)
        return [r["subs"][0][0] for r in _c1(log)]

    a, b = receivers(replicate_seed("w", 1, 0)), receivers(replicate_seed("w", 1, 1))
    assert len(a) >= HORIZON - 1 and set(a) == {"p1", "p2"}
    assert a != b


@pytest.mark.parametrize("sigma", [0.15, 0.6])
def test_blend_noise_is_blend_plus_gaussian(sigma: float) -> None:
    """Each action equals `blend_0.5` + one N(0, sigma^2) draw from the action-space RNG
    (replayed from the saved RNG state), and the noise's sample std is ~sigma."""
    router = DIAGNOSTIC_ROUTERS[f"blend_noise_{sigma}"]
    blend = ordering_blend(0.5)
    noise: list[np.ndarray] = []

    def checked(e: TokenomicsAECEnv, agent: str) -> np.ndarray:
        rng = e.action_space(agent).np_random
        state = rng.bit_generator.state
        act = router(e, agent)
        after = rng.bit_generator.state
        rng.bit_generator.state = state
        base = blend(e, agent)
        draw = rng.normal(0.0, sigma, size=base.shape)
        assert rng.bit_generator.state == after  # the router drew exactly this
        np.testing.assert_array_equal(
            act, np.clip(base + draw, -10, 10).astype(np.float32))
        inside = np.abs(base) < 10 - 5 * sigma  # unclipped entries
        noise.append((act - base)[inside])
        return act

    _record(_config(3, composite=True), checked, action_seed=replicate_seed("w", 1, 0))
    sample = np.concatenate(noise)
    assert sample.size > 100
    assert abs(float(sample.std()) - sigma) < 0.2 * sigma
    assert abs(float(sample.mean())) < 0.2 * sigma


def test_even_split_random_leftover() -> None:
    for r in _c1(_record(_config(2), DIAGNOSTIC_ROUTERS["even_split_random_leftover"],
                         action_seed=1)[0]):
        assert sorted(q for _, q, _ in r["subs"]) == [1, 1]

    def odd_units(seed: int) -> list[str]:
        log, _ = _record(_config(3), DIAGNOSTIC_ROUTERS["even_split_random_leftover"],
                         action_seed=seed)
        out = []
        for r in _c1(log):
            assert sorted(q for _, q, _ in r["subs"]) == [1, 2]
            out.append(next(p for p, q, _ in r["subs"] if q == 2))
        return out

    a, b = odd_units(replicate_seed("w", 1, 0)), odd_units(replicate_seed("w", 1, 1))
    assert a != b
    assert set(a) == {"p1", "p2"}


# --- outside the roster ---

FROZEN_V2 = {  # EXP-0047 frozen set (tests/test_frozen_v2.py)
    "worlds": "7b09f536e9880bc31370694f80eaca828587621eb5daa16fdf64e80ed05ee7ff",
    "roster": "5050508ecb25f0eda0ae0a61b0dde4ba948a6e9b0d901cd14cde28e0db161f38",
    "params": "e2aee97ac38524d709862bc956c43940413ac0cb44c846304a4b8c3d324c3a99",
}
SIGNAL_CHECK = [sys.executable, "-m", "tokenomics.eval.signal_check", "--seeds", "1",
                "--horizon", "12"]


def test_diagnostic_router_added() -> None:
    """Scenario "Diagnostic router added": frozen v2 content hashes unchanged, the baseline
    dictionaries unchanged and disjoint from the diagnostic routers, and signal_check's rows
    identical with and without the diagnostic modules imported."""
    assert load_registry(REGISTRY_V2).version == FROZEN_V2["worlds"]
    assert load_roster(ROSTER_V2).version == FROZEN_V2["roster"]
    assert load_side_effect_params(PARAMS_V2).version == FROZEN_V2["params"]
    assert list(ORDERING_BASELINES) == ["even_split", "proportional", "obs_price", "blend_0.5",
                                        "shock_reactive", "random"]
    assert list(PRODUCING_BASELINES) == ["noop", "always_restore", "always_both",
                                         "restore_when_low", "stock_aware", "entity_planner",
                                         "obs_stock_aware", "random"]
    assert not set(DIAGNOSTIC_ROUTERS) & set(ORDERING_BASELINES)
    root = Path(__file__).resolve().parents[1]
    plain = subprocess.run(SIGNAL_CHECK, cwd=root, capture_output=True, text=True, check=True)
    loaded = subprocess.run(
        [sys.executable, "-c", "import sys, runpy; import tokenomics.eval.routing_audit; "
         "sys.argv = ['signal_check', '--seeds', '1', '--horizon', '12']; "
         "runpy.run_module('tokenomics.eval.signal_check', run_name='__main__')"],
        cwd=root, capture_output=True, text=True, check=True)
    assert plain.stdout == loaded.stdout and plain.stdout.count("\n") == 13


# --- the driver ---


def test_job_list_contracts() -> None:
    """Each roster member runs the contract its roster_v2.json entry records; every diagnostic
    router joint_stack; deterministic once, stochastic R times with the gate's seeds."""
    roster = load_roster(ROSTER_V2)
    recorded = {(m["producing"], m["ordering"]): m["contract"]
                for m in json.loads(ROSTER_V2.read_text())["members"]}
    assert recorded[("entity_planner", "obs_price")] == "legacy"
    table = router_table(roster)
    world = select_worlds(load_registry(REGISTRY_V2), sizes=[12], max_worlds=1)[:1]
    jobs = build_jobs(world, table, roster.replicates, 100, episodes_per_world=1)
    assert len(jobs) == 9 + 12 * 5  # 69 runs per world-episode (D2)
    for j in jobs:
        if j["ordering"] in DIAGNOSTIC_ROUTERS:
            assert j["contract"] == JOINT_STACK
        else:
            assert j["contract"] == recorded[(j["producing"], j["ordering"])]
        stochastic = j["ordering"] == "random" or ROUTER_KIND.get(j["ordering"]) == STOCHASTIC
        assert j["action_seed"] == (replicate_seed(j["world"], j["episode"], j["replicate"])
                                    if stochastic else None)
    assert {j["ordering"] for j in jobs if j["contract"] != JOINT_STACK} == {"obs_price"}


def _row(world: str, size: int, regime: str, episode: int, router: str, replicate: int,
         units: float, **trace: float) -> dict[str, str]:
    from tokenomics.eval.routing_audit import TRACE_PREFIX, TRACE_SCALARS

    d = {"world": world, "size": str(size), "regime": regime, "episode": str(episode),
         "subject": f"entity_planner/{router}", "replicate": str(replicate),
         "units": repr(float(units)), "dead": "false"}
    d.update({TRACE_PREFIX + k: repr(float(trace.get(k, 1.0))) for k in TRACE_SCALARS})
    return d


def test_edge_arithmetic_and_the_two_of_three_rule() -> None:
    """Equal-weight world means: world A has 2 episodes, world B one. Edge per episode =
    random's replicate mean minus the best deterministic member."""
    roster = load_roster(ROSTER_V2)
    table = router_table(roster, ["even_split", "proportional", "random", "round_robin"])
    rows: list[dict[str, str]] = []
    cells = {(12, "shocked"): 0.9, (42, "calm"): 0.5, (42, "shocked"): 0.85}
    for (size, regime), share in cells.items():
        for world, episodes in ((f"A{size}{regime}", (1, 2)), (f"B{size}{regime}", (1,))):
            for ep in episodes:
                base = 10.0 * ep
                rows += [_row(world, size, regime, ep, "even_split", 0, base),
                         _row(world, size, regime, ep, "proportional", 0, base - 4),
                         _row(world, size, regime, ep, "random", 0, base + 2 * ep + 1),
                         _row(world, size, regime, ep, "random", 1, base + 2 * ep - 1),
                         _row(world, size, regime, ep, "round_robin", 0,
                              base + share * 2 * ep)]
    out = edge_table(rows, table)
    assert set().union(*map(set, out)) <= set(EDGE_COLUMNS)
    ep_rows = [r for r in out if r["level"] == "episode" and r["router"] == "random"]
    assert {r["edge"] for r in ep_rows} == {2.0, 4.0}
    cell = {(r["size"], r["regime"], r["router"]): r for r in out if r["level"] == "cell"}
    # edge world means: A = (2 + 4) / 2 = 3, B = 2 -> 2.5 (not the episode mean 8/3)
    assert cell[(12, "shocked", "random")]["edge"] == 2.5
    assert cell[(12, "shocked", "round_robin")]["recovered_fraction"] == pytest.approx(0.9)
    assert cell[(42, "calm", "round_robin")]["recovers"] is False
    rule = {r["router"]: r for r in out if r["level"] == "rule"}
    assert rule["round_robin"]["cells_recovered"] == 2 and rule["round_robin"]["recovers_most"]
    assert rule["even_split"]["cells_recovered"] == 0
    assert cell[(12, "shocked", "random")]["worst_beats_count"] == 3  # min replicate > best
    split = [r for r in out if r["level"] == "split_2x2"]
    assert {r["router"] for r in split} == {"round_robin", "even_split"}


def test_rule_boundaries() -> None:
    """At exactly 0.8 a router recovers the edge (D2 "at least 0.8"); one recovered dominant cell
    is not "most"; a det/random split-degree gap above the tolerance is not matched."""
    roster = load_roster(ROSTER_V2)
    names = ["even_split", "proportional", "random", "round_robin", "proportional_whole",
             "random_whole", "even_split_random_leftover"]
    table = router_table(roster, names)
    gains = {  # cell -> router -> gain over the best deterministic member (edge 5)
        (12, "shocked"): {"round_robin": 4, "proportional_whole": 4},
        (42, "calm"): {"round_robin": 3.95, "proportional_whole": 1},
        (42, "shocked"): {"round_robin": 0, "proportional_whole": 0}}
    degree = {"even_split": 2.0, "even_split_random_leftover": 2.0, "random_whole": 1.5}
    rows: list[dict[str, str]] = []
    for (size, regime), g in gains.items():
        w = f"w{size}{regime}"
        units = {"even_split": 10, "proportional": 6, "random": 15, "random_whole": 10,
                 "even_split_random_leftover": 10, **{k: 10 + v for k, v in g.items()}}
        rows += [_row(w, size, regime, 1, n, 0, u, split_degree_consumer=degree.get(n, 1.0))
                 for n, u in units.items()]
    out = edge_table(rows, table)
    cell = {(r["size"], r["regime"], r["router"]): r for r in out if r["level"] == "cell"}
    assert cell[(12, "shocked", "round_robin")]["recovered_fraction"] == 0.8
    assert cell[(12, "shocked", "round_robin")]["recovers"] is True
    assert cell[(42, "calm", "round_robin")]["recovers"] is False  # 0.79
    rule = {r["router"]: r for r in out if r["level"] == "rule"}
    assert rule["round_robin"]["cells_recovered"] == 1
    assert rule["round_robin"]["recovers_most"] is False
    split = {(r["size"], r["regime"], r["router"]): r for r in out if r["level"] == "split_2x2"}
    whole = split[(12, "shocked", "random_whole")]
    assert whole["split_degree_gap"] == 0.5 and whole["matched_split_degree"] is False
    assert split[(12, "shocked", "round_robin")]["matched_split_degree"] is False
    assert split[(12, "shocked", "even_split")]["split_degree_gap"] == 0.0
    assert split[(12, "shocked", "even_split")]["matched_split_degree"] is True


def _real_kwargs() -> dict:
    registry = load_registry(REGISTRY_V2)
    return dict(registry=registry, roster=load_roster(ROSTER_V2),
                worlds=select_worlds(registry), horizon=100, replicates=5,
                source_commit="a" * 40,
                filters={k: None for k in ("sizes", "regimes", "max_worlds",
                                           "episodes_per_world", "routers", "shock_profile")})


def test_real_run_clean_configuration_has_no_problem() -> None:
    assert real_run_problems(**_real_kwargs()) == []


@pytest.mark.parametrize("change,needle", [
    ("unfrozen", "is not frozen"),
    ("commit", "40-hex source commit"),
    ("horizon", "horizon 12 != 100"),
    ("replicates", "R = 1 != roster R 5"),
    ("unscreened", "has not screened the development split"),
    ("smoke", "test-only shock profile"),
    ("mixed", "mixed versions"),
    *[(f"filter:{k}", f"--{k.replace('_', '-')} is set") for k in (
        "sizes", "regimes", "max_worlds", "episodes_per_world", "routers", "shock_profile")],
])
def test_real_run_guards(change: str, needle: str) -> None:
    import dataclasses

    from tokenomics.eval.registry import build_worlds_v2
    from tokenomics.eval.roster import ROSTER_V1

    kw = _real_kwargs()
    if change == "unfrozen":
        kw["roster"] = dataclasses.replace(kw["roster"], frozen=False, screened_on=None,
                                           pruning=None)
    elif change == "commit":
        kw["source_commit"] = "abc123"
    elif change == "horizon":
        kw["horizon"] = 12
    elif change == "replicates":
        kw["replicates"] = 1
    elif change == "unscreened":
        kw["registry"] = build_worlds_v2()
    elif change == "smoke":
        kw["worlds"] = select_worlds(kw["registry"], shock_profile=SHOCKS_I_SMOKE)
    elif change == "mixed":
        kw["roster"] = load_roster(ROSTER_V1)
    else:
        kw["filters"][change.split(":", 1)[1]] = [1]
    problems = real_run_problems(**kw)
    assert any(needle in p for p in problems), problems


def test_dry_run_writes_the_exp_layout(tmp_path: Path) -> None:
    """The task's dry run: 1 world per size (shocked), 1 episode, horizon 12, shocks-i-smoke,
    3 routers, R = 1, workers 1, in under 2 min; a rerun resumes with nothing to run; a real run
    with filters is refused."""
    kw: dict = dict(registry=load_registry(REGISTRY_V2), roster=load_roster(ROSTER_V2),
                    workers=1, horizon=12, replicates=1, regimes=["shocked"], max_worlds=1,
                    episodes_per_world=1, routers=["even_split", "random", "round_robin"],
                    shock_profile=SHOCKS_I_SMOKE)
    logs: list[str] = []
    start = time.perf_counter()
    assert run_audit(out_dir=tmp_path / "exp", dry_run=True, log=logs.append, **kw) == 0
    assert time.perf_counter() - start < 120
    out = tmp_path / "exp"
    for name in ("spec.json", "episodes.csv", "routing_metrics.csv", "edge.csv",
                 "contract_failures.csv", "RESULT.md"):
        assert (out / name).exists(), name
    assert read_failures(out) == []
    spec = json.loads((out / "spec.json").read_text())
    assert spec["status"] == "complete" and spec["kind"] == "dry-run"
    assert spec["pins"]["shock_profile_override"] == SHOCKS_I_SMOKE
    episodes = (out / "episodes.csv").read_text().splitlines()
    assert len(episodes) == 1 + 2 * 3 and "trace_split_degree_consumer" in episodes[0]
    assert "DRY RUN" in (out / "RESULT.md").read_text()
    assert any(line.startswith("rule,") for line in (out / "edge.csv").read_text().splitlines())
    logs.clear()
    assert run_audit(out_dir=out, dry_run=True, log=logs.append, **kw) == 0
    assert any("0 to run" in line for line in logs)
    with pytest.raises(ValueError, match="not a valid EXP-0048 run"):
        run_audit(out_dir=tmp_path / "real", dry_run=False, log=logs.append, **kw)


def test_failed_job_makes_the_run_invalid(tmp_path: Path) -> None:
    from tokenomics.eval.routing_audit import run_job

    def flaky(job: dict) -> dict[str, str]:
        if job["ordering"] == "round_robin":
            job = {**job, "ordering": "no_such_router"}
        return run_job(job)

    code = run_audit(out_dir=tmp_path / "exp", registry=load_registry(REGISTRY_V2),
                     roster=load_roster(ROSTER_V2), horizon=4, replicates=1, sizes=[12],
                     regimes=["calm"], max_worlds=1, episodes_per_world=1,
                     routers=["even_split", "random", "round_robin"], dry_run=True,
                     run_job_fn=flaky, log=lambda s: None)
    assert code == 2
    assert len(read_failures(tmp_path / "exp")) == 1
    assert json.loads((tmp_path / "exp" / "spec.json").read_text())["status"] == "invalid"
