"""Side-effect metrics in the env (design D3, add-measurement-gate-harness; side-effect-guards).

The env reports raw quantities only (no X/Y verdict, which is `eval/gate.py`'s job). The
fixtures pin each metric to a hand-computed value: a falling run-down ratio, a starved
consumer, a known inversion pair, a known spare fraction, plus the edge cases (zero
denominators -> None, zero-capacity producers, early death, no cycle played).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tokenomics.domain.common import Capacity, CapacityType, ConsumerId, Product
from tokenomics.domain.config import (
    Configuration,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
from tokenomics.domain.system import ConsumerOrderOutcome, CycleResult, ProducerAuctionRecord
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.side_effects import (
    InversionCounter,
    SideEffectTracker,
    capacity_ratio,
    fill_rates,
    spare_fraction,
)

from test_env_consumer_accounting import P, Q, _config, _run, _scheduled, _SilentC2Env

SIDE_EFFECT_KEYS = {
    "r_start", "r_end", "consumer_fill_rate",
    "inversion_inverted_pairs", "inversion_differing_pairs", "inversion_rate",
    "min_capacity_ratio", "min_spare_fraction",
    "investment_rejection_rate", "consumer_rejection_rate", "holding_units_per_cycle",
}


# --- run-down -------------------------------------------------------------------------

def test_falling_run_down_ratio() -> None:
    # One producer, max 100, 5% degradation (5/cycle), no restoration: after 4 cycles
    # current = 80, so r falls 1.0 -> 0.8 and its minimum is the last value.
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 5))), horizon=4)
    stats = _run(env, prod="noop")
    assert stats["r_start"] == 1.0
    assert stats["r_end"] == pytest.approx(0.8)
    assert stats["min_capacity_ratio"] == pytest.approx(0.8)
    assert stats["r_end"] < stats["r_start"]


def test_restoration_holds_run_down_flat() -> None:
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 1))), horizon=6)
    stats = _run(env, prod="always_restore")
    assert stats["r_start"] == stats["r_end"] == stats["min_capacity_ratio"] == 1.0
    assert stats["investment_completed"] == 6
    assert stats["investment_rejection_rate"] == 0.0


def test_early_death_r_end_is_the_death_cycle() -> None:
    # 50% of max per cycle: 100 -> 50 -> 0, death after cycle 2. r_end is read after the
    # death cycle, so the collapse shows as run-down (r_end = 0).
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 1)), degradation=50), horizon=12)
    stats = _run(env, prod="noop")
    assert stats["dead"] is True and stats["cycles"] == 2
    assert stats["r_start"] == 1.0
    assert stats["r_end"] == 0.0
    assert stats["min_capacity_ratio"] == 0.0
    # Auctions: cycle 1 at capacity 100, cycle 2 at 50 (pre-degradation), 2 units each.
    assert stats["min_spare_fraction"] == pytest.approx(1 - 2 / 50)


@dataclass
class _Producer:
    capacities: dict[CapacityType, Capacity]
    max_capacities: dict[CapacityType, Capacity]


def test_capacity_ratio_means_types_then_producers() -> None:
    a = _Producer({"x": 50, "y": 100}, {"x": 100, "y": 100})  # (0.5 + 1.0) / 2 = 0.75
    b = _Producer({"x": 0}, {"x": 40})  # 0.0
    assert capacity_ratio([a, b]) == pytest.approx(0.375)
    # A producer-mean, not a pooled-capacity mean: pooled would be 150 / 240.
    assert capacity_ratio([a, b]) != pytest.approx(150 / 240)


def test_capacity_ratio_skips_zero_max_and_empty() -> None:
    assert capacity_ratio([_Producer({"x": 0}, {"x": 0}), _Producer({"x": 3}, {"x": 4})]) == 0.75
    assert capacity_ratio([_Producer({}, {})]) is None
    assert capacity_ratio([]) is None


def test_capacity_ratio_two_producer_env() -> None:
    cfg = _two_producer_config(big=100, small=10)
    env = TokenomicsAECEnv(config=cfg, horizon=1)
    stats = _run(env, prod="noop")
    # 5% of max, ceil: big loses 5 (0.95), small loses 1 (0.9).
    assert stats["r_end"] == pytest.approx((0.95 + 0.9) / 2)


# --- starvation -----------------------------------------------------------------------

def test_starved_consumer_fill_rate_zero() -> None:
    # c2's Q x50 needs 150 capacity-1 against 100: it can never clear, so c2 starves
    # while c1 is fully served.
    cfg = _config(_scheduled("c1", (P, 1)), _scheduled("c2", (Q, 50)))
    env = TokenomicsAECEnv(config=cfg, horizon=4)
    stats = _run(env, prod="noop")
    assert stats["consumer_fill_rate"] == {"c1": 1.0, "c2": 0.0}
    # Resolved consumer units: c1 4 completed; c2's orders placed in cycles 1 and 2
    # expired (in cycles 3 and 4); those placed in cycles 3 and 4 are still open at the
    # horizon, so they are not resolved and stay out of the rate.
    assert stats["consumer_rejected_units"] == 100
    assert stats["consumer_rejection_rate"] == pytest.approx(100 / 104)


def test_zero_order_consumer_has_no_fill_rate() -> None:
    cfg = _config(_scheduled("c1", (P, 1)), _scheduled("c2", (P, 1)))
    stats = _run(_SilentC2Env(config=cfg, horizon=3))
    assert stats["consumer_fill_rate"] == {"c1": 1.0, "c2": None}


def test_fill_rates_partial() -> None:
    assert fill_rates({"a": 4, "b": 0}, {"a": 1, "b": 0}) == {"a": 0.25, "b": None}


# --- inversion ------------------------------------------------------------------------

def _out(cid: ConsumerId, product: Product, q: int, tokens: int, placed: int, done: bool) -> ConsumerOrderOutcome:
    return ConsumerOrderOutcome(cid, product, q, tokens, placed, done)


def test_known_inversion_pair() -> None:
    inv = InversionCounter()
    inv.add(_out("a", P, 2, 10, 1, True))  # completed at 5 tokens/unit
    inv.add(_out("b", P, 3, 30, 1, False))  # rejected at 10 tokens/unit -> inverted
    assert (inv.inverted, inv.differing, inv.rate) == (1, 1, 1.0)


def test_inversion_pair_rules() -> None:
    inv = InversionCounter()
    inv.add(_out("a", P, 1, 20, 1, False))  # rejected, 20/unit
    inv.add(_out("b", P, 1, 30, 1, True))  # completed 30/unit vs 20: differing, not inverted
    inv.add(_out("c", P, 2, 40, 1, True))  # completed 20/unit vs 20: equal, not counted
    inv.add(_out("d", P, 1, 99, 2, False))  # other placement cycle: no partner
    inv.add(_out("e", Q, 1, 99, 1, False))  # other product: no partner
    inv.add(_out("f", P, 1, 1, 2, False))  # rejected vs rejected: not a pair
    assert (inv.inverted, inv.differing) == (0, 1)
    assert inv.rate == 0.0
    # Arrival order does not matter: a completion resolving after the rejection pairs too.
    inv.add(_out("g", P, 1, 5, 2, True))  # 5/unit < d's 99 (inverted), > f's 1 (not)
    assert (inv.inverted, inv.differing) == (1, 3)


def test_inversion_groups_by_placement_not_resolution_cycle() -> None:
    # A completion placed in cycle 2 and a rejection placed in cycle 1 can resolve in the
    # same cycle; they never shared an auction window, so they are not a pair.
    inv = InversionCounter()
    inv.add(_out("a", P, 1, 99, 1, False))
    inv.add(_out("b", P, 1, 1, 2, True))
    assert (inv.inverted, inv.differing, inv.rate) == (0, 0, None)


def test_inversion_exact_tokens_per_unit() -> None:
    # Equal ratios with different quantities are equal (10/3 == 20/6), and integers past
    # float precision still compare exactly: 2**53 + 1 and 2**53 are one float.
    inv = InversionCounter()
    inv.add(_out("a", P, 3, 10, 1, True))
    inv.add(_out("b", P, 6, 20, 1, False))
    assert inv.differing == 0
    inv.add(_out("c", P, 1, 2**53 + 1, 2, True))
    inv.add(_out("d", P, 1, 2**53, 2, False))
    assert (inv.inverted, inv.differing) == (0, 1)


def _two_producer_config(big: int, small: int) -> Configuration:
    """Two producers of P (2 capacity-1 per unit): `big` and `small` capacity."""
    def producer(pid: str, cap: int) -> ProducingAgentConfig:
        return ProducingAgentConfig(
            pid, {"capacity-1": cap}, 5,
            Restoration(P, 50, "capacity-1"), Upgrade(P, 40, "capacity-1"),
            produces=frozenset({P, Q}),
        )
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(P, {"capacity-1": 2}), ProcessSheet(Q, {"capacity-1": 3})],
        producer_configs=[producer("big", big), producer("small", small)],
        consumers=[_scheduled("c1", (P, 1)), _scheduled("c2", (P, 5))],
    )
    cfg.validate()
    return cfg


def _route(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
    """Misrouting orderer: c1 (1 unit, the high tokens per unit) goes to `small`, which
    can never fit a P bundle; c2 (5 units, low tokens per unit) goes to `big`."""
    target = "small" if agent == "order_c1" else "big"
    scores = np.array([10.0 if pid == target else -10.0 for pid in env.spec_.producer_ids])
    return np.tile(scores, env.spec_.k_stream)


def test_known_inversion_pair_in_env() -> None:
    env = TokenomicsAECEnv(config=_two_producer_config(big=100, small=1), horizon=4)
    env.reset(seed=0)
    for agent in env.agent_iter(max_iter=1000):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        if agent.startswith("prod_"):
            env.step(np.zeros(env.action_space(agent).shape))
        else:
            env.step(_route(env, agent))
    stats = env.episode_stats
    # c2 completes every cycle at 50 tokens/unit; c1's 1-unit orders (250 tokens/unit)
    # placed in cycles 1 and 2 expire in cycles 3 and 4 -> one inverted pair per placement
    # cycle. c1's cycle-3 and cycle-4 orders are still open at the horizon: unpaired.
    assert stats["consumer_fill_rate"] == {"c1": 0.0, "c2": 1.0}
    assert stats["inversion_inverted_pairs"] == 2
    assert stats["inversion_differing_pairs"] == 2
    assert stats["inversion_rate"] == 1.0


# --- spare fraction -------------------------------------------------------------------

def test_known_spare_fraction() -> None:
    # Capacity 100 at the auction; P x5 = 10 capacity-1 accepted -> spare 0.9.
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 5))), horizon=1)
    stats = _run(env, prod="noop")
    assert stats["min_spare_fraction"] == pytest.approx(0.9)


def test_spare_fraction_minimum_over_cycles() -> None:
    # Capacity at auction falls 100, 95, 90, 85 (degradation lands after the auction)
    # while 10 units clear each cycle: the minimum is the last, 1 - 10/85.
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 5))), horizon=4)
    stats = _run(env, prod="noop")
    assert stats["min_spare_fraction"] == pytest.approx(1 - 10 / 85)


def test_auction_records_in_domain_result() -> None:
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 5))), horizon=1)
    env.reset(seed=0)
    results: list[CycleResult] = []
    complete = env.system.complete_cycle

    def recording() -> CycleResult:
        results.append(complete())
        return results[-1]

    env.system.complete_cycle = recording  # type: ignore[method-assign]
    _drive(env)
    assert results[0].producer_auctions == (ProducerAuctionRecord("p1", 100, 10),)


def test_spare_fraction_zero_capacity_skipped() -> None:
    assert spare_fraction(ProducerAuctionRecord("p", 0, 0)) is None
    assert spare_fraction(ProducerAuctionRecord("p", 40, 10)) == 0.75
    tracker = SideEffectTracker(r_start=1.0)
    tracker.observe_cycle(CycleResult(0, producer_auctions=(ProducerAuctionRecord("p", 0, 0),)), 0.5)
    assert tracker.min_spare_fraction is None
    tracker.observe_cycle(CycleResult(0, producer_auctions=(
        ProducerAuctionRecord("p", 0, 0), ProducerAuctionRecord("q", 20, 15),
    )), 0.4)
    assert tracker.min_spare_fraction == 0.25
    assert (tracker.r_end, tracker.min_capacity_ratio) == (0.4, 0.4)


def test_zero_capacity_producer_in_env_skipped() -> None:
    # `small` (max 1) degrades to 0 after cycle 1 and keeps auctioning at capacity 0;
    # those producer-cycles are skipped, so the minimum comes from real auctions.
    env = TokenomicsAECEnv(config=_two_producer_config(big=100, small=1), horizon=3)
    stats = _run(env, prod="noop")
    assert stats["min_spare_fraction"] is not None
    assert 0.0 <= stats["min_spare_fraction"] <= 1.0


# --- zero denominators / bookkeeping --------------------------------------------------

def _drive(env: TokenomicsAECEnv) -> None:
    for agent in env.agent_iter(max_iter=1000):
        _, _, term, trunc, _ = env.last()
        env.step(None if term or trunc else np.zeros(env.action_space(agent).shape))


def test_keys_present_after_reset_with_none_rates() -> None:
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 1))), horizon=3)
    env.reset(seed=0)
    s = env.episode_stats
    assert SIDE_EFFECT_KEYS <= set(s)
    assert s["r_start"] == 1.0
    for k in ("r_end", "min_capacity_ratio", "min_spare_fraction", "inversion_rate",
              "investment_rejection_rate", "consumer_rejection_rate", "holding_units_per_cycle"):
        assert s[k] is None, k
    assert s["consumer_fill_rate"] == {"c1": None}
    assert (s["inversion_inverted_pairs"], s["inversion_differing_pairs"]) == (0, 0)


def test_no_investment_gives_none_rate_not_zero() -> None:
    stats = _run(TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 1))), horizon=3), prod="noop")
    assert stats["investment_completed"] == stats["investment_rejected"] == 0
    assert stats["investment_rejection_rate"] is None
    assert stats["consumer_rejection_rate"] == 0.0


def test_phase1_rates_match_totals() -> None:
    env = TokenomicsAECEnv(horizon=12)
    stats = _run(env, prod="restore_when_low", order="random")
    assert stats["investment_rejection_rate"] == pytest.approx(
        stats["investment_rejected"] / (stats["investment_completed"] + stats["investment_rejected"])
    )
    assert stats["holding_units_per_cycle"] == pytest.approx(stats["holding_units"] / stats["cycles"])
    assert stats["min_capacity_ratio"] <= stats["r_end"]
    assert 0 <= stats["inversion_inverted_pairs"] <= stats["inversion_differing_pairs"]


def test_reset_restarts_tracker() -> None:
    env = TokenomicsAECEnv(config=_config(_scheduled("c1", (P, 5))), horizon=4)
    first = dict(_run(env, prod="noop"))
    second = _run(env, prod="noop")
    for k in SIDE_EFFECT_KEYS:
        assert first[k] == second[k], k
