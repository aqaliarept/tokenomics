"""EXP-0007A: stable product-stream ordering agents (ADR-0052).

Behaviour-preservation tests. Splitting a source's orders across per-product
ordering agents must not change per-order routing, so deterministic baselines
must produce bit-identical episode metrics under the product-stream contract.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pettingzoo.test import api_test

from tokenomics.domain.config import load_config
from tokenomics.env.aec_env import TokenomicsAECEnv, DEFAULT_CONFIG
from tokenomics.env.spaces import (
    STREAM_PRODUCER_FEATURES,
    STREAM_PRODUCT_FEATURES,
    STREAM_SLOT_SCALARS,
    build_spec,
    config_fingerprint,
)
from tokenomics.eval.baselines import PRODUCING_BASELINES, ORDERING_BASELINES
from tokenomics.eval.runner import run_episode


def _stream_obs_dim(P: int, K: int) -> int:
    return STREAM_PRODUCT_FEATURES + K * STREAM_SLOT_SCALARS + P * STREAM_PRODUCER_FEATURES + 1


def _episode(order_policy: str, *, product_stream: bool) -> dict:
    env = TokenomicsAECEnv(horizon=100, product_stream_ordering=product_stream)
    return run_episode(
        env,
        PRODUCING_BASELINES["stock_aware"],
        ORDERING_BASELINES[order_policy],
        seed=0,
    )


def test_proportional_parity() -> None:
    # Tracer bullet: per-order routing is independent of agent grouping, so the
    # product-stream contract must reproduce the legacy metrics exactly.
    legacy = _episode("proportional", product_stream=False)
    stream = _episode("proportional", product_stream=True)
    assert stream == legacy


def test_even_split_parity() -> None:
    legacy = _episode("even_split", product_stream=False)
    stream = _episode("even_split", product_stream=True)
    assert stream == legacy


def test_stream_agents_are_restricted_and_deterministic() -> None:
    # Restricted coverage (ADR-0052): consumer pool products + producer
    # replenishment/restoration/upgrade/inventory requirement products only.
    env = TokenomicsAECEnv(horizon=100, product_stream_ordering=True)
    order_agents = [a for a in env.possible_agents if a.startswith("order_")]
    assert order_agents == [
        "order_c1__30",
        "order_p1__11",
        "order_p2__10",
        "order_p3__10",
        "order_p3__11",
        "order_p4__10",
        "order_p4__11",
        "order_p4__20",
    ]
    # stable across construction
    again = TokenomicsAECEnv(horizon=100, product_stream_ordering=True)
    assert again.possible_agents == env.possible_agents


def test_stream_agent_holds_only_its_product() -> None:
    # The whole point of ADR-0052: a stream agent never multiplexes products.
    env = TokenomicsAECEnv(horizon=100, product_stream_ordering=True)
    seen_nonempty = False

    def checking_orderer(e: TokenomicsAECEnv, agent: str):
        _, did = e._role[agent]
        product = int(did.split("__")[1])
        incoming = e.system.ordering_agents[did].incoming
        nonlocal seen_nonempty
        seen_nonempty = seen_nonempty or bool(incoming)
        assert all(oi.product == product for oi in incoming.values())
        return ORDERING_BASELINES["proportional"](e, agent)

    run_episode(env, PRODUCING_BASELINES["stock_aware"], checking_orderer, seed=0)
    assert seen_nonempty  # the invariant was actually exercised


def test_pettingzoo_api_stream_mode() -> None:
    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    api_test(env, num_cycles=20, verbose_progress=False)


# --- product_stream_v1 observation contract (orderer_obs_version 3) ---


def test_stream_mode_uses_obs_version_three() -> None:
    # Tracer: product-stream mode selects the simplified per-stream obs contract,
    # whose width scales with producers (P rows) + backlog (K), never with the
    # product vocabulary (no Q one-hot).
    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    assert env.spec_.orderer_obs_version == 3
    expected = _stream_obs_dim(env.spec_.num_producers, env.spec_.k)
    assert env.spec_.ordering_obs_dim == expected

    env.reset(seed=0)
    order_agent = next(a for a in env.possible_agents if a.startswith("order_"))
    assert env.observe(order_agent).shape == (expected,)


def _relabel(config, f):
    """Apply a product-id bijection f to every product reference in a config."""
    sheets = [
        replace(ps, product=f(ps.product), inputs={f(g): n for g, n in ps.inputs.items()})
        for ps in config.process_sheets
    ]
    producers = []
    for p in config.producer_configs:
        ie = p.inventory_expansion
        producers.append(replace(
            p,
            produces=frozenset(f(x) for x in p.produces),
            restoration=replace(p.restoration, require=f(p.restoration.require)),
            upgrade=replace(p.upgrade, require=f(p.upgrade.require)),
            inventory_expansion=replace(ie, require=f(ie.require)) if ie is not None else None,
        ))
    consumers = [
        replace(c, product=f(c.product), products=tuple(f(x) for x in c.products))
        for c in config.consumers
    ]
    return replace(config, process_sheets=sheets, producer_configs=producers, consumers=consumers)


def _all_observations(config, seed: int, *, corrected: bool = False) -> list[np.ndarray]:
    """Drive a deterministic baseline episode, recording each acting agent's
    observation in order."""
    env = TokenomicsAECEnv(config=config, horizon=40, product_stream_ordering=True,
                           corrected_stream_k=corrected)
    env.reset(seed=seed)
    for a in env.possible_agents:
        env.action_space(a).seed(seed)
    obs_log: list[np.ndarray] = []
    max_iter = env.horizon * len(env.possible_agents) * 3 + 100
    for agent in env.agent_iter(max_iter=max_iter):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        obs_log.append(env.observe(agent).copy())
        role = env._role[agent][0]
        policy = PRODUCING_BASELINES["stock_aware"] if role == "prod" else ORDERING_BASELINES["proportional"]
        env.step(policy(env, agent))
    return obs_log


def test_stream_obs_dim_independent_of_product_count() -> None:
    # Scales with producers, not products: adding an orderable product (no new
    # producer, no new replenish stream) leaves the v3 width unchanged, while the
    # product-multiplexing v2 width grows.
    base = load_config(DEFAULT_CONFIG)
    extra_sheet = replace(base.process_sheets[0], product=999, require={"capacity-4": 2}, inputs={})
    p4 = next(p for p in base.producer_configs if p.id == "p4")
    grown_p4 = replace(p4, produces=p4.produces | {999})
    grown = replace(
        base,
        process_sheets=[*base.process_sheets, extra_sheet],
        producer_configs=[grown_p4 if p.id == "p4" else p for p in base.producer_configs],
    )
    v3_base = build_spec(base, orderer_obs_version=3)
    v3_grown = build_spec(grown, orderer_obs_version=3)
    assert v3_grown.num_producers == v3_base.num_producers
    assert v3_grown.ordering_obs_dim == v3_base.ordering_obs_dim  # no Q dependence
    # sanity: the product-multiplexing v2 contract DOES grow with product count.
    assert (
        build_spec(grown, orderer_obs_version=2).ordering_obs_dim
        > build_spec(base, orderer_obs_version=2).ordering_obs_dim
    )


def test_fingerprint_separates_stream_from_legacy_contracts() -> None:
    # A v3 (stream) checkpoint must not be mis-scored against a v1/v2 economy.
    cfg = load_config(DEFAULT_CONFIG)
    fps = {config_fingerprint(cfg, v) for v in (1, 2, 3)}
    assert len(fps) == 3


def test_stream_obs_invariant_to_product_relabeling() -> None:
    # Decisions depend on economy structure, not product identity: a monotonic
    # relabel of every product id yields bit-identical observations throughout the
    # episode (and hence identical routing), since the v3 obs carries no product id.
    base = load_config(DEFAULT_CONFIG)
    relabeled = _relabel(base, lambda x: x + 100)
    base_obs = _all_observations(base, seed=0)
    relabeled_obs = _all_observations(relabeled, seed=0)
    assert len(base_obs) == len(relabeled_obs)
    for a, b in zip(base_obs, relabeled_obs):
        assert np.array_equal(a, b)


def test_stream_producer_rows_are_keyed_by_producer() -> None:
    # Producer-row equivariance: permuting the producer order relocates each
    # producer's row but never changes its contents, so a permutation of producers
    # maps cleanly onto a permutation of the action columns.
    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    env.reset(seed=0)
    from tokenomics.env.encode import encode_ordering_stream

    spec = env.spec_
    view = env.system.product_stream_view("c1__30")
    obs = encode_ordering_stream(spec, view, 0.0)

    permuted = replace(spec, producer_ids=list(reversed(spec.producer_ids)))
    obs_perm = encode_ordering_stream(permuted, view, 0.0)

    head = STREAM_PRODUCT_FEATURES + spec.k * STREAM_SLOT_SCALARS
    F = STREAM_PRODUCER_FEATURES

    def row(vec, ordering, pid):
        j = ordering.index(pid)
        return vec[head + j * F : head + (j + 1) * F]

    for pid in spec.producer_ids:
        assert np.array_equal(row(obs, spec.producer_ids, pid), row(obs_perm, permuted.producer_ids, pid))


def test_decode_routes_by_producer_order() -> None:
    # Action-column equivariance: a logit peaked at a producer routes the order to
    # that producer regardless of where it sits in the producer ordering.
    from tokenomics.domain.order import OrderInfo
    from tokenomics.domain.ordering_agent import OrderingAgentView
    from tokenomics.env.decode import decode_ordering

    spec = build_spec(load_config(DEFAULT_CONFIG), orderer_obs_version=3)
    infos = {pid: None for pid in spec.producer_ids}
    oi = OrderInfo("o1", tokens=10, product=30, quantity=4)
    # pretend two producers are eligible for this order
    p_a, p_b = spec.producer_ids[0], spec.producer_ids[2]
    view = OrderingAgentView(incoming={"o1": oi}, eligible={"o1": {p_a: infos[p_a], p_b: infos[p_b]}})

    for target in (p_a, p_b):
        action = np.zeros((spec.k, spec.num_producers), dtype=np.float32)
        action[0, spec.producer_ids.index(target)] = 10.0  # peak on the target column
        cmd = decode_ordering(spec, action.flatten(), view)
        routed = {s.producer_id: s.quantity for s in cmd.orders["o1"] if s.quantity > 0}
        assert max(routed, key=routed.get) == target


def test_stream_view_is_single_product_with_full_producer_rows() -> None:
    # Domain owns the stream-local facts (ADR-0052): the view fixes one product,
    # carries a row for every producer, and its backlog holds only that product.
    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    env.reset(seed=0)
    view = env.system.product_stream_view("c1__30")
    assert view.product == 30
    assert set(view.producers) == set(env.spec_.producer_ids)
    assert set(view.producer_products) == set(env.spec_.producer_ids)
    assert all(slot.product == 30 for slot in view.backlog)


def test_stream_encoder_length_matches_spec() -> None:
    # Lockstep guard: encoder width == spec.ordering_obs_dim for both an active and
    # an idle stream (an idle producer stream still emits product + producer rows).
    from tokenomics.env.encode import encode_ordering_stream

    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    env.reset(seed=0)
    dim = env.spec_.ordering_obs_dim
    for aid in ("c1__30", "p1__11"):  # active (consumer order placed) and idle
        vec = encode_ordering_stream(env.spec_, env.system.product_stream_view(aid), 0.0)
        assert vec.shape == (dim,)


# --- eval bars must be scored in the candidate's agent layout (review P2) ---


def test_random_hindsight_baseline_is_layout_sensitive() -> None:
    # The premise behind the eval-bar fix: `random` ordering draws action-space RNG
    # over the agent set, and the (source, product) stream split changes that set —
    # so the random-inclusive hindsight baseline is a *different bar* under each layout, even
    # though the deterministic baselines are grouping-invariant (parity). A stream-mode
    # candidate must therefore be scored against a stream-mode hindsight baseline.
    from tokenomics.eval.baselines import PRODUCING_BASELINES, ORDERING_BASELINES
    from tokenomics.eval.runner import evaluate
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(8, depth=3)  # s8 hindsight-best is stock_aware+random
    sa, rnd = PRODUCING_BASELINES["stock_aware"], ORDERING_BASELINES["random"]
    legacy = evaluate(sa, rnd, seeds=range(3), horizon=100, config=cfg, product_stream=False)
    stream = evaluate(sa, rnd, seeds=range(3), horizon=100, config=cfg, product_stream=True)
    assert legacy["consumer_completed_units"][0] != stream["consumer_completed_units"][0]


def test_eval_m2_scores_hindsight_baseline_in_candidate_layout() -> None:
    # Regression for review P2: eval_m2 must report an hb_cu computed in the SAME
    # layout as the (stream-mode) M2 candidate — i.e. the stream-mode hindsight baseline, not a
    # legacy-mode one. The M2 policy quality is irrelevant here (random init); hb_cu is
    # baseline-only, so we only assert the bar's layout.
    pytest.importorskip("torch")
    from tokenomics.eval.baselines import PRODUCING_BASELINES, ORDERING_BASELINES
    from tokenomics.eval.orderer_oracle import HINDSIGHT_BASELINE_COMBOS
    from tokenomics.eval.runner import evaluate
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.orderer_improve import eval_m2
    from tokenomics.train.mlp_actor import GaussianActor, save_policies

    import tempfile, os

    depth, seed, eval_seeds = 3, 8, 3
    cfg = build_scale_config(seed, depth=depth)
    probe = TokenomicsAECEnv(config=cfg, horizon=1, product_stream_ordering=True)
    prod = GaussianActor(probe.observation_space(probe._prod_agents[0]).shape[0],
                         probe.spec_.producing_action_dim)
    order = GaussianActor(probe.observation_space(probe._order_agents[0]).shape[0],
                          probe.spec_.ordering_action_dim)
    with tempfile.TemporaryDirectory() as d:
        save = os.path.join(d, "m2.pt")
        save_policies(prod, order, 5.0, save, config=cfg, orderer_obs_version=3)
        rows = eval_m2(save, depth=depth, seeds=[seed], eval_seeds=eval_seeds,
                       horizon=100, product_stream=True, log=lambda *a, **k: None)

    row = rows[0]
    pp, oo = HINDSIGHT_BASELINE_COMBOS[row["hb_name"]]
    direct = evaluate(PRODUCING_BASELINES[pp], ORDERING_BASELINES[oo], seeds=range(eval_seeds),
                      horizon=100, config=cfg, product_stream=True)
    assert row["hb_cu"] == round(direct["consumer_completed_units"][0], 1)


# --- corrected product-stream contract (orderer_obs_version 4, O0) ---


def test_k_stream_corrected_at_v4() -> None:
    # O0 tracer: a product-stream agent serves ONE (source, product) stream, so its
    # backlog is bounded by per-stream concurrency — ORDER_LIFETIME replenishment orders
    # plus the INVESTMENT_STREAMS gated investment orders that can share the product —
    # never by the producer input-product width (max_rel) that inflates the global legacy
    # k. The corrected contract (v4) exposes that constant bound as k_stream; legacy v3
    # keeps k.
    from tokenomics.env.spaces import INVESTMENT_STREAMS, ORDER_LIFETIME

    cfg = load_config(DEFAULT_CONFIG)
    v3 = build_spec(cfg, orderer_obs_version=3)
    v4 = build_spec(cfg, orderer_obs_version=4)
    assert v4.k_stream == ORDER_LIFETIME + INVESTMENT_STREAMS
    assert v3.k_stream == v3.k


def _grow_max_rel(base):
    """Add a composite product (new inputs) to the producer that currently
    maximizes replenishable-input count, so the global legacy k (= ... +
    ORDER_LIFETIME*max_rel) must grow while producer count stays fixed."""
    spec = build_spec(base, orderer_obs_version=3)
    top = max(spec.producer_inputs, key=lambda p: len(spec.producer_inputs[p]))
    new_inputs = [90001, 90002, 90003]
    cap = next(iter(build_spec(base).producer_owns[top]))
    primal = [replace(base.process_sheets[0], product=g, require={cap: 1}, inputs={})
              for g in new_inputs]
    composite = replace(base.process_sheets[0], product=90000, require={cap: 2},
                        inputs={g: 1 for g in new_inputs})
    grown_top = next(p for p in base.producer_configs if p.id == top)
    grown_top = replace(grown_top, produces=grown_top.produces | {90000})
    return replace(
        base,
        process_sheets=[*base.process_sheets, composite, *primal],
        producer_configs=[grown_top if p.id == top else p for p in base.producer_configs],
    )


def test_v4_obs_action_dims_invariant_to_product_width() -> None:
    # O0 core win / acceptance gate: with producer count fixed, widening the product
    # set (here growing a producer's replenishable inputs) must NOT change the
    # corrected (v4) orderer obs/action dims. The legacy v3 contract still leaks this
    # width through the global k — asserted to prove the test actually moves max_rel.
    base = load_config(DEFAULT_CONFIG)
    grown = _grow_max_rel(base)

    v3_base, v3_grown = build_spec(base, orderer_obs_version=3), build_spec(grown, orderer_obs_version=3)
    v4_base, v4_grown = build_spec(base, orderer_obs_version=4), build_spec(grown, orderer_obs_version=4)

    assert v3_grown.num_producers == v3_base.num_producers
    assert v4_grown.num_producers == v4_base.num_producers
    # v3 leaks: obs AND action grow with product/input width.
    assert v3_grown.ordering_obs_dim > v3_base.ordering_obs_dim
    assert v3_grown.ordering_action_dim > v3_base.ordering_action_dim
    # v4 corrected: both invariant.
    assert v4_grown.ordering_obs_dim == v4_base.ordering_obs_dim
    assert v4_grown.ordering_action_dim == v4_base.ordering_action_dim


def test_v4_stream_encoder_length_matches_spec() -> None:
    # O0 lockstep: the corrected (v4) encoder emits exactly v4.ordering_obs_dim for an
    # active and an idle stream. The domain stream view is version-agnostic, so the
    # only difference from v3 is the narrowed k_stream backlog width.
    from tokenomics.env.encode import encode_ordering_stream

    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    env.reset(seed=0)
    v4 = build_spec(env.config, orderer_obs_version=4)
    dim = v4.ordering_obs_dim
    assert dim < env.spec_.ordering_obs_dim  # v4 is strictly narrower than legacy v3 here
    for aid in ("c1__30", "p1__11"):  # active and idle
        vec = encode_ordering_stream(v4, env.system.product_stream_view(aid), 0.0)
        assert vec.shape == (dim,)


def test_v4_decode_reshapes_and_routes_at_k_stream() -> None:
    # O0: the corrected (v4) action vector is k_stream*P, not k*P. decode must reshape
    # at the narrowed bound and still route a peaked logit to the right producer.
    from tokenomics.domain.order import OrderInfo
    from tokenomics.domain.ordering_agent import OrderingAgentView
    from tokenomics.env.decode import decode_ordering

    spec = build_spec(load_config(DEFAULT_CONFIG), orderer_obs_version=4)
    assert spec.ordering_action_dim == spec.k_stream * spec.num_producers
    oi = OrderInfo("o1", tokens=10, product=30, quantity=4)
    p_a, p_b = spec.producer_ids[0], spec.producer_ids[2]
    view = OrderingAgentView(incoming={"o1": oi}, eligible={"o1": {p_a: None, p_b: None}})

    for target in (p_a, p_b):
        action = np.zeros((spec.k_stream, spec.num_producers), dtype=np.float32)
        action[0, spec.producer_ids.index(target)] = 10.0
        cmd = decode_ordering(spec, action.flatten(), view)
        routed = {s.producer_id: s.quantity for s in cmd.orders["o1"] if s.quantity > 0}
        assert max(routed, key=routed.get) == target


def test_v4_bound_covers_investment_orders_sharing_a_stream() -> None:
    # O0 correctness (review fix): a product-stream agent is keyed by (source, product),
    # NOT by order type. A producer's replenishment orders for input X (up to
    # ORDER_LIFETIME in the TTL window) and its gated investment orders whose
    # means-of-production product is also X collapse into the SAME (producer, X) stream.
    # There are INVESTMENT_STREAMS independently-gated investment types, so a single
    # stream can hold ORDER_LIFETIME + INVESTMENT_STREAMS concurrent orders. The backlog
    # bound must give every one of them a controllable decode slot — otherwise decode
    # falls back to uniform routing for the overflow (hidden, uncontrolled orders).
    from tokenomics.domain.order import OrderInfo
    from tokenomics.domain.ordering_agent import OrderingAgentView
    from tokenomics.env.decode import decode_ordering
    from tokenomics.env.spaces import INVESTMENT_STREAMS, ORDER_LIFETIME

    spec = build_spec(load_config(DEFAULT_CONFIG), orderer_obs_version=4)
    n = ORDER_LIFETIME + INVESTMENT_STREAMS
    assert spec.k_stream >= n  # the bound must cover the worst-case per-stream concurrency

    p_a, p_b = spec.producer_ids[0], spec.producer_ids[2]
    oids = [f"o{i}" for i in range(n)]
    incoming = {o: OrderInfo(o, tokens=10, product=30, quantity=4) for o in oids}
    eligible = {o: {p_a: None, p_b: None} for o in oids}
    view = OrderingAgentView(incoming=incoming, eligible=eligible)

    # Peak each order's own slot toward an alternating target; every order must route
    # there — including the ones past ORDER_LIFETIME, which only have a slot if the bound
    # accounts for investment orders.
    targets = [p_a if i % 2 == 0 else p_b for i in range(n)]
    action = np.zeros((spec.k_stream, spec.num_producers), dtype=np.float32)
    for k, target in enumerate(targets):
        action[k, spec.producer_ids.index(target)] = 10.0
    cmd = decode_ordering(spec, action.flatten(), view)
    for o, target in zip(sorted(oids), targets):
        routed = {s.producer_id: s.quantity for s in cmd.orders[o] if s.quantity > 0}
        assert max(routed, key=routed.get) == target, f"{o} not controlled by its slot"


def test_corrected_stream_k_env_selects_v4_same_layout() -> None:
    # O0 env wiring: opting into the corrected contract selects version 4 and the
    # narrowed obs/action shapes, but leaves the (source, product) agent set identical
    # to v3 — only the tensor widths shrink, not the agent layout.
    v3 = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    v4 = TokenomicsAECEnv(horizon=40, product_stream_ordering=True, corrected_stream_k=True)
    assert v4.spec_.orderer_obs_version == 4
    assert v4.possible_agents == v3.possible_agents
    assert v4.spec_.ordering_obs_dim < v3.spec_.ordering_obs_dim
    assert v4.spec_.ordering_action_dim < v3.spec_.ordering_action_dim
    # episode runs end to end under the corrected contract.
    res = run_episode(v4, PRODUCING_BASELINES["stock_aware"], ORDERING_BASELINES["proportional"], seed=0)
    assert res["cycles"] > 0


def test_v4_obs_invariant_to_product_relabeling() -> None:
    # O0 acceptance gate: product relabeling leaves the corrected (v4) orderer
    # observations bit-identical — the contract carries no product id.
    base = load_config(DEFAULT_CONFIG)
    relabeled = _relabel(base, lambda x: x + 100)
    base_obs = _all_observations(base, seed=0, corrected=True)
    relabeled_obs = _all_observations(relabeled, seed=0, corrected=True)
    assert len(base_obs) == len(relabeled_obs) and len(base_obs) > 0
    for a, b in zip(base_obs, relabeled_obs):
        assert np.array_equal(a, b)


def test_v4_producer_rows_are_keyed_by_producer() -> None:
    # O0 acceptance gate: producer-row permutation only relocates each producer's row
    # (and its action column), never its contents — under the narrowed k_stream head.
    from tokenomics.env.encode import encode_ordering_stream

    spec = build_spec(load_config(DEFAULT_CONFIG), orderer_obs_version=4)
    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True, corrected_stream_k=True)
    env.reset(seed=0)
    view = env.system.product_stream_view("c1__30")
    obs = encode_ordering_stream(spec, view, 0.0)
    permuted = replace(spec, producer_ids=list(reversed(spec.producer_ids)))
    obs_perm = encode_ordering_stream(permuted, view, 0.0)

    head = STREAM_PRODUCT_FEATURES + spec.k_stream * STREAM_SLOT_SCALARS
    F = STREAM_PRODUCER_FEATURES

    def row(vec, ordering, pid):
        j = ordering.index(pid)
        return vec[head + j * F : head + (j + 1) * F]

    for pid in spec.producer_ids:
        assert np.array_equal(row(obs, spec.producer_ids, pid), row(obs_perm, permuted.producer_ids, pid))


def test_v3_contract_unchanged_and_distinct_from_v4() -> None:
    # O0 back-compat gate: the legacy v3 contract keeps the global-k widths (loadable
    # for old checkpoints), and its fingerprint is distinct from v4 so a v3 checkpoint
    # can't be mis-loaded into the corrected tensor shape.
    from tokenomics.env.spaces import INVESTMENT_STREAMS, ORDER_LIFETIME

    cfg = load_config(DEFAULT_CONFIG)
    v3 = build_spec(cfg, orderer_obs_version=3)
    v4 = build_spec(cfg, orderer_obs_version=4)
    assert v3.k_stream == v3.k  # v3 unchanged: still the global legacy bound
    assert v4.k_stream == ORDER_LIFETIME + INVESTMENT_STREAMS
    # distinct contract: dims differ (here v3.k > the v4 constant bound) and fingerprints separate.
    assert v3.ordering_obs_dim != v4.ordering_obs_dim
    assert config_fingerprint(cfg, 3) != config_fingerprint(cfg, 4)
