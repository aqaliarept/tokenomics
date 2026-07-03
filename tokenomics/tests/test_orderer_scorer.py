"""PRODUCT_AGNOSTIC O2: learned orderer candidate-scorer contract (local).

Contract-only TDD (no training): the scorer maps a ProductStreamView to one score
per producer row, with shared weights so it is order-equivariant over producer rows
and runs on any producer count P. The deterministic masked allocator (O1) turns those
scores into conserved SubOrders. Real distillation/training is cloud work.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

from tokenomics.env.aec_env import TokenomicsAECEnv
from tokenomics.eval.baselines import PRODUCING_BASELINES, ORDERING_BASELINES
from tokenomics.eval.runner import run_episode


def _active_stream_view():
    """Drive a few cycles, return a product-stream view with a non-empty backlog and
    the env (for producer ids)."""
    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True, corrected_stream_k=True)
    env.reset(seed=0)
    # step a couple cycles so orders are in flight
    for _ in range(3):
        for agent in list(env.agents):
            if env.terminations[agent] or env.truncations[agent]:
                continue
            role = env._role[agent][0]
            pol = PRODUCING_BASELINES["stock_aware"] if role == "prod" else ORDERING_BASELINES["proportional"]
            if env.agent_selection == agent:
                env.step(pol(env, agent))
    return env, env.system.product_stream_view("c1__30")


def test_scorer_emits_one_score_per_producer() -> None:
    # Tracer: forward_view returns a 1-D score tensor of length P (one per producer
    # row), no fixed K*P head.
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    env, view = _active_stream_view()
    model = OrdererScoreModel()
    scores = model.forward_view(view)
    assert scores.shape == (len(view.producers),)


def test_score_follows_producer_stats_not_position() -> None:
    # Row-order equivariance: swapping two producers' stats (keeping ids/sort order
    # fixed) swaps their scores — the shared row encoder scores content, not slot.
    import torch
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    env, view = _active_stream_view()
    model = OrdererScoreModel().eval()
    pids = sorted(view.producers)
    a, b = pids[0], pids[-1]
    with torch.no_grad():
        base = model.forward_view(view)
    swapped = replace(
        view,
        producers={**view.producers, a: view.producers[b], b: view.producers[a]},
        producer_products={**view.producer_products, a: view.producer_products[b],
                           b: view.producer_products[a]},
    )
    with torch.no_grad():
        new = model.forward_view(swapped)
    ia, ib = pids.index(a), pids.index(b)
    assert torch.allclose(new[ia], base[ib]) and torch.allclose(new[ib], base[ia])


def test_runs_on_variable_producer_count() -> None:
    # No fixed head: the same model scores a view with a dropped producer; output
    # length tracks P at runtime.
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    env, view = _active_stream_view()
    model = OrdererScoreModel()
    drop = sorted(view.producers)[0]
    fewer = replace(
        view,
        producers={k: v for k, v in view.producers.items() if k != drop},
        producer_products={k: v for k, v in view.producer_products.items() if k != drop},
    )
    assert model.forward_view(fewer).shape == (len(view.producers) - 1,)


def test_scores_bridge_to_masked_allocator() -> None:
    # O1+O2 integration: model scores + an eligibility mask from the view decode into
    # conserved SubOrders routed only to eligible producers.
    import torch
    from tokenomics.env.allocator import allocate_suborders
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    env, view = _active_stream_view()
    model = OrdererScoreModel().eval()
    pids = model.producer_ids(view)
    with torch.no_grad():
        scores = model.forward_view(view).numpy()
    mask = np.array([view.producer_products[p].eligible > 0 for p in pids])
    if not mask.any():
        pytest.skip("no eligible producer in this stream snapshot")
    subs = allocate_suborders(pids, scores, mask, quantity=9, tokens=90)
    assert sum(s.quantity for s in subs) == 9 and sum(s.tokens for s in subs) == 90
    assert all(mask[pids.index(s.producer_id)] for s in subs)


def test_schema_and_determinism() -> None:
    # Schema gate: row feature width ties to the domain dataclasses (incl. the
    # active_investments 3-tuple); a checkpoint version exists; eval is deterministic.
    import dataclasses as dc
    import torch
    from tokenomics.domain.orderer_view import (
        ProducerProductStats, ProducerStats, ProductStats)
    from tokenomics.train.orderer_scorer import (
        ORDERER_SCORER_CHECKPOINT_VERSION, _PRODUCER_ROW_DIM, _PRODUCT_DIM,
        OrdererScoreModel)

    assert _PRODUCT_DIM == len(dc.fields(ProductStats))
    # ProducerStats has one 3-wide tuple field (active_investments).
    assert _PRODUCER_ROW_DIM == len(dc.fields(ProducerStats)) - 1 + 3 + len(dc.fields(ProducerProductStats))
    assert ORDERER_SCORER_CHECKPOINT_VERSION >= 1

    env, view = _active_stream_view()
    model = OrdererScoreModel().eval()
    with torch.no_grad():
        assert torch.allclose(model.forward_view(view), model.forward_view(view))


def test_scorer_policy_emits_v4_action_and_drives_episode() -> None:
    # Deploy path: the scorer policy emits a k_stream*P action (per-producer scores
    # tiled across slots) that the v4 env decodes via the masked allocator; an episode
    # runs end to end with the learned orderer.
    from tokenomics.train.orderer_scorer import OrdererScoreModel, scorer_ordering_policy

    env = TokenomicsAECEnv(horizon=30, product_stream_ordering=True, corrected_stream_k=True)
    pol = scorer_ordering_policy(OrdererScoreModel())
    res = run_episode(env, PRODUCING_BASELINES["stock_aware"], pol, seed=0)
    assert res["cycles"] > 0
    # one orderer action has the corrected width.
    env.reset(seed=0)
    oa = next(a for a in env.agents if a.startswith("order_"))
    while env.agent_selection != oa:
        a = env.agent_selection
        role = env._role[a][0]
        p = PRODUCING_BASELINES["stock_aware"] if role == "prod" else ORDERING_BASELINES["proportional"]
        env.step(p(env, a))
    assert pol(env, oa).shape == (env.spec_.k_stream * env.spec_.num_producers,)


def test_teacher_logits_rank_and_mask() -> None:
    # Teacher target tracks makeable throughput and masks ineligible producers.
    from tokenomics.train.orderer_scorer import teacher_logits

    env, view = _active_stream_view()
    pids, target, mask = teacher_logits(view)
    assert mask.tolist() == [view.producer_products[p].eligible > 0 for p in pids]
    # a producer that can make more units gets a >= target than one that can make fewer.
    mk = {p: view.producer_products[p].makeable_qty_now for p in pids}
    hi = max(pids, key=lambda p: mk[p]); lo = min(pids, key=lambda p: mk[p])
    if mk[hi] > mk[lo]:
        assert target[pids.index(hi)] >= target[pids.index(lo)]


def test_smoke_distill_roundtrip() -> None:
    # End-to-end smoke (local-allowed): collect a few samples, fit a few iters, the
    # checkpoint round-trips, and eval returns a ratio row per held-out topology.
    import tempfile, os
    from tokenomics.train.orderer_scorer import (
        collect_scorer_samples, fit_scorer, save_scorer, load_scorer,
        eval_scorer, OrdererScoreModel)

    samples = collect_scorer_samples([0], depth=3, horizon=12)
    assert samples
    model = fit_scorer(OrdererScoreModel(hidden=16), samples, iters=5)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "scorer.pt")
        save_scorer(model, path)
        reloaded = load_scorer(path)
    rows = eval_scorer(reloaded, depth=3, seeds=[6], eval_seeds=1, horizon=12)
    assert rows and "ratio" in rows[0] and rows[0]["producers"] >= 1


def test_collect_scorer_samples_honors_explicit_configs() -> None:
    # A replay/curriculum caller passes explicit Configurations (e.g. archived tasks) so the
    # orderer trains on exactly those topologies, not a build_scale_config(seed, depth) stand-in.
    from tokenomics.train.orderer_scorer import collect_scorer_samples
    from tokenomics.eval.scale import build_scale_config

    cfgs = [build_scale_config(0, depth=2), build_scale_config(1, depth=2)]
    samples = collect_scorer_samples([0, 1], depth=5, horizon=12, configs=cfgs)  # depth ignored
    assert samples


def test_collect_scorer_samples_rejects_misaligned_configs() -> None:
    from tokenomics.train.orderer_scorer import collect_scorer_samples
    from tokenomics.eval.scale import build_scale_config

    with pytest.raises(ValueError):
        collect_scorer_samples([0, 1], depth=2, horizon=12, configs=[build_scale_config(0, depth=2)])


# --- v4 orderer oracle relabel (EXP-0032): teach the orderer from the mining bar's own
# per-cycle search routing instead of the proportional throughput teacher. The collector
# reuses `_search_best_pol`; targets are the committed routing, mean-reduced over active
# backlog slots then centered + clipped (softmax shift-invariance). ---


def test_center_clip_eligible_centers_and_clips() -> None:
    # A sampled routing's logits have arbitrary scale; softmax is shift-invariant, so the
    # relabel target centers the eligible logits to zero mean and clips to the action bound.
    # Ineligible rows are floored (masked out of the distill loss anyway).
    from tokenomics.train.orderer_scorer import _BOUND, _center_clip_eligible

    logits = np.array([1.0, 3.0, 5.0, 999.0], dtype=np.float32)
    mask = np.array([True, True, True, False])
    out = _center_clip_eligible(logits, mask)
    assert np.allclose(out[mask], [-2.0, 0.0, 2.0])   # eligible mean 3 -> centered
    assert abs(out[mask].mean()) < 1e-6
    assert out[3] == -_BOUND                          # ineligible floored


def test_reduce_committed_means_active_slots() -> None:
    # The scorer represents one per-producer vector per stream, but a committed sampled
    # routing has arbitrary per-slot rows. Reduce (k_stream, P) -> (P,) by mean over the
    # occupied backlog slots; empty slots beyond the backlog are excluded.
    from tokenomics.train.orderer_scorer import _reduce_committed

    action = np.array([0.0, 10.0, 2.0, 20.0, 99.0, 99.0], dtype=np.float32)  # 3 slots x 2 producers
    reduced = _reduce_committed(action, k_stream=3, num_producers=2, n_occupied=2)
    assert reduced.tolist() == [1.0, 15.0]            # mean of the first 2 rows only


def test_collect_oracle_orderer_samples_parity() -> None:
    # The oracle-relabel collector yields ScorerSamples that feed the existing masked-distill
    # path unchanged: target/mask are aligned to sorted(view.producers), mask is the view's
    # eligibility, and only backlog+eligible views are captured.
    from tokenomics.train.orderer_scorer import (
        collect_oracle_orderer_samples, fit_scorer, OrdererScoreModel)
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(0, depth=2)
    samples = collect_oracle_orderer_samples(
        [cfg], [0], prod=PRODUCING_BASELINES["stock_aware"],
        n_samples=2, sample_scale=1.0, lookahead=3, horizon=12)
    assert samples
    for s in samples:
        pids = sorted(s.view.producers)
        assert s.target.shape == (len(pids),)
        assert s.mask.shape == (len(pids),)
        assert s.mask.dtype == bool and s.mask.any()
        assert s.mask.tolist() == [s.view.producer_products[p].eligible > 0 for p in pids]
    fit_scorer(OrdererScoreModel(hidden=16), samples, iters=2)  # consumed unchanged


def test_collect_oracle_orderer_samples_reproducible() -> None:
    # Deterministic teacher: same (configs, seeds, params) -> identical samples (the search
    # sample RNG is seeded by (seed, cycle)), so a re-run reproduces the distill set.
    from tokenomics.train.orderer_scorer import collect_oracle_orderer_samples
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(0, depth=2)
    kw = dict(prod=PRODUCING_BASELINES["stock_aware"], n_samples=2, lookahead=3, horizon=12)
    a = collect_oracle_orderer_samples([cfg], [0], **kw)
    b = collect_oracle_orderer_samples([cfg], [0], **kw)
    assert len(a) == len(b) and a
    for sa, sb in zip(a, b):
        assert np.array_equal(sa.target, sb.target)
        assert np.array_equal(sa.mask, sb.mask)


def test_collect_oracle_orderer_samples_rejects_misaligned() -> None:
    from tokenomics.train.orderer_scorer import collect_oracle_orderer_samples
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config

    with pytest.raises(ValueError):
        collect_oracle_orderer_samples(
            [build_scale_config(0, depth=2)], [0, 1],
            prod=PRODUCING_BASELINES["stock_aware"], horizon=12)


def test_representability_row_reports_recovery_ratios() -> None:
    # Smoke (local-allowed): the oracle-relabel representability row distills an orderer on a
    # task's search targets and reports on-task champ/search/oracle/bar units + recovery ratios
    # in the right ordering (search teacher is the champ-producer ceiling the oracle chases).
    from tokenomics.eval.curriculum import TaskRecord
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.orderer_representability import representability_row
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    rec = TaskRecord(
        task_id="t", task_family="scale", generator_version="test", serializer_version="v1",
        world_seed=0, episode_seed=0,
        world_spec={"generator": "build_scale_config", "params": {"depth": 2}},
    )
    row = representability_row(rec, PRODUCING_BASELINES["stock_aware"], OrdererScoreModel(),
                              iters=3, horizon=12, eval_seeds=1)
    for k in ("champ_cu", "search_cu", "oracle_cu", "bar_cu", "recovered",
              "recovered_vs_teacher", "recovered_vs_bar"):
        assert k in row
    assert row["n_samples"] > 0
    # the search teacher (champ producer + per-cycle search) is the on-task ceiling it distills.
    assert row["search_cu"] >= row["champ_cu"] - 1e-6


def test_emit_representability_writes_exp_layout(tmp_path) -> None:
    # The verification run writes a pullable EXP layout (summary.csv + spec.json + RESULT.md);
    # the RESULT gate line is the mean recovered_vs_teacher. No compute — fake rows.
    from tokenomics.eval.orderer_representability import _emit_representability

    rows = [{"task_id": "t1", "champ_cu": 10.0, "search_cu": 20.0, "oracle_cu": 16.0,
             "bar_cu": 22.0, "recovered": 6.0, "recovered_vs_teacher": 0.6,
             "recovered_vs_bar": 0.5, "n_samples": 5}]
    _emit_representability(tmp_path, rows, meta={"iters": 300})
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "spec.json").exists()
    result = (tmp_path / "RESULT.md").read_text()
    assert "recovered_vs_teacher" in result and "green-light" in result


def test_headroom_row_reports_measured_headroom() -> None:
    # Probe re-selection: the cheap headroom row measures teacher_headroom = search - champ under
    # the frozen producer (no distill), the ceiling >= the deployed stack, with a normalized frac.
    from tokenomics.eval.curriculum import TaskRecord
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.orderer_representability import headroom_row
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    rec = TaskRecord(
        task_id="t", task_family="scale", generator_version="test", serializer_version="v1",
        world_seed=0, episode_seed=0,
        world_spec={"generator": "build_scale_config", "params": {"depth": 2}},
    )
    row = headroom_row(rec, PRODUCING_BASELINES["stock_aware"], OrdererScoreModel(),
                       horizon=12, eval_seeds=1)
    for k in ("champ_cu", "search_cu", "teacher_headroom", "headroom_frac"):
        assert k in row
    assert row["search_cu"] >= row["champ_cu"] - 1e-6
    assert abs(row["teacher_headroom"] - (row["search_cu"] - row["champ_cu"])) < 0.02


def test_qualifies_uses_headroom_floor() -> None:
    # A task is a valid representability probe only if its measured headroom clears the floor.
    from tokenomics.eval.orderer_representability import _qualifies

    assert _qualifies({"headroom_frac": 0.05}, 0.02) is True
    assert _qualifies({"headroom_frac": 0.01}, 0.02) is False
    assert _qualifies({"headroom_frac": None}, 0.02) is False
