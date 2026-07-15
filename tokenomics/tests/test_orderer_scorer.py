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


def test_masked_variance_explained_invariants() -> None:
    # Stage A0 (EXP-0036): VE = 1 - model_mse/zero_mse over eligible entries, where zero_mse is
    # the variance a route-uniform (zero) predictor leaves — the targets are already centered by
    # _center_clip_eligible, so mean(target^2) IS that baseline. This is the metric that was
    # recomputed by hand from samples.pkl.gz in the diagnosis; make it first-class.
    from tokenomics.train.orderer_scorer import masked_variance_explained

    t = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    assert masked_variance_explained(t, np.zeros_like(t)) == 0.0        # zero predictor -> 0
    assert masked_variance_explained(t, t) == 1.0                       # perfect predictor -> 1
    assert abs(masked_variance_explained(t, t / 2) - 0.75) < 1e-6       # half error -> 3/4 explained
    assert masked_variance_explained(np.zeros(3, np.float32), np.zeros(3)) is None  # degenerate


def test_scorer_sample_metrics_emits_variance_explained() -> None:
    # Stage A0: the training-state metric dict carries sample_variance_explained, tied to the pure
    # helper over ALL eligible entries (concat of masked pred/target across samples) — so the probe
    # writes VE directly instead of the diagnosis recomputing it from samples.pkl.gz.
    import numpy as np
    from tokenomics.train.orderer_scorer import (
        collect_oracle_orderer_samples, masked_variance_explained,
        scorer_sample_metrics, OrdererScoreModel)
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(0, depth=2)
    samples = collect_oracle_orderer_samples(
        [cfg], [0], prod=PRODUCING_BASELINES["stock_aware"],
        n_samples=2, sample_scale=1.0, lookahead=3, horizon=12)
    model = OrdererScoreModel(hidden=16)
    metrics = scorer_sample_metrics(model, samples)
    assert "sample_variance_explained" in metrics

    preds, targets = [], []
    for s in samples:
        m = np.asarray(s.mask, dtype=bool)
        preds.append(model.forward_view(s.view).detach().cpu().numpy()[m])
        targets.append(np.asarray(s.target, np.float32)[m])
    expected = masked_variance_explained(np.concatenate(targets), np.concatenate(preds))
    assert abs(metrics["sample_variance_explained"] - expected) < 1e-6


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


def test_averaged_oracle_collector_k1_matches_single() -> None:
    # Stage A0 / H1: the K-independent-search collector commits the MEAN routing vector per cycle.
    # With k_searches=1 there is nothing to average, so it must reproduce the single-search collector
    # byte-for-byte (same targets/masks/views) — the denoising variant is a strict generalization.
    import numpy as np
    from tokenomics.train.orderer_scorer import (
        collect_oracle_orderer_samples, collect_averaged_oracle_orderer_samples)
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(0, depth=2)
    kw = dict(prod=PRODUCING_BASELINES["stock_aware"], n_samples=2, sample_scale=1.0,
              lookahead=3, horizon=12)
    single = collect_oracle_orderer_samples([cfg], [0], **kw)
    avg = collect_averaged_oracle_orderer_samples([cfg], [0], k_searches=1, **kw)
    assert len(avg) == len(single) and single
    for a, b in zip(avg, single):
        assert np.allclose(a.target, b.target) and a.mask.tolist() == b.mask.tolist()


def test_averaged_oracle_collector_k4_denoises_target() -> None:
    # H1 discriminating check: averaging K RNG-varied searches per state yields a target whose
    # variance-over-uniform differs from the single search's — a valid, distill-ready sample set.
    from tokenomics.train.orderer_scorer import (
        collect_averaged_oracle_orderer_samples, fit_scorer, OrdererScoreModel)
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(0, depth=2)
    avg = collect_averaged_oracle_orderer_samples(
        [cfg], [0], k_searches=4, prod=PRODUCING_BASELINES["stock_aware"],
        n_samples=2, sample_scale=1.0, lookahead=3, horizon=12)
    assert avg
    for s in avg:
        assert s.target.shape == s.mask.shape and s.mask.any()
    fit_scorer(OrdererScoreModel(hidden=16), avg, iters=2)  # consumed by the unchanged distill path


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


def test_headroom_row_reports_baseline_columns() -> None:
    # Stage A0 / H3 STOP-gate table: the headroom row reports cumulative consumer units under the
    # named baselines even_split and proportional alongside champion (champ_cu) and search-oracle
    # (search_cu), so `oracle ~= even_split -> STOP` can be read off directly.
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
    for k in ("champ_cu", "search_cu", "even_split_cu", "proportional_cu", "teacher_headroom"):
        assert k in row
    # search-oracle is the ceiling; both named baselines are feasible orderings (non-negative units).
    assert row["search_cu"] >= row["champ_cu"] - 1e-6
    assert row["even_split_cu"] >= 0.0 and row["proportional_cu"] >= 0.0


def test_a0_probe_row_reports_headroom_and_ve_single_vs_averaged() -> None:
    # Stage A0 decision table (EXP-0036): one row carries the H3 STOP-gate baselines plus H1's
    # discriminating check — variance-explained of a scorer fit on the single-search target vs on
    # the K-averaged (denoised) target. The gate reads oracle-vs-even_split (headroom) then whether
    # ve_averaged jumps over ve_single.
    from tokenomics.eval.curriculum import TaskRecord
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.orderer_representability import a0_probe_row
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    rec = TaskRecord(
        task_id="t", task_family="scale", generator_version="test", serializer_version="v1",
        world_seed=0, episode_seed=0,
        world_spec={"generator": "build_scale_config", "params": {"depth": 2}},
    )
    row = a0_probe_row(rec, PRODUCING_BASELINES["stock_aware"], OrdererScoreModel(),
                       iters=3, horizon=12, eval_seeds=1, k_searches=2)
    for k in ("champ_cu", "search_cu", "even_split_cu", "proportional_cu",
              "ve_single", "ve_averaged", "k_searches", "n_samples_single", "n_samples_averaged"):
        assert k in row
    assert row["k_searches"] == 2
    # VE is upper-bounded by 1 (can't explain more than all the variance); it has no lower bound
    # (a barely-trained scorer can be worse than the route-uniform baseline).
    import math
    for v in (row["ve_single"], row["ve_averaged"]):
        assert math.isfinite(v) and v <= 1.0 + 1e-9


def test_a0_probe_row_reports_named_only_search_capture() -> None:
    # Representation diagnostic (does the orderer headroom live INSIDE a named-policy vocabulary?):
    # the A0 row reports search_named_cu — the SAME per-cycle search but restricted to the named
    # candidates only (n_samples=0, still switching even_split/proportional per cycle) — and
    # named_capture_frac = (search_named - even_split) / (search_full - even_split): the share of the
    # search's headroom a named-policy teacher can reach. ~1 => switch representation; ~0 => keep
    # the continuous target and denoise it (H1).
    from tokenomics.eval.curriculum import TaskRecord
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.orderer_representability import a0_probe_row
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    rec = TaskRecord(
        task_id="t", task_family="scale", generator_version="test", serializer_version="v1",
        world_seed=0, episode_seed=0,
        world_spec={"generator": "build_scale_config", "params": {"depth": 2}},
    )
    row = a0_probe_row(rec, PRODUCING_BASELINES["stock_aware"], OrdererScoreModel(),
                       iters=3, horizon=12, eval_seeds=1, k_searches=2)
    assert "search_named_cu" in row and "named_capture_frac" in row
    assert row["search_named_cu"] >= 0.0
    # named-only is the full search minus its Gaussian samples: a subset of candidates, so it cannot
    # exceed the full search's on-task ceiling (allow a small slack for trajectory divergence).
    assert row["search_named_cu"] <= row["search_cu"] + 5.0


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


def test_emit_a0_writes_exp_layout(tmp_path) -> None:
    # Stage A0 writes a pullable EXP layout; RESULT.md is headroom-first then the VE jump. No compute.
    from tokenomics.eval.orderer_representability import _emit_a0

    rows = [{"task_id": "t1", "champ_cu": 100.0, "search_cu": 140.0, "even_split_cu": 120.0,
             "proportional_cu": 118.0, "teacher_headroom": 40.0, "headroom_frac": 0.4,
             "k_searches": 4, "ve_single": 0.1, "ve_averaged": 0.55,
             "n_samples_single": 300, "n_samples_averaged": 305}]
    _emit_a0(tmp_path, rows, meta={"iters": 1000, "k_searches": 4})
    assert (tmp_path / "summary.csv").exists() and (tmp_path / "spec.json").exists()
    result = (tmp_path / "RESULT.md").read_text()
    assert "ve_averaged" in result and "Headroom" in result


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


def test_headroom_row_anchors_at_record_episode_seed(monkeypatch) -> None:
    # EXP-0036 probe bug regression: the probe evaluated at seeds range(eval_seeds) while the
    # mined gap lives at the record's episode_seed block (`_episode_seeds`) — a different demand
    # trace entirely (champ alive at 0-2, dead at the archived 77-79). The row must anchor every
    # eval at the archive convention.
    from tokenomics.eval import runner as runner_mod
    from tokenomics.eval import orderer_oracle as oracle_mod
    from tokenomics.eval.curriculum import TaskRecord
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.orderer_representability import headroom_row
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    seen: list[int] = []

    def fake_run_episode(env, prod, order, seed):
        seen.append(seed)
        return {"consumer_completed_units": float(seed)}

    def fake_search(env, prod, cand, n_samples, sample_scale, lookahead, seed):
        seen.append(seed)
        return {"consumer_completed_units": float(seed)}, None

    monkeypatch.setattr(runner_mod, "run_episode", fake_run_episode)
    monkeypatch.setattr(oracle_mod, "search_orderer_episode", fake_search)

    rec = TaskRecord(
        task_id="t", task_family="scale", generator_version="test", serializer_version="v1",
        world_seed=7, episode_seed=7,
        world_spec={"generator": "build_scale_config", "params": {"depth": 2}},
    )
    row = headroom_row(rec, PRODUCING_BASELINES["stock_aware"], OrdererScoreModel(),
                       horizon=12, eval_seeds=2)
    assert sorted(set(seen)) == [7, 8], f"evals must run at episode seeds 7-8, saw {sorted(set(seen))}"
    assert row["champ_cu"] == pytest.approx(7.5)  # mean over the anchored block, not range(2)


def test_deploy_trace_records_cycle_capacity() -> None:
    # EXP-0036 retry diagnostic: the representability probe must save enough deploy-time
    # state to identify whether the distilled orderer dies after leaving teacher states.
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.orderer_representability import (
        _trace_deployed_orderer_episode,
        _trace_summary,
    )
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(
        0, depth=2, products_per_layer=2, producers_per_layer=2,
        num_consumers=1, quantity_max=2,
    )
    stats, rows = _trace_deployed_orderer_episode(
        cfg,
        PRODUCING_BASELINES["stock_aware"],
        ORDERING_BASELINES["proportional"],
        seed=0,
        horizon=3,
    )

    assert stats["cycles"] == 3
    assert len(rows) == 3
    first = rows[0]
    for key in (
        "cycle", "dead", "consumer_completed_units_delta",
        "replenishment_completed_delta", "min_capacity_type_current",
        "capacity_capacity_0_current", "capacity_capacity_0_reserved",
    ):
        assert key in first
    summary = _trace_summary(rows)
    assert summary["trace_cycles"] == 3
    assert summary["trace_min_capacity_type_current"] == min(
        r["min_capacity_type_current"] for r in rows
    )


def test_trace_summary_dead_across_combined_seed_rows() -> None:
    # The debug trace concatenates rows from multiple eval seeds. A death in seed A must not
    # be hidden just because the final row comes from seed B and is alive.
    from tokenomics.eval.orderer_representability import _trace_summary

    summary = _trace_summary([
        {"seed": 1, "cycle": 8, "dead": True, "min_capacity_type_current": 0},
        {"seed": 2, "cycle": 100, "dead": False, "min_capacity_type_current": 4},
    ])

    assert summary["trace_dead"] is True
    assert summary["trace_dead_seed_count"] == 1
    assert summary["trace_first_dead_cycle"] == 8
    assert summary["trace_zero_capacity_seed_count"] == 1


def test_ordering_blend_spans_even_split_to_proportional() -> None:
    # Stage A1b vocabulary (A0b diagnostic): `ordering_blend(alpha)` is the logit-space blend
    # `alpha*even_split + (1-alpha)*proportional`. even_split emits zero logits, so the family is a
    # temperature on the proportional logits (shares ~ throughput^(1-alpha)) — exactly what a
    # per-stream blend head can deploy, so search vocabulary == deployable head.
    from tokenomics.eval.baselines import (
        ordering_blend, ordering_even_split, ordering_proportional)

    env, _ = _active_stream_view()
    agent = next(a for a in env.possible_agents if env._role[a][0] == "order")
    even = ordering_even_split(env, agent)
    prop = ordering_proportional(env, agent)
    assert np.allclose(ordering_blend(1.0)(env, agent), even)
    assert np.allclose(ordering_blend(0.0)(env, agent), prop)
    half = ordering_blend(0.5)(env, agent)
    assert np.allclose(half, 0.5 * even + 0.5 * prop)
    assert half.dtype == np.float32


def test_collect_choice_samples_records_committed_vocab_choice() -> None:
    # A0b / D1 (categorical learnability): drive the named+blend search episode (n_samples=0)
    # and record, per product-stream view with a backlog, WHICH vocabulary entry the per-cycle
    # search committed plus the decisiveness margin (best minus second-best rollout units).
    # This is the dataset that tests whether the committed named choice is predictable from the
    # stream observation — the learnability question named-capture cannot answer.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.orderer_scorer import collect_choice_samples

    cfg = build_scale_config(0, depth=2, products_per_layer=2, producers_per_layer=2,
                             num_consumers=1, quantity_max=2)
    samples, _, _, vocab = collect_choice_samples(
        [cfg], [0], prod=PRODUCING_BASELINES["stock_aware"], alphas=(0.5,),
        lookahead=2, horizon=4)
    assert vocab == ["even_split", "blend_0.5", "proportional"]
    assert samples, "an active tiny economy must yield backlog stream views"
    for s in samples:
        assert 0 <= s.choice < len(vocab)
        assert s.margin >= 0.0
        assert s.view.backlog


def test_collect_choice_samples_also_records_per_cycle_global_state() -> None:
    # A0c / D1 discriminator: the same search pass must also snapshot the FULL global state
    # (env.global_state(), ADR-0004) once per cycle, paired with the committed choice + margin —
    # the dataset that separates "choice needs global context" from "choice is rollout noise".
    # Only cycles that produced stream samples are recorded, so the global probe's majority
    # floor is computed on the same cycle population as the per-stream one.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.orderer_scorer import collect_choice_samples

    cfg = build_scale_config(0, depth=2, products_per_layer=2, producers_per_layer=2,
                             num_consumers=1, quantity_max=2)
    samples, global_samples, _, vocab = collect_choice_samples(
        [cfg], [0], prod=PRODUCING_BASELINES["stock_aware"], alphas=(0.5,),
        lookahead=2, horizon=4)
    assert vocab == ["even_split", "blend_0.5", "proportional"]
    assert samples and global_samples
    # one global sample per cycle with stream samples -> never more than stream samples
    assert len(global_samples) <= len(samples)
    dim = global_samples[0].view.shape
    stream_pairs = {(s.choice, s.margin) for s in samples}
    for g in global_samples:
        assert g.view.ndim == 1 and g.view.shape == dim and g.view.dtype == np.float32
        assert (g.choice, g.margin) in stream_pairs


def test_choice_model_fits_and_reports_accuracy_vs_majority() -> None:
    # A0b / D1 machinery: a per-stream categorical head (same encoders, mean-pooled producer
    # rows) predicts the committed vocabulary choice from the view; `choice_metrics` reports
    # top-1 accuracy against the majority-class floor, overall and on decisive (margin>0)
    # samples only. On its own training set a converged fit must at least match majority —
    # the model can always represent the constant predictor.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.orderer_scorer import (
        OrdererChoiceModel, choice_metrics, collect_choice_samples, fit_choice_model)

    cfg = build_scale_config(0, depth=2, products_per_layer=2, producers_per_layer=2,
                             num_consumers=1, quantity_max=2)
    samples, _, _, vocab = collect_choice_samples(
        [cfg], [0], prod=PRODUCING_BASELINES["stock_aware"], alphas=(0.5,),
        lookahead=2, horizon=4)
    model = OrdererChoiceModel(n_choices=len(vocab))
    logits = model.forward_view(samples[0].view)
    assert logits.shape == (len(vocab),)

    fit_choice_model(model, samples, iters=150)
    m = choice_metrics(model, samples)
    for k in ("top1", "majority", "n", "top1_decisive", "majority_decisive", "n_decisive"):
        assert k in m
    assert m["n"] == len(samples)
    assert 0.0 <= m["majority"] <= 1.0
    assert m["top1"] >= m["majority"] - 1e-9


def test_global_choice_model_fits_from_standardized_global_state() -> None:
    # A0c machinery: a plain MLP on the FULL global state predicts the committed vocabulary
    # choice; inputs are standardized with train-set stats (set_norm — EXP-0009 lesson: raw
    # global_state magnitudes wreck optimization). Reuses fit_choice_model/choice_metrics via
    # the shared `.view/.choice/.margin` sample shape. On its own training set a converged fit
    # must at least match the majority floor.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.orderer_scorer import (
        GlobalChoiceModel, choice_metrics, collect_choice_samples, fit_choice_model)

    cfg = build_scale_config(0, depth=2, products_per_layer=2, producers_per_layer=2,
                             num_consumers=1, quantity_max=2)
    _, gsamples, _, vocab = collect_choice_samples(
        [cfg], [0], prod=PRODUCING_BASELINES["stock_aware"], alphas=(0.5,),
        lookahead=2, horizon=4)
    assert gsamples
    model = GlobalChoiceModel(n_choices=len(vocab), state_dim=gsamples[0].view.shape[0])
    model.set_norm([g.view for g in gsamples])
    logits = model.forward_view(gsamples[0].view)
    assert logits.shape == (len(vocab),)

    fit_choice_model(model, gsamples, iters=150)
    m = choice_metrics(model, gsamples)
    assert m["n"] == len(gsamples)
    assert m["top1"] >= m["majority"] - 1e-9


def test_a0b_diag_row_reports_blend_capture_and_choice_learnability() -> None:
    # A0b decision row: D2 = does adding alpha-blend candidates to the named-only search recover
    # more of the full search's headroom (search_blend_cu, blend_capture_frac)? D1 = is the
    # committed vocabulary choice predictable from the stream view on a HELD-OUT seed
    # (choice_top1 vs choice_majority)? Together they test what named_capture_frac left open.
    from tokenomics.eval.curriculum import TaskRecord
    from tokenomics.eval.orderer_representability import a0b_diag_row

    rec = TaskRecord(
        task_id="t", task_family="scale", generator_version="test", serializer_version="v1",
        world_seed=0, episode_seed=0,
        world_spec={"generator": "build_scale_config", "params": {"depth": 2}},
    )
    row = a0b_diag_row(rec, PRODUCING_BASELINES["stock_aware"], iters=3, horizon=12,
                       eval_seeds=2, alphas=(0.5,))
    for k in ("search_cu", "even_split_cu", "search_named_cu", "search_blend_cu",
              "named_capture_frac", "blend_capture_frac", "choice_top1", "choice_majority",
              "choice_top1_decisive", "choice_majority_decisive",
              "n_choice_train", "n_choice_heldout", "choice_heldout", "vocab",
              # A0c global-state probe (missing-context vs choice-noise discriminator)
              "gchoice_top1", "gchoice_majority", "gchoice_top1_decisive",
              "gchoice_majority_decisive", "n_gchoice_train", "n_gchoice_heldout"):
        assert k in row
    # blend vocabulary is a superset of the named one; per-cycle greedy search does not guarantee
    # episode-level monotonicity, so allow the same small trajectory-divergence slack as A0.
    assert row["search_blend_cu"] >= 0.0
    assert row["search_blend_cu"] <= row["search_cu"] + 5.0
    # eval_seeds=2 -> the choice split holds out the last episode seed (a real generalization read)
    assert row["choice_heldout"] is True
    assert row["vocab"] == ["even_split", "blend_0.5", "proportional"]


def test_emit_a0b_writes_exp_layout(tmp_path) -> None:
    # A0b writes a pullable EXP layout; RESULT.md reads blend-capture (D2) then held-out choice
    # accuracy vs majority (D1). No compute — fake rows.
    from tokenomics.eval.orderer_representability import _emit_a0b

    rows = [{"task_id": "t1", "search_cu": 1000.0, "even_split_cu": 500.0,
             "search_named_cu": 850.0, "search_blend_cu": 950.0,
             "named_capture_frac": 0.7, "blend_capture_frac": 0.9,
             "choice_top1": 0.8, "choice_majority": 0.55,
             "choice_top1_decisive": 0.85, "choice_majority_decisive": 0.5,
             "n_choice_decisive": 40, "n_choice_train": 200, "n_choice_heldout": 100,
             "gchoice_top1": 0.7, "gchoice_majority": 0.55,
             "gchoice_top1_decisive": 0.75, "gchoice_majority_decisive": 0.5,
             "n_gchoice_train": 80, "n_gchoice_heldout": 40,
             "choice_heldout": True, "vocab": ["even_split", "blend_0.5", "proportional"]}]
    _emit_a0b(tmp_path, rows, meta={"iters": 1000, "alphas": [0.5]})
    assert (tmp_path / "summary.csv").exists() and (tmp_path / "spec.json").exists()
    result = (tmp_path / "RESULT.md").read_text()
    assert "blend_capture_frac" in result and "choice_top1" in result
    # A0c: the global-state probe (missing-context vs choice-noise) must be in the read order
    assert "gchoice_top1" in result
    assert "gchoice_top1" in (tmp_path / "summary.csv").read_text()


def test_collect_choice_samples_records_cycle_traces_with_candidate_tables() -> None:
    # A0t trace analysis (EXP-0036): the same search pass must persist, per searched cycle, the
    # FULL candidate score table the argmax throws away — plus a JSON-serializable domain-state
    # snapshot readable without the encoder. This turns "top-1 vs majority" into per-cycle
    # REGRET (units[teacher pick] - units[student pick]) and lets a human read WHY the teacher
    # chose differently at the cycles that cost units.
    import json

    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.orderer_scorer import collect_choice_samples

    cfg = build_scale_config(0, depth=2, products_per_layer=2, producers_per_layer=2,
                             num_consumers=1, quantity_max=2)
    samples, gsamples, traces, vocab = collect_choice_samples(
        [cfg], [0], prod=PRODUCING_BASELINES["stock_aware"], alphas=(0.5,),
        lookahead=2, horizon=4)
    assert samples and traces
    # traces share the global-sample cycle population 1:1 (same majority-floor basis)
    assert len(traces) == len(gsamples)
    for t, g in zip(traces, gsamples):
        assert len(t.cand_units) == len(vocab)
        assert t.choice == g.choice == int(np.argmax(t.cand_units))
        assert t.margin == g.margin and t.margin >= 0.0
        assert t.champion_units is None  # no champion policy handed in
        assert t.seed == 0 and 0 <= t.cycle < 4
        json.dumps(t.state)  # snapshot must be JSON-serializable
        prods = t.state["producers"]
        assert prods
        for p in prods.values():
            assert p["capacities"] and p["max_capacities"]  # per-TYPE detail, not summed
        assert "orders" in t.state and "open_demand" in t.state
        assert t.state["cycle"] == t.cycle


def test_collect_choice_samples_scores_champion_at_the_searched_states() -> None:
    # A0t champion regret: handing a `champion` ordering policy in must score it per searched
    # cycle by the SAME lookahead rollout as the vocabulary candidates — the direct measure of
    # what each champion routing decision costs vs the teacher at the teacher's own states.
    # Determinism gives an exact oracle: a "champion" that IS the proportional baseline must
    # reproduce the proportional column of the candidate table.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.orderer_scorer import collect_choice_samples

    cfg = build_scale_config(0, depth=2, products_per_layer=2, producers_per_layer=2,
                             num_consumers=1, quantity_max=2)
    _, _, traces, vocab = collect_choice_samples(
        [cfg], [0], prod=PRODUCING_BASELINES["stock_aware"], alphas=(0.5,),
        lookahead=2, horizon=4, champion=ORDERING_BASELINES["proportional"])
    assert traces
    prop = vocab.index("proportional")
    for t in traces:
        assert t.champion_units is not None
        assert t.champion_units == t.cand_units[prop]


def test_a0t_diag_row_reports_heldout_regret_for_probe_and_champion() -> None:
    # A0t decision row: replaces "top-1 vs majority" with what a miss COSTS. On the held-out
    # seed, fit the global-state probe on train seeds, then per searched cycle read
    # probe_regret = units[teacher pick] - units[probe pick] (>= 0 by argmax construction) and
    # champ_regret = units[teacher pick] - champion rollout units (negative = champion beats
    # the whole vocabulary at that state). Per-cycle detail keeps the candidate table + state
    # snapshot so a human can read the expensive cycles.
    from tokenomics.eval.curriculum import TaskRecord
    from tokenomics.eval.orderer_representability import a0t_diag_row

    rec = TaskRecord(
        task_id="t", task_family="scale", generator_version="test", serializer_version="v1",
        world_seed=0, episode_seed=0,
        world_spec={"generator": "build_scale_config", "params": {"depth": 2}},
    )
    row, detail = a0t_diag_row(rec, PRODUCING_BASELINES["stock_aware"],
                               ORDERING_BASELINES["proportional"], iters=3, horizon=12,
                               eval_seeds=2, alphas=(0.5,))
    for k in ("task_id", "world_seed", "vocab", "n_traces_train", "n_traces_heldout",
              "probe_top1", "probe_regret_total", "probe_regret_mean", "probe_regret_max",
              "n_probe_miss", "champ_regret_total", "champ_regret_mean", "champ_regret_max",
              "n_champ_worse", "n_champ_better", "local_headroom_total",
              "probe_regret_frac", "champ_regret_frac"):
        assert k in row
    cycles = detail["cycles"]
    assert len(cycles) == row["n_traces_heldout"] > 0
    for c in cycles:
        assert c["probe_regret"] >= 0.0  # teacher is the argmax of the same table
        assert c["probe_regret"] == c["cand_units"][c["choice"]] - c["cand_units"][c["probe_pred"]]
        assert c["champ_regret"] == c["cand_units"][c["choice"]] - c["champion_units"]
        assert "state" in c and "margin" in c


def test_emit_a0t_writes_exp_layout_with_top_regret_dump(tmp_path) -> None:
    # A0t writes a pullable EXP layout; RESULT.md must lead with the regret read (not top-1)
    # and dump the highest-regret held-out cycles WITH their state snapshots — the material a
    # human reads to infer which missing observation would have explained the teacher's pick.
    # Full per-cycle traces go to artifacts/traces.json.gz.
    import gzip
    import json

    from tokenomics.eval.orderer_representability import _emit_a0t

    vocab = ["even_split", "blend_0.5", "proportional"]
    cyc = [{"seed": 0, "cycle": 3, "cand_units": [10.0, 12.0, 7.0], "choice": 1, "margin": 2.0,
            "probe_pred": 2, "probe_regret": 5.0, "champion_units": 6.0, "champ_regret": 6.0,
            "local_headroom": 2.0, "state": {"cycle": 3, "producers": {"p0": {}}, "orders": [],
                                             "open_demand": {}}},
           {"seed": 0, "cycle": 4, "cand_units": [9.0, 9.0, 9.0], "choice": 0, "margin": 0.0,
            "probe_pred": 0, "probe_regret": 0.0, "champion_units": 9.0, "champ_regret": 0.0,
            "local_headroom": 0.0, "state": {"cycle": 4, "producers": {"p0": {}}, "orders": [],
                                             "open_demand": {}}}]
    rows = [{"task_id": "t1", "world_seed": 0, "vocab": vocab, "n_traces_train": 20,
             "n_traces_heldout": 2, "probe_top1": 0.5, "probe_regret_total": 5.0,
             "probe_regret_mean": 2.5, "probe_regret_max": 5.0, "n_probe_miss": 1,
             "champ_regret_total": 6.0, "champ_regret_mean": 3.0, "champ_regret_max": 6.0,
             "n_champ_worse": 1, "n_champ_better": 0, "local_headroom_total": 2.0,
             "probe_regret_frac": 2.5, "champ_regret_frac": 3.0}]
    details = [{"task_id": "t1", "world_seed": 0, "vocab": vocab, "heldout_seed": 2,
                "cycles": cyc}]
    _emit_a0t(tmp_path, rows, details, meta={"iters": 5})
    assert (tmp_path / "summary.csv").exists() and (tmp_path / "spec.json").exists()
    result = (tmp_path / "RESULT.md").read_text()
    assert "probe_regret" in result and "champ_regret" in result
    assert "Top-regret" in result or "top-regret" in result
    # the highest-regret cycle appears with its state snapshot rendered
    assert '"cycle": 3' in result and "producers" in result
    with gzip.open(tmp_path / "artifacts" / "traces.json.gz", "rt") as f:
        assert json.load(f)[0]["cycles"][0]["cand_units"] == [10.0, 12.0, 7.0]


def test_collect_deploy_traces_follows_the_champion_trajectory() -> None:
    # A0u failure chains: drive the episode with the CHAMPION orderer (not the searched policy)
    # and, at every cycle of ITS OWN trajectory, score the vocabulary + the champion by the same
    # lookahead rollout. This is the dataset that answers where a 0-unit game goes wrong: the
    # counterfactual teacher pick exists per cycle, regret localizes the failure, the snapshot
    # explains it. Determinism oracle: champion==proportional => its rollout score reproduces
    # the proportional table column, and the episode equals a plain proportional episode.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.runner import run_episode
    from tokenomics.train.orderer_scorer import collect_deploy_traces

    cfg = build_scale_config(0, depth=2, products_per_layer=2, producers_per_layer=2,
                             num_consumers=1, quantity_max=2)
    episodes, vocab = collect_deploy_traces(
        [cfg], [0], prod=PRODUCING_BASELINES["stock_aware"],
        champion=ORDERING_BASELINES["proportional"], alphas=(0.5,), lookahead=2, horizon=4)
    assert vocab == ["even_split", "blend_0.5", "proportional"]
    (ep,) = episodes
    assert ep["seed"] == 0 and ep["traces"]
    prop = vocab.index("proportional")
    prev_delivered = 0
    for t in ep["traces"]:
        assert t.champion_units == t.cand_units[prop]
        assert t.choice == int(np.argmax(t.cand_units))  # counterfactual teacher pick
        d = t.state["progress"]["units_delivered"]
        assert d >= prev_delivered  # cumulative progress line, non-decreasing
        prev_delivered = d
    # episode outcome must equal a plain proportional episode (same trajectory)
    ref = run_episode(joint_stack_env(cfg, 4), PRODUCING_BASELINES["stock_aware"],
                      ORDERING_BASELINES["proportional"], seed=0)
    assert ep["consumer_completed_units"] == ref["consumer_completed_units"]
    assert ep["dead"] == bool(ref["dead"])


def test_a0u_rows_localize_failure_onset_from_deploy_traces() -> None:
    # A0u chain rows: per (task, seed) episode, champ_regret[c] = max(vocab table + champion) -
    # champion (>= 0: the counterfactual best includes the champion itself), onset = first cycle
    # where regret clears the threshold, peak = its argmax. These localize WHERE the badly-played
    # game starts losing; the trace detail keeps tables + snapshots for reading WHY.
    from tokenomics.eval.orderer_representability import a0u_rows
    from tokenomics.train.orderer_scorer import CycleTrace

    def tr(cycle, units, champ):
        return CycleTrace(seed=7, cycle=cycle, cand_units=units, choice=int(np.argmax(units)),
                          margin=0.0, champion_units=champ,
                          state={"cycle": cycle, "producers": {}, "orders": [],
                                 "open_demand": {}, "progress": {"units_delivered": 0,
                                                                 "units_rejected": 0}})
    episodes = [{
        "seed": 7,
        "traces": [tr(0, [10.0, 9.0, 8.0], 10.0),   # regret 0 (champion == best)
                   tr(1, [12.0, 11.0, 8.0], 11.0),  # regret 1 (below onset threshold)
                   tr(2, [30.0, 20.0, 10.0], 12.0), # regret 18 -> onset
                   tr(3, [25.0, 40.0, 10.0], 15.0)],# regret 25 -> peak
        "consumer_completed_units": 3, "consumer_rejected_units": 50,
        "dead": True, "cycles": 4,
    }]
    rows, details = a0u_rows("task-x", 45, episodes, ["even_split", "blend_0.5", "proportional"],
                             onset=5.0)
    (row,), (det,) = rows, details
    assert row["task_id"] == "task-x" and row["seed"] == 7 and row["dead"] is True
    assert row["champ_cu"] == 3 and row["cycles"] == 4
    assert row["regret_total"] == 0.0 + 1.0 + 18.0 + 25.0
    assert row["onset_cycle"] == 2 and row["peak_cycle"] == 3 and row["regret_max"] == 25.0
    assert row["n_regret_pos"] == 3
    assert [c["champ_regret"] for c in det["cycles"]] == [0.0, 1.0, 18.0, 25.0]
    assert det["cycles"][2]["teacher_pick"] == "even_split"


def test_emit_a0u_writes_failure_chain_dump(tmp_path) -> None:
    # A0u layout: per-episode chain sections must render a progress line (delivered units +
    # regret along the champion's own game) and the top-regret cycles with snapshots; full
    # chains go to artifacts/deploy_traces.json.gz.
    import gzip
    import json

    from tokenomics.eval.orderer_representability import _emit_a0u

    vocab = ["even_split", "blend_0.5", "proportional"]
    cycles = [{"cycle": 0, "cand_units": [10.0, 9.0, 8.0], "choice": 0,
               "teacher_pick": "even_split", "margin": 1.0, "champion_units": 10.0,
               "champ_regret": 0.0,
               "state": {"cycle": 0, "producers": {"p0": {}}, "orders": [], "open_demand": {},
                         "progress": {"units_delivered": 0, "units_rejected": 0}}},
              {"cycle": 1, "cand_units": [30.0, 20.0, 10.0], "choice": 0,
               "teacher_pick": "even_split", "margin": 10.0, "champion_units": 12.0,
               "champ_regret": 18.0,
               "state": {"cycle": 1, "producers": {"p0": {}}, "orders": [], "open_demand": {},
                         "progress": {"units_delivered": 0, "units_rejected": 9}}}]
    rows = [{"task_id": "t1", "world_seed": 45, "seed": 7, "champ_cu": 0, "rejected_units": 50,
             "dead": True, "cycles": 2, "n_traces": 2, "regret_total": 18.0, "regret_max": 18.0,
             "onset_cycle": 1, "peak_cycle": 1, "n_regret_pos": 1}]
    details = [{"task_id": "t1", "world_seed": 45, "seed": 7, "vocab": vocab, "cycles": cycles}]
    _emit_a0u(tmp_path, rows, details, meta={"onset": 5.0})
    assert (tmp_path / "summary.csv").exists() and (tmp_path / "spec.json").exists()
    result = (tmp_path / "RESULT.md").read_text()
    assert "onset_cycle" in result and "champ_regret" in result
    assert "Failure chain" in result or "failure chain" in result
    assert '"units_rejected": 9' in result  # snapshot of the onset cycle is rendered
    with gzip.open(tmp_path / "artifacts" / "deploy_traces.json.gz", "rt") as f:
        assert json.load(f)[0]["cycles"][1]["champ_regret"] == 18.0


def test_cycle_feature_blocks_named_layout_with_per_type_truth() -> None:
    # A0f missing-observation probe (EXP-0036 Task 1): per decision-time cycle, a dict of NAMED
    # flat observation blocks so a probe can be fit on cumulative block sets and rank which
    # missing observation family carries the teacher's choice. Layout is deterministic (sorted
    # producers x sorted capacity types x sorted products); every block is finite float32. The
    # per-type capacity block must carry the PER-TYPE truth the lossy global_state sums away.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.orderer_scorer import cycle_feature_blocks

    cfg = build_scale_config(0, depth=2, products_per_layer=2, producers_per_layer=2,
                             num_consumers=1, quantity_max=2)
    env = TokenomicsAECEnv(config=cfg, horizon=4, entity_producer_obs=True,
                           product_stream_ordering=True, corrected_stream_k=True)
    env.reset(seed=0)
    blocks = cycle_feature_blocks(env)
    expected = {"gstate", "stream", "per_type_capacity", "order_queue", "inventory",
                "inbound", "relational", "compact"}
    assert set(blocks) == expected
    for name, arr in blocks.items():
        assert arr.ndim == 1 and arr.dtype == np.float32, name
        assert np.isfinite(arr).all(), name

    producers = sorted(env.system.producing_agents)
    types = sorted({t for p in env.system.producing_agents.values() for t in p.max_capacities})
    products = sorted(env.system.process_sheets)
    assert blocks["gstate"].shape == np.asarray(env.global_state(), dtype=np.float32).shape
    assert blocks["per_type_capacity"].shape == (len(producers) * len(types) * 4,)
    assert blocks["inventory"].shape == (len(producers) * len(products),)
    assert blocks["inbound"].shape == (len(producers) * len(products),)
    # first producer, first type: [current, max, reserved, requested] from the live domain state
    p0 = env.system.producing_agents[producers[0]]
    assert blocks["per_type_capacity"][0] == float(p0.capacities.get(types[0], 0))
    assert blocks["per_type_capacity"][1] == float(p0.max_capacities.get(types[0], 0))


def test_collect_choice_samples_attaches_feature_blocks_when_requested() -> None:
    # A0f dataset: `with_features=True` must attach the decision-time feature blocks to every
    # CycleTrace (same key set + dims across traces, so block matrices stack); the default
    # keeps traces lean (features None) for the existing A0t/A0u callers.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.orderer_scorer import collect_choice_samples

    cfg = build_scale_config(0, depth=2, products_per_layer=2, producers_per_layer=2,
                             num_consumers=1, quantity_max=2)
    _, _, plain, _ = collect_choice_samples(
        [cfg], [0], prod=PRODUCING_BASELINES["stock_aware"], alphas=(0.5,),
        lookahead=2, horizon=4)
    assert plain and all(t.features is None for t in plain)

    _, _, traces, _ = collect_choice_samples(
        [cfg], [0], prod=PRODUCING_BASELINES["stock_aware"], alphas=(0.5,),
        lookahead=2, horizon=4, with_features=True)
    assert traces
    keys = set(traces[0].features)
    assert "per_type_capacity" in keys and "stream" in keys
    dims = {k: v.shape for k, v in traces[0].features.items()}
    for t in traces:
        assert set(t.features) == keys
        assert {k: v.shape for k, v in t.features.items()} == dims


def test_history_features_stack_previous_cycles_zero_padded_per_seed() -> None:
    # A0f history block: per trace, the concatenation of the previous k cycles' `compact`
    # vectors (most recent first), zero-padded at episode start and NEVER leaking across
    # episode seeds — history is a backward proxy for what the lookahead unrolls forward.
    from tokenomics.train.orderer_scorer import CycleTrace, history_features

    def tr(seed, cycle, val):
        return CycleTrace(seed=seed, cycle=cycle, cand_units=[0.0], choice=0, margin=0.0,
                          champion_units=None, state={},
                          features={"compact": np.asarray([val, -val], dtype=np.float32)})

    traces = [tr(0, 0, 1.0), tr(0, 1, 2.0), tr(0, 2, 3.0), tr(1, 0, 7.0), tr(1, 1, 8.0)]
    h = history_features(traces, k=2)
    assert h.shape == (5, 4) and h.dtype == np.float32
    np.testing.assert_array_equal(h[0], [0, 0, 0, 0])          # episode start: all padding
    np.testing.assert_array_equal(h[1], [1, -1, 0, 0])         # cycle-1 first, then padding
    np.testing.assert_array_equal(h[2], [2, -2, 1, -1])        # cycle-1, cycle-2
    np.testing.assert_array_equal(h[3], [0, 0, 0, 0])          # new seed: no cross-episode leak
    np.testing.assert_array_equal(h[4], [7, -7, 0, 0])


def test_value_table_model_recovers_a_linear_candidate_table() -> None:
    # A0f probe head: regress the FULL candidate value table (not a hard argmax) from a flat
    # feature vector; the pick is the predicted argmax but quality is measured by REGRET
    # against the true table, so tied winners are free (HANDOVER: ties are sets, accuracy is
    # the wrong metric). On a noiseless linear table the fit must drive train regret ~0.
    import torch
    from tokenomics.train.orderer_scorer import (
        ValueTableModel, fit_value_table, value_table_metrics)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6)).astype(np.float32)
    W = rng.normal(size=(6, 3)).astype(np.float32)
    Y = (X @ W).astype(np.float32)
    torch.manual_seed(0)
    model = ValueTableModel(state_dim=6, n_choices=3)
    model.set_norm(X)
    assert model.forward_states(torch.from_numpy(X[:2])).shape == (2, 3)

    fit_value_table(model, X, Y, iters=400)
    m = value_table_metrics(model, X, Y, even_idx=0)
    for k in ("top1", "winner_hit", "regret_total", "regret_frac", "headroom_total", "n"):
        assert k in m
    assert m["n"] == 80
    assert m["regret_frac"] is not None and m["regret_frac"] < 0.1
    assert m["winner_hit"] >= m["top1"]  # winner-set credit can only add to exact-argmax credit


def test_a0f_feature_row_emits_cumulative_block_ladder() -> None:
    # A0f decision rows: per task, one row per block set — the lossy `gstate` reference, the
    # local `stream` block, then the handover ladder added cumulatively — each carrying train
    # AND held-out regret_frac (train regret separates "cannot even fit" from "does not
    # generalize"). Dims must strictly grow along the cumulative ladder.
    from tokenomics.eval.curriculum import TaskRecord
    from tokenomics.eval.orderer_representability import a0f_feature_row

    rec = TaskRecord(
        task_id="t", task_family="scale", generator_version="test", serializer_version="v1",
        world_seed=0, episode_seed=0,
        world_spec={"generator": "build_scale_config", "params": {"depth": 2}},
    )
    rows, detail = a0f_feature_row(rec, PRODUCING_BASELINES["stock_aware"], iters=3,
                                   horizon=12, eval_seeds=2, alphas=(0.5,), k_history=2)
    names = [r["block_set"] for r in rows]
    assert names == ["gstate", "stream", "+per_type_capacity", "+order_queue", "+inventory",
                     "+inbound", "+history", "+relational"]
    for r in rows:
        for k in ("task_id", "world_seed", "block_set", "blocks", "dim", "n_train",
                  "n_heldout", "train_regret_frac", "train_top1", "regret_frac", "top1",
                  "winner_hit", "regret_total", "headroom_total"):
            assert k in r
        assert r["n_heldout"] > 0
    ladder = [r["dim"] for r in rows[1:]]
    assert all(a < b for a, b in zip(ladder, ladder[1:]))
    # detail persists the traces (with features) for local re-analysis without a pod re-run
    assert detail["task_id"] == "t" and detail["vocab"] == ["even_split", "blend_0.5",
                                                            "proportional"]
    assert len(detail["train"]) == rows[0]["n_train"]
    assert len(detail["heldout"]) == rows[0]["n_heldout"]
    c = detail["heldout"][0]
    assert "cand_units" in c and "features" in c and isinstance(
        c["features"]["stream"], list)


def test_emit_a0f_writes_exp_layout_with_regret_ladder(tmp_path) -> None:
    # A0f layout: summary.csv + spec.json + RESULT.md leading with the per-block regret_frac
    # ladder (the Task 1 definition of done: a table with regret_frac per feature block);
    # traces with features go to artifacts for local re-fitting.
    import gzip
    import json

    from tokenomics.eval.orderer_representability import _emit_a0f

    rows = [{"task_id": "t1", "world_seed": 0, "block_set": "gstate", "blocks": ["gstate"],
             "dim": 40, "n_train": 200, "n_heldout": 100, "train_regret_frac": 0.05,
             "train_top1": 0.9, "regret_frac": 0.9, "top1": 0.3, "winner_hit": 0.4,
             "regret_total": 450.0, "headroom_total": 500.0},
            {"task_id": "t1", "world_seed": 0, "block_set": "+per_type_capacity",
             "blocks": ["stream", "per_type_capacity"], "dim": 150, "n_train": 200,
             "n_heldout": 100, "train_regret_frac": 0.02, "train_top1": 0.95,
             "regret_frac": 0.4, "top1": 0.6, "winner_hit": 0.7, "regret_total": 200.0,
             "headroom_total": 500.0}]
    details = [{"task_id": "t1", "world_seed": 0, "vocab": ["even_split", "proportional"],
                "heldout_seed": 2, "train": [], "heldout": []}]
    _emit_a0f(tmp_path, rows, details, meta={"iters": 5, "k_history": 4})
    assert (tmp_path / "summary.csv").exists() and (tmp_path / "spec.json").exists()
    result = (tmp_path / "RESULT.md").read_text()
    assert "regret_frac" in result and "+per_type_capacity" in result
    assert "block" in result.lower()
    with gzip.open(tmp_path / "artifacts" / "feature_traces.json.gz", "rt") as f:
        assert json.load(f)[0]["task_id"] == "t1"


def test_cycle_set_features_reshape_named_blocks_into_producer_rows() -> None:
    # Producer-count-agnostic value table (EXP-0036 HANDOVER "Proposed Model"): the set view
    # is the SAME information as the flat A0f blocks, reshaped — a fixed-dim stream/system
    # context plus one feature row per producer — so probes on both views isolate the set
    # inductive bias, not a feature change. Rows must slice back to the flat block values.
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.orderer_scorer import (
        cycle_feature_blocks, cycle_set_features, set_features_from_blocks)

    cfg = build_scale_config(0, depth=2, products_per_layer=2, producers_per_layer=2,
                             num_consumers=1, quantity_max=2)
    env = TokenomicsAECEnv(config=cfg, horizon=4, entity_producer_obs=True,
                           product_stream_ordering=True, corrected_stream_k=True)
    env.reset(seed=0)
    blocks = cycle_feature_blocks(env)
    got = cycle_set_features(env)
    assert set(got) == {"context", "producers"}

    P = len(env.system.producing_agents)
    T = len({t for p in env.system.producing_agents.values() for t in p.max_capacities})
    G = len(env.system.process_sheets)
    assert got["producers"].shape == (P, 4 * T + 2 * G + 6 * G)
    assert got["context"].shape == (blocks["stream"].shape[0]
                                    + blocks["order_queue"].shape[0]
                                    + blocks["compact"].shape[0],)
    np.testing.assert_array_equal(
        got["context"],
        np.concatenate([blocks["stream"], blocks["order_queue"], blocks["compact"]]))
    # producer 0's row = its per-type-capacity slice ‖ inventory row ‖ inbound row ‖ its
    # relational column gathered across products (relational block is product-major)
    row0 = got["producers"][0]
    np.testing.assert_array_equal(row0[:4 * T], blocks["per_type_capacity"][:4 * T])
    np.testing.assert_array_equal(row0[4 * T:4 * T + G], blocks["inventory"][:G])
    np.testing.assert_array_equal(row0[4 * T + G:4 * T + 2 * G], blocks["inbound"][:G])
    rel = blocks["relational"].reshape(G, P, 6)
    np.testing.assert_array_equal(row0[4 * T + 2 * G:], rel[:, 0, :].reshape(-1))
    # reconstruction from serialized flat blocks (the saved A0f traces) must be identical
    rebuilt = set_features_from_blocks(blocks, n_producers=P, n_types=T, n_products=G)
    np.testing.assert_array_equal(rebuilt["producers"], got["producers"])
    np.testing.assert_array_equal(rebuilt["context"], got["context"])


def test_producer_set_value_model_is_permutation_invariant_and_count_agnostic() -> None:
    # HANDOVER invariants: reordering producer IDs must not change candidate values; adding /
    # removing producers changes only the input set, never the model shape; padded rows with
    # mask 0 are exactly absent; the candidate table size is fixed by the vocabulary.
    import torch
    from tokenomics.train.orderer_scorer import ProducerSetValueModel

    rng = np.random.default_rng(0)
    ctx = rng.normal(size=(4, 7)).astype(np.float32)
    rows = rng.normal(size=(4, 3, 5)).astype(np.float32)
    torch.manual_seed(0)
    model = ProducerSetValueModel(context_dim=7, producer_dim=5, n_choices=5)
    model.set_norm(ctx, rows)
    model.eval()
    with torch.no_grad():
        out = model.forward_tables(torch.from_numpy(ctx), torch.from_numpy(rows))
        assert out.shape == (4, 5)
        perm = model.forward_tables(torch.from_numpy(ctx),
                                    torch.from_numpy(rows[:, [2, 0, 1], :]))
        torch.testing.assert_close(out, perm)
        # same model, different producer count — no shape change
        more = model.forward_tables(torch.from_numpy(ctx),
                                    torch.from_numpy(rng.normal(size=(4, 6, 5))
                                                     .astype(np.float32)))
        assert more.shape == (4, 5)
        # a masked pad row must not alter the table
        padded = np.concatenate([rows, np.full((4, 1, 5), 99.0, dtype=np.float32)], axis=1)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]] * 4)
        masked = model.forward_tables(torch.from_numpy(ctx), torch.from_numpy(padded), mask)
        torch.testing.assert_close(out, masked)


def test_fit_set_value_table_recovers_pooled_linear_table() -> None:
    # The set probe must be able to LEARN a table that depends on the producer set through a
    # symmetric pool (here: mean over rows) plus context — the function class the HANDOVER
    # model claims. Noiseless target → train regret_frac ~0, metrics keyed like the flat probe.
    import torch
    from tokenomics.train.orderer_scorer import (
        ProducerSetValueModel, fit_set_value_table, set_value_table_metrics)

    rng = np.random.default_rng(1)
    ctx = rng.normal(size=(80, 4)).astype(np.float32)
    rows = rng.normal(size=(80, 3, 6)).astype(np.float32)
    Wp = rng.normal(size=(6, 3)).astype(np.float32)
    Wc = rng.normal(size=(4, 3)).astype(np.float32)
    Y = (rows.mean(axis=1) @ Wp + ctx @ Wc).astype(np.float32)
    torch.manual_seed(0)
    model = ProducerSetValueModel(context_dim=4, producer_dim=6, n_choices=3)
    model.set_norm(ctx, rows)
    fit_set_value_table(model, ctx, rows, Y, iters=400)
    m = set_value_table_metrics(model, ctx, rows, Y, even_idx=0)
    for k in ("top1", "winner_hit", "regret_total", "regret_frac", "headroom_total", "n"):
        assert k in m
    assert m["n"] == 80
    assert m["regret_frac"] is not None and m["regret_frac"] < 0.1


def test_aggregate_set_window_means_context_and_producers() -> None:
    # Window summary the Task-5 rows and the deployed policy both use: context =
    # concat[mean-over-window, last], producers = mean-over-window (per producer row).
    from tokenomics.train.orderer_scorer import aggregate_set_window

    feats = [
        {"context": np.array([0.0, 2.0], np.float32),
         "producers": np.array([[1.0], [3.0]], np.float32)},
        {"context": np.array([2.0, 4.0], np.float32),
         "producers": np.array([[3.0], [5.0]], np.float32)},
    ]
    out = aggregate_set_window(feats)
    # context: mean=[1,3], last=[2,4]
    np.testing.assert_allclose(out["context"], [1.0, 3.0, 2.0, 4.0])
    # producers: per-row mean over the two cycles
    np.testing.assert_allclose(out["producers"], [[2.0], [4.0]])
    assert out["producers"].shape == (2, 1)
    with pytest.raises(ValueError):
        aggregate_set_window([])


def test_blend_decision_is_conservative() -> None:
    # Default stickiness: a candidate must beat the default by MORE than the threshold to win.
    from tokenomics.train.orderer_scorer import blend_decision

    default = 2
    # nothing clears +1.0 over the default value (table[2]=0.5): stay on default
    assert blend_decision(np.array([1.0, 1.2, 0.5, 1.4, 0.0]), default, threshold=1.0) == default
    # candidate 3 clears default+1.0 (2.0 - 0.5 = 1.5 > 1.0); it's the best over-threshold one
    assert blend_decision(np.array([1.0, 1.7, 0.5, 2.0, 0.3]), default, threshold=1.0) == 3
    # exactly at threshold does NOT switch (strict >)
    assert blend_decision(np.array([0.0, 0.0, 0.0, 1.0, 0.0]), default, threshold=1.0) == default
    # ties among over-threshold candidates: first max wins
    assert blend_decision(np.array([5.0, 5.0, 0.0, 0.0, 0.0]), default, threshold=1.0) == 0
