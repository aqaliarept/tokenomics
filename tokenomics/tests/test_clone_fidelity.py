"""Clone-fidelity diagnostics (ADR-0049 Phase 1 follow-up).

The warm-start gate fails (clone+base 82 << 290 bar) although the obs_stock_aware
expert itself clears it — a clone-FIDELITY problem. Raw MSE is misleading: the
producer decode is gated (replenish fires only for raw>0, a toggle only for raw>0,
masked dims are no-ops, decode.py), so a low-MSE clone can still decode the wrong
command. These diagnostics score the clone on DECODED behavior, the thing the gate
actually exercises."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def test_decoded_command_match_reads_commands_not_raw_vectors():
    # Tracer: collect real producer transitions under the expert, then a "clone" that
    # reproduces the expert action verbatim must decode-match every sample (1.0). A
    # clone that flips the restoration-toggle sign decodes a different command on the
    # rows where the expert restored, so the score drops below 1.0 — proving the metric
    # reads decode_producing output, not the raw action distance.
    from tokenomics.env.spaces import build_spec
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.clone_fidelity import (
        collect_producer_samples,
        decoded_command_match,
    )

    cfg = build_scale_config(0, depth=2)
    spec = build_spec(cfg)
    expert = PRODUCING_BASELINES["obs_stock_aware"]
    order = ORDERING_BASELINES["proportional"]

    samples = collect_producer_samples(cfg, expert, order, seeds=range(1), horizon=20)
    assert samples
    assert all(s.expert_action.shape == (spec.producing_action_dim,) for s in samples)

    expert_act = np.stack([s.expert_action for s in samples])
    assert decoded_command_match(spec, samples, expert_act) == 1.0

    g = spec.num_inputs
    flipped = expert_act.copy()
    flipped[:, g] = -flipped[:, g] - 1.0  # invert restoration toggle sign on every row
    assert decoded_command_match(spec, samples, flipped) < 1.0


def _depth2_samples(horizon=40):
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.clone_fidelity import collect_producer_samples

    cfg = build_scale_config(0, depth=2)
    samples = collect_producer_samples(
        cfg, PRODUCING_BASELINES["obs_stock_aware"], ORDERING_BASELINES["proportional"],
        seeds=range(1), horizon=horizon,
    )
    return cfg, samples


def test_fidelity_report_perfect_clone_is_clean_and_grouped_by_layer():
    # A clone that emits the expert action verbatim is perfect on every facet: zero
    # decoded mismatch, no toggle errors, exact replen quantities, ~0 MSE. The report
    # is also grouped by the supply-chain layers actually present (depth-2 -> {0, 1}).
    from tokenomics.env.spaces import build_spec
    from tokenomics.train.clone_fidelity import clone_fidelity_report

    cfg, samples = _depth2_samples()
    spec = build_spec(cfg)
    expert_act = np.stack([s.expert_action for s in samples])

    rep = clone_fidelity_report(spec, samples, expert_act)

    assert rep.overall.decoded_mismatch == 0.0
    assert rep.overall.restore_fp == 0 and rep.overall.restore_fn == 0
    assert rep.overall.replen_exact == 1.0
    assert rep.overall.raw_mse == pytest.approx(0.0)
    assert rep.overall.active_mse == pytest.approx(0.0)
    assert rep.overall.raw_mse_by_dim.shape == (spec.producing_action_dim,)
    assert set(rep.by_layer) == {s.layer for s in samples}


def test_fidelity_report_flags_dropped_restoration():
    # A clone that never restores (toggle pinned off) turns every expert restoration
    # into a false negative and raises decoded mismatch. Pins that the report scores
    # the GATED command, counting exactly the expert's decoded restorations.
    from tokenomics.env.decode import decode_producing
    from tokenomics.env.spaces import ACTION_LOGIT_BOUND, build_spec
    from tokenomics.train.clone_fidelity import clone_fidelity_report

    cfg, samples = _depth2_samples()
    spec = build_spec(cfg)
    g = spec.num_inputs
    expert_restores = sum(
        decode_producing(spec, s.expert_action, s.view).do_restoration for s in samples
    )
    assert expert_restores > 0  # precondition: the expert restores in this rollout

    never = np.stack([s.expert_action for s in samples])
    never[:, g] = -ACTION_LOGIT_BOUND  # restoration toggle pinned off

    rep = clone_fidelity_report(spec, samples, never)
    assert rep.overall.restore_tp == 0
    assert rep.overall.restore_fn == expert_restores
    assert rep.overall.decoded_mismatch > 0.0


def test_producer_clone_overfits_decoded_command_on_tiny_set():
    # Step 2 (ADR-0049): on ONE depth-3 config / one seed, the decode-aware producer
    # fit must reach near-perfect DECODED command match. This isolates loss/labeling
    # from data coverage — plain full-vector MSE reaches only ~0.15 here (restoration
    # false positives at decode's >0 boundary), so a failure means the training
    # procedure cannot fit, not that the dataset is too narrow.
    from tokenomics.env.spaces import build_spec
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.clone_fidelity import (
        collect_producer_samples,
        decoded_command_match,
        fit_producer_clone,
        predict_actions,
    )
    from tokenomics.train.smoke_ppo import (
        DEFAULT_ORDER_LOG_STD,
        DEFAULT_PROD_LOG_STD,
        DEFAULT_PROD_REPLEN_BIAS,
        DEFAULT_PROD_REPLEN_LOG_STD,
        DEFAULT_PROD_RESTORE_BIAS,
        _build_clone_actors,
    )

    cfg = build_scale_config(0, depth=3)
    spec = build_spec(cfg)
    samples = collect_producer_samples(
        cfg, PRODUCING_BASELINES["obs_stock_aware"],
        ORDERING_BASELINES["proportional"], seeds=range(1), horizon=40,
    )

    import torch
    torch.manual_seed(0)
    prod, _order, bound = _build_clone_actors(
        cfg, prod_log_std=DEFAULT_PROD_LOG_STD,
        prod_replen_log_std=DEFAULT_PROD_REPLEN_LOG_STD,
        prod_replen_bias=DEFAULT_PROD_REPLEN_BIAS,
        prod_restore_bias=DEFAULT_PROD_RESTORE_BIAS, log_std_init=DEFAULT_ORDER_LOG_STD,
    )
    fit_producer_clone(prod, samples, spec, bound, epochs=3000, lr=3e-3)
    pred = predict_actions(prod, samples, bound)

    assert decoded_command_match(spec, samples, pred) >= 0.9

    # Freeze the exact regressions the decode-aware loss fixed (plain MSE: restore
    # fp=267, replen_exact=0.86, decoded match 0.15): toggle sign is classified, not
    # regressed, and replen quantities round to the expert's.
    from tokenomics.train.clone_fidelity import clone_fidelity_report

    rep = clone_fidelity_report(spec, samples, pred)
    assert rep.overall.restore_fp == 0
    assert rep.overall.restore_fn == 0
    assert rep.overall.replen_exact == 1.0


def test_dagger_producer_first_saves_high_fidelity_loadable_checkpoint(tmp_path):
    # Step 3/4 wiring: the producer-first DAgger driver clones with the decode-aware
    # loss, relabels learner-visited states under the BASE orderer (deployment
    # condition), and saves a PPO-loadable checkpoint whose producer decodes the
    # expert command on most held states. Depth-2 / small so it is a wiring smoke.
    from tokenomics.env.spaces import build_spec
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.clone_fidelity import (
        collect_producer_samples,
        dagger_producer_first,
        decoded_command_match,
        load_producer_actor,
        predict_actions,
    )
    from tokenomics.train.smoke_ppo import load_policies

    cfg = build_scale_config(0, depth=2)
    spec = build_spec(cfg)
    save = tmp_path / "pf.pt"
    dagger_producer_first(
        cfg, PRODUCING_BASELINES["obs_stock_aware"],
        ORDERING_BASELINES["proportional"], ORDERING_BASELINES["proportional"],
        dagger_iters=2, seeds=range(1), horizon=20, epochs=400, lr=3e-3,
        save=str(save), log=lambda *a, **k: None,
    )

    assert save.exists()
    prod_policy, order_policy = load_policies(str(save))  # PPO-compatible schema
    assert callable(prod_policy) and callable(order_policy)

    prod, bound = load_producer_actor(str(save))
    samples = collect_producer_samples(
        cfg, PRODUCING_BASELINES["obs_stock_aware"],
        ORDERING_BASELINES["proportional"], seeds=range(1), horizon=20,
    )
    pred = predict_actions(prod, samples, bound)
    assert decoded_command_match(spec, samples, pred) >= 0.7


def test_producer_clone_fit_freezes_masked_action_dims():
    # Freeze active-mask semantics: a producing-action dim that decode no-ops (mask 0
    # on every row) must receive NO clone gradient, even given a strong label — its
    # output-layer params stay frozen, while an active dim's params move. This is what
    # keeps the clone from fitting dims that cannot change the decoded command.
    import numpy as np
    import torch

    from tokenomics.train.smoke_ppo import GaussianActor, _fit_producer_clone

    g = 2  # 2 replenish dims + 3 toggles
    adim = g + 3
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((16, 5)).astype(np.float32)
    act = np.zeros((16, adim), dtype=np.float32)
    act[:, 0] = 5.0          # active replen dim 0 -> a real target
    act[:, g] = 8.0          # restoration toggle -> strong positive label ...
    masks = np.ones((16, adim), dtype=np.float32)
    masks[:, g] = 0.0        # ... but masked on every row (e.g. restoration running)

    torch.manual_seed(0)
    actor = GaussianActor(obs_dim=5, act_dim=adim)
    head = actor.actor[-1]
    w_g0, b_g = head.weight[g].detach().clone(), head.bias[g].detach().clone()
    w_a0, b_a = head.weight[0].detach().clone(), head.bias[0].detach().clone()

    _fit_producer_clone(actor, obs, act, masks, np.ones(16, dtype=np.float32),
                        g=g, bound=10.0, epochs=200, lr=1e-2)

    # masked toggle dim: its head row + bias get zero gradient -> unchanged
    assert torch.equal(head.weight[g].detach(), w_g0)
    assert torch.equal(head.bias[g].detach(), b_g)
    # active replen dim: its params moved
    assert not torch.equal(head.weight[0].detach(), w_a0) or not torch.equal(
        head.bias[0].detach(), b_a
    )


def test_dagger_producer_first_multi_topology_fits_every_config(tmp_path):
    # Phase 2 (ADR-0049): multi-topology sampling. One producer net trained across
    # several same-shape scale topologies (seeds vary mechanics) must decode the
    # expert command on EACH, not just one — the start against operating-point /
    # slot memorization.
    from tokenomics.env.spaces import build_spec
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.clone_fidelity import (
        collect_producer_samples,
        dagger_producer_first,
        decoded_command_match,
        load_producer_actor,
        predict_actions,
    )

    cfgs = [build_scale_config(s, depth=2) for s in (0, 1)]
    save = tmp_path / "multi.pt"
    dagger_producer_first(
        cfgs, PRODUCING_BASELINES["obs_stock_aware"],
        ORDERING_BASELINES["proportional"], ORDERING_BASELINES["proportional"],
        dagger_iters=2, seeds=range(1), horizon=20, epochs=1200, lr=3e-3,
        save=str(save), log=lambda *a, **k: None,
    )

    prod, bound = load_producer_actor(str(save))
    for cfg in cfgs:
        spec = build_spec(cfg)
        samples = collect_producer_samples(
            cfg, PRODUCING_BASELINES["obs_stock_aware"],
            ORDERING_BASELINES["proportional"], seeds=range(1), horizon=20,
        )
        pred = predict_actions(prod, samples, bound)
        assert decoded_command_match(spec, samples, pred) >= 0.7


def test_held_out_report_scores_unseen_topologies():
    # Phase 2 step 6: a held-out topology suite normalizes each unseen topology by its
    # OWN base+base, so the ratio (not absolute units) is the fair judgment. Scoring the
    # base producer AS the clone must give ratio ≈ 1.0 (the scripted domain is
    # deterministic; clone == base here).
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.train.warm_start import held_out_report

    base_prod = PRODUCING_BASELINES["stock_aware"]
    rows = held_out_report(
        base_prod, base_prod, ORDERING_BASELINES["proportional"],
        depth=2, seeds=[5, 6], eval_seeds=2, horizon=40, log=lambda *a, **k: None,
    )
    assert [r.seed for r in rows] == [5, 6]
    assert all(r.units > 0 and r.base_units > 0 for r in rows)
    assert all(r.ratio == pytest.approx(1.0) for r in rows)  # clone == base
    assert all(0.0 <= r.dead <= 1.0 for r in rows)
