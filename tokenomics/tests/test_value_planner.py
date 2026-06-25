"""EXP-0009 value-guided planner: the orderer search teacher scores every candidate
routing by a full `lookahead`-cycle deepcopy rollout (`orderer_oracle._score_candidates`),
which dominates label-gen compute. EXP-0009 replaces the rollout tail with a learned value
estimate: roll only `short` cycles under the candidate, then add `V(global_state)` for the
remaining window. The accept criterion is the planner matching the rollout-only teacher's
consumer units at fewer simulated cycles (ALPHAZERO_EXPERT_ITERATION_APPROACH.md EXP-0009).

The value model reads the existing versioned system state (`env.global_state()`, the
centralized-critic input) so no new state contract is introduced. These tests pin the new
pieces as vertical slices, kept tiny/local per the compute rule.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def test_value_model_fit_tracks_targets():
    """A `ValueModel` fit by MSE on (global_state, windowed-units) pairs learns to predict
    the target ordering: after fitting on a tiny synthetic set, a high-target state scores
    above a low-target state. This is the regressor the planner's terminal value relies on."""
    from tokenomics.train.value_model import ValueModel, ValueSample, fit_value_model

    rng = np.random.default_rng(0)
    state_dim = 6
    # synthetic linear target: units = sum(state) so the model has a learnable signal
    states = rng.standard_normal((64, state_dim)).astype(np.float32)
    samples = [ValueSample(s, float(s.sum())) for s in states]

    model = ValueModel(state_dim)
    fit_value_model(model, samples, epochs=400, lr=1e-2)

    hi = np.full(state_dim, 1.0, dtype=np.float32)
    lo = np.full(state_dim, -1.0, dtype=np.float32)
    import torch

    with torch.no_grad():
        v_hi = float(model(torch.as_tensor(hi)))
        v_lo = float(model(torch.as_tensor(lo)))
    assert v_hi > v_lo


def _cfg_prod_order():
    """Tiny depth-2 topology + fixed (stock_aware, proportional) policies for collection /
    planner tests — small enough to run the search locally per the compute rule."""
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(0, depth=2)
    return cfg, PRODUCING_BASELINES["stock_aware"], ORDERING_BASELINES["proportional"]


def test_collect_value_samples_targets_are_windowed_units():
    """Each collected sample's target is the consumer units delivered over the next `window`
    cycles from that state. Verified without reimplementing the sim: a window spanning the
    whole episode makes the first sample's target the full-episode consumer units (matched
    against an independent episode run), and the small-window targets telescope out of the
    full-window cumulative."""
    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.eval.orderer_oracle import _baseline_units
    from tokenomics.train.value_model import collect_value_samples

    cfg, prod, order = _cfg_prod_order()
    horizon = 12
    window = 3

    full = collect_value_samples(cfg, prod, order, window=horizon, seeds=[0], horizon=horizon)
    small = collect_value_samples(cfg, prod, order, window=window, seeds=[0], horizon=horizon)

    # independent oracle: the same episode's total consumer units
    total = _baseline_units(TokenomicsAECEnv(config=cfg, horizon=horizon), prod, order, 0)[
        "consumer_completed_units"
    ]
    assert full[0].target == pytest.approx(total)          # window covers the whole episode
    assert len(small) == len(full)                          # one sample per cycle, same rollout

    # small-window target[p] = cum[p+W] - cum[p]; with cum[p] = total - full[p].target this is
    # full[p].target - full[p+W].target (full[].target beyond the episode end is 0).
    for p in range(len(small)):
        nxt = full[p + window].target if p + window < len(full) else 0.0
        assert small[p].target == pytest.approx(full[p].target - nxt)


def test_valued_scoring_reduces_to_rollout_and_composes_terminal_value():
    """Value-guided candidate scoring (`_score_candidates_valued`) generalizes the existing
    full-rollout scorer: with `window == lookahead` and no value it reproduces
    `_score_candidates` exactly (same candidates, same units). A terminal `value_fn` adds its
    estimate onto the short-rollout units — a constant value shifts every candidate's score by
    that constant (so it never changes the short-rollout ranking, the value model carries the
    state-dependent signal)."""
    from tokenomics.eval.orderer_oracle import _score_candidates, _score_candidates_valued

    cfg, prod, order = _cfg_prod_order()
    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.eval.baselines import ORDERING_BASELINES

    env = TokenomicsAECEnv(config=cfg, horizon=10)
    env.reset(seed=0)
    cands = {"proportional": ORDERING_BASELINES["proportional"],
             "even_split": ORDERING_BASELINES["even_split"]}
    adim = env.spec_.ordering_action_dim
    kw = dict(n_samples=3, sample_scale=1.0, seed=0, cur=0, adim=adim)

    rollout = _score_candidates(env, prod, cands, lookahead=4, **kw)
    valued_none, cyc = _score_candidates_valued(env, prod, cands, window=4, value_fn=None, **kw)
    assert [n for n, _, _ in valued_none] == [n for n, _, _ in rollout]      # same candidate set
    for (_, _, sv), (_, _, sr) in zip(valued_none, rollout):
        assert sv == pytest.approx(sr)                                       # reduces to rollout
    assert cyc > 0                                                           # simulated some cycles

    const = 7.0
    valued_c, _ = _score_candidates_valued(
        env, prod, cands, window=2, value_fn=lambda s: const, **kw)
    base_c, _ = _score_candidates_valued(env, prod, cands, window=2, value_fn=None, **kw)
    for (_, _, sc), (_, _, sb) in zip(valued_c, base_c):
        assert sc == pytest.approx(sb + const)                              # additive terminal value


def test_value_tail_is_zeroed_at_terminal_states():
    """The value model predicts units-to-go from a cycle start, so a terminal state (system
    death or horizon reached) must score a 0 tail — no future cycle can deliver units. A
    rollout that runs all the way to the episode end must therefore ignore `value_fn`, while a
    rollout that stops mid-episode (non-terminal) still adds the terminal estimate. Without
    this, terminal candidates near the end of an episode get arbitrary extra score and corrupt
    the ranking vs. the full rollout."""
    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.eval.orderer_oracle import _rollout_window

    cfg, prod, order = _cfg_prod_order()
    horizon = 8
    const = 100.0

    # window spans the whole episode → rollout ends terminal → tail must be 0
    env = TokenomicsAECEnv(config=cfg, horizon=horizon)
    env.reset(seed=0)
    term_score, term_cycles = _rollout_window(env, prod, order, horizon + 5, value_fn=lambda s: const)
    env2 = TokenomicsAECEnv(config=cfg, horizon=horizon)
    env2.reset(seed=0)
    plain_score, _ = _rollout_window(env2, prod, order, horizon + 5, value_fn=None)
    assert term_cycles < horizon + 5                     # episode ended before the window
    assert term_score == pytest.approx(plain_score)      # terminal tail dropped

    # short window stops mid-episode (non-terminal) → tail is added
    env3 = TokenomicsAECEnv(config=cfg, horizon=horizon)
    env3.reset(seed=0)
    mid_valued, _ = _rollout_window(env3, prod, order, 2, value_fn=lambda s: const)
    env4 = TokenomicsAECEnv(config=cfg, horizon=horizon)
    env4.reset(seed=0)
    mid_plain, _ = _rollout_window(env4, prod, order, 2, value_fn=None)
    assert mid_valued == pytest.approx(mid_plain + const)


def test_planner_episode_rollout_mode_matches_search_and_value_mode_is_cheaper():
    """`planner_orderer_episode` is one driver in two modes. In rollout-only mode
    (`window == lookahead`, no value) it reproduces the legacy `search_orderer_episode`
    consumer units exactly — same candidates, same per-cycle argmax. In value-guided mode a
    shorter window + terminal value commits a full episode while simulating strictly fewer
    rollout cycles (the EXP-0009 compute lever), and still returns valid episode stats."""
    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.eval.baselines import ORDERING_BASELINES
    from tokenomics.eval.orderer_oracle import (
        planner_orderer_episode,
        search_orderer_episode,
    )

    cfg, prod, _ = _cfg_prod_order()
    cands = {"proportional": ORDERING_BASELINES["proportional"],
             "even_split": ORDERING_BASELINES["even_split"]}
    kw = dict(n_samples=4, sample_scale=1.0, seed=0)
    horizon = 12

    legacy_stats, _ = search_orderer_episode(
        TokenomicsAECEnv(config=cfg, horizon=horizon), prod, cands, lookahead=6, **kw)
    roll_stats, roll_cycles = planner_orderer_episode(
        TokenomicsAECEnv(config=cfg, horizon=horizon), prod, cands, window=6, value_fn=None, **kw)
    assert roll_stats["consumer_completed_units"] == pytest.approx(
        legacy_stats["consumer_completed_units"])                       # generalizes legacy driver

    val_stats, val_cycles = planner_orderer_episode(
        TokenomicsAECEnv(config=cfg, horizon=horizon), prod, cands,
        window=2, value_fn=lambda s: 0.0, **kw)
    assert val_cycles < roll_cycles                                     # fewer simulated cycles
    assert "consumer_completed_units" in val_stats                      # valid episode


def test_ranking_agreement_is_one_when_scorings_coincide():
    """`ranking_agreement_episode` reports the per-cycle fraction where the value-guided argmax
    candidate matches the full-rollout argmax — the offline ranking-quality diagnostic. When
    the value-guided scoring coincides with the full rollout (window == lookahead, zero terminal
    value: the trivially exact value), it must report perfect agreement; in general it is a
    fraction in [0, 1]."""
    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.eval.baselines import ORDERING_BASELINES
    from tokenomics.eval.orderer_oracle import ranking_agreement_episode

    cfg, prod, _ = _cfg_prod_order()
    cands = {"proportional": ORDERING_BASELINES["proportional"],
             "even_split": ORDERING_BASELINES["even_split"]}
    kw = dict(n_samples=4, sample_scale=1.0, seed=0)

    exact = ranking_agreement_episode(
        TokenomicsAECEnv(config=cfg, horizon=12), prod, cands,
        lookahead=6, short=6, value_fn=lambda s: 0.0, **kw)
    assert exact == pytest.approx(1.0)

    approx = ranking_agreement_episode(
        TokenomicsAECEnv(config=cfg, horizon=12), prod, cands,
        lookahead=6, short=2, value_fn=lambda s: 0.0, **kw)
    assert 0.0 <= approx <= 1.0


def test_value_planner_eval_runs_end_to_end_with_compute_savings():
    """The accept harness wires the whole loop: collect value targets, fit the model, then
    score the value-guided planner against the rollout-only teacher on held-out topologies,
    reporting per-seed consumer units, simulated cycles, and ranking agreement. The EXP-0009
    accept verdict (units match at fewer cycles) is read off these rows on the real run; the
    local invariant pinned here is that the value-guided planner commits a valid episode while
    simulating strictly fewer cycles than the full rollout."""
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.train.value_model import run_value_planner_eval

    rows = run_value_planner_eval(
        depth=2, prod=PRODUCING_BASELINES["stock_aware"],
        order=ORDERING_BASELINES["proportional"],
        candidates=["proportional", "even_split"], n_samples=3, sample_scale=1.0,
        lookahead=4, short=1, train_seeds=[0], held_out_seeds=[1], label_seeds=1,
        horizon=10, epochs=60, lr=1e-2, seed=0, log=lambda *a, **k: None)

    assert len(rows) == 1
    for r in rows:
        assert r["value_cycles"] < r["rollout_cycles"]      # compute lever holds
        assert r["rollout_units"] >= 0 and r["value_units"] >= 0
        assert 0.0 <= r["agreement"] <= 1.0
