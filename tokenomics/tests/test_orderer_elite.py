"""EXP-0008 elite-set distillation: EXP-0006B distilled a Gaussian around the *single*
argmax routing per cycle, so the learned std only captured spread *across* states. On
random-dominated cells (s6/s9) many routings tie within a cycle; collapsing them to one
arbitrary winner throws away that within-state ambiguity. EXP-0008 keeps the top-K
elite routings per state (within eps of the best) and distills over all of them, so the
masked Gaussian fits the elite *centroid + spread* instead of one chosen action
(ALPHAZERO_EXPERT_ITERATION_APPROACH.md "elite_set_distillation").

These tests pin the new pieces, kept tiny/local per the compute rule:
  1. elite selection: best-first, within-eps, capped at K, always >= 1,
  2. the search teacher returns the elite set, not just the argmax,
  3. collection emits one sample per elite action per state (multi-label augmentation),
  4. the elite path threads end to end into a schema-unchanged stochastic checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenomics.env import TokenomicsAECEnv
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.orderer_oracle import (
    _search_best_pol,
    _search_elite_pols,
    _select_elite,
)
from tokenomics.eval.scale import build_scale_config
from tokenomics.train.orderer_improve import (
    collect_orderer_samples,
    run_orderer_improve,
)


def _pol(tag):
    """A throwaway routing policy (identity tag); selection only reads the score."""
    def p(env, agent):
        return tag
    return p


def test_select_elite_keeps_best_first_within_eps_capped_at_k():
    """Elite = every routing scoring >= (1-eps)*best, best first, capped at top-K. A
    clearly-worse routing is dropped even when K leaves room; near-ties are kept."""
    scored = [
        ("a", _pol("a"), 100.0),  # best
        ("b", _pol("b"), 96.0),   # within 5% -> kept
        ("c", _pol("c"), 80.0),   # 20% worse -> dropped
        ("d", _pol("d"), 99.0),   # within 5% -> kept
    ]
    elite = _select_elite(scored, eps=0.05, k=10)

    names = [n for n, _, _ in elite]
    assert names[0] == "a"            # best first
    assert names == ["a", "d", "b"]   # sorted by units desc, c excluded


def test_search_elite_returns_sorted_nonempty_elite_and_best_matches_argmax():
    """The teacher's elite variant scores the same baselines + sampled routings as the
    argmax search, but returns the whole within-eps elite (best first, capped at K) rather
    than one winner. Its top entry is exactly the argmax search's winner, so EXP-0008 only
    *adds* near-optimal labels, never changes the chosen best."""
    cfg = build_scale_config(0, depth=2)
    env = TokenomicsAECEnv(config=cfg, horizon=10)
    env.reset(seed=0)
    prod = PRODUCING_BASELINES["stock_aware"]
    cands = {
        "proportional": ORDERING_BASELINES["proportional"],
        "even_split": ORDERING_BASELINES["even_split"],
    }
    adim = env.spec_.ordering_action_dim
    kw = dict(n_samples=4, sample_scale=1.0, lookahead=3, seed=0, cur=0, adim=adim)

    elite = _search_elite_pols(env, prod, cands, eps=0.2, k=5, **kw)
    units = [u for _, u in elite]
    assert 1 <= len(elite) <= 5             # non-empty, capped at K
    assert units == sorted(units, reverse=True)  # best first

    # the elite's top routing is the argmax search's winner (same obs/action at this state)
    best_pol, _ = _search_best_pol(env, prod, cands, **kw)
    order_agent = env._order_agents[0]
    assert np.allclose(elite[0][0](env, order_agent), best_pol(env, order_agent))


def test_elite_collection_augments_labels_not_states():
    """Elite collection (elite_k>1) emits one OrdererSample per elite action per state, so a
    tie-heavy run yields strictly more rows than the single-argmax collection — but over the
    *same* visited states (augmentation adds labels, not new states). This is the multi-label
    dataset the masked Gaussian then fits its centroid+spread to."""
    cfg = build_scale_config(0, depth=2)
    prod = PRODUCING_BASELINES["stock_aware"]
    cands = {
        "proportional": ORDERING_BASELINES["proportional"],
        "even_split": ORDERING_BASELINES["even_split"],
    }
    kw = dict(n_samples=4, sample_scale=1.0, lookahead=2, seeds=[0], horizon=8)

    base = collect_orderer_samples(cfg, prod, cands, **kw)               # elite_k=1 (default)
    elite = collect_orderer_samples(cfg, prod, cands, elite_k=6, elite_eps=0.99, **kw)

    assert len(elite) > len(base)                       # extra near-optimal labels
    base_states = {s.obs.tobytes() for s in base}
    elite_states = {s.obs.tobytes() for s in elite}
    assert base_states == elite_states                  # same states, more labels


def _synth_producer_ckpt(cfg, path: str) -> None:
    """A valid `save_policies` checkpoint with the right dims (quality irrelevant — the
    elite path exercises the *wiring*, not producer skill). Legacy v1 obs (depth-2)."""
    from tokenomics.train.smoke_ppo import GaussianActor, save_policies

    probe = TokenomicsAECEnv(config=cfg, horizon=1)
    prod = GaussianActor(probe.observation_space(probe._prod_agents[0]).shape[0],
                         probe.spec_.producing_action_dim)
    order = GaussianActor(probe.observation_space(probe._order_agents[0]).shape[0],
                          probe.spec_.ordering_action_dim)
    save_policies(prod, order, 1.0, path, config=cfg, orderer_obs_version=1)


def test_elite_run_threads_through_to_deployable_stochastic_ckpt(tmp_path):
    """End-to-end wiring: `run_orderer_improve(elite_k>1, stochastic=True)` threads the
    elite labels through DAgger into the masked Gaussian NLL fit and saves an orderer with a
    *learned* (un-clamped) std that deploys by sampling — the same checkpoint schema as the
    EXP-0006B champion, so the champion-iteration gate works unchanged. Tiny depth-2 search
    keeps it local per the compute rule."""
    from tokenomics.train.smoke_ppo import load_order_actor

    cfg = build_scale_config(0, depth=2)
    prod_ckpt = str(tmp_path / "prod.pt")
    out = str(tmp_path / "m2_elite.pt")
    _synth_producer_ckpt(cfg, prod_ckpt)

    rows = run_orderer_improve(
        depth=2, prod_ckpt=prod_ckpt, train_seeds=1, held_out_seeds=1,
        candidates=["proportional", "even_split"], n_samples=2, sample_scale=1.0,
        lookahead=1, dagger_iters=1, label_seeds=1, horizon=6, epochs=80, lr=3e-2,
        seed=0, log_std_save=-2.5, eval_seeds=1, save=out, elite_k=4, elite_eps=0.5,
        stochastic=True, log=lambda *a, **k: None,
    )

    log_std = load_order_actor(out)[0].log_std.detach().numpy()
    assert not np.allclose(log_std, -2.5)   # learned std kept (not clamped to deploy floor)
    assert rows                              # elite deploy path ran through eval


def test_run_orderer_improve_skips_internal_eval_when_eval_seeds_zero(tmp_path, monkeypatch):
    """`eval_seeds=0` trains + saves the checkpoint but skips the held-out eval, returning no
    rows — so a caller that scores the result itself (the champion-iteration driver, which
    re-evals candidate AND champion under identical conditions) does not pay for a redundant
    second candidate eval."""
    import tokenomics.train.orderer_improve as oi

    called: list[int] = []
    monkeypatch.setattr(oi, "eval_m2", lambda *a, **k: called.append(1) or [])

    cfg = build_scale_config(0, depth=2)
    prod_ckpt = str(tmp_path / "prod.pt")
    out = str(tmp_path / "m2.pt")
    _synth_producer_ckpt(cfg, prod_ckpt)

    rows = run_orderer_improve(
        depth=2, prod_ckpt=prod_ckpt, train_seeds=1, held_out_seeds=1,
        candidates=["proportional"], n_samples=1, sample_scale=1.0, lookahead=1,
        dagger_iters=1, label_seeds=1, horizon=6, epochs=40, lr=3e-2, seed=0,
        log_std_save=-2.5, eval_seeds=0, save=out, stochastic=True,
        log=lambda *a, **k: None,
    )

    assert rows == []          # no eval rows returned
    assert called == []        # the expensive held-out eval was not run
    from pathlib import Path
    assert Path(out).exists()  # checkpoint still trained + saved
