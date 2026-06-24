"""Champion regression gate (EXP-0006B). Pins the *shipped* orderer champion
(`m2_v3_obs_stochastic.pt` = v4 producer + enriched v2 obs + learned stochastic orderer)
so a later retrain/refactor cannot silently swap in a weaker or non-stochastic policy.

Distinct from `test_orderer_stochastic.py`, which tests the distillation *mechanism* on
synthetic data — these tests load the actual artifact and assert its provenance + the
empirical bar it was accepted on (RESULT.md: beats the strict envelope on all held-out
seeds 6,7,8,9, zero death; the 40-eval-seed audit clears even at the lower 95% CI bound).

Fast structural + P2 gates run in the default suite. The full envelope gate is a ~13-min
eval marked `champion` (deselected by default via pyproject `addopts`; opt in with
`pytest -m champion`). All skip cleanly if the LFS artifact is not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from tokenomics.eval.runner import evaluate
from tokenomics.eval.scale import build_scale_config
from tokenomics.train.orderer_improve import _stochastic_closure, eval_m2
from tokenomics.train.smoke_ppo import load_order_actor, load_policies

# repo_root/experiments/...  — tests/ -> project(tokenomics) -> repo_root
CHAMPION = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "EXP-0006B-orderer-stochastic"
    / "artifacts"
    / "m2_v3_obs_stochastic.pt"
)
requires_champion = pytest.mark.skipif(
    not CHAMPION.exists(), reason="champion artifact absent (git-lfs not pulled)"
)

# Provenance pinned in EXP-0006B/spec.json + RESULT.md.
_CHECKPOINT_VERSION = 6
_V2_FINGERPRINT = "548120bd72921234"  # enriched-v2 obs fingerprint (seed-0 economy)
_MASKED_SIGMA = np.exp(-2.0)  # log_std init for decode-inactive orderer dims


@requires_champion
def test_champion_provenance_is_the_enriched_v2_stochastic_orderer():
    """The shipped checkpoint is the exact accepted artifact: current schema version and
    the enriched-v2 obs fingerprint (a v1 model on the same economy fingerprints
    differently, ADR-0051) — so an eval cannot mistake a legacy or differently-trained
    orderer for the champion."""
    ck = torch.load(CHAMPION, weights_only=True)
    assert ck["version"] == _CHECKPOINT_VERSION
    assert ck["config_fingerprint"] == _V2_FINGERPRINT
    assert ck["order_act"] == 180  # ordering action dim for the depth-3 scale economy


@requires_champion
def test_champion_orderer_head_is_learned_stochastic_not_collapsed():
    """The accepted lever is routing *diversity*: the orderer must deploy a learned
    per-dim std, not a collapsed deterministic mean. The masked Gaussian NLL left the
    decode-inactive dims at init (σ≈0.135) and grew the decode-active routing dims —
    64 dims rose above init, peaking at σ≈0.6 (RESULT.md). A deterministic clone (all
    dims clamped to the deploy floor) would fail every clause."""
    sigma = np.exp(torch.load(CHAMPION, weights_only=True)["order"]["log_std"].numpy())
    grew = int((sigma > _MASKED_SIGMA + 1e-3).sum())
    assert grew == 64  # decode-active routing dims that learned a wider spread
    assert sigma.max() == pytest.approx(0.605, abs=0.02)  # the learned diversity, not noise
    # masking held: most dims stayed pinned at the init floor
    assert int(np.isclose(sigma, _MASKED_SIGMA, atol=1e-3).sum()) >= 100


def _champion_deploy():
    """The champion as deployed: greedy v4 producer + the orderer *sampled* from its
    learned log_std (not clip(mean)) — exactly how `eval_m2(stochastic=True)` scores it."""
    prod, _ = load_policies(CHAMPION)
    actor, bound = load_order_actor(CHAMPION)
    return prod, _stochastic_closure(actor, bound)


@requires_champion
def test_champion_eval_is_order_independent_per_topology():
    """P2 at the artifact level: the stochastic orderer draws its noise from the env's
    per-episode action-space RNG, so a held-out topology must score *identically* whether
    evaluated alone or after another topology in the same process — i.e. the per-seed CSV
    is reproducible and order-free (a global-RNG regression would break this)."""
    prod, order = _champion_deploy()
    cfg9, cfg6 = build_scale_config(9, depth=3), build_scale_config(6, depth=3)

    r1 = evaluate(prod, order, seeds=range(2), horizon=30, config=cfg9, enriched_orderer_obs=True)
    evaluate(prod, order, seeds=range(2), horizon=30, config=cfg6, enriched_orderer_obs=True)
    r2 = evaluate(prod, order, seeds=range(2), horizon=30, config=cfg9, enriched_orderer_obs=True)

    assert r1["consumer_completed_units"] == r2["consumer_completed_units"]


# Held-out topologies + the 40-eval-seed audit bar (EXP-0006B RESULT.md confidence audit).
_HELD_OUT = [6, 7, 8, 9]
_AUDIT_SEEDS = 40


@pytest.mark.champion
@requires_champion
def test_champion_beats_strict_envelope_at_lower_ci_bound():
    """The acceptance bar, as a gate: scored over `_AUDIT_SEEDS` stochastic eval seeds the
    champion must beat the strict random-inclusive envelope on *every* held-out topology
    at its lower 95% CI bound (not merely on the mean — the stronger claim the audit
    established), with zero death. Slow (~13 min full eval) → `pytest -m champion`."""
    rows = eval_m2(
        str(CHAMPION), depth=3, seeds=_HELD_OUT, eval_seeds=_AUDIT_SEEDS, horizon=100,
        enriched_orderer_obs=True, stochastic=True, log=lambda *a, **k: None,
    )
    assert len(rows) == len(_HELD_OUT)
    for r in rows:
        assert r["m2_dead"] == 0.0, f"seed {r['seed']}: death {r['m2_dead']}"
        # clears the strict envelope even at the lower 95% CI bound (audit-grade)
        assert r["m2_cu"] - r["m2_ci"] > r["env_cu"], (
            f"seed {r['seed']}: {r['m2_cu']}-{r['m2_ci']} !> env {r['env_cu']}"
        )
