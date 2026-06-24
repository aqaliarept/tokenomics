"""EXP-0006B stochastic orderer distillation: EXP-0006A found the residual seed-9 gap
is *distillation form*, not state — s9's envelope-best is `stock_aware+random`, so a
single deterministic mean cannot reproduce the winning routing *diversity* regardless
of inputs. The fix is to distill a *stochastic* orderer (learned per-dim log_std,
sampled at deploy) instead of collapsing the teacher's per-cycle variety into one mean.

These tests pin the three behaviours that change vs the deterministic M2 clone:
  1. the fit learns log_std (masked Gaussian NLL recovers the label spread),
  2. deploy samples within the action bound (and is reproducible per seed),
  3. `--stochastic` threads end to end so the saved orderer deploys a non-trivial std.
Kept tiny (synthetic data / shrunk search) so they stay local per the compute rule.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.encode import _price
from tokenomics.eval.baselines import (
    ORDERING_BASELINES,
    PRODUCING_BASELINES,
    ordering_obs_price,
)
from tokenomics.eval.scale import build_scale_config
from tokenomics.train.orderer_improve import (
    OrdererSample,
    _fit_orderer_clone,
    _stochastic_closure,
    run_orderer_improve,
)
from tokenomics.train.smoke_ppo import GaussianActor, load_order_actor, save_policies


class _FakeEnv:
    """Minimal stand-in: the stochastic deploy `Policy` needs `env.observe(agent)` and a
    per-agent action space whose `np_random` `run_episode` seeds (so the orderer's
    sampling is reproducible per episode seed, like the `random` baseline)."""

    def __init__(self, obs: np.ndarray, act_dim: int, seed: int) -> None:
        from gymnasium.spaces import Box

        self._obs = obs
        self._space = Box(-1e3, 1e3, shape=(act_dim,), dtype=np.float32)
        self._space.seed(seed)

    def observe(self, agent: str) -> np.ndarray:
        return self._obs

    def action_space(self, agent: str):
        return self._space


def test_stochastic_fit_learns_log_std_from_label_spread():
    """Masked Gaussian NLL (stochastic=True) drives the per-dim std toward the empirical
    spread of the teacher labels on decode-active dims — the whole point of stochastic
    distillation. A decode-inactive dim (mask=0) gets no gradient, so its log_std stays
    at init. Synthetic: constant obs, dim0 active with labels ±0.5 (std 0.5), dim1 masked."""
    act_dim = 2
    actor = GaussianActor(obs_dim=2, act_dim=act_dim, log_std_init=-2.0)
    obs = np.zeros(2, dtype=np.float32)
    mask = np.array([1.0, 0.0], dtype=np.float32)  # only dim0 allocates
    labels = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
    samples = [
        OrdererSample(obs.copy(),
                      np.array([v, 7.0], dtype=np.float32),  # dim1 garbage, masked out
                      mask.copy())
        for v in labels
    ]

    _fit_orderer_clone(actor, samples, bound=1.0, epochs=2000, lr=3e-2, stochastic=True)

    std = actor.log_std.exp().detach().numpy()
    # dim0: MLE std recovers the ±0.5 label spread (mean fits ~0)
    assert std[0] == pytest.approx(0.5, abs=0.1)
    # dim1: masked → no gradient → unchanged from init exp(-2.0)
    assert std[1] == pytest.approx(np.exp(-2.0), abs=1e-4)


def test_stochastic_closure_samples_within_bound_varies_and_is_seed_reproducible():
    """Deploy = sample a~Normal(mean, std), clipped to ±bound (vs greedy clip(mean)).
    With a non-trivial std the action varies call to call (that is the diversity we are
    distilling) and never escapes the bound. Crucially the noise comes from the env's
    per-agent action-space RNG (seeded per episode by `run_episode`), NOT a closure RNG —
    so a given episode seed scores identically regardless of what else was sampled
    before (order-independent, reproducible CSV)."""
    actor = GaussianActor(obs_dim=3, act_dim=4, log_std_init=0.0)  # std = 1.0
    obs = np.ones(3, dtype=np.float32)
    bound = 0.5
    pol = _stochastic_closure(actor, bound)

    env = _FakeEnv(obs, act_dim=4, seed=7)
    a1, a2 = pol(env, "order_0"), pol(env, "order_0")
    assert not np.allclose(a1, a2)               # sampled, not the fixed mean
    assert np.all(np.abs(a1) <= bound) and np.all(np.abs(a2) <= bound)  # action bound

    # order-independence: an unrelated rollout draws from a different env first ...
    pol(_FakeEnv(obs, act_dim=4, seed=999), "order_0")
    # ... yet a freshly seed-7 env reproduces a1 exactly (RNG lives on the env, per seed)
    assert np.allclose(pol(_FakeEnv(obs, act_dim=4, seed=7), "order_0"), a1)


def _capture_obs_price(cfg, *, enriched: bool, target_cycle: int = 5):
    """Drive a deterministic white-box pair (stock_aware + proportional, both obs-version
    -agnostic) to the first orderer decision at `target_cycle`, then read `obs_price`
    there. Returns (action, prices the spec's price indices pick out, white-box cut-off
    prices). The driving policies never depend on the obs version, so a v1 and a v2 env
    reach the *same* domain state — only the obs encoding differs."""
    env = TokenomicsAECEnv(config=cfg, horizon=30, enriched_orderer_obs=enriched)
    env.reset(seed=0)
    prod, order = PRODUCING_BASELINES["stock_aware"], ORDERING_BASELINES["proportional"]
    for agent in env.agent_iter(max_iter=env.horizon * len(env.possible_agents) * 3 + 100):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        role = env._role[agent][0]
        if role == "order" and env.current_cycle >= target_cycle:
            spec = env.spec_
            obs = np.asarray(env.observe(agent), dtype=np.float32)
            wb = np.array([_price(spec, env.system.producer_infos[pid].cut_off_price)
                           for pid in spec.producer_ids], dtype=np.float32)
            return ordering_obs_price(env, agent), obs[spec.ordering_price_indices()], wb
        env.step(prod(env, agent) if role == "prod" else order(env, agent))
    raise AssertionError("no orderer decision reached")


def test_obs_price_reads_cutoff_prices_under_enriched_v2_and_matches_v1():
    """[review P1] obs_price is a search-teacher candidate; it routes by the observed
    producer cut-off price. Under enriched v2 obs the price block moved (strided in the
    per-producer block, not the v1 contiguous slot), so a hardcoded v1 offset reads
    garbage and silently changes the candidate. The spec's version-aware price indices
    must (a) actually point at the cut-off prices in v2, and (b) make obs_price emit the
    SAME action in v1 and v2 — holding the candidate set fixed across the EXP-0006A/B A/B."""
    cfg = build_scale_config(0, depth=3)

    act_v2, picked_v2, wb_v2 = _capture_obs_price(cfg, enriched=True)
    assert wb_v2.sum() > 0  # meaningful: prices are non-zero/varied at this cycle
    assert np.allclose(picked_v2, wb_v2)  # (a) v2 indices read the real cut-off prices

    act_v1, _, _ = _capture_obs_price(cfg, enriched=False)
    assert np.allclose(act_v1, act_v2)  # (b) identical candidate action across versions


def _synth_producer_ckpt(cfg, path: str) -> None:
    """A valid `save_policies` checkpoint with the right dims (quality irrelevant — this
    test exercises the stochastic *wiring*, with the enriched v2 orderer obs)."""
    probe = TokenomicsAECEnv(config=cfg, horizon=1, enriched_orderer_obs=True)
    prod = GaussianActor(probe.observation_space(probe._prod_agents[0]).shape[0],
                         probe.spec_.producing_action_dim)
    order = GaussianActor(probe.observation_space(probe._order_agents[0]).shape[0],
                          probe.spec_.ordering_action_dim)
    save_policies(prod, order, 1.0, path, config=cfg, orderer_obs_version=2)


def test_stochastic_run_deploys_unclamped_learned_std(tmp_path):
    """End-to-end wiring: `run_orderer_improve(stochastic=True)` distills via the NLL fit
    and, unlike the deterministic M2 path, does NOT clamp the orderer `log_std` to the
    deploy floor at save — so the saved orderer carries a *learned* std that deploys by
    sampling. Tiny depth-2 search keeps it local."""
    cfg = build_scale_config(0, depth=2)
    prod_ckpt = str(tmp_path / "prod.pt")
    out = str(tmp_path / "m2_stoch.pt")
    _synth_producer_ckpt(cfg, prod_ckpt)

    rows = run_orderer_improve(
        depth=2, prod_ckpt=prod_ckpt, train_seeds=1, held_out_seeds=1,
        candidates=["proportional"], n_samples=1, sample_scale=1.0, lookahead=1,
        dagger_iters=1, label_seeds=1, horizon=6, epochs=80, lr=3e-2, seed=0,
        log_std_save=-2.5, eval_seeds=1, save=out,
        enriched_orderer_obs=True, stochastic=True, log=lambda *a, **k: None,
    )

    log_std = load_order_actor(out)[0].log_std.detach().numpy()
    # clamp skipped: not collapsed onto the deterministic deploy floor (-2.5)
    assert not np.allclose(log_std, -2.5)
    assert rows  # stochastic deploy path ran through eval without error
