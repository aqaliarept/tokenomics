"""add-m6-joint-mpg-gnn 2.2b / 2.3: the team potential Φ and the MAPPO learner (L0 code checks).

- **Potential:** the tracker reads only (the env's own rewards and stats are unchanged) and its unit
  term equals the episode's consumer units; ν charges a death once.
- **Credit:** GAE over cycles is the textbook recursion; the recorded old log-probability equals the
  recomputed one at the same parameters (PPO's ratio starts at 1).
- **Update:** a tiny dry train runs end to end, writes its checkpoint and curve, resumes, and the
  synchronous update is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenomics.env.potential import PotentialTracker, PotentialWeights  # noqa: E402
from tokenomics.train import m6_imitation as im  # noqa: E402
from tokenomics.train import m6_mappo as mp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "experiments/EXP-0058-es-gradient-probe/bar"
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
MODELS = REPO / "experiments/EXP-0100-m6-imitation-v2/artifacts/fit/models.pt"
W = PotentialWeights(lam=2.0, mu=3.0, nu=100.0, x=0.75, y=0.73)


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs = pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(MODELS)
                                and (TRAINING / "training_worlds.json").exists()),
                           reason="records absent")


def _job(horizon: int = 8) -> dict:
    ts, train, _mon = im._training(str(TRAINING))
    return im._jobs(ts, train[:1], "x", (0,), horizon)[0]


@needs
def test_potential_tracker_reads_only_and_counts_the_units() -> None:
    from tokenomics.eval.baselines import producing_policy
    from tokenomics.eval.routing_audit import ordering_policy
    from tokenomics.eval.runner import run_episode

    def play(track: bool) -> tuple[dict, PotentialTracker | None]:
        env = im._episode_env(_job())
        tr = None
        if track:
            tr = PotentialTracker(W)
            tr.attach(env)
        run_episode(env, producing_policy(im.BASE_PRODUCER), ordering_policy(im.BASE_ORDERER),
                    seed=_job()["episode"], action_seed=None)
        return dict(env.episode_stats), tr

    plain, _ = play(False)
    tracked, tr = play(True)
    assert plain["consumer_completed_units"] == tracked["consumer_completed_units"]
    assert plain["return"] == tracked["return"]  # the env's reward is untouched
    assert len(tr.terms) == tracked["cycles"]
    assert tr.totals()["units"] == tracked["consumer_completed_units"]
    assert all(t.rundown >= 0 and t.starving >= 0 for t in tr.terms)
    t0 = tr.terms[0]
    assert tr.rewards()[0] == pytest.approx(t0.units - W.lam * t0.rundown - W.mu * t0.starving
                                            - W.nu * t0.death)


def test_unit_scaled_reward_and_calibration_constants() -> None:
    """The normalised reward: s · units − λ rundown − μ starving − ν death per cycle; calibrate's
    B_we = the base's mean episode units per world-episode, λ / μ = σ(normalised units) / σ(term),
    pooled within world-episode, ν = H; a run's unit scale = H / B_we."""
    from tokenomics.env.potential import PotentialTerms

    t = PotentialTerms(units=30.0, rundown=0.5, starving=2, death=1)
    assert t.phi(W, unit_scale=0.04) == pytest.approx(30 * 0.04 - 2.0 * 0.5 - 3.0 * 2 - 100.0)
    assert t.phi(W) == pytest.approx(30 - 1.0 - 6 - 100.0)
    rows = [{"world": "a", "episode": 1, "units": u, "rundown": r, "starving": s}
            for u, r, s in ((100, 1.0, 2), (120, 3.0, 4))]
    rows += [{"world": "b", "episode": 2, "units": u, "rundown": r, "starving": s}
             for u, r, s in ((200, 2.0, 1), (200, 2.0, 3))]
    c = mp.calibrate_weights(rows, horizon=10)
    assert c["bar"] == {"a|1": 110.0, "b|2": 200.0} and c["nu"] == 10.0
    import statistics as st
    na = [100 * 10 / 110, 120 * 10 / 110]
    su = ((sum((x - st.mean(na)) ** 2 for x in na)) / 2) ** 0.5  # pooled, df 2; b has 0 spread
    sr = ((sum((x - 2.0) ** 2 for x in (1.0, 3.0))) / 2) ** 0.5
    ss = ((2 * 1.0 + 2 * 1.0) / 2) ** 0.5
    assert c["lambda"] == pytest.approx(su / sr) and c["mu"] == pytest.approx(su / ss)
    cfg = mp.MappoConfig(horizon=10)
    assert mp.unit_scale(c, "a", 1, cfg) == pytest.approx(10 / 110)
    assert mp.unit_scale(None, "a", 1, cfg) == cfg.dry_unit_scale
    with pytest.raises(ValueError, match="no base units"):
        mp.unit_scale(c, "z", 9, cfg)


def test_gae_matches_the_recursion() -> None:
    r, v = [1.0, 0.0, 2.0], [0.5, 0.4, 0.3]
    adv, ret = mp.gae(r, v, gamma=0.9, lam=0.8)
    d2 = 2.0 - 0.3
    d1 = 0.0 + 0.9 * 0.3 - 0.4
    d0 = 1.0 + 0.9 * 0.4 - 0.5
    a2 = d2
    a1 = d1 + 0.9 * 0.8 * a2
    a0 = d0 + 0.9 * 0.8 * a1
    assert adv == pytest.approx([a0, a1, a2])
    assert ret == pytest.approx([a0 + 0.5, a1 + 0.4, a2 + 0.3])


@needs
def test_recorded_logp_equals_the_recomputed_one() -> None:
    from tokenomics.eval.runner import run_episode

    pm, om = im.load_models(MODELS)
    ex = mp.ActorExtras(-1.5)
    pol = mp.RecordingPolicies(pm, om, ex, np.random.default_rng(0))
    run_episode(im._episode_env(_job(6)), pol.producing, pol.ordering, seed=_job()["episode"],
                action_seed=None)
    assert {d["role"] for d in pol.decisions} == {"p", "o"}
    with torch.no_grad():
        for d in pol.decisions:
            lp = (mp.producer_logp(pm, ex, d["f"], d["bits"], d["z"], d["mask"])[0]
                  if d["role"] == "p" else mp.orderer_logp(om, d["f"], d["picks"])[0])
            assert float(lp) == pytest.approx(d["logp"], abs=1e-5)


@needs
def test_tiny_dry_train_runs_resumes_and_checkpoints(tmp_path: Path) -> None:
    cfg = mp.MappoConfig(iterations=1, episodes_per_iter=2, horizon=6, ppo_epochs=1, batch=16,
                         workers=1, monitor_every=1)
    kw = dict(training_dir=str(TRAINING), atlas=str(ATLAS_V2), init_models=str(MODELS),
              out_dir=str(tmp_path), weights=W, dry_run=True, max_world_episodes=2,
              log=lambda s: None)
    assert mp.run_train(cfg=cfg, **kw) == 0
    rec = [json.loads(x) for x in (tmp_path / "curve.jsonl").read_text().splitlines()]
    assert len(rec) == 1 and np.isfinite(rec[0]["value"]) and "monitor" in rec[0]
    assert (tmp_path / "latest.pt").exists() and (tmp_path / "models.pt").exists()
    im.load_models(tmp_path / "models.pt")  # the deployed format
    assert mp.run_train(cfg=cfg, **kw) == 0  # resume: iteration 0 is done, nothing reruns
    assert len((tmp_path / "curve.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="pins differ"):
        mp.run_train(cfg=mp.MappoConfig(**{**cfg.__dict__, "lr_actor": 1e-2}), **kw)


@needs
def test_sync_update_is_deterministic(tmp_path: Path) -> None:
    ts, train, _mon = im._training(str(TRAINING))
    cfg = mp.MappoConfig(iterations=1, episodes_per_iter=1, horizon=6, ppo_epochs=1, batch=8,
                         workers=2)
    out = []
    for _ in range(2):
        torch.manual_seed(0)
        pm, om = im.load_models(MODELS)
        mods = {"producer": pm, "orderer": om, "extras": mp.ActorExtras(-1.5),
                "critic": mp.CentralCritic.from_actors(pm, om)}
        snap = tmp_path / f"snap_{_}.pt"
        torch.save(mp._state(mods), snap)
        jobs = mp.iteration_jobs(ts, train, cfg, 0, str(snap), W, 1)
        dec, cri = mp.build_batch([mp.rollout_job(j) for j in jobs], cfg)
        opt = torch.optim.Adam([p for m in mods.values() for p in m.parameters()], lr=1e-3)
        mp._sync_steps(mods, opt, dec, cri, cfg, epoch_seed=0, workers=2)
        out.append([p.detach().clone() for m in mods.values() for p in m.parameters()])
    assert all(torch.equal(a, b) for a, b in zip(*out))


@needs
def test_rollout_reward_is_the_unit_scaled_potential(tmp_path: Path) -> None:
    """A job carries s = H / B_we from the calibration, and the rollout's rewards are the tracker's
    Φ_t at that scale: units · s per cycle when the penalty weights are 0."""
    ts, train, _mon = im._training(str(TRAINING))
    cfg = mp.MappoConfig(horizon=6)
    w, e = train[0]
    cal = {"bar": {f"{ww}|{ee}": 300.0 for ww, ee in train}}
    pm, om = im.load_models(MODELS)
    mods = {"producer": pm, "orderer": om, "extras": mp.ActorExtras(-1.5),
            "critic": mp.CentralCritic.from_actors(pm, om)}
    snap = tmp_path / "snap.pt"
    torch.save(mp._state(mods), snap)
    zero = PotentialWeights(0.0, 0.0, 0.0, 0.75, 0.73)
    job = [j for j in mp.iteration_jobs(ts, train, cfg, 0, str(snap), zero, len(train), cal)
           if (j["world"], j["episode"]) == (w, e)][0]
    assert job["unit_scale"] == pytest.approx(6 / 300.0)
    tr = mp.rollout_job(job)
    assert sum(tr["rewards"]) == pytest.approx(tr["terms"]["units"] * 6 / 300.0)
