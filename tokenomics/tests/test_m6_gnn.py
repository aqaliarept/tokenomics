"""add-m6-joint-mpg-gnn L0 code checks: the fully learned GNN policies (D2 / D3) and Stage I imitation (D4).

- **Labels:** deterministic given the view (the old GNN's unstable label, EXP-0040).
- **Information:** ADR-0058 only; a policy reading anything else raises (leak test).
- **Topology-agnostic:** permuting producer rows permutes the orderer's logits, and one model runs at 12 and
  42 producers and on an unseen topology with zero contract failures (legal commands).
- **Stage I dry chain:** collect → fit → check runs and reports the agreement metrics.
- **Fit (v2, DIAGNOSTICS 31):** the orderer loss stays finite and falls on raw counts; the sync fit is
  deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenomics.train import m6_gnn as g  # noqa: E402
from tokenomics.train import m6_imitation as im  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
TRAINING = REPO / "experiments/EXP-0058-es-gradient-probe/bar"
PHASE1 = REPO / "tokenomics/tokenomics/config/phase1.json"
PREV = REPO / "experiments/EXP-0100-m6-imitation-v2/artifacts/fit/models.pt"


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs = pytest.mark.skipif(not (_lfs(ATLAS_V2) and (TRAINING / "training_worlds.json").exists()),
                           reason="records absent")


def _job(world: str, horizon: int) -> dict:
    from tokenomics.eval.decoder_check import MWWK, decoder_arm, select_views, world_entry
    from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster, member_id
    from tokenomics.eval.routing_audit import build_jobs
    from tokenomics.eval.teacher_audit import load_atlas

    view = select_views(load_atlas(ATLAS_V2), [world])[0]
    arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
    arm = {**arm, "producing": im.BASE_PRODUCER, "subject": member_id(im.BASE_PRODUCER, arm["ordering"])}
    return {**build_jobs([world_entry(view)], [arm], 1, horizon, None, {})[0], "order_ttl": 2}


class _Proxy:
    """Exposes only `allowed` attributes of the wrapped object; anything else raises."""

    def __init__(self, obj: object, allowed: tuple[str, ...]) -> None:
        object.__setattr__(self, "_o", obj)
        object.__setattr__(self, "_a", allowed)

    def __getattr__(self, name: str) -> object:
        if name not in object.__getattribute__(self, "_a"):
            raise AttributeError(f"policy read system.{name}")
        return getattr(object.__getattribute__(self, "_o"), name)


def _restricted(env: object, allowed: tuple[str, ...]) -> SimpleNamespace:
    from tokenomics.eval.routing_diagnostics import DRAW_KEY_ATTR

    ns = SimpleNamespace(system=_Proxy(env.system, allowed), spec_=env.spec_, _role=env._role)
    setattr(ns, DRAW_KEY_ATTR, getattr(env, DRAW_KEY_ATTR, None))
    return ns


PRODUCER_INFO = ("producer_entity_view", "cycle_counter", "investment_fund")
ORDERER_INFO = ("product_stream_view", "ordering_agents", "producer_infos", "cycle_counter")


@needs
def test_policies_read_only_their_own_view() -> None:
    """ADR-0058: the producer GNN reads its own entity view (and the fund level, as k5_rf5);
    the orderer GNN its own stream view and routing view. Any other read raises."""
    from tokenomics.eval.runner import run_episode

    env = im._episode_env(_job("ws-atlas-v1-12-0111", 6))
    pp, op = g.gnn_producing_policy(g.ProducerGNN()), g.gnn_ordering_policy(g.OrdererGNN())
    run_episode(env, lambda e, a: pp(_restricted(e, PRODUCER_INFO), a),
                lambda e, a: op(_restricted(e, ORDERER_INFO), a), seed=1)


@needs
def test_labels_are_deterministic_given_the_view() -> None:
    """The same keyed episode records the same features and labels twice (k5_rf5 and MWW's
    weights are deterministic functions of the view; MWWk's draw is not part of the label)."""
    job = _job("ws-atlas-v1-12-0111", 6)
    a, b = im.record_job(job), im.record_job(job)
    assert len(a["producer"]) == len(b["producer"]) > 0 and len(a["orderer"]) > 0
    for (fa, ya), (fb, yb) in zip(a["producer"], b["producer"]):
        assert all(np.array_equal(ya[k], yb[k]) for k in ya)
        assert np.array_equal(fa["input"], fb["input"]) and np.array_equal(fa["capacity"],
                                                                            fb["capacity"])
    for (_fa, ya), (_fb, yb) in zip(a["orderer"], b["orderer"]):
        assert all(np.allclose(x, y) for x, y in zip(ya, yb))
        assert all(abs(x.sum() - 1) < 1e-5 for x in ya if len(x))


def test_orderer_logits_permute_with_producer_rows() -> None:
    """No ids are features: permuting the producer rows permutes each slot's logits."""
    torch.manual_seed(0)
    m = g.OrdererGNN()
    rng = np.random.default_rng(0)
    f = {"global": rng.normal(size=(1, 9)).astype(np.float32),
         "producer": rng.normal(size=(5, m.enc["producer"][0].in_features)).astype(np.float32),
         "order": rng.normal(size=(2, 7)).astype(np.float32), "elig": [[0, 1, 2, 3, 4], [1, 3]],
         "pids": list("abcde")}
    perm = [3, 0, 4, 1, 2]  # new row j = old row perm[j]
    inv = {old: new for new, old in enumerate(perm)}
    f2 = {**f, "producer": f["producer"][perm], "pids": [f["pids"][i] for i in perm],
          "elig": [[inv[j] for j in el] for el in f["elig"]]}
    with torch.no_grad():
        a, b = m(f), m(f2)
    for la, lb in zip(a, b):
        assert np.allclose(np.sort(la.numpy()), np.sort(lb.numpy()), atol=1e-5)
    assert np.allclose(a[1].numpy(), b[1].numpy(), atol=1e-5)  # same producers, same order


@needs
@pytest.mark.parametrize("world", ["ws-atlas-v1-12-0111", "ws-atlas-v1-42-0000", "phase1"])
def test_one_model_runs_legally_at_12_42_and_an_unseen_topology(world: str) -> None:
    """Random nonzero weights, one parameter set, zero contract failures (the env decode keeps
    every command legal) at 12 and 42 producers and on phase1 (a topology outside the atlas)."""
    from tokenomics.domain.config import load_config
    from tokenomics.eval.contract import ContractMonitor
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.routing_diagnostics import DrawKey, set_draw_key
    from tokenomics.eval.runner import run_episode

    torch.manual_seed(3)
    pm, om = g.ProducerGNN(), g.OrdererGNN()
    if world == "phase1":
        env = joint_stack_env(load_config(PHASE1), 8)
        set_draw_key(env, DrawKey("phase1", 1, 0))
    else:
        env = im._episode_env(_job(world, 8))
    mon = ContractMonitor(env)
    run_episode(env, g.gnn_producing_policy(pm), g.gnn_ordering_policy(om), seed=1, monitor=mon)
    mon.verify(dead=bool(env.episode_stats["dead"]), cycles=env.episode_stats["cycles"], horizon=8)
    assert sum(p.numel() for p in pm.parameters()) == sum(
        p.numel() for p in g.ProducerGNN().parameters())  # size-independent


@needs
def test_stage_one_dry_chain(tmp_path: Path) -> None:
    """collect (2 training world-episodes, horizon 6) -> fit (1 epoch) -> check (1 monitor
    world-episode): runs end to end; fit reports the agreement metrics; check writes a reading."""
    c, f, k = tmp_path / "c", tmp_path / "f", tmp_path / "k"
    assert im.run_collect(str(TRAINING), str(c), 1, 6, 2, dry_run=True, atlas_path=str(ATLAS_V2),
                          horizon_42=6) == 0
    meta = json.loads((c / "collect.json").read_text())
    assert meta["features"] == g.FEATURES_V3 and meta["counts"]["train"]["producer_42"] > 0
    assert im.run_fit(str(c), str(f), 1, 0, dry_run=True) == 0
    doc = json.loads((f / "fit.json").read_text())
    for key in ("exact_command", "gate_accuracy", "replenish_mae_units", "orderer_kl",
                "orderer_top1"):
        assert key in doc["train_agreement"] and key in doc["monitor_agreement"]
    assert im.run_check(str(TRAINING), str(ATLAS_V2), str(f / "models.pt"), str(k), 1, 6, 1,
                        dry_run=True, prev_models=str(PREV), episodes_42=1, horizon_42=6) == 0
    assert set(doc["monitor_agreement"]["producer_heads"]["toggles"]) == set(im.TOGGLES)
    spec = json.loads((f / "spec.json").read_text())
    assert spec["fit_mode"] in ("sync", "hogwild")
    chk = json.loads((k / "imitation_check.json").read_text())
    assert chk["reading"] in ("faithful", "short") and chk["rule"] == im.V3_RULE
    assert set(chk["b_42"]["over_ceiling"]) == {"gnn", "k5_rf5"} and chk["c_vs_exp0100"]["F"]


def _raw_orderer_samples(n: int, seed: int = 0) -> list:
    """Orderer samples with EXP-0099-scale raw counts (thousands) and a learnable soft label:
    the MWW-like weights follow the makeable-units column."""
    rng = np.random.default_rng(seed)
    width = g.osc._PRODUCER_ROW_DIM + len(g.OUTCOME_KINDS)
    mk = g.osc._ORDERER_PRODUCER_ROW_FEATURES.index("makeable_qty_now")
    out = []
    for _ in range(n):
        k = int(rng.integers(2, 6))
        rows = rng.uniform(0, 1, size=(k, width)).astype(np.float32)
        rows[:, list(g.ORDERER_LOG1P["producer"])] *= rng.uniform(100, 6000)
        glob = rng.uniform(0, 6000, size=(1, g.osc._PRODUCT_DIM)).astype(np.float32)
        y = rows[:, mk] / rows[:, mk].sum()
        f = {"global": glob, "producer": rows, "order": rng.uniform(0, 3, (1, 7)).astype(
            np.float32), "elig": [list(range(k))], "pids": [f"p{i}" for i in range(k)]}
        out.append((f, [y.astype(np.float32)]))
    return out


def test_orderer_fit_loss_stays_finite_and_falls_on_raw_counts() -> None:
    """DIAGNOSTICS 31: on raw counts up to ~6,000 the v2 fit (normaliser + clipping) keeps the
    orderer's loss finite and falling, and its weights bounded."""
    samples = {"producer": [], "orderer": _raw_orderer_samples(256)}
    _pm, om, curve, info = im.fit(samples, 12, batch=32, lr=3e-3, log=lambda s: None)
    losses = [c["orderer_loss"] for c in curve]
    assert all(np.isfinite(losses)) and losses[-1] < 0.9 * losses[0]
    assert max(float(p.detach().abs().max()) for p in om.parameters()) < 50
    assert info["normaliser"]["orderer"]["producer"]["rows"] > 0
    f = samples["orderer"][0][0]  # the fitted normaliser bounds raw counts in the thousands
    for block in ("global", "producer", "order"):
        z = om.norm(block, f[block])
        assert float(z.abs().max()) <= g.NORM_CLIP and torch.isfinite(z).all()
    assert float(np.abs(f["producer"]).max()) > 100


def test_sync_fit_is_deterministic() -> None:
    """Two synchronous data-parallel fits (2 workers) give the same weights bit for bit."""
    samples = {"producer": [], "orderer": _raw_orderer_samples(96, seed=1)}
    a = im.fit(samples, 2, batch=16, workers=2, mode="sync", log=lambda s: None)
    b = im.fit(samples, 2, batch=16, workers=2, mode="sync", log=lambda s: None)
    assert all(np.isfinite([c["orderer_loss"] for c in a[2]]))
    for x, y in zip(a[1].state_dict().values(), b[1].state_dict().values()):
        assert torch.equal(x, y)


def test_shard_files_give_the_same_fit_as_in_memory(tmp_path: Path) -> None:
    """EXP-0105's out-of-memory fix: each worker loads only its shard file (samples[w::workers]);
    the sync fit is bit-identical to the in-memory one, and so is the normaliser."""
    samples = {"producer": [], "orderer": _raw_orderer_samples(96, seed=2)}
    a = im.fit(samples, 2, batch=16, workers=2, mode="sync", log=lambda s: None)
    sh = im.write_shards(samples, 2, tmp_path / "sh", 50_000)
    assert len(sh.paths) == 2 and sh.counts == (0, 96)
    b = im.fit(sh, 2, batch=16, workers=2, mode="sync", log=lambda s: None)
    for x, y in zip(a[1].state_dict().values(), b[1].state_dict().values()):
        assert torch.equal(x, y)
    h = im.fit(sh, 1, batch=16, workers=2, mode="hogwild", log=lambda s: None)
    assert np.isfinite(h[2][-1]["orderer_loss"])
    with pytest.raises(ValueError, match="shard count"):
        im.fit(sh, 1, workers=3, log=lambda s: None)


def test_real_stage_one_runs_need_the_pinned_preregistration(tmp_path: Path) -> None:
    """A real collect / fit / check refuses without the pin (v5, Stage I v3), a changed file, other
    settings; the frozen v2 still matches its code."""
    assert im.prereg_problems() == [] and im.freeze_prereg_v5()[0] == "MATCH"
    assert im.freeze_prereg()[0] == "MATCH"
    with pytest.raises(ValueError, match="horizon 6"):
        im.run_collect(str(TRAINING), str(tmp_path / "c"), 1, 6, None)
    assert any("--epochs 3" in x for x in im.real_problems(horizon=100, max_world_episodes=None,
                                                           epochs=3))
    import tokenomics.train.m6_imitation as mod

    old = mod.PREREG_V5_SHA256
    try:
        mod.PREREG_V5_SHA256 = "0" * 64
        assert "does not match the pinned" in mod.prereg_problems()[0]
    finally:
        mod.PREREG_V5_SHA256 = old


def test_ceiling_margin_is_signed_and_scale_free() -> None:
    """DIAGNOSTICS 32: < 0 iff one more upgrade puts degradation past one restoration per cycle;
    bounded in [-1, 1] at any scale (a restore_effect = 0 row reads -1); v3 adds it as the 17th
    capacity column; a v2 checkpoint keeps 16."""
    row = SimpleNamespace(degradation_rate=6, max=198, upgrade_effect=16, restore_effect=12)
    assert g.ceiling_margin(row) == pytest.approx((12 - 13) / 13)  # EXP-0104: 13 vs restores 12
    big = SimpleNamespace(degradation_rate=6, max=1980, upgrade_effect=160, restore_effect=120)
    assert g.ceiling_margin(big) == pytest.approx((120 - 129) / 129)
    for mx in (10, 1000, 100000):  # a row without a restoration: -1 at any scale
        none = SimpleNamespace(degradation_rate=6, max=mx, upgrade_effect=0, restore_effect=0)
        assert g.ceiling_margin(none) == -1.0
    idle = SimpleNamespace(degradation_rate=0, max=50, upgrade_effect=0, restore_effect=0)
    assert g.ceiling_margin(idle) == 0.0
    ok = SimpleNamespace(degradation_rate=6, max=150, upgrade_effect=16, restore_effect=12)
    assert g.ceiling_margin(ok) > 0
    assert g.CAPACITY_DIM == {g.FEATURES_V2: 16, g.FEATURES_V3: 17}
    if _lfs(PREV):
        pm, _om = im.load_models(PREV)
        assert pm.features == g.FEATURES_V2 and pm.enc["capacity"][0].in_features == 16

