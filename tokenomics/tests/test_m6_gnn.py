"""add-m6-joint-mpg-gnn L0 code checks: the fully learned GNN policies (D2 / D3) and Stage I imitation (D4).

- **Labels:** deterministic given the view (the old GNN's unstable label, EXP-0040).
- **Information:** ADR-0058 only; a policy reading anything else raises (leak test).
- **Topology-agnostic:** permuting producer rows permutes the orderer's logits, and one model runs at 12 and
  42 producers and on an unseen topology with zero contract failures (legal commands).
- **Stage I dry chain:** collect → fit → check runs and reports the agreement metrics.
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
    assert im.run_collect(str(TRAINING), str(c), 1, 6, 2) == 0
    assert im.run_fit(str(c), str(f), 1, 0) == 0
    doc = json.loads((f / "fit.json").read_text())
    for key in ("exact_command", "gate_accuracy", "replenish_mae_units", "orderer_kl",
                "orderer_top1"):
        assert key in doc["train_agreement"] and key in doc["monitor_agreement"]
    assert im.run_check(str(TRAINING), str(ATLAS_V2), str(f / "models.pt"), str(k), 1, 6, 1) == 0
    chk = json.loads((k / "imitation_check.json").read_text())
    assert chk["reading"] in ("faithful", "short") and chk["rule"] == im.IMIT_RULE
