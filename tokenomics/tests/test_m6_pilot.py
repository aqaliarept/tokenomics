"""EXP-0113 (pre-registration v10): the capped Stage 1 pilot: the loss, the gate reading, a smoke."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenomics.train import m6_imitation as im  # noqa: E402
from tokenomics.train import m6_pilot as pl  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "experiments/EXP-0058-es-gradient-probe/bar"
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
E10 = REPO / "experiments/EXP-0110-m6-stage0/artifacts"
PREV = REPO / "experiments/EXP-0100-m6-imitation-v2/artifacts/fit/models.pt"


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs = pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(E10 / "start_models.pt") and _lfs(PREV)
                                and (TRAINING / "training_worlds.json").exists()),
                           reason="records absent")


def _gt(g: float, f: float, cd: int = 0, rd: int = 0, cr: float = 0.5, rr: float = 0.5,
        cs: float = 0.5, rs: float = 0.5) -> dict:
    return {"g": g, "F": f, "candidate": {"deaths": cd, "rundown_share": cr, "starvation_share": cs},
            "reference": {"deaths": rd, "rundown_share": rr, "starvation_share": rs}}


def test_gate_reading() -> None:
    assert pl.gate_reading(_gt(-50, 40), final=False) == "stop"
    assert pl.gate_reading(_gt(10, 40, cd=1), final=False) == "stop"
    assert pl.gate_reading(_gt(10, 40, cr=0.6), final=False) == "stop"
    assert pl.gate_reading(_gt(-10, 40), final=False) == "continue"
    assert pl.gate_reading(_gt(50, 40), final=True) == "gain"
    assert pl.gate_reading(_gt(50, 40, cs=0.6), final=True) == "worse"
    assert pl.gate_reading(_gt(-30, 40), final=True) == "hold"
    assert pl.gate_reading(_gt(-50, 40), final=True) == "worse"


@needs
def test_ppo_loss_at_the_start_is_the_policy_gradient() -> None:
    """At the start (old log-prob = current), the ratio is 1 and the KL to the start is 0."""
    import pickle

    from tokenomics.train import m6_gnn as g
    from tokenomics.train import m6_stage0b as sb

    pm, _om = im.load_models(E10 / "start_models.pt")
    pm0, _ = im.load_models(E10 / "start_models.pt")
    mon = REPO / "experiments/EXP-0105-m6-imitation-v3/artifacts/collect/samples_monitor.pkl"
    if not _lfs(mon):
        pytest.skip("samples absent")
    with mon.open("rb") as fh:
        prod = pickle.load(fh)["producer"][:20]
    batch = []
    for f, _y in prod:
        f4 = g.v3_to_v4(f)
        with torch.no_grad():
            tl = sb.tempered_logits(pm.heads(f4)[0])
        bits = (torch.sigmoid(tl).numpy() > 0.5).astype(np.float32)
        lp = float(torch.distributions.Bernoulli(logits=tl).log_prob(torch.from_numpy(bits)).sum())
        batch.append({"f": f4, "bits": bits, "old_lp": lp, "adv": 1.0})
    loss, st = pl.ppo_loss(pm, pm0, batch)
    assert st["clip_frac"] == 0.0 and abs(st["kl"]) < 1e-6
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pm.parameters())


@needs
def test_pilot_dry_run(tmp_path: Path) -> None:
    """A true smoke: 2 iterations of 2 seeds at horizon 4, a 1-world gate at each; the registered
    config is restored afterwards (the pre-registration still matches)."""
    assert pl.run(str(TRAINING), str(ATLAS_V2), str(E10), str(PREV), str(tmp_path), 1,
                  dry_run=True) == 0
    doc = json.loads((tmp_path / "reading.json").read_text())
    assert doc["reading"] in ("gain", "hold", "worse", "stopped_early")
    curve = [json.loads(x) for x in (tmp_path / "curve.jsonl").read_text().splitlines()]
    assert len(curve) >= 1 and "gate" in curve[0] and set(curve[0]["flip_rate"]) == {
        "restore", "upgrade", "expand"}
    assert (tmp_path / "checkpoint.pt").exists() and (tmp_path / "steps.jsonl").exists()
    assert pl.prereg_problems() == []


def test_v10_is_frozen_and_pinned() -> None:
    assert pl.prereg_problems() == [] and pl.freeze()[0] == "MATCH"


def test_v11_changes_only_the_length_and_the_gates() -> None:
    diff = {k for k in pl.CFG if pl.CFG[k] != pl.CFG_V11[k]}
    assert diff == {"iterations", "gate_iterations"} and set(pl.CFG) == set(pl.CFG_V11)
    assert pl.CFG_V11["iterations"] == 40 and pl.CFG_V11["gate_iterations"] == [15, 20, 30, 40]
    doc = pl.registered_prereg_v11()
    assert doc["keeps"]["m6_prereg_v10.json"] == pl.PREREG_V10_SHA256
    assert doc["gate"]["stop_only"] == [15, 20, 30] and doc["gate"]["deciding"] == 40


def test_v11_is_frozen_and_pinned() -> None:
    assert pl.prereg_v11_problems() == []
    assert pl.freeze(pl.PREREG_V11_PATH, pl.registered_prereg_v11())[0] == "MATCH"


def test_v11_needs_its_resume_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resume-from"):
        pl.run("t", "a", "e", "r", str(tmp_path), 1, dry_run=True, prereg="v11")
    with pytest.raises(ValueError, match="resume-from"):
        pl.run("t", "a", "e", "r", str(tmp_path), 1, dry_run=True, resume_from=str(tmp_path))


@needs
def test_continuation_dry_run_resumes_the_pilot(tmp_path: Path) -> None:
    """A v10 smoke (iterations 1-2), then the v11 smoke resumed from its checkpoint (iterations 3-4):
    the resume pins hold, a tampered checkpoint is refused, the curve starts after the resumed
    iteration, and EXP-0100's reference rows reproduce."""
    a, b = tmp_path / "a", tmp_path / "b"
    assert pl.run(str(TRAINING), str(ATLAS_V2), str(E10), str(PREV), str(a), 1, dry_run=True) == 0
    start = str(E10 / "start_models.pt")
    assert pl.resume_problems(a, start, real=False) == []
    assert any("EXP-0113" in x for x in pl.resume_problems(a, start, real=True))
    assert pl.run(str(TRAINING), str(ATLAS_V2), str(E10), str(PREV), str(b), 1, dry_run=True,
                  prereg="v11", resume_from=str(a)) == 0
    doc = json.loads((b / "reading.json").read_text())
    assert doc["prereg_version"] == "v11" and doc["resumed_from"]["iteration"] == 2
    assert doc["resumed_from"]["ref_reproduction"] == "MATCH"
    curve = [json.loads(x) for x in (b / "curve.jsonl").read_text().splitlines()]
    assert curve[0]["iteration"] == 3 and "gate" in curve[0]
    d = torch.load(a / "checkpoint.pt", weights_only=False)
    k = next(iter(d["producer"]))
    d["producer"][k] = d["producer"][k] + 1.0
    torch.save(d, a / "checkpoint.pt")
    assert any("differs from the gated candidate" in x
               for x in pl.resume_problems(a, start, real=False))
    assert pl.prereg_problems() == [] and pl.prereg_v11_problems() == []
