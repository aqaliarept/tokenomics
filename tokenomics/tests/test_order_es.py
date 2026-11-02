"""Task 5.2: the step-2 ES loop and the gradient-signal probe (add-m4-orderer-redesign design
D6 / D7; learned-orderer "Outcome training by bounded ES", "Relative disqualification")."""

from __future__ import annotations

import copy
import csv
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tokenomics.eval.routing_audit import job_config
from tokenomics.train import order_es as es
from tokenomics.train import order_es_worlds as ow
from tokenomics.train.per_order_scorer import (ARMS, PSI_FEATURES, FeatureNormaliser, RawRows,
                                               N_PHI, N_PSI, input_transforms, transform_psi)

REPO = Path(__file__).resolve().parents[2]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"


def _lfs_present(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs_inputs = pytest.mark.skipif(not (_lfs_present(ATLAS_V2) and _lfs_present(ORDERER)),
                                  reason="EXP-0052 atlas / EXP-0012 checkpoint absent")
QUIET: Callable[[str], None] = lambda m: None  # noqa: E731


@pytest.fixture(scope="module")
def bar_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """5.1's 2-world horizon-12 dry-run bar (1 atlas + 1 fresh world, 2 episodes each)."""
    if not (_lfs_present(ATLAS_V2) and _lfs_present(ORDERER)):
        pytest.skip("EXP-0052 atlas / EXP-0012 checkpoint absent")
    out = tmp_path_factory.mktemp("es") / "bar"
    assert ow.main(["--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER), "--dry-run",
                    "--atlas-worlds", "1", "--target", "2", "--min-feasible", "1",
                    "--horizon", "12", "--shock-profile", "shocks-i-smoke", "--workers", "1",
                    "--out-dir", str(out)]) == 0
    return out


# --- registered values ---


def test_registered_values() -> None:
    assert es.PAIRS == 12 and es.H_HOLDING == 0.1 and es.WEIGHT_DECAY == 0.005
    assert es.ADAM["lr"] == 0.02 and es.GENERATIONS == {"pilot": 20, "main": 40}
    assert es.MONITOR_EVERY == 10 and es.WEAK_MINIBATCH == 24 and ow.MINIBATCH == 12
    assert (es.SIGNAL_COSINE, es.WEAK_COSINE, es.A2_RANK2_BELOW) == (0.3, 0.1, 0.1)
    # parameter counts unchanged by the log1p transform (D5 as amended by task 4.3)
    assert {a: ARMS[a].n_params for a in ("A1", "A2", "A2-R2")} == {"A1": 44, "A2": 268,
                                                                    "A2-R2": 156}
    for a in ("A1", "A2", "A2-R2"):
        assert np.all(ARMS[a].sigma() == 0.05)


# --- pure ES ---


def test_centered_ranks_ties_and_worst() -> None:
    r = es.centered_ranks([3.0, 1.0, 2.0, 5.0], [False] * 4)
    assert r.tolist() == pytest.approx([1 / 6, -0.5, -1 / 6, 0.5])
    # ties share the average rank
    r = es.centered_ranks([1.0, 2.0, 2.0, 3.0], [False] * 4)
    assert r[1] == r[2] == pytest.approx(0.0)
    # a disqualified member takes the worst rank whatever its fitness; several share it
    r = es.centered_ranks([9.0, 1.0, 2.0, 8.0], [True, False, False, True])
    assert r[0] == r[3] == pytest.approx(-0.5 + 0.5 / 3) and r[0] < r[1] < r[2]
    r = es.centered_ranks([9.0, 1.0, 2.0, 3.0], [True, False, False, False])
    assert r[0] == pytest.approx(-0.5) and r[0] == r.min()
    with pytest.raises(ValueError):
        es.centered_ranks([float("nan"), 1.0], [False, False])


def test_estimate_uses_pair_rank_differences() -> None:
    """D6: centered ranks over all 2n members; the estimate weighs eps_i by r(+i) - r(-i) and
    divides by 2 n sigma_j (per parameter)."""
    rng = np.random.default_rng(1)
    n, p = 5, 7
    eps = rng.standard_normal((n, p))
    sigma = np.linspace(0.01, 0.2, p)
    labels = es.member_labels(n)
    fit = {lab: float(rng.normal()) for lab, _, _ in labels}
    worst = {lab: False for lab, _, _ in labels}
    rk = es.rank_and_estimate(eps, sigma, fit, worst)
    allr = es.centered_ranks([fit[lab] for lab, _, _ in labels], [False] * (2 * n))
    want = np.zeros(p)
    for i in range(n):
        want += (allr[i] - allr[n + i]) * eps[i]
    assert np.allclose(rk.estimate, want / (2 * n * sigma))
    # a pair whose members tie contributes nothing
    tied = dict(fit, **{"-02": fit["+02"]})
    r2 = es.rank_and_estimate(eps, sigma, tied, worst)
    assert r2.ranks["+02"] == r2.ranks["-02"]
    # disqualified / non-finite members rank worst
    dq = es.rank_and_estimate(eps, sigma, dict(fit, **{"+00": 1e9}), dict(worst, **{"+00": True}))
    assert dq.ranks["+00"] == -0.5 and dq.worst["+00"]
    nf = es.rank_and_estimate(eps, sigma, dict(fit, **{"-01": None}), worst)
    assert nf.ranks["-01"] == -0.5 and nf.worst["-01"]


def test_member_thetas_apply_the_arm_sigma_vector() -> None:
    rng = np.random.default_rng(2)
    theta = rng.normal(size=44)
    sigma = np.linspace(0.01, 0.1, 44)
    eps = rng.standard_normal((3, 44))
    th = es.member_thetas(theta, eps, sigma)
    assert np.array_equal(th[es.CENTER], theta) and len(th) == 7
    for lab, sign, i in es.member_labels(3):
        assert np.allclose((th[lab] - theta) / eps[i], sign * sigma)
    # the noise stream is a pure function of (opt seed, generation)
    assert np.array_equal(es.noise(0, 3, 12, 44), es.noise(0, 3, 12, 44))
    assert not np.array_equal(es.noise(0, 3, 12, 44), es.noise(1, 3, 12, 44))
    assert not np.array_equal(es.noise(0, 3, 12, 44), es.noise(0, 4, 12, 44))


def _toy_es(f1: Callable[[np.ndarray], float], theta: np.ndarray, gens: int, pairs: int = 12,
            sigma: float = 0.05) -> np.ndarray:
    adam = es.AdamState.zeros(len(theta))
    s = np.full(len(theta), sigma)
    for g in range(1, gens + 1):
        eps = es.noise(0, g, pairs, len(theta))
        th = es.member_thetas(theta, eps, s)
        rk = es.rank_and_estimate(eps, s, {k: f1(v) for k, v in th.items()},
                                  {k: False for k in th})
        theta, adam = es.adam_ascend(theta, rk.estimate, adam, weight_decay=0.0)
    return theta


def test_quadratic_toy_moves_toward_the_optimum() -> None:
    opt = np.linspace(-0.5, 0.5, 10)

    def f(t: np.ndarray) -> float:
        return -float(np.sum((t - opt) ** 2))

    start = np.zeros(10)
    end = _toy_es(f, start, 40)
    assert np.linalg.norm(end - opt) < 0.3 * np.linalg.norm(start - opt)
    # weight decay pulls toward 0 (coupled L2): a flat fitness shrinks theta
    t, a = np.ones(4), es.AdamState.zeros(4)
    t2, _ = es.adam_ascend(t, np.zeros(4), a)
    assert np.all(t2 < t)


def test_probe_cosine_noiseless_and_noise() -> None:
    opt = np.linspace(-1, 1, 44)

    def f(t: np.ndarray) -> float:
        return -float(np.sum((t - opt) ** 2))

    n, p = 12, 44
    eps = es.probe_noise("A1", n, p)
    sigma = ARMS["A1"].sigma()
    th = es.member_thetas(np.zeros(p), eps, sigma)
    worst = {k: False for k in th}
    fit = {k: f(v) for k, v in th.items()}
    g1 = es.rank_and_estimate(eps, sigma, fit, worst).estimate
    g2 = es.rank_and_estimate(eps, sigma, dict(fit), worst).estimate
    assert es.cosine(g1, g2) == pytest.approx(1.0)
    # pure noise: fitness unrelated to theta and independent across the two minibatches
    n = 3000
    eps = es.noise(0, 1, n, p)
    rng = np.random.default_rng(3)
    labs = [lab for lab, _, _ in es.member_labels(n)]
    w = {lab: False for lab in labs}
    a = es.rank_and_estimate(eps, sigma, {lab: float(rng.normal()) for lab in labs}, w)
    b = es.rank_and_estimate(eps, sigma, {lab: float(rng.normal()) for lab in labs}, w)
    assert abs(es.cosine(a.estimate, b.estimate)) < 0.1
    assert es.cosine(np.zeros(3), np.ones(3)) is None
    # CRN: a shared world effect cancels in paired differences, not in unpaired ones
    world = {0: 5.0, 1: -5.0}
    labs12 = [lab for lab, _, _ in es.member_labels(12)]
    fs = [{lab: world[k] + 0.01 * rng.normal() for lab in labs12} for k in (0, 1)]
    vr = es.variance_ratio(fs[0], fs[1], 12)
    assert vr is not None and vr < 0.01


def test_probe_reading_rule() -> None:
    r = es.probe_reading
    assert r({"A1": 0.3, "A2": 0.5})["reading"] == "signal"
    assert r({"A1": 0.2999, "A2": 0.5})["reading"] == "weak"
    assert r({"A1": 0.2, "A2": 0.5})["minibatch"] == 24
    assert r({"A1": 0.1, "A2": 0.5})["reading"] == "weak"
    assert r({"A1": 0.0999, "A2": 0.5})["reading"] == "no_signal"
    assert r({"A1": None, "A2": 0.5})["reading"] == "no_signal"
    assert r({"A1": 0.4, "A2": 0.05})["a2_arm"] == "A2-R2"
    assert r({"A1": 0.4, "A2": 0.05})["a2_rank"] == 2
    assert r({"A1": 0.4, "A2": 0.1})["a2_arm"] == "A2"
    assert r({"A1": 0.05, "A2": 0.05})["a2_arm"] == "A2"  # no_signal: stop, no rank change
    assert r({"A1": 0.4, "A2": None})["a2_arm"] == "A2-R2"


# --- fitness and relative disqualification ---


def _bar(**kw: object) -> es.BarRow:
    d = dict(world="w", episode=1, bar=200.0, demand_per_cycle=10.0, a0_units=150.0,
             a0_dead=False, a0_rundown=False, a0_starved=frozenset(), demand_digest="d",
             shock_digest="s")
    d.update(kw)
    return es.BarRow(**d)  # type: ignore[arg-type]


def _row(**kw: str) -> dict[str, str]:
    d = {"world": "w", "episode": "1", "units": "100.0", "dead": "false", "r_start": "1.0",
         "r_end": "0.9", "consumer_fill_rate": json.dumps({"c0": 0.9, "c1": 0.9}),
         "holding_units_per_cycle": "2.0", "nonfinite": "false"}
    d.update(kw)
    return d


def test_control_also_starves_a_consumer() -> None:
    """Spec "Control also starves a consumer": member and A0 both leave c0 below Y -> not
    disqualified for c0; a consumer only the member starves -> disqualified."""
    y = ow.Y_STARVATION
    both = _row(consumer_fill_rate=json.dumps({"c0": y - 0.1, "c1": 0.9}))
    assert es.disqualifying(both, _bar(a0_starved=frozenset({"c0"}))) == []
    assert es.disqualifying(both, _bar()) == ["starvation:c0"]
    other = _row(consumer_fill_rate=json.dumps({"c0": y - 0.1, "c1": y - 0.2}))
    assert es.disqualifying(other, _bar(a0_starved=frozenset({"c0"}))) == ["starvation:c1"]
    assert es.disqualifying(_row(consumer_fill_rate=json.dumps({"c0": y})), _bar()) == []
    # death and run-down are relative too
    assert es.disqualifying(_row(dead="true"), _bar()) == ["death"]
    assert es.disqualifying(_row(dead="true"), _bar(a0_dead=True)) == []
    low = _row(r_end=repr(ow.X_RUNDOWN * 0.99))
    assert es.disqualifying(low, _bar()) == ["rundown"]
    assert es.disqualifying(low, _bar(a0_rundown=True)) == []
    assert es.disqualifying(_row(r_end=repr(ow.X_RUNDOWN)), _bar()) == []


def test_fitness_and_member_score() -> None:
    bar = {("w", 1): _bar(), ("w", 2): _bar(episode=2, bar=100.0, demand_per_cycle=5.0)}
    assert es.we_fitness(_row(), bar[("w", 1)]) == pytest.approx(100 / 200 - 0.1 * 2 / 10)
    rows = [_row(), _row(episode="2", units="50.0", holding_units_per_cycle="0.0")]
    s = es.score_member(rows, bar)
    assert s.fitness == pytest.approx(((0.5 - 0.02) + 0.5) / 2) and not s.disqualified
    s = es.score_member([_row(), _row(episode="2", dead="true")], bar)
    assert s.disqualified and s.reasons == ("w:2:death",) and s.fitness is not None
    s = es.score_member([_row(), _row(nonfinite="true")], bar)
    assert s.nonfinite and s.fitness is None


# --- pre-registration ---


def _frozen() -> dict:
    d = es.draft_prereg()
    d["status"] = es.FROZEN
    d["readings"] = {"fixture": True}
    return d


TRAIN_OK = dict(mode=es.TRAIN, arm="A1", opt_seed=0, generations=20, minibatch=12)


def test_unregistered_change_refused() -> None:
    """Spec "Unregistered change refused": arms, seeds, scales or rules that differ from the
    recorded pre-registration refuse a real run."""
    ok = _frozen()
    assert es.prereg_problems(ok, **TRAIN_OK) == []
    assert es.prereg_problems(ok, **{**TRAIN_OK, "arm": "A2-R2", "generations": 40,
                                     "minibatch": 24, "opt_seed": 1}) == []
    assert es.prereg_problems(ok, mode=es.PROBE, arms=["A1", "A2"], minibatch=6) == []

    def bad(edit: Callable[[dict], None], **kw: object) -> list[str]:
        d = copy.deepcopy(ok)
        edit(d)
        return es.prereg_problems(d, **{**TRAIN_OK, **kw})

    for edit, where in (
            (lambda d: d["sigma"].update(w=0.1), "sigma.w"),
            (lambda d: d["arms"]["A2"].update(rank=3), "arms.A2.rank"),
            (lambda d: d["es"]["adam"].update(lr=0.05), "es.adam.lr"),
            (lambda d: d["es"]["fitness"].update(h=0.2), "es.fitness.h"),
            (lambda d: d["es"].update(pairs=8), "es.pairs"),
            (lambda d: d["es"].update(opt_seeds=[0, 1, 2]), "es.opt_seeds"),
            (lambda d: d["normaliser"].update(clip=4.0), "normaliser.clip"),
            (lambda d: d["normaliser"].update(transforms={}), "normaliser.transforms"),
            (lambda d: d["policy"].update(decoder_version="x"), "policy.decoder_version"),
            (lambda d: d["probe"]["readings"].update(signal=0.2), "probe.readings.signal")):
        got = bad(edit)
        assert len(got) == 1 and where in got[0], (where, got)
    # run choices outside the registration
    for kw, word in ((dict(opt_seed=2), "opt seed"), (dict(generations=30), "generations"),
                     (dict(minibatch=13), "minibatch"), (dict(arm="A0"), "arm"),
                     (dict(pairs=6), "pairs"), (dict(monitor_every=5), "monitor")):
        got = es.prereg_problems(ok, **{**TRAIN_OK, **kw})
        assert len(got) == 1 and word in got[0], (kw, got)
    assert es.prereg_problems(ok, mode=es.PROBE, arms=["A1"], minibatch=6)
    # a draft is not a frozen pre-registration; a missing field is a schema problem
    draft = es.draft_prereg()
    assert es.prereg_problems(draft, **TRAIN_OK) == [
        "pre-registration status is 'draft', a real run needs 'frozen'"]
    d = copy.deepcopy(ok)
    del d["es"]["fitness"]["h"]
    assert es.prereg_problems(d, **TRAIN_OK) == ["pre-registration lacks es.fitness.h"]
    assert es.prereg_schema_problems([]) == ["pre-registration is not a JSON object"]


def test_frozen_file_matches_the_module() -> None:
    """The frozen pre-registration (task 5.3; it superseded 5.2's draft) holds this module's
    registered values, and a real run accepts it."""
    doc = json.loads(es.PREREG_PATH.read_text())
    assert doc["status"] == es.FROZEN and es.prereg_schema_problems(doc) == []
    for sec, want in es.registered_sections().items():
        assert doc[sec] == want, sec
    assert doc["normaliser"]["transforms"] == {"psi.eligible_producer_count": "log1p"}
    assert es.prereg_problems(doc, **TRAIN_OK) == [] and es.prereg_file_problems(
        es.PREREG_PATH) == []


# --- log1p (supervisor decision 3) ---


def test_log1p_transform_and_version_refusals() -> None:
    assert input_transforms() == {"psi.eligible_producer_count": "log1p"}
    j = PSI_FEATURES.index("eligible_producer_count")
    psi = np.arange(N_PSI, dtype=np.float64) + 1.0
    t = transform_psi(psi)
    assert t[j] == pytest.approx(np.log1p(psi[j])) and psi[j] == j + 1.0  # input untouched
    assert np.array_equal(np.delete(t, j), np.delete(psi, j))
    nm = FeatureNormaliser.identity()
    d = nm.to_dict()
    assert d["transforms"] == input_transforms() and d["version"] == "median-iqr-log1p-v2"
    for k, v in (("version", "median-iqr-v1"), ("transforms", {})):
        with pytest.raises(ValueError):
            FeatureNormaliser.from_dict({**d, k: v})
    # a constant transformed column (log1p(4) on every row) has zero spread exactly -> scale 1,
    # not np.std's ~1e-16 float noise
    rows = RawRows([np.zeros((3, N_PHI))], [transform_psi(np.full((100, N_PSI), 4.0))])
    assert FeatureNormaliser.fit(rows).psi_scale[j] == 1.0
    old = {k: v for k, v in d.items() if k != "transforms"}
    with pytest.raises(ValueError):
        FeatureNormaliser.from_dict({**old, "version": "median-iqr-v1"})


@needs_inputs
def test_log1p_keeps_42_producer_static_counts_inside_the_clip() -> None:
    """On the loaded 42-producer atlas worlds (5.1's static-check path), the static config counts
    stay strictly inside +-5 under a normaliser fitted on the 12-producer atlas train worlds'
    products; untransformed, E = 14 would sit at the clip (z = (14 - 4) / 2 = 5)."""
    from tokenomics.eval.teacher_audit import load_atlas
    from tokenomics.eval.world_space import config_from_record

    atlas = load_atlas(ATLAS_V2)
    c12 = [config_from_record(c.record) for c in ow.atlas_candidates(atlas)]
    psi = []
    for cfg in c12:
        vals = ow.static_psi_values(cfg)
        for k in range(len(vals["product_depth"])):
            row = np.zeros(N_PSI)
            for name in ow.STATIC_PSI:
                row[PSI_FEATURES.index(name)] = vals[name][k]
            psi.append(row)
    raw = np.asarray(psi)
    rows = RawRows()
    rows.phi.append(np.zeros((1, N_PHI)))
    rows.psi.append(transform_psi(raw))  # fit rows carry the transformed values
    nm = FeatureNormaliser.fit(rows)
    configs42 = ow.reads_42_configs(atlas)
    assert len(configs42) >= 13
    static = ow.static_psi_check(nm, configs42)
    assert len(static) == len(configs42) * len(ow.STATIC_PSI)
    for s in static:
        assert s["at_clip"] == 0.0 and abs(s["z_max"]) < 5.0 and abs(s["z_min"]) < 5.0, s
    e = [s for s in static if s["feature"] == "eligible_producer_count"]
    assert max(s["raw_max"] for s in e) == 14.0 and {s["transform"] for s in e} == {"log1p"}
    j = PSI_FEATURES.index("eligible_producer_count")
    untransformed = FeatureNormaliser.fit(RawRows([np.zeros((1, N_PHI))], [raw]))
    assert (14.0 - untransformed.psi_median[j]) / untransformed.psi_scale[j] >= 5.0


# --- inputs, jobs, CRN ---


@needs_inputs
def test_inputs_minibatches_and_pair_crn(bar_dir: Path, tmp_path: Path) -> None:
    """Members of a pair (and the center) see identical world-episodes and keys; generation g's
    minibatch is schedule.json's for every member; jobs go longest first."""
    inp = es.load_inputs(bar_dir)
    sched = json.loads((bar_dir / "schedule.json").read_text())
    pin = {"path": str(ORDERER), "sha256": "x"}
    for g in (1, 2, 3):
        mb = es.minibatch_for(inp, 0, g, int(sched["minibatch_size"]))
        assert mb == [(d["world"], d["episode"]) for d in sched["minibatches"]["0"][g - 1]]
        eps = es.noise(0, g, 3, 44)
        jobs = es.generation_jobs(inp, pin, "A1", es.member_thetas(np.zeros(44), eps,
                                                                   ARMS["A1"].sigma()), mb, g)
        seen: dict[str, set] = {}
        for j in jobs:
            seen.setdefault(j["member"], set()).add((j["world"], j["episode"], j["replicate"],
                                                     j["world_seed"], json.dumps(j["generator"],
                                                                                 sort_keys=True)))
        assert len(seen) == 7 and len({frozenset(v) for v in seen.values()}) == 1
        assert {(w, e) for w, e, *_ in next(iter(seen.values()))} == set(mb)
        assert all(j["replicate"] == ow.TRAIN_REPLICATE for j in jobs)
    # a tampered schedule is refused against the recorded stream
    bad = tmp_path / "bad"
    shutil.copytree(bar_dir, bad)
    doc = json.loads((bad / "schedule.json").read_text())
    doc["minibatches"]["0"][0] = doc["minibatches"]["0"][0][::-1]
    doc["content_hash"] = ow.content_hash(doc)
    (bad / "schedule.json").write_text(json.dumps(doc))
    tampered = es.load_inputs(bad)
    if len(doc["minibatches"]["0"][0]) > 1:
        with pytest.raises(ValueError, match="recorded stream"):
            es.minibatch_for(tampered, 0, 1, int(doc["minibatch_size"]))
    jobs = [{"cost": c, "i": i} for i, c in enumerate([1.0, 3.0, 2.0, 3.0])]
    assert [j["i"] for j in es.longest_first(jobs)] == [1, 3, 2, 0]


def test_size_refusals() -> None:
    assert es.size_problems({"a": 12}) == []
    assert "training size is 12" in es.size_problems({"a": 42})[0]
    assert "> 42" in es.size_problems({"a": 120})[0]


@needs_inputs
def test_training_size_other_than_12_refused(bar_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    shutil.copytree(bar_dir, bad)
    doc = json.loads((bad / "training_worlds.json").read_text())
    doc["worlds"][0]["record"]["size"] = 42
    doc["content_hash"] = ow.content_hash(doc)
    (bad / "training_worlds.json").write_text(json.dumps(doc))
    nd = json.loads((bad / "normaliser.json").read_text())
    nd["training_worlds_sha256"] = doc["content_hash"]
    (bad / "normaliser.json").write_text(json.dumps(nd))
    with pytest.raises(ValueError, match="training size is 12"):
        es.load_inputs(bad)


# --- driver: fake jobs (resume equivalence, non-finite members) ---


def _fake(bar_dir: Path, nonfinite: set[str] = frozenset()) -> Callable[[dict], dict[str, str]]:
    """Synthetic rows: units a smooth function of theta, the bar's digests, A0 = the bar's A0;
    members in `nonfinite` raise NonFiniteChoice."""
    bar = es.read_bar(bar_dir / "bar.csv")

    def run(job: dict) -> dict[str, str]:
        b = bar[(job["world"], job["episode"])]
        row = {c: "" for c in es.ROW_COLUMNS}
        row.update({"world": job["world"], "size": str(job["size"]), "regime": job["regime"],
                    "episode": str(job["episode"]), "subject": job["subject"],
                    "replicate": str(job["replicate"]), "job_id": job["job_id"],
                    "kind": job["kind"], "arm": job["arm"], "generation": job["generation"],
                    "member": job["member"], "sign": str(job["sign"]),
                    "pair": str(job["pair"]), "batch": str(job["batch"]),
                    "theta_sha256": job["theta_sha256"], "ordering_decode": "choice-v1",
                    "episode_seconds": "0.01", "job_seconds": "0.01"})
        if job["member"] in nonfinite:
            row.update({"nonfinite": "true", "nonfinite_message": "forced"})
            return row
        th = np.asarray(job["theta"])
        units = b.a0_units if job["arm"] == "A0" else (
            b.a0_units + 10.0 * float(np.tanh(th.sum())) - float(np.sum(th ** 2))
            + 0.1 * job["replicate"])
        row.update({"units": repr(float(units)), "dead": "false", "r_start": "1.0",
                    "r_end": "1.0", "consumer_fill_rate": json.dumps({"c0": 1.0}),
                    "holding_units_per_cycle": "0.0", "demand_digest": b.demand_digest,
                    "shock_digest": b.shock_digest, "nonfinite": "false"})
        return row
    return run


def _train(bar_dir: Path, out: Path, gens: int, **kw: object) -> int:
    args: dict = dict(bar_dir=bar_dir, out_dir=out, orderer_ckpt=str(ORDERER), arm="A1",
                      opt_seed=0, generations=gens, pairs=2, monitor_every=1, dry_run=True,
                      run_job_fn=_fake(bar_dir), log=QUIET)
    args.update(kw)
    return es.run_train(**args)


@needs_inputs
def test_resume_equivalence_2_plus_1_equals_3(bar_dir: Path, tmp_path: Path) -> None:
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    assert _train(bar_dir, a, 2) == 0
    assert _train(bar_dir, b, 3, resume_from=str(es.checkpoint_path(a, 2))) == 0
    assert _train(bar_dir, c, 3) == 0
    kb, kc = es.load_checkpoint(b), es.load_checkpoint(c)
    assert kb["generation"] == kc["generation"] == 3
    assert kb["theta"] == kc["theta"] and kb["adam"] == kc["adam"]
    strip = [{k: v for k, v in h.items() if k != "seconds"} for h in kc["history"]]
    assert [{k: v for k, v in h.items() if k != "seconds"} for h in kb["history"]] == strip
    assert kb["monitor"] == kc["monitor"] and len(kc["monitor"]) == 4  # g = 0, 1, 2, 3
    assert not np.array_equal(np.asarray(kc["theta"]), np.zeros(44))  # it moved
    # the same dir resumes with nothing to play; other pins are refused
    rows = (c / "members.csv").read_text()
    assert _train(bar_dir, c, 3) == 0 and (c / "members.csv").read_text() == rows
    with pytest.raises(ValueError, match="different pins"):
        _train(bar_dir, c, 3, opt_seed=1)
    with pytest.raises(ValueError, match="other training pins"):
        _train(bar_dir, tmp_path / "d", 3, opt_seed=1,
               resume_from=str(es.checkpoint_path(a, 2)))
    with pytest.raises(ValueError, match="other training pins"):
        _train(bar_dir, tmp_path / "e", 3, pairs=3, resume_from=str(a))
    # a mid-generation stop: drop generation 3's checkpoint and half its rows, then resume
    d = tmp_path / "f"
    shutil.copytree(c, d)
    es.checkpoint_path(d, 3).unlink()
    lines = (d / "members.csv").read_text().splitlines(keepends=True)
    keep = [x for x in lines if ":g3:" not in x] + [x for x in lines if ":g3:" in x][:3]
    (d / "members.csv").write_text("".join(keep))
    played: list[str] = []
    fake = _fake(bar_dir)
    assert _train(bar_dir, d, 3, run_job_fn=lambda j: (played.append(j["job_id"]),
                                                       fake(j))[1]) == 0
    assert es.load_checkpoint(d)["theta"] == kc["theta"]
    assert len(played) == 5 * 2 - 3  # the rest of generation 3 (its monitor rows are recorded)
    assert [m["generation"] for m in es.load_checkpoint(d)["monitor"]] == [0, 1, 2, 3]
    fit = list(csv.DictReader((d / "member_fitness.csv").open()))
    assert sorted(int(r["generation"]) for r in fit) == [1] * 5 + [2] * 5 + [3] * 5


@needs_inputs
def test_nonfinite_member_ranked_worst_and_run_continues(bar_dir: Path, tmp_path: Path) -> None:
    logs: list[str] = []
    out = tmp_path / "nf"
    assert _train(bar_dir, out, 2, run_job_fn=_fake(bar_dir, {"+01"}), log=logs.append) == 0
    assert (out / "contract_failures.csv").read_text().count("\n") == 1  # header only
    fit = list(csv.DictReader((out / "member_fitness.csv").open()))
    nf = [r for r in fit if r["member"] == "+01"]
    assert len(nf) == 2 and all(r["nonfinite"] == "True" and r["fitness"] == "" for r in nf)
    assert all(float(r["rank"]) == -0.5 for r in nf)
    assert any("NONFINITE" in m and "ranked worst" in m for m in logs)
    assert json.loads((out / "spec.json").read_text())["status"] == "complete"


@needs_inputs
def test_probe_fake_recomputes_and_tolerates_the_bar_subdir(bar_dir: Path,
                                                            tmp_path: Path) -> None:
    """The probe's out dir holds the bar run as `bar/` (EXP-0058); the cosine recomputes from
    probe_fitness.csv; a rerun resumes with nothing to play."""
    out = tmp_path / "exp0058"
    shutil.copytree(bar_dir, out / "bar")
    code = es.run_probe(bar_dir=out / "bar", out_dir=out, orderer_ckpt=str(ORDERER),
                        pairs=3, dry_run=True, run_job_fn=_fake(out / "bar"), log=QUIET)
    assert code == 0
    doc = json.loads((out / "probe.json").read_text())
    fit = list(csv.DictReader((out / "probe_fitness.csv").open()))
    assert len(fit) == 2 * 2 * 6
    again = es.probe_from_rows(fit, ["A1", "A2"], 3)
    assert json.loads(json.dumps(again)) == {k: doc[k] for k in ("arms", "reading")}
    assert doc["reading"]["reading"] in ("signal", "weak", "no_signal")
    assert (out / "bar" / "spec.json").exists()
    episodes = (out / "probe_episodes.csv").read_text()
    assert es.run_probe(bar_dir=out / "bar", out_dir=out, orderer_ckpt=str(ORDERER), pairs=3,
                        dry_run=True, run_job_fn=_fake(out / "bar"), log=QUIET) == 0
    assert (out / "probe_episodes.csv").read_text() == episodes
    with pytest.raises(ValueError, match="must differ"):
        es.run_probe(bar_dir=out / "bar", out_dir=out / "bar", orderer_ckpt=str(ORDERER),
                     dry_run=True, log=QUIET)


@needs_inputs
def test_real_run_refusals(bar_dir: Path, tmp_path: Path) -> None:
    prereg = tmp_path / "prereg.json"
    doc = _frozen()
    doc["sigma"]["w"] = 0.1  # an unregistered change
    prereg.write_text(json.dumps(doc))
    with pytest.raises(ValueError) as exc:
        es.run_train(bar_dir=bar_dir, out_dir=tmp_path / "real", orderer_ckpt=str(ORDERER),
                     arm="A1", opt_seed=0, generations=20, prereg_path=str(prereg), log=QUIET)
    msg = str(exc.value)
    for want in ("not a valid ES training run", "dry run", "horizon 12", "source commit",
                 "sigma.w", "no probe reading"):
        assert want in msg, want
    with pytest.raises(ValueError, match="not a valid ES probe run"):
        es.run_probe(bar_dir=bar_dir, out_dir=tmp_path / "probe", orderer_ckpt=str(ORDERER),
                     log=QUIET)
    with pytest.raises(ValueError, match="trainable arms"):
        _train(bar_dir, tmp_path / "a0", 1, arm="A0")


# --- the real worker: a 2-world, 2-generation, 2-pair horizon-12 dry run ---


@needs_inputs
def test_dry_run_end_to_end(bar_dir: Path, tmp_path: Path,
                            monkeypatch: pytest.MonkeyPatch) -> None:
    keys: list[tuple[str, tuple]] = []
    real_set = es.set_draw_key

    def spy(env: object, key: object) -> None:
        keys.append((current["member"], (key.world, key.episode, key.replicate)))  # type: ignore
        real_set(env, key)  # type: ignore[arg-type]

    current: dict = {}
    real_run = es.run_es_job

    def run(job: dict) -> dict[str, str]:
        current.update(job)
        return real_run(job)

    monkeypatch.setattr(es, "set_draw_key", spy)
    monkeypatch.setattr(es, "run_es_job", run)  # the CLI looks it up at call time
    out = tmp_path / "EXP-train"
    started = time.perf_counter()
    code = es.main(["--bar-dir", str(bar_dir), "--orderer-ckpt", str(ORDERER), "--out-dir",
                    str(out), "--arm", "A1", "--generations", "2", "--pairs", "2",
                    "--monitor-every", "2", "--workers", "1", "--dry-run"])
    elapsed = time.perf_counter() - started
    assert code == 0
    for f in ("spec.json", "members.csv", "monitor.csv", "member_fitness.csv",
              "generations.csv", "monitor_summary.csv", "contract_failures.csv",
              "candidate.json", "RESULT.md"):
        assert (out / f).exists(), f
    assert (out / "contract_failures.csv").read_text().count("\n") == 1
    spec = json.loads((out / "spec.json").read_text())
    assert spec["status"] == "complete" and spec["kind"] == "dry-run"
    tp = spec["pins"]["training"]
    assert tp["ordering_decode"] == "choice-v1"  # decode pins merged
    assert tp["per_order"]["normaliser_sha256"] == tp["inputs"]["normaliser_sha256"]
    assert tp["inputs"]["normaliser_sha256"] == json.loads(
        (bar_dir / "normaliser.json").read_text())["sha256"]
    members = list(csv.DictReader((out / "members.csv").open()))
    sched = json.loads((bar_dir / "schedule.json").read_text())
    for g in (1, 2):
        rows = [r for r in members if r["generation"] == str(g)]
        mb = {(d["world"], str(d["episode"])) for d in sched["minibatches"]["0"][g - 1]}
        assert len({r["member"] for r in rows}) == 5
        for m in {r["member"] for r in rows}:
            assert {(r["world"], r["episode"]) for r in rows if r["member"] == m} == mb
        assert {r["ordering_decode"] for r in rows} == {"choice-v1"}
        assert len({(r["world"], r["episode"], r["demand_digest"], r["shock_digest"])
                    for r in rows}) == len(mb)  # every member saw the same digests
    # keys: a pair's members (and the center) were keyed identically, never by the member
    by_member: dict[str, list] = {}
    for m, k in keys:
        by_member.setdefault(m, []).append(k)
    gen_keys = {m: sorted(v) for m, v in by_member.items() if m not in ("A0",)}
    assert gen_keys["+00"] == gen_keys["-00"] and gen_keys["+01"] == gen_keys["-01"]
    # the monitor plays the center and A0 at replicates 0 and 1 (A0 once per run dir)
    mon = list(csv.DictReader((out / "monitor.csv").open()))
    mw = len(sched["monitor"]["world_episodes"])
    assert {(r["member"], r["generation"], r["replicate"]) for r in mon} == {
        (m, g, r) for m, g in (("c", "0"), ("c", "2"), ("A0", "a0")) for r in ("0", "1")}
    assert len(mon) == 3 * 2 * mw
    # A0's replicate-0 monitor units reproduce the bar; the centre at theta = 0 equals A0
    bar = es.read_bar(bar_dir / "bar.csv")
    for r in mon:
        if r["member"] == "A0" and r["replicate"] == "0":
            assert float(r["units"]) == bar[(r["world"], int(r["episode"]))].a0_units
    a0 = {(r["world"], r["episode"], r["replicate"]): r["units"] for r in mon
          if r["member"] == "A0"}
    c0 = {(r["world"], r["episode"], r["replicate"]): r["units"] for r in mon
          if r["member"] == "c" and r["generation"] == "0"}
    assert a0 == c0
    # a checkpoint per generation; the candidate is a loadable choice-v1 composite
    assert [p.name for p in sorted((out / "checkpoints").glob("*.json"))] == [
        "gen-0000.json", "gen-0001.json", "gen-0002.json"]
    from tokenomics.train.per_order_scorer import load_composite, per_order_env

    inp = es.load_inputs(bar_dir)
    w, e = inp.world_episodes[0]
    job = es.make_job(inp, {"path": str(ORDERER), "sha256": "x"}, kind="member", arm="A1",
                      theta=np.zeros(44), generation=0, member="c", world=w, episode=e,
                      replicate=0)
    pol = load_composite(out / "candidate.json", ORDERER, per_order_env(job_config(job), 2))
    assert np.array_equal(pol.theta, np.asarray(es.load_checkpoint(out)["theta"]))
    assert elapsed < 300, elapsed


@needs_inputs
def test_worker_turns_nonfinite_choice_into_a_ranked_row(bar_dir: Path) -> None:
    """The real worker catches `NonFiniteChoice` (theta_tau = -800 -> tau = 0) as a row with
    `nonfinite` true, never a contract failure."""
    inp = es.load_inputs(bar_dir)
    w, e = inp.world_episodes[0]
    theta = np.zeros(44)
    theta[43] = -800.0
    from tokenomics.eval.provenance import sha256_file

    job = es.make_job(inp, {"path": str(ORDERER), "sha256": sha256_file(str(ORDERER))},
                      kind=es.MEMBER, arm="A1", theta=theta, generation=1, member="+00",
                      world=w, episode=e, replicate=0, sign=1, pair=0)
    d = es.run_es_job(job)
    assert "scope" not in d and d["nonfinite"] == "true" and "temperature" in \
        d["nonfinite_message"]
    s = es.score_member([d], inp.bar)
    assert s.nonfinite and s.fitness is None


@needs_inputs
def test_bar_fit_rows_carry_the_transformed_count(bar_dir: Path) -> None:
    """The policy transforms before its sink: A0's fit rows hold log1p(E), never raw E."""
    j = PSI_FEATURES.index("eligible_producer_count")
    seen = set()
    for f in (bar_dir / ow.FIT_DIR).glob("*.npz"):
        with np.load(f) as z:
            col = np.asarray(z["psi"])[:, j]
        e = np.expm1(col)
        assert np.allclose(e, np.round(e)) and np.all(np.round(e) >= 1)
        seen |= set(np.round(e).astype(int).tolist())
    assert seen and max(seen) <= 12


@needs_inputs
def test_digest_mismatch_and_a0_reproduction_invalidate(bar_dir: Path, tmp_path: Path) -> None:
    fake = _fake(bar_dir)

    def wrong_digest(job: dict) -> dict[str, str]:
        d = fake(job)
        if job["member"] == "-00":
            d["demand_digest"] = "0" * 16
        return d

    out = tmp_path / "digest"
    assert _train(bar_dir, out, 1, run_job_fn=wrong_digest) == es.CONTRACT_INVALID
    assert json.loads((out / "spec.json").read_text())["status"] == "invalid"
    fails = list(csv.DictReader((out / "contract_failures.csv").open()))
    assert [f["check"] for f in fails] == ["digest_pairing"]

    def a0_off(job: dict) -> dict[str, str]:
        d = fake(job)
        if job["arm"] == "A0":
            d["units"] = repr(float(d["units"]) + 1.0)
        return d

    # a rerun on the same dir stays invalid: the earlier session's row is never forgotten
    assert _train(bar_dir, tmp_path / "digest", 1) == es.CONTRACT_INVALID
    out = tmp_path / "a0"
    assert _train(bar_dir, out, 1, run_job_fn=a0_off) == es.CONTRACT_INVALID
    fails = list(csv.DictReader((out / "contract_failures.csv").open()))
    assert [f["check"] for f in fails] == [es.A0_REPRODUCTION]


# --- review round 1 ---


@needs_inputs
@pytest.mark.parametrize("p", [0.0, 0.02, 0.1, 0.5, 0.9])
def test_log1p_floor_on_decision_shaped_rows(p: float) -> None:
    """Concern 1: the real fit runs on one psi row per decision, whose E = 2 share p is not
    fixed. With the registered scale floor, E = 7 / 14 stay strictly inside +-5 on every loaded
    42-producer atlas world whatever p is."""
    from tokenomics.eval.teacher_audit import load_atlas

    j = PSI_FEATURES.index("eligible_producer_count")
    n = 1000
    raw = np.zeros((n, N_PSI))
    raw[:, j] = np.where(np.arange(n) < round(p * n), 2.0, 4.0)
    nm = FeatureNormaliser.fit(RawRows([np.zeros((1, N_PHI))], [transform_psi(raw)]))
    assert nm.psi_scale[j] >= np.log1p(4.0) - np.log1p(2.0) - 1e-15
    static = ow.static_psi_check(nm, ow.reads_42_configs(load_atlas(ATLAS_V2)))
    e = [s for s in static if s["feature"] == "eligible_producer_count"]
    assert e and max(s["raw_max"] for s in e) == 14.0
    assert all(s["at_clip"] == 0.0 and abs(s["z_max"]) < 5.0 for s in e)
    # without the floor (the plain zero-spread rule) p = 0.02 / 0.1 / 0.9 would pin E = 14
    from tokenomics.train.per_order_scorer import FeatureNormaliser as FN

    med, scale = FN._column_stats(transform_psi(raw))
    z = (np.log1p(14.0) - med[j]) / scale[j]
    assert (z >= 5.0) == (p in (0.02, 0.1, 0.9))


def test_scale_floor_recorded_and_refused_below() -> None:
    from tokenomics.train.per_order_scorer import SCALE_FLOORS

    assert SCALE_FLOORS == {"psi.eligible_producer_count": np.log1p(4.0) - np.log1p(2.0)}
    d = FeatureNormaliser.identity().to_dict()
    assert d["scale_floors"] == SCALE_FLOORS
    j = PSI_FEATURES.index("eligible_producer_count")
    low = dict(d, psi_scale=[0.1 if k == j else v for k, v in enumerate(d["psi_scale"])])
    with pytest.raises(ValueError, match="below its registered floor"):
        FeatureNormaliser.from_dict(low)
    with pytest.raises(ValueError, match="scale floors"):
        FeatureNormaliser.from_dict(dict(d, scale_floors={}))
    # the 5.1 report names the floor as the column's rule
    raw = np.zeros((100, N_PSI))
    raw[:, j] = 4.0
    raw[:5, j] = 2.0  # 5 % on E = 2: std ~ 0.11 < floor
    rows = RawRows([np.zeros((3, N_PHI))], [transform_psi(raw)])
    nm, report = ow.fit_normaliser(rows)
    r = next(x for x in report if x["block"] == "psi" and x["index"] == j)
    assert r["rule"] == "floor" and r["scale"] == SCALE_FLOORS["psi.eligible_producer_count"]


@needs_inputs
def test_static_clip_flag_stops_a_real_run(bar_dir: Path, tmp_path: Path) -> None:
    """Concern 1 (b): a bar whose normaliser flags a pinned 42-producer static count is refused
    by a real run (registered stop branch, to the supervisor)."""
    assert es.registered_sections()["es"]["static_clip_42_stop"] == es.STATIC_CLIP_STOP
    bad = tmp_path / "bar"
    shutil.copytree(bar_dir, bad)
    nd = json.loads((bad / "normaliser.json").read_text())
    common = dict(prereg=_frozen(), prereg_issues=[], source_commit="a" * 40,
                  orderer_sha=es.ROLE_CHECKPOINTS["orderer"], filters={})
    clean = es.real_run_problems(inp=es.load_inputs(bad), **common)
    assert not any("pins 42-producer" in m for m in clean)
    assert any("bar run's pins differ" in m and "selection.target" in m for m in clean)
    nd["flags"]["static_psi_clip_42"] = ["eligible_producer_count on w42 (1.000)"]
    (bad / "normaliser.json").write_text(json.dumps(nd))
    got = es.real_run_problems(inp=es.load_inputs(bad), **common)
    assert sum("pins 42-producer static counts at the clip" in m for m in got) == 1


def _registered_bar_pins(bar_dir: Path) -> dict:
    """The dry-run bar's pins with every registered D6 field set to its registered value: a
    bar pin set a real run accepts."""
    pins = json.loads((bar_dir / "spec.json").read_text())["pins"]
    reg = es.bar_registration()
    pins.update({k: copy.deepcopy(v) for k, v in reg.items() if k != "replicates"})
    return pins


@needs_inputs
def test_worlds_settings_registered_and_compared(bar_dir: Path) -> None:
    """Concern 2: D6's training-world / bar settings are a pre-registered section, and a real
    run compares the bar's pins with it: a single-field change is refused."""
    reg = es.registered_sections()["worlds"]
    sel = reg["selection"]
    assert (sel["target"], sel["atlas_worlds"], sel["min_feasible"]) == (64, 23, 50)
    assert (sel["max_replaced_share"], sel["rho_loaded"], sel["saturation_fill"]) == \
        (0.25, 0.28, 0.95)
    assert sel["episodes_per_world"] == 2 and sel["fresh_pool"] == 256
    assert reg["replicates"]["MWW"] == 2 and "shock_reactive" in reg["arms"]
    ok = _registered_bar_pins(bar_dir)
    assert es.bar_pin_problems(ok) == []
    for path, value in ((("selection", "min_feasible"), 49),
                        (("selection", "max_replaced_share"), 0.3),
                        (("selection", "rho_loaded"), 0.3),
                        (("selection", "saturation_fill"), 0.9),
                        (("selection", "target"), 60),
                        (("selection", "episodes_per_world"), 3),
                        (("selection", "fresh_pool"), 128),
                        (("selection", "fresh_atlas"), "other"),
                        (("guards", "x_rundown"), 0.7),
                        (("schedule", "monitor"), 8), (("horizon",), 50),
                        (("roster_version",), "x")):
        pins = copy.deepcopy(ok)
        node = pins
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        got = es.bar_pin_problems(pins)
        assert len(got) == 1 and ".".join(path) in got[0], (path, got)
    pins = copy.deepcopy(ok)
    pins["arms"] = pins["arms"][:-2] + pins["arms"][-1:]
    assert "arms" in es.bar_pin_problems(pins)[0]
    pins = copy.deepcopy(ok)
    next(a for a in pins["arm_table"] if a["arm"] == "MWW")["replicates"] = 1
    assert "replicates.MWW" in es.bar_pin_problems(pins)[0]
    # the dry-run bar itself is refused on its filters
    dry = json.loads((bar_dir / "spec.json").read_text())["pins"]
    assert "selection.target" in es.bar_pin_problems(dry)[0]
    # and the pre-registration's worlds section is enforced like any other
    d = _frozen()
    d["worlds"]["selection"]["min_feasible"] = 40
    got = es.prereg_problems(d, **TRAIN_OK)
    assert len(got) == 1 and "worlds.selection.min_feasible" in got[0]


def test_adam_bias_corrected_steps() -> None:
    """Concern 3 (M4): two Adam steps equal an independent bias-corrected computation; the first
    step moves every coordinate by lr (|m_hat / sqrt(v_hat)| = 1)."""
    theta = np.array([0.5, -1.0, 2.0])
    g1, g2 = np.array([1.0, -3.0, 0.5]), np.array([-2.0, 1.0, 0.0])
    st = es.AdamState.zeros(3)
    t1, st = es.adam_ascend(theta, g1, st)
    lr, b1, b2, eps, wd = 0.02, 0.9, 0.999, 1e-8, 0.005
    d1 = -g1 + wd * theta
    m, v = (1 - b1) * d1, (1 - b2) * d1 ** 2
    want1 = theta - lr * (m / (1 - b1)) / (np.sqrt(v / (1 - b2)) + eps)
    assert np.allclose(t1, want1, rtol=0, atol=1e-15)
    assert np.allclose(np.abs(t1 - theta), lr, atol=1e-8)
    t2, st = es.adam_ascend(t1, g2, st)
    d2 = -g2 + wd * t1
    m, v = b1 * m + (1 - b1) * d2, b2 * v + (1 - b2) * d2 ** 2
    want2 = t1 - lr * (m / (1 - b1 ** 2)) / (np.sqrt(v / (1 - b2 ** 2)) + eps)
    assert np.allclose(t2, want2, rtol=0, atol=1e-15) and st.t == 2


def test_probe_reading_gates_training() -> None:
    """Concern 3 (M13, M5, M12): training needs a `signal` / `weak` probe; `weak` -> minibatch
    24, `signal` -> 12; A2 vs A2-R2 follows the probe's `a2_arm`; a dry-run probe is refused."""
    def doc(reading: str, a2_arm: str = "A2", **kw: object) -> dict:
        r = es.probe_reading({"A1": {"signal": 0.5, "weak": 0.2, "no_signal": 0.0}[reading],
                              "A2": 0.5 if a2_arm == "A2" else 0.0})
        assert r["reading"] == reading and r["a2_arm"] == (a2_arm if reading != "no_signal"
                                                           else "A2")
        return {"status": "complete", "dry_run": False, "reading": r, **kw}

    P = es._probe_doc_problems
    assert P(doc("signal"), "A1", 12) == [] and P(doc("weak"), "A1", 24) == []
    assert len(P(doc("no_signal"), "A1", 12)) == 1 and "no_signal" in P(
        doc("no_signal"), "A1", 12)[0]
    assert "minibatch 12" in P(doc("weak"), "A1", 12)[0]
    assert "minibatch 24" in P(doc("signal"), "A1", 24)[0]
    assert P(doc("signal", "A2-R2"), "A2-R2", 12) == []
    assert "A2-R2" in P(doc("signal", "A2-R2"), "A2", 12)[0]
    assert "A2" in P(doc("signal", "A2"), "A2-R2", 12)[0]
    assert P(doc("signal", "A2-R2"), "A1", 12) == []
    assert "not a complete real probe" in P(doc("signal", dry_run=True), "A1", 12)[0]


def test_monitor_summary_mean_and_window() -> None:
    """Concern 3 (M16, M17): the center's monitor mean is over both replicates; improvement_10 is
    against generation g - 10, not generation 0."""
    rows = [{"units": u, "nonfinite": "false", "replicate": r}
            for u, r in (("10", "0"), ("30", "1"), ("20", "0"), ("40", "1"))]
    hist = [{"generation": 0, "center_units": 0.0}, {"generation": 10, "center_units": 20.0}]
    s = es.monitor_summary(20, rows, 15.0, hist)
    assert s["center_units"] == 25.0 and s["g_tr_raw"] == 10.0
    assert s["improvement_10"] == 5.0
    assert es.monitor_summary(10, rows, 15.0, hist[:1])["improvement_10"] == 25.0
    assert es.monitor_summary(5, rows, 15.0, hist)["improvement_10"] is None


@needs_inputs
def test_member_noise_keyed_by_generation(bar_dir: Path, tmp_path: Path) -> None:
    """Concern 3 (M19): generation g's members are theta_g-1 +- sigma * noise(seed, g)."""
    seen: dict[tuple[str, str], np.ndarray] = {}
    fake = _fake(bar_dir)

    def spy(job: dict) -> dict[str, str]:
        if job["kind"] == es.MEMBER:
            seen[(job["generation"], job["member"])] = np.asarray(job["theta"])
        return fake(job)

    assert _train(bar_dir, tmp_path / "n", 2, run_job_fn=spy) == 0
    sigma = ARMS["A1"].sigma()
    for g in (1, 2):
        eps = es.noise(0, g, 2, 44)
        for lab, sign, i in es.member_labels(2):
            assert np.allclose(seen[(str(g), lab)] - seen[(str(g), "c")], sign * sigma * eps[i],
                               rtol=0, atol=1e-15)
    assert not np.allclose(es.noise(0, 1, 2, 44), es.noise(0, 2, 2, 44))


@needs_inputs
def test_resume_guards(bar_dir: Path, tmp_path: Path) -> None:
    """Concern 3 (M1, M7, M18): a resume across a pre-registration change is refused; a row of an
    earlier session with another theta is refused; resuming at or past the target is refused."""
    p1, p2 = tmp_path / "p1.json", tmp_path / "p2.json"
    p1.write_text(json.dumps(es.draft_prereg()))
    p2.write_text(json.dumps(es.draft_prereg(), indent=2))  # same content, other file hash
    a = tmp_path / "a"
    assert _train(bar_dir, a, 2, prereg_path=str(p1)) == 0
    with pytest.raises(ValueError, match="prereg_sha256"):
        _train(bar_dir, tmp_path / "b", 3, prereg_path=str(p2),
               resume_from=str(es.checkpoint_path(a, 2)))
    assert _train(bar_dir, tmp_path / "b2", 3, prereg_path=str(p1),
                  resume_from=str(es.checkpoint_path(a, 2))) == 0
    for target in (2, 1):
        with pytest.raises(ValueError, match=">= --generations"):
            _train(bar_dir, tmp_path / f"c{target}", target, prereg_path=str(p1),
                   resume_from=str(es.checkpoint_path(a, 2)))
    # an earlier session's row whose theta is not this member's
    es.checkpoint_path(a, 2).unlink()
    rows = list(csv.DictReader((a / "members.csv").open()))
    target = next(r for r in rows if r["generation"] == "2")
    target["theta_sha256"] = "0" * 64
    with (a / "members.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(es.ROW_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="another theta"):
        _train(bar_dir, a, 2, prereg_path=str(p1))
