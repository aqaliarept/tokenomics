"""Order TTL as a global world setting (add-m4-orderer-redesign task 7.12): the default 2 is
byte-identical to the old constant; other TTLs reach every order through the configuration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pettingzoo.test import api_test

from tokenomics.domain.config import DEFAULT_ORDER_TTL, MAX_ORDER_TTL, load_config
from tokenomics.domain.order import ORDER_TTL, ConsumerRequestRejected, Order, OrderStillProcessing
from tokenomics.domain.requests import ConsumerRequest
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.spaces import build_spec, config_fingerprint
from tokenomics.eval import decoder_check as dc
from tokenomics.eval import routing_audit as ra
from tokenomics.eval.roster import build_roster_v2

TESTS = Path(__file__).resolve().parent
COMPOSITE = TESTS / "fixtures" / "composite.json"
PHASE1 = TESTS.parents[0] / "tokenomics" / "config" / "phase1.json"
REPO = TESTS.parents[1]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"


def _lfs_present(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


def test_default_is_the_old_constant_and_unset() -> None:
    assert DEFAULT_ORDER_TTL == ORDER_TTL == 2 and MAX_ORDER_TTL == 4
    cfg = load_config(PHASE1)
    explicit = replace(cfg, order_ttl=2)
    assert cfg.order_ttl is None and explicit.order_ttl is None and explicit == cfg
    assert cfg.ttl == 2
    assert config_fingerprint(explicit) == config_fingerprint(cfg)
    assert ra.config_digest(explicit) == ra.config_digest(cfg)
    assert build_spec(explicit) == build_spec(cfg)


def test_non_default_ttl_changes_identity_and_shapes() -> None:
    cfg = load_config(PHASE1)
    base = build_spec(cfg)
    assert (base.order_ttl, base.order_lifetime) == (2, 3)
    for ttl in (3, 4):
        other = replace(cfg, order_ttl=ttl)
        spec = build_spec(other)
        assert (spec.order_ttl, spec.order_lifetime) == (ttl, ttl + 1)
        assert spec.k > base.k
        assert config_fingerprint(other) != config_fingerprint(cfg)
        assert ra.config_digest(other) != ra.config_digest(cfg)


@pytest.mark.parametrize("bad", [0, 5, -1])
def test_out_of_range_ttl_refused(bad: int) -> None:
    with pytest.raises(ValueError):
        replace(load_config(PHASE1), order_ttl=bad).validate()


@pytest.mark.parametrize("ttl", [1, 2, 3, 4])
def test_unfilled_order_expires_after_ttl_cycles(ttl: int) -> None:
    req = ConsumerRequest(consumer_id=0, product="p", tokens=10, quantity=1)
    order = Order(1, 10, "p", 1, req, None, True, ttl=ttl)
    for age in range(ttl):
        score, event = order.complete_cycle()
        assert (score, type(event)) == (1, OrderStillProcessing), age
    score, event = order.complete_cycle()
    assert score == 5 and isinstance(event, ConsumerRequestRejected)


def test_system_orders_carry_the_world_ttl() -> None:
    from tokenomics.domain import build_system

    system = build_system(replace(load_config(COMPOSITE), order_ttl=4))
    assert system.order_ttl == 4
    default = build_system(load_config(COMPOSITE))
    assert default.order_ttl == 2


@pytest.mark.parametrize("ttl", [2, 4])
def test_pettingzoo_api_at_ttl(ttl: int) -> None:
    cfg = replace(load_config(COMPOSITE), order_ttl=ttl)
    api_test(TokenomicsAECEnv(config=cfg, horizon=40), num_cycles=20, verbose_progress=False)


def test_job_config_applies_and_caches_the_ttl() -> None:
    job = {"world": "w-ttl-test", "world_seed": 0,
           "generator": {"generator": "sparse_wide_family",
                         "params": {"producers": 12, "regime": "calm", "schedule_length": 7}}}
    default = ra.job_config(job)
    ttl4 = ra.job_config({**job, "order_ttl": 4})
    assert default.ttl == 2 and default.order_ttl is None and ttl4.ttl == 4
    assert replace(ttl4, order_ttl=None) == default
    assert ra.job_config({**job, "order_ttl": 4}) is ttl4
    assert ra.job_config(job) is default


def test_sweep_refuses_other_arms_and_references(tmp_path: Path) -> None:
    for kw in ({"arms": ["TR"]}, {"exp0052_episodes": "x.csv"}):
        with pytest.raises(ValueError, match="TTL sweep"):
            dc.run_decoder_check(atlas_path=ATLAS_V2, out_dir=tmp_path / "o",
                                 orderer_ckpt=str(ORDERER), roster=build_roster_v2(),
                                 order_ttl=4, dry_run=True, **kw)
    with pytest.raises(SystemExit):
        dc.main(["--mode", "arm-compare", "--arm", "utwf", "--order-ttl", "4",
                 "--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER),
                 "--out-dir", str(tmp_path / "a")])


@pytest.mark.skipif(not (_lfs_present(ATLAS_V2) and _lfs_present(ORDERER)),
                    reason="EXP-0052 atlas / EXP-0012 checkpoint absent")
def test_ttl4_sweep_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "EXP-ttl4"
    assert dc.main(["--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER), "--dry-run",
                    "--order-ttl", "4", "--worlds", "ws-atlas-v1-12-0060", "--horizon", "12",
                    "--workers", "1", "--out-dir", str(out)]) == 0
    spec = json.loads((out / "spec.json").read_text())
    assert spec["result"]["order_ttl"] == 4 and spec["result"]["profile"] == "ttl_sweep"
    assert spec["result"]["contract_failures"] == 0
    reading = json.loads((out / "reading.json").read_text())
    assert reading["readings_withheld"] is True and reading["arms"] == list(dc.TTL_SWEEP_ARMS)
    assert (out / "contract_failures.csv").read_text().count("\n") == 1
