"""Combined held-out suite (NEXT STEPS stage 4): a *fixed, versioned* seed manifest
pins the base config, the combined perturbation axes, and the eval seeds, so the
held-out evaluation is reproducible and reviewable. The verdict (`beats_baseline`) is
a pure function of two metric dicts, so the release bar — trained reliably beats the
strongest observation-compatible baseline — is unit-testable without training."""

from __future__ import annotations

from pathlib import Path

from tokenomics.domain import load_config
from tokenomics.env.spaces import build_spec

FIX = Path(__file__).parent / "fixtures"


def _dims(cfg):
    sp = build_spec(cfg)
    return (sp.producing_obs_dim, sp.producing_action_dim,
            sp.ordering_obs_dim, sp.ordering_action_dim)


def test_seed_manifest_round_trips_through_json():
    from tokenomics.eval.held_out import SeedManifest

    m = SeedManifest(
        version="heldout-v1",
        base=str(FIX / "competition.json"),
        seeds=(10000, 10001, 10002),
        degradation_factor=1.5,
        quantity_max=3,
        capacity_factor=0.8,
        cost_factor=1.2,
    )
    back = SeedManifest.from_json(m.to_json())
    assert back == m  # exact round-trip -> the manifest is a stable, versioned artifact


def test_seed_manifest_config_applies_combined_axes_and_preserves_dims():
    from tokenomics.eval.held_out import SeedManifest

    base = load_config(FIX / "competition.json")
    m = SeedManifest(
        version="t", base=str(FIX / "competition.json"), seeds=(0,),
        degradation_factor=2.0, quantity_max=2, capacity_factor=0.7, cost_factor=1.5,
    )
    cfg = m.config()
    cfg.validate()
    assert _dims(cfg) == _dims(base)                              # net stays loadable
    assert cfg.producer_configs[0].degradation == round(base.producer_configs[0].degradation * 2.0)
    assert all(c.quantity_max == 2 for c in cfg.consumers)
    assert cfg != base


def test_run_manifest_evaluates_policy_over_manifest_seeds_and_config():
    from tokenomics.eval.baselines import ordering_even_split, producing_stock_aware
    from tokenomics.eval.held_out import SeedManifest, run_manifest

    m = SeedManifest(
        version="t", base=str(FIX / "competition.json"), seeds=(10000, 10001),
        quantity_max=3, capacity_factor=0.8,
    )
    agg = run_manifest(producing_stock_aware(0.6), ordering_even_split, m, horizon=30)
    # returns the standard Aggregated metric dict the verdict reads
    for k in ("return", "consumer_completed_units", "dead_rate"):
        assert k in agg and isinstance(agg[k], tuple)


def test_beats_baseline_passes_when_trained_dominates():
    from tokenomics.eval.held_out import beats_baseline

    trained = {"return": (1300.0, 0.0), "consumer_completed_units": (120.0, 0.0),
               "dead_rate": (0.0, 0.0)}
    baseline = {"return": (1255.0, 0.0), "consumer_completed_units": (100.0, 0.0),
                "dead_rate": (0.0, 0.0)}
    v = beats_baseline(trained, baseline)
    assert v.passed
    assert v.reasons == ()  # nothing to complain about


def test_beats_baseline_fails_and_explains_each_shortfall():
    from tokenomics.eval.held_out import beats_baseline

    # worse return, fewer units, AND more death than the baseline -> three reasons
    trained = {"return": (1000.0, 0.0), "consumer_completed_units": (90.0, 0.0),
               "dead_rate": (0.3, 0.0)}
    baseline = {"return": (1255.0, 0.0), "consumer_completed_units": (100.0, 0.0),
                "dead_rate": (0.0, 0.0)}
    v = beats_baseline(trained, baseline)
    assert not v.passed
    assert len(v.reasons) == 3
    joined = " ".join(v.reasons).lower()
    assert "return" in joined and "unit" in joined and "death" in joined


def test_load_manifest_resolves_relative_base_against_manifest_dir(tmp_path):
    from tokenomics.eval.held_out import SeedManifest, load_manifest

    # a checked-in manifest references its base by a path relative to itself, so the
    # pair is portable regardless of the working directory it is loaded from.
    (tmp_path / "base.json").write_text((FIX / "competition.json").read_text())
    m = SeedManifest(version="v1", base="base.json", seeds=(10000,))
    (tmp_path / "heldout.json").write_text(m.to_json())

    loaded = load_manifest(str(tmp_path / "heldout.json"))
    cfg = loaded.config()  # resolves base.json next to the manifest, not under cwd
    cfg.validate()


def test_shipped_manifest_and_base_are_packaged_resources():
    # The shipped default suite must survive a wheel/sdist install, not just the
    # source checkout: the manifest AND its base config must live inside the package
    # tree as importable resources (and be declared package-data so they ship).
    from importlib.resources import files

    manifests = files("tokenomics.eval").joinpath("manifests")
    assert manifests.joinpath("heldout_v1.json").is_file()
    assert manifests.joinpath("heldout_v1_base.json").is_file()


def test_shipped_default_manifest_loads_and_builds_valid_config():
    from tokenomics.eval.held_out import DEFAULT_MANIFEST, load_manifest

    m = load_manifest(DEFAULT_MANIFEST)
    assert m.version == "heldout-v1"
    assert len(m.seeds) >= 5 and all(s >= 10000 for s in m.seeds)  # held-out range
    cfg = m.config()  # base resolved next to the manifest in the package
    cfg.validate()


def test_beats_baseline_higher_death_alone_fails():
    from tokenomics.eval.held_out import beats_baseline

    # wins on return and units but dies more often -> survival is non-negotiable
    trained = {"return": (1400.0, 0.0), "consumer_completed_units": (130.0, 0.0),
               "dead_rate": (0.1, 0.0)}
    baseline = {"return": (1255.0, 0.0), "consumer_completed_units": (100.0, 0.0),
                "dead_rate": (0.0, 0.0)}
    v = beats_baseline(trained, baseline)
    assert not v.passed
    assert any("death" in r.lower() for r in v.reasons)
