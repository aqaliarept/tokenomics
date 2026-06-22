"""Producer-first warm-start gate (ADR-0049 / Phase 1).

The depth-3 wall is producer-side: `clone+base` (cloned producer, baseline orderer)
is the gate column. `swap_gate` runs the 4 swap combos and renders a PASS/FAIL
verdict on that column — `clone+base units >= pass_frac * base+base units` with the
clone's death rate at/below `max_dead`. These tests pin the verdict semantics through
the public result, using cheap deterministic baselines so the math is exact (the
scripted domain is deterministic; stock_aware/proportional add no RNG)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_swap_gate_passes_when_clone_matches_base():
    # Tracer: when the "clone" IS the baseline, clone+base must equal base+base, so a
    # 0.8 bar is cleared and the gate passes. Pins the combo wiring (prod+order order)
    # and the verdict's happy path.
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.warm_start import swap_gate

    cfg = build_scale_config(0, depth=1)
    bp = PRODUCING_BASELINES["stock_aware"]
    bo = ORDERING_BASELINES["proportional"]

    res = swap_gate(bp, bo, bp, bo, config=cfg, swap_seeds=2, horizon=30,
                    pass_frac=0.8, max_dead=0.05)

    assert {r.name for r in res.rows} == {
        "base+base", "clone+clone", "clone+base", "base+clone"
    }
    assert res.clone_base.units == res.base_base.units  # same policies, deterministic
    assert res.clone_base.units > 0
    assert res.passed


def test_swap_gate_fails_when_bar_exceeds_clone():
    # Threshold isolation: clone==base so clone+base == base+base, but a 2× bar can
    # never be met by parity. Pins the units half of the verdict independently of the
    # policies' actual strength.
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.warm_start import swap_gate

    cfg = build_scale_config(0, depth=1)
    bp = PRODUCING_BASELINES["stock_aware"]
    bo = ORDERING_BASELINES["proportional"]

    res = swap_gate(bp, bo, bp, bo, config=cfg, swap_seeds=2, horizon=30,
                    pass_frac=2.0, max_dead=0.05)

    assert not res.passed


def test_swap_gate_flags_a_worse_producer():
    # The gate's purpose: a producer that never restores (noop) starves the chain, so
    # clone+base must fall below base+base and fail the 0.8 bar — while base+clone
    # (baseline producer) stays healthy. Depth-2 so degradation actually bites.
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.warm_start import swap_gate

    cfg = build_scale_config(0, depth=2)
    noop = PRODUCING_BASELINES["noop"]
    bp = PRODUCING_BASELINES["stock_aware"]
    bo = ORDERING_BASELINES["proportional"]

    res = swap_gate(noop, bo, bp, bo, config=cfg, swap_seeds=2, horizon=60,
                    pass_frac=0.8, max_dead=0.05)

    assert res.clone_base.units < res.base_base.units
    assert not res.passed
    # column wiring: clone+base = clone PRODUCER + base orderer (the failing noop),
    # base+clone = base producer + clone orderer (healthy) — so they must not be equal
    # and base+clone clears the failing clone+base.
    assert res.row("base+clone").units > res.clone_base.units


def test_run_warm_start_bc_clones_saves_and_gates(tmp_path):
    # Driver wiring (Phase 1): resolve the expert, clone it, save a PPO-loadable
    # checkpoint, load it back, and return the swap gate. Tiny epochs/seeds keep it a
    # smoke of the plumbing, not a training run.
    from tokenomics.train.smoke_ppo import load_policies
    from tokenomics.train.warm_start import GateResult, run_warm_start

    save = tmp_path / "clone.pt"
    res = run_warm_start(
        depth=1, mode="bc", expert="obs_stock_aware", order_expert="proportional",
        dagger_iters=1, seeds=1, horizon=20, epochs=20, lr=3e-3, seed=0,
        log_std_save=-2.5, save=str(save), swap_seeds=1, pass_frac=0.8,
        max_dead=0.05, log=lambda *a, **k: None,
    )

    assert isinstance(res, GateResult)
    assert save.exists()
    prod_policy, order_policy = load_policies(str(save))  # schema-compatible
    assert callable(prod_policy) and callable(order_policy)


def test_run_warm_start_dagger_with_slot_permutations_trains_and_gates(tmp_path):
    # Phase 2 wiring: `perms>0` expands each training seed into slot-permuted relabelings
    # (different sorted-id -> config maps) and DAgger concatenates their per-config decode
    # masks. The real risk is dim/mask mismatch across relabeled specs; this smokes that
    # the permuted set flows through dagger_producer_first and still yields a gateable,
    # schema-compatible checkpoint. Tiny budget — plumbing, not a training run.
    from tokenomics.train.smoke_ppo import load_policies
    from tokenomics.train.warm_start import GateResult, run_warm_start

    save = tmp_path / "clone_perm.pt"
    res = run_warm_start(
        depth=1, mode="dagger", expert="obs_stock_aware", order_expert="proportional",
        dagger_iters=1, seeds=1, horizon=20, epochs=20, lr=3e-3, seed=0,
        log_std_save=-2.5, save=str(save), train_seeds=1, perms=2, perm_seed=1,
        swap_seeds=1, pass_frac=0.8, max_dead=0.05, log=lambda *a, **k: None,
    )

    assert isinstance(res, GateResult)
    assert save.exists()
    prod_policy, order_policy = load_policies(str(save))
    assert callable(prod_policy) and callable(order_policy)
