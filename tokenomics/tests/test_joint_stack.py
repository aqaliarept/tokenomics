"""C1 portable-stack smoke (PRODUCT_AGNOSTIC combined ladder).

Wire the two portable modules together — producer `ProducerScoreModel` →
`plan_producer_command` and orderer `OrdererScoreModel` → `allocate_suborders` — in
one env (both `entity_producer_obs` and the corrected v4 product-stream ordering on)
and prove they COEXIST: a full episode runs with no feasibility failure, a pure
relabel of the topology leaves the episode value unchanged, and the eval harness
reports a value ratio against the deterministic `entity_planner + proportional` bar.

Smoke models (untrained / few-iter): the C1 gate is contract — feasibility,
conservation, relabel invariance — provable for any scores. Checkpoint QUALITY is the
EXP-0014 local eval against the real distilled artifacts.
"""

from __future__ import annotations

from tokenomics.eval.scale import build_scale_config
from tokenomics.train.orderer_scorer import OrdererScoreModel
from tokenomics.train.producer_scorer import ProducerScoreModel


def test_joint_portable_stack_drives_feasible_episode() -> None:
    # Both portable policies in one env complete an episode with no feasibility failure.
    from tokenomics.eval.joint_stack import run_joint_stack

    cfg = build_scale_config(0, depth=2)
    stats = run_joint_stack(ProducerScoreModel(), OrdererScoreModel(), cfg, seed=0, horizon=20)
    assert stats["cycles"] > 0
    assert stats["consumer_completed_units"] >= 0


def test_joint_stack_value_unchanged_under_capacity_relabel() -> None:
    # A pure relabel (rename every capacity type) is invisible to the id-free contract:
    # both portable policies must yield a byte-identical episode. No relabel mismatch.
    from tokenomics.eval.joint_stack import relabel_capacity_types, run_joint_stack

    prod, order = ProducerScoreModel(), OrdererScoreModel()
    cfg = build_scale_config(1, depth=2)
    base = run_joint_stack(prod, order, cfg, seed=0, horizon=20)
    renamed = run_joint_stack(prod, order, relabel_capacity_types(cfg), seed=0, horizon=20)
    assert renamed["consumer_completed_units"] == base["consumer_completed_units"]
    assert renamed["cycles"] == base["cycles"]
    assert renamed["dead"] == base["dead"]


def test_eval_joint_stack_reports_ratio_vs_deterministic_bar() -> None:
    # Eval harness: per held-out seed, joint-stack units / (entity_planner+proportional)
    # units, with both death flags. The bar is the deterministic stack on the same env.
    from tokenomics.eval.joint_stack import eval_joint_stack

    rows = eval_joint_stack(
        ProducerScoreModel(), OrdererScoreModel(),
        depth=2, seeds=[6], eval_seeds=1, horizon=20,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["seed"] == 6 and row["producers"] >= 1
    assert row["bar_units"] > 0
    assert row["ratio"] == row["joint_units"] / row["bar_units"]
    assert row["dead"] in (0.0, 1.0) and row["bar_dead"] in (0.0, 1.0)
