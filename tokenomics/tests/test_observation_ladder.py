"""EXP-0038A missing-observation audit: the cumulative deployable-observation ladder over the
EXP-0038 dump. Torch-free measurement; one behavior per test (HANDOVER metrics)."""

from __future__ import annotations

import csv

from tokenomics.eval import observation_ladder as ol


def _write(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _row(label: float, **feats: float) -> dict:
    """A minimal producer-row record: a teacher label plus named feature columns."""
    return {"teacher_label": label, **feats}


def test_alias_spread_ranges_over_groups_seen_at_least_min_count() -> None:
    # Two feature groups keyed on column "a". Group a=0 seen 5x with labels spanning 4 units;
    # group a=1 seen 3x (< min_count) is ignored. Spread is measured only where the block repeats.
    rows = [_row(10.0, a=0), _row(12.0, a=0), _row(14.0, a=0), _row(11.0, a=0), _row(13.0, a=0),
            _row(1.0, a=1), _row(2.0, a=1), _row(9.0, a=1)]
    s = ol.alias_spread(rows, ["a"], min_count=5)
    assert s["n_groups"] == 1
    assert s["median_range"] == 4.0
    assert s["max_range"] == 4.0
    assert s["frac_gt1"] == 1.0


def test_load_dump_joins_order_context_onto_each_producer_row(tmp_path) -> None:
    # One decision (task t, seed 1, cycle 3, order o1) with two eligible producers. The order
    # context lives in decisions.csv; each producer row in producers.csv must inherit it by the
    # (task_id, seed, cycle, order_id) key so a block can key on both producer + order fields.
    d = tmp_path
    _write(str(d / "decisions.csv"), [
        {"task_id": "t", "seed": "1", "cycle": "3", "order_id": "o1", "stream_index": "0",
         "product": "2", "quantity": "7", "order_age": "1", "ttl_remaining": "1",
         "backlog_size": "4", "same_product_backlog": "2"},
    ])
    _write(str(d / "producers.csv"), [
        {"task_id": "t", "seed": "1", "cycle": "3", "order_id": "o1", "producer_id": "p0",
         "eligible": "True", "teacher_label": "12.0", "actor_score": "0.1",
         **{f"feat_{i}": str(i) for i in range(7)}},
        {"task_id": "t", "seed": "1", "cycle": "3", "order_id": "o1", "producer_id": "p1",
         "eligible": "True", "teacher_label": "3.0", "actor_score": "0.2",
         **{f"feat_{i}": str(10 + i) for i in range(7)}},
    ])
    rows = ol.load_dump(str(d))
    assert len(rows) == 2
    r0 = rows[0]
    assert r0["producer_id"] == "p0" and r0["teacher_label"] == 12.0
    assert r0["quantity"] == 7 and r0["product"] == 2 and r0["backlog_size"] == 4
    assert r0["feat_0"] == 0.0 and r0["feat_6"] == 6.0
    # decision key groups the two producer rows of the same order
    assert rows[0]["decision_key"] == rows[1]["decision_key"]


import pytest

DUMP = "/workspaces/tokenomics/experiments/EXP-0038-actor-teacher-error-microscope/dump"


@pytest.mark.skipif(not __import__("os").path.isdir(DUMP), reason="EXP-0038 dump not present")
def test_current_row_reproduces_exp0038_headline_and_order_collapses_it() -> None:
    # Guardrail: current_row alias spread must reproduce the EXP-0038 headline exactly, and adding
    # the deployable +order block must collapse the median label range to 0 (the preliminary
    # finding this audit rests on). Reads the real checked-in dump.
    rows = ol.load_dump(DUMP)
    cur = ol.alias_spread(rows, ol.block_columns("current_row"))
    assert cur["n_groups"] == 902
    assert cur["median_range"] == 10.0
    assert round(cur["mean_range"], 2) == 14.31
    assert cur["max_range"] == 90.0
    assert round(cur["frac_gt1"], 3) == 0.669

    order = ol.alias_spread(rows, ol.block_columns("+order"))
    assert order["median_range"] == 0.0
    assert order["frac_gt1"] < 0.3


def _decision(dkey, feat0s, *, quantity=1, label_fn=None):
    """A synthetic routing decision: several eligible producers differing only in feat_0, sharing
    one order context. label_fn(feat0, quantity) sets the teacher label so a test can encode a
    known dependence (e.g. an interaction)."""
    rows = []
    for j, f0 in enumerate(feat0s):
        row = {"decision_key": dkey, "producer_id": f"p{j}", "task_id": dkey[0], "seed": dkey[1],
               "teacher_label": float(label_fn(f0, quantity)),
               **{f"feat_{i}": 0.0 for i in range(7)},
               **{c: 0 for c in ol.ORDER_COLS + ol.QUEUE_COLS}}
        row["feat_0"] = float(f0)
        row["quantity"] = quantity
        rows.append(row)
    return rows


def test_probe_fold_recovers_additive_label_from_current_row() -> None:
    # Label = 2*feat_0 (pure per-producer signal). A ridge probe on current_row must rank producers
    # right on held-out data: ~zero regret, high variance explained. Plumbing check.
    train, test = [], []
    for c in range(30):
        train += _decision(("t", "1", c), [0.1 * (c % 5), 0.2 + 0.1 * (c % 3)],
                           label_fn=lambda f, q: 2 * f)
    for c in range(10):
        test += _decision(("t", "2", c), [0.15 * (c % 4), 0.05 + 0.1 * (c % 3)],
                          label_fn=lambda f, q: 2 * f)
    m = ol.probe_fold(train, test, ol.block_columns("current_row"), kind="ridge")
    assert m["mean_regret"] < 0.05
    # centered within-decision r2 with ridge shrinkage: recovery is clear (ranking exact, regret ~0)
    assert m["r2"] > 0.7


def test_probe_fold_needs_order_interaction_to_re_rank_when_best_flips_with_quantity() -> None:
    # The best producer FLIPS with order quantity: label = feat_0 for small orders (q=1, high feat_0
    # wins) but 1-feat_0 for large orders (q=5, low feat_0 wins). quantity is constant across a
    # decision's producers, so current_row (feat_0 only) is forced to one ranking and is wrong half
    # the time -> high held-out regret. +order supplies feat_0 x quantity interactions that recover
    # the flip -> regret near 0. This is the licensing mechanism: order context must re-rank WITHIN
    # a decision, not just shift the level.
    import random
    rng = random.Random(0)
    flip = lambda f, q: f if q == 1 else 1.0 - f
    def make(seed, n):
        out = []
        for c in range(n):
            q = rng.choice([1, 5])
            out += _decision((f"t", seed, c), [rng.uniform(0, 1), rng.uniform(0, 1)],
                             quantity=q, label_fn=flip)
        return out
    train = make("1", 80) + make("2", 80)
    test = make("3", 60)
    cur = ol.probe_fold(train, test, ol.block_columns("current_row"), kind="ridge")
    order = ol.probe_fold(train, test, ol.block_columns("+order"), kind="ridge")
    assert cur["mean_regret"] > 0.15          # current_row cannot see the flip
    assert order["mean_regret"] < 0.03         # +order interactions recover it
    assert order["mean_regret"] < cur["mean_regret"]


def test_heldout_probe_runs_leave_one_seed_out_per_task() -> None:
    # One task, three seeds; label is additive in feat_0. heldout_probe holds each seed out once,
    # trains on the other two, and returns per-task averaged fold metrics keyed by task_id.
    rows = []
    for seed in ("1", "2", "3"):
        for c in range(20):
            rows += _decision(("ta", seed, c), [0.1 * (c % 5), 0.3 + 0.1 * (c % 4)],
                              label_fn=lambda f, q: 2 * f)
    out = ol.heldout_probe(rows, ol.block_columns("current_row"), kind="ridge")
    assert set(out) == {"ta"}
    assert out["ta"]["n_folds"] == 3
    assert out["ta"]["mean_regret"] < 0.05
    assert out["ta"]["r2"] > 0.9


def test_floor_metrics_scores_random_and_deployed_actor_without_training() -> None:
    # Floors need no fit: 'random' = expected regret of picking a uniform eligible producer;
    # 'actor' = the deployed EXP-0037 residual (actor_score column) argmax. Two decisions, best
    # label 10. Random expected regret = mean shortfall vs a uniform pick; actor picks argmax score.
    rows = [
        {"task_id": "t", "decision_key": ("t", "1", 0), "teacher_label": 10.0, "actor_score": 0.9},
        {"task_id": "t", "decision_key": ("t", "1", 0), "teacher_label": 0.0, "actor_score": 0.1},
        {"task_id": "t", "decision_key": ("t", "1", 1), "teacher_label": 4.0, "actor_score": 0.2},
        {"task_id": "t", "decision_key": ("t", "1", 1), "teacher_label": 6.0, "actor_score": 0.8},
    ]
    rnd = ol.floor_metrics(rows, kind="random")["t"]
    act = ol.floor_metrics(rows, kind="actor")["t"]
    # random: dec0 shortfall vs mean(10,0)=5 -> regret 5; dec1 vs mean(4,6)=5 -> regret 1; mean 3.
    assert abs(rnd["mean_regret"] - 3.0) < 1e-9
    # actor: dec0 picks label 10 (regret 0), dec1 picks label 6 (regret 0) -> perfect here.
    assert act["mean_regret"] == 0.0
    assert act["top1_match"] == 1.0


def test_run_audit_emits_all_five_tables(tmp_path) -> None:
    # End-to-end on the tiny synthetic dump: run_audit must write the five auditable tables + spec.
    import os
    d = tmp_path / "dump"
    d.mkdir()
    decs, prods = [], []
    for seed in ("1", "2", "3"):
        for c in range(12):
            dk = ("ta", seed, str(c), f"o{c}")
            decs.append({"task_id": "ta", "seed": seed, "cycle": str(c), "order_id": f"o{c}",
                         "stream_index": "0", "product": "1", "quantity": str(1 + c % 3),
                         "order_age": "0", "ttl_remaining": "1", "backlog_size": "2",
                         "same_product_backlog": "1"})
            for j, f0 in enumerate((0.2, 0.8)):
                prods.append({"task_id": "ta", "seed": seed, "cycle": str(c), "order_id": f"o{c}",
                              "producer_id": f"p{j}", "eligible": "True",
                              "teacher_label": str(10 * f0), "actor_score": str(f0),
                              **{f"feat_{i}": ("0.0" if i else str(f0)) for i in range(7)}})
    _write(str(d / "decisions.csv"), decs)
    _write(str(d / "producers.csv"), prods)
    out = tmp_path / "audit"
    ol.run_audit(str(d), str(out), log=lambda *a, **k: None)
    for name in ("block_summary.csv", "alias_by_block.csv", "probe_by_block.csv",
                 "top_mistakes_by_block.csv", "spec.json"):
        assert os.path.getsize(str(out / name)) > 0, name


def test_run_audit_includes_full_state_and_history_blocks_when_present(tmp_path) -> None:
    # A full=True re-dump adds per-producer b4_*/h_* columns; ladder_for must append
    # +full_producer_state and +history, and run_audit must score them like any other block.
    d = tmp_path / "dump"
    d.mkdir()
    decs, prods = [], []
    for seed in ("1", "2", "3"):
        for c in range(12):
            decs.append({"task_id": "ta", "seed": seed, "cycle": str(c), "order_id": f"o{c}",
                         "stream_index": "0", "product": "1", "quantity": str(1 + c % 3),
                         "order_age": "0", "ttl_remaining": "1", "backlog_size": "2",
                         "same_product_backlog": "1"})
            for j, f0 in enumerate((0.2, 0.8)):
                prods.append({"task_id": "ta", "seed": seed, "cycle": str(c), "order_id": f"o{c}",
                              "producer_id": f"p{j}", "eligible": "True",
                              "teacher_label": str(10 * f0), "actor_score": str(f0),
                              **{f"feat_{i}": ("0.0" if i else str(f0)) for i in range(7)},
                              "b4_cap_min": str(f0), "b4_cap_std": "0.1",
                              "h_routed_ema": str(f0 * 2)})
    _write(str(d / "decisions.csv"), decs)
    _write(str(d / "producers.csv"), prods)

    rows = ol.load_dump(str(d))
    assert rows[0]["b4_cap_min"] == 0.2 and rows[0]["h_routed_ema"] == 0.4
    names = [b for b, _ in ol.ladder_for(rows)]
    assert names == ["current_row", "+order", "+local_queue",
                     "+full_producer_state", "+history"]

    out = tmp_path / "audit"
    res = ol.run_audit(str(d), str(out), log=lambda *a, **k: None)
    blocks = [r["block"] for r in res["block_summary"]]
    assert "+full_producer_state" in blocks and "+history" in blocks


def test_run_audit_survives_task_with_single_seed(tmp_path) -> None:
    # A 1-seed task admits no leave-one-seed-out fold, so heldout_probe returns nothing for it.
    # run_audit must still emit tables (alias only), with None probe metrics, not KeyError.
    d = tmp_path / "dump"
    d.mkdir()
    decs, prods = [], []
    for c in range(8):
        decs.append({"task_id": "ta", "seed": "1", "cycle": str(c), "order_id": f"o{c}",
                     "stream_index": "0", "product": "1", "quantity": "1", "order_age": "0",
                     "ttl_remaining": "1", "backlog_size": "2", "same_product_backlog": "1"})
        for j, f0 in enumerate((0.2, 0.8)):
            prods.append({"task_id": "ta", "seed": "1", "cycle": str(c), "order_id": f"o{c}",
                          "producer_id": f"p{j}", "eligible": "True", "teacher_label": str(10 * f0),
                          "actor_score": str(f0),
                          **{f"feat_{i}": ("0.0" if i else str(f0)) for i in range(7)}})
    _write(str(d / "decisions.csv"), decs)
    _write(str(d / "producers.csv"), prods)
    res = ol.run_audit(str(d), str(tmp_path / "audit"), log=lambda *a, **k: None)
    row = res["block_summary"][0]
    assert row["ridge_regret"] is None and row["ridge_beats_actor_ntasks"] == 0


def test_routing_metrics_excludes_flat_decisions_and_is_tie_aware() -> None:
    # A flat-teacher decision (both labels equal) carries no real choice and must be dropped from
    # regret/r2/top1 (P1). A tie at the best label counts as a top-1 match, matching the floor (P3).
    test = [
        {"decision_key": ("t", "1", 0), "teacher_label": 5.0},   # flat -> excluded
        {"decision_key": ("t", "1", 0), "teacher_label": 5.0},
        {"decision_key": ("t", "1", 1), "teacher_label": 8.0},   # real choice, best=8
        {"decision_key": ("t", "1", 1), "teacher_label": 2.0},
    ]
    pred = [0.0, 1.0, 0.9, 0.1]  # dec0 irrelevant (flat); dec1 picks the label-8 producer
    m = ol._routing_metrics(test, pred)
    assert m["n_decisions"] == 1          # only the non-flat decision scored
    assert m["mean_regret"] == 0.0        # picked the best
    assert m["top1_match"] == 1.0


def test_floor_and_probe_share_decision_weighting_and_real_choice_filter() -> None:
    # actor floor is scored through the same _routing_metrics as the probe: real-choice-only,
    # decision-weighted, tie-aware. A dominated actor_score ranking must show up as positive regret.
    rows = []
    for c in range(6):
        dk = ("t", "1", c)
        rows += [
            {"task_id": "t", "decision_key": dk, "teacher_label": 9.0, "actor_score": 0.1},
            {"task_id": "t", "decision_key": dk, "teacher_label": 1.0, "actor_score": 0.9},
        ]
    act = ol.floor_metrics(rows, kind="actor")["t"]
    assert act["n_decisions"] == 6            # every decision is a real choice
    assert act["mean_regret"] == 8.0          # actor always picks the worse producer
    assert act["top1_match"] == 0.0


def test_ladder_discovers_producer_product_v1_pp_blocks(tmp_path) -> None:
    # An EXP-0038B producer_product_v1 dump adds per-producer pp_* columns (P1 throughput, P2 opp,
    # P3 dep, P4 base, P6 fulfil) + reused h_ history. ladder_for must append one cumulative block
    # per pp_ prefix (in ladder order) plus +history, and run_audit must score each like any block.
    d = tmp_path / "dump"
    d.mkdir()
    decs, prods = [], []
    for seed in ("1", "2", "3"):
        for c in range(12):
            decs.append({"task_id": "ta", "seed": seed, "cycle": str(c), "order_id": f"o{c}",
                         "stream_index": "0", "product": "1", "quantity": str(1 + c % 3),
                         "order_age": "0", "ttl_remaining": "1", "backlog_size": "2",
                         "same_product_backlog": "1"})
            for j, f0 in enumerate((0.2, 0.8)):
                prods.append({"task_id": "ta", "seed": seed, "cycle": str(c), "order_id": f"o{c}",
                              "producer_id": f"p{j}", "eligible": "True",
                              "teacher_label": str(10 * f0), "actor_score": str(f0),
                              **{f"feat_{i}": ("0.0" if i else str(f0)) for i in range(7)},
                              "pp_throughput_current_min": str(f0), "pp_throughput_current_std": "0.1",
                              "pp_opp_capacity_opportunity_cost": str(1 - f0),
                              "pp_dep_bottleneck_cover": str(f0 * 2),
                              "pp_base_margin_to_best": str(-f0),
                              "pp_fulfill_est_cycles_to_complete": str(f0),
                              "h_routed_ema": str(f0 * 2)})
    _write(str(d / "decisions.csv"), decs)
    _write(str(d / "producers.csv"), prods)

    rows = ol.load_dump(str(d))
    assert rows[0]["pp_throughput_current_min"] == 0.2 and rows[0]["pp_base_margin_to_best"] == -0.2
    names = [b for b, _ in ol.ladder_for(rows)]
    assert names == ["current_row", "+order", "+local_queue", "+pp_throughput", "+pp_opportunity",
                     "+pp_dependency", "+pp_base", "+pp_fulfilment", "+history"]

    out = tmp_path / "audit"
    res = ol.run_audit(str(d), str(out), log=lambda *a, **k: None)
    blocks = [r["block"] for r in res["block_summary"]]
    assert "+pp_throughput" in blocks and "+pp_base" in blocks and "+pp_fulfilment" in blocks


def test_split_cols_routes_sys_columns_to_interaction_context() -> None:
    # EXP-0038B Phase 2: sys_* columns are (mostly) decision-constant, so _split_cols must route them
    # into `context` (interaction-only), alongside order/queue fields — never into `current` where
    # they would enter additively. Producer-varying columns (feat_/pp_/h_) stay in `current`.
    cols = ["feat_0", "pp_throughput_current_min", "h_routed_ema", "quantity",
            "sys_demand_target_queued_qty", "sys_capacity_target_free_hhi"]
    current, context = ol._split_cols(cols)
    assert set(current) == {"feat_0", "pp_throughput_current_min", "h_routed_ema"}
    assert set(context) == {"quantity", "sys_demand_target_queued_qty", "sys_capacity_target_free_hhi"}


def test_sys_context_re_ranks_only_through_interaction_with_producer_feature() -> None:
    # The Phase-2 licensing mechanism: a decision-constant sys_* field can re-rank producers only via
    # its interaction with a producer-varying feature. Here the best producer FLIPS with a
    # decision-constant sys value: label = feat_0 when sys=0 (high feat_0 wins) but 1-feat_0 when
    # sys=1 (low feat_0 wins). A ridge on feat_0 alone is forced to one ranking and is wrong half the
    # time; adding the sys column (as interaction context) recovers feat_0 x sys and the flip.
    import random
    rng = random.Random(0)

    def make(seed, n):
        out = []
        for c in range(n):
            s = rng.choice([0.0, 1.0])
            rows = []
            for j, f0 in enumerate((rng.uniform(0, 1), rng.uniform(0, 1))):
                lab = f0 if s == 0.0 else 1.0 - f0
                rows.append({"decision_key": ("t", seed, c), "producer_id": f"p{j}",
                             "task_id": "t", "seed": seed, "teacher_label": float(lab),
                             **{f"feat_{i}": 0.0 for i in range(7)},
                             **{col: 0 for col in ol.ORDER_COLS + ol.QUEUE_COLS},
                             "feat_0": float(f0), "sys_demand_target_queued_qty": s})
            out += rows
        return out

    train = make("1", 90) + make("2", 90)
    test = make("3", 60)
    base = ["feat_0"]
    with_sys = ["feat_0", "sys_demand_target_queued_qty"]
    m_base = ol.probe_fold(train, test, base, kind="ridge")
    m_sys = ol.probe_fold(train, test, with_sys, kind="ridge")
    assert m_base["mean_regret"] > 0.10                 # feat_0 alone cannot see the flip
    assert m_sys["mean_regret"] < 0.05                  # sys x feat_0 interaction recovers it
    assert m_sys["mean_regret"] < m_base["mean_regret"] / 2


def test_ladder_discovers_producer_product_system_v1_sys_blocks(tmp_path) -> None:
    # A producer_product_system_v1 dump adds per-producer sys_* columns on top of the Phase-1 pp_*
    # blocks + h_ history. ladder_for must append one cumulative block per sys_ prefix (in ladder
    # order) AFTER +history, and run_audit must score each like any block.
    d = tmp_path / "dump"
    d.mkdir()
    decs, prods = [], []
    for seed in ("1", "2", "3"):
        for c in range(12):
            decs.append({"task_id": "ta", "seed": seed, "cycle": str(c), "order_id": f"o{c}",
                         "stream_index": "0", "product": "1", "quantity": str(1 + c % 3),
                         "order_age": "0", "ttl_remaining": "1", "backlog_size": "2",
                         "same_product_backlog": "1"})
            for j, f0 in enumerate((0.2, 0.8)):
                prods.append({"task_id": "ta", "seed": seed, "cycle": str(c), "order_id": f"o{c}",
                              "producer_id": f"p{j}", "eligible": "True",
                              "teacher_label": str(10 * f0), "actor_score": str(f0),
                              **{f"feat_{i}": ("0.0" if i else str(f0)) for i in range(7)},
                              "pp_throughput_current_min": str(f0),
                              "pp_opp_capacity_opportunity_cost": str(1 - f0),
                              "pp_dep_bottleneck_cover": str(f0 * 2),
                              "pp_base_margin_to_best": str(-f0),
                              "pp_fulfill_est_cycles_to_complete": str(f0),
                              "h_routed_ema": str(f0 * 2),
                              "sys_demand_target_queued_qty": str(1 + c % 3),
                              "sys_capacity_target_free_hhi": "0.5",
                              "sys_recipe_product_depth": "2",
                              "sys_cutoff_order_urgency": "0.5",
                              "sys_actor_recent_completed": str(f0)})
    _write(str(d / "decisions.csv"), decs)
    _write(str(d / "producers.csv"), prods)

    rows = ol.load_dump(str(d))
    assert rows[0]["sys_demand_target_queued_qty"] == 1.0 and rows[0]["sys_recipe_product_depth"] == 2.0
    names = [b for b, _ in ol.ladder_for(rows)]
    assert names == ["current_row", "+order", "+local_queue", "+pp_throughput", "+pp_opportunity",
                     "+pp_dependency", "+pp_base", "+pp_fulfilment", "+history",
                     "+sys_demand", "+sys_capacity", "+sys_recipe", "+sys_cutoff", "+sys_actor"]

    out = tmp_path / "audit"
    res = ol.run_audit(str(d), str(out), log=lambda *a, **k: None)
    blocks = [r["block"] for r in res["block_summary"]]
    assert "+sys_demand" in blocks and "+sys_capacity" in blocks and "+sys_actor" in blocks
