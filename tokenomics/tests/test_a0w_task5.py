"""EXP-0036 Stage A0w (HANDOVER Task 5) pure-logic contract: window/episode blend-policy
value tables. The heavy full-horizon collection + distill + deployed rollout is cloud work;
here we pin only the pure target-construction and baseline/decision-regret math."""

from __future__ import annotations

import numpy as np

from tokenomics.eval.orderer_representability import (
    _const_regret_frac,
    _decision_regret_frac,
    episode_value_table,
    window_value_rows,
)


def test_episode_value_table_orders_by_vocab() -> None:
    order = ["even_split", "blend_0.5", "proportional"]
    tbl = episode_value_table({"proportional": 3.0, "even_split": 1.0, "blend_0.5": 2.0}, order)
    np.testing.assert_allclose(tbl, [1.0, 2.0, 3.0])
    assert tbl.dtype == np.float32


def test_window_value_rows_pad_dead_policies_and_align_to_reference() -> None:
    order = ["a", "b", "c"]
    # c "dies" at cycle 3 (only 3 deltas); windows must pad it with 0 up to ref_len=4.
    deltas = {"a": [1.0, 1.0, 1.0, 1.0], "b": [0.0, 5.0, 0.0, 0.0], "c": [2.0, 0.0, 0.0]}
    rows = window_value_rows(deltas, ref_len=4, window=2, order=order)
    assert [s for s, _ in rows] == [0, 1, 2]  # full length-2 windows only
    np.testing.assert_allclose(rows[0][1], [2.0, 5.0, 2.0])  # [0:2] sums, c padded fine
    np.testing.assert_allclose(rows[2][1], [2.0, 0.0, 0.0])  # [2:4]: c already 0
    # a window longer than the reference trajectory yields no rows (short episode)
    assert window_value_rows(deltas, ref_len=4, window=5, order=order) == []


def test_const_regret_frac_is_even_split_normalized() -> None:
    # rows: best is col1 both times; even_split=col0. Picking col0 loses the whole headroom.
    y = np.array([[0.0, 10.0, 4.0], [0.0, 6.0, 2.0]], dtype=np.float32)
    assert _const_regret_frac(y, pick=0, even=0) == 1.0  # always-even_split = full regret
    assert _const_regret_frac(y, pick=1, even=0) == 0.0  # always-best = zero regret
    # picking col2: regret (10-4)+(6-2)=10 over headroom (10-0)+(6-0)=16
    assert _const_regret_frac(y, pick=2, even=0) == round(10 / 16, 4)


def test_decision_rule_sticks_to_default_until_threshold_cleared() -> None:
    y = np.array([[0.0, 10.0, 4.0], [0.0, 6.0, 2.0]], dtype=np.float32)
    # default = col2. Huge threshold => never switch => regret = const pick col2, 0 switches.
    pred = y.copy()
    rf_hi, sw_hi = _decision_regret_frac(pred, y, even=0, default_idx=2, threshold=100.0)
    assert rf_hi == _const_regret_frac(y, pick=2, even=0) and sw_hi == 0.0
    # zero threshold => follows the predicted argmax (col1 both rows) => zero regret, all switch.
    rf_lo, sw_lo = _decision_regret_frac(pred, y, even=0, default_idx=2, threshold=0.0)
    assert rf_lo == 0.0 and sw_lo == 1.0
