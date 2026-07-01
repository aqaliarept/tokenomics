"""Ablation: config-fingerprint load gate vs portable schema-version gate (step 14).

The portable checkpoint contract claims a scorer loads across topologies (relabel, wider
product/producer counts) as long as the feature *schema* is unchanged. This ablation runs
BOTH candidate load gates on the SAME portable checkpoint across those topologies and shows
the fingerprint gate rejects every non-identical topology while the schema gate loads them
all — the empirical case for schema-version loading over a fingerprint gate.
"""

from __future__ import annotations

import pytest


def test_schema_gate_loads_where_fingerprint_gate_rejects(tmp_path):
    pytest.importorskip("torch")

    from tokenomics.eval.gate_ablation import gate_ablation_rows

    rows = gate_ablation_rows(str(tmp_path / "producer_scorer.pt"))
    by_name = {r["target"]: r for r in rows}

    # identical base topology: both gates load
    base = by_name["base"]
    assert base["schema_gate_loads"] is True
    assert base["fingerprint_gate_loads"] is True

    # relabel + wider product + wider producer: schema gate loads all, fingerprint none
    for name in ("capacity_relabel", "wider_product", "wider_producer"):
        r = by_name[name]
        assert r["fingerprint_changed"] is True
        assert r["schema_gate_loads"] is True, name
        assert r["fingerprint_gate_loads"] is False, name
