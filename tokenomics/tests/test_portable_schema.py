"""Portable checkpoint schema gate (PRODUCT_AGNOSTIC ordered-plan step 13).

Portable scorer checkpoints must load by *feature schema* — named features per entity
type, entity/relation types, and normalization / decoder / architecture versions — not by
a topology fingerprint. A real schema/semantics change must reject; product count, producer
count, and relabeling (none of which live in the schema) must NOT.
"""

from __future__ import annotations

import pytest


def test_identical_schema_is_compatible():
    from tokenomics.train.portable_schema import PortableSchema, assert_compatible

    schema = PortableSchema(
        schema_version=1,
        entity_feature_names={"row": ("inventory", "inbound")},
        entity_types=("global", "row"),
        relation_types=("row_uses_cap",),
        normalization_version="log1p-v1",
        decoder_version="planner-v1",
        architecture_version="shared-row-scorer-v1",
    )
    # a byte-identical schema loads cleanly (no raise, returns None)
    assert assert_compatible(schema, schema) is None


def test_renamed_feature_rejects():
    from tokenomics.train.portable_schema import PortableSchema, assert_compatible

    saved = PortableSchema(
        schema_version=1,
        entity_feature_names={"row": ("inventory", "inbound")},
        entity_types=("global", "row"),
        relation_types=("row_uses_cap",),
        normalization_version="log1p-v1",
        decoder_version="planner-v1",
        architecture_version="shared-row-scorer-v1",
    )
    from dataclasses import replace

    current = replace(saved, entity_feature_names={"row": ("inventory", "RENAMED")})
    with pytest.raises(ValueError, match="feature"):
        assert_compatible(saved, current)


def test_producer_checkpoint_carries_named_schema_and_gates_on_it(tmp_path):
    """A freshly saved producer scorer checkpoint records its `PortableSchema`, and load
    rejects a checkpoint whose saved schema no longer matches the current code (a real
    feature/semantics drift), naming the offending field."""
    pytest.importorskip("torch")
    import torch

    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        load_producer_score_model,
        producer_portable_schema,
        save_producer_score_model,
    )

    path = tmp_path / "producer_scorer.pt"
    save_producer_score_model(ProducerScoreModel(hidden=16), path)

    ckpt = torch.load(path, weights_only=True)
    assert "schema" in ckpt  # schema block persisted
    # the current-code schema is what load asserts against
    from tokenomics.train.portable_schema import PortableSchema
    assert PortableSchema.from_dict(ckpt["schema"]) == producer_portable_schema()

    load_producer_score_model(path)  # matching schema loads

    # tamper the saved schema's feature names -> load must reject
    ckpt["schema"]["entity_feature_names"]["input"] = ["inventory", "BOGUS", "x", "y"]
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="feature|schema"):
        load_producer_score_model(path)


def test_producer_load_tolerates_legacy_checkpoint_without_schema(tmp_path):
    """Existing v2 checkpoints (EXP-0013/0021) were saved before the schema block existed.
    They must keep loading — the named gate is additive, not a hard break of live
    artifacts."""
    pytest.importorskip("torch")
    import torch

    from tokenomics.train.producer_scorer import (
        PRODUCER_SCORER_CHECKPOINT_VERSION,
        ProducerScoreModel,
        load_producer_score_model,
    )

    model = ProducerScoreModel(hidden=16)
    legacy = {  # exactly the pre-step-13 save shape (no "schema")
        "version": PRODUCER_SCORER_CHECKPOINT_VERSION,
        "hidden": 16,
        "upgrade_threshold": 0.6,
        "inventory_expansion_threshold": 0.5,
        "state_dict": model.state_dict(),
    }
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)
    loaded = load_producer_score_model(path)
    assert loaded.upgrade_threshold == 0.6


def test_producer_load_ignores_non_gating_config_fingerprint(tmp_path):
    """A recorded config fingerprint is audit metadata, never a load gate: two checkpoints
    with different fingerprints but the same schema both load (portability by schema, not by
    topology)."""
    pytest.importorskip("torch")
    import torch

    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        load_producer_score_model,
        save_producer_score_model,
    )

    path = tmp_path / "producer_scorer.pt"
    save_producer_score_model(ProducerScoreModel(hidden=16), path,
                              config_fingerprint="some-wide-topology-hash")
    ckpt = torch.load(path, weights_only=True)
    assert ckpt["config_fingerprint"] == "some-wide-topology-hash"  # stored
    load_producer_score_model(path)  # ...but not gated on


def test_orderer_checkpoint_carries_named_schema_and_gates_on_it(tmp_path):
    """The orderer scorer checkpoint carries its `PortableSchema` and load rejects a real
    feature-schema drift by name (the same additive named gate as the producer side)."""
    pytest.importorskip("torch")
    import torch

    from tokenomics.train.orderer_scorer import (
        OrdererScoreModel,
        load_scorer,
        orderer_portable_schema,
        save_scorer,
    )
    from tokenomics.train.portable_schema import PortableSchema

    path = str(tmp_path / "orderer_scorer.pt")
    save_scorer(OrdererScoreModel(hidden=16), path)

    ckpt = torch.load(path, weights_only=False)
    assert "schema" in ckpt
    assert PortableSchema.from_dict(ckpt["schema"]) == orderer_portable_schema()

    load_scorer(path)  # matching schema loads

    ckpt["schema"]["entity_types"] = ["product", "RENAMED_ROW"]
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="entity_types|schema"):
        load_scorer(path)


def test_orderer_load_tolerates_legacy_checkpoint_without_schema(tmp_path):
    """The live EXP-0012 orderer checkpoint predates the schema block. It must still load —
    the old version + feature-dim gate stays as the legacy floor."""
    pytest.importorskip("torch")
    import torch

    from tokenomics.train.orderer_scorer import (
        ORDERER_SCORER_CHECKPOINT_VERSION,
        OrdererScoreModel,
        _PRODUCER_ROW_DIM,
        _PRODUCT_DIM,
        load_scorer,
    )

    model = OrdererScoreModel(hidden=16)
    legacy = {  # exactly the pre-step-13 save shape (no "schema")
        "scorer_version": ORDERER_SCORER_CHECKPOINT_VERSION,
        "hidden": 16,
        "product_dim": _PRODUCT_DIM,
        "producer_row_dim": _PRODUCER_ROW_DIM,
        "state": model.state_dict(),
    }
    path = str(tmp_path / "legacy_orderer.pt")
    torch.save(legacy, path)
    load_scorer(path)  # loads unchanged
