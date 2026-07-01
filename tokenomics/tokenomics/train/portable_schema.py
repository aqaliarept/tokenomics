"""Portable checkpoint schema gate (PRODUCT_AGNOSTIC ordered-plan step 13).

A portable scorer checkpoint should load by what it *means*, not by the topology it was
trained on. The load gate is this `PortableSchema`: the named features per entity/row type,
the entity and relation types the encoder walks, and the normalization / decoder / model
architecture versions. None of these carry product count, producer count, or product/
capacity ids — so relabeling or widening the topology cannot make a checkpoint reject,
while a real feature/semantics change does.

The full economy/config fingerprint is intentionally NOT part of this gate. If a checkpoint
records one it is audit metadata for transfer reports (`config_fingerprint`), read but never
compared on load — that is the point of a portable checkpoint (see the plan's "Long-term").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortableSchema:
    """The feature-schema identity a portable scorer checkpoint loads against.

    Two checkpoints are load-compatible iff every field here matches. Fields are all
    topology-free by construction: feature *names* (not counts of products/producers),
    entity/relation *type* sets, and opaque version strings for the normalization, the
    deterministic decoder/planner, and the model architecture."""

    schema_version: int
    entity_feature_names: dict[str, tuple[str, ...]]
    entity_types: tuple[str, ...]
    relation_types: tuple[str, ...]
    normalization_version: str
    decoder_version: str
    architecture_version: str

    def to_dict(self) -> dict:
        """Plain-dict form for `torch.save` (tuples survive, dicts of tuples survive)."""
        return {
            "schema_version": self.schema_version,
            "entity_feature_names": {k: list(v) for k, v in self.entity_feature_names.items()},
            "entity_types": list(self.entity_types),
            "relation_types": list(self.relation_types),
            "normalization_version": self.normalization_version,
            "decoder_version": self.decoder_version,
            "architecture_version": self.architecture_version,
        }

    @staticmethod
    def from_dict(d: dict) -> "PortableSchema":
        return PortableSchema(
            schema_version=int(d["schema_version"]),
            entity_feature_names={k: tuple(v) for k, v in d["entity_feature_names"].items()},
            entity_types=tuple(d["entity_types"]),
            relation_types=tuple(d["relation_types"]),
            normalization_version=str(d["normalization_version"]),
            decoder_version=str(d["decoder_version"]),
            architecture_version=str(d["architecture_version"]),
        )


def assert_compatible(saved: PortableSchema, current: PortableSchema) -> None:
    """Raise `ValueError` iff the saved schema is not load-compatible with `current`.

    Every mismatch names the exact field so a real feature/semantics change fails loudly,
    while topology changes (absent from the schema) can never trigger it."""
    if saved.schema_version != current.schema_version:
        raise ValueError(
            f"portable schema_version {saved.schema_version} != {current.schema_version}"
        )
    if saved.architecture_version != current.architecture_version:
        raise ValueError(
            f"architecture_version {saved.architecture_version!r} != "
            f"{current.architecture_version!r}"
        )
    if saved.normalization_version != current.normalization_version:
        raise ValueError(
            f"normalization_version {saved.normalization_version!r} != "
            f"{current.normalization_version!r}"
        )
    if saved.decoder_version != current.decoder_version:
        raise ValueError(
            f"decoder_version {saved.decoder_version!r} != {current.decoder_version!r}"
        )
    if tuple(saved.entity_types) != tuple(current.entity_types):
        raise ValueError(
            f"entity_types {saved.entity_types} != {current.entity_types}"
        )
    if tuple(saved.relation_types) != tuple(current.relation_types):
        raise ValueError(
            f"relation_types {saved.relation_types} != {current.relation_types}"
        )
    if saved.entity_feature_names != current.entity_feature_names:
        raise ValueError(
            f"feature schema mismatch: {saved.entity_feature_names} != "
            f"{current.entity_feature_names}"
        )
