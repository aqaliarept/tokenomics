"""Ablation: config-fingerprint load gate vs portable schema-version gate (step 14).

Runs both candidate load gates on the SAME portable producer checkpoint across topologies
that a portable policy must transfer to — a capacity relabel, a wider product count, a wider
producer count. The schema gate (`load_producer_score_model`, keyed on the topology-free
`PortableSchema`) loads all of them; a config-fingerprint gate (keyed on
`env.spaces.config_fingerprint`, which folds in producer ids / capacity types / vocab / dims)
rejects every non-identical topology. That is the empirical case for schema-version loading.

Local + deterministic: no training, no pod. The checkpoint weights are irrelevant here —
only the LOAD GATE behavior is under test — so an untrained scorer is used.
"""

from __future__ import annotations

from ..domain.config import Configuration
from ..env.spaces import config_fingerprint
from .joint_stack import relabel_capacity_types
from .scale import build_scale_config


def load_with_fingerprint_gate(path: str, expected_fingerprint: str):
    """Counterfactual STRICT load gate: accept only if the checkpoint's stamped
    `config_fingerprint` equals the target economy's fingerprint. This is the legacy
    flat-actor gate applied to a portable checkpoint — it rejects any topology change.
    Returns the loaded model, or raises `ValueError` on fingerprint mismatch."""
    import torch

    from ..train.producer_scorer import load_producer_score_model

    ckpt = torch.load(path, weights_only=True)
    stamped = ckpt.get("config_fingerprint")
    if stamped != expected_fingerprint:
        raise ValueError(
            f"fingerprint gate: checkpoint {stamped!r} != target economy {expected_fingerprint!r}"
        )
    return load_producer_score_model(path)


def gate_ablation_rows(ckpt_path: str, *, seed: int = 6, depth: int = 3) -> list[dict]:
    """Save one portable producer checkpoint stamped with the BASE economy's fingerprint,
    then, for each target topology, report whether the schema gate and the fingerprint gate
    each accept it. Row keys: `target`, `fingerprint_changed`, `schema_gate_loads`,
    `fingerprint_gate_loads`."""
    from ..train.producer_scorer import (
        ProducerScoreModel,
        load_producer_score_model,
        save_producer_score_model,
    )

    base = build_scale_config(seed, depth=depth)
    base_fp = config_fingerprint(base, producer_obs_version=2)
    save_producer_score_model(ProducerScoreModel(hidden=16), ckpt_path, config_fingerprint=base_fp)

    targets: dict[str, Configuration] = {
        "base": base,
        "capacity_relabel": relabel_capacity_types(base),
        "wider_product": build_scale_config(seed, depth=depth, products_per_layer=8),
        "wider_producer": build_scale_config(seed, depth=depth, producers_per_layer=8),
    }

    rows: list[dict] = []
    for name, cfg in targets.items():
        target_fp = config_fingerprint(cfg, producer_obs_version=2)
        schema_ok = _accepts(lambda: load_producer_score_model(ckpt_path))
        fp_ok = _accepts(lambda: load_with_fingerprint_gate(ckpt_path, target_fp))
        rows.append({
            "target": name,
            "fingerprint_changed": target_fp != base_fp,
            "schema_gate_loads": schema_ok,
            "fingerprint_gate_loads": fp_ok,
        })
    return rows


def _accepts(load_fn) -> bool:
    try:
        load_fn()
        return True
    except ValueError:
        return False


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--ckpt", default="/tmp/gate_ablation_scorer.pt")
    args = ap.parse_args()
    rows = gate_ablation_rows(args.ckpt, seed=args.seed, depth=args.depth)
    print("target,fingerprint_changed,schema_gate_loads,fingerprint_gate_loads")
    for r in rows:
        print(f"{r['target']},{r['fingerprint_changed']},{r['schema_gate_loads']},"
              f"{r['fingerprint_gate_loads']}")


if __name__ == "__main__":
    main()
