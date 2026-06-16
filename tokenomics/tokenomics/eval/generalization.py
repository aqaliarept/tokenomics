"""Controlled-generalization perturbations (NEXT STEPS stage 3).

Single-axis config transforms toward UNSEEN/harder settings — the opposite of the
curriculum *easers* in `train.smoke_ppo` (which ramp toward p=0). Each changes
exactly one dimension so a generalization failure is attributable to it ("change one
dimension at a time to identify failures"). Pure (no torch, no RL): just
`dataclasses.replace` over the immutable config, so they import cleanly into the
torch-optional eval layer.

Every perturbation preserves the obs/action dimensions of the base config (it never
adds/removes a producer, product, or capacity type — only numeric values), so a
policy or spec built on the base stays valid on the perturbed config ("keep action
and observation dimensions fixed across this suite")."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from ..domain.config import Configuration


def perturb_degradation(cfg: Configuration, factor: float) -> Configuration:
    """Scale every producer's per-cycle degradation by `factor` (>1 = harsher,
    unseen decay). Capacity types are unchanged, so dims are preserved."""
    producers = [
        replace(p, degradation=round(p.degradation * factor))
        for p in cfg.producer_configs
    ]
    return replace(cfg, producer_configs=producers)


def perturb_quantity(cfg: Configuration, quantity_max: int) -> Configuration:
    """Raise every consumer's demand `quantity_max` to an unseen value (the lower
    bound and product pool are left alone — only the quantity axis moves). The
    consumer pool, not the producer/product set, changes, so dims are preserved."""
    consumers = [replace(c, quantity_max=quantity_max) for c in cfg.consumers]
    return replace(cfg, consumers=consumers)


def perturb_producer_capacity(cfg: Configuration, factor: float) -> Configuration:
    """Scale every producer's owned capacity by `factor` (numeric producer variation;
    <1 = leaner producers, an unseen starting base). Each capacity TYPE the producer
    owns is kept (only its value changes), so dims are preserved."""
    producers = [
        replace(p, capacities={t: round(v * factor) for t, v in p.capacities.items()})
        for p in cfg.producer_configs
    ]
    return replace(cfg, producer_configs=producers)


def perturb_costs(cfg: Configuration, factor: float) -> Configuration:
    """Scale every *set* investment cost (restoration / upgrade / inventory expansion
    / per-unit replenishment) by `factor` (>1 = dearer means of production, an unseen
    cost regime). A `None` cost (unbounded target) stays `None` — we never introduce a
    target where there was none. Only token amounts change, so dims are preserved."""
    def _scale(c: int | None) -> int | None:
        return None if c is None else round(c * factor)

    producers = []
    for p in cfg.producer_configs:
        restoration = replace(p.restoration, cost=_scale(p.restoration.cost))
        upgrade = replace(p.upgrade, cost=_scale(p.upgrade.cost))
        expansion = (
            replace(p.inventory_expansion, cost=_scale(p.inventory_expansion.cost))
            if p.inventory_expansion is not None
            else None
        )
        producers.append(
            replace(
                p,
                restoration=restoration,
                upgrade=upgrade,
                inventory_expansion=expansion,
                replenish_cost_per_unit=_scale(p.replenish_cost_per_unit),
            )
        )
    return replace(cfg, producer_configs=producers)


def combined_config(
    cfg: Configuration,
    *,
    degradation_factor: float = 1.0,
    quantity_max: int | None = None,
    capacity_factor: float = 1.0,
    cost_factor: float = 1.0,
) -> Configuration:
    """Apply several single-axis perturbations at once (NEXT STEPS stage 4: the
    *combined* held-out suite, vs stage 3's one-axis-at-a-time). Each axis composes
    the corresponding `perturb_*`; neutral defaults (factor 1.0, `quantity_max=None`)
    leave that axis untouched, so the all-default call is the identity. Still
    dims-preserving — every component perturbation is."""
    out = cfg
    if degradation_factor != 1.0:
        out = perturb_degradation(out, degradation_factor)
    if quantity_max is not None:
        out = perturb_quantity(out, quantity_max)
    if capacity_factor != 1.0:
        out = perturb_producer_capacity(out, capacity_factor)
    if cost_factor != 1.0:
        out = perturb_costs(out, cost_factor)
    return out


def random_domain_config(
    cfg: Configuration,
    rng: np.random.Generator,
    *,
    degradation_range: tuple[float, float] = (1.0, 1.0),
    quantity_choices: Sequence[int] | None = None,
    capacity_range: tuple[float, float] = (1.0, 1.0),
    cost_range: tuple[float, float] = (1.0, 1.0),
) -> Configuration:
    """Draw ONE dims-preserving domain-randomized config off `cfg` (NEXT STEPS step-3
    fix). Per training iter the trainer samples a fresh operating point from these
    ranges, so the policy sees a *distribution* — the consolidated fix the stage 2-4
    overfit findings (open-loop policy memorizes one operating point) all pointed to.

    Each factor is drawn uniformly from its range (quantity_max uniformly from
    `quantity_choices` if given), then applied via `combined_config`, so the result is
    dims-preserving. Pure: a given `rng` state yields a deterministic config."""
    deg = float(rng.uniform(*degradation_range))
    cap = float(rng.uniform(*capacity_range))
    cost = float(rng.uniform(*cost_range))
    qty = None if not quantity_choices else int(rng.choice(np.asarray(quantity_choices)))
    return combined_config(
        cfg, degradation_factor=deg, quantity_max=qty, capacity_factor=cap, cost_factor=cost
    )


def generalization_suite(base_cfg: Configuration) -> dict[str, Configuration]:
    """One config per dims-preserving generalization axis (stage 3), each perturbing
    a SINGLE dimension off `base_cfg` so a failure is attributable to that axis.

    Axes that need a different topology than canonical phase1 — `product mixture` and
    `multiple viable producers` (phase1 has one declarer per product, ADR-0028) —
    are NOT included here; run the suite on a multi-declarer base (e.g.
    competition.json) to exercise them. The axes below apply to any config."""
    return {
        "higher_degradation": perturb_degradation(base_cfg, 2.0),
        "unseen_quantities": perturb_quantity(
            base_cfg, quantity_max=max(c.quantity_max for c in base_cfg.consumers) + 3
        ),
        "leaner_producers": perturb_producer_capacity(base_cfg, 0.7),
    }
