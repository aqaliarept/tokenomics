"""Combined held-out suite (NEXT STEPS stage 4).

A `SeedManifest` is a *fixed, versioned* description of a held-out evaluation: the
base config, the combined perturbation axes (degradation / quantity / capacity /
cost, stacked via `generalization.combined_config`), and the eval seeds. It
serializes to JSON so the exact held-out workload is checked in and reproducible —
"use fixed versioned seed manifests".

`beats_baseline` is the stage-4 release bar as a pure function of two metric dicts
(the `Aggregated` mean/std tuples from `runner.evaluate`): the trained policy must
*reliably outperform the strongest observation-compatible baseline* — survive at
least as well, fulfil at least as many units, and earn strictly more return. Pure
(no torch) so the bar is unit-testable without training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..domain import load_config
from ..domain.config import Configuration
from .baselines import Policy
from .generalization import combined_config
from .runner import Aggregated, evaluate


# the shipped, versioned held-out suite (stage 4): combined competition base +
# mild degradation + randomized quantity + leaner capacity + dearer costs.
DEFAULT_MANIFEST = str(Path(__file__).parent / "manifests" / "heldout_v1.json")


@dataclass(frozen=True)
class SeedManifest:
    """A reproducible held-out evaluation: base config + combined axes + seeds.

    Neutral defaults (factor 1.0, `quantity_max=None`) leave an axis untouched, so a
    manifest names exactly the perturbations it stacks. `seeds` are held-out env seeds
    (disjoint from training/validation, ADR-0011); convention is seeds from 10_000 up,
    clear of validation seeds and signal_check's default 0..N."""

    version: str
    base: str
    seeds: tuple[int, ...]
    degradation_factor: float = 1.0
    quantity_max: int | None = None
    capacity_factor: float = 1.0
    cost_factor: float = 1.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> SeedManifest:
        d = json.loads(text)
        d["seeds"] = tuple(d["seeds"])
        return cls(**d)

    def config(self) -> Configuration:
        """Load the base config and apply the manifest's combined perturbations."""
        return combined_config(
            load_config(self.base),
            degradation_factor=self.degradation_factor,
            quantity_max=self.quantity_max,
            capacity_factor=self.capacity_factor,
            cost_factor=self.cost_factor,
        )


def load_manifest(path: str) -> SeedManifest:
    """Read a versioned manifest JSON from disk. A relative `base` is resolved against
    the manifest's own directory, so a checked-in (manifest, base) pair is portable
    regardless of the working directory."""
    p = Path(path)
    m = SeedManifest.from_json(p.read_text())
    if not Path(m.base).is_absolute():
        m = replace(m, base=str((p.parent / m.base).resolve()))
    return m


def run_manifest(
    prod_policy: Policy, order_policy: Policy, manifest: SeedManifest, horizon: int = 100
) -> Aggregated:
    """Evaluate a policy pair on the manifest's combined-perturbed config over its
    fixed held-out seeds. Same harness for the trained policy and the baseline, so the
    two are scored head-to-head on the identical workload."""
    return evaluate(
        prod_policy, order_policy, seeds=manifest.seeds, horizon=horizon,
        config=manifest.config(),
    )


@dataclass(frozen=True)
class SuiteVerdict:
    passed: bool
    reasons: tuple[str, ...]


def beats_baseline(
    trained: Aggregated, baseline: Aggregated, *, unit_margin: float = 0.0
) -> SuiteVerdict:
    """Stage-4 bar: the trained policy reliably outperforms the strongest
    observation-compatible baseline. All three must hold (survival is non-negotiable,
    so a return/units win cannot buy back extra death):

    - death no worse than the baseline,
    - at least as many fulfilled units (within `unit_margin`),
    - strictly higher return.

    Returns a verdict carrying one reason per failed condition (empty when it passes).
    Reads only the metric means (index 0 of each `(mean, std)` tuple)."""
    reasons: list[str] = []
    t_ret, b_ret = trained["return"][0], baseline["return"][0]
    t_units, b_units = trained["consumer_completed_units"][0], baseline["consumer_completed_units"][0]
    t_dead, b_dead = trained["dead_rate"][0], baseline["dead_rate"][0]

    if t_dead > b_dead:
        reasons.append(f"higher death rate: trained {t_dead:.2f} > baseline {b_dead:.2f}")
    if t_units < b_units - unit_margin:
        reasons.append(f"fewer fulfilled units: trained {t_units:.1f} < baseline {b_units:.1f}")
    if t_ret <= b_ret:
        reasons.append(f"return not higher: trained {t_ret:.1f} <= baseline {b_ret:.1f}")

    return SuiteVerdict(passed=not reasons, reasons=tuple(reasons))
