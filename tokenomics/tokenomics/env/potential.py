"""The M6 team potential Φ (add-m6-joint-mpg-gnn D1): the RL stage's identical-interest reward.

Per played cycle t:

    Φ_t = units_t − λ · Σ_p max(0, X − ρ_p,t) − μ · starving_t − ν · death_t

- `units_t`: consumer units completed in cycle t (`CycleResult.consumer_completed_units`);
- `ρ_p,t`: producer p's capacity ratio after cycle t (mean over owned types of current / max), read from
  the domain's end-of-cycle snapshot (`CycleResult.end_capacities`), as `r_end` is. It is a per-cycle,
  per-producer signal, not the episode-end snapshot: M5's v8 term used end-ρ, whose noise (SD 0.034)
  swamped a 0.01 shift (EXP-0097);
- `starving_t`: consumers whose fill rate so far (fulfilled / ordered units over the played-cycle cohort
  up to t, the side-effect tracker's cohort) is below Y; consumers that ordered nothing are skipped;
- `death_t`: 1 on the cycle the system dies.

X and Y are the frozen side-effect params v2, passed in (the env does not read `eval/`). λ, μ and ν are
set by the RL pre-registration (M5's noise-scaled rule for λ and μ; ν ≈ the episode's base units). The
tracker only reads: it wraps `System.complete_cycle` like the routing tracer and changes no reward the env
emits, so every existing reader and reward stays as it was. The domain stays pure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..domain.common import ConsumerId
from ..domain.system import CycleResult, System

if TYPE_CHECKING:
    from .aec_env import TokenomicsAECEnv


@dataclass(frozen=True)
class PotentialWeights:
    """D1's weights and thresholds. `x` / `y` = the side-effect params' X / Y."""

    lam: float
    mu: float
    nu: float
    x: float
    y: float


@dataclass(frozen=True)
class PotentialTerms:
    """One cycle's raw terms; `phi` applies the weights."""

    units: float
    rundown: float  # Σ_p max(0, X − ρ_p,t)
    starving: int
    death: int

    def phi(self, w: PotentialWeights) -> float:
        return self.units - w.lam * self.rundown - w.mu * self.starving - w.nu * self.death


def producer_ratio(capacities: dict, max_capacities: dict) -> float | None:
    """One producer's ρ: mean over owned types (max > 0) of current / max; None if none."""
    terms = [capacities[t] / m for t, m in max_capacities.items() if m > 0]
    return sum(terms) / len(terms) if terms else None


@dataclass
class PotentialTracker:
    """Per-episode Φ terms. `attach(env)` before the episode; `terms` holds one `PotentialTerms` per
    played cycle, in order. Re-attaching (or the env's reset) restarts the episode."""

    weights: PotentialWeights
    terms: list[PotentialTerms] = field(default_factory=list)
    _ordered: dict[ConsumerId, int] = field(default_factory=dict)
    _fulfilled: dict[ConsumerId, int] = field(default_factory=dict)

    def attach(self, env: "TokenomicsAECEnv") -> None:
        original_reset = env.reset

        def reset(*args: object, **kwargs: object) -> object:
            out = original_reset(*args, **kwargs)
            self._start(env)
            return out

        env.reset = reset  # type: ignore[method-assign]

    def _start(self, env: "TokenomicsAECEnv") -> None:
        self.terms = []
        self._ordered, self._fulfilled = {}, {}
        system: System = env.system
        original: Callable[[], CycleResult] = system.complete_cycle

        def complete_cycle() -> CycleResult:
            # the cohort the env counts: orders first played in this cycle (`_units_placed_for`)
            placed = env._units_placed_for(system.cycle_counter)
            result = original()
            self._observe(placed, result, system.is_dead())
            return result

        system.complete_cycle = complete_cycle  # type: ignore[method-assign]

    def _observe(self, placed: dict[ConsumerId, int], result: CycleResult, dead: bool) -> None:
        for cid, q in placed.items():
            self._ordered[cid] = self._ordered.get(cid, 0) + q
        for out in result.consumer_outcomes:
            if out.completed:
                self._fulfilled[out.consumer_id] = self._fulfilled.get(out.consumer_id, 0) + out.quantity
        w = self.weights
        rundown = 0.0
        for snap in result.end_capacities:
            r = producer_ratio(snap.capacities, snap.max_capacities)
            if r is not None:
                rundown += max(0.0, w.x - r)
        starving = sum(1 for cid, q in self._ordered.items()
                       if q > 0 and self._fulfilled.get(cid, 0) / q < w.y)
        self.terms.append(PotentialTerms(float(result.consumer_completed_units), rundown, starving,
                                         int(dead)))

    def rewards(self) -> list[float]:
        """Φ_t per played cycle (the identical-interest team reward)."""
        return [t.phi(self.weights) for t in self.terms]

    def totals(self) -> dict[str, float]:
        """Episode sums of each raw term (the calibration input of λ / μ)."""
        return {"units": sum(t.units for t in self.terms),
                "rundown": sum(t.rundown for t in self.terms),
                "starving": float(sum(t.starving for t in self.terms)),
                "death": float(sum(t.death for t in self.terms))}
