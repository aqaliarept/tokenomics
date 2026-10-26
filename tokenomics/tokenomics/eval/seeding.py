"""Random-baseline replicate seeds and the replicate mean (design D5, add-measurement-gate-harness;
gate-harness "Random-baseline replicates").

`run_episode(seed=s)` seeds the demand RNG (env reset) and, by default, the action spaces with
the same `s`. A random roster member that reused `s` for its actions would couple its luck to
the episode's demand draw, so each replicate gets its own action seed, derived by sha256 from
(world id, episode seed, replicate index): stable across processes (no `hash()`), and distinct
per replicate without a seed table to freeze.
"""

from __future__ import annotations

import hashlib
from statistics import fmean
from typing import Sequence


def replicate_seed(world: str, episode: int, replicate: int) -> int:
    """Action seed of replicate `replicate` of episode seed `episode` on world `world` (D5):
    `int(sha256(f"{world}|{episode}|{replicate}")[:16], 16)`, a 64-bit non-negative int.

    The spec requires it to differ from the episode's event seed. A 64-bit hash equals a given
    seed with probability 2**-64, so instead of perturbing (a second, never-exercised formula
    the frozen gate rows would silently depend on) the collision raises: the run stops loudly
    and the exact D5 formula stays the only one that ever produces a seed."""
    digest = hashlib.sha256(f"{world}|{episode}|{replicate}".encode()).hexdigest()
    seed = int(digest[:16], 16)
    if seed == episode:
        raise ValueError(
            f"replicate seed of ({world!r}, {episode}, {replicate}) equals the event seed "
            f"{episode}; the spec forbids reusing the event seed as an action seed"
        )
    return seed


def replicate_mean(values: Sequence[float]) -> float:
    """A random member's per-episode value: the mean over its R replicates (D5).

    Not the max: a max over replicates rewards luck and grows with R, which would inflate the
    hindsight baseline just by running more replicates."""
    if not values:
        raise ValueError("replicate_mean needs at least one replicate value")
    return fmean(values)
