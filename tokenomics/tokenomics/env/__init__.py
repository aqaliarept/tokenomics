"""RL adapter layer: PettingZoo AEC environment over the pure-Python domain."""

from __future__ import annotations

from .aec_env import TokenomicsAECEnv


def env(**kwargs: object) -> TokenomicsAECEnv:
    """PettingZoo-style factory."""
    return TokenomicsAECEnv(**kwargs)


__all__ = ["TokenomicsAECEnv", "env"]
