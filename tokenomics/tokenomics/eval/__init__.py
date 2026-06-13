"""Evaluation: heuristic baselines + metrics harness for the signal check."""

from __future__ import annotations

from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from .runner import evaluate, run_episode

__all__ = ["PRODUCING_BASELINES", "ORDERING_BASELINES", "run_episode", "evaluate"]
