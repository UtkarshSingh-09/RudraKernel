"""Composable rubric primitives for reward components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class Rubric:
    """A composable reward unit.

    Each reward component exposes a Rubric instance to make the reward
    system inspectable and composable for OpenEnv-style evaluation.
    """

    key: str
    description: str
    scorer: Callable[..., float]

    def score(self, *args: Any, **kwargs: Any) -> float:
        return float(self.scorer(*args, **kwargs))
