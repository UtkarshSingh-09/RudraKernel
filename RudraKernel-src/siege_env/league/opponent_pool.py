"""Frozen opponent league utilities for SIEGE Step 20."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(slots=True)
class FrozenOpponent:
    opponent_id: str
    policy_tag: str
    tier: int


class FrozenOpponentPool:
    def __init__(self, *, seed: int = 0) -> None:
        self._seed = seed
        self._rng = Random(seed)
        self._catalog = [
            FrozenOpponent("opp_alpha", "baseline_v1", 1),
            FrozenOpponent("opp_beta", "curriculum_v2", 2),
            FrozenOpponent("opp_gamma", "adversarial_v1", 2),
            FrozenOpponent("opp_delta", "hardening_v3", 3),
            FrozenOpponent("opp_epsilon", "stress_test_v1", 3),
        ]

    def sample(self, *, k: int = 3) -> list[FrozenOpponent]:
        k = max(1, min(k, len(self._catalog)))
        return self._rng.sample(self._catalog, k=k)
