"""Cross-episode reputation tracker (Step-aligned scaffold)."""

from __future__ import annotations


class ReputationMemory:
    """Keeps exponentially-smoothed reputation per agent across episodes."""

    def __init__(self, *, alpha: float = 0.2) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        self._alpha = alpha
        self._scores: dict[int, float] = {}

    def update(self, agent_id: int, *, reliable: bool) -> float:
        prior = self._scores.get(agent_id, 0.5)
        target = 1.0 if reliable else 0.0
        posterior = (1.0 - self._alpha) * prior + self._alpha * target
        self._scores[agent_id] = max(0.0, min(1.0, posterior))
        return self._scores[agent_id]

    def snapshot(self) -> dict[int, float]:
        return dict(self._scores)

    def reset(self) -> None:
        self._scores.clear()
