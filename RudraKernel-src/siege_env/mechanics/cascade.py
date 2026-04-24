"""Epistemic cascade mechanics for SIEGE Step 13."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CascadeSnapshot:
    """Compact summary of cascade state at a step."""

    mean_confidence: float
    herd_strength: float
    triggered: bool


class EpistemicCascadeEngine:
    """Detect herd-behavior collapse across claim confidences."""

    def __init__(self, *, trigger_threshold: float = 0.82, min_agents: int = 4) -> None:
        self._trigger_threshold = trigger_threshold
        self._min_agents = min_agents

    def evaluate(self, confidences: list[float]) -> CascadeSnapshot:
        if not confidences:
            return CascadeSnapshot(mean_confidence=0.0, herd_strength=0.0, triggered=False)
        clipped = [max(0.0, min(1.0, float(c))) for c in confidences]
        mean_conf = sum(clipped) / len(clipped)
        herd_strength = sum(1.0 for c in clipped if c >= self._trigger_threshold) / len(clipped)
        triggered = len(clipped) >= self._min_agents and herd_strength >= 0.75
        return CascadeSnapshot(
            mean_confidence=round(mean_conf, 4),
            herd_strength=round(herd_strength, 4),
            triggered=triggered,
        )
