"""Weighted coalition voting and ratification logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True)
class CoalitionResult:
    """Coalition vote tally result."""

    support_weight: float
    oppose_weight: float
    abstain_weight: float
    total_weight: float
    support_ratio: float
    ratified: bool


class CoalitionVoting:
    """Weighted voting with configurable ratification threshold."""

    def __init__(self, *, ratification_threshold: float = 0.6, neutral_weight: float = 0.5) -> None:
        if not 0.0 < ratification_threshold <= 1.0:
            raise ValueError("ratification_threshold must be in (0, 1].")
        if not 0.0 <= neutral_weight <= 1.0:
            raise ValueError("neutral_weight must be in [0, 1].")
        self._ratification_threshold = ratification_threshold
        self._neutral_weight = neutral_weight

    def tally(
        self,
        *,
        votes: Mapping[int, bool | None],
        trust_weights: Mapping[int, float],
    ) -> CoalitionResult:
        support_weight = 0.0
        oppose_weight = 0.0
        abstain_weight = 0.0

        for agent_id, vote in votes.items():
            weight = float(trust_weights.get(agent_id, self._neutral_weight))
            weight = max(0.0, min(1.0, weight))
            if vote is True:
                support_weight += weight
            elif vote is False:
                oppose_weight += weight
            else:
                abstain_weight += weight

        decided_weight = support_weight + oppose_weight
        total_weight = decided_weight
        support_ratio = support_weight / decided_weight if decided_weight > 0.0 else 0.0
        ratified = support_weight > oppose_weight and support_ratio >= self._ratification_threshold

        return CoalitionResult(
            support_weight=round(support_weight, 6),
            oppose_weight=round(oppose_weight, 6),
            abstain_weight=round(abstain_weight, 6),
            total_weight=round(total_weight, 6),
            support_ratio=round(support_ratio, 6),
            ratified=ratified,
        )
