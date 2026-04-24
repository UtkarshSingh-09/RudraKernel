"""Trust calibration reward (R4) based on Brier scoring."""

from __future__ import annotations

from typing import Mapping


def compute_r4_trust_calibration(
    *,
    trust_scores: Mapping[int, float],
    agent_reliability: Mapping[int, bool],
) -> float:
    """Compute trust calibration score using 1 - mean Brier loss."""

    common_agent_ids = [agent_id for agent_id in trust_scores if agent_id in agent_reliability]
    if not common_agent_ids:
        return 0.0

    brier_sum = 0.0
    for agent_id in common_agent_ids:
        predicted = max(0.0, min(1.0, float(trust_scores[agent_id])))
        actual = 1.0 if agent_reliability[agent_id] else 0.0
        brier_sum += (predicted - actual) ** 2

    mean_brier = brier_sum / len(common_agent_ids)
    score = 1.0 - mean_brier
    return round(max(0.0, min(1.0, score)), 6)
