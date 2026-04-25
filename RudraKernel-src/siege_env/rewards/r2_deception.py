"""Deception penetration reward (R2) for pathogen role."""

from __future__ import annotations

from siege_env.models import SIEGEAction
from siege_env.rewards.rubric import Rubric


def compute_r2_deception(
    action: SIEGEAction,
    *,
    seat_role: str,
    ground_truth_root_cause: str,
) -> float:
    """Reward pathogen for pushing an incorrect diagnosis."""

    if seat_role not in {"pathogen", "cooperative_pathogen"}:
        return 0.0
    if action.tool_name != "diagnose":
        return 0.0
    if action.arguments.root_cause == ground_truth_root_cause:
        return 0.0
    return 1.0


R2_RUBRIC = Rubric(
    key="r2_deception",
    description="Pathogen reward for successful deceptive diagnosis.",
    scorer=compute_r2_deception,
)
