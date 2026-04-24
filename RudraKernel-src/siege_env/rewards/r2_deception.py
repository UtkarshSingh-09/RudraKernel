"""Deception penetration reward (R2) for pathogen role."""

from __future__ import annotations

from siege_env.models import SIEGEAction


def compute_r2_deception(
    action: SIEGEAction,
    *,
    seat_role: str,
    ground_truth_root_cause: str,
) -> float:
    """Reward pathogen for pushing an incorrect diagnosis."""

    if seat_role != "pathogen":
        return 0.0
    if action.tool_name != "diagnose":
        return 0.0
    if action.arguments.root_cause == ground_truth_root_cause:
        return 0.0
    return 1.0
