"""Detection reward (R3) for immune role challenge actions."""

from __future__ import annotations

from typing import Any, Mapping

from siege_env.models import SIEGEAction
from siege_env.rewards.rubric import Rubric


def compute_r3_detection(
    action: SIEGEAction,
    *,
    seat_role: str,
    ground_truth_root_cause: str,
    claims_by_id: Mapping[str, dict[str, Any]],
) -> float:
    """Reward immune role for correctly challenging incorrect claims."""

    if seat_role != "immune":
        return 0.0
    if action.tool_name != "challenge":
        return 0.0

    challenged_claim = claims_by_id.get(action.arguments.claim_id)
    if challenged_claim is None:
        return 0.0

    challenged_root_cause = challenged_claim.get("root_cause")
    return 1.0 if challenged_root_cause != ground_truth_root_cause else 0.0


R3_RUBRIC = Rubric(
    key="r3_detection",
    description="Immune reward for correctly challenging false claims.",
    scorer=compute_r3_detection,
)
