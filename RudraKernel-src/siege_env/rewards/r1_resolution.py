"""Resolution reward (R1) for minimal Step 04 environment loop."""

from __future__ import annotations

from siege_env.models import SIEGEAction
from siege_env.rewards.rubric import Rubric


def compute_r1_resolution(action: SIEGEAction, ground_truth_root_cause: str) -> float:
    """Return 1.0 for correct diagnose action, otherwise 0.0."""

    if action.tool_name != "diagnose":
        return 0.0

    predicted_root_cause = action.arguments.root_cause
    if predicted_root_cause == ground_truth_root_cause:
        return 1.0
    return 0.0


R1_RUBRIC = Rubric(
    key="r1_resolution",
    description="Correct root-cause diagnosis accuracy.",
    scorer=compute_r1_resolution,
)
