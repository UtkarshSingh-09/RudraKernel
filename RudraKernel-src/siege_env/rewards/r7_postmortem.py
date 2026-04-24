"""R7 postmortem quality reward component (Step 19)."""

from __future__ import annotations

from siege_env.models import SIEGEAction


def compute_r7_postmortem(action: SIEGEAction, *, ground_truth_root_cause: str) -> float:
    if action.tool_name != "postmortem":
        return 0.0

    root_match = action.arguments.root_cause == ground_truth_root_cause
    timeline_events = [event.event.strip().lower() for event in action.arguments.timeline]
    unique_events = len(set(timeline_events))
    analysis = action.arguments.misdiagnosis_analysis.strip()

    score = 0.1
    if root_match:
        score += 0.5
    if unique_events >= 2:
        score += 0.2
    if len(analysis) >= 60:
        score += 0.2

    # Exploit resistance: penalize trivial template-parroting summaries.
    if unique_events == 1 or len(analysis) < 30:
        score -= 0.2

    return max(0.0, min(1.0, round(score, 4)))
