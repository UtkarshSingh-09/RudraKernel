"""Temporal reward (R6) — rewards fast correct diagnoses, penalises slow ones.

R6 design
---------
R6 answers the question: *given that the agent got the diagnosis right,
how quickly did it act on the available evidence?*

  r6 = urgency_multiplier * base_r6_score

where:

  base_r6_score   = 1.0 if action is a correct diagnose, else 0.0
  urgency_multi   = mean freshness across the evidence signals cited in
                    the action's evidence list at the current step
                    (falls back to 1.0 if no evidence list provided)

This gives the full R6 range of [0, 1]:
  • Correct diagnosis on step 0 with fresh evidence → R6 ≈ 1.0
  • Correct diagnosis on step 20 with stale evidence → R6 ≈ 0.1
  • Wrong diagnosis → R6 = 0.0

The urgency multiplier is supplied externally (computed by
TemporalEvidenceTracker.urgency()) so that R6 stays a pure scoring
function with no hidden state — easy to test and compose.
"""

from __future__ import annotations

from siege_env.models import SIEGEAction


def compute_r6_temporal(
    action: SIEGEAction,
    ground_truth_root_cause: str,
    *,
    urgency_multiplier: float = 1.0,
) -> float:
    """Compute the temporal reward component R6.

    Args:
        action: The agent's action for this step.
        ground_truth_root_cause: The true root cause for this episode.
        urgency_multiplier: Pre-computed freshness-based multiplier in [0, 1].
            Defaults to 1.0 (no temporal penalty) when caller does not
            supply freshness data.

    Returns:
        R6 score in [0.0, 1.0].
    """
    if not (0.0 <= urgency_multiplier <= 1.0):
        raise ValueError(
            f"urgency_multiplier must be in [0, 1], got {urgency_multiplier}"
        )

    if action.tool_name != "diagnose":
        return 0.0

    if action.arguments.root_cause != ground_truth_root_cause:
        return 0.0

    return round(urgency_multiplier, 4)
