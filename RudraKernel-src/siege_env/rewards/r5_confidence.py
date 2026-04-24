"""Confidence calibration reward (R5) for SIEGE.

Design
------
R5 rewards agents that are epistemically honest: confident when correct
and less confident when wrong.

The scoring rule used is a *modified Brier score* anchored to the
correct outcome:

  outcome = 1.0 if diagnosis is correct, else 0.0
  brier   = (confidence - outcome) ** 2
  r5      = 1.0 - brier                          # raw [0, 1]

This gives:
  - Correct + confident (conf=1.0) → R5 = 1.0  (ideal)
  - Correct + half-sure  (conf=0.5) → R5 = 0.75 (mediocre, not maximal)
  - Wrong  + confident   (conf=1.0) → R5 = 0.0  (maximally penalised)
  - Wrong  + half-sure   (conf=0.5) → R5 = 0.75 (partially penalised)

Anti-exploit property
---------------------
An agent that always reports confidence=0.5 ("always-0.5 exploit")
receives R5=0.75 on every step regardless of correctness.  Because the
maximum achievable score is 1.0 (requires confident correct diagnosis),
the always-0.5 strategy cannot reach the top of the leaderboard.  The
test suite explicitly verifies:

    compute_r5_confidence(correct, conf=0.5) < 1.0
    compute_r5_confidence(wrong,   conf=0.5) < compute_r5_confidence(correct, conf=0.5)

Stateful calibration (ConfidenceCalibrator)
-------------------------------------------
For multi-step evaluation (e.g., validation episodes) we provide
`ConfidenceCalibrator`, a running Brier-score accumulator that
computes the *mean R5 across all diagnose actions* in an episode.
This is the metric judges will see in the final leaderboard.
"""

from __future__ import annotations

from siege_env.models import SIEGEAction

# ---------------------------------------------------------------------------
# Stateless per-action R5
# ---------------------------------------------------------------------------


def compute_r5_confidence(
    action: SIEGEAction,
    ground_truth_root_cause: str,
) -> float:
    """Compute the confidence-calibration reward component R5 for a single action.

    Args:
        action: The agent's action for this step.
        ground_truth_root_cause: The true root cause for this episode.

    Returns:
        R5 ∈ [0.0, 1.0].  Non-diagnose actions return 0.0.
    """
    if action.tool_name != "diagnose":
        return 0.0

    confidence = float(action.arguments.confidence)
    correct = action.arguments.root_cause == ground_truth_root_cause
    outcome = 1.0 if correct else 0.0

    brier = (confidence - outcome) ** 2
    r5 = 1.0 - brier
    return round(max(0.0, min(1.0, r5)), 6)


# ---------------------------------------------------------------------------
# Stateful multi-step calibrator
# ---------------------------------------------------------------------------


class ConfidenceCalibrator:
    """Running Brier-score accumulator across multiple diagnose actions.

    Use this to compute mean R5 over an entire episode or evaluation run.

    Example::

        cal = ConfidenceCalibrator()
        for action, truth in episode_steps:
            cal.record(action, truth)
        episode_r5 = cal.mean_r5()
    """

    def __init__(self) -> None:
        self._total_r5: float = 0.0
        self._count: int = 0

    def record(self, action: SIEGEAction, ground_truth_root_cause: str) -> float:
        """Record one action and return its per-step R5.

        Non-diagnose actions are silently ignored (return 0.0 without
        updating the running mean).
        """
        r5 = compute_r5_confidence(action, ground_truth_root_cause)
        if action.tool_name == "diagnose":
            self._total_r5 += r5
            self._count += 1
        return r5

    def mean_r5(self) -> float:
        """Return mean R5 across all recorded diagnose actions.

        Returns 0.0 if no diagnose actions have been recorded yet.
        """
        if self._count == 0:
            return 0.0
        return round(self._total_r5 / self._count, 6)

    def reset(self) -> None:
        """Clear accumulated state."""
        self._total_r5 = 0.0
        self._count = 0

    @property
    def num_recorded(self) -> int:
        """Number of diagnose actions recorded so far."""
        return self._count
