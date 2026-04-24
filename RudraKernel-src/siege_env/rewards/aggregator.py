"""Reward aggregation scaffold for SIEGE."""

from __future__ import annotations

from typing import Any

from siege_env.models import SIEGEAction
from siege_env.rewards.r1_resolution import compute_r1_resolution
from siege_env.rewards.r2_deception import compute_r2_deception
from siege_env.rewards.r3_detection import compute_r3_detection
from siege_env.rewards.r4_trust_calibration import compute_r4_trust_calibration


def aggregate_rewards(
    action: SIEGEAction,
    *,
    ground_truth_root_cause: str,
    seat_role: str = "immune",
    claims_by_id: dict[str, dict[str, Any]] | None = None,
    trust_scores: dict[int, float] | None = None,
    agent_reliability: dict[int, bool] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Aggregate reward components (R1-R4)."""

    r1 = compute_r1_resolution(action, ground_truth_root_cause)
    r2 = compute_r2_deception(
        action,
        seat_role=seat_role,
        ground_truth_root_cause=ground_truth_root_cause,
    )
    r3 = compute_r3_detection(
        action,
        seat_role=seat_role,
        ground_truth_root_cause=ground_truth_root_cause,
        claims_by_id=claims_by_id or {},
    )
    r4 = compute_r4_trust_calibration(
        trust_scores=trust_scores or {},
        agent_reliability=agent_reliability or {},
    )

    total = max(0.0, min(1.0, max(r1, r2, r3, r4)))
    return total, {
        "r1_resolution": r1,
        "r2_deception": r2,
        "r3_detection": r3,
        "r4_trust_calibration": r4,
    }
