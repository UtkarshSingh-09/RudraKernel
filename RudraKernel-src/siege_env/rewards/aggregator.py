"""Reward aggregation scaffold for SIEGE."""

from __future__ import annotations

from typing import Any

from siege_env.models import SIEGEAction
from siege_env.rewards.r1_resolution import compute_r1_resolution
from siege_env.rewards.r2_deception import compute_r2_deception
from siege_env.rewards.r3_detection import compute_r3_detection
from siege_env.rewards.r4_trust_calibration import compute_r4_trust_calibration
from siege_env.rewards.r5_confidence import compute_r5_confidence
from siege_env.rewards.r6_temporal import compute_r6_temporal


def aggregate_rewards(
    action: SIEGEAction,
    *,
    ground_truth_root_cause: str,
    seat_role: str = "immune",
    claims_by_id: dict[str, dict[str, Any]] | None = None,
    trust_scores: dict[int, float] | None = None,
    agent_reliability: dict[int, bool] | None = None,
    urgency_multiplier: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    """Aggregate reward components (R1-R5, R6)."""

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
    r5 = compute_r5_confidence(action, ground_truth_root_cause)
    r6 = compute_r6_temporal(
        action,
        ground_truth_root_cause,
        urgency_multiplier=urgency_multiplier,
    )

    total = max(0.0, min(1.0, max(r1, r2, r3, r4, r5, r6)))
    return total, {
        "r1_resolution": r1,
        "r2_deception": r2,
        "r3_detection": r3,
        "r4_trust_calibration": r4,
        "r5_confidence": r5,
        "r6_temporal": r6,
    }


# Step 17 append-only extension: include R9 correlation reward
from siege_env.rewards.r9_correlation import compute_r9_correlation

_ORIGINAL_AGGREGATE_REWARDS_STEP17 = aggregate_rewards


def aggregate_rewards(
    action: SIEGEAction,
    *,
    ground_truth_root_cause: str,
    seat_role: str = "immune",
    claims_by_id: dict[str, dict[str, Any]] | None = None,
    trust_scores: dict[int, float] | None = None,
    agent_reliability: dict[int, bool] | None = None,
    urgency_multiplier: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    base_total, components = _ORIGINAL_AGGREGATE_REWARDS_STEP17(
        action,
        ground_truth_root_cause=ground_truth_root_cause,
        seat_role=seat_role,
        claims_by_id=claims_by_id,
        trust_scores=trust_scores,
        agent_reliability=agent_reliability,
        urgency_multiplier=urgency_multiplier,
    )
    r9 = compute_r9_correlation(
        action,
        claims_by_id=claims_by_id,
        ground_truth_root_cause=ground_truth_root_cause,
    )
    components = dict(components)
    components["r9_correlation"] = r9
    total = max(0.0, min(1.0, max(base_total, r9)))
    return total, components


# Step 18 append-only extension: include R8 severity-speed reward
from siege_env.rewards.r8_severity_speed import compute_r8_severity_speed

_ORIGINAL_AGGREGATE_REWARDS_STEP18 = aggregate_rewards


def aggregate_rewards(
    action: SIEGEAction,
    *,
    ground_truth_root_cause: str,
    seat_role: str = "immune",
    claims_by_id: dict[str, dict[str, Any]] | None = None,
    trust_scores: dict[int, float] | None = None,
    agent_reliability: dict[int, bool] | None = None,
    urgency_multiplier: float = 1.0,
    incident_severity: str = "warning",
) -> tuple[float, dict[str, Any]]:
    base_total, components = _ORIGINAL_AGGREGATE_REWARDS_STEP18(
        action,
        ground_truth_root_cause=ground_truth_root_cause,
        seat_role=seat_role,
        claims_by_id=claims_by_id,
        trust_scores=trust_scores,
        agent_reliability=agent_reliability,
        urgency_multiplier=urgency_multiplier,
    )
    r8 = compute_r8_severity_speed(action, incident_severity=incident_severity)
    components = dict(components)
    components["r8_severity_speed"] = r8
    total = max(0.0, min(1.0, max(base_total, r8)))
    return total, components


# Step 19 append-only extension: include R7 postmortem quality reward
from siege_env.rewards.r7_postmortem import compute_r7_postmortem

_ORIGINAL_AGGREGATE_REWARDS_STEP19 = aggregate_rewards


def aggregate_rewards(
    action: SIEGEAction,
    *,
    ground_truth_root_cause: str,
    seat_role: str = "immune",
    claims_by_id: dict[str, dict[str, Any]] | None = None,
    trust_scores: dict[int, float] | None = None,
    agent_reliability: dict[int, bool] | None = None,
    urgency_multiplier: float = 1.0,
    incident_severity: str = "warning",
) -> tuple[float, dict[str, Any]]:
    base_total, components = _ORIGINAL_AGGREGATE_REWARDS_STEP19(
        action,
        ground_truth_root_cause=ground_truth_root_cause,
        seat_role=seat_role,
        claims_by_id=claims_by_id,
        trust_scores=trust_scores,
        agent_reliability=agent_reliability,
        urgency_multiplier=urgency_multiplier,
        incident_severity=incident_severity,
    )
    r7 = compute_r7_postmortem(action, ground_truth_root_cause=ground_truth_root_cause)
    components = dict(components)
    components["r7_postmortem"] = r7
    total = max(0.0, min(1.0, max(base_total, r7)))
    return total, components
