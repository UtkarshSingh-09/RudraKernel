"""Reward modules and aggregators."""

from siege_env.rewards.aggregator import aggregate_rewards
from siege_env.rewards.r1_resolution import R1_RUBRIC, compute_r1_resolution
from siege_env.rewards.r2_deception import R2_RUBRIC, compute_r2_deception
from siege_env.rewards.r3_detection import R3_RUBRIC, compute_r3_detection
from siege_env.rewards.r4_trust_calibration import R4_RUBRIC, compute_r4_trust_calibration
from siege_env.rewards.r5_confidence import R5_RUBRIC, ConfidenceCalibrator, compute_r5_confidence
from siege_env.rewards.r6_temporal import R6_RUBRIC, compute_r6_temporal
from siege_env.rewards.r7_postmortem import R7_RUBRIC, compute_r7_postmortem
from siege_env.rewards.r8_severity_speed import R8_RUBRIC, compute_r8_severity_speed
from siege_env.rewards.r9_correlation import R9_RUBRIC, compute_r9_correlation

COMPOSED_RUBRICS = [
    R1_RUBRIC,
    R2_RUBRIC,
    R3_RUBRIC,
    R4_RUBRIC,
    R5_RUBRIC,
    R6_RUBRIC,
    R7_RUBRIC,
    R8_RUBRIC,
    R9_RUBRIC,
]

__all__ = [
    "aggregate_rewards",
    "compute_r1_resolution",
    "compute_r2_deception",
    "compute_r3_detection",
    "compute_r4_trust_calibration",
    "compute_r5_confidence",
    "ConfidenceCalibrator",
    "compute_r6_temporal",
    "compute_r7_postmortem",
    "compute_r8_severity_speed",
    "compute_r9_correlation",
    "COMPOSED_RUBRICS",
]
