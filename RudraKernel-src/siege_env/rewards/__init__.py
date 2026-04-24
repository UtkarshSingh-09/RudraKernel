"""Reward modules and aggregators."""

from siege_env.rewards.aggregator import aggregate_rewards
from siege_env.rewards.r1_resolution import compute_r1_resolution
from siege_env.rewards.r2_deception import compute_r2_deception
from siege_env.rewards.r3_detection import compute_r3_detection
from siege_env.rewards.r4_trust_calibration import compute_r4_trust_calibration

__all__ = [
    "aggregate_rewards",
    "compute_r1_resolution",
    "compute_r2_deception",
    "compute_r3_detection",
    "compute_r4_trust_calibration",
]
