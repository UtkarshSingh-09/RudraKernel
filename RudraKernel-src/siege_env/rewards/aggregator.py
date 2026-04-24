"""Reward aggregation scaffold for SIEGE."""

from __future__ import annotations

from typing import Any

from siege_env.models import SIEGEAction
from siege_env.rewards.r1_resolution import compute_r1_resolution


def aggregate_rewards(
    action: SIEGEAction,
    *,
    ground_truth_root_cause: str,
) -> tuple[float, dict[str, Any]]:
    """Aggregate reward components (Step 04 uses only R1)."""

    r1 = compute_r1_resolution(action, ground_truth_root_cause)
    total = max(0.0, min(1.0, r1))
    return total, {"r1_resolution": r1}
