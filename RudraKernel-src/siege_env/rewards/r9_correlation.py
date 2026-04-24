"""R9 correlation reward component (Step 17)."""

from __future__ import annotations

from typing import Any

from siege_env.models import SIEGEAction


FALSE_CORRELATION = "type1_false_correlation"


def compute_r9_correlation(
    action: SIEGEAction,
    *,
    claims_by_id: dict[str, dict[str, Any]] | None = None,
    ground_truth_root_cause: str = "",
) -> float:
    if action.tool_name != "challenge":
        return 0.0
    if action.arguments.flaw_type != FALSE_CORRELATION:
        return 0.0

    claim = (claims_by_id or {}).get(action.arguments.claim_id)
    if not claim:
        return 0.1

    claim_root = str(claim.get("root_cause", "")).strip()
    return 1.0 if claim_root and claim_root != ground_truth_root_cause else 0.0
