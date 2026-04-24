"""Information asymmetry mechanics for SIEGE Step 15."""

from __future__ import annotations


def visibility_for_step(step_number: int, role: str) -> str:
    if role == "pathogen":
        return "delayed"
    if step_number <= 1:
        return "metrics_only"
    if step_number <= 3:
        return "traces_only"
    return "full"


def filter_evidence_for_visibility(
    evidence: list[dict[str, object]],
    *,
    visibility_level: str,
) -> list[dict[str, object]]:
    if visibility_level == "full":
        return list(evidence)
    if visibility_level == "traces_only":
        return list(evidence[: max(1, min(2, len(evidence)))])
    if visibility_level == "metrics_only":
        return list(evidence[:1])
    if visibility_level == "delayed":
        return list(evidence[: max(1, min(1, len(evidence)))])
    return list(evidence)
