"""Severity escalation mechanics for SIEGE Step 18."""

from __future__ import annotations


def compute_incident_severity(step_number: int) -> str:
    if step_number <= 1:
        return "warning"
    if step_number <= 3:
        return "critical"
    return "outage"


def severity_score(severity: str) -> float:
    mapping = {"warning": 0.3, "critical": 0.7, "outage": 1.0}
    return mapping.get(severity, 0.0)
