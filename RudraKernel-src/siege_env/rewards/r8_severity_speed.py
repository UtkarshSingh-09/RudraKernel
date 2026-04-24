"""R8 severity-speed reward component (Step 18)."""

from __future__ import annotations

from siege_env.models import SIEGEAction


def compute_r8_severity_speed(action: SIEGEAction, *, incident_severity: str) -> float:
    if incident_severity == "outage":
        if action.tool_name == "escalate":
            return 1.0
        if action.tool_name == "diagnose":
            return 0.6
        return 0.0

    if incident_severity == "critical":
        if action.tool_name == "escalate":
            return 0.8
        if action.tool_name == "diagnose":
            return 0.5
        return 0.0

    # warning state: avoid rewarding premature escalation exploit
    if action.tool_name == "escalate":
        return 0.2
    if action.tool_name == "diagnose":
        return 0.4
    return 0.0
