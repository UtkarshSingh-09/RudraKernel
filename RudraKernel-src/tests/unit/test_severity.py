"""Unit tests for severity escalation mechanics."""

from __future__ import annotations

from siege_env.mechanics.severity_escalation import compute_incident_severity


class TestSeverityEscalation:
    def test_early_step_is_warning(self) -> None:
        assert compute_incident_severity(0) == "warning"
        assert compute_incident_severity(1) == "warning"

    def test_mid_step_is_critical(self) -> None:
        assert compute_incident_severity(2) == "critical"
        assert compute_incident_severity(3) == "critical"

    def test_late_step_is_outage(self) -> None:
        assert compute_incident_severity(4) == "outage"
        assert compute_incident_severity(10) == "outage"

    def test_return_type_is_string(self) -> None:
        result = compute_incident_severity(2)
        assert isinstance(result, str)
        assert result in ("warning", "critical", "outage")
