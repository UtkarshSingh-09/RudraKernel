"""Gate tests for Step 18 — Severity Escalation + R8."""

from __future__ import annotations

from siege_env.mechanics.severity_escalation import compute_incident_severity
from siege_env.models.actions import EscalateArgs, SIEGEAction
from siege_env.rewards.r8_severity_speed import compute_r8_severity_speed
from siege_env.server.siege_environment import SIEGEEnvironment


def test_incident_severity_progression_by_step() -> None:
    assert compute_incident_severity(0) == "warning"
    assert compute_incident_severity(2) == "critical"
    assert compute_incident_severity(5) == "outage"


def test_r8_rewards_fast_escalation_when_severity_outage() -> None:
    action = SIEGEAction(
        tool_name="escalate",
        arguments=EscalateArgs(concern="major outage", blast_radius_estimate=["db", "api"]),
    )
    assert compute_r8_severity_speed(action, incident_severity="outage") == 1.0


def test_r8_lower_reward_for_warning_escalation() -> None:
    action = SIEGEAction(
        tool_name="escalate",
        arguments=EscalateArgs(concern="minor signal", blast_radius_estimate=["api"]),
    )
    assert compute_r8_severity_speed(action, incident_severity="warning") == 0.2


def test_environment_observation_exposes_escalated_severity() -> None:
    env = SIEGEEnvironment(seed=15)
    obs = env.reset()
    assert obs.incident_severity in {"warning", "critical", "outage"}
    assert "severity_score" in obs.incident_dashboard


def test_exploit_always_escalate_not_maximal_in_warning_state() -> None:
    action = SIEGEAction(
        tool_name="escalate",
        arguments=EscalateArgs(concern="always escalate exploit", blast_radius_estimate=["x"]),
    )
    reward = compute_r8_severity_speed(action, incident_severity="warning")
    assert reward < 1.0
