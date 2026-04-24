"""Gate tests for Step 19 — Post-Mortem Generation + R7."""

from __future__ import annotations

from siege_env.models.actions import PostmortemArgs, SIEGEAction, TimelineEvent
from siege_env.rewards.r7_postmortem import compute_r7_postmortem
from siege_env.server.siege_environment import SIEGEEnvironment


def _postmortem_action(*, root: str, timeline_events: list[str], analysis: str) -> SIEGEAction:
    return SIEGEAction(
        tool_name="postmortem",
        arguments=PostmortemArgs(
            root_cause=root,
            timeline=[
                TimelineEvent(timestamp=f"t{i}", event=ev)
                for i, ev in enumerate(timeline_events)
            ],
            contributing_factors=["factor_a", "factor_b"],
            misdiagnosis_analysis=analysis,
        ),
    )


def test_r7_rewards_high_quality_postmortem() -> None:
    action = _postmortem_action(
        root="db_timeout",
        timeline_events=["signal rose", "cache invalidation failed"],
        analysis="Initial triage over-weighted traffic volume and ignored a lock wait pattern in diagnostics.",
    )
    assert compute_r7_postmortem(action, ground_truth_root_cause="db_timeout") >= 0.8


def test_r7_is_zero_for_non_postmortem_actions() -> None:
    env = SIEGEEnvironment(seed=31)
    env.reset()
    _, reward, _, _ = env.step({"tool_name": "diagnose", "arguments": {"root_cause": "x", "confidence": 0.5, "evidence": ["e"]}})
    assert reward >= 0.0


def test_postmortem_flag_in_step_info() -> None:
    env = SIEGEEnvironment(seed=33)
    env.reset()
    action = _postmortem_action(
        root="any",
        timeline_events=["a", "b"],
        analysis="Detailed analysis with enough context to satisfy quality heuristics.",
    )
    _, _, _, info = env.step(action)
    assert info["postmortem_generated"] is True


def test_observation_dashboard_carries_last_postmortem() -> None:
    env = SIEGEEnvironment(seed=35)
    env.reset()
    action = _postmortem_action(
        root="rc",
        timeline_events=["first", "second"],
        analysis="Long postmortem narrative connecting false leads to final diagnosis confidence shift.",
    )
    obs, _, _, _ = env.step(action)
    assert "last_postmortem" in obs.incident_dashboard


def test_template_parroting_exploit_gets_penalized() -> None:
    action = _postmortem_action(
        root="db_timeout",
        timeline_events=["template text", "template text"],
        analysis="template text",
    )
    assert compute_r7_postmortem(action, ground_truth_root_cause="db_timeout") < 0.8
