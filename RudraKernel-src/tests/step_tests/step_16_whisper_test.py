"""Gate tests for Step 16 — Whisper / Private Channels."""

from __future__ import annotations

from siege_env.mechanics.whisper import build_whisper_event
from siege_env.models.actions import SIEGEAction, WhisperArgs
from siege_env.server.siege_environment import SIEGEEnvironment


def test_build_whisper_event_contains_expected_fields() -> None:
    event = build_whisper_event(
        sender_agent_id=0,
        target_agent_id=2,
        message="check replica lag",
        step_number=1,
    ).to_dict()
    assert event["sender_agent_id"] == 0
    assert event["target_agent_id"] == 2
    assert event["message"] == "check replica lag"


def test_whisper_action_logged_in_step_info_counter() -> None:
    env = SIEGEEnvironment(seed=5)
    env.reset()
    action = SIEGEAction(
        tool_name="whisper",
        arguments=WhisperArgs(target_agent_id=2, message="sync privately"),
    )
    _, _, _, info = env.step(action)
    assert info["whispers_logged"] >= 1


def test_whisper_log_visible_in_observation() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    action = SIEGEAction(
        tool_name="whisper",
        arguments=WhisperArgs(target_agent_id=3, message="private note"),
    )
    obs, _, _, _ = env.step(action)
    assert len(obs.whisper_log) >= 1


def test_whisper_inbox_receives_messages_to_seat_agent() -> None:
    env = SIEGEEnvironment(seed=9)
    env.reset()
    action = SIEGEAction(
        tool_name="whisper",
        arguments=WhisperArgs(target_agent_id=0, message="for you"),
    )
    obs, _, _, _ = env.step(action)
    assert len(obs.whisper_inbox) >= 1
