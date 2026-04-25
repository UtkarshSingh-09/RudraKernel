"""Unit tests for whisper/private channel mechanics."""

from __future__ import annotations

from siege_env.mechanics.whisper import WhisperEvent, build_whisper_event


class TestWhisper:
    def test_build_whisper_event(self) -> None:
        event = build_whisper_event(
            sender_agent_id=0, target_agent_id=3, message="Watch Agent4", step_number=5,
        )
        assert isinstance(event, WhisperEvent)
        assert event.sender_agent_id == 0
        assert event.target_agent_id == 3
        assert event.step_number == 5

    def test_whisper_to_dict(self) -> None:
        event = build_whisper_event(
            sender_agent_id=1, target_agent_id=2, message="Suspicious claim", step_number=3,
        )
        d = event.to_dict()
        assert d["sender_agent_id"] == 1
        assert d["target_agent_id"] == 2
        assert "message" in d

    def test_self_whisper_is_valid(self) -> None:
        event = build_whisper_event(
            sender_agent_id=0, target_agent_id=0, message="Note to self", step_number=1,
        )
        assert event.sender_agent_id == event.target_agent_id
