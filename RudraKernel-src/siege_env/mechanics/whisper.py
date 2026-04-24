"""Whisper/private channel mechanics for SIEGE Step 16."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WhisperEvent:
    sender_agent_id: int
    target_agent_id: int
    message: str
    step_number: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sender_agent_id": self.sender_agent_id,
            "target_agent_id": self.target_agent_id,
            "message": self.message,
            "step_number": self.step_number,
        }


def build_whisper_event(
    *,
    sender_agent_id: int,
    target_agent_id: int,
    message: str,
    step_number: int,
) -> WhisperEvent:
    return WhisperEvent(
        sender_agent_id=sender_agent_id,
        target_agent_id=target_agent_id,
        message=message,
        step_number=step_number,
    )
