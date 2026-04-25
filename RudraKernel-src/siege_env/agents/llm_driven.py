"""Tier-2 LLM-driven NPC scaffold.

For hackathon safety and deterministic local tests, this class currently uses a
prompt-template fallback that behaves deterministically without network calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any


@dataclass(slots=True)
class LLMDrivenAgent:
    agent_id: int
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = Random(self.seed + self.agent_id)

    def generate_claim(self, incident: dict[str, Any], *, step_number: int) -> dict[str, Any]:
        signals = list(incident.get("observable_signals", []))
        root_cause = str(incident.get("root_cause", "unknown"))
        focus = signals[step_number % len(signals)] if signals else "unknown_signal"
        confidence = round(0.55 + self._rng.random() * 0.35, 2)
        return {
            "agent_id": self.agent_id,
            "claim_id": f"llm-{self.agent_id}-{step_number}",
            "root_cause": root_cause,
            "confidence": confidence,
            "reasoning": f"Signal {focus} is consistent with {root_cause}.",
        }
