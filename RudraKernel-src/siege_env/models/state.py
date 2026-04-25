"""State dataclass for the SIEGE environment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping


@dataclass(slots=True)
class SIEGEState:
    """Serializable internal environment state snapshot."""

    episode_id: str
    step_count: int
    incident_template_id: str
    ground_truth_root_cause: str
    current_tier: int
    arms_race_score: float
    trigger_activated: bool = False
    cooperative_steps: int = 0
    trigger_step: int | None = None

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must be a non-empty string.")
        if self.step_count < 0:
            raise ValueError("step_count must be non-negative.")
        if self.cooperative_steps < 0:
            raise ValueError("cooperative_steps must be non-negative.")
        if not self.incident_template_id.strip():
            raise ValueError("incident_template_id must be a non-empty string.")
        if not self.ground_truth_root_cause.strip():
            raise ValueError("ground_truth_root_cause must be a non-empty string.")
        if self.current_tier not in {1, 2, 3}:
            raise ValueError("current_tier must be one of 1, 2, or 3.")
        if not isfinite(self.arms_race_score):
            raise ValueError("arms_race_score must be finite.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the state into a JSON-serializable mapping."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SIEGEState:
        """Construct a state object from a mapping or decoded JSON object."""

        return cls(
            episode_id=str(payload["episode_id"]),
            step_count=int(payload["step_count"]),
            incident_template_id=str(payload["incident_template_id"]),
            ground_truth_root_cause=str(payload["ground_truth_root_cause"]),
            current_tier=int(payload["current_tier"]),
            arms_race_score=float(payload["arms_race_score"]),
            trigger_activated=bool(payload.get("trigger_activated", False)),
            cooperative_steps=int(payload.get("cooperative_steps", 0)),
            trigger_step=(
                int(payload["trigger_step"])
                if payload.get("trigger_step") is not None
                else None
            ),
        )

    def to_json(self) -> str:
        """Serialize the state into JSON for snapshots or debugging."""

        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> SIEGEState:
        """Deserialize a state object from a JSON string."""

        return cls.from_dict(json.loads(payload))
