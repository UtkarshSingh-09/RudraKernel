"""Observation dataclass for the SIEGE environment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Literal, Mapping


Role = Literal["immune", "pathogen"]
VisibilityLevel = Literal["metrics_only", "traces_only", "full", "delayed"]
IncidentSeverity = Literal["warning", "critical", "outage"]

_ALLOWED_ROLES = {"immune", "pathogen"}
_ALLOWED_VISIBILITY_LEVELS = {"metrics_only", "traces_only", "full", "delayed"}
_ALLOWED_SEVERITIES = {"warning", "critical", "outage"}


def _normalize_agent_scores(
    raw_mapping: Mapping[object, object], *, field_name: str
) -> dict[int, float]:
    normalized: dict[int, float] = {}
    for raw_agent_id, raw_score in raw_mapping.items():
        try:
            agent_id = int(raw_agent_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} keys must be integer-like agent IDs.") from exc

        score = float(raw_score)
        if agent_id < 0 or agent_id > 7:
            raise ValueError(f"{field_name} keys must be between 0 and 7.")
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{field_name} values must be finite scores between 0.0 and 1.0.")

        normalized[agent_id] = score
    return normalized


@dataclass(slots=True)
class SIEGEObservation:
    """Serializable observation emitted to the seat agent each step."""

    incident_dashboard: dict[str, Any]
    agent_claims: list[dict[str, Any]]
    trust_scores: dict[int, float]
    coalition_status: dict[str, Any]
    step_number: int
    slo_status: dict[str, Any]
    your_role: Role
    available_evidence: list[dict[str, Any]]
    visibility_level: VisibilityLevel
    whisper_inbox: list[dict[str, Any]]
    whisper_log: list[dict[str, Any]]
    incident_severity: IncidentSeverity
    red_herring_signals: list[dict[str, Any]]
    reputation_history: dict[int, float]
    active_incidents: list[dict[str, Any]]
    seat_agent_id: int
    action_error: str | None = None

    def __post_init__(self) -> None:
        if self.step_number < 0:
            raise ValueError("step_number must be non-negative.")
        if self.your_role not in _ALLOWED_ROLES:
            raise ValueError(f"your_role must be one of {_ALLOWED_ROLES}.")
        if self.visibility_level not in _ALLOWED_VISIBILITY_LEVELS:
            raise ValueError(
                f"visibility_level must be one of {_ALLOWED_VISIBILITY_LEVELS}."
            )
        if self.incident_severity not in _ALLOWED_SEVERITIES:
            raise ValueError(f"incident_severity must be one of {_ALLOWED_SEVERITIES}.")
        if self.seat_agent_id < 0 or self.seat_agent_id > 7:
            raise ValueError("seat_agent_id must be between 0 and 7.")
        if self.action_error is not None and not self.action_error.strip():
            raise ValueError("action_error must be a non-empty string when provided.")

        self.trust_scores = _normalize_agent_scores(
            self.trust_scores,
            field_name="trust_scores",
        )
        self.reputation_history = _normalize_agent_scores(
            self.reputation_history,
            field_name="reputation_history",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the observation into a JSON-serializable mapping."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SIEGEObservation:
        """Construct an observation from a mapping or decoded JSON object."""

        return cls(
            incident_dashboard=dict(payload["incident_dashboard"]),
            agent_claims=list(payload["agent_claims"]),
            trust_scores=_normalize_agent_scores(
                payload["trust_scores"],
                field_name="trust_scores",
            ),
            coalition_status=dict(payload["coalition_status"]),
            step_number=int(payload["step_number"]),
            slo_status=dict(payload["slo_status"]),
            your_role=payload["your_role"],
            available_evidence=list(payload["available_evidence"]),
            visibility_level=payload["visibility_level"],
            whisper_inbox=list(payload["whisper_inbox"]),
            whisper_log=list(payload["whisper_log"]),
            incident_severity=payload["incident_severity"],
            red_herring_signals=list(payload["red_herring_signals"]),
            reputation_history=_normalize_agent_scores(
                payload["reputation_history"],
                field_name="reputation_history",
            ),
            active_incidents=list(payload["active_incidents"]),
            seat_agent_id=int(payload["seat_agent_id"]),
            action_error=payload.get("action_error"),
        )

    def to_json(self) -> str:
        """Serialize the observation into JSON for replay or transport."""

        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> SIEGEObservation:
        """Deserialize an observation from a JSON string."""

        return cls.from_dict(json.loads(payload))
