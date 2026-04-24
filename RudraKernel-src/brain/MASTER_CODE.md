# MASTER CODE - Last Updated: 2026-04-24T17:53:09+00:00

# Files Tracked: 30

## siege_env/__init__.py (last modified: 2026-04-24T17:52:44+00:00)
```python
"""SIEGE environment package."""

from siege_env.models import SIEGEAction, SIEGEObservation, SIEGEState
from siege_env.server import SIEGEEnvironment

__all__ = ["SIEGEAction", "SIEGEEnvironment", "SIEGEObservation", "SIEGEState"]

```

## siege_env/agents/__init__.py (last modified: 2026-04-24T15:24:19+00:00)
```python
"""Agent population modules."""
```

## siege_env/curriculum/__init__.py (last modified: 2026-04-24T15:24:19+00:00)
```python
"""Curriculum scheduling modules."""
```

## siege_env/incidents/__init__.py (last modified: 2026-04-24T17:23:08+00:00)
```python
"""Incident templates and generation utilities."""

from siege_env.incidents.generator import generate_variant
from siege_env.incidents.loader import load_templates

__all__ = ["generate_variant", "load_templates"]

```

## siege_env/incidents/generator.py (last modified: 2026-04-24T17:23:03+00:00)
```python
"""Deterministic incident variant generation from seed templates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _rotated(values: list[str], offset: int) -> list[str]:
    if not values:
        return []
    normalized = offset % len(values)
    return values[normalized:] + values[:normalized]


def generate_variant(template: dict[str, Any], variant_index: int) -> dict[str, Any]:
    """Generate a deterministic variant while preserving schema contract."""

    if variant_index < 0:
        raise ValueError("variant_index must be non-negative.")

    variant = deepcopy(template)
    variant["id"] = f"{template['id']}_v{variant_index:03d}"
    variant["observable_signals"] = _rotated(list(template["observable_signals"]), variant_index)
    variant["flaw_types"] = _rotated(list(template["flaw_types"]), variant_index)
    variant["blast_radius"] = _rotated(list(template["blast_radius"]), variant_index)
    return variant

```

## siege_env/incidents/loader.py (last modified: 2026-04-24T17:22:57+00:00)
```python
"""Template loading and validation for SIEGE incident seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TEMPLATE_KEYS = (
    "id",
    "source_url",
    "root_cause",
    "observable_signals",
    "flaw_types",
    "blast_radius",
)
TEMPLATES_PATH = Path(__file__).with_name("templates.json")


def _validate_template(raw_template: dict[str, Any], index: int) -> dict[str, Any]:
    missing = [key for key in REQUIRED_TEMPLATE_KEYS if key not in raw_template]
    if missing:
        missing_joined = ", ".join(missing)
        raise ValueError(f"Template at index {index} is missing required keys: {missing_joined}")

    template = {key: raw_template[key] for key in REQUIRED_TEMPLATE_KEYS}
    if not isinstance(template["id"], str) or not template["id"].strip():
        raise ValueError(f"Template at index {index} has invalid 'id'.")
    if not isinstance(template["source_url"], str) or not template["source_url"].startswith("https://"):
        raise ValueError(f"Template '{template['id']}' has invalid 'source_url'.")
    if not isinstance(template["root_cause"], str) or not template["root_cause"].strip():
        raise ValueError(f"Template '{template['id']}' has invalid 'root_cause'.")

    for list_key in ("observable_signals", "flaw_types", "blast_radius"):
        value = template[list_key]
        if not isinstance(value, list) or not value:
            raise ValueError(f"Template '{template['id']}' has invalid '{list_key}'.")
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f"Template '{template['id']}' contains invalid values in '{list_key}'.")

    return {
        "id": template["id"].strip(),
        "source_url": template["source_url"].strip(),
        "root_cause": template["root_cause"].strip(),
        "observable_signals": [item.strip() for item in template["observable_signals"]],
        "flaw_types": [item.strip() for item in template["flaw_types"]],
        "blast_radius": [item.strip() for item in template["blast_radius"]],
    }


def load_templates(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate incident templates from disk."""

    target_path = path or TEMPLATES_PATH
    raw_payload = json.loads(target_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, list):
        raise ValueError("Incident templates payload must be a list.")

    return [_validate_template(item, idx) for idx, item in enumerate(raw_payload)]

```

## siege_env/league/__init__.py (last modified: 2026-04-24T15:24:19+00:00)
```python
"""Opponent league modules."""
```

## siege_env/mechanics/__init__.py (last modified: 2026-04-24T15:24:19+00:00)
```python
"""Mechanics modules."""
```

## siege_env/models/__init__.py (last modified: 2026-04-24T17:11:21+00:00)
```python
"""Data models for SIEGE."""

from siege_env.models.actions import (
    ACTION_ARGS_BY_TOOL,
    ChallengeArgs,
    DiagnoseArgs,
    EscalateArgs,
    PostmortemArgs,
    RatifyArgs,
    SIEGEAction,
    WhisperArgs,
)
from siege_env.models.observations import SIEGEObservation
from siege_env.models.state import SIEGEState

__all__ = [
    "ACTION_ARGS_BY_TOOL",
    "ChallengeArgs",
    "DiagnoseArgs",
    "EscalateArgs",
    "PostmortemArgs",
    "RatifyArgs",
    "SIEGEAction",
    "SIEGEObservation",
    "SIEGEState",
    "WhisperArgs",
]

```

## siege_env/models/actions.py (last modified: 2026-04-24T17:11:21+00:00)
```python
"""Strict action schemas for the SIEGE tool surface."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


ToolName: TypeAlias = Literal[
    "diagnose",
    "challenge",
    "ratify",
    "escalate",
    "whisper",
    "postmortem",
]
FlawType: TypeAlias = Literal[
    "type1_false_correlation",
    "type2_scope_inflation",
    "type3_tunnel_vision",
    "type4_blame_shifting",
    "type5_premature_closure",
]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=1500)]
MessageText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
ClaimId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
EvidenceItem = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class SIEGEBaseModel(BaseModel):
    """Shared strict settings for SIEGE Pydantic models."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AlternativeHypothesis(SIEGEBaseModel):
    """Fallback diagnosis candidate with a calibrated confidence score."""

    diagnosis: ShortText
    confidence: float = Field(ge=0.0, le=1.0)


class TimelineEvent(SIEGEBaseModel):
    """Structured postmortem event used during reflective summaries."""

    timestamp: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
    ]
    event: MessageText


class DiagnoseArgs(SIEGEBaseModel):
    """Primary diagnosis action emitted during live incident response."""

    root_cause: ShortText
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=10)
    alternative_hypotheses: list[AlternativeHypothesis] = Field(default_factory=list, max_length=5)


class ChallengeArgs(SIEGEBaseModel):
    """Challenge a peer claim with flaw taxonomy reasoning."""

    target_agent_id: int = Field(ge=0, le=7)
    claim_id: ClaimId
    flaw_type: FlawType
    reasoning: LongText


class RatifyArgs(SIEGEBaseModel):
    """Vote on whether a claim should be ratified by the coalition."""

    claim_id: ClaimId
    vote: bool


class EscalateArgs(SIEGEBaseModel):
    """Escalate an incident when the blast radius or severity expands."""

    concern: MessageText
    blast_radius_estimate: list[ShortText] = Field(min_length=1, max_length=10)


class WhisperArgs(SIEGEBaseModel):
    """Private channel between two agents."""

    target_agent_id: int = Field(ge=0, le=7)
    message: MessageText


class PostmortemArgs(SIEGEBaseModel):
    """Structured reflective summary emitted after incident resolution."""

    root_cause: ShortText
    timeline: list[TimelineEvent] = Field(min_length=1, max_length=20)
    contributing_factors: list[ShortText] = Field(min_length=1, max_length=10)
    misdiagnosis_analysis: LongText


ActionArguments: TypeAlias = (
    DiagnoseArgs | ChallengeArgs | RatifyArgs | EscalateArgs | WhisperArgs | PostmortemArgs
)

ACTION_ARGS_BY_TOOL: dict[ToolName, type[SIEGEBaseModel]] = {
    "diagnose": DiagnoseArgs,
    "challenge": ChallengeArgs,
    "ratify": RatifyArgs,
    "escalate": EscalateArgs,
    "whisper": WhisperArgs,
    "postmortem": PostmortemArgs,
}


class SIEGEAction(SIEGEBaseModel):
    """Validated action payload with tool-aware argument coercion."""

    tool_name: ToolName
    arguments: ActionArguments

    @model_validator(mode="before")
    @classmethod
    def _coerce_tool_specific_arguments(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        tool_name = data.get("tool_name")
        if tool_name not in ACTION_ARGS_BY_TOOL:
            return data

        arguments = data.get("arguments")
        if arguments is None:
            return data

        model_cls = ACTION_ARGS_BY_TOOL[tool_name]
        if isinstance(arguments, model_cls):
            return data

        updated = dict(data)
        updated["arguments"] = model_cls.model_validate(arguments)
        return updated

    @model_validator(mode="after")
    def _ensure_arguments_match_tool(self) -> SIEGEAction:
        expected_model = ACTION_ARGS_BY_TOOL[self.tool_name]
        if not isinstance(self.arguments, expected_model):
            raise ValueError(
                f"Arguments for tool '{self.tool_name}' must use {expected_model.__name__}."
            )
        return self

    @classmethod
    def tool_schema(cls, tool_name: ToolName) -> dict[str, Any]:
        """Return the JSON schema for a specific tool payload."""

        return ACTION_ARGS_BY_TOOL[tool_name].model_json_schema()

```

## siege_env/models/observations.py (last modified: 2026-04-24T17:11:21+00:00)
```python
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

```

## siege_env/models/state.py (last modified: 2026-04-24T17:11:21+00:00)
```python
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

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must be a non-empty string.")
        if self.step_count < 0:
            raise ValueError("step_count must be non-negative.")
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
        )

    def to_json(self) -> str:
        """Serialize the state into JSON for snapshots or debugging."""

        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> SIEGEState:
        """Deserialize a state object from a JSON string."""

        return cls.from_dict(json.loads(payload))

```

## siege_env/replay/__init__.py (last modified: 2026-04-24T15:24:19+00:00)
```python
"""Replay logging and playback modules."""
```

## siege_env/rewards/__init__.py (last modified: 2026-04-24T17:52:36+00:00)
```python
"""Reward modules and aggregators."""

from siege_env.rewards.aggregator import aggregate_rewards
from siege_env.rewards.r1_resolution import compute_r1_resolution

__all__ = ["aggregate_rewards", "compute_r1_resolution"]

```

## siege_env/rewards/aggregator.py (last modified: 2026-04-24T17:51:49+00:00)
```python
"""Reward aggregation scaffold for SIEGE."""

from __future__ import annotations

from typing import Any

from siege_env.models import SIEGEAction
from siege_env.rewards.r1_resolution import compute_r1_resolution


def aggregate_rewards(
    action: SIEGEAction,
    *,
    ground_truth_root_cause: str,
) -> tuple[float, dict[str, Any]]:
    """Aggregate reward components (Step 04 uses only R1)."""

    r1 = compute_r1_resolution(action, ground_truth_root_cause)
    total = max(0.0, min(1.0, r1))
    return total, {"r1_resolution": r1}

```

## siege_env/rewards/r1_resolution.py (last modified: 2026-04-24T17:51:44+00:00)
```python
"""Resolution reward (R1) for minimal Step 04 environment loop."""

from __future__ import annotations

from siege_env.models import SIEGEAction


def compute_r1_resolution(action: SIEGEAction, ground_truth_root_cause: str) -> float:
    """Return 1.0 for correct diagnose action, otherwise 0.0."""

    if action.tool_name != "diagnose":
        return 0.0

    predicted_root_cause = action.arguments.root_cause
    if predicted_root_cause == ground_truth_root_cause:
        return 1.0
    return 0.0

```

## siege_env/server/__init__.py (last modified: 2026-04-24T17:52:40+00:00)
```python
"""Server modules for SIEGE environment."""

from siege_env.server.siege_environment import SIEGEEnvironment

__all__ = ["SIEGEEnvironment"]

```

## siege_env/server/app.py (last modified: 2026-04-24T17:52:48+00:00)
```python
"""FastAPI server scaffold for SIEGE Step 01."""

from __future__ import annotations

from fastapi import FastAPI

from siege_env.server.siege_environment import SIEGEEnvironment


app = FastAPI(title="SIEGE Environment", version="0.1.0")
env = SIEGEEnvironment(seed=7)


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness endpoint used for local and container smoke tests."""
    return {"status": "ok"}


@app.get("/env/reset")
def reset() -> dict[str, object]:
    """Reset the minimal environment and return the starting observation."""
    observation = env.reset()
    return {"observation": observation.to_dict()}

```

## siege_env/server/siege_environment.py (last modified: 2026-04-24T17:52:32+00:00)
```python
"""Minimal Step 04 SIEGE environment implementation."""

from __future__ import annotations

from dataclasses import replace
from random import Random
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from siege_env.incidents import load_templates
from siege_env.models import SIEGEAction, SIEGEObservation, SIEGEState
from siege_env.rewards.aggregator import aggregate_rewards

try:
    from openenv import MCPEnvironment
except ImportError:  # pragma: no cover - fallback for local development.
    class MCPEnvironment:  # type: ignore[no-redef]
        """Fallback base when OpenEnv is not installed in local test environments."""


class SIEGEEnvironment(MCPEnvironment):
    """Single-seat SIEGE environment with minimal R1 reward loop."""

    def __init__(self, *, seed: int = 0, max_steps: int = 5) -> None:
        self._rng = Random(seed)
        self._max_steps = max_steps
        self._templates = load_templates()
        self._state: SIEGEState | None = None
        self._agent_claims: list[dict[str, Any]] = []
        self._done = False
        self._last_reward_components: dict[str, Any] = {"r1_resolution": 0.0}

    def reset(self) -> SIEGEObservation:
        template = self._rng.choice(self._templates)
        episode_id = f"episode-{uuid4().hex[:8]}"
        self._state = SIEGEState(
            episode_id=episode_id,
            step_count=0,
            incident_template_id=template["id"],
            ground_truth_root_cause=template["root_cause"],
            current_tier=1,
            arms_race_score=0.0,
        )
        self._agent_claims = []
        self._done = False
        self._last_reward_components = {"r1_resolution": 0.0}
        return self._build_observation(template=template, action_error=None)

    def step(self, action_payload: SIEGEAction | dict[str, Any]) -> tuple[SIEGEObservation, float, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        template = self._template_by_id(self._state.incident_template_id)
        if self._done:
            observation = self._build_observation(template=template, action_error=None)
            return observation, 0.0, True, {"already_done": True}

        self._state = replace(self._state, step_count=self._state.step_count + 1)

        try:
            action = SIEGEAction.model_validate(action_payload)
        except ValidationError as exc:
            self._done = self._state.step_count >= self._max_steps
            observation = self._build_observation(template=template, action_error=str(exc))
            return observation, -0.05, self._done, {"invalid_action": True}

        reward, components = aggregate_rewards(
            action,
            ground_truth_root_cause=self._state.ground_truth_root_cause,
        )
        self._last_reward_components = components

        if action.tool_name == "diagnose":
            self._agent_claims.append(
                {
                    "agent_id": 0,
                    "claim_id": f"claim-{self._state.step_count:03d}",
                    "root_cause": action.arguments.root_cause,
                }
            )

        self._done = (action.tool_name == "diagnose" and reward == 1.0) or (
            self._state.step_count >= self._max_steps
        )
        observation = self._build_observation(template=template, action_error=None)
        info = {"invalid_action": False, "reward_components": components}
        return observation, reward, self._done, info

    def state(self) -> SIEGEState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() before state().")
        return self._state

    def _template_by_id(self, template_id: str) -> dict[str, Any]:
        for template in self._templates:
            if template["id"] == template_id:
                return template
        raise RuntimeError(f"Template '{template_id}' not found.")

    def _build_observation(self, *, template: dict[str, Any], action_error: str | None) -> SIEGEObservation:
        if self._state is None:
            raise RuntimeError("Environment not initialized.")

        if self._state.step_count <= 1:
            severity = "warning"
        elif self._state.step_count <= 3:
            severity = "critical"
        else:
            severity = "outage"

        visible_signals = template["observable_signals"][: max(1, min(len(template["observable_signals"]), self._state.step_count + 1))]
        available_evidence = [{"type": "signal", "value": signal} for signal in visible_signals]
        active_status = "resolved" if self._done else "active"

        return SIEGEObservation(
            incident_dashboard={
                "template_id": template["id"],
                "signals": visible_signals,
            },
            agent_claims=list(self._agent_claims),
            trust_scores={idx: 0.5 for idx in range(1, 8)},
            coalition_status={"votes_for": [], "votes_against": []},
            step_number=self._state.step_count,
            slo_status={"breached": self._state.step_count >= self._max_steps},
            your_role="immune",
            available_evidence=available_evidence,
            visibility_level="full",
            whisper_inbox=[],
            whisper_log=[],
            incident_severity=severity,
            red_herring_signals=[],
            reputation_history={idx: 0.5 for idx in range(1, 8)},
            active_incidents=[
                {
                    "incident_id": template["id"],
                    "status": active_status,
                }
            ],
            seat_agent_id=0,
            action_error=action_error,
        )

```

## siege_env/trust/__init__.py (last modified: 2026-04-24T15:24:19+00:00)
```python
"""Trust and coalition logic modules."""
```

## siege_env/utils/__init__.py (last modified: 2026-04-24T15:24:19+00:00)
```python
"""Shared utility modules."""
```

## tests/__init__.py (last modified: 2026-04-24T17:09:33+00:00)
```python
"""Test package for SIEGE."""

```

## tests/conftest.py (last modified: 2026-04-24T17:09:33+00:00)
```python
"""Shared pytest fixtures and test path bootstrap for SIEGE."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

```

## tests/master_suite.py (last modified: 2026-04-24T17:51:39+00:00)
```python
"""Master test suite entrypoint aggregating the project test surface."""

from tests.step_tests.step_00_bootstrap_test import *  # noqa: F401,F403
from tests.step_tests.step_01_scaffold_test import *  # noqa: F401,F403
from tests.step_tests.step_02_models_test import *  # noqa: F401,F403
from tests.step_tests.step_03_incidents_test import *  # noqa: F401,F403
from tests.step_tests.step_04_minimal_env_test import *  # noqa: F401,F403

```

## tests/step_tests/__init__.py (last modified: 2026-04-24T17:09:33+00:00)
```python
"""Step-gated tests for SIEGE."""

```

## tests/step_tests/step_00_bootstrap_test.py (last modified: 2026-04-24T17:14:53+00:00)
```python
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_required_structure_exists() -> None:
    required_paths = [
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / ".pre-commit-config.yaml",
        ROOT / "pyproject.toml",
        ROOT / "Makefile",
        ROOT / "brain" / "tools" / "update_brain.py",
        ROOT / "brain" / "tools" / "compile_master_code.py",
        ROOT / "tests" / "step_tests" / "step_00_bootstrap_test.py",
    ]
    for path in required_paths:
        assert path.exists(), f"Missing required bootstrap path: {path}"


def test_ci_config_parses() -> None:
    ci_file = ROOT / ".github" / "workflows" / "ci.yml"
    data = yaml.safe_load(ci_file.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "jobs" in data


def test_make_test_all_runs_cleanly() -> None:
    if os.getenv("SIEGE_BOOTSTRAP_SELFTEST") == "1":
        return

    env = os.environ.copy()
    env["SIEGE_BOOTSTRAP_SELFTEST"] = "1"
    result = subprocess.run(
        ["make", "test-all"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_update_brain_creates_snapshot() -> None:
    tracked_files = {
        ROOT / "brain" / "MASTER_CODE.md": (ROOT / "brain" / "MASTER_CODE.md").read_text(
            encoding="utf-8"
        ),
        ROOT / "brain" / "CHANGELOG.md": (ROOT / "brain" / "CHANGELOG.md").read_text(
            encoding="utf-8"
        ),
        ROOT / "brain" / "CONTEXT.md": (ROOT / "brain" / "CONTEXT.md").read_text(
            encoding="utf-8"
        ),
        ROOT / "brain" / "ROADMAP_STATUS.md": (
            ROOT / "brain" / "ROADMAP_STATUS.md"
        ).read_text(encoding="utf-8"),
    }
    before = set((ROOT / "brain" / "snapshots").glob("step_00_*.json"))
    try:
        result = subprocess.run(
            [
                "python3",
                "brain/tools/update_brain.py",
                "--step",
                "00",
                "--title",
                "Bootstrap",
                "--owner",
                "Utkarsh",
                "--reviewer",
                "Ankit",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        after = set((ROOT / "brain" / "snapshots").glob("step_00_*.json"))
        assert len(after) >= len(before) + 1
    finally:
        after = set((ROOT / "brain" / "snapshots").glob("step_00_*.json"))
        for snapshot_path in after - before:
            snapshot_path.unlink()
        for file_path, original_contents in tracked_files.items():
            file_path.write_text(original_contents, encoding="utf-8")

```

## tests/step_tests/step_01_scaffold_test.py (last modified: 2026-04-24T17:20:36+00:00)
```python
from __future__ import annotations

import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    info = subprocess.run(
        ["docker", "info"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return info.returncode == 0


def test_import_and_health_route() -> None:
    from siege_env.server.app import app

    routes = {route.path for route in app.routes}
    assert "/health" in routes


def test_openenv_manifest_has_required_keys() -> None:
    manifest = ROOT / "openenv.yaml"
    assert manifest.exists()
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert data["name"] == "siege_env"
    assert data["runtime"]["entrypoint"] == "siege_env.server.app:app"
    assert data["runtime"]["healthcheck"] == "/health"


@pytest.mark.skipif(not _docker_ready(), reason="Docker daemon not available")
def test_docker_build_succeeds() -> None:
    result = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            "siege_env/server/Dockerfile",
            "-t",
            "siege-step01-test",
            ".",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.skipif(not _docker_ready(), reason="Docker daemon not available")
def test_container_health_endpoint() -> None:
    port = _free_port()
    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "-p",
            f"{port}:8000",
            "siege-step01-test",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + "\n" + run.stderr
    container_id = run.stdout.strip()
    try:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + 20
        last_error = ""
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    body = response.read().decode("utf-8")
                    assert response.status == 200
                    assert "ok" in body
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                last_error = str(exc)
                time.sleep(0.5)
        raise AssertionError(f"Container health endpoint did not become ready: {last_error}")
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
```

## tests/step_tests/step_02_models_test.py (last modified: 2026-04-24T17:09:33+00:00)
```python
from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from siege_env.models.actions import (
    ChallengeArgs,
    DiagnoseArgs,
    EscalateArgs,
    PostmortemArgs,
    RatifyArgs,
    SIEGEAction,
    WhisperArgs,
)
from siege_env.models.observations import SIEGEObservation
from siege_env.models.state import SIEGEState


def build_observation() -> SIEGEObservation:
    return SIEGEObservation(
        incident_dashboard={"alerts": [{"name": "latency_p99", "value": 420}]},
        agent_claims=[
            {
                "agent_id": 2,
                "claim_id": "claim-001",
                "root_cause": "query_plan_regression",
            }
        ],
        trust_scores={1: 0.35, 2: 0.82, 3: 0.54},
        coalition_status={"claim_id": "claim-001", "votes_for": [2], "votes_against": []},
        step_number=3,
        slo_status={"budget_remaining_pct": 12.5, "breached": False},
        your_role="immune",
        available_evidence=[{"type": "metric", "name": "latency_p99"}],
        visibility_level="full",
        whisper_inbox=[{"from_agent": 3, "message": "Check the database traces."}],
        whisper_log=[{"from_agent": 3, "to_agent": 0}],
        incident_severity="critical",
        red_herring_signals=[{"type": "deploy", "service": "frontend"}],
        reputation_history={1: 0.40, 2: 0.91},
        active_incidents=[{"incident_id": "inc-001", "status": "active"}],
        seat_agent_id=0,
        action_error=None,
    )


def build_state() -> SIEGEState:
    return SIEGEState(
        episode_id="episode-001",
        step_count=3,
        incident_template_id="cloudflare-regex-2019",
        ground_truth_root_cause="regex_backtracking",
        current_tier=2,
        arms_race_score=0.27,
    )


def test_diagnose_args_accept_valid_payload() -> None:
    args = DiagnoseArgs.model_validate(
        {
            "root_cause": "query_plan_regression",
            "confidence": 0.72,
            "evidence": ["latency_p99_spike", "no_traffic_increase"],
            "alternative_hypotheses": [
                {"diagnosis": "connection_pool_exhaustion", "confidence": 0.18},
                {"diagnosis": "load_spike", "confidence": 0.10},
            ],
        }
    )
    assert args.root_cause == "query_plan_regression"
    assert len(args.alternative_hypotheses) == 2


def test_diagnose_args_reject_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        DiagnoseArgs.model_validate(
            {
                "root_cause": "query_plan_regression",
                "confidence": 1.2,
                "evidence": ["latency_p99_spike"],
                "alternative_hypotheses": [],
            }
        )


def test_challenge_args_reject_unknown_flaw_type() -> None:
    with pytest.raises(ValidationError):
        ChallengeArgs.model_validate(
            {
                "target_agent_id": 4,
                "claim_id": "claim-001",
                "flaw_type": "made_up_taxonomy",
                "reasoning": "This claim does not line up with the observed trace timings.",
            }
        )


def test_ratify_args_reject_blank_claim_id() -> None:
    with pytest.raises(ValidationError):
        RatifyArgs.model_validate({"claim_id": "   ", "vote": True})


def test_escalate_args_round_trip_json() -> None:
    args = EscalateArgs.model_validate(
        {
            "concern": "Blast radius is spreading across payments and API traffic.",
            "blast_radius_estimate": ["payments", "api-gateway"],
        }
    )
    restored = EscalateArgs.model_validate_json(args.model_dump_json())
    assert restored == args


def test_whisper_args_reject_invalid_target_agent() -> None:
    with pytest.raises(ValidationError):
        WhisperArgs.model_validate({"target_agent_id": 8, "message": "Check agent 5."})


def test_postmortem_args_requires_non_empty_timeline() -> None:
    with pytest.raises(ValidationError):
        PostmortemArgs.model_validate(
            {
                "root_cause": "regex_backtracking",
                "timeline": [],
                "contributing_factors": ["unsafe regex rollout"],
                "misdiagnosis_analysis": "We anchored on load instead of tracing the regex path.",
            }
        )


def test_siege_action_parses_tool_specific_arguments() -> None:
    action = SIEGEAction.model_validate(
        {
            "tool_name": "diagnose",
            "arguments": {
                "root_cause": "query_plan_regression",
                "confidence": 0.72,
                "evidence": ["latency_p99_spike"],
                "alternative_hypotheses": [],
            },
        }
    )
    assert isinstance(action.arguments, DiagnoseArgs)

    with pytest.raises(ValidationError):
        SIEGEAction.model_validate(
            {
                "tool_name": "diagnose",
                "arguments": {
                    "target_agent_id": 1,
                    "message": "This is clearly the wrong payload.",
                },
            }
        )


def test_observation_round_trip_json_and_validation() -> None:
    observation = build_observation()
    restored = SIEGEObservation.from_json(observation.to_json())
    assert restored == observation

    with pytest.raises(ValueError):
        replace(observation, incident_severity="catastrophic")


def test_state_round_trip_json_and_validation() -> None:
    state = build_state()
    restored = SIEGEState.from_json(state.to_json())
    assert restored == state

    with pytest.raises(ValueError):
        replace(state, current_tier=4)

```

## tests/step_tests/step_03_incidents_test.py (last modified: 2026-04-24T17:22:19+00:00)
```python
from __future__ import annotations

from siege_env.incidents.generator import generate_variant
from siege_env.incidents.loader import load_templates


REQUIRED_KEYS = {
    "id",
    "source_url",
    "root_cause",
    "observable_signals",
    "flaw_types",
    "blast_radius",
}


def test_all_five_seed_templates_load() -> None:
    templates = load_templates()
    assert len(templates) == 5


def test_templates_contain_required_ground_truth_fields() -> None:
    templates = load_templates()
    for template in templates:
        assert REQUIRED_KEYS.issubset(set(template.keys()))
        assert template["id"]
        assert template["root_cause"]
        assert template["source_url"].startswith("https://")
        assert isinstance(template["observable_signals"], list)
        assert isinstance(template["flaw_types"], list)
        assert isinstance(template["blast_radius"], list)


def test_variant_generator_produces_valid_variant() -> None:
    template = load_templates()[0]
    variant = generate_variant(template, variant_index=7)
    assert REQUIRED_KEYS.issubset(set(variant.keys()))
    assert variant["id"].startswith(f"{template['id']}_v")
    assert variant["root_cause"] == template["root_cause"]
    assert variant["source_url"] == template["source_url"]
    assert len(variant["observable_signals"]) == len(template["observable_signals"])

```

## tests/step_tests/step_04_minimal_env_test.py (last modified: 2026-04-24T17:51:36+00:00)
```python
from __future__ import annotations

from siege_env.models import SIEGEAction, SIEGEObservation, SIEGEState
from siege_env.server.siege_environment import SIEGEEnvironment


def _valid_diagnose_action(root_cause: str) -> dict[str, object]:
    return {
        "tool_name": "diagnose",
        "arguments": {
            "root_cause": root_cause,
            "confidence": 0.81,
            "evidence": ["latency_p99_spike"],
            "alternative_hypotheses": [],
        },
    }


def test_reset_returns_valid_observation() -> None:
    env = SIEGEEnvironment(seed=7)
    obs = env.reset()
    assert isinstance(obs, SIEGEObservation)
    assert obs.step_number == 0
    assert obs.action_error is None


def test_step_accepts_valid_action() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    action = SIEGEAction.model_validate(_valid_diagnose_action(env.state().ground_truth_root_cause))
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, SIEGEObservation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_done_is_reachable() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    _, _, done, _ = env.step(_valid_diagnose_action(env.state().ground_truth_root_cause))
    assert done is True


def test_reward_is_clamped_between_zero_and_one() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    _, reward, _, _ = env.step(_valid_diagnose_action("wrong_cause"))
    assert 0.0 <= reward <= 1.0


def test_state_serializes_round_trip() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    state = env.state()
    restored = SIEGEState.from_json(state.to_json())
    assert restored == state


def test_invalid_action_is_handled_gracefully() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    obs, reward, done, info = env.step({"tool_name": "diagnose", "arguments": {"bad": "payload"}})
    assert reward == -0.05
    assert done is False
    assert obs.action_error is not None
    assert info["invalid_action"] is True


def test_multi_step_episode_works() -> None:
    env = SIEGEEnvironment(seed=7, max_steps=3)
    env.reset()
    _, _, done_1, _ = env.step(_valid_diagnose_action("wrong_cause"))
    _, _, done_2, _ = env.step(
        {
            "tool_name": "escalate",
            "arguments": {
                "concern": "potential blast radius increase",
                "blast_radius_estimate": ["api-gateway"],
            },
        }
    )
    _, reward_3, done_3, _ = env.step(_valid_diagnose_action(env.state().ground_truth_root_cause))
    assert done_1 is False
    assert done_2 is False
    assert done_3 is True
    assert reward_3 == 1.0

```
