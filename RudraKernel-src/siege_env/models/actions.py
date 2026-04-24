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
