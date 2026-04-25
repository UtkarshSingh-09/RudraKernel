"""Unit tests for SIEGE Pydantic action + observation + state models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

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


# --- Action Schema Tests ---

class TestDiagnoseArgs:
    def test_valid_diagnose(self) -> None:
        args = DiagnoseArgs(root_cause="db_overload", confidence=0.85, evidence=["log1"])
        assert args.root_cause == "db_overload"
        assert args.confidence == 0.85

    def test_empty_root_cause_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DiagnoseArgs(root_cause="", confidence=0.5, evidence=["e1"])

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            DiagnoseArgs(root_cause="test", confidence=1.5, evidence=["e1"])

    def test_empty_evidence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DiagnoseArgs(root_cause="test", confidence=0.5, evidence=[])


class TestChallengeArgs:
    def test_valid_challenge(self) -> None:
        args = ChallengeArgs(
            target_agent_id=3,
            claim_id="claim-001",
            flaw_type="type1_false_correlation",
            reasoning="This is clearly a false correlation because the timing is wrong.",
        )
        assert args.target_agent_id == 3
        assert args.flaw_type == "type1_false_correlation"

    def test_invalid_flaw_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChallengeArgs(
                target_agent_id=1,
                claim_id="c1",
                flaw_type="not_a_flaw",  # type: ignore[arg-type]
                reasoning="x" * 20,
            )


class TestSIEGEAction:
    def test_diagnose_round_trip(self) -> None:
        payload = {
            "tool_name": "diagnose",
            "arguments": {"root_cause": "db_crash", "confidence": 0.9, "evidence": ["log"]},
        }
        action = SIEGEAction.model_validate(payload)
        assert action.tool_name == "diagnose"
        assert isinstance(action.arguments, DiagnoseArgs)

    def test_json_round_trip(self) -> None:
        payload = {
            "tool_name": "ratify",
            "arguments": {"claim_id": "claim-42", "vote": True},
        }
        action = SIEGEAction.model_validate(payload)
        dumped = action.model_dump()
        restored = SIEGEAction.model_validate(dumped)
        assert restored.tool_name == action.tool_name

    def test_wrong_arguments_for_tool_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SIEGEAction.model_validate({
                "tool_name": "diagnose",
                "arguments": {"claim_id": "c1", "vote": True},  # ratify args, not diagnose
            })

    def test_invalid_tool_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SIEGEAction.model_validate({
                "tool_name": "reset",  # reserved name
                "arguments": {},
            })

    def test_all_six_tools_have_schema(self) -> None:
        for tool in ("diagnose", "challenge", "ratify", "escalate", "whisper", "postmortem"):
            schema = SIEGEAction.tool_schema(tool)  # type: ignore[arg-type]
            assert isinstance(schema, dict)
            assert "properties" in schema


# --- Observation Tests ---

class TestSIEGEObservation:
    @pytest.fixture()
    def valid_obs_data(self) -> dict:
        return {
            "incident_dashboard": {"template_id": "test"},
            "agent_claims": [],
            "trust_scores": {1: 0.5, 2: 0.6},
            "coalition_status": {"votes_for": [], "votes_against": []},
            "step_number": 0,
            "slo_status": {"breached": False},
            "your_role": "immune",
            "available_evidence": [],
            "visibility_level": "full",
            "whisper_inbox": [],
            "whisper_log": [],
            "incident_severity": "warning",
            "red_herring_signals": [],
            "reputation_history": {1: 0.5, 2: 0.6},
            "active_incidents": [],
            "seat_agent_id": 0,
        }

    def test_valid_observation_creates(self, valid_obs_data: dict) -> None:
        obs = SIEGEObservation(**valid_obs_data)
        assert obs.step_number == 0

    def test_negative_step_rejected(self, valid_obs_data: dict) -> None:
        valid_obs_data["step_number"] = -1
        with pytest.raises(ValueError):
            SIEGEObservation(**valid_obs_data)

    def test_invalid_role_rejected(self, valid_obs_data: dict) -> None:
        valid_obs_data["your_role"] = "villain"
        with pytest.raises(ValueError):
            SIEGEObservation(**valid_obs_data)

    def test_json_round_trip(self, valid_obs_data: dict) -> None:
        obs = SIEGEObservation(**valid_obs_data)
        j = obs.to_json()
        restored = SIEGEObservation.from_json(j)
        assert restored.step_number == obs.step_number

    def test_dict_round_trip(self, valid_obs_data: dict) -> None:
        obs = SIEGEObservation(**valid_obs_data)
        d = obs.to_dict()
        restored = SIEGEObservation.from_dict(d)
        assert restored.your_role == obs.your_role


# --- State Tests ---

class TestSIEGEState:
    def test_valid_state(self) -> None:
        state = SIEGEState(
            episode_id="ep-001",
            step_count=0,
            incident_template_id="gitlab_2017",
            ground_truth_root_cause="db_deletion",
            current_tier=1,
            arms_race_score=0.0,
        )
        assert state.episode_id == "ep-001"

    def test_empty_episode_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            SIEGEState(
                episode_id="  ",
                step_count=0,
                incident_template_id="t1",
                ground_truth_root_cause="rc",
                current_tier=1,
                arms_race_score=0.0,
            )

    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(ValueError):
            SIEGEState(
                episode_id="ep-1",
                step_count=0,
                incident_template_id="t1",
                ground_truth_root_cause="rc",
                current_tier=4,
                arms_race_score=0.0,
            )

    def test_json_round_trip(self) -> None:
        state = SIEGEState(
            episode_id="ep-1", step_count=3, incident_template_id="t1",
            ground_truth_root_cause="rc", current_tier=2, arms_race_score=0.5,
        )
        j = state.to_json()
        restored = SIEGEState.from_json(j)
        assert restored.step_count == state.step_count
