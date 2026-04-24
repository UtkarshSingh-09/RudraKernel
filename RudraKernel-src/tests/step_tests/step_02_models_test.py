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
