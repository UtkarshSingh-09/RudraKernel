from __future__ import annotations

from siege_env.models import SIEGEAction
from siege_env.rewards.aggregator import aggregate_rewards
from siege_env.rewards.r4_trust_calibration import compute_r4_trust_calibration
from siege_env.server.siege_environment import SIEGEEnvironment


def _diagnose(root_cause: str) -> SIEGEAction:
    return SIEGEAction.model_validate(
        {
            "tool_name": "diagnose",
            "arguments": {
                "root_cause": root_cause,
                "confidence": 0.77,
                "evidence": ["latency_p99_spike"],
                "alternative_hypotheses": [],
            },
        }
    )


def test_r4_perfect_calibration_scores_one() -> None:
    score = compute_r4_trust_calibration(
        trust_scores={1: 1.0, 2: 0.0, 3: 1.0},
        agent_reliability={1: True, 2: False, 3: True},
    )
    assert score == 1.0


def test_r4_uniform_trust_exploit_is_penalized() -> None:
    score = compute_r4_trust_calibration(
        trust_scores={1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
        agent_reliability={1: True, 2: False, 3: True, 4: False},
    )
    assert score == 0.75


def test_aggregator_includes_r4_component() -> None:
    action = _diagnose("wrong_cause")
    total, components = aggregate_rewards(
        action,
        ground_truth_root_cause="true_cause",
        seat_role="immune",
        claims_by_id={},
        trust_scores={1: 0.9, 2: 0.1},
        agent_reliability={1: True, 2: False},
    )
    assert "r4_trust_calibration" in components
    assert 0.0 <= components["r4_trust_calibration"] <= 1.0
    assert 0.0 <= total <= 1.0


def test_exploit_r1_non_diagnose_spam_gets_no_r1() -> None:
    action = SIEGEAction.model_validate(
        {
            "tool_name": "escalate",
            "arguments": {"concern": "spam", "blast_radius_estimate": ["api"]},
        }
    )
    _, components = aggregate_rewards(
        action,
        ground_truth_root_cause="true_cause",
        seat_role="immune",
        claims_by_id={},
        trust_scores={},
        agent_reliability={},
    )
    assert components["r1_resolution"] == 0.0


def test_exploit_r2_pathogen_truthful_diagnose_gets_no_r2() -> None:
    action = _diagnose("true_cause")
    _, components = aggregate_rewards(
        action,
        ground_truth_root_cause="true_cause",
        seat_role="pathogen",
        claims_by_id={},
        trust_scores={},
        agent_reliability={},
    )
    assert components["r2_deception"] == 0.0


def test_exploit_r3_challenge_unknown_claim_gets_no_r3() -> None:
    action = SIEGEAction.model_validate(
        {
            "tool_name": "challenge",
            "arguments": {
                "target_agent_id": 1,
                "claim_id": "missing-claim",
                "flaw_type": "type1_false_correlation",
                "reasoning": "Might be wrong based on observed signals and timing drift.",
            },
        }
    )
    _, components = aggregate_rewards(
        action,
        ground_truth_root_cause="true_cause",
        seat_role="immune",
        claims_by_id={},
        trust_scores={},
        agent_reliability={},
    )
    assert components["r3_detection"] == 0.0


def test_exploit_r4_uniform_trust_cannot_reach_perfect_score_in_env_context() -> None:
    env = SIEGEEnvironment(seed=12)
    obs = env.reset()
    _, _, _, info = env.step(_diagnose("wrong_cause"))
    r4 = info["reward_components"]["r4_trust_calibration"]
    assert r4 < 1.0
    assert obs.trust_scores and all(value == 0.5 for value in obs.trust_scores.values())
