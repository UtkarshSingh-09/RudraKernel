"""Gate tests for Step 17 — Red Herrings + R9."""

from __future__ import annotations

from siege_env.mechanics.red_herrings import generate_red_herrings
from siege_env.models.actions import ChallengeArgs, DiagnoseArgs, SIEGEAction
from siege_env.rewards.r9_correlation import compute_r9_correlation
from siege_env.server.siege_environment import SIEGEEnvironment


def test_red_herrings_are_deterministic_for_seed_and_step() -> None:
    a = generate_red_herrings(seed=42, step_number=2)
    b = generate_red_herrings(seed=42, step_number=2)
    assert a == b


def test_r9_rewards_correct_false_correlation_challenge() -> None:
    action = SIEGEAction(
        tool_name="challenge",
        arguments=ChallengeArgs(
            target_agent_id=1,
            claim_id="c1",
            flaw_type="type1_false_correlation",
            reasoning="Signal is correlated but not causal.",
        ),
    )
    claims = {"c1": {"root_cause": "wrong_cause"}}
    assert (
        compute_r9_correlation(action, claims_by_id=claims, ground_truth_root_cause="real_cause")
        == 1.0
    )


def test_r9_zero_for_non_challenge_actions() -> None:
    action = SIEGEAction(
        tool_name="diagnose",
        arguments=DiagnoseArgs(root_cause="x", confidence=0.7, evidence=["e"]),
    )
    assert compute_r9_correlation(action, claims_by_id={}, ground_truth_root_cause="x") == 0.0


def test_environment_observation_contains_red_herrings() -> None:
    env = SIEGEEnvironment(seed=21)
    obs = env.reset()
    assert len(obs.red_herring_signals) >= 1


def test_exploit_always_challenge_wrong_flaw_type_gets_no_r9() -> None:
    action = SIEGEAction(
        tool_name="challenge",
        arguments=ChallengeArgs(
            target_agent_id=1,
            claim_id="c2",
            flaw_type="type3_tunnel_vision",
            reasoning="Always challenging for reward exploit attempt.",
        ),
    )
    claims = {"c2": {"root_cause": "wrong_cause"}}
    assert (
        compute_r9_correlation(action, claims_by_id=claims, ground_truth_root_cause="real_cause")
        == 0.0
    )
