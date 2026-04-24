from __future__ import annotations

from siege_env.models import SIEGEObservation
from siege_env.server.siege_environment import SIEGEEnvironment


def _diagnose(root_cause: str) -> dict[str, object]:
    return {
        "tool_name": "diagnose",
        "arguments": {
            "root_cause": root_cause,
            "confidence": 0.71,
            "evidence": ["latency_p99_spike"],
            "alternative_hypotheses": [],
        },
    }


def _challenge(claim_id: str, target_agent_id: int) -> dict[str, object]:
    return {
        "tool_name": "challenge",
        "arguments": {
            "target_agent_id": target_agent_id,
            "claim_id": claim_id,
            "flaw_type": "type1_false_correlation",
            "reasoning": "Claim conflicts with observed evidence progression across traces.",
        },
    }


def _env_with_role(role: str) -> tuple[SIEGEEnvironment, SIEGEObservation]:
    # Role is sampled each reset, so retry deterministically until desired role appears.
    env = SIEGEEnvironment(seed=11 if role == "pathogen" else 7)
    for _ in range(50):
        obs = env.reset()
        if obs.your_role == role:
            return env, obs
    raise AssertionError(f"Could not sample role '{role}' in 50 resets.")


def test_role_field_is_emitted_in_observation() -> None:
    env = SIEGEEnvironment(seed=3)
    obs = env.reset()
    assert obs.your_role in {"immune", "pathogen"}


def test_role_assignment_approximately_matches_seventy_thirty_split() -> None:
    env = SIEGEEnvironment(seed=123)
    pathogen_count = 0
    total_episodes = 200
    for _ in range(total_episodes):
        if env.reset().your_role == "pathogen":
            pathogen_count += 1
    ratio = pathogen_count / total_episodes
    assert 0.2 <= ratio <= 0.4


def test_r2_positive_for_pathogen_wrong_diagnosis() -> None:
    env, _ = _env_with_role("pathogen")
    obs, reward, _, info = env.step(_diagnose("fabricated_root_cause"))
    assert obs.action_error is None
    assert info["reward_components"]["r2_deception"] > 0.0
    assert reward >= info["reward_components"]["r2_deception"]


def test_r2_zero_for_pathogen_correct_diagnosis() -> None:
    env, _ = _env_with_role("pathogen")
    _, _, _, info = env.step(_diagnose(env.state().ground_truth_root_cause))
    assert info["reward_components"]["r2_deception"] == 0.0


def test_r2_zero_for_immune_wrong_diagnosis() -> None:
    env, _ = _env_with_role("immune")
    _, _, _, info = env.step(_diagnose("fabricated_root_cause"))
    assert info["reward_components"]["r2_deception"] == 0.0


def test_r3_positive_for_immune_challenging_incorrect_claim() -> None:
    env, obs = _env_with_role("immune")
    incorrect_claim = next(
        claim for claim in obs.agent_claims if claim["root_cause"] != env.state().ground_truth_root_cause
    )
    _, reward, _, info = env.step(_challenge(incorrect_claim["claim_id"], incorrect_claim["agent_id"]))
    assert info["reward_components"]["r3_detection"] > 0.0
    assert reward >= info["reward_components"]["r3_detection"]


def test_r3_zero_for_immune_challenging_correct_claim() -> None:
    env, obs = _env_with_role("immune")
    correct_claim = next(
        claim for claim in obs.agent_claims if claim["root_cause"] == env.state().ground_truth_root_cause
    )
    _, _, _, info = env.step(_challenge(correct_claim["claim_id"], correct_claim["agent_id"]))
    assert info["reward_components"]["r3_detection"] == 0.0


def test_r3_zero_for_pathogen_challenge() -> None:
    env, obs = _env_with_role("pathogen")
    claim = obs.agent_claims[0]
    _, _, _, info = env.step(_challenge(claim["claim_id"], claim["agent_id"]))
    assert info["reward_components"]["r3_detection"] == 0.0


def test_reward_components_include_r2_and_r3() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    _, _, _, info = env.step(_diagnose("fabricated_root_cause"))
    assert "r2_deception" in info["reward_components"]
    assert "r3_detection" in info["reward_components"]
