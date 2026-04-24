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
