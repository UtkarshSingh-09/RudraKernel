"""Integration test: full episode lifecycle from reset to done."""

from __future__ import annotations

from siege_env.models.actions import SIEGEAction
from siege_env.server.siege_environment import SIEGEEnvironment


class TestFullEpisode:
    def test_episode_runs_to_completion(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        obs = env.reset()
        assert obs.step_number == 0

        done = False
        step = 0
        while not done and step < 10:
            action = SIEGEAction(
                tool_name="diagnose",
                arguments={"root_cause": "test_cause", "confidence": 0.5, "evidence": ["e1"]},
            )
            obs, reward, done, info = env.step(action.model_dump())
            step += 1

        assert done is True
        assert step <= 6  # max_steps=5, done on step 5 at latest

    def test_reward_is_bounded(self) -> None:
        env = SIEGEEnvironment(seed=0, max_steps=5)
        env.reset()
        action = SIEGEAction(
            tool_name="diagnose",
            arguments={"root_cause": "whatever", "confidence": 0.5, "evidence": ["e"]},
        )
        _, reward, _, _ = env.step(action.model_dump())
        assert -1.0 <= reward <= 1.0

    def test_state_after_reset_is_valid(self) -> None:
        env = SIEGEEnvironment(seed=7, max_steps=5)
        env.reset()
        state = env.state()
        assert state.step_count == 0
        assert state.current_tier in {1, 2, 3}
        assert state.episode_id.startswith("episode-")

    def test_multiple_episodes_have_different_ids(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=3)
        env.reset()
        id1 = env.state().episode_id
        env.reset()
        id2 = env.state().episode_id
        assert id1 != id2

    def test_observation_contains_agent_claims(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        obs = env.reset()
        assert isinstance(obs.agent_claims, list)
        assert len(obs.agent_claims) > 0

    def test_info_contains_reward_components(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        env.reset()
        action = SIEGEAction(
            tool_name="diagnose",
            arguments={"root_cause": "x", "confidence": 0.5, "evidence": ["e"]},
        )
        _, _, _, info = env.step(action.model_dump())
        assert "reward_components" in info
