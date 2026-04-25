"""Integration test: invalid action handling."""

from __future__ import annotations

from siege_env.server.siege_environment import SIEGEEnvironment


class TestInvalidActions:
    def test_empty_payload_returns_penalty(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        env.reset()
        _, reward, _, info = env.step({})
        assert reward < 0
        assert info.get("invalid_action") is True

    def test_missing_tool_name_returns_penalty(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        env.reset()
        _, reward, _, info = env.step({"arguments": {"root_cause": "x"}})
        assert reward < 0

    def test_missing_arguments_returns_penalty(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        env.reset()
        _, reward, _, info = env.step({"tool_name": "diagnose"})
        assert reward < 0

    def test_step_before_reset_raises(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        try:
            env.step({"tool_name": "diagnose", "arguments": {"root_cause": "x", "confidence": 0.5, "evidence": ["e"]}})
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_state_before_reset_raises(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        try:
            env.state()
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_step_after_done_returns_zero_reward(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=1)
        env.reset()
        action = {"tool_name": "diagnose", "arguments": {"root_cause": "x", "confidence": 0.5, "evidence": ["e"]}}
        env.step(action)
        _, reward, done, info = env.step(action)
        assert done is True
        assert reward == 0.0
        assert info.get("already_done") is True
