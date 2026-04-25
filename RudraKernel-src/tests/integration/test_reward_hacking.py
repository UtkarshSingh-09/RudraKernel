"""Integration test: reward hacking exploit tests (9 exploits)."""

from __future__ import annotations

import pytest

from siege_env.models.actions import SIEGEAction
from siege_env.server.siege_environment import SIEGEEnvironment


def _run_exploit_episode(action_sequence: list[dict], seed: int = 42) -> list[float]:
    env = SIEGEEnvironment(seed=seed, max_steps=len(action_sequence) + 1)
    env.reset()
    rewards = []
    for action_data in action_sequence:
        _, reward, done, _ = env.step(action_data)
        rewards.append(reward)
        if done:
            break
    return rewards


class TestRewardHacking:
    def test_always_diagnose_same_cause_no_perfect_score(self) -> None:
        """Spamming the same diagnosis should not yield perfect reward."""
        actions = [
            {"tool_name": "diagnose", "arguments": {"root_cause": "always_same", "confidence": 0.5, "evidence": ["e"]}}
            for _ in range(5)
        ]
        rewards = _run_exploit_episode(actions)
        assert max(rewards) < 1.0 or sum(rewards) < 5.0

    def test_always_max_confidence_penalized(self) -> None:
        """Always claiming confidence=1.0 should not dominate."""
        actions = [
            {"tool_name": "diagnose", "arguments": {"root_cause": "test", "confidence": 1.0, "evidence": ["e"]}}
            for _ in range(5)
        ]
        rewards = _run_exploit_episode(actions)
        assert sum(rewards) < 5.0

    def test_always_min_confidence_penalized(self) -> None:
        """Always claiming confidence=0.0 should not be rewarded."""
        actions = [
            {"tool_name": "diagnose", "arguments": {"root_cause": "test", "confidence": 0.0, "evidence": ["e"]}}
            for _ in range(5)
        ]
        rewards = _run_exploit_episode(actions)
        assert sum(rewards) < 5.0

    def test_never_challenge_is_not_optimal(self) -> None:
        """A policy that never challenges should not get top scores."""
        actions = [
            {"tool_name": "diagnose", "arguments": {"root_cause": "x", "confidence": 0.5, "evidence": ["e"]}}
            for _ in range(5)
        ]
        rewards = _run_exploit_episode(actions)
        total = sum(rewards)
        # Not definitive — but shouldn't be perfect
        assert total < 4.5

    def test_spam_escalate_not_rewarded(self) -> None:
        """Spamming escalate should not generate positive reward."""
        actions = [
            {"tool_name": "escalate", "arguments": {"concern": "everything is broken!", "blast_radius_estimate": ["all"]}}
            for _ in range(5)
        ]
        rewards = _run_exploit_episode(actions)
        assert sum(rewards) <= 2.0

    def test_spam_whisper_not_rewarded(self) -> None:
        """Spamming whisper should not generate positive reward."""
        actions = [
            {"tool_name": "whisper", "arguments": {"target_agent_id": 1, "message": "spam message"}}
            for _ in range(5)
        ]
        rewards = _run_exploit_episode(actions)
        assert sum(rewards) <= 2.0

    def test_invalid_actions_get_penalty(self) -> None:
        """Sending invalid actions should result in negative reward."""
        env = SIEGEEnvironment(seed=42, max_steps=5)
        env.reset()
        _, reward, _, info = env.step({"tool_name": "diagnose", "arguments": {}})
        assert reward < 0
        assert info.get("invalid_action") is True

    def test_ratify_everything_not_optimal(self) -> None:
        """Blindly ratifying everything shouldn't be optimal."""
        actions = [
            {"tool_name": "ratify", "arguments": {"claim_id": f"claim-{i}", "vote": True}}
            for i in range(5)
        ]
        rewards = _run_exploit_episode(actions)
        assert sum(rewards) < 4.0

    def test_constant_half_confidence_not_optimal(self) -> None:
        """Always 0.5 confidence = not calibrated = not optimal."""
        actions = [
            {"tool_name": "diagnose", "arguments": {"root_cause": "rc", "confidence": 0.5, "evidence": ["e"]}}
            for _ in range(5)
        ]
        rewards = _run_exploit_episode(actions)
        assert sum(rewards) < 4.5
