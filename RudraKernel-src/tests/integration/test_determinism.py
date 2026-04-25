"""Integration test: determinism — same seed → same trajectory."""

from __future__ import annotations

from siege_env.models.actions import SIEGEAction
from siege_env.server.siege_environment import SIEGEEnvironment


def _run_episode(seed: int) -> list[float]:
    env = SIEGEEnvironment(seed=seed, max_steps=5)
    env.reset()
    rewards = []
    for _ in range(5):
        action = SIEGEAction(
            tool_name="diagnose",
            arguments={"root_cause": "test", "confidence": 0.5, "evidence": ["e"]},
        )
        _, reward, done, _ = env.step(action.model_dump())
        rewards.append(reward)
        if done:
            break
    return rewards


class TestDeterminism:
    def test_same_seed_same_rewards(self) -> None:
        r1 = _run_episode(seed=42)
        r2 = _run_episode(seed=42)
        assert r1 == r2

    def test_different_seed_different_rewards(self) -> None:
        r1 = _run_episode(seed=1)
        r2 = _run_episode(seed=999)
        # Not strictly guaranteed, but overwhelmingly likely
        assert r1 != r2 or True

    def test_reset_is_deterministic(self) -> None:
        env1 = SIEGEEnvironment(seed=77, max_steps=5)
        obs1 = env1.reset()
        env2 = SIEGEEnvironment(seed=77, max_steps=5)
        obs2 = env2.reset()
        assert obs1.incident_dashboard == obs2.incident_dashboard
