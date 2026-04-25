"""Performance test: environment step latency benchmark."""

from __future__ import annotations

import time

from siege_env.models.actions import SIEGEAction
from siege_env.server.siege_environment import SIEGEEnvironment


class TestLatency:
    def test_reset_under_100ms(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        start = time.perf_counter()
        env.reset()
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"reset took {elapsed_ms:.1f}ms (limit: 100ms)"

    def test_step_under_50ms(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        env.reset()
        action = SIEGEAction(
            tool_name="diagnose",
            arguments={"root_cause": "test", "confidence": 0.5, "evidence": ["e"]},
        )
        start = time.perf_counter()
        env.step(action.model_dump())
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50, f"step took {elapsed_ms:.1f}ms (limit: 50ms)"

    def test_full_episode_under_500ms(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        start = time.perf_counter()
        env.reset()
        for _ in range(5):
            action = SIEGEAction(
                tool_name="diagnose",
                arguments={"root_cause": "test", "confidence": 0.5, "evidence": ["e"]},
            )
            _, _, done, _ = env.step(action.model_dump())
            if done:
                break
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"full episode took {elapsed_ms:.1f}ms (limit: 500ms)"
