"""Unit tests for tiered curriculum scheduler."""

from __future__ import annotations

import pytest

from siege_env.curriculum.tiered_scheduler import TieredScheduler


class TestTieredScheduler:
    def test_starts_at_tier_1(self) -> None:
        sched = TieredScheduler()
        assert sched.current_tier == 1

    def test_tier_escalation_on_sustained_wins(self) -> None:
        sched = TieredScheduler(window=10, escalate_threshold=0.7, cooldown=0)
        for _ in range(15):
            sched.record_episode(won=True)
        assert sched.current_tier >= 2

    def test_tier_does_not_exceed_3(self) -> None:
        sched = TieredScheduler(window=5, escalate_threshold=0.7, cooldown=0)
        for _ in range(200):
            sched.record_episode(won=True)
        assert sched.current_tier <= 3

    def test_attacker_ahead_on_fresh_scheduler(self) -> None:
        sched = TieredScheduler()
        assert sched.attacker_ahead() is True

    def test_deescalation_on_losses(self) -> None:
        sched = TieredScheduler(window=5, escalate_threshold=0.7, deescalate_threshold=0.3, cooldown=0)
        # First escalate
        for _ in range(10):
            sched.record_episode(won=True)
        tier_after_wins = sched.current_tier
        # Then lose a lot
        for _ in range(20):
            sched.record_episode(won=False)
        assert sched.current_tier <= tier_after_wins

    def test_config_returns_tier_config(self) -> None:
        sched = TieredScheduler()
        config = sched.config
        assert config.tier == 1
        assert config.num_pathogens >= 1

    def test_reset_returns_to_tier_1(self) -> None:
        sched = TieredScheduler(window=5, escalate_threshold=0.7, cooldown=0)
        for _ in range(20):
            sched.record_episode(won=True)
        sched.reset()
        assert sched.current_tier == 1

    def test_win_rate_zero_before_window(self) -> None:
        sched = TieredScheduler(window=10)
        sched.record_episode(won=True)
        assert sched.win_rate() == 0.0  # not enough data
