"""Gate test for Step 09 — Tiered Curriculum Scheduler.

5 tests:
1. Fresh scheduler always starts at Tier 1.
2. Escalates to Tier 2 after sustained high win-rate.
3. De-escalates back to Tier 1 after sustained low win-rate.
4. Never escalates above Tier 3 regardless of win-rate.
5. attacker_ahead() invariant holds correctly.
"""

from __future__ import annotations

from siege_env.curriculum.tiered_scheduler import TIER_CONFIGS, TieredScheduler


def _feed(scheduler: TieredScheduler, *, wins: int, losses: int) -> None:
    """Feed a block of wins then losses (or vice-versa) to the scheduler."""
    for _ in range(wins):
        scheduler.record_episode(won=True)
    for _ in range(losses):
        scheduler.record_episode(won=False)


def test_starts_at_tier_1() -> None:
    """A fresh scheduler must start at Tier 1."""
    s = TieredScheduler()
    assert s.current_tier == 1
    assert s.config == TIER_CONFIGS[1]


def test_escalates_on_high_winrate() -> None:
    """10 consecutive wins (window=10) must push scheduler from Tier 1 → 2."""
    s = TieredScheduler(window=10, escalate_threshold=0.70, cooldown=0)
    _feed(s, wins=10, losses=0)
    assert s.current_tier == 2


def test_deescalates_on_low_winrate() -> None:
    """After reaching Tier 2, 10 consecutive losses must drop back to Tier 1."""
    s = TieredScheduler(window=10, escalate_threshold=0.70, deescalate_threshold=0.30, cooldown=0)
    # Escalate to Tier 2 first
    _feed(s, wins=10, losses=0)
    assert s.current_tier == 2
    # Now tank the win-rate
    _feed(s, wins=0, losses=10)
    assert s.current_tier == 1


def test_no_escalation_above_tier_3() -> None:
    """Scheduler must never exceed Tier 3 no matter how many wins are recorded."""
    s = TieredScheduler(window=10, escalate_threshold=0.70, cooldown=0)
    # Push through all tiers
    _feed(s, wins=50, losses=0)
    assert s.current_tier == 3


def test_attacker_ahead_invariant() -> None:
    """attacker_ahead() must be True until agent clears the escalation threshold."""
    s = TieredScheduler(window=10, escalate_threshold=0.70, cooldown=0)

    # Not enough data yet → attacker assumed ahead
    assert s.attacker_ahead() is True

    # 6 wins out of 10 = 60% < 70% threshold → still ahead
    _feed(s, wins=6, losses=4)
    assert s.attacker_ahead() is True

    # 8 wins in next window = 80% ≥ 70% → agent is beating the tier
    s2 = TieredScheduler(window=10, escalate_threshold=0.70, cooldown=0)
    _feed(s2, wins=8, losses=2)
    assert s2.attacker_ahead() is False
