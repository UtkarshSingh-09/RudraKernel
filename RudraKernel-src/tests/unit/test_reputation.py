"""Unit tests for cross-episode reputation tracking."""

from __future__ import annotations

from siege_env.trust.reputation import ReputationMemory


class TestReputationMemory:
    def test_initial_reputation_is_neutral(self) -> None:
        store = ReputationMemory()
        # First update returns from prior of 0.5
        score = store.update(agent_id=1, reliable=True)
        assert 0.5 <= score <= 1.0  # should move toward 1.0

    def test_reliable_increases_reputation(self) -> None:
        store = ReputationMemory()
        s1 = store.update(agent_id=1, reliable=True)
        s2 = store.update(agent_id=1, reliable=True)
        assert s2 >= s1

    def test_unreliable_decreases_reputation(self) -> None:
        store = ReputationMemory()
        store.update(agent_id=1, reliable=True)
        store.update(agent_id=1, reliable=True)
        high = store.snapshot().get(1, 0.5)
        store.update(agent_id=1, reliable=False)
        low = store.snapshot().get(1, 0.5)
        assert low < high

    def test_snapshot_returns_all_agents(self) -> None:
        store = ReputationMemory()
        store.update(agent_id=1, reliable=True)
        store.update(agent_id=2, reliable=False)
        snap = store.snapshot()
        assert 1 in snap
        assert 2 in snap

    def test_reset_clears_scores(self) -> None:
        store = ReputationMemory()
        store.update(agent_id=1, reliable=True)
        store.reset()
        assert store.snapshot() == {}

    def test_invalid_alpha_rejected(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            ReputationMemory(alpha=0.0)
        with pytest.raises(ValueError):
            ReputationMemory(alpha=1.5)
