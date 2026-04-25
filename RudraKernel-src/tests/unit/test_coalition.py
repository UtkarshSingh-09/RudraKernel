"""Unit tests for coalition weighted voting."""

from __future__ import annotations

import pytest

from siege_env.trust.coalition import CoalitionVoting


class TestCoalitionVoting:
    def test_unanimous_support_ratifies(self) -> None:
        cv = CoalitionVoting(ratification_threshold=0.6)
        result = cv.tally(
            votes={0: True, 1: True, 2: True, 3: True},
            trust_weights={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        )
        assert result.ratified is True

    def test_no_votes_not_ratified(self) -> None:
        cv = CoalitionVoting(ratification_threshold=0.6)
        result = cv.tally(votes={}, trust_weights={})
        assert result.ratified is False

    def test_majority_oppose_not_ratified(self) -> None:
        cv = CoalitionVoting(ratification_threshold=0.6)
        result = cv.tally(
            votes={0: True, 1: False, 2: False, 3: False},
            trust_weights={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        )
        assert result.ratified is False

    def test_weighted_votes_respect_trust(self) -> None:
        cv = CoalitionVoting(ratification_threshold=0.5)
        # Agent 0 has very high trust but votes yes
        # Agents 1-3 have low trust and vote no
        result = cv.tally(
            votes={0: True, 1: False, 2: False, 3: False},
            trust_weights={0: 0.9, 1: 0.1, 2: 0.1, 3: 0.1},
        )
        assert result.ratified is True  # 0.9 > 0.3

    def test_abstain_weight_tracked(self) -> None:
        cv = CoalitionVoting()
        result = cv.tally(
            votes={0: True, 1: None},
            trust_weights={0: 1.0, 1: 0.5},
        )
        assert result.abstain_weight > 0

    def test_support_ratio_correct(self) -> None:
        cv = CoalitionVoting()
        result = cv.tally(
            votes={0: True, 1: False},
            trust_weights={0: 0.8, 1: 0.2},
        )
        expected_ratio = 0.8 / (0.8 + 0.2)
        assert abs(result.support_ratio - expected_ratio) < 0.01

    def test_invalid_threshold_rejected(self) -> None:
        with pytest.raises(ValueError):
            CoalitionVoting(ratification_threshold=0.0)
        with pytest.raises(ValueError):
            CoalitionVoting(ratification_threshold=1.5)
