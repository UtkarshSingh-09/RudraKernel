"""Unit tests for the N×N Bayesian trust network."""

from __future__ import annotations

import pytest

from siege_env.trust.network import BayesianTrustNetwork


class TestBayesianTrustNetwork:
    def test_initial_trust_is_prior(self) -> None:
        net = BayesianTrustNetwork(agent_count=4, prior=0.5)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert net.get_trust(i, j) == pytest.approx(0.5)

    def test_self_trust_is_one(self) -> None:
        net = BayesianTrustNetwork(agent_count=4, prior=0.5)
        for i in range(4):
            assert net.get_trust(i, i) == 1.0

    def test_correct_claim_increases_trust(self) -> None:
        net = BayesianTrustNetwork(agent_count=4)
        before = net.get_trust(0, 1)
        net.update(observer_id=0, target_id=1, claim_correct=True)
        after = net.get_trust(0, 1)
        assert after > before

    def test_incorrect_claim_decreases_trust(self) -> None:
        net = BayesianTrustNetwork(agent_count=4)
        before = net.get_trust(0, 1)
        net.update(observer_id=0, target_id=1, claim_correct=False)
        after = net.get_trust(0, 1)
        assert after < before

    def test_trust_stays_in_bounds(self) -> None:
        net = BayesianTrustNetwork(agent_count=4)
        for _ in range(200):
            net.update(observer_id=0, target_id=1, claim_correct=True)
        assert 0.0 <= net.get_trust(0, 1) <= 1.0

        for _ in range(400):
            net.update(observer_id=0, target_id=1, claim_correct=False)
        assert 0.0 <= net.get_trust(0, 1) <= 1.0

    def test_matrix_is_asymmetric(self) -> None:
        net = BayesianTrustNetwork(agent_count=4)
        net.update(observer_id=0, target_id=1, claim_correct=True)
        # Only agent 0's view of agent 1 changes, not reverse
        assert net.get_trust(0, 1) != net.get_trust(1, 0)

    def test_as_matrix_returns_correct_shape(self) -> None:
        net = BayesianTrustNetwork(agent_count=8)
        matrix = net.as_matrix()
        assert len(matrix) == 8
        assert all(len(row) == 8 for row in matrix)

    def test_invalid_agent_count_rejected(self) -> None:
        with pytest.raises(ValueError):
            BayesianTrustNetwork(agent_count=1)

    def test_out_of_bounds_agent_raises(self) -> None:
        net = BayesianTrustNetwork(agent_count=4)
        with pytest.raises(ValueError):
            net.get_trust(5, 0)
