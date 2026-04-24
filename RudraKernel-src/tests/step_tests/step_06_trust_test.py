from __future__ import annotations

import pytest

from siege_env.trust.coalition import CoalitionVoting
from siege_env.trust.network import BayesianTrustNetwork


def test_trust_matrix_shape_is_n_by_n() -> None:
    network = BayesianTrustNetwork(agent_count=8)
    matrix = network.as_matrix()
    assert len(matrix) == 8
    assert all(len(row) == 8 for row in matrix)


def test_diagonal_is_one_by_default() -> None:
    network = BayesianTrustNetwork(agent_count=8)
    assert all(network.get_trust(i, i) == 1.0 for i in range(8))


def test_initial_off_diagonal_uses_prior() -> None:
    network = BayesianTrustNetwork(agent_count=8, prior=0.55)
    assert network.get_trust(0, 1) == 0.55
    assert network.get_trust(6, 3) == 0.55


def test_positive_evidence_increases_trust() -> None:
    network = BayesianTrustNetwork(agent_count=8, prior=0.5)
    before = network.get_trust(0, 1)
    network.update(observer_id=0, target_id=1, claim_correct=True)
    after = network.get_trust(0, 1)
    assert after > before


def test_negative_evidence_decreases_trust() -> None:
    network = BayesianTrustNetwork(agent_count=8, prior=0.5)
    before = network.get_trust(0, 1)
    network.update(observer_id=0, target_id=1, claim_correct=False)
    after = network.get_trust(0, 1)
    assert after < before


def test_repeated_updates_remain_bounded() -> None:
    network = BayesianTrustNetwork(agent_count=8, prior=0.5)
    for _ in range(100):
        network.update(observer_id=0, target_id=1, claim_correct=True)
        network.update(observer_id=0, target_id=1, claim_correct=False)
    value = network.get_trust(0, 1)
    assert 0.0 <= value <= 1.0


def test_invalid_agent_index_raises_error() -> None:
    network = BayesianTrustNetwork(agent_count=8)
    with pytest.raises(ValueError):
        network.get_trust(-1, 2)
    with pytest.raises(ValueError):
        network.update(observer_id=0, target_id=8, claim_correct=True)


def test_self_update_keeps_identity_trust_fixed() -> None:
    network = BayesianTrustNetwork(agent_count=8)
    network.update(observer_id=2, target_id=2, claim_correct=False)
    assert network.get_trust(2, 2) == 1.0


def test_weighted_ratification_passes_with_threshold() -> None:
    voting = CoalitionVoting(ratification_threshold=0.6)
    trust_weights = {1: 0.9, 2: 0.7, 3: 0.3}
    votes = {1: True, 2: True, 3: False}
    result = voting.tally(votes=votes, trust_weights=trust_weights)
    assert result.ratified is True


def test_weighted_ratification_fails_below_threshold() -> None:
    voting = CoalitionVoting(ratification_threshold=0.75)
    trust_weights = {1: 0.9, 2: 0.4, 3: 0.7}
    votes = {1: True, 2: False, 3: False}
    result = voting.tally(votes=votes, trust_weights=trust_weights)
    assert result.ratified is False


def test_tie_is_not_ratified() -> None:
    voting = CoalitionVoting(ratification_threshold=0.5)
    trust_weights = {1: 0.6, 2: 0.6}
    votes = {1: True, 2: False}
    result = voting.tally(votes=votes, trust_weights=trust_weights)
    assert result.support_weight == result.oppose_weight
    assert result.ratified is False


def test_abstentions_are_ignored() -> None:
    voting = CoalitionVoting(ratification_threshold=0.5)
    trust_weights = {1: 0.8, 2: 0.2, 3: 0.9}
    votes = {1: True, 2: None, 3: False}
    result = voting.tally(votes=votes, trust_weights=trust_weights)
    assert result.total_weight == 1.7
    assert result.abstain_weight == 0.2


def test_missing_weight_defaults_to_neutral() -> None:
    voting = CoalitionVoting(ratification_threshold=0.5)
    votes = {1: True, 2: False}
    result = voting.tally(votes=votes, trust_weights={1: 0.9})
    assert result.total_weight == 1.4
    assert result.ratified is True


def test_invalid_threshold_raises_error() -> None:
    with pytest.raises(ValueError):
        CoalitionVoting(ratification_threshold=0.0)
    with pytest.raises(ValueError):
        CoalitionVoting(ratification_threshold=1.2)
