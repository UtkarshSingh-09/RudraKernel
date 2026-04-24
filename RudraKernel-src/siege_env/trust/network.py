"""Bayesian trust network for multi-agent credibility tracking."""

from __future__ import annotations


class BayesianTrustNetwork:
    """N x N trust matrix with simple Bayesian updates."""

    def __init__(
        self,
        *,
        agent_count: int,
        prior: float = 0.5,
        p_correct_if_trusted: float = 0.8,
        p_correct_if_untrusted: float = 0.3,
    ) -> None:
        if agent_count < 2:
            raise ValueError("agent_count must be >= 2.")
        if not 0.0 < prior < 1.0:
            raise ValueError("prior must be between 0 and 1 (exclusive).")
        if not 0.0 < p_correct_if_untrusted < p_correct_if_trusted < 1.0:
            raise ValueError("likelihood parameters must satisfy 0 < untrusted < trusted < 1.")

        self._agent_count = agent_count
        self._p_correct_if_trusted = p_correct_if_trusted
        self._p_correct_if_untrusted = p_correct_if_untrusted
        self._matrix: list[list[float]] = []
        for observer in range(agent_count):
            row: list[float] = []
            for target in range(agent_count):
                row.append(1.0 if observer == target else prior)
            self._matrix.append(row)

    @property
    def agent_count(self) -> int:
        return self._agent_count

    def _validate_agent(self, agent_id: int) -> None:
        if agent_id < 0 or agent_id >= self._agent_count:
            raise ValueError(f"agent_id {agent_id} out of bounds for {self._agent_count} agents.")

    def get_trust(self, observer_id: int, target_id: int) -> float:
        self._validate_agent(observer_id)
        self._validate_agent(target_id)
        return self._matrix[observer_id][target_id]

    def update(self, *, observer_id: int, target_id: int, claim_correct: bool) -> float:
        self._validate_agent(observer_id)
        self._validate_agent(target_id)

        if observer_id == target_id:
            self._matrix[observer_id][target_id] = 1.0
            return 1.0

        prior = self._matrix[observer_id][target_id]
        if claim_correct:
            likelihood_trusted = self._p_correct_if_trusted
            likelihood_untrusted = self._p_correct_if_untrusted
        else:
            likelihood_trusted = 1.0 - self._p_correct_if_trusted
            likelihood_untrusted = 1.0 - self._p_correct_if_untrusted

        numerator = likelihood_trusted * prior
        denominator = numerator + (likelihood_untrusted * (1.0 - prior))
        posterior = numerator / denominator
        posterior = max(0.0, min(1.0, posterior))
        self._matrix[observer_id][target_id] = posterior
        return posterior

    def as_matrix(self) -> list[list[float]]:
        return [list(row) for row in self._matrix]
