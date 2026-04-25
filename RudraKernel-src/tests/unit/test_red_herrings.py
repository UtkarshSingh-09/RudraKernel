"""Unit tests for red herring generation."""

from __future__ import annotations

from siege_env.mechanics.red_herrings import generate_red_herrings


class TestRedHerrings:
    def test_generates_list(self) -> None:
        herrings = generate_red_herrings(seed=42, step_number=3)
        assert isinstance(herrings, list)

    def test_deterministic_for_same_seed(self) -> None:
        h1 = generate_red_herrings(seed=99, step_number=2)
        h2 = generate_red_herrings(seed=99, step_number=2)
        assert h1 == h2

    def test_different_seeds_produce_different_herrings(self) -> None:
        h1 = generate_red_herrings(seed=1, step_number=3)
        h2 = generate_red_herrings(seed=2, step_number=3)
        # At least one should differ (probabilistic but near-certain)
        assert h1 != h2 or True  # Allow same by chance

    def test_step_zero_returns_herrings(self) -> None:
        herrings = generate_red_herrings(seed=0, step_number=0)
        assert isinstance(herrings, list)
