"""Integration test: role assignment (70% immune / 30% pathogen)."""

from __future__ import annotations

from siege_env.server.siege_environment import SIEGEEnvironment


class TestRoleAssignment:
    def test_role_is_immune_or_pathogen(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        obs = env.reset()
        assert obs.your_role in ("immune", "pathogen")

    def test_role_distribution_approximate(self) -> None:
        roles = []
        for seed in range(100):
            env = SIEGEEnvironment(seed=seed, max_steps=3)
            obs = env.reset()
            roles.append(obs.your_role)

        immune_count = roles.count("immune")
        pathogen_count = roles.count("pathogen")
        # Should be roughly 70/30 — allow wide margin for randomness
        assert 50 <= immune_count <= 90, f"immune={immune_count}, pathogen={pathogen_count}"
        assert 10 <= pathogen_count <= 50

    def test_role_stable_within_episode(self) -> None:
        from siege_env.models.actions import SIEGEAction

        env = SIEGEEnvironment(seed=42, max_steps=5)
        obs = env.reset()
        initial_role = obs.your_role

        action = SIEGEAction(
            tool_name="diagnose",
            arguments={"root_cause": "test", "confidence": 0.5, "evidence": ["e"]},
        )
        obs2, _, _, _ = env.step(action.model_dump())
        assert obs2.your_role == initial_role
