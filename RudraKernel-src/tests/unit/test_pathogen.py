"""Unit tests for pathogen strategy library."""

from __future__ import annotations

from siege_env.agents.pathogen_strategies import PathogenStrategy


class TestPathogenStrategy:
    def test_build_trust_phase(self) -> None:
        strat = PathogenStrategy(agent_id=4, seed=42, strike_step=5)
        assert strat.stage(0) == "build_trust"
        assert strat.stage(4) == "build_trust"

    def test_strike_phase(self) -> None:
        strat = PathogenStrategy(agent_id=4, seed=42, strike_step=5)
        assert strat.stage(5) == "strike"
        assert strat.stage(10) == "strike"

    def test_build_phase_echoes_true_root_cause(self) -> None:
        strat = PathogenStrategy(agent_id=4, seed=42, strike_step=5)
        template = {
            "root_cause": "db_deletion",
            "observable_signals": ["sig1", "sig2"],
            "blast_radius": ["br1", "br2"],
        }
        claim = strat.generate_claim(template, step_number=0)
        assert claim["root_cause"] == "db_deletion"

    def test_strike_phase_injects_false_cause(self) -> None:
        strat = PathogenStrategy(agent_id=4, seed=42, strike_step=3)
        template = {
            "root_cause": "db_deletion",
            "observable_signals": ["sig1", "sig2"],
            "blast_radius": ["br1", "br2"],
        }
        claim = strat.generate_claim(template, step_number=5)
        assert claim["root_cause"] != "db_deletion"
        assert "false_positive" in claim["root_cause"]

    def test_claim_has_required_fields(self) -> None:
        strat = PathogenStrategy(agent_id=4, seed=42)
        template = {
            "root_cause": "test",
            "observable_signals": ["s1"],
            "blast_radius": ["b1"],
        }
        claim = strat.generate_claim(template, step_number=0)
        assert "agent_id" in claim
        assert "claim_id" in claim
        assert "root_cause" in claim
        assert "confidence" in claim
        assert 0.0 <= claim["confidence"] <= 1.0

    def test_trust_trajectory_builds_then_drops(self) -> None:
        strat = PathogenStrategy(agent_id=4, seed=42, strike_step=3)
        trajectory = strat.trust_trajectory(steps=8)
        # Trust should rise during build phase then drop during strike
        assert trajectory[2] > trajectory[0]  # build phase
        assert trajectory[5] < trajectory[2]  # strike phase

    def test_deterministic_claims(self) -> None:
        strat1 = PathogenStrategy(agent_id=4, seed=42)
        strat2 = PathogenStrategy(agent_id=4, seed=42)
        template = {"root_cause": "test", "observable_signals": ["s1"], "blast_radius": ["b1"]}
        c1 = strat1.generate_claim(template, step_number=3)
        c2 = strat2.generate_claim(template, step_number=3)
        assert c1 == c2
