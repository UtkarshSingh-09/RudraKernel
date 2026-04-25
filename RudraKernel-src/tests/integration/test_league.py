"""Integration test: frozen opponent league."""

from __future__ import annotations

from siege_env.league.opponent_pool import FrozenOpponentPool
from siege_env.server.siege_environment import SIEGEEnvironment


class TestLeague:
    def test_league_roster_in_reset_observation(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        obs = env.reset()
        assert "league_roster" in obs.incident_dashboard

    def test_league_roster_has_opponents(self) -> None:
        env = SIEGEEnvironment(seed=42, max_steps=5)
        obs = env.reset()
        roster = obs.incident_dashboard["league_roster"]
        assert isinstance(roster, list)
        assert len(roster) > 0

    def test_frozen_pool_deterministic(self) -> None:
        pool1 = FrozenOpponentPool(seed=42)
        pool2 = FrozenOpponentPool(seed=42)
        r1 = [o.opponent_id for o in pool1.sample(k=3)]
        r2 = [o.opponent_id for o in pool2.sample(k=3)]
        assert r1 == r2

    def test_opponent_has_required_fields(self) -> None:
        pool = FrozenOpponentPool(seed=0)
        opponents = pool.sample(k=3)
        for opp in opponents:
            assert hasattr(opp, "opponent_id")
            assert hasattr(opp, "policy_tag")
            assert hasattr(opp, "tier")
