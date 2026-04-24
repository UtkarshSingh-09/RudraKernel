"""Gate tests for Step 20 — Frozen Opponent League."""

from __future__ import annotations

from siege_env.league.opponent_pool import FrozenOpponentPool
from siege_env.server.siege_environment import SIEGEEnvironment


def test_frozen_pool_sampling_count() -> None:
    pool = FrozenOpponentPool(seed=1)
    roster = pool.sample(k=3)
    assert len(roster) == 3


def test_frozen_pool_deterministic_for_seed() -> None:
    a = [o.opponent_id for o in FrozenOpponentPool(seed=42).sample(k=3)]
    b = [o.opponent_id for o in FrozenOpponentPool(seed=42).sample(k=3)]
    assert a == b


def test_environment_reset_contains_league_roster() -> None:
    env = SIEGEEnvironment(seed=2)
    obs = env.reset()
    assert "league_roster" in obs.incident_dashboard
    assert len(obs.incident_dashboard["league_roster"]) == 3


def test_league_roster_persists_into_step_observation() -> None:
    env = SIEGEEnvironment(seed=8)
    env.reset()
    obs, _, _, _ = env.step({"tool_name": "diagnose", "arguments": {"root_cause": "x", "confidence": 0.5, "evidence": ["e"]}})
    assert "league_roster" in obs.incident_dashboard
