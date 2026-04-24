"""Gate tests for Step 13 — Epistemic Cascade Failures."""

from __future__ import annotations

from siege_env.mechanics.cascade import EpistemicCascadeEngine
from siege_env.models.actions import DiagnoseArgs, SIEGEAction
from siege_env.server.siege_environment import SIEGEEnvironment


def test_cascade_engine_triggered_for_high_herding() -> None:
    engine = EpistemicCascadeEngine(trigger_threshold=0.8, min_agents=4)
    snapshot = engine.evaluate([0.91, 0.88, 0.85, 0.82, 0.3])
    assert snapshot.triggered is True


def test_cascade_engine_not_triggered_for_sparse_confidence() -> None:
    engine = EpistemicCascadeEngine(trigger_threshold=0.8, min_agents=4)
    snapshot = engine.evaluate([0.9, 0.1, 0.4, 0.2, 0.5])
    assert snapshot.triggered is False


def test_environment_info_contains_cascade_block() -> None:
    env = SIEGEEnvironment(seed=11)
    env.reset()
    state = env.state()
    action = SIEGEAction(
        tool_name="diagnose",
        arguments=DiagnoseArgs(
            root_cause=state.ground_truth_root_cause,
            confidence=0.8,
            evidence=["signal"],
        ),
    )
    _, _, _, info = env.step(action)
    assert "cascade" in info
    assert {"mean_confidence", "herd_strength", "triggered"}.issubset(info["cascade"].keys())


def test_environment_observation_contains_cascade_metadata() -> None:
    env = SIEGEEnvironment(seed=17)
    obs = env.reset()
    assert "cascade" in obs.incident_dashboard
