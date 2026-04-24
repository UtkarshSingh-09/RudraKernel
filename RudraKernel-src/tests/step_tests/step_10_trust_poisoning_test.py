"""Gate test for Step 10 — Trust Poisoning Strategy (scripted pathogen).

6 tests covering:
1. Stage returns 'build_trust' before strike_step.
2. Stage returns 'strike' at and after strike_step.
3. Build-trust claims always echo the true root cause.
4. Strike claims always produce a false root cause.
5. Trust trajectory rises during Phase 1.
6. Trust trajectory drops sharply after Phase 2 begins.
"""

from __future__ import annotations

from siege_env.agents.pathogen_strategies import PathogenStrategy


_TEMPLATE: dict = {
    "id": "test_incident",
    "source_url": "https://example.com/incident",
    "root_cause": "database_timeout",
    "observable_signals": ["high_latency", "connection_errors", "retry_storms"],
    "flaw_types": ["type1_false_correlation"],
    "blast_radius": ["api_layer", "cache_layer", "db_layer"],
}


def test_stage_build_trust_before_strike() -> None:
    """Steps before strike_step must return 'build_trust'."""
    p = PathogenStrategy(agent_id=0, seed=42, strike_step=5)
    for step in range(5):
        assert p.stage(step) == "build_trust", f"Expected build_trust at step {step}"


def test_stage_strike_at_and_after_strike_step() -> None:
    """Steps >= strike_step must return 'strike'."""
    p = PathogenStrategy(agent_id=0, seed=42, strike_step=5)
    for step in range(5, 12):
        assert p.stage(step) == "strike", f"Expected strike at step {step}"


def test_build_trust_claims_echo_true_root_cause() -> None:
    """During Phase 1, every claim must echo the true root cause."""
    p = PathogenStrategy(agent_id=1, seed=7, strike_step=5)
    true_root_cause = _TEMPLATE["root_cause"]
    for step in range(5):
        claim = p.generate_claim(_TEMPLATE, step_number=step)
        assert claim["root_cause"] == true_root_cause, (
            f"Phase 1 step {step} should echo true root cause"
        )
        assert claim["stage"] == "build_trust"


def test_strike_claims_produce_false_root_cause() -> None:
    """During Phase 2, claims must NOT match the true root cause."""
    p = PathogenStrategy(agent_id=1, seed=7, strike_step=5)
    true_root_cause = _TEMPLATE["root_cause"]
    for step in range(5, 10):
        claim = p.generate_claim(_TEMPLATE, step_number=step)
        assert claim["root_cause"] != true_root_cause, (
            f"Phase 2 step {step} should inject false root cause"
        )
        assert claim["stage"] == "strike"


def test_trust_trajectory_rises_during_phase_1() -> None:
    """Trust should increase monotonically during build-trust phase."""
    p = PathogenStrategy(agent_id=2, seed=99, strike_step=8)
    traj = p.trust_trajectory(steps=8)  # all Phase 1 steps
    assert traj[0] == 0.5, "Trajectory should start at initial_trust=0.5"
    for i in range(1, len(traj)):
        assert traj[i] >= traj[i - 1], (
            f"Trust should not decrease in Phase 1: step {i-1}={traj[i-1]}, step {i}={traj[i]}"
        )
    assert traj[-1] > traj[0], "Trust must be higher after Phase 1 than at start"


def test_trust_drops_after_strike_begins() -> None:
    """Trust must fall below its Phase-1 peak once Phase 2 strikes land."""
    p = PathogenStrategy(agent_id=3, seed=55, strike_step=5)
    # Run enough steps to see the drop: 5 build + 5 strike
    traj = p.trust_trajectory(steps=10)
    phase1_peak = max(traj[:5])
    phase2_final = traj[9]
    assert phase2_final < phase1_peak, (
        f"Trust after strikes ({phase2_final:.4f}) must be lower than "
        f"Phase 1 peak ({phase1_peak:.4f})"
    )
