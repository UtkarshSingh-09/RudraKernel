"""Gate test for Step 11 — Temporal Evidence Dynamics + R6.

5 tests covering:
1. EvidenceRecord is created with correct fields.
2. Freshness decays correctly over steps.
3. Urgency is floored at min_urgency for stale evidence.
4. R6 = 0 for wrong diagnose, R6 = urgency for correct diagnose.
5. R6 with no-evidence fallback (urgency=1.0) equals full R6.
"""

from __future__ import annotations

import math

from siege_env.mechanics.temporal_evidence import TemporalEvidenceTracker
from siege_env.models.actions import SIEGEAction, DiagnoseArgs
from siege_env.rewards.r6_temporal import compute_r6_temporal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diagnose_action(root_cause: str) -> SIEGEAction:
    return SIEGEAction(
        tool_name="diagnose",
        arguments=DiagnoseArgs(
            root_cause=root_cause,
            confidence=0.9,
            evidence=["observed high latency spike"],
        ),
    )


def _non_diagnose_action() -> SIEGEAction:
    from siege_env.models.actions import ChallengeArgs
    return SIEGEAction(
        tool_name="challenge",
        arguments=ChallengeArgs(
            target_agent_id=1,
            claim_id="claim-001",
            flaw_type="type1_false_correlation",
            reasoning="The evidence presented does not support the claimed causation.",
        ),
    )


# ---------------------------------------------------------------------------
# Test 1: Evidence observation and freshness at step 0
# ---------------------------------------------------------------------------

def test_freshness_at_observation_step() -> None:
    """Freshness must be 1.0 immediately when a signal is observed."""
    tracker = TemporalEvidenceTracker(decay_rate=0.15)
    tracker.observe("signal_A", step=3)
    freshness = tracker.freshness("signal_A", current_step=3)
    assert freshness == 1.0, f"Expected 1.0, got {freshness}"


# ---------------------------------------------------------------------------
# Test 2: Freshness decays exponentially over steps
# ---------------------------------------------------------------------------

def test_freshness_decays_over_steps() -> None:
    """Freshness should follow exp(-decay * age)."""
    decay_rate = 0.2
    tracker = TemporalEvidenceTracker(decay_rate=decay_rate)
    tracker.observe("signal_B", step=0)

    for age in range(1, 8):
        expected = math.exp(-decay_rate * age)
        actual = tracker.freshness("signal_B", current_step=age)
        assert abs(actual - expected) < 1e-9, (
            f"Step {age}: expected {expected:.6f}, got {actual:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Urgency is floored at min_urgency for old evidence
# ---------------------------------------------------------------------------

def test_urgency_floored_at_min() -> None:
    """After many steps the urgency must not go below min_urgency."""
    min_urgency = 0.10
    tracker = TemporalEvidenceTracker(decay_rate=0.5, min_urgency=min_urgency)
    tracker.observe("old_signal", step=0)
    # After 20 steps, exp(-0.5 * 20) ≈ 2e-5 which is < min_urgency
    urgency = tracker.urgency("old_signal", current_step=20)
    assert urgency == min_urgency, f"Expected {min_urgency}, got {urgency}"


# ---------------------------------------------------------------------------
# Test 4: R6 = 0 for wrong answer, R6 = urgency for correct answer
# ---------------------------------------------------------------------------

def test_r6_correct_vs_wrong_diagnose() -> None:
    """R6 is 0 for wrong root cause and equals urgency_multiplier for correct."""
    truth = "database_timeout"
    urgency = 0.72

    wrong_action = _diagnose_action("network_partition")
    assert compute_r6_temporal(wrong_action, truth, urgency_multiplier=urgency) == 0.0

    right_action = _diagnose_action(truth)
    r6 = compute_r6_temporal(right_action, truth, urgency_multiplier=urgency)
    assert abs(r6 - urgency) < 1e-6, f"Expected {urgency}, got {r6}"


# ---------------------------------------------------------------------------
# Test 5: R6 defaults to full score (urgency=1.0) for correct fast diagnosis
# ---------------------------------------------------------------------------

def test_r6_full_score_no_temporal_penalty() -> None:
    """When no urgency multiplier is given, correct diagnose gives R6 = 1.0."""
    truth = "config_drift"
    action = _diagnose_action(truth)
    r6 = compute_r6_temporal(action, truth)  # default urgency_multiplier=1.0
    assert r6 == 1.0, f"Expected 1.0, got {r6}"

    # Non-diagnose action should still give 0
    non_diag = _non_diagnose_action()
    assert compute_r6_temporal(non_diag, truth) == 0.0
