"""Gate test for Step 12 — Confidence Calibration + R5.

5 tests covering:
1. Non-diagnose action returns R5 = 0.0.
2. Correct diagnosis with maximum confidence returns R5 = 1.0.
3. Calibration curve: correct + conf=0.5 gives 0.75 (not 1.0).
4. Always-0.5 exploit: wrong diagnosis + conf=0.5 returns 0.75 < correct + conf=1.0.
5. ConfidenceCalibrator: mean R5 across multiple actions is correctly averaged.
"""

from __future__ import annotations

from siege_env.models.actions import SIEGEAction, DiagnoseArgs, ChallengeArgs
from siege_env.rewards.r5_confidence import compute_r5_confidence, ConfidenceCalibrator


_TRUTH = "database_timeout"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diagnose(root_cause: str, confidence: float) -> SIEGEAction:
    return SIEGEAction(
        tool_name="diagnose",
        arguments=DiagnoseArgs(
            root_cause=root_cause,
            confidence=confidence,
            evidence=["latency_p99_spike"],
        ),
    )


def _challenge() -> SIEGEAction:
    return SIEGEAction(
        tool_name="challenge",
        arguments=ChallengeArgs(
            target_agent_id=2,
            claim_id="claim-007",
            flaw_type="type3_tunnel_vision",
            reasoning="The agent ignored recent deployment signals that contradict this hypothesis.",
        ),
    )


# ---------------------------------------------------------------------------
# Test 1: Non-diagnose action → R5 = 0
# ---------------------------------------------------------------------------

def test_r5_non_diagnose_is_zero() -> None:
    """Non-diagnose actions must return 0.0."""
    r5 = compute_r5_confidence(_challenge(), _TRUTH)
    assert r5 == 0.0, f"Expected 0.0 for non-diagnose, got {r5}"


# ---------------------------------------------------------------------------
# Test 2: Correct + fully confident → R5 = 1.0
# ---------------------------------------------------------------------------

def test_r5_correct_full_confidence_is_one() -> None:
    """Correct diagnosis with confidence=1.0 must return R5=1.0."""
    r5 = compute_r5_confidence(_diagnose(_TRUTH, 1.0), _TRUTH)
    assert r5 == 1.0, f"Expected 1.0, got {r5}"


# ---------------------------------------------------------------------------
# Test 3: Calibration curve — correct + 0.5 conf gives 0.75
# ---------------------------------------------------------------------------

def test_r5_correct_half_confidence_is_0_75() -> None:
    """Correct diagnosis with confidence=0.5 should give R5=0.75 (not 1.0)."""
    r5 = compute_r5_confidence(_diagnose(_TRUTH, 0.5), _TRUTH)
    assert abs(r5 - 0.75) < 1e-6, f"Expected 0.75, got {r5}"
    # Critically: not the maximum possible score
    assert r5 < 1.0, "Always-half confidence must not achieve maximum R5"


# ---------------------------------------------------------------------------
# Test 4: Always-0.5 exploit prevention
# ---------------------------------------------------------------------------

def test_r5_always_half_exploit_blocked() -> None:
    """Wrong diagnosis + confidence=0.5 must score LOWER than correct + confidence=1.0."""
    r5_exploit = compute_r5_confidence(_diagnose("wrong_root_cause", 0.5), _TRUTH)
    r5_ideal = compute_r5_confidence(_diagnose(_TRUTH, 1.0), _TRUTH)

    # wrong + 0.5 confidence → 1 - (0.5-0)^2 = 0.75
    assert abs(r5_exploit - 0.75) < 1e-6, (
        f"Wrong+0.5conf should give 0.75 per Brier, got {r5_exploit}"
    )
    # ideal (correct+1.0) → 1.0; exploit cannot reach this ceiling
    assert r5_exploit < r5_ideal, (
        f"Always-0.5 exploit ({r5_exploit}) must be < ideal ({r5_ideal})"
    )
    # Also: wrong + overconfident (conf=1.0) → R5=0.0
    r5_overconfident_wrong = compute_r5_confidence(_diagnose("wrong_root_cause", 1.0), _TRUTH)
    assert r5_overconfident_wrong == 0.0, (
        f"Overconfident wrong diagnosis must give 0.0, got {r5_overconfident_wrong}"
    )


# ---------------------------------------------------------------------------
# Test 5: ConfidenceCalibrator mean R5 across multiple actions
# ---------------------------------------------------------------------------

def test_confidence_calibrator_mean_r5() -> None:
    """ConfidenceCalibrator must correctly average R5 across diagnose actions."""
    cal = ConfidenceCalibrator()

    # Non-diagnose action — should not affect mean
    cal.record(_challenge(), _TRUTH)
    assert cal.num_recorded == 0
    assert cal.mean_r5() == 0.0

    # Correct + conf=1.0 → R5=1.0
    cal.record(_diagnose(_TRUTH, 1.0), _TRUTH)
    # Correct + conf=0.5 → R5=0.75
    cal.record(_diagnose(_TRUTH, 0.5), _TRUTH)
    # Wrong  + conf=1.0 → R5=0.0
    cal.record(_diagnose("wrong_cause", 1.0), _TRUTH)

    assert cal.num_recorded == 3

    # mean = (1.0 + 0.75 + 0.0) / 3 = 0.5833...
    expected = round((1.0 + 0.75 + 0.0) / 3, 6)
    actual = cal.mean_r5()
    assert abs(actual - expected) < 1e-5, f"Expected mean {expected}, got {actual}"

    # Reset clears state
    cal.reset()
    assert cal.num_recorded == 0
    assert cal.mean_r5() == 0.0
