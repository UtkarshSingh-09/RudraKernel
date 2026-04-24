"""Trust poisoning strategy library for scripted pathogen NPCs.

A pathogen agent operates in two phases:

  Phase 1 — BUILD TRUST (steps 0 .. strike_step - 1)
    The pathogen acts like a cooperative agent: it always echoes the
    correct root cause and expresses high confidence. Observers' trust
    scores for the pathogen therefore rise toward 1.0.

  Phase 2 — STRIKE (steps strike_step .. end)
    The pathogen injects a false diagnosis with high confidence.
    Because trust is already high, the false claim carries significant
    weight in coalition voting.

This two-phase pattern is the core adversarial mechanic SIEGE trains
the main agent to detect and counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Literal


PathoStage = Literal["build_trust", "strike"]

_BUILD_CONFIDENCE_LOW = 0.85
_BUILD_CONFIDENCE_HIGH = 0.97
_STRIKE_CONFIDENCE_LOW = 0.88
_STRIKE_CONFIDENCE_HIGH = 0.99


@dataclass(slots=True)
class PathogenStrategy:
    """Scripted two-phase trust-poisoning strategy.

    Args:
        agent_id: Unique agent index (0-based).
        seed: Deterministic seed for reproducibility.
        strike_step: First step on which the pathogen switches to Phase 2.
            Steps [0, strike_step) are Phase 1 (build trust).
            Steps [strike_step, ∞) are Phase 2 (strike).
    """

    agent_id: int
    seed: int
    strike_step: int = 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stage(self, step_number: int) -> PathoStage:
        """Return the active strategy stage for the given step."""
        return "build_trust" if step_number < self.strike_step else "strike"

    def generate_claim(
        self,
        template: dict[str, Any],
        *,
        step_number: int,
    ) -> dict[str, Any]:
        """Generate a phase-appropriate diagnosis claim.

        Phase 1: mirrors the true root cause to build trust.
        Phase 2: injects a false root cause to poison coalition voting.
        """
        rng = Random(self.seed + step_number * 131)
        true_root_cause = str(template["root_cause"])
        signals = [str(s) for s in template["observable_signals"]]
        blast_radius = [str(b) for b in template["blast_radius"]]
        claim_id = f"path-{self.agent_id:02d}-step-{step_number:03d}"

        if self.stage(step_number) == "build_trust":
            root_cause = true_root_cause
            low, high = _BUILD_CONFIDENCE_LOW, _BUILD_CONFIDENCE_HIGH
        else:
            # Fabricate a plausible-sounding but wrong diagnosis
            false_target = blast_radius[rng.randrange(len(blast_radius))]
            root_cause = f"false_positive_{false_target}_misconfiguration"
            low, high = _STRIKE_CONFIDENCE_LOW, _STRIKE_CONFIDENCE_HIGH

        confidence = round(rng.uniform(low, high), 2)
        evidence_size = min(len(signals), 2)
        evidence = rng.sample(signals, k=evidence_size)

        return {
            "agent_id": self.agent_id,
            "claim_id": claim_id,
            "root_cause": root_cause,
            "confidence": confidence,
            "stage": self.stage(step_number),
            "evidence": evidence,
        }

    def trust_trajectory(
        self,
        *,
        steps: int,
        initial_trust: float = 0.5,
        p_correct_if_trusted: float = 0.8,
        p_correct_if_untrusted: float = 0.3,
    ) -> list[float]:
        """Simulate the Bayesian trust trajectory an observer would assign
        to this pathogen over `steps` steps.

        Useful for testing the build-then-strike signature in isolation
        without running a full episode.

        Returns:
            List of trust values, one per step (0-indexed).
        """
        trust = initial_trust
        trajectory: list[float] = []

        for step in range(steps):
            trajectory.append(round(trust, 4))
            stage = self.stage(step)
            claim_correct = stage == "build_trust"  # Phase 1 always correct

            if claim_correct:
                likelihood_trusted = p_correct_if_trusted
                likelihood_untrusted = p_correct_if_untrusted
            else:
                likelihood_trusted = 1.0 - p_correct_if_trusted
                likelihood_untrusted = 1.0 - p_correct_if_untrusted

            numerator = likelihood_trusted * trust
            denominator = (
                likelihood_trusted * trust
                + likelihood_untrusted * (1.0 - trust)
            )
            trust = numerator / denominator if denominator > 0.0 else trust

        return trajectory
