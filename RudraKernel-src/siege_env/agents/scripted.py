"""Rule-based scripted NPC policies for Phase A."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Literal

NPCRole = Literal["lead", "verifier", "contrarian"]
ROLE_CONFIDENCE_BOUNDS: dict[NPCRole, tuple[float, float]] = {
    "lead": (0.72, 0.95),
    "verifier": (0.45, 0.78),
    "contrarian": (0.25, 0.62),
}


@dataclass(slots=True)
class ScriptedNPCAgent:
    """Deterministic, role-conditioned scripted NPC."""

    agent_id: int
    role: NPCRole
    seed: int

    def generate_claim(self, template: dict[str, Any], *, step_number: int) -> dict[str, Any]:
        """Generate a plausible diagnosis claim based on role policy."""

        rng = Random(self.seed + step_number * 101)
        root_cause = str(template["root_cause"])
        signals = [str(signal) for signal in template["observable_signals"]]
        blast_radius = [str(item) for item in template["blast_radius"]]
        low, high = ROLE_CONFIDENCE_BOUNDS[self.role]
        confidence = round(rng.uniform(low, high), 2)

        if self.role == "contrarian":
            # Contrarian produces plausible-but-often-wrong diagnoses in early scripted phase.
            guessed_root_cause = (
                f"suspected_{blast_radius[rng.randrange(len(blast_radius))]}_regression"
            )
        else:
            guessed_root_cause = root_cause

        evidence_size = min(len(signals), 2)
        evidence = rng.sample(signals, k=evidence_size)
        claim_id = f"npc-{self.agent_id:02d}-step-{step_number:03d}"

        return {
            "agent_id": self.agent_id,
            "claim_id": claim_id,
            "root_cause": guessed_root_cause,
            "confidence": confidence,
            "role": self.role,
            "evidence": evidence,
        }
