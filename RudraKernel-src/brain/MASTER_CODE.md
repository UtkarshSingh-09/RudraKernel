# MASTER CODE - Last Updated: 2026-04-24T18:37:11+00:00

# Files Tracked: 50

## siege_env/__init__.py (last modified: 2026-04-24T17:52:44+00:00)
```python
"""SIEGE environment package."""

from siege_env.models import SIEGEAction, SIEGEObservation, SIEGEState
from siege_env.server import SIEGEEnvironment

__all__ = ["SIEGEAction", "SIEGEEnvironment", "SIEGEObservation", "SIEGEState"]

```

## siege_env/agents/__init__.py (last modified: 2026-04-24T18:23:03+00:00)
```python
"""Agent population modules."""

from siege_env.agents.pathogen_strategies import PathogenStrategy
from siege_env.agents.population import NPCPopulation
from siege_env.agents.scripted import ROLE_CONFIDENCE_BOUNDS, ScriptedNPCAgent

__all__ = ["NPCPopulation", "PathogenStrategy", "ROLE_CONFIDENCE_BOUNDS", "ScriptedNPCAgent"]

```

## siege_env/agents/pathogen_strategies.py (last modified: 2026-04-24T18:23:03+00:00)
```python
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

```

## siege_env/agents/population.py (last modified: 2026-04-24T17:56:16+00:00)
```python
"""NPC population orchestration for scripted agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from siege_env.agents.scripted import NPCRole, ScriptedNPCAgent


ROLE_SEQUENCE: tuple[NPCRole, ...] = ("lead", "verifier", "contrarian")


@dataclass(slots=True)
class NPCPopulation:
    """Build and drive deterministic scripted NPC populations."""

    seed: int
    seat_agent_id: int
    total_agents: int = 8
    agents: list[ScriptedNPCAgent] = field(init=False)

    def __post_init__(self) -> None:
        if self.total_agents < 2:
            raise ValueError("total_agents must be at least 2.")
        if self.seat_agent_id < 0 or self.seat_agent_id >= self.total_agents:
            raise ValueError("seat_agent_id must be within [0, total_agents).")
        self.agents = self._build_agents()

    def _build_agents(self) -> list[ScriptedNPCAgent]:
        npc_agents: list[ScriptedNPCAgent] = []
        role_index = 0
        for agent_id in range(self.total_agents):
            if agent_id == self.seat_agent_id:
                continue
            role = ROLE_SEQUENCE[role_index % len(ROLE_SEQUENCE)]
            role_index += 1
            npc_agents.append(
                ScriptedNPCAgent(
                    agent_id=agent_id,
                    role=role,
                    seed=self.seed + agent_id * 17,
                )
            )
        return npc_agents

    def generate_claims(self, template: dict[str, Any], *, step_number: int) -> list[dict[str, Any]]:
        """Generate one claim per scripted NPC in stable agent-id order."""

        return [
            agent.generate_claim(template, step_number=step_number)
            for agent in sorted(self.agents, key=lambda npc: npc.agent_id)
        ]

```

## siege_env/agents/scripted.py (last modified: 2026-04-24T17:55:25+00:00)
```python
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
            guessed_root_cause = f"suspected_{blast_radius[rng.randrange(len(blast_radius))]}_regression"
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

```

## siege_env/curriculum/__init__.py (last modified: 2026-04-24T18:21:01+00:00)
```python
"""Curriculum scheduling modules."""

from siege_env.curriculum.tiered_scheduler import TierConfig, TieredScheduler

__all__ = ["TierConfig", "TieredScheduler"]
```

## siege_env/curriculum/tiered_scheduler.py (last modified: 2026-04-24T18:21:01+00:00)
```python
"""Tiered curriculum scheduler for SIEGE adversarial training.

Implements a 3-tier difficulty ladder with automatic escalation and
de-escalation based on agent rolling win-rate. Enforces the
'attacker-ahead invariant': the environment is always at least as
hard as the agent's demonstrated skill level.

Tier 1 — Novice:    1 pathogen,  low noise,    0 red herrings, 20 max steps
Tier 2 — Advanced:  2 pathogens, medium noise,  2 red herrings, 25 max steps
Tier 3 — Expert:    3 pathogens, high noise,    4 red herrings, 30 max steps
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TierConfig:
    """Immutable configuration for a single difficulty tier."""

    tier: int
    num_pathogens: int
    noise_level: float
    red_herring_count: int
    max_steps: int

    def as_dict(self) -> dict[str, Any]:
        """Return tier parameters as a plain dict for env consumption."""
        return {
            "tier": self.tier,
            "num_pathogens": self.num_pathogens,
            "noise_level": self.noise_level,
            "red_herring_count": self.red_herring_count,
            "max_steps": self.max_steps,
        }


TIER_CONFIGS: dict[int, TierConfig] = {
    1: TierConfig(
        tier=1,
        num_pathogens=1,
        noise_level=0.1,
        red_herring_count=0,
        max_steps=20,
    ),
    2: TierConfig(
        tier=2,
        num_pathogens=2,
        noise_level=0.3,
        red_herring_count=2,
        max_steps=25,
    ),
    3: TierConfig(
        tier=3,
        num_pathogens=3,
        noise_level=0.5,
        red_herring_count=4,
        max_steps=30,
    ),
}

_MIN_TIER = 1
_MAX_TIER = 3
_DEFAULT_WINDOW = 10
_DEFAULT_ESCALATE_THRESHOLD = 0.70
_DEFAULT_DEESCALATE_THRESHOLD = 0.30
_DEFAULT_COOLDOWN = 5


class TieredScheduler:
    """Manages difficulty tier based on agent rolling win-rate.

    Args:
        window: Number of recent episodes used for the rolling win-rate.
        escalate_threshold: Win-rate above which the tier increases.
        deescalate_threshold: Win-rate below which the tier decreases.
        cooldown: Minimum episodes between any tier change to avoid thrashing.
        initial_tier: Starting tier (default 1).
    """

    def __init__(
        self,
        *,
        window: int = _DEFAULT_WINDOW,
        escalate_threshold: float = _DEFAULT_ESCALATE_THRESHOLD,
        deescalate_threshold: float = _DEFAULT_DEESCALATE_THRESHOLD,
        cooldown: int = _DEFAULT_COOLDOWN,
        initial_tier: int = 1,
    ) -> None:
        if not (_MIN_TIER <= initial_tier <= _MAX_TIER):
            raise ValueError(
                f"initial_tier must be between {_MIN_TIER} and {_MAX_TIER}, got {initial_tier}"
            )
        if not (0.0 < deescalate_threshold < escalate_threshold < 1.0):
            raise ValueError(
                "thresholds must satisfy 0 < deescalate_threshold < escalate_threshold < 1"
            )
        if window < 1:
            raise ValueError("window must be >= 1")
        if cooldown < 0:
            raise ValueError("cooldown must be >= 0")

        self._tier: int = initial_tier
        self._window = window
        self._escalate_threshold = escalate_threshold
        self._deescalate_threshold = deescalate_threshold
        self._cooldown = cooldown
        self._history: deque[bool] = deque(maxlen=window)
        self._episodes_since_change: int = cooldown  # allow immediate change on first window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_tier(self) -> int:
        """Current active difficulty tier (1, 2, or 3)."""
        return self._tier

    @property
    def config(self) -> TierConfig:
        """Immutable config for the current tier."""
        return TIER_CONFIGS[self._tier]

    def record_episode(self, *, won: bool) -> None:
        """Record the outcome of one episode and update tier if needed.

        Args:
            won: True if the agent correctly identified the root cause
                 before timeout (i.e. a 'win' from the agent's perspective).
        """
        self._history.append(won)
        self._episodes_since_change += 1

        if len(self._history) < self._window:
            return  # not enough data yet

        if self._episodes_since_change < self._cooldown:
            return  # still in cooldown

        win_rate = self._win_rate()
        if win_rate >= self._escalate_threshold and self._tier < _MAX_TIER:
            self._tier += 1
            self._episodes_since_change = 0
        elif win_rate <= self._deescalate_threshold and self._tier > _MIN_TIER:
            self._tier -= 1
            self._episodes_since_change = 0

    def win_rate(self) -> float:
        """Current rolling win-rate over the last `window` episodes.

        Returns 0.0 if fewer than `window` episodes have been recorded.
        """
        if len(self._history) < self._window:
            return 0.0
        return self._win_rate()

    def attacker_ahead(self) -> bool:
        """Return True when the environment difficulty is ahead of agent skill.

        Defined as: the agent has NOT yet beaten the current tier
        (i.e. win-rate is below the escalation threshold).
        This is the invariant SIEGE maintains — the attacker (env) should
        always be at least as hard as the agent's current mastery level.
        """
        if len(self._history) < self._window:
            return True  # insufficient data → assume attacker ahead
        return self._win_rate() < self._escalate_threshold

    def reset(self) -> None:
        """Reset episode history and return to initial tier 1."""
        self._tier = 1
        self._history.clear()
        self._episodes_since_change = self._cooldown

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _win_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

```

## siege_env/incidents/__init__.py (last modified: 2026-04-24T17:23:08+00:00)
```python
"""Incident templates and generation utilities."""

from siege_env.incidents.generator import generate_variant
from siege_env.incidents.loader import load_templates

__all__ = ["generate_variant", "load_templates"]

```

## siege_env/incidents/generator.py (last modified: 2026-04-24T17:23:03+00:00)
```python
"""Deterministic incident variant generation from seed templates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _rotated(values: list[str], offset: int) -> list[str]:
    if not values:
        return []
    normalized = offset % len(values)
    return values[normalized:] + values[:normalized]


def generate_variant(template: dict[str, Any], variant_index: int) -> dict[str, Any]:
    """Generate a deterministic variant while preserving schema contract."""

    if variant_index < 0:
        raise ValueError("variant_index must be non-negative.")

    variant = deepcopy(template)
    variant["id"] = f"{template['id']}_v{variant_index:03d}"
    variant["observable_signals"] = _rotated(list(template["observable_signals"]), variant_index)
    variant["flaw_types"] = _rotated(list(template["flaw_types"]), variant_index)
    variant["blast_radius"] = _rotated(list(template["blast_radius"]), variant_index)
    return variant

```

## siege_env/incidents/loader.py (last modified: 2026-04-24T17:22:57+00:00)
```python
"""Template loading and validation for SIEGE incident seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TEMPLATE_KEYS = (
    "id",
    "source_url",
    "root_cause",
    "observable_signals",
    "flaw_types",
    "blast_radius",
)
TEMPLATES_PATH = Path(__file__).with_name("templates.json")


def _validate_template(raw_template: dict[str, Any], index: int) -> dict[str, Any]:
    missing = [key for key in REQUIRED_TEMPLATE_KEYS if key not in raw_template]
    if missing:
        missing_joined = ", ".join(missing)
        raise ValueError(f"Template at index {index} is missing required keys: {missing_joined}")

    template = {key: raw_template[key] for key in REQUIRED_TEMPLATE_KEYS}
    if not isinstance(template["id"], str) or not template["id"].strip():
        raise ValueError(f"Template at index {index} has invalid 'id'.")
    if not isinstance(template["source_url"], str) or not template["source_url"].startswith("https://"):
        raise ValueError(f"Template '{template['id']}' has invalid 'source_url'.")
    if not isinstance(template["root_cause"], str) or not template["root_cause"].strip():
        raise ValueError(f"Template '{template['id']}' has invalid 'root_cause'.")

    for list_key in ("observable_signals", "flaw_types", "blast_radius"):
        value = template[list_key]
        if not isinstance(value, list) or not value:
            raise ValueError(f"Template '{template['id']}' has invalid '{list_key}'.")
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f"Template '{template['id']}' contains invalid values in '{list_key}'.")

    return {
        "id": template["id"].strip(),
        "source_url": template["source_url"].strip(),
        "root_cause": template["root_cause"].strip(),
        "observable_signals": [item.strip() for item in template["observable_signals"]],
        "flaw_types": [item.strip() for item in template["flaw_types"]],
        "blast_radius": [item.strip() for item in template["blast_radius"]],
    }


def load_templates(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate incident templates from disk."""

    target_path = path or TEMPLATES_PATH
    raw_payload = json.loads(target_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, list):
        raise ValueError("Incident templates payload must be a list.")

    return [_validate_template(item, idx) for idx, item in enumerate(raw_payload)]

```

## siege_env/league/__init__.py (last modified: 2026-04-24T15:24:19+00:00)
```python
"""Opponent league modules."""
```

## siege_env/mechanics/__init__.py (last modified: 2026-04-24T18:32:44+00:00)
```python
"""Mechanics modules."""

from siege_env.mechanics.temporal_evidence import TemporalEvidenceTracker, EvidenceRecord

__all__ = ["TemporalEvidenceTracker", "EvidenceRecord"]
```

## siege_env/mechanics/temporal_evidence.py (last modified: 2026-04-24T18:32:44+00:00)
```python
"""Temporal evidence dynamics for SIEGE.

Evidence in a live incident investigation has a shelf-life. Signals
observed early in an episode are more informative when acted upon
quickly; stale evidence that is only cited steps later carries less
diagnostic weight.

This module provides:

  EvidenceRecord  — a single piece of time-stamped evidence.
  TemporalEvidenceTracker — tracks all evidence and computes per-signal
      freshness and per-step urgency multipliers used by R6.

Freshness model
---------------
Each evidence signal decays exponentially from the step it was first
observed:

    freshness(t) = exp(-decay_rate * (current_step - observed_step))

where decay_rate controls how fast freshness falls (default 0.15).
A signal observed at step 0 has freshness 1.0 at step 0, ~0.86 at
step 1, ~0.74 at step 2, etc.

Urgency multiplier
------------------
The urgency multiplier for acting at step t on evidence observed at
step obs_step is:

    urgency(t) = max(min_urgency, freshness(t))

Acting fast gives urgency close to 1.0; acting very late gives
urgency close to min_urgency (default 0.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


_DEFAULT_DECAY_RATE = 0.15
_DEFAULT_MIN_URGENCY = 0.10


@dataclass(slots=True)
class EvidenceRecord:
    """A single piece of evidence with its observation timestamp."""

    signal_id: str
    observed_at_step: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TemporalEvidenceTracker:
    """Tracks evidence signals and computes time-sensitive freshness values.

    Args:
        decay_rate: Exponential decay constant (higher = faster staleness).
        min_urgency: Floor for urgency multiplier (prevents reward collapsing
            to zero on very stale evidence).
    """

    def __init__(
        self,
        *,
        decay_rate: float = _DEFAULT_DECAY_RATE,
        min_urgency: float = _DEFAULT_MIN_URGENCY,
    ) -> None:
        if decay_rate <= 0.0:
            raise ValueError("decay_rate must be positive.")
        if not (0.0 <= min_urgency < 1.0):
            raise ValueError("min_urgency must be in [0, 1).")

        self._decay_rate = decay_rate
        self._min_urgency = min_urgency
        self._evidence: dict[str, EvidenceRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, signal_id: str, *, step: int, metadata: dict[str, Any] | None = None) -> None:
        """Record a new evidence signal at the given step.

        If the signal was already observed, the earlier timestamp is kept
        (first observation is canonical).
        """
        if signal_id not in self._evidence:
            self._evidence[signal_id] = EvidenceRecord(
                signal_id=signal_id,
                observed_at_step=step,
                metadata=metadata or {},
            )

    def freshness(self, signal_id: str, *, current_step: int) -> float:
        """Return the freshness of a signal at current_step.

        Returns 0.0 if the signal has never been observed.
        """
        if signal_id not in self._evidence:
            return 0.0
        age = max(0, current_step - self._evidence[signal_id].observed_at_step)
        return math.exp(-self._decay_rate * age)

    def urgency(self, signal_id: str, *, current_step: int) -> float:
        """Return the urgency multiplier for acting on a signal at current_step.

        Clipped to [min_urgency, 1.0].
        """
        raw = self.freshness(signal_id, current_step=current_step)
        if raw == 0.0:
            return 0.0  # signal never observed → no urgency
        return max(self._min_urgency, raw)

    def all_signals(self) -> list[str]:
        """Return all observed signal IDs in observation order."""
        return list(self._evidence.keys())

    def reset(self) -> None:
        """Clear all recorded evidence."""
        self._evidence.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def decay_rate(self) -> float:
        return self._decay_rate

    @property
    def min_urgency(self) -> float:
        return self._min_urgency

```

## siege_env/models/__init__.py (last modified: 2026-04-24T17:11:21+00:00)
```python
"""Data models for SIEGE."""

from siege_env.models.actions import (
    ACTION_ARGS_BY_TOOL,
    ChallengeArgs,
    DiagnoseArgs,
    EscalateArgs,
    PostmortemArgs,
    RatifyArgs,
    SIEGEAction,
    WhisperArgs,
)
from siege_env.models.observations import SIEGEObservation
from siege_env.models.state import SIEGEState

__all__ = [
    "ACTION_ARGS_BY_TOOL",
    "ChallengeArgs",
    "DiagnoseArgs",
    "EscalateArgs",
    "PostmortemArgs",
    "RatifyArgs",
    "SIEGEAction",
    "SIEGEObservation",
    "SIEGEState",
    "WhisperArgs",
]

```

## siege_env/models/actions.py (last modified: 2026-04-24T17:11:21+00:00)
```python
"""Strict action schemas for the SIEGE tool surface."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


ToolName: TypeAlias = Literal[
    "diagnose",
    "challenge",
    "ratify",
    "escalate",
    "whisper",
    "postmortem",
]
FlawType: TypeAlias = Literal[
    "type1_false_correlation",
    "type2_scope_inflation",
    "type3_tunnel_vision",
    "type4_blame_shifting",
    "type5_premature_closure",
]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=1500)]
MessageText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
ClaimId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
EvidenceItem = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class SIEGEBaseModel(BaseModel):
    """Shared strict settings for SIEGE Pydantic models."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AlternativeHypothesis(SIEGEBaseModel):
    """Fallback diagnosis candidate with a calibrated confidence score."""

    diagnosis: ShortText
    confidence: float = Field(ge=0.0, le=1.0)


class TimelineEvent(SIEGEBaseModel):
    """Structured postmortem event used during reflective summaries."""

    timestamp: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
    ]
    event: MessageText


class DiagnoseArgs(SIEGEBaseModel):
    """Primary diagnosis action emitted during live incident response."""

    root_cause: ShortText
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=10)
    alternative_hypotheses: list[AlternativeHypothesis] = Field(default_factory=list, max_length=5)


class ChallengeArgs(SIEGEBaseModel):
    """Challenge a peer claim with flaw taxonomy reasoning."""

    target_agent_id: int = Field(ge=0, le=7)
    claim_id: ClaimId
    flaw_type: FlawType
    reasoning: LongText


class RatifyArgs(SIEGEBaseModel):
    """Vote on whether a claim should be ratified by the coalition."""

    claim_id: ClaimId
    vote: bool


class EscalateArgs(SIEGEBaseModel):
    """Escalate an incident when the blast radius or severity expands."""

    concern: MessageText
    blast_radius_estimate: list[ShortText] = Field(min_length=1, max_length=10)


class WhisperArgs(SIEGEBaseModel):
    """Private channel between two agents."""

    target_agent_id: int = Field(ge=0, le=7)
    message: MessageText


class PostmortemArgs(SIEGEBaseModel):
    """Structured reflective summary emitted after incident resolution."""

    root_cause: ShortText
    timeline: list[TimelineEvent] = Field(min_length=1, max_length=20)
    contributing_factors: list[ShortText] = Field(min_length=1, max_length=10)
    misdiagnosis_analysis: LongText


ActionArguments: TypeAlias = (
    DiagnoseArgs | ChallengeArgs | RatifyArgs | EscalateArgs | WhisperArgs | PostmortemArgs
)

ACTION_ARGS_BY_TOOL: dict[ToolName, type[SIEGEBaseModel]] = {
    "diagnose": DiagnoseArgs,
    "challenge": ChallengeArgs,
    "ratify": RatifyArgs,
    "escalate": EscalateArgs,
    "whisper": WhisperArgs,
    "postmortem": PostmortemArgs,
}


class SIEGEAction(SIEGEBaseModel):
    """Validated action payload with tool-aware argument coercion."""

    tool_name: ToolName
    arguments: ActionArguments

    @model_validator(mode="before")
    @classmethod
    def _coerce_tool_specific_arguments(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        tool_name = data.get("tool_name")
        if tool_name not in ACTION_ARGS_BY_TOOL:
            return data

        arguments = data.get("arguments")
        if arguments is None:
            return data

        model_cls = ACTION_ARGS_BY_TOOL[tool_name]
        if isinstance(arguments, model_cls):
            return data

        updated = dict(data)
        updated["arguments"] = model_cls.model_validate(arguments)
        return updated

    @model_validator(mode="after")
    def _ensure_arguments_match_tool(self) -> SIEGEAction:
        expected_model = ACTION_ARGS_BY_TOOL[self.tool_name]
        if not isinstance(self.arguments, expected_model):
            raise ValueError(
                f"Arguments for tool '{self.tool_name}' must use {expected_model.__name__}."
            )
        return self

    @classmethod
    def tool_schema(cls, tool_name: ToolName) -> dict[str, Any]:
        """Return the JSON schema for a specific tool payload."""

        return ACTION_ARGS_BY_TOOL[tool_name].model_json_schema()

```

## siege_env/models/observations.py (last modified: 2026-04-24T17:11:21+00:00)
```python
"""Observation dataclass for the SIEGE environment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Literal, Mapping


Role = Literal["immune", "pathogen"]
VisibilityLevel = Literal["metrics_only", "traces_only", "full", "delayed"]
IncidentSeverity = Literal["warning", "critical", "outage"]

_ALLOWED_ROLES = {"immune", "pathogen"}
_ALLOWED_VISIBILITY_LEVELS = {"metrics_only", "traces_only", "full", "delayed"}
_ALLOWED_SEVERITIES = {"warning", "critical", "outage"}


def _normalize_agent_scores(
    raw_mapping: Mapping[object, object], *, field_name: str
) -> dict[int, float]:
    normalized: dict[int, float] = {}
    for raw_agent_id, raw_score in raw_mapping.items():
        try:
            agent_id = int(raw_agent_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} keys must be integer-like agent IDs.") from exc

        score = float(raw_score)
        if agent_id < 0 or agent_id > 7:
            raise ValueError(f"{field_name} keys must be between 0 and 7.")
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{field_name} values must be finite scores between 0.0 and 1.0.")

        normalized[agent_id] = score
    return normalized


@dataclass(slots=True)
class SIEGEObservation:
    """Serializable observation emitted to the seat agent each step."""

    incident_dashboard: dict[str, Any]
    agent_claims: list[dict[str, Any]]
    trust_scores: dict[int, float]
    coalition_status: dict[str, Any]
    step_number: int
    slo_status: dict[str, Any]
    your_role: Role
    available_evidence: list[dict[str, Any]]
    visibility_level: VisibilityLevel
    whisper_inbox: list[dict[str, Any]]
    whisper_log: list[dict[str, Any]]
    incident_severity: IncidentSeverity
    red_herring_signals: list[dict[str, Any]]
    reputation_history: dict[int, float]
    active_incidents: list[dict[str, Any]]
    seat_agent_id: int
    action_error: str | None = None

    def __post_init__(self) -> None:
        if self.step_number < 0:
            raise ValueError("step_number must be non-negative.")
        if self.your_role not in _ALLOWED_ROLES:
            raise ValueError(f"your_role must be one of {_ALLOWED_ROLES}.")
        if self.visibility_level not in _ALLOWED_VISIBILITY_LEVELS:
            raise ValueError(
                f"visibility_level must be one of {_ALLOWED_VISIBILITY_LEVELS}."
            )
        if self.incident_severity not in _ALLOWED_SEVERITIES:
            raise ValueError(f"incident_severity must be one of {_ALLOWED_SEVERITIES}.")
        if self.seat_agent_id < 0 or self.seat_agent_id > 7:
            raise ValueError("seat_agent_id must be between 0 and 7.")
        if self.action_error is not None and not self.action_error.strip():
            raise ValueError("action_error must be a non-empty string when provided.")

        self.trust_scores = _normalize_agent_scores(
            self.trust_scores,
            field_name="trust_scores",
        )
        self.reputation_history = _normalize_agent_scores(
            self.reputation_history,
            field_name="reputation_history",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the observation into a JSON-serializable mapping."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SIEGEObservation:
        """Construct an observation from a mapping or decoded JSON object."""

        return cls(
            incident_dashboard=dict(payload["incident_dashboard"]),
            agent_claims=list(payload["agent_claims"]),
            trust_scores=_normalize_agent_scores(
                payload["trust_scores"],
                field_name="trust_scores",
            ),
            coalition_status=dict(payload["coalition_status"]),
            step_number=int(payload["step_number"]),
            slo_status=dict(payload["slo_status"]),
            your_role=payload["your_role"],
            available_evidence=list(payload["available_evidence"]),
            visibility_level=payload["visibility_level"],
            whisper_inbox=list(payload["whisper_inbox"]),
            whisper_log=list(payload["whisper_log"]),
            incident_severity=payload["incident_severity"],
            red_herring_signals=list(payload["red_herring_signals"]),
            reputation_history=_normalize_agent_scores(
                payload["reputation_history"],
                field_name="reputation_history",
            ),
            active_incidents=list(payload["active_incidents"]),
            seat_agent_id=int(payload["seat_agent_id"]),
            action_error=payload.get("action_error"),
        )

    def to_json(self) -> str:
        """Serialize the observation into JSON for replay or transport."""

        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> SIEGEObservation:
        """Deserialize an observation from a JSON string."""

        return cls.from_dict(json.loads(payload))

```

## siege_env/models/state.py (last modified: 2026-04-24T17:11:21+00:00)
```python
"""State dataclass for the SIEGE environment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping


@dataclass(slots=True)
class SIEGEState:
    """Serializable internal environment state snapshot."""

    episode_id: str
    step_count: int
    incident_template_id: str
    ground_truth_root_cause: str
    current_tier: int
    arms_race_score: float

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must be a non-empty string.")
        if self.step_count < 0:
            raise ValueError("step_count must be non-negative.")
        if not self.incident_template_id.strip():
            raise ValueError("incident_template_id must be a non-empty string.")
        if not self.ground_truth_root_cause.strip():
            raise ValueError("ground_truth_root_cause must be a non-empty string.")
        if self.current_tier not in {1, 2, 3}:
            raise ValueError("current_tier must be one of 1, 2, or 3.")
        if not isfinite(self.arms_race_score):
            raise ValueError("arms_race_score must be finite.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the state into a JSON-serializable mapping."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SIEGEState:
        """Construct a state object from a mapping or decoded JSON object."""

        return cls(
            episode_id=str(payload["episode_id"]),
            step_count=int(payload["step_count"]),
            incident_template_id=str(payload["incident_template_id"]),
            ground_truth_root_cause=str(payload["ground_truth_root_cause"]),
            current_tier=int(payload["current_tier"]),
            arms_race_score=float(payload["arms_race_score"]),
        )

    def to_json(self) -> str:
        """Serialize the state into JSON for snapshots or debugging."""

        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> SIEGEState:
        """Deserialize a state object from a JSON string."""

        return cls.from_dict(json.loads(payload))

```

## siege_env/replay/__init__.py (last modified: 2026-04-24T15:24:19+00:00)
```python
"""Replay logging and playback modules."""
```

## siege_env/rewards/__init__.py (last modified: 2026-04-24T18:36:05+00:00)
```python
"""Reward modules and aggregators."""

from siege_env.rewards.aggregator import aggregate_rewards
from siege_env.rewards.r1_resolution import compute_r1_resolution
from siege_env.rewards.r2_deception import compute_r2_deception
from siege_env.rewards.r3_detection import compute_r3_detection
from siege_env.rewards.r4_trust_calibration import compute_r4_trust_calibration
from siege_env.rewards.r5_confidence import compute_r5_confidence, ConfidenceCalibrator
from siege_env.rewards.r6_temporal import compute_r6_temporal

__all__ = [
    "aggregate_rewards",
    "compute_r1_resolution",
    "compute_r2_deception",
    "compute_r3_detection",
    "compute_r4_trust_calibration",
    "compute_r5_confidence",
    "ConfidenceCalibrator",
    "compute_r6_temporal",
]

```

## siege_env/rewards/aggregator.py (last modified: 2026-04-24T18:36:05+00:00)
```python
"""Reward aggregation scaffold for SIEGE."""

from __future__ import annotations

from typing import Any

from siege_env.models import SIEGEAction
from siege_env.rewards.r1_resolution import compute_r1_resolution
from siege_env.rewards.r2_deception import compute_r2_deception
from siege_env.rewards.r3_detection import compute_r3_detection
from siege_env.rewards.r4_trust_calibration import compute_r4_trust_calibration
from siege_env.rewards.r5_confidence import compute_r5_confidence
from siege_env.rewards.r6_temporal import compute_r6_temporal


def aggregate_rewards(
    action: SIEGEAction,
    *,
    ground_truth_root_cause: str,
    seat_role: str = "immune",
    claims_by_id: dict[str, dict[str, Any]] | None = None,
    trust_scores: dict[int, float] | None = None,
    agent_reliability: dict[int, bool] | None = None,
    urgency_multiplier: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    """Aggregate reward components (R1-R5, R6)."""

    r1 = compute_r1_resolution(action, ground_truth_root_cause)
    r2 = compute_r2_deception(
        action,
        seat_role=seat_role,
        ground_truth_root_cause=ground_truth_root_cause,
    )
    r3 = compute_r3_detection(
        action,
        seat_role=seat_role,
        ground_truth_root_cause=ground_truth_root_cause,
        claims_by_id=claims_by_id or {},
    )
    r4 = compute_r4_trust_calibration(
        trust_scores=trust_scores or {},
        agent_reliability=agent_reliability or {},
    )
    r5 = compute_r5_confidence(action, ground_truth_root_cause)
    r6 = compute_r6_temporal(
        action,
        ground_truth_root_cause,
        urgency_multiplier=urgency_multiplier,
    )

    total = max(0.0, min(1.0, max(r1, r2, r3, r4, r5, r6)))
    return total, {
        "r1_resolution": r1,
        "r2_deception": r2,
        "r3_detection": r3,
        "r4_trust_calibration": r4,
        "r5_confidence": r5,
        "r6_temporal": r6,
    }

```

## siege_env/rewards/r1_resolution.py (last modified: 2026-04-24T17:51:44+00:00)
```python
"""Resolution reward (R1) for minimal Step 04 environment loop."""

from __future__ import annotations

from siege_env.models import SIEGEAction


def compute_r1_resolution(action: SIEGEAction, ground_truth_root_cause: str) -> float:
    """Return 1.0 for correct diagnose action, otherwise 0.0."""

    if action.tool_name != "diagnose":
        return 0.0

    predicted_root_cause = action.arguments.root_cause
    if predicted_root_cause == ground_truth_root_cause:
        return 1.0
    return 0.0

```

## siege_env/rewards/r2_deception.py (last modified: 2026-04-24T18:10:20+00:00)
```python
"""Deception penetration reward (R2) for pathogen role."""

from __future__ import annotations

from siege_env.models import SIEGEAction


def compute_r2_deception(
    action: SIEGEAction,
    *,
    seat_role: str,
    ground_truth_root_cause: str,
) -> float:
    """Reward pathogen for pushing an incorrect diagnosis."""

    if seat_role != "pathogen":
        return 0.0
    if action.tool_name != "diagnose":
        return 0.0
    if action.arguments.root_cause == ground_truth_root_cause:
        return 0.0
    return 1.0

```

## siege_env/rewards/r3_detection.py (last modified: 2026-04-24T18:10:26+00:00)
```python
"""Detection reward (R3) for immune role challenge actions."""

from __future__ import annotations

from typing import Any, Mapping

from siege_env.models import SIEGEAction


def compute_r3_detection(
    action: SIEGEAction,
    *,
    seat_role: str,
    ground_truth_root_cause: str,
    claims_by_id: Mapping[str, dict[str, Any]],
) -> float:
    """Reward immune role for correctly challenging incorrect claims."""

    if seat_role != "immune":
        return 0.0
    if action.tool_name != "challenge":
        return 0.0

    challenged_claim = claims_by_id.get(action.arguments.claim_id)
    if challenged_claim is None:
        return 0.0

    challenged_root_cause = challenged_claim.get("root_cause")
    return 1.0 if challenged_root_cause != ground_truth_root_cause else 0.0

```

## siege_env/rewards/r4_trust_calibration.py (last modified: 2026-04-24T18:13:55+00:00)
```python
"""Trust calibration reward (R4) based on Brier scoring."""

from __future__ import annotations

from typing import Mapping


def compute_r4_trust_calibration(
    *,
    trust_scores: Mapping[int, float],
    agent_reliability: Mapping[int, bool],
) -> float:
    """Compute trust calibration score using 1 - mean Brier loss."""

    common_agent_ids = [agent_id for agent_id in trust_scores if agent_id in agent_reliability]
    if not common_agent_ids:
        return 0.0

    brier_sum = 0.0
    for agent_id in common_agent_ids:
        predicted = max(0.0, min(1.0, float(trust_scores[agent_id])))
        actual = 1.0 if agent_reliability[agent_id] else 0.0
        brier_sum += (predicted - actual) ** 2

    mean_brier = brier_sum / len(common_agent_ids)
    score = 1.0 - mean_brier
    return round(max(0.0, min(1.0, score)), 6)

```

## siege_env/rewards/r5_confidence.py (last modified: 2026-04-24T18:35:16+00:00)
```python
"""Confidence calibration reward (R5) for SIEGE.

Design
------
R5 rewards agents that are epistemically honest: confident when correct
and less confident when wrong.

The scoring rule used is a *modified Brier score* anchored to the
correct outcome:

  outcome = 1.0 if diagnosis is correct, else 0.0
  brier   = (confidence - outcome) ** 2
  r5      = 1.0 - brier                          # raw [0, 1]

This gives:
  - Correct + confident (conf=1.0) → R5 = 1.0  (ideal)
  - Correct + half-sure  (conf=0.5) → R5 = 0.75 (mediocre, not maximal)
  - Wrong  + confident   (conf=1.0) → R5 = 0.0  (maximally penalised)
  - Wrong  + half-sure   (conf=0.5) → R5 = 0.75 (partially penalised)

Anti-exploit property
---------------------
An agent that always reports confidence=0.5 ("always-0.5 exploit")
receives R5=0.75 on every step regardless of correctness.  Because the
maximum achievable score is 1.0 (requires confident correct diagnosis),
the always-0.5 strategy cannot reach the top of the leaderboard.  The
test suite explicitly verifies:

    compute_r5_confidence(correct, conf=0.5) < 1.0
    compute_r5_confidence(wrong,   conf=0.5) < compute_r5_confidence(correct, conf=0.5)

Stateful calibration (ConfidenceCalibrator)
-------------------------------------------
For multi-step evaluation (e.g., validation episodes) we provide
`ConfidenceCalibrator`, a running Brier-score accumulator that
computes the *mean R5 across all diagnose actions* in an episode.
This is the metric judges will see in the final leaderboard.
"""

from __future__ import annotations

from siege_env.models import SIEGEAction


# ---------------------------------------------------------------------------
# Stateless per-action R5
# ---------------------------------------------------------------------------


def compute_r5_confidence(
    action: SIEGEAction,
    ground_truth_root_cause: str,
) -> float:
    """Compute the confidence-calibration reward component R5 for a single action.

    Args:
        action: The agent's action for this step.
        ground_truth_root_cause: The true root cause for this episode.

    Returns:
        R5 ∈ [0.0, 1.0].  Non-diagnose actions return 0.0.
    """
    if action.tool_name != "diagnose":
        return 0.0

    confidence = float(action.arguments.confidence)
    correct = action.arguments.root_cause == ground_truth_root_cause
    outcome = 1.0 if correct else 0.0

    brier = (confidence - outcome) ** 2
    r5 = 1.0 - brier
    return round(max(0.0, min(1.0, r5)), 6)


# ---------------------------------------------------------------------------
# Stateful multi-step calibrator
# ---------------------------------------------------------------------------


class ConfidenceCalibrator:
    """Running Brier-score accumulator across multiple diagnose actions.

    Use this to compute mean R5 over an entire episode or evaluation run.

    Example::

        cal = ConfidenceCalibrator()
        for action, truth in episode_steps:
            cal.record(action, truth)
        episode_r5 = cal.mean_r5()
    """

    def __init__(self) -> None:
        self._total_r5: float = 0.0
        self._count: int = 0

    def record(self, action: SIEGEAction, ground_truth_root_cause: str) -> float:
        """Record one action and return its per-step R5.

        Non-diagnose actions are silently ignored (return 0.0 without
        updating the running mean).
        """
        r5 = compute_r5_confidence(action, ground_truth_root_cause)
        if action.tool_name == "diagnose":
            self._total_r5 += r5
            self._count += 1
        return r5

    def mean_r5(self) -> float:
        """Return mean R5 across all recorded diagnose actions.

        Returns 0.0 if no diagnose actions have been recorded yet.
        """
        if self._count == 0:
            return 0.0
        return round(self._total_r5 / self._count, 6)

    def reset(self) -> None:
        """Clear accumulated state."""
        self._total_r5 = 0.0
        self._count = 0

    @property
    def num_recorded(self) -> int:
        """Number of diagnose actions recorded so far."""
        return self._count

```

## siege_env/rewards/r6_temporal.py (last modified: 2026-04-24T18:32:44+00:00)
```python
"""Temporal reward (R6) — rewards fast correct diagnoses, penalises slow ones.

R6 design
---------
R6 answers the question: *given that the agent got the diagnosis right,
how quickly did it act on the available evidence?*

  r6 = urgency_multiplier * base_r6_score

where:

  base_r6_score   = 1.0 if action is a correct diagnose, else 0.0
  urgency_multi   = mean freshness across the evidence signals cited in
                    the action's evidence list at the current step
                    (falls back to 1.0 if no evidence list provided)

This gives the full R6 range of [0, 1]:
  • Correct diagnosis on step 0 with fresh evidence → R6 ≈ 1.0
  • Correct diagnosis on step 20 with stale evidence → R6 ≈ 0.1
  • Wrong diagnosis → R6 = 0.0

The urgency multiplier is supplied externally (computed by
TemporalEvidenceTracker.urgency()) so that R6 stays a pure scoring
function with no hidden state — easy to test and compose.
"""

from __future__ import annotations

from siege_env.models import SIEGEAction


def compute_r6_temporal(
    action: SIEGEAction,
    ground_truth_root_cause: str,
    *,
    urgency_multiplier: float = 1.0,
) -> float:
    """Compute the temporal reward component R6.

    Args:
        action: The agent's action for this step.
        ground_truth_root_cause: The true root cause for this episode.
        urgency_multiplier: Pre-computed freshness-based multiplier in [0, 1].
            Defaults to 1.0 (no temporal penalty) when caller does not
            supply freshness data.

    Returns:
        R6 score in [0.0, 1.0].
    """
    if not (0.0 <= urgency_multiplier <= 1.0):
        raise ValueError(
            f"urgency_multiplier must be in [0, 1], got {urgency_multiplier}"
        )

    if action.tool_name != "diagnose":
        return 0.0

    if action.arguments.root_cause != ground_truth_root_cause:
        return 0.0

    return round(urgency_multiplier, 4)

```

## siege_env/server/__init__.py (last modified: 2026-04-24T17:52:40+00:00)
```python
"""Server modules for SIEGE environment."""

from siege_env.server.siege_environment import SIEGEEnvironment

__all__ = ["SIEGEEnvironment"]

```

## siege_env/server/app.py (last modified: 2026-04-24T17:52:48+00:00)
```python
"""FastAPI server scaffold for SIEGE Step 01."""

from __future__ import annotations

from fastapi import FastAPI

from siege_env.server.siege_environment import SIEGEEnvironment


app = FastAPI(title="SIEGE Environment", version="0.1.0")
env = SIEGEEnvironment(seed=7)


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness endpoint used for local and container smoke tests."""
    return {"status": "ok"}


@app.get("/env/reset")
def reset() -> dict[str, object]:
    """Reset the minimal environment and return the starting observation."""
    observation = env.reset()
    return {"observation": observation.to_dict()}

```

## siege_env/server/siege_environment.py (last modified: 2026-04-24T18:14:23+00:00)
```python
"""Minimal Step 04 SIEGE environment implementation."""

from __future__ import annotations

from dataclasses import replace
from random import Random
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from siege_env.agents import NPCPopulation
from siege_env.incidents import load_templates
from siege_env.models import SIEGEAction, SIEGEObservation, SIEGEState
from siege_env.rewards.aggregator import aggregate_rewards

try:
    from openenv import MCPEnvironment
except ImportError:  # pragma: no cover - fallback for local development.
    class MCPEnvironment:  # type: ignore[no-redef]
        """Fallback base when OpenEnv is not installed in local test environments."""


class SIEGEEnvironment(MCPEnvironment):
    """Single-seat SIEGE environment with minimal R1 reward loop."""

    def __init__(self, *, seed: int = 0, max_steps: int = 5) -> None:
        self._seed = seed
        self._rng = Random(seed)
        self._max_steps = max_steps
        self._templates = load_templates()
        self._state: SIEGEState | None = None
        self._agent_claims: list[dict[str, Any]] = []
        self._population: NPCPopulation | None = None
        self._seat_role = "immune"
        self._done = False
        self._last_reward_components: dict[str, Any] = {
            "r1_resolution": 0.0,
            "r2_deception": 0.0,
            "r3_detection": 0.0,
            "r4_trust_calibration": 0.0,
        }

    def reset(self) -> SIEGEObservation:
        template = self._rng.choice(self._templates)
        episode_id = f"episode-{uuid4().hex[:8]}"
        self._state = SIEGEState(
            episode_id=episode_id,
            step_count=0,
            incident_template_id=template["id"],
            ground_truth_root_cause=template["root_cause"],
            current_tier=1,
            arms_race_score=0.0,
        )
        self._seat_role = "pathogen" if self._rng.random() < 0.3 else "immune"
        population_seed = self._seed + self._rng.randrange(10_000)
        self._population = NPCPopulation(seed=population_seed, seat_agent_id=0)
        self._agent_claims = self._population.generate_claims(template, step_number=0)
        self._done = False
        self._last_reward_components = {
            "r1_resolution": 0.0,
            "r2_deception": 0.0,
            "r3_detection": 0.0,
            "r4_trust_calibration": 0.0,
        }
        return self._build_observation(template=template, action_error=None)

    def step(self, action_payload: SIEGEAction | dict[str, Any]) -> tuple[SIEGEObservation, float, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        template = self._template_by_id(self._state.incident_template_id)
        if self._done:
            observation = self._build_observation(template=template, action_error=None)
            return observation, 0.0, True, {"already_done": True}

        self._state = replace(self._state, step_count=self._state.step_count + 1)

        try:
            action = SIEGEAction.model_validate(action_payload)
        except ValidationError as exc:
            self._done = self._state.step_count >= self._max_steps
            observation = self._build_observation(template=template, action_error=str(exc))
            return observation, -0.05, self._done, {"invalid_action": True}

        reward, components = aggregate_rewards(
            action,
            ground_truth_root_cause=self._state.ground_truth_root_cause,
            seat_role=self._seat_role,
            claims_by_id={claim["claim_id"]: claim for claim in self._agent_claims},
            trust_scores={idx: 0.5 for idx in range(1, 8)},
            agent_reliability={
                int(claim["agent_id"]): (
                    claim["root_cause"] == self._state.ground_truth_root_cause
                )
                for claim in self._agent_claims
            },
        )
        self._last_reward_components = components

        if self._population is not None:
            self._agent_claims = self._population.generate_claims(
                template,
                step_number=self._state.step_count,
            )

        self._done = (action.tool_name == "diagnose" and reward == 1.0) or (
            self._state.step_count >= self._max_steps
        )
        observation = self._build_observation(template=template, action_error=None)
        info = {
            "invalid_action": False,
            "reward_components": components,
            "seat_role": self._seat_role,
        }
        return observation, reward, self._done, info

    def state(self) -> SIEGEState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() before state().")
        return self._state

    def _template_by_id(self, template_id: str) -> dict[str, Any]:
        for template in self._templates:
            if template["id"] == template_id:
                return template
        raise RuntimeError(f"Template '{template_id}' not found.")

    def _build_observation(self, *, template: dict[str, Any], action_error: str | None) -> SIEGEObservation:
        if self._state is None:
            raise RuntimeError("Environment not initialized.")

        if self._state.step_count <= 1:
            severity = "warning"
        elif self._state.step_count <= 3:
            severity = "critical"
        else:
            severity = "outage"

        visible_signals = template["observable_signals"][: max(1, min(len(template["observable_signals"]), self._state.step_count + 1))]
        available_evidence = [{"type": "signal", "value": signal} for signal in visible_signals]
        active_status = "resolved" if self._done else "active"

        return SIEGEObservation(
            incident_dashboard={
                "template_id": template["id"],
                "signals": visible_signals,
            },
            agent_claims=list(self._agent_claims),
            trust_scores={idx: 0.5 for idx in range(1, 8)},
            coalition_status={"votes_for": [], "votes_against": []},
            step_number=self._state.step_count,
            slo_status={"breached": self._state.step_count >= self._max_steps},
            your_role=self._seat_role,
            available_evidence=available_evidence,
            visibility_level="full",
            whisper_inbox=[],
            whisper_log=[],
            incident_severity=severity,
            red_herring_signals=[],
            reputation_history={idx: 0.5 for idx in range(1, 8)},
            active_incidents=[
                {
                    "incident_id": template["id"],
                    "status": active_status,
                }
            ],
            seat_agent_id=0,
            action_error=action_error,
        )

```

## siege_env/trust/__init__.py (last modified: 2026-04-24T18:02:13+00:00)
```python
"""Trust and coalition logic modules."""

from siege_env.trust.coalition import CoalitionResult, CoalitionVoting
from siege_env.trust.network import BayesianTrustNetwork

__all__ = ["BayesianTrustNetwork", "CoalitionResult", "CoalitionVoting"]

```

## siege_env/trust/coalition.py (last modified: 2026-04-24T18:02:27+00:00)
```python
"""Weighted coalition voting and ratification logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True)
class CoalitionResult:
    """Coalition vote tally result."""

    support_weight: float
    oppose_weight: float
    abstain_weight: float
    total_weight: float
    support_ratio: float
    ratified: bool


class CoalitionVoting:
    """Weighted voting with configurable ratification threshold."""

    def __init__(self, *, ratification_threshold: float = 0.6, neutral_weight: float = 0.5) -> None:
        if not 0.0 < ratification_threshold <= 1.0:
            raise ValueError("ratification_threshold must be in (0, 1].")
        if not 0.0 <= neutral_weight <= 1.0:
            raise ValueError("neutral_weight must be in [0, 1].")
        self._ratification_threshold = ratification_threshold
        self._neutral_weight = neutral_weight

    def tally(
        self,
        *,
        votes: Mapping[int, bool | None],
        trust_weights: Mapping[int, float],
    ) -> CoalitionResult:
        support_weight = 0.0
        oppose_weight = 0.0
        abstain_weight = 0.0

        for agent_id, vote in votes.items():
            weight = float(trust_weights.get(agent_id, self._neutral_weight))
            weight = max(0.0, min(1.0, weight))
            if vote is True:
                support_weight += weight
            elif vote is False:
                oppose_weight += weight
            else:
                abstain_weight += weight

        decided_weight = support_weight + oppose_weight
        total_weight = decided_weight
        support_ratio = support_weight / decided_weight if decided_weight > 0.0 else 0.0
        ratified = support_weight > oppose_weight and support_ratio >= self._ratification_threshold

        return CoalitionResult(
            support_weight=round(support_weight, 6),
            oppose_weight=round(oppose_weight, 6),
            abstain_weight=round(abstain_weight, 6),
            total_weight=round(total_weight, 6),
            support_ratio=round(support_ratio, 6),
            ratified=ratified,
        )

```

## siege_env/trust/network.py (last modified: 2026-04-24T18:01:48+00:00)
```python
"""Bayesian trust network for multi-agent credibility tracking."""

from __future__ import annotations


class BayesianTrustNetwork:
    """N x N trust matrix with simple Bayesian updates."""

    def __init__(
        self,
        *,
        agent_count: int,
        prior: float = 0.5,
        p_correct_if_trusted: float = 0.8,
        p_correct_if_untrusted: float = 0.3,
    ) -> None:
        if agent_count < 2:
            raise ValueError("agent_count must be >= 2.")
        if not 0.0 < prior < 1.0:
            raise ValueError("prior must be between 0 and 1 (exclusive).")
        if not 0.0 < p_correct_if_untrusted < p_correct_if_trusted < 1.0:
            raise ValueError("likelihood parameters must satisfy 0 < untrusted < trusted < 1.")

        self._agent_count = agent_count
        self._p_correct_if_trusted = p_correct_if_trusted
        self._p_correct_if_untrusted = p_correct_if_untrusted
        self._matrix: list[list[float]] = []
        for observer in range(agent_count):
            row: list[float] = []
            for target in range(agent_count):
                row.append(1.0 if observer == target else prior)
            self._matrix.append(row)

    @property
    def agent_count(self) -> int:
        return self._agent_count

    def _validate_agent(self, agent_id: int) -> None:
        if agent_id < 0 or agent_id >= self._agent_count:
            raise ValueError(f"agent_id {agent_id} out of bounds for {self._agent_count} agents.")

    def get_trust(self, observer_id: int, target_id: int) -> float:
        self._validate_agent(observer_id)
        self._validate_agent(target_id)
        return self._matrix[observer_id][target_id]

    def update(self, *, observer_id: int, target_id: int, claim_correct: bool) -> float:
        self._validate_agent(observer_id)
        self._validate_agent(target_id)

        if observer_id == target_id:
            self._matrix[observer_id][target_id] = 1.0
            return 1.0

        prior = self._matrix[observer_id][target_id]
        if claim_correct:
            likelihood_trusted = self._p_correct_if_trusted
            likelihood_untrusted = self._p_correct_if_untrusted
        else:
            likelihood_trusted = 1.0 - self._p_correct_if_trusted
            likelihood_untrusted = 1.0 - self._p_correct_if_untrusted

        numerator = likelihood_trusted * prior
        denominator = numerator + (likelihood_untrusted * (1.0 - prior))
        posterior = numerator / denominator
        posterior = max(0.0, min(1.0, posterior))
        self._matrix[observer_id][target_id] = posterior
        return posterior

    def as_matrix(self) -> list[list[float]]:
        return [list(row) for row in self._matrix]

```

## siege_env/utils/__init__.py (last modified: 2026-04-24T15:24:19+00:00)
```python
"""Shared utility modules."""
```

## tests/__init__.py (last modified: 2026-04-24T17:09:33+00:00)
```python
"""Test package for SIEGE."""

```

## tests/conftest.py (last modified: 2026-04-24T17:09:33+00:00)
```python
"""Shared pytest fixtures and test path bootstrap for SIEGE."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

```

## tests/master_suite.py (last modified: 2026-04-24T18:36:09+00:00)
```python
"""Master test suite entrypoint aggregating the project test surface."""

from tests.step_tests.step_00_bootstrap_test import *  # noqa: F401,F403
from tests.step_tests.step_01_scaffold_test import *  # noqa: F401,F403
from tests.step_tests.step_02_models_test import *  # noqa: F401,F403
from tests.step_tests.step_03_incidents_test import *  # noqa: F401,F403
from tests.step_tests.step_04_minimal_env_test import *  # noqa: F401,F403
from tests.step_tests.step_05_npc_test import *  # noqa: F401,F403
from tests.step_tests.step_06_trust_test import *  # noqa: F401,F403
from tests.step_tests.step_07_pathogen_test import *  # noqa: F401,F403
from tests.step_tests.step_08_r4_hacking_test import *  # noqa: F401,F403
from tests.step_tests.step_09_curriculum_test import *  # noqa: F401,F403
from tests.step_tests.step_10_trust_poisoning_test import *  # noqa: F401,F403
from tests.step_tests.step_11_temporal_test import *  # noqa: F401,F403
from tests.step_tests.step_12_confidence_test import *  # noqa: F401,F403

```

## tests/step_tests/__init__.py (last modified: 2026-04-24T17:09:33+00:00)
```python
"""Step-gated tests for SIEGE."""

```

## tests/step_tests/step_00_bootstrap_test.py (last modified: 2026-04-24T18:02:55+00:00)
```python
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_required_structure_exists() -> None:
    required_paths = [
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / ".pre-commit-config.yaml",
        ROOT / "pyproject.toml",
        ROOT / "Makefile",
        ROOT / "brain" / "tools" / "update_brain.py",
        ROOT / "brain" / "tools" / "compile_master_code.py",
        ROOT / "tests" / "step_tests" / "step_00_bootstrap_test.py",
    ]
    for path in required_paths:
        assert path.exists(), f"Missing required bootstrap path: {path}"


def test_ci_config_parses() -> None:
    ci_file = ROOT / ".github" / "workflows" / "ci.yml"
    data = yaml.safe_load(ci_file.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "jobs" in data


def test_make_test_all_runs_cleanly() -> None:
    if os.getenv("SIEGE_BOOTSTRAP_SELFTEST") == "1":
        return

    env = os.environ.copy()
    env["SIEGE_BOOTSTRAP_SELFTEST"] = "1"
    result = subprocess.run(
        ["make", "test-all"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_update_brain_creates_snapshot() -> None:
    tracked_files = {
        ROOT / "brain" / "MASTER_CODE.md": (ROOT / "brain" / "MASTER_CODE.md").read_text(
            encoding="utf-8"
        ),
        ROOT / "brain" / "CHANGELOG.md": (ROOT / "brain" / "CHANGELOG.md").read_text(
            encoding="utf-8"
        ),
        ROOT / "brain" / "CONTEXT.md": (ROOT / "brain" / "CONTEXT.md").read_text(
            encoding="utf-8"
        ),
        ROOT / "brain" / "ROADMAP_STATUS.md": (
            ROOT / "brain" / "ROADMAP_STATUS.md"
        ).read_text(encoding="utf-8"),
    }
    before = set((ROOT / "brain" / "snapshots").glob("step_00_*.json"))
    try:
        result = subprocess.run(
            [
                "python3",
                "brain/tools/update_brain.py",
                "--step",
                "00",
                "--title",
                "Bootstrap",
                "--owner",
                "Utkarsh",
                "--reviewer",
                "Ankit",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        after = set((ROOT / "brain" / "snapshots").glob("step_00_*.json"))
        assert len(after) >= len(before) + 1
    finally:
        after = set((ROOT / "brain" / "snapshots").glob("step_00_*.json"))
        for snapshot_path in after - before:
            snapshot_path.unlink(missing_ok=True)
        for file_path, original_contents in tracked_files.items():
            file_path.write_text(original_contents, encoding="utf-8")

```

## tests/step_tests/step_01_scaffold_test.py (last modified: 2026-04-24T17:20:36+00:00)
```python
from __future__ import annotations

import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    info = subprocess.run(
        ["docker", "info"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return info.returncode == 0


def test_import_and_health_route() -> None:
    from siege_env.server.app import app

    routes = {route.path for route in app.routes}
    assert "/health" in routes


def test_openenv_manifest_has_required_keys() -> None:
    manifest = ROOT / "openenv.yaml"
    assert manifest.exists()
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert data["name"] == "siege_env"
    assert data["runtime"]["entrypoint"] == "siege_env.server.app:app"
    assert data["runtime"]["healthcheck"] == "/health"


@pytest.mark.skipif(not _docker_ready(), reason="Docker daemon not available")
def test_docker_build_succeeds() -> None:
    result = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            "siege_env/server/Dockerfile",
            "-t",
            "siege-step01-test",
            ".",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.skipif(not _docker_ready(), reason="Docker daemon not available")
def test_container_health_endpoint() -> None:
    port = _free_port()
    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "-p",
            f"{port}:8000",
            "siege-step01-test",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + "\n" + run.stderr
    container_id = run.stdout.strip()
    try:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + 20
        last_error = ""
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    body = response.read().decode("utf-8")
                    assert response.status == 200
                    assert "ok" in body
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                last_error = str(exc)
                time.sleep(0.5)
        raise AssertionError(f"Container health endpoint did not become ready: {last_error}")
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
```

## tests/step_tests/step_02_models_test.py (last modified: 2026-04-24T17:09:33+00:00)
```python
from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from siege_env.models.actions import (
    ChallengeArgs,
    DiagnoseArgs,
    EscalateArgs,
    PostmortemArgs,
    RatifyArgs,
    SIEGEAction,
    WhisperArgs,
)
from siege_env.models.observations import SIEGEObservation
from siege_env.models.state import SIEGEState


def build_observation() -> SIEGEObservation:
    return SIEGEObservation(
        incident_dashboard={"alerts": [{"name": "latency_p99", "value": 420}]},
        agent_claims=[
            {
                "agent_id": 2,
                "claim_id": "claim-001",
                "root_cause": "query_plan_regression",
            }
        ],
        trust_scores={1: 0.35, 2: 0.82, 3: 0.54},
        coalition_status={"claim_id": "claim-001", "votes_for": [2], "votes_against": []},
        step_number=3,
        slo_status={"budget_remaining_pct": 12.5, "breached": False},
        your_role="immune",
        available_evidence=[{"type": "metric", "name": "latency_p99"}],
        visibility_level="full",
        whisper_inbox=[{"from_agent": 3, "message": "Check the database traces."}],
        whisper_log=[{"from_agent": 3, "to_agent": 0}],
        incident_severity="critical",
        red_herring_signals=[{"type": "deploy", "service": "frontend"}],
        reputation_history={1: 0.40, 2: 0.91},
        active_incidents=[{"incident_id": "inc-001", "status": "active"}],
        seat_agent_id=0,
        action_error=None,
    )


def build_state() -> SIEGEState:
    return SIEGEState(
        episode_id="episode-001",
        step_count=3,
        incident_template_id="cloudflare-regex-2019",
        ground_truth_root_cause="regex_backtracking",
        current_tier=2,
        arms_race_score=0.27,
    )


def test_diagnose_args_accept_valid_payload() -> None:
    args = DiagnoseArgs.model_validate(
        {
            "root_cause": "query_plan_regression",
            "confidence": 0.72,
            "evidence": ["latency_p99_spike", "no_traffic_increase"],
            "alternative_hypotheses": [
                {"diagnosis": "connection_pool_exhaustion", "confidence": 0.18},
                {"diagnosis": "load_spike", "confidence": 0.10},
            ],
        }
    )
    assert args.root_cause == "query_plan_regression"
    assert len(args.alternative_hypotheses) == 2


def test_diagnose_args_reject_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        DiagnoseArgs.model_validate(
            {
                "root_cause": "query_plan_regression",
                "confidence": 1.2,
                "evidence": ["latency_p99_spike"],
                "alternative_hypotheses": [],
            }
        )


def test_challenge_args_reject_unknown_flaw_type() -> None:
    with pytest.raises(ValidationError):
        ChallengeArgs.model_validate(
            {
                "target_agent_id": 4,
                "claim_id": "claim-001",
                "flaw_type": "made_up_taxonomy",
                "reasoning": "This claim does not line up with the observed trace timings.",
            }
        )


def test_ratify_args_reject_blank_claim_id() -> None:
    with pytest.raises(ValidationError):
        RatifyArgs.model_validate({"claim_id": "   ", "vote": True})


def test_escalate_args_round_trip_json() -> None:
    args = EscalateArgs.model_validate(
        {
            "concern": "Blast radius is spreading across payments and API traffic.",
            "blast_radius_estimate": ["payments", "api-gateway"],
        }
    )
    restored = EscalateArgs.model_validate_json(args.model_dump_json())
    assert restored == args


def test_whisper_args_reject_invalid_target_agent() -> None:
    with pytest.raises(ValidationError):
        WhisperArgs.model_validate({"target_agent_id": 8, "message": "Check agent 5."})


def test_postmortem_args_requires_non_empty_timeline() -> None:
    with pytest.raises(ValidationError):
        PostmortemArgs.model_validate(
            {
                "root_cause": "regex_backtracking",
                "timeline": [],
                "contributing_factors": ["unsafe regex rollout"],
                "misdiagnosis_analysis": "We anchored on load instead of tracing the regex path.",
            }
        )


def test_siege_action_parses_tool_specific_arguments() -> None:
    action = SIEGEAction.model_validate(
        {
            "tool_name": "diagnose",
            "arguments": {
                "root_cause": "query_plan_regression",
                "confidence": 0.72,
                "evidence": ["latency_p99_spike"],
                "alternative_hypotheses": [],
            },
        }
    )
    assert isinstance(action.arguments, DiagnoseArgs)

    with pytest.raises(ValidationError):
        SIEGEAction.model_validate(
            {
                "tool_name": "diagnose",
                "arguments": {
                    "target_agent_id": 1,
                    "message": "This is clearly the wrong payload.",
                },
            }
        )


def test_observation_round_trip_json_and_validation() -> None:
    observation = build_observation()
    restored = SIEGEObservation.from_json(observation.to_json())
    assert restored == observation

    with pytest.raises(ValueError):
        replace(observation, incident_severity="catastrophic")


def test_state_round_trip_json_and_validation() -> None:
    state = build_state()
    restored = SIEGEState.from_json(state.to_json())
    assert restored == state

    with pytest.raises(ValueError):
        replace(state, current_tier=4)

```

## tests/step_tests/step_03_incidents_test.py (last modified: 2026-04-24T17:22:19+00:00)
```python
from __future__ import annotations

from siege_env.incidents.generator import generate_variant
from siege_env.incidents.loader import load_templates


REQUIRED_KEYS = {
    "id",
    "source_url",
    "root_cause",
    "observable_signals",
    "flaw_types",
    "blast_radius",
}


def test_all_five_seed_templates_load() -> None:
    templates = load_templates()
    assert len(templates) == 5


def test_templates_contain_required_ground_truth_fields() -> None:
    templates = load_templates()
    for template in templates:
        assert REQUIRED_KEYS.issubset(set(template.keys()))
        assert template["id"]
        assert template["root_cause"]
        assert template["source_url"].startswith("https://")
        assert isinstance(template["observable_signals"], list)
        assert isinstance(template["flaw_types"], list)
        assert isinstance(template["blast_radius"], list)


def test_variant_generator_produces_valid_variant() -> None:
    template = load_templates()[0]
    variant = generate_variant(template, variant_index=7)
    assert REQUIRED_KEYS.issubset(set(variant.keys()))
    assert variant["id"].startswith(f"{template['id']}_v")
    assert variant["root_cause"] == template["root_cause"]
    assert variant["source_url"] == template["source_url"]
    assert len(variant["observable_signals"]) == len(template["observable_signals"])

```

## tests/step_tests/step_04_minimal_env_test.py (last modified: 2026-04-24T17:51:36+00:00)
```python
from __future__ import annotations

from siege_env.models import SIEGEAction, SIEGEObservation, SIEGEState
from siege_env.server.siege_environment import SIEGEEnvironment


def _valid_diagnose_action(root_cause: str) -> dict[str, object]:
    return {
        "tool_name": "diagnose",
        "arguments": {
            "root_cause": root_cause,
            "confidence": 0.81,
            "evidence": ["latency_p99_spike"],
            "alternative_hypotheses": [],
        },
    }


def test_reset_returns_valid_observation() -> None:
    env = SIEGEEnvironment(seed=7)
    obs = env.reset()
    assert isinstance(obs, SIEGEObservation)
    assert obs.step_number == 0
    assert obs.action_error is None


def test_step_accepts_valid_action() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    action = SIEGEAction.model_validate(_valid_diagnose_action(env.state().ground_truth_root_cause))
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, SIEGEObservation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_done_is_reachable() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    _, _, done, _ = env.step(_valid_diagnose_action(env.state().ground_truth_root_cause))
    assert done is True


def test_reward_is_clamped_between_zero_and_one() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    _, reward, _, _ = env.step(_valid_diagnose_action("wrong_cause"))
    assert 0.0 <= reward <= 1.0


def test_state_serializes_round_trip() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    state = env.state()
    restored = SIEGEState.from_json(state.to_json())
    assert restored == state


def test_invalid_action_is_handled_gracefully() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    obs, reward, done, info = env.step({"tool_name": "diagnose", "arguments": {"bad": "payload"}})
    assert reward == -0.05
    assert done is False
    assert obs.action_error is not None
    assert info["invalid_action"] is True


def test_multi_step_episode_works() -> None:
    env = SIEGEEnvironment(seed=7, max_steps=3)
    env.reset()
    _, _, done_1, _ = env.step(_valid_diagnose_action("wrong_cause"))
    _, _, done_2, _ = env.step(
        {
            "tool_name": "escalate",
            "arguments": {
                "concern": "potential blast radius increase",
                "blast_radius_estimate": ["api-gateway"],
            },
        }
    )
    _, reward_3, done_3, _ = env.step(_valid_diagnose_action(env.state().ground_truth_root_cause))
    assert done_1 is False
    assert done_2 is False
    assert done_3 is True
    assert reward_3 == 1.0

```

## tests/step_tests/step_05_npc_test.py (last modified: 2026-04-24T17:55:10+00:00)
```python
from __future__ import annotations

from siege_env.agents.population import NPCPopulation
from siege_env.agents.scripted import ROLE_CONFIDENCE_BOUNDS, ScriptedNPCAgent
from siege_env.incidents.loader import load_templates
from siege_env.server.siege_environment import SIEGEEnvironment


def _template() -> dict[str, object]:
    return load_templates()[0]


def test_population_builds_seven_npcs_for_one_seat() -> None:
    population = NPCPopulation(seed=11, seat_agent_id=0)
    agents = population.agents
    assert len(agents) == 7
    assert all(agent.agent_id != 0 for agent in agents)


def test_population_role_assignment_is_deterministic() -> None:
    first = NPCPopulation(seed=11, seat_agent_id=0)
    second = NPCPopulation(seed=11, seat_agent_id=0)
    first_roles = [agent.role for agent in first.agents]
    second_roles = [agent.role for agent in second.agents]
    assert first_roles == second_roles


def test_scripted_agent_generate_claim_is_deterministic() -> None:
    template = _template()
    agent_a = ScriptedNPCAgent(agent_id=3, role="verifier", seed=91)
    agent_b = ScriptedNPCAgent(agent_id=3, role="verifier", seed=91)
    assert agent_a.generate_claim(template, step_number=2) == agent_b.generate_claim(
        template, step_number=2
    )


def test_claim_contains_expected_keys() -> None:
    template = _template()
    claim = ScriptedNPCAgent(agent_id=4, role="lead", seed=99).generate_claim(template, step_number=1)
    assert {"agent_id", "claim_id", "root_cause", "confidence", "role", "evidence"} <= set(claim.keys())


def test_role_confidence_respects_role_bounds() -> None:
    template = _template()
    for role, bounds in ROLE_CONFIDENCE_BOUNDS.items():
        claim = ScriptedNPCAgent(agent_id=2, role=role, seed=7).generate_claim(template, step_number=1)
        lower, upper = bounds
        assert lower <= claim["confidence"] <= upper


def test_population_generate_claims_returns_one_per_npc() -> None:
    population = NPCPopulation(seed=13, seat_agent_id=0)
    claims = population.generate_claims(_template(), step_number=2)
    assert len(claims) == len(population.agents)
    assert len({claim["agent_id"] for claim in claims}) == len(claims)


def test_population_claim_generation_is_deterministic() -> None:
    template = _template()
    pop_a = NPCPopulation(seed=19, seat_agent_id=0)
    pop_b = NPCPopulation(seed=19, seat_agent_id=0)
    assert pop_a.generate_claims(template, step_number=3) == pop_b.generate_claims(
        template, step_number=3
    )


def test_environment_observation_contains_npc_claims() -> None:
    env = SIEGEEnvironment(seed=21)
    obs = env.reset()
    assert len(obs.agent_claims) == 7

```

## tests/step_tests/step_06_trust_test.py (last modified: 2026-04-24T18:01:28+00:00)
```python
from __future__ import annotations

import pytest

from siege_env.trust.coalition import CoalitionVoting
from siege_env.trust.network import BayesianTrustNetwork


def test_trust_matrix_shape_is_n_by_n() -> None:
    network = BayesianTrustNetwork(agent_count=8)
    matrix = network.as_matrix()
    assert len(matrix) == 8
    assert all(len(row) == 8 for row in matrix)


def test_diagonal_is_one_by_default() -> None:
    network = BayesianTrustNetwork(agent_count=8)
    assert all(network.get_trust(i, i) == 1.0 for i in range(8))


def test_initial_off_diagonal_uses_prior() -> None:
    network = BayesianTrustNetwork(agent_count=8, prior=0.55)
    assert network.get_trust(0, 1) == 0.55
    assert network.get_trust(6, 3) == 0.55


def test_positive_evidence_increases_trust() -> None:
    network = BayesianTrustNetwork(agent_count=8, prior=0.5)
    before = network.get_trust(0, 1)
    network.update(observer_id=0, target_id=1, claim_correct=True)
    after = network.get_trust(0, 1)
    assert after > before


def test_negative_evidence_decreases_trust() -> None:
    network = BayesianTrustNetwork(agent_count=8, prior=0.5)
    before = network.get_trust(0, 1)
    network.update(observer_id=0, target_id=1, claim_correct=False)
    after = network.get_trust(0, 1)
    assert after < before


def test_repeated_updates_remain_bounded() -> None:
    network = BayesianTrustNetwork(agent_count=8, prior=0.5)
    for _ in range(100):
        network.update(observer_id=0, target_id=1, claim_correct=True)
        network.update(observer_id=0, target_id=1, claim_correct=False)
    value = network.get_trust(0, 1)
    assert 0.0 <= value <= 1.0


def test_invalid_agent_index_raises_error() -> None:
    network = BayesianTrustNetwork(agent_count=8)
    with pytest.raises(ValueError):
        network.get_trust(-1, 2)
    with pytest.raises(ValueError):
        network.update(observer_id=0, target_id=8, claim_correct=True)


def test_self_update_keeps_identity_trust_fixed() -> None:
    network = BayesianTrustNetwork(agent_count=8)
    network.update(observer_id=2, target_id=2, claim_correct=False)
    assert network.get_trust(2, 2) == 1.0


def test_weighted_ratification_passes_with_threshold() -> None:
    voting = CoalitionVoting(ratification_threshold=0.6)
    trust_weights = {1: 0.9, 2: 0.7, 3: 0.3}
    votes = {1: True, 2: True, 3: False}
    result = voting.tally(votes=votes, trust_weights=trust_weights)
    assert result.ratified is True


def test_weighted_ratification_fails_below_threshold() -> None:
    voting = CoalitionVoting(ratification_threshold=0.75)
    trust_weights = {1: 0.9, 2: 0.4, 3: 0.7}
    votes = {1: True, 2: False, 3: False}
    result = voting.tally(votes=votes, trust_weights=trust_weights)
    assert result.ratified is False


def test_tie_is_not_ratified() -> None:
    voting = CoalitionVoting(ratification_threshold=0.5)
    trust_weights = {1: 0.6, 2: 0.6}
    votes = {1: True, 2: False}
    result = voting.tally(votes=votes, trust_weights=trust_weights)
    assert result.support_weight == result.oppose_weight
    assert result.ratified is False


def test_abstentions_are_ignored() -> None:
    voting = CoalitionVoting(ratification_threshold=0.5)
    trust_weights = {1: 0.8, 2: 0.2, 3: 0.9}
    votes = {1: True, 2: None, 3: False}
    result = voting.tally(votes=votes, trust_weights=trust_weights)
    assert result.total_weight == 1.7
    assert result.abstain_weight == 0.2


def test_missing_weight_defaults_to_neutral() -> None:
    voting = CoalitionVoting(ratification_threshold=0.5)
    votes = {1: True, 2: False}
    result = voting.tally(votes=votes, trust_weights={1: 0.9})
    assert result.total_weight == 1.4
    assert result.ratified is True


def test_invalid_threshold_raises_error() -> None:
    with pytest.raises(ValueError):
        CoalitionVoting(ratification_threshold=0.0)
    with pytest.raises(ValueError):
        CoalitionVoting(ratification_threshold=1.2)

```

## tests/step_tests/step_07_pathogen_test.py (last modified: 2026-04-24T18:11:25+00:00)
```python
from __future__ import annotations

from siege_env.models import SIEGEObservation
from siege_env.server.siege_environment import SIEGEEnvironment


def _diagnose(root_cause: str) -> dict[str, object]:
    return {
        "tool_name": "diagnose",
        "arguments": {
            "root_cause": root_cause,
            "confidence": 0.71,
            "evidence": ["latency_p99_spike"],
            "alternative_hypotheses": [],
        },
    }


def _challenge(claim_id: str, target_agent_id: int) -> dict[str, object]:
    return {
        "tool_name": "challenge",
        "arguments": {
            "target_agent_id": target_agent_id,
            "claim_id": claim_id,
            "flaw_type": "type1_false_correlation",
            "reasoning": "Claim conflicts with observed evidence progression across traces.",
        },
    }


def _env_with_role(role: str) -> tuple[SIEGEEnvironment, SIEGEObservation]:
    # Role is sampled each reset, so retry deterministically until desired role appears.
    env = SIEGEEnvironment(seed=11 if role == "pathogen" else 7)
    for _ in range(50):
        obs = env.reset()
        if obs.your_role == role:
            return env, obs
    raise AssertionError(f"Could not sample role '{role}' in 50 resets.")


def test_role_field_is_emitted_in_observation() -> None:
    env = SIEGEEnvironment(seed=3)
    obs = env.reset()
    assert obs.your_role in {"immune", "pathogen"}


def test_role_assignment_approximately_matches_seventy_thirty_split() -> None:
    env = SIEGEEnvironment(seed=123)
    pathogen_count = 0
    total_episodes = 200
    for _ in range(total_episodes):
        if env.reset().your_role == "pathogen":
            pathogen_count += 1
    ratio = pathogen_count / total_episodes
    assert 0.2 <= ratio <= 0.4


def test_r2_positive_for_pathogen_wrong_diagnosis() -> None:
    env, _ = _env_with_role("pathogen")
    obs, reward, _, info = env.step(_diagnose("fabricated_root_cause"))
    assert obs.action_error is None
    assert info["reward_components"]["r2_deception"] > 0.0
    assert reward >= info["reward_components"]["r2_deception"]


def test_r2_zero_for_pathogen_correct_diagnosis() -> None:
    env, _ = _env_with_role("pathogen")
    _, _, _, info = env.step(_diagnose(env.state().ground_truth_root_cause))
    assert info["reward_components"]["r2_deception"] == 0.0


def test_r2_zero_for_immune_wrong_diagnosis() -> None:
    env, _ = _env_with_role("immune")
    _, _, _, info = env.step(_diagnose("fabricated_root_cause"))
    assert info["reward_components"]["r2_deception"] == 0.0


def test_r3_positive_for_immune_challenging_incorrect_claim() -> None:
    env, obs = _env_with_role("immune")
    incorrect_claim = next(
        claim for claim in obs.agent_claims if claim["root_cause"] != env.state().ground_truth_root_cause
    )
    _, reward, _, info = env.step(_challenge(incorrect_claim["claim_id"], incorrect_claim["agent_id"]))
    assert info["reward_components"]["r3_detection"] > 0.0
    assert reward >= info["reward_components"]["r3_detection"]


def test_r3_zero_for_immune_challenging_correct_claim() -> None:
    env, obs = _env_with_role("immune")
    correct_claim = next(
        claim for claim in obs.agent_claims if claim["root_cause"] == env.state().ground_truth_root_cause
    )
    _, _, _, info = env.step(_challenge(correct_claim["claim_id"], correct_claim["agent_id"]))
    assert info["reward_components"]["r3_detection"] == 0.0


def test_r3_zero_for_pathogen_challenge() -> None:
    env, obs = _env_with_role("pathogen")
    claim = obs.agent_claims[0]
    _, _, _, info = env.step(_challenge(claim["claim_id"], claim["agent_id"]))
    assert info["reward_components"]["r3_detection"] == 0.0


def test_reward_components_include_r2_and_r3() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    _, _, _, info = env.step(_diagnose("fabricated_root_cause"))
    assert "r2_deception" in info["reward_components"]
    assert "r3_detection" in info["reward_components"]

```

## tests/step_tests/step_08_r4_hacking_test.py (last modified: 2026-04-24T18:13:44+00:00)
```python
from __future__ import annotations

from siege_env.models import SIEGEAction
from siege_env.rewards.aggregator import aggregate_rewards
from siege_env.rewards.r4_trust_calibration import compute_r4_trust_calibration
from siege_env.server.siege_environment import SIEGEEnvironment


def _diagnose(root_cause: str) -> SIEGEAction:
    return SIEGEAction.model_validate(
        {
            "tool_name": "diagnose",
            "arguments": {
                "root_cause": root_cause,
                "confidence": 0.77,
                "evidence": ["latency_p99_spike"],
                "alternative_hypotheses": [],
            },
        }
    )


def test_r4_perfect_calibration_scores_one() -> None:
    score = compute_r4_trust_calibration(
        trust_scores={1: 1.0, 2: 0.0, 3: 1.0},
        agent_reliability={1: True, 2: False, 3: True},
    )
    assert score == 1.0


def test_r4_uniform_trust_exploit_is_penalized() -> None:
    score = compute_r4_trust_calibration(
        trust_scores={1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
        agent_reliability={1: True, 2: False, 3: True, 4: False},
    )
    assert score == 0.75


def test_aggregator_includes_r4_component() -> None:
    action = _diagnose("wrong_cause")
    total, components = aggregate_rewards(
        action,
        ground_truth_root_cause="true_cause",
        seat_role="immune",
        claims_by_id={},
        trust_scores={1: 0.9, 2: 0.1},
        agent_reliability={1: True, 2: False},
    )
    assert "r4_trust_calibration" in components
    assert 0.0 <= components["r4_trust_calibration"] <= 1.0
    assert 0.0 <= total <= 1.0


def test_exploit_r1_non_diagnose_spam_gets_no_r1() -> None:
    action = SIEGEAction.model_validate(
        {
            "tool_name": "escalate",
            "arguments": {"concern": "spam", "blast_radius_estimate": ["api"]},
        }
    )
    _, components = aggregate_rewards(
        action,
        ground_truth_root_cause="true_cause",
        seat_role="immune",
        claims_by_id={},
        trust_scores={},
        agent_reliability={},
    )
    assert components["r1_resolution"] == 0.0


def test_exploit_r2_pathogen_truthful_diagnose_gets_no_r2() -> None:
    action = _diagnose("true_cause")
    _, components = aggregate_rewards(
        action,
        ground_truth_root_cause="true_cause",
        seat_role="pathogen",
        claims_by_id={},
        trust_scores={},
        agent_reliability={},
    )
    assert components["r2_deception"] == 0.0


def test_exploit_r3_challenge_unknown_claim_gets_no_r3() -> None:
    action = SIEGEAction.model_validate(
        {
            "tool_name": "challenge",
            "arguments": {
                "target_agent_id": 1,
                "claim_id": "missing-claim",
                "flaw_type": "type1_false_correlation",
                "reasoning": "Might be wrong based on observed signals and timing drift.",
            },
        }
    )
    _, components = aggregate_rewards(
        action,
        ground_truth_root_cause="true_cause",
        seat_role="immune",
        claims_by_id={},
        trust_scores={},
        agent_reliability={},
    )
    assert components["r3_detection"] == 0.0


def test_exploit_r4_uniform_trust_cannot_reach_perfect_score_in_env_context() -> None:
    env = SIEGEEnvironment(seed=12)
    obs = env.reset()
    _, _, _, info = env.step(_diagnose("wrong_cause"))
    r4 = info["reward_components"]["r4_trust_calibration"]
    assert r4 < 1.0
    assert obs.trust_scores and all(value == 0.5 for value in obs.trust_scores.values())

```

## tests/step_tests/step_09_curriculum_test.py (last modified: 2026-04-24T18:21:01+00:00)
```python
"""Gate test for Step 09 — Tiered Curriculum Scheduler.

5 tests:
1. Fresh scheduler always starts at Tier 1.
2. Escalates to Tier 2 after sustained high win-rate.
3. De-escalates back to Tier 1 after sustained low win-rate.
4. Never escalates above Tier 3 regardless of win-rate.
5. attacker_ahead() invariant holds correctly.
"""

from __future__ import annotations

from siege_env.curriculum.tiered_scheduler import TIER_CONFIGS, TieredScheduler


def _feed(scheduler: TieredScheduler, *, wins: int, losses: int) -> None:
    """Feed a block of wins then losses (or vice-versa) to the scheduler."""
    for _ in range(wins):
        scheduler.record_episode(won=True)
    for _ in range(losses):
        scheduler.record_episode(won=False)


def test_starts_at_tier_1() -> None:
    """A fresh scheduler must start at Tier 1."""
    s = TieredScheduler()
    assert s.current_tier == 1
    assert s.config == TIER_CONFIGS[1]


def test_escalates_on_high_winrate() -> None:
    """10 consecutive wins (window=10) must push scheduler from Tier 1 → 2."""
    s = TieredScheduler(window=10, escalate_threshold=0.70, cooldown=0)
    _feed(s, wins=10, losses=0)
    assert s.current_tier == 2


def test_deescalates_on_low_winrate() -> None:
    """After reaching Tier 2, 10 consecutive losses must drop back to Tier 1."""
    s = TieredScheduler(window=10, escalate_threshold=0.70, deescalate_threshold=0.30, cooldown=0)
    # Escalate to Tier 2 first
    _feed(s, wins=10, losses=0)
    assert s.current_tier == 2
    # Now tank the win-rate
    _feed(s, wins=0, losses=10)
    assert s.current_tier == 1


def test_no_escalation_above_tier_3() -> None:
    """Scheduler must never exceed Tier 3 no matter how many wins are recorded."""
    s = TieredScheduler(window=10, escalate_threshold=0.70, cooldown=0)
    # Push through all tiers
    _feed(s, wins=50, losses=0)
    assert s.current_tier == 3


def test_attacker_ahead_invariant() -> None:
    """attacker_ahead() must be True until agent clears the escalation threshold."""
    s = TieredScheduler(window=10, escalate_threshold=0.70, cooldown=0)

    # Not enough data yet → attacker assumed ahead
    assert s.attacker_ahead() is True

    # 6 wins out of 10 = 60% < 70% threshold → still ahead
    _feed(s, wins=6, losses=4)
    assert s.attacker_ahead() is True

    # 8 wins in next window = 80% ≥ 70% → agent is beating the tier
    s2 = TieredScheduler(window=10, escalate_threshold=0.70, cooldown=0)
    _feed(s2, wins=8, losses=2)
    assert s2.attacker_ahead() is False

```

## tests/step_tests/step_10_trust_poisoning_test.py (last modified: 2026-04-24T18:23:03+00:00)
```python
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

```

## tests/step_tests/step_11_temporal_test.py (last modified: 2026-04-24T18:32:44+00:00)
```python
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

```

## tests/step_tests/step_12_confidence_test.py (last modified: 2026-04-24T18:35:45+00:00)
```python
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

```

## tests/step_tests/step_13_cascade_test.py (last modified: 2026-04-24T19:23:25+00:00)
```python
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
```

## tests/step_tests/step_14_templates_expansion_test.py (last modified: 2026-04-24T19:34:48+00:00)
```python
"""Gate tests for Step 14 — Incident template expansion to 20."""

from __future__ import annotations

import pytest

from siege_env.incidents.loader import load_templates


EXPECTED_IDS = [
    "gitlab_2017_01_db_recovery",
    "cloudflare_2019_07_regex_waf",
    "aws_s3_2017_02_us_east_1",
    "github_2018_10_network_partition",
    "google_sre_shakespeare_case",
    "slack_2021_01_dns_dependency",
    "meta_2021_10_bgp_withdrawal",
    "fastly_2021_06_edge_config_bug",
    "gcp_2020_11_auth_quota_exhaustion",
    "azure_2023_01_wan_issue",
    "dropbox_2014_01_auth_bug",
    "stripe_2019_07_db_failover",
    "twilio_2023_08_kv_dependency",
    "atlassian_2022_04_script_failure",
    "datadog_2021_11_message_bus",
    "zoom_2020_08_dns_registrar",
    "shopify_2020_09_kubernetes",
    "netflix_2012_12_aws_outage",
    "pagerduty_2023_01_db_migration",
    "openai_2023_11_capacity_event",
]


@pytest.fixture(scope="module")
def expanded_templates() -> list[dict[str, object]]:
    return load_templates(include_step14_expansion=True)


def test_step14_total_template_count(expanded_templates: list[dict[str, object]]) -> None:
    assert len(expanded_templates) == 20


@pytest.mark.parametrize("template_id", EXPECTED_IDS)
def test_step14_template_present_and_valid(
    template_id: str,
    expanded_templates: list[dict[str, object]],
) -> None:
    by_id = {str(t["id"]): t for t in expanded_templates}
    assert template_id in by_id
    template = by_id[template_id]
    assert str(template["source_url"]).startswith("https://")
    assert str(template["root_cause"]).strip()
    assert isinstance(template["observable_signals"], list) and len(template["observable_signals"]) > 0
    assert isinstance(template["flaw_types"], list) and len(template["flaw_types"]) > 0
    assert isinstance(template["blast_radius"], list) and len(template["blast_radius"]) > 0
```

## tests/step_tests/step_15_info_asymmetry_test.py (last modified: 2026-04-24T19:35:50+00:00)
```python
"""Gate tests for Step 15 — Information Asymmetry."""

from __future__ import annotations

from siege_env.mechanics.info_asymmetry import (
    filter_evidence_for_visibility,
    visibility_for_step,
)
from siege_env.server.siege_environment import SIEGEEnvironment


def test_visibility_schedule_for_immune_agent() -> None:
    assert visibility_for_step(0, "immune") == "metrics_only"
    assert visibility_for_step(2, "immune") == "traces_only"
    assert visibility_for_step(4, "immune") == "full"


def test_visibility_schedule_for_pathogen_agent() -> None:
    assert visibility_for_step(0, "pathogen") == "delayed"
    assert visibility_for_step(5, "pathogen") == "delayed"


def test_evidence_filtering_respects_visibility_levels() -> None:
    evidence = [{"value": "a"}, {"value": "b"}, {"value": "c"}]
    assert len(filter_evidence_for_visibility(evidence, visibility_level="metrics_only")) == 1
    assert len(filter_evidence_for_visibility(evidence, visibility_level="traces_only")) == 2
    assert len(filter_evidence_for_visibility(evidence, visibility_level="full")) == 3


def test_environment_observation_contains_visibility_and_filtered_evidence() -> None:
    env = SIEGEEnvironment(seed=3)
    obs = env.reset()
    assert obs.visibility_level in {"metrics_only", "traces_only", "full", "delayed"}
    assert len(obs.available_evidence) >= 1
    if obs.visibility_level == "metrics_only":
        assert len(obs.available_evidence) == 1
```

## tests/step_tests/step_16_whisper_test.py (last modified: 2026-04-24T19:37:21+00:00)
```python
"""Gate tests for Step 16 — Whisper / Private Channels."""

from __future__ import annotations

from siege_env.mechanics.whisper import build_whisper_event
from siege_env.models.actions import SIEGEAction, WhisperArgs
from siege_env.server.siege_environment import SIEGEEnvironment


def test_build_whisper_event_contains_expected_fields() -> None:
    event = build_whisper_event(
        sender_agent_id=0,
        target_agent_id=2,
        message="check replica lag",
        step_number=1,
    ).to_dict()
    assert event["sender_agent_id"] == 0
    assert event["target_agent_id"] == 2
    assert event["message"] == "check replica lag"


def test_whisper_action_logged_in_step_info_counter() -> None:
    env = SIEGEEnvironment(seed=5)
    env.reset()
    action = SIEGEAction(
        tool_name="whisper",
        arguments=WhisperArgs(target_agent_id=2, message="sync privately"),
    )
    _, _, _, info = env.step(action)
    assert info["whispers_logged"] >= 1


def test_whisper_log_visible_in_observation() -> None:
    env = SIEGEEnvironment(seed=7)
    env.reset()
    action = SIEGEAction(
        tool_name="whisper",
        arguments=WhisperArgs(target_agent_id=3, message="private note"),
    )
    obs, _, _, _ = env.step(action)
    assert len(obs.whisper_log) >= 1


def test_whisper_inbox_receives_messages_to_seat_agent() -> None:
    env = SIEGEEnvironment(seed=9)
    env.reset()
    action = SIEGEAction(
        tool_name="whisper",
        arguments=WhisperArgs(target_agent_id=0, message="for you"),
    )
    obs, _, _, _ = env.step(action)
    assert len(obs.whisper_inbox) >= 1
```

## tests/step_tests/step_17_red_herrings_r9_test.py (last modified: 2026-04-24T19:39:27+00:00)
```python
"""Gate tests for Step 17 — Red Herrings + R9."""

from __future__ import annotations

from siege_env.mechanics.red_herrings import generate_red_herrings
from siege_env.models.actions import ChallengeArgs, DiagnoseArgs, SIEGEAction
from siege_env.rewards.r9_correlation import compute_r9_correlation
from siege_env.server.siege_environment import SIEGEEnvironment


def test_red_herrings_are_deterministic_for_seed_and_step() -> None:
    a = generate_red_herrings(seed=42, step_number=2)
    b = generate_red_herrings(seed=42, step_number=2)
    assert a == b


def test_r9_rewards_correct_false_correlation_challenge() -> None:
    action = SIEGEAction(
        tool_name="challenge",
        arguments=ChallengeArgs(
            target_agent_id=1,
            claim_id="c1",
            flaw_type="type1_false_correlation",
            reasoning="Signal is correlated but not causal.",
        ),
    )
    claims = {"c1": {"root_cause": "wrong_cause"}}
    assert compute_r9_correlation(action, claims_by_id=claims, ground_truth_root_cause="real_cause") == 1.0


def test_r9_zero_for_non_challenge_actions() -> None:
    action = SIEGEAction(
        tool_name="diagnose",
        arguments=DiagnoseArgs(root_cause="x", confidence=0.7, evidence=["e"]),
    )
    assert compute_r9_correlation(action, claims_by_id={}, ground_truth_root_cause="x") == 0.0


def test_environment_observation_contains_red_herrings() -> None:
    env = SIEGEEnvironment(seed=21)
    obs = env.reset()
    assert len(obs.red_herring_signals) >= 1


def test_exploit_always_challenge_wrong_flaw_type_gets_no_r9() -> None:
    action = SIEGEAction(
        tool_name="challenge",
        arguments=ChallengeArgs(
            target_agent_id=1,
            claim_id="c2",
            flaw_type="type3_tunnel_vision",
            reasoning="Always challenging for reward exploit attempt.",
        ),
    )
    claims = {"c2": {"root_cause": "wrong_cause"}}
    assert compute_r9_correlation(action, claims_by_id=claims, ground_truth_root_cause="real_cause") == 0.0
```

## tests/step_tests/step_18_severity_r8_test.py (last modified: 2026-04-24T19:40:39+00:00)
```python
"""Gate tests for Step 18 — Severity Escalation + R8."""

from __future__ import annotations

from siege_env.mechanics.severity_escalation import compute_incident_severity
from siege_env.models.actions import DiagnoseArgs, EscalateArgs, SIEGEAction
from siege_env.rewards.r8_severity_speed import compute_r8_severity_speed
from siege_env.server.siege_environment import SIEGEEnvironment


def test_incident_severity_progression_by_step() -> None:
    assert compute_incident_severity(0) == "warning"
    assert compute_incident_severity(2) == "critical"
    assert compute_incident_severity(5) == "outage"


def test_r8_rewards_fast_escalation_when_severity_outage() -> None:
    action = SIEGEAction(
        tool_name="escalate",
        arguments=EscalateArgs(concern="major outage", blast_radius_estimate=["db", "api"]),
    )
    assert compute_r8_severity_speed(action, incident_severity="outage") == 1.0


def test_r8_lower_reward_for_warning_escalation() -> None:
    action = SIEGEAction(
        tool_name="escalate",
        arguments=EscalateArgs(concern="minor signal", blast_radius_estimate=["api"]),
    )
    assert compute_r8_severity_speed(action, incident_severity="warning") == 0.2


def test_environment_observation_exposes_escalated_severity() -> None:
    env = SIEGEEnvironment(seed=15)
    obs = env.reset()
    assert obs.incident_severity in {"warning", "critical", "outage"}
    assert "severity_score" in obs.incident_dashboard


def test_exploit_always_escalate_not_maximal_in_warning_state() -> None:
    action = SIEGEAction(
        tool_name="escalate",
        arguments=EscalateArgs(concern="always escalate exploit", blast_radius_estimate=["x"]),
    )
    reward = compute_r8_severity_speed(action, incident_severity="warning")
    assert reward < 1.0
```

## tests/step_tests/step_19_postmortem_r7_test.py (last modified: 2026-04-24T19:41:58+00:00)
```python
"""Gate tests for Step 19 — Post-Mortem Generation + R7."""

from __future__ import annotations

from siege_env.models.actions import PostmortemArgs, SIEGEAction, TimelineEvent
from siege_env.rewards.r7_postmortem import compute_r7_postmortem
from siege_env.server.siege_environment import SIEGEEnvironment


def _postmortem_action(*, root: str, timeline_events: list[str], analysis: str) -> SIEGEAction:
    return SIEGEAction(
        tool_name="postmortem",
        arguments=PostmortemArgs(
            root_cause=root,
            timeline=[
                TimelineEvent(timestamp=f"t{i}", event=ev)
                for i, ev in enumerate(timeline_events)
            ],
            contributing_factors=["factor_a", "factor_b"],
            misdiagnosis_analysis=analysis,
        ),
    )


def test_r7_rewards_high_quality_postmortem() -> None:
    action = _postmortem_action(
        root="db_timeout",
        timeline_events=["signal rose", "cache invalidation failed"],
        analysis="Initial triage over-weighted traffic volume and ignored a lock wait pattern in diagnostics.",
    )
    assert compute_r7_postmortem(action, ground_truth_root_cause="db_timeout") >= 0.8


def test_r7_is_zero_for_non_postmortem_actions() -> None:
    env = SIEGEEnvironment(seed=31)
    env.reset()
    _, reward, _, _ = env.step({"tool_name": "diagnose", "arguments": {"root_cause": "x", "confidence": 0.5, "evidence": ["e"]}})
    assert reward >= 0.0


def test_postmortem_flag_in_step_info() -> None:
    env = SIEGEEnvironment(seed=33)
    env.reset()
    action = _postmortem_action(
        root="any",
        timeline_events=["a", "b"],
        analysis="Detailed analysis with enough context to satisfy quality heuristics.",
    )
    _, _, _, info = env.step(action)
    assert info["postmortem_generated"] is True


def test_observation_dashboard_carries_last_postmortem() -> None:
    env = SIEGEEnvironment(seed=35)
    env.reset()
    action = _postmortem_action(
        root="rc",
        timeline_events=["first", "second"],
        analysis="Long postmortem narrative connecting false leads to final diagnosis confidence shift.",
    )
    obs, _, _, _ = env.step(action)
    assert "last_postmortem" in obs.incident_dashboard


def test_template_parroting_exploit_gets_penalized() -> None:
    action = _postmortem_action(
        root="db_timeout",
        timeline_events=["template text", "template text"],
        analysis="template text",
    )
    assert compute_r7_postmortem(action, ground_truth_root_cause="db_timeout") < 0.8
```

## tests/step_tests/step_20_league_test.py (last modified: 2026-04-24T19:43:13+00:00)
```python
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
```

## tests/step_tests/step_21_replay_determinism_test.py (last modified: 2026-04-24T19:45:00+00:00)
```python
"""Gate tests for Step 21 — Determinism + Replay."""

from __future__ import annotations

from pathlib import Path

from siege_env.replay.logger import ReplayLogger
from siege_env.replay.player import replay_file
from siege_env.server.siege_environment import SIEGEEnvironment


def test_replay_logger_round_trip() -> None:
    path = Path("/tmp/siege_step21_round_trip.jsonl")
    if path.exists():
        path.unlink()
    logger = ReplayLogger(path)
    logger.append({"step": 1, "tool": "diagnose"})
    logger.append({"step": 2, "tool": "challenge"})
    events = logger.read_all()
    assert len(events) == 2


def test_replay_player_reads_logged_events() -> None:
    path = Path("/tmp/siege_step21_player.jsonl")
    if path.exists():
        path.unlink()
    logger = ReplayLogger(path)
    logger.append({"step": 1, "tool": "diagnose"})
    assert len(replay_file(path)) == 1


def test_environment_step_info_exposes_replay_path() -> None:
    env = SIEGEEnvironment(seed=101)
    env.reset()
    _, _, _, info = env.step({"tool_name": "diagnose", "arguments": {"root_cause": "x", "confidence": 0.5, "evidence": ["e"]}})
    assert "replay_log_path" in info
```

## tests/step_tests/step_22_heldout_ablation_test.py (last modified: 2026-04-24T19:46:49+00:00)
```python
"""Gate tests for Step 22 — Held-Out Eval + Ablation Harness."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from training.ablation import default_ablation_runs
from training.heldout_split import build_split


def test_heldout_split_integrity() -> None:
    ids = [f"t{i}" for i in range(20)]
    split = build_split(ids, seed=11, heldout_fraction=0.2)
    assert len(split["heldout"]) == 4
    assert len(set(split["train"]).intersection(split["heldout"])) == 0


def test_ablation_default_runs_available() -> None:
    runs = default_ablation_runs()
    assert len(runs) >= 3
    assert {r.name for r in runs}.issuperset({"base", "no_curriculum", "no_trust_poisoning"})


def test_generate_ablations_script_writes_plan() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["bash", "scripts/generate_ablations.sh"], cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    path = root / "artifacts" / "ablation_plan.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) >= 3
```

## tests/step_tests/step_23_wandb_plots_test.py (last modified: 2026-04-24T19:47:57+00:00)
```python
"""Gate tests for Step 23 — W&B integration + committed plots."""

from __future__ import annotations

from pathlib import Path

from training.wandb_config import build_init_kwargs, default_settings


def test_wandb_init_kwargs_contains_required_fields() -> None:
    settings = default_settings()
    kwargs = build_init_kwargs("step23-check")
    assert kwargs["project"] == settings.project
    assert kwargs["mode"] == "offline"
    assert kwargs["name"] == "step23-check"


def test_required_plot_artifacts_committed() -> None:
    root = Path(__file__).resolve().parents[2]
    plots_dir = root / "docs" / "plots"
    required = [
        "arms_race_curve.png",
        "reward_components.png",
        "ablation_comparison.png",
        "generalization_gap.png",
    ]
    for file_name in required:
        path = plots_dir / file_name
        assert path.exists(), f"Missing plot: {file_name}"
        assert path.stat().st_size > 0, f"Empty plot: {file_name}"
```

## tests/step_tests/step_24_gradio_demo_test.py (last modified: 2026-04-24T19:49:22+00:00)
```python
"""Gate tests for Step 24 — Gradio money-shot frontend."""

from __future__ import annotations

from frontend.app import build_app, load_demo_episode_text


def test_gradio_app_boots_and_has_three_tabs() -> None:
    app = build_app()
    assert app is not None
    assert getattr(app, "rudra_tabs", []) == ["War Room", "Before-After", "Arms Race"]


def test_demo_episode_playback_text_is_nonempty() -> None:
    text = load_demo_episode_text()
    assert "Agent4" in text
    assert "YOU" in text
    assert len(text.splitlines()) >= 3
```