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
