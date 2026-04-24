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
