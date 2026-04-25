"""Seed utilities for deterministic SIEGE runs."""

from __future__ import annotations

import random


def seed_python(seed: int) -> None:
    """Seed Python's global RNG."""
    random.seed(seed)


def derive_seed(base_seed: int, namespace: str) -> int:
    """Derive a stable integer seed from a base seed and namespace label."""
    mix = f"{base_seed}:{namespace}"
    return abs(hash(mix)) % (2**31)
