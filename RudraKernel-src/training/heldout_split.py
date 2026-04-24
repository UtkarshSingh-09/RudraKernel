"""Held-out split utilities for Step 22."""

from __future__ import annotations

from random import Random


def build_split(
    template_ids: list[str], *, seed: int = 0, heldout_fraction: float = 0.2
) -> dict[str, list[str]]:
    rng = Random(seed)
    ordered = list(dict.fromkeys(template_ids))
    rng.shuffle(ordered)
    heldout_count = max(1, int(round(len(ordered) * heldout_fraction))) if ordered else 0
    heldout = sorted(ordered[:heldout_count])
    train = sorted(ordered[heldout_count:])
    return {"train": train, "heldout": heldout}
