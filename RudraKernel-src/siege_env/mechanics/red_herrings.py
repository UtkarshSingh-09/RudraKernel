"""Red herring signal generator for SIEGE Step 17."""

from __future__ import annotations

from random import Random

RED_HERRING_POOL = [
    "minor_network_jitter",
    "non_impacting_cache_miss_burst",
    "stale_dashboard_annotation",
    "background_batch_spike",
]


def generate_red_herrings(
    *, seed: int, step_number: int, count: int = 2
) -> list[dict[str, object]]:
    rng = Random(seed + (step_number * 13))
    picks = rng.sample(RED_HERRING_POOL, k=min(count, len(RED_HERRING_POOL)))
    return [{"signal": signal, "is_red_herring": True} for signal in picks]
