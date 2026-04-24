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
