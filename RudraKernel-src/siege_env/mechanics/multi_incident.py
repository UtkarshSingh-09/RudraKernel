"""Multi-incident helper utilities.

Allows constructing small bundles of concurrent incidents for future steps.
"""

from __future__ import annotations

from typing import Any


def build_incident_bundle(
    primary: dict[str, Any], secondary: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return a normalized active-incidents payload."""
    incidents = [dict(primary)] + [dict(item) for item in secondary]
    return {
        "active_count": len(incidents),
        "incidents": incidents,
    }
