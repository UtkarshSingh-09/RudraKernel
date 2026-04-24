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
    if not isinstance(template["source_url"], str) or not template["source_url"].startswith(
        "https://"
    ):
        raise ValueError(f"Template '{template['id']}' has invalid 'source_url'.")
    if not isinstance(template["root_cause"], str) or not template["root_cause"].strip():
        raise ValueError(f"Template '{template['id']}' has invalid 'root_cause'.")

    for list_key in ("observable_signals", "flaw_types", "blast_radius"):
        value = template[list_key]
        if not isinstance(value, list) or not value:
            raise ValueError(f"Template '{template['id']}' has invalid '{list_key}'.")
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(
                f"Template '{template['id']}' contains invalid values in '{list_key}'."
            )

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


# Step 14 append-only extension: expanded template loading (5 seed + 15 additional)
EXPANSION_TEMPLATES_PATH = Path(__file__).with_name("templates_expansion_step14.json")
_ORIGINAL_LOAD_TEMPLATES_STEP14 = load_templates


def load_templates(
    path: Path | None = None,
    *,
    include_step14_expansion: bool = False,
) -> list[dict[str, Any]]:
    seed_templates = _ORIGINAL_LOAD_TEMPLATES_STEP14(path)
    if not include_step14_expansion:
        return seed_templates

    raw_expansion = json.loads(EXPANSION_TEMPLATES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_expansion, list):
        raise ValueError("Step 14 expansion payload must be a list.")

    validated_expansion = [_validate_template(item, idx) for idx, item in enumerate(raw_expansion)]

    merged = seed_templates + validated_expansion
    seen: set[str] = set()
    for template in merged:
        if template["id"] in seen:
            raise ValueError(f"Duplicate template id detected: {template['id']}")
        seen.add(template["id"])
    return merged
