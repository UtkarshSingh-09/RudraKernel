from __future__ import annotations

INTERNAL_TABS: list[str] = ["War Room", "Before-After", "Arms Race"]

DISPLAY_TAB_NAMES: dict[str, str] = {
    "War Room": "Incident Command",
    "Before-After": "Before-After",
    "Arms Race": "Training Curve",
}


def get_display_tab_name(internal_name: str) -> str:
    return DISPLAY_TAB_NAMES.get(internal_name, internal_name)
