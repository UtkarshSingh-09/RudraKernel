"""Validation helpers used across environment/server boundaries."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from siege_env.models import SIEGEAction


def validate_action_payload(
    payload: SIEGEAction | dict[str, Any],
) -> tuple[SIEGEAction | None, str | None]:
    """Validate a user action payload into a SIEGEAction.

    Returns (action, error_message). Only one side is non-None.
    """
    try:
        action = SIEGEAction.model_validate(payload)
    except ValidationError as exc:
        return None, str(exc)
    return action, None
