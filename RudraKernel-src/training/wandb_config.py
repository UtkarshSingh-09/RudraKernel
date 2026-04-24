"""W&B integration helpers for Step 23."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class WandbSettings:
    project: str
    entity: str | None
    mode: str = "offline"


def default_settings() -> WandbSettings:
    return WandbSettings(project="rudrakernel-siege", entity=None, mode="offline")


def build_init_kwargs(run_name: str) -> dict[str, Any]:
    settings = default_settings()
    return {
        "project": settings.project,
        "entity": settings.entity,
        "mode": settings.mode,
        "name": run_name,
    }
