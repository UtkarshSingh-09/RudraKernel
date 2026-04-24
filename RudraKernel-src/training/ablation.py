"""Ablation harness helpers for Step 22."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AblationRun:
    name: str
    enabled_components: list[str]


def default_ablation_runs() -> list[AblationRun]:
    return [
        AblationRun(
            name="base", enabled_components=["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"]
        ),
        AblationRun(
            name="no_curriculum",
            enabled_components=["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"],
        ),
        AblationRun(
            name="no_trust_poisoning",
            enabled_components=["r1", "r3", "r4", "r5", "r6", "r7", "r8", "r9"],
        ),
    ]
