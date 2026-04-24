"""Gate tests for Step 15 — Information Asymmetry."""

from __future__ import annotations

from siege_env.mechanics.info_asymmetry import (
    filter_evidence_for_visibility,
    visibility_for_step,
)
from siege_env.server.siege_environment import SIEGEEnvironment


def test_visibility_schedule_for_immune_agent() -> None:
    assert visibility_for_step(0, "immune") == "metrics_only"
    assert visibility_for_step(2, "immune") == "traces_only"
    assert visibility_for_step(4, "immune") == "full"


def test_visibility_schedule_for_pathogen_agent() -> None:
    assert visibility_for_step(0, "pathogen") == "delayed"
    assert visibility_for_step(5, "pathogen") == "delayed"


def test_evidence_filtering_respects_visibility_levels() -> None:
    evidence = [{"value": "a"}, {"value": "b"}, {"value": "c"}]
    assert len(filter_evidence_for_visibility(evidence, visibility_level="metrics_only")) == 1
    assert len(filter_evidence_for_visibility(evidence, visibility_level="traces_only")) == 2
    assert len(filter_evidence_for_visibility(evidence, visibility_level="full")) == 3


def test_environment_observation_contains_visibility_and_filtered_evidence() -> None:
    env = SIEGEEnvironment(seed=3)
    obs = env.reset()
    assert obs.visibility_level in {"metrics_only", "traces_only", "full", "delayed"}
    assert len(obs.available_evidence) >= 1
    if obs.visibility_level == "metrics_only":
        assert len(obs.available_evidence) == 1
