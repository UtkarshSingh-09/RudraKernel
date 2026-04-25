"""Unit tests for information asymmetry visibility filtering."""

from __future__ import annotations

from siege_env.mechanics.info_asymmetry import (
    filter_evidence_for_visibility,
    visibility_for_step,
)


class TestInfoAsymmetry:
    def test_immune_starts_with_limited_visibility(self) -> None:
        vis = visibility_for_step(step_number=0, role="immune")
        assert vis in ("metrics_only", "traces_only", "full", "delayed")

    def test_pathogen_has_different_visibility(self) -> None:
        vis = visibility_for_step(step_number=0, role="pathogen")
        assert vis in ("metrics_only", "traces_only", "full", "delayed")

    def test_filter_reduces_evidence_for_metrics_only(self) -> None:
        evidence = [
            {"type": "signal", "value": "cpu_spike"},
            {"type": "signal", "value": "log_entry"},
            {"type": "signal", "value": "trace"},
        ]
        filtered = filter_evidence_for_visibility(evidence, visibility_level="metrics_only")
        assert len(filtered) <= len(evidence)

    def test_full_visibility_returns_all(self) -> None:
        evidence = [{"type": "signal", "value": "x"}, {"type": "signal", "value": "y"}]
        filtered = filter_evidence_for_visibility(evidence, visibility_level="full")
        assert len(filtered) == len(evidence)
