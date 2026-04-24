"""Gate tests for Step 23 — W&B integration + committed plots."""

from __future__ import annotations

from pathlib import Path

from training.wandb_config import build_init_kwargs, default_settings


def test_wandb_init_kwargs_contains_required_fields() -> None:
    settings = default_settings()
    kwargs = build_init_kwargs("step23-check")
    assert kwargs["project"] == settings.project
    assert kwargs["mode"] == "offline"
    assert kwargs["name"] == "step23-check"


def test_required_plot_artifacts_committed() -> None:
    root = Path(__file__).resolve().parents[2]
    plots_dir = root / "docs" / "plots"
    required = [
        "arms_race_curve.png",
        "reward_components.png",
        "ablation_comparison.png",
        "generalization_gap.png",
    ]
    for file_name in required:
        path = plots_dir / file_name
        assert path.exists(), f"Missing plot: {file_name}"
        assert path.stat().st_size > 0, f"Empty plot: {file_name}"
