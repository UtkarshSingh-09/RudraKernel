"""Gate tests for Step 25 — Unsloth GRPO training variant."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_grpo_unsloth_script_exists() -> None:
    """Verify Unsloth GRPO training script is present."""
    root = Path(__file__).resolve().parents[2]
    script = root / "training" / "grpo_train_unsloth.py"
    assert script.exists(), "grpo_train_unsloth.py not found"

    content = script.read_text()
    assert "GRPOTrainingConfig" in content
    assert "run_grpo_training" in content
    assert "FastLanguageModel" in content


def test_grpo_config_dataclass() -> None:
    """Verify GRPOTrainingConfig is properly defined."""
    try:
        from training.grpo_train_unsloth import GRPOTrainingConfig

        config = GRPOTrainingConfig(
            trajectory_episodes=10,
            num_train_epochs=1,
        )
        assert config.trajectory_episodes == 10
        assert config.num_train_epochs == 1
        assert config.model_name == "unsloth/Qwen2.5-4B-Instruct-bnb-4bit"
    except ImportError:
        # Unsloth not installed; that's OK for CI
        pytest.skip("Unsloth not installed")


def test_grpo_summary_dataclass() -> None:
    """Verify GRPOTrainingSummary is properly defined."""
    try:
        from training.grpo_train_unsloth import GRPOTrainingSummary

        summary = GRPOTrainingSummary(
            model_name="test_model",
            num_epochs=2,
            total_trajectories=100,
            final_reward_mean=1.5,
            final_reward_std=0.3,
            best_reward=3.0,
            final_train_loss=0.5,
            learning_rate=1e-4,
            total_tokens_processed=50000,
            training_duration_seconds=300.0,
            checkpoint_path="/path/to/checkpoint",
            metrics_path="/path/to/metrics.json",
            wandb_run_url="https://wandb.ai/test",
            completed_at="2026-04-25T...",
        )
        assert summary.total_trajectories == 100
        assert summary.final_reward_mean == 1.5
    except ImportError:
        pytest.skip("Unsloth not installed")


def test_pyproject_has_grpo_optional_deps() -> None:
    """Verify pyproject.toml includes grpo optional dependencies."""
    root = Path(__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    content = pyproject.read_text()

    assert "[project.optional-dependencies]" in content
    assert "grpo = [" in content
    assert "unsloth" in content
    assert "trl" in content
    assert "transformers" in content


def test_colab_guide_exists() -> None:
    """Verify Colab training guide is documented."""
    root = Path(__file__).resolve().parents[2]
    guide = root / "training" / "GRPO_COLAB_GUIDE.md"
    assert guide.exists(), "GRPO_COLAB_GUIDE.md not found"

    content = guide.read_text()
    assert "Colab" in content
    assert "Cell 1" in content
    assert "grpo_train_unsloth.py" in content
    assert "wandb" in content.lower()
