"""Gate tests for Step 25 — GRPO training script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from training.grpo_train import TrainingConfig, run_step25_training


def test_step25_mini_run_completes_and_writes_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "training_artifacts"
    summary = run_step25_training(
        TrainingConfig(
            name="step25-gate",
            seed=123,
            episodes=50,
            baseline_episodes=6,
            max_steps=5,
            output_dir=str(output_dir),
        )
    )

    assert summary.episodes_completed == 50
    assert summary.non_zero_gradient_signal is True

    checkpoint_path = Path(summary.checkpoint_path)
    metrics_path = Path(summary.metrics_path)
    assert checkpoint_path.exists()
    assert metrics_path.exists()

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["non_zero_gradient_signal"] is True


def test_step25_cli_runs_and_prints_summary(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "cli_artifacts"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.grpo_train",
            "--episodes",
            "10",
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    payload = json.loads(result.stdout)
    assert payload["episodes_completed"] == 10
    assert Path(payload["checkpoint_path"]).exists()
    assert Path(payload["metrics_path"]).exists()


def test_step25_colab_guide_exists() -> None:
    root = Path(__file__).resolve().parents[2]
    guide_path = root / "training" / "GRPO_COLAB_GUIDE.md"
    assert guide_path.exists(), "Missing Step 25 Colab guide."
    content = guide_path.read_text(encoding="utf-8")
    assert "grpo_train_unsloth" in content, "Guide must reference grpo_train_unsloth."
