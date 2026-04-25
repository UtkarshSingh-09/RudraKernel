"""E2E test: training loop — verify the GRPO training entrypoint works end-to-end."""

from __future__ import annotations

from pathlib import Path

from training.grpo_train import TrainingConfig, run_step25_training


class TestTrainingLoop:
    def test_mini_training_run_produces_artifacts(self, tmp_path: Path) -> None:
        output = tmp_path / "e2e_training"
        summary = run_step25_training(
            TrainingConfig(
                name="e2e-test",
                seed=42,
                episodes=10,
                baseline_episodes=3,
                max_steps=3,
                output_dir=str(output),
            )
        )
        assert summary.episodes_completed == 10
        assert Path(summary.checkpoint_path).exists()
        assert Path(summary.metrics_path).exists()

    def test_training_produces_non_zero_gradient(self, tmp_path: Path) -> None:
        output = tmp_path / "e2e_gradient"
        summary = run_step25_training(
            TrainingConfig(
                name="e2e-gradient",
                seed=7,
                episodes=20,
                baseline_episodes=5,
                max_steps=5,
                output_dir=str(output),
            )
        )
        assert summary.non_zero_gradient_signal is True
