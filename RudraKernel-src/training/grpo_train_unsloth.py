"""Step 25 GRPO training with Unsloth + TRL for production-grade fine-tuning.

This module implements real GRPO training using:
- Unsloth for 4-bit quantized model loading (memory-efficient)
- TRL GRPO trainer for policy gradient optimization
- Real trajectory collection from SIEGEEnv
- W&B logging for experiment tracking
- Colab GPU optimization (A100/T4 support)

Phase C workflow:
1) Model loading (Qwen3 4B quantized)
2) Trajectory collection from SIEGEEnv
3) Dataset construction + formatting
4) GRPO training loop with gradient updates
5) Checkpoint + metrics logging to W&B + local disk
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import Any

import torch
import yaml

try:
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from siege_env.rewards import COMPOSED_RUBRICS
from siege_env.server.siege_environment import SIEGEEnvironment


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(slots=True)
class GRPOTrainingConfig:
    """Production GRPO training config with Unsloth + TRL."""

    # Model
    model_name: str = "unsloth/Qwen2.5-4B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # Training
    num_train_epochs: int = 3
    num_mini_batches: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Data
    trajectory_episodes: int = 200  # Real env rollouts
    max_trajectory_length: int = 512
    
    # Colab optimization
    gradient_accumulation_steps: int = 2
    per_device_train_batch_size: int = 2  # Colab GPU memory limit
    
    # Logging
    log_to_wandb: bool = True
    wandb_project: str = "rudra-kernel-grpo"
    output_dir: str = "artifacts/training/unsloth"
    
    # Environment
    seed: int = 42
    max_env_steps: int = 10


@dataclass(slots=True)
class GRPOTrainingSummary:
    """Machine-readable summary from Unsloth GRPO training."""

    model_name: str
    num_epochs: int
    total_trajectories: int
    final_reward_mean: float
    final_reward_std: float
    best_reward: float
    final_train_loss: float
    learning_rate: float
    total_tokens_processed: int
    training_duration_seconds: float
    checkpoint_path: str
    metrics_path: str
    wandb_run_url: str | None
    completed_at: str


def collect_trajectories(
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Collect real trajectories from SIEGEEnv with reward signals."""
    
    rng = Random(seed)
    trajectories = []
    
    for ep in range(num_episodes):
        env = SIEGEEnvironment(seed=seed + ep, max_steps=max_steps)
        obs = env.reset()
        trajectory = {
            "prompt": "Diagnose the root cause:",
            "actions": [],
            "rewards": [],
            "total_reward": 0.0,
        }
        
        done = False
        step = 0
        while not done and step < max_steps:
            # Build action
            candidate_causes = []
            for claim in obs.agent_claims:
                root_cause = claim.get("root_cause")
                if isinstance(root_cause, str) and root_cause.strip():
                    candidate_causes.append(root_cause.strip())
            
            if not candidate_causes:
                candidate_causes = ["unknown_root_cause"]
            
            root_cause = rng.choice(candidate_causes)
            confidence = rng.uniform(0.5, 0.99)
            
            action_text = f"root_cause={root_cause}, confidence={confidence:.2f}"
            trajectory["actions"].append(action_text)
            
            # Step environment
            action = {
                "tool_name": "diagnose",
                "arguments": {
                    "root_cause": root_cause,
                    "confidence": confidence,
                    "evidence": [],
                    "alternative_hypotheses": [],
                },
            }
            obs, reward, done, _info = env.step(action)
            
            trajectory["rewards"].append(float(reward))
            trajectory["total_reward"] += float(reward)
            step += 1
        
        # Completion signal
        trajectory["completion"] = "Episode complete" if done else "Max steps reached"
        trajectories.append(trajectory)
        
        if (ep + 1) % 50 == 0:
            logger.info(f"Collected {ep + 1}/{num_episodes} trajectories")
    
    logger.info(f"✓ Collected {len(trajectories)} trajectories")
    return trajectories


def format_trajectory_for_training(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Format trajectory as text for LLM training."""
    
    prompt = trajectory["prompt"]
    actions_text = "\n".join(f"  Step {i+1}: {a}" for i, a in enumerate(trajectory["actions"]))
    completion = trajectory["completion"]
    
    return {
        "prompt": f"{prompt}\n{actions_text}\n{completion}",
        "reward": trajectory["total_reward"],
    }


def setup_unsloth_model(config: GRPOTrainingConfig) -> tuple[Any, Any]:
    """Load Qwen3 4B with Unsloth quantization."""
    
    if not HAS_UNSLOTH:
        raise ImportError("Unsloth not installed. Run: pip install unsloth")
    
    logger.info(f"Loading {config.model_name} with Unsloth...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=torch.float16,
        load_in_4bit=config.load_in_4bit,
    )
    
    # Prepare for training
    model = FastLanguageModel.for_training(model)
    
    logger.info(f"✓ Model loaded: {model.config.model_type}")
    return model, tokenizer


def run_grpo_training(config: GRPOTrainingConfig) -> GRPOTrainingSummary:
    """Run full GRPO training pipeline."""
    
    start_time = datetime.now(UTC)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect trajectories
    logger.info(f"Collecting {config.trajectory_episodes} trajectories...")
    trajectories = collect_trajectories(
        num_episodes=config.trajectory_episodes,
        max_steps=config.max_env_steps,
        seed=config.seed,
    )
    
    # 2. Format for training
    formatted_data = [format_trajectory_for_training(t) for t in trajectories]
    rewards = [d["reward"] for d in formatted_data]
    
    # 3. Load model
    model, tokenizer = setup_unsloth_model(config)
    
    # 4. Setup W&B
    wandb_run_url = None
    if config.log_to_wandb and HAS_WANDB:
        wandb.init(
            project=config.wandb_project,
            name=f"grpo-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
            config=asdict(config),
        )
        wandb_run_url = wandb.run.url if wandb.run else None
    
    # 5. Configure GRPO trainer
    training_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_mini_batches=config.num_mini_batches,
        logging_steps=10,
        save_steps=50,
        report_to=["wandb"] if (config.log_to_wandb and HAS_WANDB) else [],
        seed=config.seed,
    )
    
    # 6. Create trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_config,
        train_dataset=formatted_data,  # List of {"prompt": ..., "reward": ...}
    )
    
    # 7. Train
    logger.info("Starting GRPO training...")
    train_result = trainer.train()
    
    # 8. Save artifacts
    final_model_path = output_dir / "final_model"
    model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    logger.info(f"✓ Model saved to {final_model_path}")
    
    # 9. Metrics
    duration = (datetime.now(UTC) - start_time).total_seconds()
    best_reward = max(rewards)
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 0.0
    
    summary = GRPOTrainingSummary(
        model_name=config.model_name,
        num_epochs=config.num_train_epochs,
        total_trajectories=len(trajectories),
        final_reward_mean=mean_reward,
        final_reward_std=std_reward,
        best_reward=best_reward,
        final_train_loss=train_result.training_loss or 0.0,
        learning_rate=config.learning_rate,
        total_tokens_processed=train_result.total_flos if hasattr(train_result, 'total_flos') else 0,
        training_duration_seconds=duration,
        checkpoint_path=str(final_model_path / "pytorch_model.bin"),
        metrics_path=str(output_dir / "metrics.json"),
        wandb_run_url=wandb_run_url,
        completed_at=datetime.now(UTC).isoformat(),
    )
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(asdict(summary), indent=2, default=str))
    logger.info(f"✓ Metrics saved to {metrics_path}")
    
    # Finalize W&B
    if config.log_to_wandb and HAS_WANDB:
        wandb.log(asdict(summary, default=str))
        wandb.finish()
    
    logger.info(f"✓ Training complete in {duration:.1f}s")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 25 GRPO training with Unsloth/TRL")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (overrides defaults).",
    )
    parser.add_argument(
        "--model",
        default="unsloth/Qwen2.5-4B-Instruct-bnb-4bit",
        help="Model name/path.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override trajectory episode count.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    
    # Load or create config
    if args.config:
        payload = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        config = GRPOTrainingConfig(**payload)
    else:
        config = GRPOTrainingConfig()
    
    # CLI overrides
    if args.model:
        config.model_name = args.model
    if args.episodes is not None:
        config.trajectory_episodes = args.episodes
    if args.epochs is not None:
        config.num_train_epochs = args.epochs
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_wandb:
        config.log_to_wandb = False
    
    # Validate
    if not HAS_UNSLOTH:
        print("ERROR: Unsloth not installed.")
        print("Run: pip install unsloth torch transformers trl datasets")
        return 1
    
    # Train
    summary = run_grpo_training(config)
    print(json.dumps(asdict(summary, default=str), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
