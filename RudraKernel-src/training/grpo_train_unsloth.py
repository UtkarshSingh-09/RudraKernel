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
import inspect
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

import torch
import yaml

UNSLOTH_IMPORT_ERROR: ImportError | None = None
TRL_IMPORT_ERROR: ImportError | None = None

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except Exception as exc:
    HAS_UNSLOTH = False
    if isinstance(exc, ImportError):
        UNSLOTH_IMPORT_ERROR = exc
    else:
        UNSLOTH_IMPORT_ERROR = ImportError(str(exc))

try:
    from trl import GRPOConfig, GRPOTrainer
    HAS_TRL = True
except Exception as exc:
    HAS_TRL = False
    if isinstance(exc, ImportError):
        TRL_IMPORT_ERROR = exc
    else:
        TRL_IMPORT_ERROR = ImportError(str(exc))

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from siege_env.rewards import COMPOSED_RUBRICS
from siege_env.server.siege_environment import SIEGEEnvironment


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
UTC = timezone.utc


@dataclass(slots=True)
class GRPOTrainingConfig:
    """Production GRPO training config with Unsloth + TRL."""

    # Model
    model_name: str = "unsloth/Qwen2.5-4B-Instruct-bnb-4bit"
    max_seq_length: int = 1024
    load_in_4bit: bool = True

    # Training
    num_train_epochs: int = 3
    num_mini_batches: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Data
    trajectory_episodes: int = 200  # Real env rollouts
    max_trajectory_length: int = 256
    
    # GPU optimization (A10G 24GB)
    gradient_accumulation_steps: int = 8
    per_device_train_batch_size: int = 1  # Conserve VRAM for GRPO generation
    
    # Logging
    log_to_wandb: bool = False
    wandb_project: str = "rudra-kernel-grpo"
    output_dir: str = "artifacts/training/unsloth"
    
    # HF Hub — push trained LoRA after training
    hub_model_id: str = "ankit-choubey/siege-grpo-lora"
    
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
        env_seed = seed + ep
        env = SIEGEEnvironment(seed=env_seed, max_steps=max_steps)
        obs = env.reset()

        # Build rich prompt with observation data so model can actually see the incident
        claims_text = ""
        for idx, claim in enumerate(obs.agent_claims):
            agent_id = claim.get("agent_id", idx)
            text = claim.get("claim", claim.get("root_cause", "unknown"))
            claims_text += f"  Agent {agent_id}: {text}\n"

        evidence_text = ""
        evidence_values: list[str] = []
        for item in obs.available_evidence:
            value = item.get("value")
            if isinstance(value, str) and value.strip():
                evidence_values.append(value.strip())
                evidence_text += f"  - {value.strip()}\n"
        if not evidence_values:
            evidence_values = ["signal_unavailable"]
            evidence_text = "  - signal_unavailable\n"

        prompt = (f"[EnvSeed:{env_seed}]\n"
                  f"INCIDENT: {getattr(obs, 'incident_type', 'unknown')}\n"
                  f"SEVERITY: {getattr(obs, 'severity', 'unknown')}\n"
                  f"AGENT CLAIMS:\n{claims_text}"
                  f"EVIDENCE:\n{evidence_text}"
                  f"Diagnose the root cause. Output root_cause=<cause>, confidence=<0-1>.")

        trajectory = {
            "prompt": prompt,
            "env_seed": env_seed,
            "actions": [],
            "rewards": [],
            "total_reward": 0.0,
        }

        done = False
        step = 0
        while not done and step < max_steps:
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

            action = {
                "tool_name": "diagnose",
                "arguments": {
                    "root_cause": root_cause,
                    "confidence": confidence,
                    "evidence": evidence_values[:3],
                    "alternative_hypotheses": [],
                },
            }
            obs, reward, done, _info = env.step(action)
            trajectory["rewards"].append(float(reward))
            trajectory["total_reward"] += float(reward)
            step += 1

        trajectory["completion"] = "Episode complete" if done else "Max steps reached"
        trajectories.append(trajectory)
        
        if (ep + 1) % 50 == 0:
            logger.info(f"Collected {ep + 1}/{num_episodes} trajectories")
    
    logger.info(f"✓ Collected {len(trajectories)} trajectories")
    return trajectories


def format_trajectory_for_training(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Format trajectory as prompt dict for GRPO dataset."""
    
    prompt = trajectory["prompt"]
    actions_text = "\n".join(f"  Step {i+1}: {a}" for i, a in enumerate(trajectory["actions"]))
    
    return {
        "prompt": [
            {"role": "system", "content": "You are a site reliability engineer diagnosing incidents."},
            {"role": "user", "content": f"{prompt}\n{actions_text}"},
        ],
    }


def build_siege_reward_func(seed: int, max_steps: int) -> Any:
    """Build a reward function that scores completions using SIEGEEnv.

    FIX: All completions in a GRPO group use the SAME env seed
    (extracted from prompt). Reward variance = completion quality, not luck.
    """

    def siege_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
        """Score each completion against the same environment."""
        # Extract env seed from prompt [EnvSeed:XXX]
        prompts = kwargs.get("prompts", kwargs.get("prompt", []))
        prompt_text = ""
        if isinstance(prompts, list) and prompts:
            p = prompts[0]
            if isinstance(p, list):  # chat format
                prompt_text = p[-1].get("content", "") if p else ""
            elif isinstance(p, str):
                prompt_text = p

        # Same seed for ALL completions in this group
        env_seed = seed
        if "[EnvSeed:" in prompt_text:
            try:
                env_seed = int(prompt_text.split("[EnvSeed:")[1].split("]")[0])
            except (ValueError, IndexError):
                pass

        rewards = []
        for completion in completions:
            try:
                env = SIEGEEnvironment(seed=env_seed, max_steps=max_steps)
                obs = env.reset()

                evidence_values: list[str] = []
                for item in obs.available_evidence:
                    value = item.get("value")
                    if isinstance(value, str) and value.strip():
                        evidence_values.append(value.strip())
                if not evidence_values:
                    evidence_values = ["signal_unavailable"]

                # Parse root_cause and confidence from completion
                root_cause = "unknown_root_cause"
                confidence = 0.5
                for line in completion.split("\n"):
                    low = line.lower().strip()
                    if "root_cause" in low or "root cause" in low:
                        for sep in ["=", ":"]:
                            if sep in line:
                                val = line.split(sep, 1)[1].strip().strip("'\"")
                                if val:
                                    root_cause = val[:100]
                                    break
                    if "confidence" in low:
                        for sep in ["=", ":"]:
                            if sep in line:
                                try:
                                    confidence = float(line.split(sep, 1)[1].strip().strip("'\"% "))
                                    if confidence > 1:
                                        confidence /= 100
                                except ValueError:
                                    pass

                # Hard penalty: no structured output = no reward
                if root_cause == "unknown_root_cause":
                    rewards.append(-0.5)
                    continue

                action = {
                    "tool_name": "diagnose",
                    "arguments": {
                        "root_cause": root_cause,
                        "confidence": max(0.01, min(0.99, confidence)),
                        "evidence": evidence_values[:3],
                        "alternative_hypotheses": [],
                    },
                }
                _obs, reward, _done, _info = env.step(action)
                rewards.append(float(reward))
            except Exception:
                rewards.append(-0.5)
        return rewards

    return siege_reward_func


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
    
    # Attach LoRA adapters (required for training quantized models)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
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
    
    # 2. Format for training dataset
    from datasets import Dataset
    
    formatted_data = [format_trajectory_for_training(t) for t in trajectories]
    train_dataset = Dataset.from_list(formatted_data)
    rewards = [t["total_reward"] for t in trajectories]
    
    # 3. Build reward function
    reward_func = build_siege_reward_func(seed=config.seed, max_steps=config.max_env_steps)
    
    # 4. Load model
    model, tokenizer = setup_unsloth_model(config)
    
    # 5. Setup W&B
    wandb_run_url = None
    if config.log_to_wandb and HAS_WANDB:
        wandb.init(
            project=config.wandb_project,
            name=f"grpo-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
            config=asdict(config),
        )
        wandb_run_url = wandb.run.url if wandb.run else None
    
    # 6. Configure GRPO trainer (TRL/Unsloth APIs can differ across versions)
    grpo_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "num_mini_batches": config.num_mini_batches,
        "logging_steps": 10,
        "save_steps": 50,
        "max_completion_length": config.max_trajectory_length,
        "report_to": ["wandb"] if (config.log_to_wandb and HAS_WANDB) else [],
        "seed": config.seed,
    }

    supported_grpo_args = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    filtered_grpo_kwargs = {k: v for k, v in grpo_kwargs.items() if k in supported_grpo_args}
    dropped_args = sorted(set(grpo_kwargs.keys()) - set(filtered_grpo_kwargs.keys()))
    if dropped_args:
        logger.info("Skipping unsupported GRPOConfig args for this TRL version: %s", dropped_args)

    training_config = GRPOConfig(**filtered_grpo_kwargs)
    
    # 7. Create trainer with reward_funcs
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_config,
        "train_dataset": train_dataset,
        "reward_funcs": reward_func,
    }
    
    # Add tokenizer/processing_class based on TRL version
    trainer_init_params = set(inspect.signature(GRPOTrainer.__init__).parameters.keys())
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    
    trainer = GRPOTrainer(**trainer_kwargs)

    # Workaround: Unsloth compiled GRPOTrainer references multimodal attrs
    # that are missing for text-only models.
    for _attr in (
        "image_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "image_token",
        "vision_start_token",
        "vision_end_token",
    ):
        if not hasattr(trainer, _attr):
            setattr(trainer, _attr, None)
    if not hasattr(trainer, "pad_token"):
        trainer.pad_token = tokenizer.pad_token or ""

    # Workaround: Unsloth 2026.4.8 compiled cache references
    # truncate_with_protected_tokens without importing it.
    # Inject into builtins (fallback scope) + all loaded compiled-cache modules.
    try:
        from trl.trainer.grpo_trainer import truncate_with_protected_tokens as _trunc_fn
    except ImportError:
        def _trunc_fn(prompt_ids, prompt_mask, max_length=None, protected=None):  # type: ignore[misc]
            if max_length is not None and prompt_ids.shape[-1] > max_length:
                prompt_ids = prompt_ids[:, -max_length:]
                prompt_mask = prompt_mask[:, -max_length:]
            return prompt_ids, prompt_mask
    import builtins, sys as _sys  # noqa: E401
    builtins.truncate_with_protected_tokens = _trunc_fn  # type: ignore[attr-defined]
    for _mn, _mo in list(_sys.modules.items()):
        if "unsloth_compiled_cache" in _mn:
            _mo.__dict__.setdefault("truncate_with_protected_tokens", _trunc_fn)
            _mo.__dict__.setdefault("has_images", False)
            _mo.__dict__.setdefault("images", None)
    
    # 8. Train
    logger.info("Starting GRPO training...")
    train_result = trainer.train()
    
    # 9. Save artifacts locally
    final_model_path = output_dir / "final_model"
    model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    logger.info(f"✓ Model saved locally to {final_model_path}")
    
    # 10. Push to HF Hub (critical for ephemeral HF Spaces storage)
    hf_token = os.environ.get("HF_TOKEN", "")
    print(f"\n{'='*60}", flush=True)
    print(f"  HUB PUSH STATUS", flush=True)
    print(f"  hub_model_id: {config.hub_model_id}", flush=True)
    print(f"  HF_TOKEN present: {bool(hf_token)}", flush=True)
    print(f"{'='*60}", flush=True)
    
    if config.hub_model_id and hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            
            # Create repo if it doesn't exist
            api.create_repo(
                repo_id=config.hub_model_id,
                private=True,
                exist_ok=True,
            )
            print(f"  ✓ Repo created/verified: {config.hub_model_id}", flush=True)
            
            # Upload the saved model directory
            api.upload_folder(
                folder_path=str(final_model_path),
                repo_id=config.hub_model_id,
                commit_message="GRPO trained SIEGE LoRA adapter",
            )
            print(f"  ✓ Model uploaded to https://huggingface.co/{config.hub_model_id}", flush=True)
            logger.info(f"✓ Model pushed to https://huggingface.co/{config.hub_model_id}")
        except Exception as hub_exc:
            print(f"  ✗ PUSH FAILED: {hub_exc}", flush=True)
            logger.error(f"Failed to push to Hub: {hub_exc}")
            logger.error("Model is saved locally — push manually if needed.")
    elif config.hub_model_id and not hf_token:
        print("  ✗ HF_TOKEN env var is MISSING — cannot push!", flush=True)
        logger.warning("hub_model_id is set but HF_TOKEN env var is missing — skipping Hub push.")
        logger.warning("Set HF_TOKEN as a Space secret to auto-push after training.")
    
    # 10. Metrics
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
        wandb.log(asdict(summary))
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
        default="unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit",
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
        print("ERROR: Failed to import Unsloth.")
        if UNSLOTH_IMPORT_ERROR is not None:
            print(f"ImportError: {UNSLOTH_IMPORT_ERROR}")
        print("Run: pip install unsloth torch transformers trl datasets")
        return 1
    if not HAS_TRL:
        print("ERROR: Failed to import TRL GRPO modules.")
        if TRL_IMPORT_ERROR is not None:
            print(f"ImportError: {TRL_IMPORT_ERROR}")
        print("Run: pip install trl transformers")
        return 1
    
    # Train
    summary = run_grpo_training(config)
    print(json.dumps(asdict(summary), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
