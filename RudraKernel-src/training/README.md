# Step 25: GRPO Training Pipeline

This directory contains **two versions** of GRPO training for the RudraKernel project:

1. **Lightweight scaffold** (`grpo_train.py`) — Fast, judge-runnable, works locally & Colab
2. **Full Unsloth/TRL** (`grpo_train_unsloth.py`) — Production-grade, real model fine-tuning

---

## Quick Start

### Option 1: Lightweight (Recommended for quick demo)

**Local:**
```bash
cd /path/to/RudraKernel/RudraKernel-src
python -m training.grpo_train --episodes 50 --output-dir artifacts/training
```

**Colab:**
```python
!python training/grpo_train.py --episodes 50 --output-dir artifacts/training
```

✅ **Output:** checkpoint + metrics JSON (~5-10 seconds runtime)

---

### Option 2: Full Unsloth/TRL (Production training)

**Requires:** GPU (Colab A100/T4), Unsloth + TRL packages

**Colab:**
```python
# See GRPO_COLAB_GUIDE.md for full 8-cell step-by-step walkthrough
!python training/grpo_train_unsloth.py \
  --model "unsloth/Qwen2.5-4B-Instruct-bnb-4bit" \
  --episodes 200 \
  --epochs 3
```

✅ **Output:** Fine-tuned model checkpoint + metrics (~30-50 mins runtime)

---

## File Structure

```
training/
├── grpo_train.py                    # Lightweight scaffold (50 episodes, mock gradients)
├── grpo_train_unsloth.py            # Full Unsloth/TRL trainer (real model + gradients)
├── colab_notebook.ipynb             # Interactive Colab notebook (lightweight version)
├── GRPO_COLAB_GUIDE.md              # Step-by-step Colab walkthrough (Unsloth version)
├── configs/
│   └── base.yaml                    # Base config template
├── ablation.py                      # Ablation studies (for later)
├── heldout_split.py                 # Held-out test set generation
├── wandb_config.py                  # W&B logging config
└── README.md                        # This file
```

---

## Version Comparison

| Aspect | Lightweight | Unsloth/TRL |
|--------|------------|-----------|
| **Model** | None | Qwen3 4B (4-bit quantized) |
| **Trajectories** | Synthetic (hardcoded) | Real SIEGEEnv rollouts |
| **Gradient Updates** | No (mock metrics only) | Yes (TRL GRPO) |
| **GPU Required** | No | Yes (A100/T4) |
| **Runtime** | ~5 sec | ~30-50 min |
| **Output Size** | ~1 KB metrics | ~4 GB checkpoint |
| **W&B Logging** | No | Yes |
| **Use Case** | Demo, testing, CI | Real training, submission |

---

## Lightweight Version (`grpo_train.py`)

### Features
- **Zero GPU requirement** — runs on CPU
- **Fast** — completes in seconds
- **Deterministic** — same seed = same output
- **Judge-runnable** — works in Colab without GPU
- **Tests integrated** — gate test in `tests/step_tests/step_25_grpo_training_test.py`

### What It Does
1. Environment smoke check
2. Rubric integrity validation (all 9 R1-R9 rubrics present)
3. Scripted baseline (8 episodes, hardcoded actions)
4. Frozen policy rollout (8 episodes, random actions)
5. Mini training run (50 episodes, stochastic mode)
6. Writes checkpoint + metrics to `artifacts/training/`

### Output
```json
{
  "run_name": "base",
  "episodes_completed": 50,
  "non_zero_gradient_signal": true,
  "reward_mean": 1.27,
  "reward_std": 0.7156,
  "checkpoint_path": "artifacts/training/base_checkpoint.json",
  "metrics_path": "artifacts/training/base_metrics.json"
}
```

### CLI Usage
```bash
python -m training.grpo_train \
  --episodes 50 \
  --baseline-episodes 8 \
  --max-steps 5 \
  --output-dir artifacts/training \
  --seed 42
```

---

## Full Unsloth Version (`grpo_train_unsloth.py`)

### Features
- **Real model fine-tuning** — Qwen3 4B via Unsloth 4-bit
- **Gradient updates** — TRL GRPO trainer with real loss optimization
- **Real trajectories** — collected from live SIEGEEnv
- **W&B tracking** — live experiment monitoring
- **Production-ready** — saves model checkpoint + config

### What It Does
1. Load Qwen3 4B with Unsloth 4-bit quantization
2. Collect real trajectories from SIEGEEnv (200+ episodes)
3. Format trajectories as LLM training data
4. Configure TRL GRPO trainer
5. Run GRPO training loop (3 epochs)
6. Save fine-tuned model + metrics
7. Log to W&B (optional)

### Output
```json
{
  "model_name": "unsloth/Qwen2.5-4B-Instruct-bnb-4bit",
  "num_epochs": 3,
  "total_trajectories": 200,
  "final_reward_mean": 1.35,
  "final_reward_std": 0.68,
  "final_train_loss": 0.85,
  "training_duration_seconds": 2847,
  "checkpoint_path": "artifacts/training/unsloth/final_model/pytorch_model.bin",
  "wandb_run_url": "https://wandb.ai/..."
}
```

### CLI Usage
```bash
python training/grpo_train_unsloth.py \
  --model "unsloth/Qwen2.5-4B-Instruct-bnb-4bit" \
  --episodes 200 \
  --epochs 3 \
  --output-dir artifacts/training/unsloth \
  --no-wandb  # omit if W&B is configured
```

### Install Unsloth + Dependencies
```bash
pip install torch transformers datasets
pip install git+https://github.com/unslothai/unsloth.git
pip install trl wandb
pip install -e ".[grpo]"  # from project root
```

---

## Configuration

### Lightweight Config (`grpo_train.py`)
Edit or override via YAML:
```yaml
# training/configs/base.yaml
name: "my-run"
seed: 42
episodes: 50
baseline_episodes: 8
max_steps: 5
output_dir: "artifacts/training"
```

### Unsloth Config (in code)
Edit `GRPOTrainingConfig` dataclass in `grpo_train_unsloth.py`:
```python
@dataclass
class GRPOTrainingConfig:
    model_name: str = "unsloth/Qwen2.5-4B-Instruct-bnb-4bit"
    trajectory_episodes: int = 200
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 2  # Adjust for OOM
    log_to_wandb: bool = True
```

---

## Testing

### Lightweight Version
```bash
pytest tests/step_tests/step_25_grpo_training_test.py -v
# Runs 3 tests: mini-run, CLI smoke test, notebook validation
```

### Unsloth Version
```bash
pytest tests/step_tests/step_25_grpo_unsloth_test.py -v
# Checks: script exists, config valid, guide present, pyproject OK
```

### Master Suite (All Tests)
```bash
pytest tests/master_suite.py -q
# Should show 155+ passed
```

---

## Colab Guides

### For Lightweight Training
Open [colab_notebook.ipynb](colab_notebook.ipynb) in Colab:
1. Install deps
2. Run training
3. Validate artifacts

### For Full Unsloth Training
Follow [GRPO_COLAB_GUIDE.md](GRPO_COLAB_GUIDE.md) step-by-step:
- **8 cells** covering clone → install → train → validate
- **~30-50 min** total runtime on A100

---

## Troubleshooting

### Lightweight Version

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: siege_env` | Run `pip install -e .` from project root |
| `checkpoint doesn't exist` | Check `artifacts/training/` directory; re-run training |

### Unsloth Version

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: unsloth` | Run `pip install git+https://github.com/unslothai/unsloth.git` |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` in config |
| `Qwen model not found` | Check internet; HF model auto-downloads on first load |
| `W&B login failed` | Use `--no-wandb` or configure API key |

---

## Next Steps

1. **Lightweight runs locally?** ✅ Move to Step 26 (HF deploy)
2. **Full Unsloth running on Colab?** ✅ Push checkpoint to HF Hub
3. **Need to scale?** Increase `--episodes` or `--epochs` (will take longer)
4. **Want to iterate?** Modify trajectory collection or reward signals in `collect_trajectories()`

---

## References

- **Unsloth GitHub:** https://github.com/unslothai/unsloth
- **TRL Documentation:** https://huggingface.co/docs/trl
- **GRPO Paper:** [arxiv:2402.06358](https://arxiv.org/abs/2402.06358)
- **Qwen3 Model:** https://huggingface.co/Qwen/Qwen2.5-4B-Instruct

---

**Questions?** Check the IMPLEMENTATION_PLAN.md Section 12 for hackathon requirements.
