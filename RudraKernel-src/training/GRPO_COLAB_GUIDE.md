# Step 25: Full GRPO Training with Unsloth on Colab

## Overview

This guide walks through running **production-grade GRPO training** using Unsloth (4-bit quantized Qwen3 4B) on Google Colab.

**What you'll get:**
- Real trajectory collection from SIEGEEnv
- Fine-tuned policy model via GRPO
- W&B experiment tracking
- Checkpoint saved to disk + HuggingFace

**Runtime:** ~30–60 mins on Colab A100 GPU (shorter on T4)

---

## Prerequisites

1. **Colab GPU runtime** (A100 or T4)
   - Go to Runtime → Change runtime type → GPU (High-RAM optional)
2. **GitHub repo cloned**
3. **W&B account** (optional, for logging)

---

## Colab Cells (Step-by-Step)

### Cell 1: Clone & Setup

```python
!cd /content && rm -rf RudraKernel
!git clone https://github.com/<YOUR_USERNAME>/RudraKernel.git
%cd /content/RudraKernel/RudraKernel-src
!pwd
```

Expected output:
```
/content/RudraKernel/RudraKernel-src
```

---

### Cell 2: Install Torch + Core Deps

```python
!pip install -U pip setuptools wheel -q
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
!pip install pyyaml pydantic fastapi requests uvicorn openenv -q
```

(This installs PyTorch 2.0+ with CUDA 11.8 support for A100/T4)

---

### Cell 3: Install Unsloth + TRL Stack

```python
!pip install -U transformers datasets -q
!pip install git+https://github.com/unslothai/unsloth.git --quiet
!pip install trl wandb -q
```

**Note:** First time install ~2–3 mins. Unsloth auto-detects GPU and optimizes.

---

### Cell 4: Install RudraKernel Package

```python
%cd /content/RudraKernel/RudraKernel-src
!python -m pip install --no-build-isolation . -q
```

Expected: No errors, all packages installed.

---

### Cell 5: Login to W&B (Optional)

```python
import wandb
wandb.login()
# Follow the prompt to paste your W&B API key
```

**Skip this if you don't want cloud logging.**

---

### Cell 6: Run Full GRPO Training

```python
%cd /content/RudraKernel/RudraKernel-src
!python training/grpo_train_unsloth.py \
  --model "unsloth/Qwen2.5-4B-Instruct-bnb-4bit" \
  --episodes 200 \
  --epochs 3 \
  --output-dir artifacts/training/unsloth
```

**Parameters:**
- `--episodes 200` — real SIEGEEnv trajectory rollouts
- `--epochs 3` — GRPO training passes over data
- `--output-dir` — where to save checkpoint + metrics

**Expected runtime:**
- 200 trajectories collection: ~5–10 mins
- GRPO training (3 epochs): ~20–40 mins
- **Total: ~30–50 mins**

**Output (printed JSON):**
```json
{
  "model_name": "unsloth/Qwen2.5-4B-Instruct-bnb-4bit",
  "num_epochs": 3,
  "total_trajectories": 200,
  "final_reward_mean": 1.35,
  "final_reward_std": 0.68,
  "best_reward": 4.0,
  "final_train_loss": 0.85,
  "learning_rate": 5e-05,
  "total_tokens_processed": 1024000,
  "training_duration_seconds": 2847.3,
  "checkpoint_path": "artifacts/training/unsloth/final_model/pytorch_model.bin",
  "metrics_path": "artifacts/training/unsloth/metrics.json",
  "wandb_run_url": "https://wandb.ai/...",
  "completed_at": "2026-04-25T..."
}
```

---

### Cell 7: Validate Checkpoint

```python
from pathlib import Path
import json

metrics_path = Path("/content/RudraKernel/RudraKernel-src/artifacts/training/unsloth/metrics.json")
checkpoint_dir = Path("/content/RudraKernel/RudraKernel-src/artifacts/training/unsloth/final_model")

print("Checkpoint exists:", checkpoint_dir.exists())
print("Metrics exists:", metrics_path.exists())

if metrics_path.exists():
    m = json.loads(metrics_path.read_text())
    print("\n=== TRAINING RESULTS ===")
    print(f"Trajectories: {m['total_trajectories']}")
    print(f"Final reward mean: {m['final_reward_mean']:.3f}")
    print(f"Final train loss: {m['final_train_loss']:.3f}")
    print(f"W&B URL: {m['wandb_run_url']}")

if checkpoint_dir.exists():
    files = list(checkpoint_dir.glob("*"))
    print(f"\n=== CHECKPOINT ({len(files)} files) ===")
    for f in sorted(files)[:5]:
        print(f"  {f.name}")
    if len(files) > 5:
        print(f"  ... and {len(files)-5} more")
```

Expected output:
```
Checkpoint exists: True
Metrics exists: True

=== TRAINING RESULTS ===
Trajectories: 200
Final reward mean: 1.35
Final train loss: 0.85
W&B URL: https://wandb.ai/...

=== CHECKPOINT (8 files) ===
  adapter_config.json
  adapter_model.bin
  config.json
  generation_config.json
  pytorch_model.bin
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
```

---

### Cell 8: Load & Test Fine-Tuned Model (Optional)

```python
from unsloth import FastLanguageModel
import torch

checkpoint_path = "/content/RudraKernel/RudraKernel-src/artifacts/training/unsloth/final_model"

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Inference test
model = FastLanguageModel.for_inference(model)

prompt = "Diagnose the root cause: Database latency spike at 3 AM. Error: Connection timeout."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model output:")
print(result)
```

---

## Configuration Details

### Training Config (in code)

```python
class GRPOTrainingConfig:
    model_name = "unsloth/Qwen2.5-4B-Instruct-bnb-4bit"  # Qwen3 4B, 4-bit quantized
    num_train_epochs = 3                                  # GRPO passes
    trajectory_episodes = 200                             # Real env rollouts
    per_device_train_batch_size = 2                       # Colab GPU memory
    gradient_accumulation_steps = 2                       # Effective batch: 4
    learning_rate = 5e-5                                  # Conservative for fine-tuning
    max_trajectory_length = 512                           # Tokens per trajectory
```

### Memory Usage

| GPU | Batch Size | Epoch Time |
|-----|-----------|-----------|
| T4 (16GB) | 2 | ~15 min |
| A100 (40GB) | 4–8 | ~5–8 min |

Adjust `per_device_train_batch_size` if OOM errors occur.

---

## W&B Tracking

If W&B is enabled (default), training automatically logs:
- Training loss per step
- Learning rate schedule
- Gradient norms
- Reward statistics

**View live:** Click the `wandb_run_url` from final output.

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: unsloth` | Run Cell 3 again |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` to 1 |
| `Qwen model not found` | Check internet; model downloads on first load |
| `W&B login failed` | Skip Cell 5 (logging optional) |

---

## Next Steps

1. **Download checkpoint:** Use Colab's file browser to download final_model/
2. **Deploy to HF:** Push to Hugging Face Model Hub
3. **Evaluate:** Run held-out test set against baseline
4. **Iterate:** Adjust epochs/lr/data and re-train

---

## Key Differences from Lightweight Version

| Feature | Lightweight | Unsloth GRPO |
|---------|------------|-------------|
| Model | None | Qwen3 4B |
| Trajectories | Synthetic (5) | Real SIEGEEnv (200+) |
| Gradients | Mock | Real (SGD via TRL) |
| Training | ~30s | ~30–50 mins |
| Outputs | Metrics JSON | Fine-tuned checkpoint + metrics |
| W&B | No | Yes |
| GPU | Not needed | Required (A100/T4) |

---

**Done!** You now have a production GRPO-trained policy. 🚀
