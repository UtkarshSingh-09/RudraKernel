#!/bin/bash
set -e

echo "=== A100 GRPO Training Pipeline ==="

echo "=== Step 1: Install dependencies ==="
pip install -q unsloth "trl<0.20" datasets pyyaml torch transformers wandb

echo "=== Step 2: Verify CUDA ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo "=== Step 3: Clone and install project ==="
git clone --depth 1 https://github.com/UtkarshSingh-09/RudraKernel.git 
cd RudraKernel/RudraKernel-src
pip install -q -e . --no-deps

echo "=== Step 4: Run training ==="
python -m training.grpo_train_unsloth \
  --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
  --episodes 200 \
  --epochs 3 \
  --output-dir artifacts/training/unsloth \
  --no-wandb

echo "=== Step 5: Upload checkpoint ==="
if [ -n "$HF_TOKEN" ]; then
  hf_user_agent huggingface-cli \
    upload ankit-choubey/Rudrakernel-unsloth \
    artifacts/training/unsloth/final_model . \
    --token "$HF_TOKEN"
else
  echo "Warning: HF_TOKEN not set. Skipping HF upload."
fi

echo "=== TRAINING COMPLETE ==="
echo "Checkpoint saved to: artifacts/training/unsloth/final_model"
