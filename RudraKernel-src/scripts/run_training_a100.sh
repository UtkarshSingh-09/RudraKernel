#!/bin/bash
set -e

echo "=== A100 GRPO Training Pipeline ==="

REPO_URL="https://github.com/UtkarshSingh-09/RudraKernel.git"
REPO_DIR="RudraKernel"
PROJECT_SUBDIR="RudraKernel-src"

echo "=== Step 1: Install dependencies ==="
pip install -q unsloth "trl<0.20" datasets pyyaml torch transformers wandb

echo "=== Step 2: Verify CUDA ==="
if ! python -c "import sys, torch; ok=torch.cuda.is_available(); print(f'CUDA available: {ok}'); print(f'Device: {torch.cuda.get_device_name(0) if ok else \"CPU\"}'); sys.exit(0 if ok else 2)"; then
  echo "ERROR: No GPU detected. Run this script on HF Space/Notebook with A100 hardware enabled."
  exit 1
fi

echo "=== Step 3: Clone and install project ==="
if [ -d "${PROJECT_SUBDIR}" ] && [ -f "${PROJECT_SUBDIR}/pyproject.toml" ]; then
  cd "${PROJECT_SUBDIR}"
elif [ -d "${REPO_DIR}/${PROJECT_SUBDIR}" ] && [ -f "${REPO_DIR}/${PROJECT_SUBDIR}/pyproject.toml" ]; then
  cd "${REPO_DIR}/${PROJECT_SUBDIR}"
else
  if [ ! -d "${REPO_DIR}" ]; then
    git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
  fi
  cd "${REPO_DIR}/${PROJECT_SUBDIR}"
fi

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
  huggingface-cli \
    upload ankit-choubey/Rudrakernel-unsloth \
    artifacts/training/unsloth/final_model . \
    --token "$HF_TOKEN"
else
  echo "Warning: HF_TOKEN not set. Skipping HF upload."
fi

echo "=== TRAINING COMPLETE ==="
echo "Checkpoint saved to: artifacts/training/unsloth/final_model"
