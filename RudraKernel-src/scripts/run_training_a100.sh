#!/bin/bash
set -e

echo "=== A100 GRPO Training Pipeline ==="

REPO_URL="https://github.com/UtkarshSingh-09/RudraKernel.git"
REPO_DIR="RudraKernel"
PROJECT_SUBDIR="RudraKernel-src"
OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/tmp/rudra_unsloth}"
VENV_DIR="${TRAIN_VENV_DIR:-/tmp/rudra_train_venv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache}"

echo "=== Step 0: Prepare writable Python env ==="
python -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade pip

echo "=== Step 1: Install dependencies ==="
pip install -q --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -q unsloth "trl<0.20" datasets pyyaml transformers wandb

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
  --config training/configs/a100_grpo.yaml \
  --output-dir "$OUTPUT_DIR" \
  --no-wandb

echo "=== Step 5: Upload checkpoint ==="
if [ -n "$HF_TOKEN" ]; then
  huggingface-cli \
    upload ankit-choubey/Rudrakernel-unsloth \
    "$OUTPUT_DIR/final_model" . \
    --token "$HF_TOKEN"
else
  echo "Warning: HF_TOKEN not set. Skipping HF upload."
fi

echo "=== TRAINING COMPLETE ==="
echo "Checkpoint saved to: $OUTPUT_DIR/final_model"
