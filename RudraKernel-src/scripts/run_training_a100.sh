#!/bin/bash
set -euo pipefail

echo "=== A100 GRPO Training Pipeline ==="

PROJECT_SUBDIR="RudraKernel-src"
OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/tmp/rudra_unsloth}"
VENV_DIR="${TRAIN_VENV_DIR:-/tmp/rudra_train_venv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache}"
# HF Space containers run with UID lacking /etc/passwd entry; provide USER for getpass.
export USER="${USER:-hfuser}"
export HOME="${HOME:-/tmp/rudra_home}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
export HF_HOME="${HF_HOME:-/tmp/hf-home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/hf-home/transformers}"
mkdir -p "$HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME"

echo "=== Step 0: Prepare writable Python env ==="
python -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade pip

echo "=== Step 1: Install dependencies ==="
# Install a modern torch that satisfies xformers/unsloth; avoid torchvision/torchaudio conflicts.
pip install -q "torch>=2.10,<2.11"
pip install -q unsloth "trl<0.20" datasets pyyaml transformers wandb accelerate

echo "=== Step 2: Verify CUDA ==="
if ! python -c "import sys, torch; ok=torch.cuda.is_available(); print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {ok}'); print(f'Device: {torch.cuda.get_device_name(0) if ok else \"CPU\"}'); sys.exit(0 if ok else 2)"; then
  echo "ERROR: No GPU detected. Run this script on HF Space/Notebook with A100 hardware enabled."
  exit 1
fi

echo "=== Step 3: Locate project root ==="
if [ -f "pyproject.toml" ] && [ -d "training" ]; then
  # Running from repository root already (HF Docker /app layout)
  :
elif [ -d "${PROJECT_SUBDIR}" ] && [ -f "${PROJECT_SUBDIR}/pyproject.toml" ]; then
  cd "${PROJECT_SUBDIR}"
else
  echo "ERROR: Could not locate project root with pyproject.toml."
  echo "Current directory: $(pwd)"
  echo "Expected one of: ./, ./${PROJECT_SUBDIR}"
  exit 1
fi

# Avoid editable install in Space runtime because /app can be read-only.
# Running from repo root makes local packages importable via PYTHONPATH even when cwd changes.
PROJECT_ROOT="$(pwd)"
CONFIG_PATH="$PROJECT_ROOT/training/configs/a100_grpo.yaml"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=== Step 4: Run training ==="
python -m training.grpo_train_unsloth \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --no-wandb

echo "=== Step 5: Upload checkpoint ==="
if [ -n "${HF_TOKEN:-}" ] && [ -d "$OUTPUT_DIR/final_model" ] && command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli \
    upload ankit-choubey/Rudrakernel-unsloth \
    "$OUTPUT_DIR/final_model" . \
    --token "$HF_TOKEN"
else
  echo "Warning: HF upload skipped (missing token, model dir, or huggingface-cli)."
fi

echo "=== TRAINING COMPLETE ==="
echo "Checkpoint saved to: $OUTPUT_DIR/final_model"
