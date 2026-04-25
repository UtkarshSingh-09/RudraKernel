#!/bin/bash
# Deploy to the TRAINING HF Space (A100).
# This script temporarily swaps Dockerfile → Dockerfile.training,
# pushes via git subtree, then restores the original.
#
# Usage:
#   bash scripts/deploy_training_space.sh <remote-name>
#
# Example:
#   git remote add training https://huggingface.co/spaces/YOUR_ACCOUNT/siege-training
#   bash scripts/deploy_training_space.sh training

set -euo pipefail

REMOTE="${1:-training}"
BRANCH="${2:-main}"
SRC_DIR="RudraKernel-src"

echo "=== Deploying TRAINING mode to HF Space: $REMOTE/$BRANCH ==="

# 1. Backup current Dockerfile
cp "$SRC_DIR/Dockerfile" "$SRC_DIR/Dockerfile.api.bak"

# 2. Swap in the training Dockerfile
cp "$SRC_DIR/Dockerfile.training" "$SRC_DIR/Dockerfile"

# 3. Commit the swap
git add "$SRC_DIR/Dockerfile"
git commit -m "temp: swap Dockerfile for training Space" --allow-empty

# 4. Push to training Space via subtree
echo "Pushing to $REMOTE $BRANCH ..."
git subtree push --prefix "$SRC_DIR" "$REMOTE" "$BRANCH"

# 5. Restore original Dockerfile
cp "$SRC_DIR/Dockerfile.api.bak" "$SRC_DIR/Dockerfile"
rm "$SRC_DIR/Dockerfile.api.bak"
git add "$SRC_DIR/Dockerfile"
git commit -m "restore: revert Dockerfile to API mode"

echo ""
echo "=== ✓ Training Space deployed ==="
echo "Set these in your Space settings:"
echo "  - Hardware: A100"
echo "  - Secret: HF_TOKEN = <your token>"
echo ""
echo "The trained model will be pushed to:"
echo "  https://huggingface.co/ankit-choubey/siege-grpo-lora"
