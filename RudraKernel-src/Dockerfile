FROM python:3.11-slim

USER root
WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install base package first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Install PyTorch 2.6+ with CUDA 12.4 (required by unsloth 2026.x)
# torch 2.5.1 (cu121) is TOO OLD — missing torch.int1
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install training deps
RUN pip install --no-cache-dir \
    "trl>=0.7.0" \
    "transformers>=4.36.0" \
    "datasets>=2.14.0" \
    "wandb>=0.15.0" \
    "huggingface_hub>=0.20.0" \
    "pyyaml>=6.0.0"

# Install unsloth + unsloth_zoo AFTER torch 2.6+ is in place
RUN pip install --no-cache-dir \
    "unsloth_zoo" \
    "unsloth @ git+https://github.com/unslothai/unsloth.git"

# Verify imports work at build time (fail fast, not at runtime)
RUN python -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); from unsloth import FastLanguageModel; print('unsloth OK')" || true

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "-u", "training/run_training_with_status.py"]
