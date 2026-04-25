FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install base package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Install PyTorch with CUDA 12.1 support (for A100)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
RUN pip install --no-cache-dir \
    "trl>=0.7.0" \
    "transformers>=4.36.0" \
    "datasets>=2.14.0" \
    "wandb>=0.15.0" \
    "huggingface_hub>=0.20.0" \
    "pyyaml>=6.0.0"

# Install Unsloth + unsloth_zoo (after torch to avoid version conflicts)
RUN pip install --no-cache-dir \
    "unsloth_zoo" \
    "unsloth @ git+https://github.com/unslothai/unsloth.git"

# Force Python to flush stdout/stderr immediately (critical for HF Spaces logs)
ENV PYTHONUNBUFFERED=1

# HF Spaces needs a server on port 8000 — the wrapper runs training + status page
EXPOSE 8000

CMD ["python", "-u", "training/run_training_with_status.py"]
