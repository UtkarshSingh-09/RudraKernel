FROM python:3.11-slim

USER root
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

# Install siege-env package (no GPU deps in setup.py)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# === SINGLE COMMAND: let unsloth resolve ALL deps (torch, transformers, etc.) ===
# This is the official recommended approach from https://docs.unsloth.ai
RUN pip install --no-cache-dir unsloth

# Extra training deps not covered by unsloth
RUN pip install --no-cache-dir \
    "trl>=0.7.0" \
    "wandb>=0.15.0" \
    "huggingface_hub>=0.20.0" \
    "pyyaml>=6.0.0"

ENV PYTHONUNBUFFERED=1
ENV UNSLOTH_COMPILE_DISABLE=1
EXPOSE 8000

CMD ["python", "-u", "training/run_training_with_status.py"]
