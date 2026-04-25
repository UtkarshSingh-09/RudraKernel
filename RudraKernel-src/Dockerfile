FROM unsloth/unsloth:latest

USER root
WORKDIR /app

# Copy project
COPY . /app

# Install the siege-env package
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir "huggingface_hub>=0.20.0" "pyyaml>=6.0.0"

# Force Python to flush stdout/stderr immediately (critical for HF Spaces logs)
ENV PYTHONUNBUFFERED=1
# Disable unsloth compiled cache (causes "already registered" errors in Docker)
ENV UNSLOTH_COMPILE_DISABLE=1
ENV TRITON_CACHE_DIR=/tmp/triton_cache

# HF Spaces needs a server on port 8000 — the wrapper runs training + status page
EXPOSE 8000

# Clear the base image's ENTRYPOINT (it starts Jupyter by default)
ENTRYPOINT []
CMD ["python", "-u", "training/run_training_with_status.py"]
