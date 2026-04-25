FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY siege_env /app/siege_env
COPY openenv.yaml /app/openenv.yaml

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "siege_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
