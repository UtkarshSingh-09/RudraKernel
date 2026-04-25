FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "siege_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
