FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[dev]" gradio>=4.39.0 pandas

EXPOSE 7860

CMD ["python", "-m", "frontend.app"]
