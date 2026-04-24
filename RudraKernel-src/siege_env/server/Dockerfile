FROM python:3.10-slim

WORKDIR /app

COPY siege_env/server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY siege_env /app/siege_env

EXPOSE 8000

CMD ["uvicorn", "siege_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]