FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY app.py .
COPY model_bundle.pkl .

ENV MODEL_PATH=/app/model_bundle.pkl

EXPOSE 8081

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8081"]