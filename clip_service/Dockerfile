FROM python:3.9-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

RUN apt update \
 && apt install -y curl

COPY model_service.py .

EXPOSE 8000
CMD ["uvicorn", "model_service:app", "--host", "0.0.0.0", "--port", "8000"]
