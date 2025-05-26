FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

RUN apt update && apt install -y curl

COPY reranker_service.py .
EXPOSE 8002
CMD ["uvicorn", "reranker_service:app", "--host", "0.0.0.0", "--port", "8002"]
