FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Устанавливаем curl для проверки здоровья
RUN apt update && apt install -y curl

COPY retriever_service.py .
EXPOSE 8001
CMD ["uvicorn", "retriever_service:app", "--host", "0.0.0.0", "--port", "8001"]
