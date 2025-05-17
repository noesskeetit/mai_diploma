# Базовый Python-образ
FROM python:3.9-slim

WORKDIR /app

# Устанавливаем системные зависимости для сборки
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем Python-библиотеки
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Устанавливаем curl для проверки здоровья
RUN apt update && apt install -y curl

# Копируем сервис
COPY model_service.py .

EXPOSE 8000

# Запуск FastAPI через Uvicorn
CMD ["uvicorn", "model_service:app", "--host", "0.0.0.0", "--port", "8000"]
