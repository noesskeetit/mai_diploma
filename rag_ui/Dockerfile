FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY rag_ui_service.py .
EXPOSE 7860

CMD ["python", "rag_ui_service.py"]
