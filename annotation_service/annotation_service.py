import os
import json
import base64
import logging
import requests
from kafka import KafkaConsumer, KafkaProducer
import boto3
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("annotation_service")

# Параметры Kafka
BOOT = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").split(",")
IN_TOPIC  = os.getenv("IMAGE_TASK_INPUT_TOPIC", "image-tasks")
OUT_TOPIC = os.getenv("DB_SAVER_INPUT_TOPIC", "db-saver-input")

consumer = KafkaConsumer(
    IN_TOPIC,
    bootstrap_servers=BOOT,
    group_id=os.getenv("KAFKA_GROUP_ID_ANNOTATION", "annotation_service"),
    max_poll_interval_ms=3_600_000,
    value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    enable_auto_commit=False,

)
producer = KafkaProducer(
    bootstrap_servers=BOOT,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

# Настройки S3 и OpenRouter
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
OR_URL   = os.getenv("OPENROUTER_API_URL")
OR_KEY   = os.getenv("OPENROUTER_API_KEY")
OR_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-image-preview")

def annotate_image(bucket: str, key: str) -> str:
    """
    1) Скачивает кадр из S3
    2) Кодирует в Base64
    3) Отправляет мультимодальный запрос в OpenRouter
    4) Возвращает текст-описание
    """
    obj = s3.get_object(Bucket=bucket, Key=key)
    img_bytes = obj["Body"].read()

    # Base64 Data URL
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    payload = {
        "model": OR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                                    Пожалуйста, максимально подробно опиши это изображение, учитывая любые возможные сценарии: 
                                    - Основные объекты и их назначение (слайды, окна приложений, диаграммы, таблицы, текстовые блоки и т. д.) 
                                    - Контекст и среду (рабочий стол, IDE, терминал, веб-страница, слайд презентации или учебный материал) 
                                    - Весь текст на экране (заголовки, подписи, фрагменты кода, формулы, маркеры списка)  
                                """
                    },
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ]
    }
    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    resp = requests.post(OR_URL, headers=headers, json=payload)
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"]

# Основной цикл: читаем кадры, аннотируем и отправляем дальше
for msg in consumer:
    job = msg.value
    logger.info(f"Received task: {job}")
    try:
        caption = annotate_image(job["bucket"], job["key"])
        logger.info(f"Got caption: {caption!r}")

        out_msg = {
            "bucket":          job["bucket"],       # ← передаём bucket
            "room_uuid":       job["room"],
            "image_key":       job["key"],
            "caption":         caption,
            "timestamp_ms":    job["timestamp_ms"]
        }
        producer.send(OUT_TOPIC, out_msg)
        producer.flush()
        consumer.commit()
        logger.info('!annotation_task_message commited in consumer!')
        logger.info(f"Sent to {OUT_TOPIC}: {out_msg}")

    except Exception:
        logger.exception("Annotation failed")
