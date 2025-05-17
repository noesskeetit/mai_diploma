import os
import io
import tempfile
import logging
import boto3
import re
import json
from kafka import KafkaProducer
from PIL import Image
from dotenv import load_dotenv

from clip_client import ClipClient
from process_video_clip import process_video_clip

load_dotenv()
logger = logging.getLogger("video_preprocessor")

# S3-клиент для чтения/записи
s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('S3_URL'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
)

# Kafka Producer для отправки задач на аннотацию
producer = KafkaProducer(
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").split(","),
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)
IMAGE_TASK_TOPIC = os.getenv("IMAGE_TASK_TOPIC", "image-tasks")

clip = ClipClient()

def parse_video_filename(filename):
    """
    Извлекает room, video_id, user_id, start_demo_time из имени:
      room.video_id.user_id.start_demo_time.ogg
    """
    base = os.path.basename(filename)
    m = re.match(r"^([^.]+)\.([^.]+)\.([^.]+)\.([^.]+)\.ogg$", base)
    if not m:
        raise ValueError(f"Filename does not match pattern: {filename}")
    return m.group(1), m.group(2), m.group(3), m.group(4)

def upload_image(bucket: str, key: str, img_array):
    """
    Сохраняет RGB-массив как JPEG в S3 по ключу key.
    """
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format='JPEG')
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf, ContentType='image/jpeg')

def process_videos_in_folder(bucket: str, root_folder: str):
    """
    Для каждого .ogg в {root_folder}/raw_video/:
      1) скачиваем в temp
      2) извлекаем уникальные кадры
      3) заливаем JPEG в {root_folder}/preprocessed_video/ и
         публикуем задачу в Kafka для аннотации
    """
    raw_pref = f"{root_folder}/raw_video/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=raw_pref)
    if not resp.get('Contents'):
        logger.warning(f"No raw_video under s3://{bucket}/{raw_pref}")
        return

    for obj in resp['Contents']:
        key = obj['Key']
        if not key.lower().endswith('.ogg'):
            continue

        # временный файл для видео
        ext = os.path.splitext(key)[1]
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            # скачиваем
            logger.debug(f"Downloading s3://{bucket}/{key} → {tmp_path}")
            s3.download_file(bucket, key, tmp_path)

            # выделяем кадры
            _, frames, timestamps = process_video_clip(tmp_path, clip.encode_image)
            if not timestamps:
                logger.warning(f"No unique frames for {key}")
                continue

            # парсим имя видео
            room, video_id, user_id, start_demo_time = parse_video_filename(key)
            folder_name = f"{room}.{video_id}.{user_id}.{start_demo_time}"
            dest_pref = f"{root_folder}/preprocessed_video/{folder_name}/"

            # сохраняем кадры и публикуем задачи
            for idx, (start_ts, img) in enumerate(zip(timestamps, frames)):
                filename = (
                    f"{room}.{video_id}.{user_id}.{start_demo_time}."
                    f"{idx:02d}.jpg"
                )
                s3_key = dest_pref + filename

                # 1) загрузка в S3
                upload_image(bucket, s3_key, img)
                logger.info(f"Uploaded {s3_key}")

                # 2) отправка задачи в Kafka
                message = {
                    "bucket": bucket,
                    "key": s3_key,
                    "room": room,
                    "video_id": video_id,
                    "user_id": user_id,
                    "start_demo_time": start_demo_time,
                    "frame_id": idx,
                    "timestamp_ms": start_ts,
                }
                producer.send(IMAGE_TASK_TOPIC, message)
                logger.info(f"Sent task to {IMAGE_TASK_TOPIC}: {message}")

            producer.flush()
            logger.info(f"Flushed Kafka for {key}")

        except Exception:
            logger.exception(f"Error processing {key}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning(f"Failed to delete temp file {tmp_path}")
