import os
import io
import tempfile
import logging
import boto3
import re
from PIL import Image
from dotenv import load_dotenv

from clip_client import ClipClient
from process_video_clip import process_video_clip

load_dotenv()
logger = logging.getLogger("video_preprocessor")

def parse_video_filename(filename):
    """
    Парсит имя файла room.video_id.user_id.start_demo_time.ogg
    Возвращает: room, video_id, user_id, start_demo_time
    """
    base = os.path.basename(filename)
    m = re.match(r"^([^.]+)\.([^.]+)\.([^.]+)\.([^.]+)\.ogg$", base)
    if not m:
        raise ValueError(f"Filename does not match pattern: {filename}")
    return m.group(1), m.group(2), m.group(3), m.group(4)

# S3-клиент
s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('S3_URL'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
)

clip = ClipClient()

def upload_image(bucket: str, key: str, img_array):
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format='JPEG')
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf, ContentType='image/jpeg')

def process_videos_in_folder(bucket: str, root_folder: str):
    """
    Для каждого raw_video/*.ogg в {root_folder}/raw_video/:
      1. скачиваем s3.download_file (стриминг)
      2. делаем дедупликацию кадров
      3. заливаем JPEG-кадры в {root_folder}/preprocessed_video/{room.video_id.user_id.start_demo_time}/
         имена кадров: room.video_id.user_id.start_demo_time.XX_frame_start_time-frame_end_time.jpg
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

        logger.debug(f"Found raw video: s3://{bucket}/{key}")

        ext = os.path.splitext(key)[1]
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            # 1) Download via streaming API
            logger.debug(f"Downloading s3://{bucket}/{key} → {tmp_path}")
            s3.download_file(bucket, key, tmp_path)

            # 2) Deduplication
            frame_ids, frames, timestamps = process_video_clip(tmp_path, clip.encode_image)
            if not timestamps:
                logger.warning(f"No unique frames found for {key}")
                continue

            # 3) Формируем структуру хранения и имена файлов
            room, video_id, user_id, start_demo_time = parse_video_filename(key)
            folder_name = f"{room}.{video_id}.{user_id}.{start_demo_time}"
            dest_pref = f"{root_folder}/preprocessed_video/{folder_name}/"

            n = len(timestamps)
            end_times = [timestamps[i+1] - 1 if i + 1 < n else timestamps[i] for i in range(n)]

            for idx, (start_ts, end_ts, img) in enumerate(zip(timestamps, end_times, frames)):
                filename = (
                    f"{room}.{video_id}.{user_id}.{start_demo_time}."
                    f"{idx:02d}_frame_{start_ts}-{end_ts}.jpg"
                )
                upload_image(bucket, dest_pref + filename, img)

            logger.info(f"Successfully processed {key}: {len(frames)} images uploaded")

        except Exception:
            logger.exception(f"Error processing s3://{bucket}/{key}")

        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning(f"Failed to delete temp file {tmp_path}")
