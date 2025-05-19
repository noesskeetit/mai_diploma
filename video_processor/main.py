import os
import json
import logging
from kafka import KafkaConsumer
from dotenv import load_dotenv
from video_preprocessor import process_videos_in_folder

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
)
logger = logging.getLogger("main")

# Подписываемся на топик с задачами на обработку видеопапок
consumer = KafkaConsumer(
    os.getenv('VIDEOPROCESSOR_INPUT_TOPIC'),
    group_id=os.getenv('KAFKA_GROUP_ID'),
    max_poll_interval_ms=1_200_000,
    bootstrap_servers=[os.getenv('KAFKA_BOOTSTRAP_SERVERS')],
    value_deserializer=lambda x: json.loads(x.decode())
)

logger.info("Video-processor started, waiting for messages…")

for msg in consumer:
    bucket = msg.value.get('bucket')
    folder = msg.value.get('folder')
    logger.info(f"Job received: bucket={bucket}, folder={folder}")
    try:
        process_videos_in_folder(bucket, folder)
    except Exception:
        logger.exception("Failed to process folder")
