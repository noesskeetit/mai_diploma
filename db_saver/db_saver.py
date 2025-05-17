import os
import json
import uuid
import logging
import requests
from kafka import KafkaConsumer
from opensearchpy import OpenSearch
from pymilvus import MilvusClient, utility, DataType
from pymilvus.orm.connections import Connections
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_saver")

# Конфигурация из .env
MILVUS_URI            = os.getenv("MILVUS_URI")
COL_NAME              = os.getenv("MILVUS_COLLECTION_NAME", "images")
EMB_DIM               = int(os.getenv("EMB_DIM", "1024"))
OS_HOST               = os.getenv("OPENSEARCH_HOST")
OS_PORT               = int(os.getenv("OPENSEARCH_PORT", "9200"))
OS_USER               = os.getenv("OPENSEARCH_USER")
OS_PASS               = os.getenv("OPENSEARCH_PASS")
OS_INDEX              = os.getenv("OPENSEARCH_INDEX_NAME", "image_captions")
EMB_URL               = os.getenv("EMB_URL", "http://model-service:8000/encode_text")
BOOTSTRAP             = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").split(",")
IN_TOPIC              = os.getenv("DB_SAVER_INPUT_TOPIC", "db-saver-input")
GROUP_ID              = os.getenv("KAFKA_GROUP_ID_DB_SAVER", "db-saver-group")

# Подключаемся к Milvus
milvus = MilvusClient(MILVUS_URI)
conn_name = Connections().list_connections()[-1][0]
if not utility.has_collection(COL_NAME, using=conn_name):
    # Создаём коллекцию, если нет
    milvus.create_collection(
        collection_name=COL_NAME,
        dimension=EMB_DIM,
        primary_field_name="id",
        id_type=DataType.VARCHAR,
        vector_field_name="vector",
        metric_type="COSINE",
        auto_id=False,
        max_length=65535
    )
    logger.info(f"Created Milvus collection {COL_NAME}")

# Подключаемся к OpenSearch
os_client = OpenSearch(
    hosts=[{"host": OS_HOST, "port": OS_PORT}],
    http_auth=(OS_USER, OS_PASS),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)
# Создаём индекс при отсутствии
if not os_client.indices.exists(index=OS_INDEX):
    os_client.indices.create(
        index=OS_INDEX,
        body={"settings": {"index": {"number_of_shards": 4}}}
    )
    logger.info(f"Created OpenSearch index {OS_INDEX}")

# Kafka Consumer
consumer = KafkaConsumer(
    IN_TOPIC,
    bootstrap_servers=BOOTSTRAP,
    group_id=GROUP_ID,
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

def get_text_embedding(text: str):
    """
    Запрашивает /encode_text в clip_service и возвращает эмбеддинг.
    """
    resp = requests.post(
        EMB_URL,
        json={"text": text},
        headers={"Content-Type": "application/json"}
    )
    resp.raise_for_status()
    return resp.json().get("embedding")

# Основной цикл: сохраняем текст и вектор
for msg in consumer:
    job = msg.value
    logger.info(f"Received annotation: {job}")
    caption     = job["caption"]
    room_uuid   = job["room_uuid"]
    timestamp   = job.get("timestamp_ms")
    image_key   = job.get("image_key")

    try:
        # 1) Получаем эмбеддинг текста
        vec = get_text_embedding(caption)
        doc_id = str(uuid.uuid4())

        # 2) Индексируем описание в OpenSearch
        os_client.index(
            index=OS_INDEX,
            id=doc_id,
            body={
                "id":           doc_id,
                "room_uuid":    room_uuid,
                "caption":      caption,
                "timestamp_ms": timestamp,
                "image_key":    image_key
            },
            refresh=True
        )

        # 3) Сохраняем вектор в Milvus
        milvus.insert(
            collection_name=COL_NAME,
            data=[{"id": doc_id, "vector": vec}]
        )

        logger.info(f"Stored caption {doc_id}")

    except Exception:
        logger.exception("DB saver error")
