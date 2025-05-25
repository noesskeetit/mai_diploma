import os
import json
import logging
import time
import random
from kafka import KafkaConsumer
from opensearchpy import OpenSearch
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType
)
from dotenv import load_dotenv

# ==== Load environment ====
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_saver")

# ==== Retry-enabled HTTP session ====
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[502, 503, 504],
    allowed_methods=["POST", "GET"],
)
adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)

# ==== Configuration from .env ====
MILVUS_URI       = os.getenv("MILVUS_URI")
COL_NAME         = os.getenv("MILVUS_COLLECTION_NAME", "images")
EMB_DIM          = int(os.getenv("EMB_DIM", "1024"))
N_LIST           = int(os.getenv("N_LIST", 128))
OS_HOST          = os.getenv("OPENSEARCH_HOST")
OS_PORT          = int(os.getenv("OPENSEARCH_PORT", "9200"))
OS_USER          = os.getenv("OPENSEARCH_USER")
OS_PASS          = os.getenv("OPENSEARCH_PASS")
OS_INDEX         = os.getenv("OPENSEARCH_INDEX_NAME", "image_captions")
EMB_URL          = os.getenv("EMB_URL", "http://model-service:8000/encode_text")
BOOTSTRAP        = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").split(",")
IN_TOPIC         = os.getenv("DB_SAVER_INPUT_TOPIC", "db-saver-input")
GROUP_ID         = os.getenv("KAFKA_GROUP_ID_DB_SAVER", "db-saver-group")

# ==== Connect to Milvus ====
connections.connect(alias="default", uri=MILVUS_URI)
if not utility.has_collection(COL_NAME, using="default"):
    fields = [
        FieldSchema(name="id",        dtype=DataType.VARCHAR, is_primary=True, max_length=65535),
        FieldSchema(name="vector",    dtype=DataType.FLOAT_VECTOR, dim=EMB_DIM, metric_type="COSINE"),
        FieldSchema(name="bucket",    dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="image_key", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="room_uuid", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="caption",   dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="timestamp_ms", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="Image captions with metadata and embeddings")
    Collection(name=COL_NAME, schema=schema, using="default")
    logger.info(f"Created Milvus collection `{COL_NAME}` with extended schema")
collection = Collection(name=COL_NAME, using="default")
try:
    ivf_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": N_LIST}}
    collection.create_index(field_name="vector", index_params=ivf_params)
    logger.info(f"Created IVF_FLAT index on `{COL_NAME}` (nlist={N_LIST})")
except Exception as e:
    logger.debug(f"IVF_FLAT index creation skipped: {e}")
try:
    collection.load()
    logger.info(f"Milvus collection `{COL_NAME}` loaded into memory")
except Exception as e:
    logger.warning(f"Could not load Milvus collection: {e}")

# ==== Connect to OpenSearch ====
os_client = OpenSearch(
    hosts=[{"host": OS_HOST, "port": OS_PORT}],
    http_auth=(OS_USER, OS_PASS),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)
if not os_client.indices.exists(index=OS_INDEX):
    os_client.indices.create(
        index=OS_INDEX,
        body={"settings": {"index": {"number_of_shards": 4}}}
    )
    logger.info(f"Created OpenSearch index `{OS_INDEX}`")

# ==== Kafka Consumer ====
consumer = KafkaConsumer(
    IN_TOPIC,
    bootstrap_servers=BOOTSTRAP,
    group_id=GROUP_ID,
    max_poll_interval_ms=3_600_000,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    enable_auto_commit=False
)

# ==== Embedding helper with backoff ====
def get_text_embedding(text: str):
    resp = session.post(
        EMB_URL,
        json={"text": text},
        headers={"Content-Type": "application/json"},
        timeout=(60, 60),
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def get_text_embedding_with_backoff(text: str, max_tries: int = 5):
    for i in range(max_tries):
        try:
            return get_text_embedding(text)
        except Exception as e:
            wait = (2 ** i) + random.random()
            logger.warning(f"Embedding failed (try {i+1}/{max_tries}): {e}, retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("Failed to get embedding after retries")

# ==== Main loop ====
logger.info("DB saver started, waiting for annotations…")
for msg in consumer:
    job = msg.value
    logger.info(f"Received annotation: {job}")
    bucket    = job.get("bucket")
    key       = job.get("image_key")
    room_uuid = job.get("room_uuid")
    caption   = job.get("caption")
    timestamp = job.get("timestamp_ms")
    doc_id    = f"{bucket}/{key}"

    try:
        vec = get_text_embedding_with_backoff(caption)
        logger.debug("Successfully obtained embedding vector")

        meta = {"bucket": bucket, "image_key": key, "room_uuid": room_uuid,
                "caption": caption, "timestamp_ms": timestamp}

        os_client.index(index=OS_INDEX, id=doc_id, body=meta, refresh=True)
        record = {"id": doc_id, "vector": vec, **meta}
        collection.insert([record])
        consumer.commit()
        logger.info(f"Stored document `{doc_id}` in both OpenSearch and Milvus")
    except Exception:
        logger.exception("DB saver error")
