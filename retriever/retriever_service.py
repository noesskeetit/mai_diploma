import os
import logging
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import connections, Collection
from opensearchpy import OpenSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger("retriever-service")

# Environment settings
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION_NAME")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX_NAME")
EMB_URL = os.getenv("EMB_URL", "http://model-service:8000/encode_text")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8002/rerank")

# OpenRouter settings
OR_KEY = os.getenv("OPENROUTER_API_KEY")
OR_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OR_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")

# Search parameters (fallbacks)
VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_TOP_K", 5))
CLASSIC_SEARCH_K = int(os.getenv("CLASSIC_SEARCH_TOP_K", 5))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", 5))

# Initialize Milvus connection and collection
connections.connect(alias="default", uri=MILVUS_URI)
collection = Collection(name=MILVUS_COLLECTION, using="default")

# Initialize OpenSearch client
os_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False
)

# FastAPI app
app = FastAPI(title="Retriever Service")

# Pydantic models
class RetrieveRequest(BaseModel):
    query: str

class Context(BaseModel):
    id: str
    score: float
    bucket: str
    image_key: str
    room_uuid: str
    caption: str
    timestamp_ms: int
    source: str

class RetrieveResponse(BaseModel):
    final_answer: str
    contexts: list[Context]

# Helper functions
def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        EMB_URL,
        headers={"Content-Type": "application/json"},
        json={"text": text}
    )
    resp.raise_for_status()
    return resp.json().get("embedding", [])


def search_milvus(vector: list[float], k: int) -> list[dict]:
    results = collection.search(
        data=[vector],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=k,
        output_fields=["bucket", "image_key", "room_uuid", "caption", "timestamp_ms"]
    )
    hits = []
    for hit in results[0]:
        entity = hit.entity
        hits.append({
            "id": str(hit.id),
            "score": float(hit.score),
            "bucket": entity.bucket,
            "image_key": entity.image_key,
            "room_uuid": entity.room_uuid,
            "caption": entity.caption,
            "timestamp_ms": int(entity.timestamp_ms),
            "source": "milvus"
        })
    return hits


def search_opensearch(query: str, k: int) -> list[dict]:
    body = {
        "size": k,
        "query": {"multi_match": {"query": query, "fields": ["caption"]}}
    }
    res = os_client.search(index=OPENSEARCH_INDEX, body=body)
    hits = []
    for hit in res.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        hits.append({
            "id": hit.get("_id"),
            "score": float(hit.get("_score", 0)),
            "bucket": src.get("bucket"),
            "image_key": src.get("image_key"),
            "room_uuid": src.get("room_uuid"),
            "caption": src.get("caption"),
            "timestamp_ms": int(src.get("timestamp_ms", 0)),
            "source": "opensearch"
        })
    return hits


def rerank_results(query: str, contexts: list[dict]) -> list[dict]:
    resp = requests.post(
        RERANKER_URL,
        headers={"Content-Type": "application/json"},
        json={"query": query, "contexts": contexts}
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("ranked", [])


def generate_answer(query: str, contexts: list[dict]) -> str:
    # Собираем контексты в один текст
    text_ctx = "\n".join([f"{i+1}. {ctx['caption']}" for i, ctx in enumerate(contexts)])
    messages = [
        {"role": "system", "content": "Вы помощник, отвечающий на вопросы на основе контекстов."},
        {"role": "user", "content": (
            f"Используя следующие контексты:\n{text_ctx}\n\nВопрос: {query}\nОтвет:"
        )}
    ]
    payload = {
        "model": OR_MODEL,
        "messages": messages
    }
    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json"
    }
    resp = requests.post(OR_URL, headers=headers, json=payload)
    resp.raise_for_status()
    # Ответ в choices[0].message.content
    return resp.json()["choices"][0]["message"]["content"].strip()

# Healthcheck
@app.get("/health")
async def health():
    return {"status": "ok"}

# Main retrieval endpoint
@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    try:
        # 1. Embed query
        vector = get_embedding(req.query)

        # 2. Retrieve candidates
        milvus_hits = search_milvus(vector, VECTOR_SEARCH_K)
        os_hits = search_opensearch(req.query, CLASSIC_SEARCH_K)
        all_hits = milvus_hits + os_hits

        # 3. Deduplicate by id, keep highest score
        unique = {}
        for hit in all_hits:
            if hit['id'] not in unique or hit['score'] > unique[hit['id']]['score']:
                unique[hit['id']] = hit
        candidates = list(unique.values())

        # 4. Rerank
        ranked = rerank_results(req.query, candidates)

        # 5. Trim to max results
        topk = ranked[:MAX_RESULTS]

        # 6. Generate final answer
        final_answer = generate_answer(req.query, topk)

        return {"final_answer": final_answer, "contexts": topk}

    except Exception as e:
        logger.exception("Error in retrieve")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
