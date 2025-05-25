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

# Retrieval parameters
VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_TOP_K", 5))
CLASSIC_SEARCH_K = int(os.getenv("CLASSIC_SEARCH_TOP_K", 5))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", 5))
VARIATIONS_K = int(os.getenv("QUERY_VARIATIONS_K", 3))
HYPOTHESES_K = int(os.getenv("HYPOTHETICAL_ANSWERS_K", 3))

# Initialize Milvus and OpenSearch
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

app = FastAPI(title="Retriever Service with Query Variations + Hypothetical Answers")

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

# Helpers

def call_llm(messages: list[dict]) -> str:
    payload = {"model": OR_MODEL, "messages": messages}
    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    resp = requests.post(OR_URL, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate_query_variations(query: str, k: int) -> list[str]:
    messages = [
        {"role": "system", "content": "Paraphrase the user question to improve retrieval."},
        {"role": "user", "content": f"Provide {k} alternative formulations of the question: '{query}'"}
    ]
    content = call_llm(messages)
    lines = [line.strip("- ") for line in content.splitlines() if line.strip()]
    return lines[:k]


def generate_hypothetical_answers(question: str, k: int) -> list[str]:
    messages = [
        {"role": "system", "content": "Generate plausible answers to help retrieve relevant context."},
        {"role": "user", "content": f"Provide {k} different plausible answers for the question: '{question}'"}
    ]
    content = call_llm(messages)
    lines = [line.strip("- ") for line in content.splitlines() if line.strip()]
    return lines[:k]


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
        output_fields=["bucket","image_key","room_uuid","caption","timestamp_ms"]
    )
    hits = []
    for hit in results[0]:
        e = hit.entity
        hits.append({
            "id": str(hit.id), "score": float(hit.score),
            "bucket": e.bucket, "image_key": e.image_key,
            "room_uuid": e.room_uuid, "caption": e.caption,
            "timestamp_ms": int(e.timestamp_ms), "source": "milvus"
        })
    return hits


def search_opensearch(query: str, k: int) -> list[dict]:
    body = {"size": k, "query": {"multi_match": {"query": query, "fields": ["caption"]}}}
    res = os_client.search(index=OPENSEARCH_INDEX, body=body)
    hits = []
    for hit in res.get("hits", {}).get("hits", []):
        s = hit.get("_source", {})
        hits.append({
            "id": hit.get("_id"), "score": float(hit.get("_score", 0)),
            "bucket": s.get("bucket"), "image_key": s.get("image_key"),
            "room_uuid": s.get("room_uuid"), "caption": s.get("caption"),
            "timestamp_ms": int(s.get("timestamp_ms", 0)), "source": "opensearch"
        })
    return hits


def rerank_results(query: str, contexts: list[dict]) -> list[dict]:
    resp = requests.post(RERANKER_URL, json={"query": query, "contexts": contexts})
    resp.raise_for_status()
    return resp.json().get("ranked", [])

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    try:
        # 1. Generate top-k query variations
        variations = generate_query_variations(req.query, VARIATIONS_K)

        # 2. From each variation, generate hypothetical answers
        hypotheticals = []
        for var in variations:
            hypotheticals.extend(generate_hypothetical_answers(var, HYPOTHESES_K))

        # 3. Prepare retrieval keys: original + all hypotheticals
        keys = [req.query] + hypotheticals

        # 4. Retrieve contexts for each key
        all_hits = []
        for key in keys:
            vec = get_embedding(key)
            all_hits.extend(search_milvus(vec, VECTOR_SEARCH_K))
            all_hits.extend(search_opensearch(key, CLASSIC_SEARCH_K))

        # 5. Deduplicate by id, keeping highest score
        unique = {}
        for hit in all_hits:
            if hit['id'] not in unique or hit['score'] > unique[hit['id']]['score']:
                unique[hit['id']] = hit
        candidates = list(unique.values())

        # 6. Rerank candidates by original query
        ranked = rerank_results(req.query, candidates)
        topk = ranked[:MAX_RESULTS]

        # 7. Generate final answer
        text_ctx = "\n".join([f"{i+1}. {ctx['caption']}" for i, ctx in enumerate(topk)])
        final_messages = [
            {"role": "system", "content": "Answer the question using provided contexts."},
            {"role": "user", "content": f"Using contexts:\n{text_ctx}\nQuestion: {req.query}\nAnswer:"}
        ]
        final_answer = call_llm(final_messages)

        return {"final_answer": final_answer, "contexts": topk}
    except Exception as e:
        logger.exception("Error in multi-query RAG retrieval")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
