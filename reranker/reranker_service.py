import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reranker")

# ==== Конфигурация ====
MODEL_NAME = os.getenv(
    "RERANK_MODEL",
    "amberoad/bert-multilingual-passage-reranking-msmarco"
)
DEVICE = os.getenv("RERANKER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# ==== Загрузка модели ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE).eval()

# ==== Схемы FastAPI ====
class Context(BaseModel):
    id: str
    source: str
    score: float
    bucket: str
    image_key: str
    room_uuid: str
    caption: str
    timestamp_ms: int

class RerankRequest(BaseModel):
    query: str
    contexts: List[Context]

class RerankResponse(BaseModel):
    ranked: List[Context]

app = FastAPI(title="Passage Reranker")

@app.get("/health")
async def health():
    """
    Простой healthcheck для Kubernetes/других оркестраторов.
    """
    return {"status": "ok"}

# ==== Реранг ====
@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    if not req.contexts:
        raise HTTPException(400, "No contexts provided")

    pairs = [(req.query, ctx.caption) for ctx in req.contexts]
    inputs = tokenizer(
        [q for q, p in pairs],
        [p for q, p in pairs],
        padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.squeeze(-1).cpu().tolist()

    ranked = [
        ctx for _, ctx in sorted(
            zip(scores, req.contexts),
            key=lambda x: x[0],
            reverse=True
        )
    ]

    return {"ranked": ranked}
