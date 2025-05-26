import logging
import torch
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor
import transformers

from transformers.modeling_utils import PreTrainedModel
PreTrainedModel.initialize_weights = lambda self, *args, **kwargs: None

transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_service")
device = os.getenv('CLIP_DEVICE', "cuda" if torch.cuda.is_available() else "cpu")

logger.info(f'CLIP_DEVICE IS {device}')

app = FastAPI(title="CLIP Encoding Service")

# Загружаем CLIP-модель и процессор при старте
try:
    processor = AutoProcessor.from_pretrained(
        "jinaai/jina-clip-v2", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "jinaai/jina-clip-v2", trust_remote_code=True
    ).to(device)
    model.eval()
    logger.info("CLIP model and processor loaded")
except Exception:
    logger.exception("Failed to load CLIP model")
    raise

# Pydantic-схема для ответа
class EmbeddingResponse(BaseModel):
    embedding: list[float]

# Pydantic-схема для запроса текста
class TextRequest(BaseModel):
    text: str

@app.get("/health")
async def health():
    """health-check"""
    return {"status": "ok"}

@app.post("/encode_image", response_model=EmbeddingResponse)
async def encode_image(file: UploadFile = File(...)):
    """
    Принимает изображение (JPEG/PNG) в multipart/form-data,
    возвращает L2-нормализованный эмбеддинг.
    """
    try:
        from PIL import Image
        import io

        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        inputs = processor(images=img, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
            embeddings = embeddings.to(torch.float32)

        emb_list = embeddings[0].cpu().numpy().tolist()
        return {"embedding": emb_list}

    except Exception:
        logger.exception("Error in /encode_image")
        raise HTTPException(status_code=500, detail="Failed to encode image")

@app.post("/encode_text", response_model=EmbeddingResponse)
async def encode_text(request: TextRequest):
    """
    Принимает JSON {"text": "..."} и возвращает L2-нормализованный эмбеддинг текста.
    """
    try:
        inputs = processor(text=request.text, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
            embeddings = embeddings.to(torch.float32)

        emb_list = embeddings[0].cpu().numpy().tolist()
        return {"embedding": emb_list}

    except Exception:
        logger.exception("Error in /encode_text")
        raise HTTPException(status_code=500, detail="Failed to encode text")
