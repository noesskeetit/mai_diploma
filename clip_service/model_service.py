import logging
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_service")

app = FastAPI(title="CLIP Encoding Service")

# Загружаем CLIP-модель и процессор при старте
try:
    processor = AutoProcessor.from_pretrained(
        "jinaai/jina-clip-v2", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "jinaai/jina-clip-v2", trust_remote_code=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
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
    """
    Простой healthcheck для Kubernetes/других оркестраторов.
    """
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

        # Читаем байты и конвертим в RGB
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        # Пропускаем через процессор и модель
        inputs = processor(images=img, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)

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
        # Токенизируем и получаем фичи
        inputs = processor(text=request.text, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)

        emb_list = embeddings[0].cpu().numpy().tolist()
        return {"embedding": emb_list}

    except Exception:
        logger.exception("Error in /encode_text")
        raise HTTPException(status_code=500, detail="Failed to encode text")
