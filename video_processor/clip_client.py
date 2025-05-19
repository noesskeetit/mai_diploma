import os
import io
import logging
import requests
import torch
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
logger = logging.getLogger("clip_client")

class ClipClient:
    """
    HTTP-клиент для CLIP-сервиса: /encode_image
    """
    def __init__(self, base_url: str = None):
        base = base_url or os.getenv('CLIP_SERVICE_URL', 'http://model-service:8000')
        self.base_url = base.rstrip('/')

    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        buf = io.BytesIO()
        pil_image.save(buf, format='JPEG')
        buf.seek(0)

        files = {'file': ('image.jpg', buf, 'image/jpeg')}
        resp = requests.post(f"{self.base_url}/encode_image", files=files)
        resp.raise_for_status()

        data = resp.json().get('embedding')
        return torch.tensor(data)
