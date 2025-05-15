import os
import cv2
import logging
from PIL import Image
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()
STEP_MS = int(os.getenv('PROCESSING_TIMESTEP', '1000'))
THRESH = float(os.getenv('PROCESSING_SIMILARITY_THRESHOLD', '0.85'))

logger = logging.getLogger("process_video_clip")

def process_video_clip(video_path: str, encoder_fn):
    """
    Читает видео (OpenCV), берёт каждый кадр с шагом STEP_MS (мс),
    кодирует через encoder_fn (возвращает torch.Tensor), и удаляет
    «похожие» кадры по косинусному сходству.
    Возвращает:
      frame_ids: List[str],
      frames: List[np.ndarray],  # RGB
      timestamps: List[int]      # миллисекунды от начала
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video {video_path}")
        raise RuntimeError(f"Cannot open {video_path}")

    prev_emb = None
    next_ts = 0
    frame_ids, frames, timestamps = [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if t_ms < next_ts:
            continue
        next_ts += STEP_MS

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        try:
            emb = encoder_fn(pil)
            emb = emb.cpu()
            emb = emb / emb.norm(dim=-1, keepdim=True)
        except Exception as e:
            logger.exception(f"Encoder error: {e}")
            continue

        is_unique = (prev_emb is None or
                     F.cosine_similarity(emb, prev_emb, dim=0).item() < THRESH)
        if is_unique:
            frame_ids.append(f"{t_ms}")
            frames.append(rgb)
            timestamps.append(t_ms)
            prev_emb = emb

    cap.release()
    logger.info(f"{video_path} → unique frames: {len(frames)}")
    return frame_ids, frames, timestamps
