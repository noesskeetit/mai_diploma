import cv2
import logging
from PIL import Image
import torch.nn.functional as F
from dotenv import load_dotenv
import os

load_dotenv()
STEP_MS = int(os.getenv('PROCESSING_TIMESTEP', '1000'))  # интервал в мс
THRESH = float(os.getenv('PROCESSING_SIMILARITY_THRESHOLD', '0.65'))  # порог схожести

logger = logging.getLogger("process_video_clip")

def process_video_clip(video_path: str, encoder_fn):
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
            emb = encoder_fn(pil).cpu()
            emb = emb / emb.norm(dim=-1, keepdim=True)
        except Exception as e:
            logger.exception(f"Encoder error: {e}")
            continue

        if prev_emb is None or F.cosine_similarity(emb, prev_emb, dim=0).item() < THRESH:
            frame_ids.append(str(t_ms))
            frames.append(rgb)
            timestamps.append(t_ms)
            prev_emb = emb

    cap.release()
    logger.info(f"{video_path} → unique frames: {len(frames)}")
    return frame_ids, frames, timestamps
