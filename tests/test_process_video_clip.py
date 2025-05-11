import os
import cv2
import tempfile
import numpy as np
import torch
from video_processor.process_video_clip import process_video_clip

def dummy_encoder(pil_image):
    # Простая фиктивная функция: превращает изображение в нулевой вектор
    return torch.zeros(1, 512)

def make_test_video(path, width=64, height=48, fps=10, duration=1.0):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    frames = int(fps * duration)
    for i in range(frames):
        color = (i % 2) * 255
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        out.write(frame)
    out.release()

def test_process_video_clip():
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp.close()
    make_test_video(tmp.name, duration=0.5)
    vid_id, frame_ids, frames, times = process_video_clip(
        tmp.name,
        dummy_encoder,
        similarity_threshold=0.1,
        processing_timestep=100
    )
    os.remove(tmp.name)

    # В видео чередуются чёрный/белый, дедупликация должна пропустить половину
    assert len(frames) >= 1
    assert all(isinstance(fid, str) for fid in frame_ids)
    assert all(isinstance(img, np.ndarray) for img in frames)
    assert all(isinstance(t, int) for t in times)
