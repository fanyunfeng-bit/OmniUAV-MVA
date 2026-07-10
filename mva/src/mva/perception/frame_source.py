from __future__ import annotations
from typing import Iterator, Protocol, Tuple, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    fps: float
    def iter_frames(self) -> Iterator[Tuple[float, np.ndarray]]: ...


class UniformFrameSource:
    """按 target_fps 均匀抽帧(密集感知流基线)。t 为视频绝对时间(秒)。"""
    def __init__(self, video_path: str, target_fps: float = 5.0):
        self.video_path = video_path
        self.target_fps = float(target_fps)
        self.fps = self.target_fps

    def iter_frames(self) -> Iterator[Tuple[float, np.ndarray]]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        src_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        step = max(1, int(round(src_fps / max(1e-6, self.target_fps))))
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    yield (idx / src_fps, frame)
                idx += 1
        finally:
            cap.release()
