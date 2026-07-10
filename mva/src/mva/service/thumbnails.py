"""从视频在指定时刻抽一帧存 jpg（检索 top-1 缩略图用）。"""
from __future__ import annotations
import os
from typing import Optional

import cv2


def extract_frame(video_path: str, t_sec: float, out_path: str) -> Optional[str]:
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 兜底:取第一帧
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        return out_path if cv2.imwrite(out_path, frame) else None
    finally:
        cap.release()
