"""Reads frames from a local video file via OpenCV.

Used for offline evaluation. RTSP live source belongs in a separate module
(deferred to v2+).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional

import cv2

from mva.contracts import Frame


class FileStreamSource:
    """Reads a video file and yields Frame objects.

    Parameters
    ----------
    path : str | Path
        Path to a video file (.mp4 / .avi / etc.).
    view_id : str
        Identifier for this stream (e.g. "drone-1").
    sample_fps : float | None
        If set, downsample to roughly this rate via nearest-frame strategy.
        None = keep source FPS.
    """

    def __init__(
        self,
        path: str | Path,
        view_id: str,
        sample_fps: Optional[float] = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self.view_id = view_id
        self.sample_fps = sample_fps

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        keep_every = 1
        if self.sample_fps is not None and self.sample_fps > 0:
            keep_every = max(1, int(round(source_fps / self.sample_fps)))

        frame_idx = 0
        t_start = time.time()
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if frame_idx % keep_every == 0:
                    yield Frame(
                        view_id=self.view_id,
                        t=t_start + frame_idx / source_fps,
                        image=image,
                        telemetry=None,    # 🔌 §3.4 #1
                    )
                frame_idx += 1
        finally:
            cap.release()
