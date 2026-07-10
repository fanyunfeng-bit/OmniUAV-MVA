"""Reads frames from a directory of images, in sorted-name order.

Useful for datasets like MATRIX / MOTChallenge that ship as PNG/JPG
sequences instead of video files. Mirrors FileStreamSource's surface
(iterator yielding `Frame` objects) so downstream layers don't care
which source they're getting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import cv2

from mva.contracts import Frame


_DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


class ImageDirStreamSource:
    """Yields Frame objects from a directory of image files.

    Parameters
    ----------
    path : str | Path
        Directory containing the image files. Files are sorted by name —
        zero-padded numeric names (`0000.png`, `0001.png`, ...) sort the
        intended way; non-padded names will sort lexicographically, not
        numerically.
    view_id : str
        Identifier for this stream (e.g. "D1").
    source_fps : float
        Native frame rate the directory represents. Used to compute
        per-frame timestamps and to decide downsampling stride.
    sample_fps : float | None
        If set, downsample to roughly this rate via nearest-frame
        strategy. None = keep every frame.
    extensions : tuple[str, ...]
        File extensions considered images (lower-cased comparison).
    """

    def __init__(
        self,
        path: str | Path,
        view_id: str,
        source_fps: float = 30.0,
        sample_fps: Optional[float] = None,
        extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise NotADirectoryError(f"Image dir not found: {self.path}")
        if source_fps <= 0:
            raise ValueError("source_fps must be positive")
        self.view_id = view_id
        self.source_fps = float(source_fps)
        self.sample_fps = sample_fps
        self.extensions = tuple(e.lower() for e in extensions)

        self._files = sorted(
            p for p in self.path.iterdir()
            if p.is_file() and p.suffix.lower() in self.extensions
        )

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self) -> Iterator[Frame]:
        keep_every = 1
        if self.sample_fps is not None and self.sample_fps > 0:
            keep_every = max(1, int(round(self.source_fps / self.sample_fps)))

        for frame_idx, file_path in enumerate(self._files):
            if frame_idx % keep_every != 0:
                continue
            image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if image is None:
                # cv2 returns None on decode failure; skip but log via stderr
                # to keep tight-loop tests fast.
                continue
            yield Frame(
                view_id=self.view_id,
                t=frame_idx / self.source_fps,
                image=image,
                telemetry=None,    # 🔌 §3.4 #1
            )
