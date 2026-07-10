"""Image-directory segmenter — slices a sorted PNG/JPG sequence into
time-windowed segments.

Models the sequence as if it were a `source_fps`-fps video, so the same
`SegmenterConfig` (window_sec / stride_sec) applies. MATRIX (2 fps, ~1000
PNG/view) is the primary use case: a 10 s window covers 20 frames; we
uniformly sample K=4 from those.

Reads only the K sampled frames per segment (cv2.imread on those files),
not the whole directory — same I/O profile as the video seek variant.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from mva.segmentation.base import MIN_SEGMENT_SEC, Segment, SegmenterConfig


_DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def iter_image_dir_segments(
    image_dir: Path | str,
    view_id: str,
    source_fps: float,
    config: SegmenterConfig,
    extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS,
) -> Iterator[Segment]:
    """Yield Segments from a sorted image directory. No-op iterator if
    the directory is empty or `source_fps <= 0`."""
    import cv2

    config.validate()
    if source_fps <= 0:
        raise ValueError(f"source_fps must be > 0, got {source_fps}")

    dir_path = Path(image_dir)
    if not dir_path.is_dir():
        return

    ext_lower = tuple(e.lower() for e in extensions)
    files = sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in ext_lower
    )
    if not files:
        return

    duration = len(files) / source_fps

    seg_idx = 0
    cur = 0.0
    while cur < duration:
        seg_end = min(cur + config.window_sec, duration)
        if seg_end - cur < MIN_SEGMENT_SEC and seg_idx > 0:
            break

        start_fidx = int(cur * source_fps)
        end_fidx = min(int(seg_end * source_fps), len(files))
        if end_fidx <= start_fidx:
            seg_idx += 1
            cur += config.stride_sec
            continue
        chunk_size = end_fidx - start_fidx
        K = config.nframes_per_segment
        if chunk_size <= K:
            sampled_fidx = list(range(start_fidx, end_fidx))
        else:
            step = chunk_size / K
            sampled_fidx = [
                start_fidx + int((i + 0.5) * step) for i in range(K)
            ]

        frames = []
        frame_indices = []
        for fidx in sampled_fidx:
            img = cv2.imread(str(files[fidx]), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
                frame_indices.append(fidx)

        if frames:
            yield Segment(
                view_id=view_id,
                segment_idx=seg_idx,
                start_t=float(cur),
                end_t=float(seg_end),
                frames=frames,
                frame_indices=frame_indices,
                source_uri=str(dir_path),
                metadata={
                    "source_fps": float(source_fps),
                    "duration_sec": duration,
                    "total_files": len(files),
                },
            )

        seg_idx += 1
        cur += config.stride_sec
