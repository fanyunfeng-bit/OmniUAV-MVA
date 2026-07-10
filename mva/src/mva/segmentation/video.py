"""Video-file segmenter — slices an mp4 into time-windowed segments.

Uses `cv2.VideoCapture` + `CAP_PROP_POS_MSEC` seek so we only decode K
frames per segment, not every source frame. For MVU-Eval (4636 videos ×
~30 s each × 30 fps), this is ~100× faster than the streaming variant
that would decode every frame.

Robust to:
  - VFR / odd FPS metadata (seek by MSEC, not by frame index)
  - Short videos < MIN_SEGMENT_SEC (yields one segment containing the
    whole video — useful for unit tests with synthetic 2 s clips)
  - Trailing remainder < MIN_SEGMENT_SEC (dropped to avoid embedding a
    0.3 s tail)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from mva.segmentation.base import MIN_SEGMENT_SEC, Segment, SegmenterConfig


def iter_video_segments(
    video_path: Path | str,
    view_id: str,
    config: SegmenterConfig,
) -> Iterator[Segment]:
    """Yield Segments from `video_path`. No-op iterator if the file
    cannot be opened or has 0 frames."""
    import cv2

    config.validate()

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total <= 0:
            return
        duration = total / fps

        seg_idx = 0
        cur = 0.0
        while cur < duration:
            seg_end = min(cur + config.window_sec, duration)
            if seg_end - cur < MIN_SEGMENT_SEC and seg_idx > 0:
                # Drop tiny trailing segment, but keep the only segment
                # for very short videos (< MIN_SEGMENT_SEC total).
                break

            span = seg_end - cur
            timestamps = [
                cur + (i + 0.5) * (span / config.nframes_per_segment)
                for i in range(config.nframes_per_segment)
            ]
            frames = []
            frame_indices = []
            for t_sec in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
                ok, img = cap.read()
                if ok and img is not None:
                    frames.append(img)
                    # Best-effort source-frame index via fps;
                    # consumers should treat as approximate for VFR.
                    frame_indices.append(int(t_sec * fps))

            if frames:
                yield Segment(
                    view_id=view_id,
                    segment_idx=seg_idx,
                    start_t=float(cur),
                    end_t=float(seg_end),
                    frames=frames,
                    frame_indices=frame_indices,
                    source_uri=str(path),
                    metadata={"source_fps": fps, "duration_sec": duration},
                )

            seg_idx += 1
            cur += config.stride_sec
    finally:
        cap.release()
