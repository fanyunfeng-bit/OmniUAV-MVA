"""Segment + SegmenterConfig contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Drop trailing segments shorter than this — avoids embedding a 0.3 s tail.
# Exposed so tests / callers can reason about edge cases.
MIN_SEGMENT_SEC = 1.0


@dataclass
class Segment:
    """One time-windowed slice of a stream with K sampled frames.

    `start_t` / `end_t` are in seconds, always 0-based (the first sample
    of any source starts at t=0). `frame_indices` are the original
    integer positions in the source — for video files that's the frame
    index inside the mp4; for image directories it's the index into the
    sorted file list. Used by retrieval to map an embedding back to the
    exact source frame ("命中第 14 段的第 2 采样帧 → 原视频第 423 帧").

    `source_uri` is the source path (video file or PNG directory). Kept
    here so the consumer can do `ffmpeg -ss start_t -t (end_t - start_t)
    source_uri ...` or open the PNG corresponding to `frame_indices[i]`
    without re-resolving via the adapter.
    """

    view_id: str
    segment_idx: int
    start_t: float
    end_t: float
    frames: list[np.ndarray]            # K BGR frames, ready for L1 detect or embedder.encode_images
    frame_indices: list[int]
    source_uri: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmenterConfig:
    """Knobs for the sliding-window segmenter.

    Defaults match sentrysearch's convention (10 s non-overlapping windows,
    4 frames per window mean-pooled into one embedding).
    """

    window_sec: float = 10.0
    stride_sec: float = 10.0
    nframes_per_segment: int = 4

    def validate(self) -> None:
        if self.window_sec <= 0 or self.stride_sec <= 0:
            raise ValueError(
                f"window_sec / stride_sec must be > 0, got "
                f"window={self.window_sec}, stride={self.stride_sec}"
            )
        if self.nframes_per_segment <= 0:
            raise ValueError(
                f"nframes_per_segment must be > 0, got "
                f"{self.nframes_per_segment}"
            )
