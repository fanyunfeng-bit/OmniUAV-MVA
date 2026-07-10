"""Video / image-sequence segmentation (M2.8).

Unifies the "slice a stream into K-frames-per-window" operation across
sources. Used by `mva ingest` (the M2.8 unified pipeline) and by future
adapters via the `DatasetAdapter.iter_segments` Protocol method.

Two source-shape implementations:
  - `iter_video_segments` — `cv2.VideoCapture` + `CAP_PROP_POS_MSEC` seek
    (efficient on long videos; one decode per sampled frame, not per
    source frame).
  - `iter_image_dir_segments` — sorted file list + per-segment indexing
    (MATRIX / MOTChallenge style).

Both yield `Segment` instances with `(start_t, end_t, frames,
frame_indices, source_uri)` — `frame_indices` are the original positions
in the source so retrieval can map an embedding back to the exact source
frame.

A `SegmenterConfig` carries the three tunable knobs:
  window_sec    — segment length
  stride_sec    — stride between segment starts; equal to window_sec
                  yields non-overlapping segments
  nframes_per_segment — how many frames to uniformly sample per segment
                        (embedder mean-pools them into one vector)
"""
from mva.segmentation.base import (
    MIN_SEGMENT_SEC,
    Segment,
    SegmenterConfig,
)
from mva.segmentation.image_dir import iter_image_dir_segments
from mva.segmentation.video import iter_video_segments

__all__ = [
    "MIN_SEGMENT_SEC",
    "Segment",
    "SegmenterConfig",
    "iter_image_dir_segments",
    "iter_video_segments",
]
