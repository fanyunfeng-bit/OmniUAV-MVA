"""Unit tests for `mva index` plumbing.

Focus on the pure helpers (metadata assembly, detector aggregation) so we
don't have to spin up a real YOLO / a real embedder. The end-to-end CLI
behavior is exercised manually per PROGRESS.md "如何手动测试".
"""
from __future__ import annotations

import json
from typing import List

import numpy as np

from mva.cli.index import _build_metadata
from mva.datasets.base import IndexUnit
from mva.l1_perception import Detection


class _FakeDetector:
    """Deterministic stand-in for `mva.l1_perception.Detector`.

    Returns a fixed Detection list per `.detect(frame)` call so we can
    assert how the metadata aggregator counts class hits across N sampled
    frames per segment.
    """

    def __init__(self, per_frame: List[List[Detection]]) -> None:
        self._per_frame = per_frame
        self._call = 0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        out = self._per_frame[self._call % len(self._per_frame)]
        self._call += 1
        return out


def _seg_unit(n_frames: int = 2) -> IndexUnit:
    return IndexUnit(
        unit_id="vid::seg0001",
        scene_id="qa-0",
        view_id="vid.mp4",
        kind="image_seq",
        data=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n_frames)],
        vector_type="frame",
        metadata={"video_path": "/data/vid.mp4", "nframes_sampled": n_frames},
        document="vid [10.0-20.0s]",
        start_sec=10.0,
        end_sec=20.0,
        segment_idx=1,
    )


def _det(cls: str, conf: float = 0.9) -> Detection:
    return Detection(
        bbox=(0.0, 0.0, 4.0, 4.0), class_id=0, class_name=cls, confidence=conf,
    )


def test_build_metadata_carries_segment_fields_without_detector():
    meta = _build_metadata(_seg_unit(), detector=None)
    assert meta["video_path"] == "/data/vid.mp4"
    assert meta["start_sec"] == 10.0
    assert meta["end_sec"] == 20.0
    assert meta["segment_idx"] == 1
    # `chunk_id` mirrors segment_idx so ChromaDB id stays unique per segment
    assert meta["chunk_id"] == 1
    assert "detected_classes" not in meta


def test_build_metadata_aggregates_detection_across_frames():
    """Two sampled frames; frame A has 2 persons + 1 car, frame B has 1
    person. Aggregator should sum across frames (total persons = 3, car = 1)
    and serialize as JSON in `detected_counts_json`."""
    unit = _seg_unit(n_frames=2)
    detector = _FakeDetector([
        [_det("person"), _det("person"), _det("car")],
        [_det("person")],
    ])
    meta = _build_metadata(unit, detector=detector)
    assert meta["detected_classes"] == "car,person"
    counts = json.loads(meta["detected_counts_json"])
    assert counts == {"car": 1, "person": 3}


def test_build_metadata_skips_detection_for_image_kind():
    """ROI-crop (MATRIX) units are kind=image — detection step must be a
    no-op there (it's a per-segment aggregation, not per-crop)."""
    unit = IndexUnit(
        unit_id="tk-1", scene_id="MINI", view_id="D1",
        kind="image", data=np.zeros((8, 8, 3), dtype=np.uint8),
        vector_type="reid", metadata={"t": 0.0},
    )
    detector = _FakeDetector([[_det("person")]])
    meta = _build_metadata(unit, detector=detector)
    assert "detected_classes" not in meta
    assert "detected_counts_json" not in meta


def test_build_metadata_empty_detection_does_not_emit_keys():
    """If detector finds nothing in any sampled frame, we should NOT emit
    a `detected_classes=""` row — easier to filter on key-existence."""
    unit = _seg_unit(n_frames=2)
    detector = _FakeDetector([[], []])
    meta = _build_metadata(unit, detector=detector)
    assert "detected_classes" not in meta
    assert "detected_counts_json" not in meta
