"""Unit tests for the L1 ByteTracker wrapper (mock + boxmot paths).

The boxmot-backed path needs the real library installed; mark it
``@pytest.mark.gpu`` so CI runs only the mock path.
"""
from __future__ import annotations

import pytest

from mva.l1_perception import ByteTracker, Detection


# --------------------------------------------------------------------------
# Mock fallback — deterministic, no external deps
# --------------------------------------------------------------------------


def _det(bbox, conf=0.9, cls=("person", 0)):
    cls_name, cls_id = cls
    return Detection(
        bbox=bbox, class_id=cls_id, class_name=cls_name, confidence=conf,
    )


def test_mock_same_object_three_frames_keeps_one_track_id():
    tracker = ByteTracker(iou_threshold=0.5)
    # Slight translation each frame; IoU stays > 0.5
    out0 = tracker.update([_det((100, 100, 200, 200))], 480, 640)
    out1 = tracker.update([_det((105, 100, 205, 200))], 480, 640)
    out2 = tracker.update([_det((110, 100, 210, 200))], 480, 640)
    # All three same id
    assert out0[0][1] == out1[0][1] == out2[0][1]
    # First id is 1 by mock convention
    assert out0[0][1] == 1


def test_mock_two_distant_objects_get_separate_ids():
    tracker = ByteTracker()
    out = tracker.update(
        [_det((100, 100, 200, 200)), _det((500, 500, 600, 600))],
        480, 640,
    )
    assert len(out) == 2
    assert out[0][1] != out[1][1]


def test_mock_reset_clears_internal_state():
    tracker = ByteTracker()
    tracker.update([_det((100, 100, 200, 200))], 480, 640)
    tracker.update([_det((105, 100, 205, 200))], 480, 640)
    tracker.reset()
    # Post-reset: a brand-new object should start from id 1 again
    out = tracker.update([_det((50, 50, 100, 100))], 480, 640)
    assert out[0][1] == 1


def test_mock_empty_detection_returns_empty():
    tracker = ByteTracker()
    out = tracker.update([], 480, 640)
    assert out == []


def test_mock_low_confidence_detections_dropped():
    tracker = ByteTracker(conf_threshold=0.5)
    out = tracker.update(
        [_det((100, 100, 200, 200), conf=0.9),
         _det((300, 300, 400, 400), conf=0.2)],
        480, 640,
    )
    # The low-conf det dropped
    assert len(out) == 1
    assert out[0][0].confidence == pytest.approx(0.9)


def test_mock_low_iou_assigns_new_id():
    """Same class but the bbox shifts so much IoU < threshold → new track."""
    tracker = ByteTracker(iou_threshold=0.5)
    out0 = tracker.update([_det((100, 100, 200, 200))], 480, 640)
    # 250px shift: zero IoU
    out1 = tracker.update([_det((400, 100, 500, 200))], 480, 640)
    assert out0[0][1] != out1[0][1]


def test_mock_cross_segment_isolation_via_reset():
    """After reset() between segments, a new object should not inherit
    the previous segment's id."""
    tracker = ByteTracker()
    # segment 0
    tracker.update([_det((100, 100, 200, 200))], 480, 640)
    tracker.update([_det((105, 100, 205, 200))], 480, 640)
    tracker.reset()
    # segment 1
    out = tracker.update([_det((100, 100, 200, 200))], 480, 640)
    assert out[0][1] == 1   # fresh counter


def test_mock_multiple_tracks_persist_across_frames():
    tracker = ByteTracker()
    out0 = tracker.update(
        [_det((100, 100, 200, 200)), _det((400, 100, 500, 200))],
        480, 640,
    )
    out1 = tracker.update(
        [_det((105, 100, 205, 200)), _det((405, 100, 505, 200))],
        480, 640,
    )
    # Both ids should match across frames
    assert {o[1] for o in out0} == {o[1] for o in out1}
    # And both stayed associated to their original bbox by IoU
    id_for_left_t0 = out0[0][1]
    id_for_left_t1 = next(tid for det, tid in out1 if det.bbox[0] == 105)
    assert id_for_left_t0 == id_for_left_t1


def test_is_simple_property_reflects_default_algorithm():
    assert ByteTracker().is_simple is True
    assert ByteTracker(algorithm="iou_greedy").is_simple is True


def test_invalid_algorithm_raises():
    with pytest.raises(ValueError):
        ByteTracker(algorithm="nonexistent")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# boxmot ByteTrack opt-in path — skipped if boxmot is not installed
# --------------------------------------------------------------------------


@pytest.mark.gpu
def test_bytetrack_algorithm_runs_when_boxmot_available():
    """algorithm='bytetrack' loads boxmot and returns positive integer
    ids. Note: ByteTrack's two-stage matching can drop dets on the first
    frame (see module docstring), so the assertion is loose — we just
    confirm the integration loads + returns the right shape, not that
    the output count matches the input."""
    pytest.importorskip("boxmot")
    tracker = ByteTracker(algorithm="bytetrack")
    assert tracker.is_simple is False
    out = tracker.update(
        [_det((100, 100, 200, 200), conf=0.9),
         _det((500, 500, 600, 600), conf=0.9)],
        720, 1280,
    )
    for det, tid in out:
        assert isinstance(tid, int)
        assert tid > 0


def test_iou_greedy_accepts_optional_frame_kwarg():
    # `frame` is optional + ignored by iou_greedy (it exists for botsort CMC);
    # passing it must not break the default path (backward compat).
    import numpy as np
    tracker = ByteTracker()
    dets = [Detection(bbox=(10, 10, 30, 30), class_id=0, class_name="x",
                      confidence=0.9)]
    out = tracker.update(dets, 100, 100,
                         frame=np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(out) == 1


def test_botsort_is_valid_algorithm():
    # 'botsort' must be accepted (not a ValueError) when boxmot is available.
    pytest.importorskip("boxmot")
    tracker = ByteTracker(algorithm="botsort")
    assert tracker.is_simple is False


@pytest.mark.gpu
def test_botsort_algorithm_runs_when_boxmot_available():
    """algorithm='botsort' loads boxmot (with_reid=False) and returns positive
    integer ids. Loose like the bytetrack test — confirms integration shape."""
    import numpy as np
    pytest.importorskip("boxmot")
    tracker = ByteTracker(algorithm="botsort")
    dets = [Detection(bbox=(100, 100, 140, 160), class_id=2, class_name="car",
                      confidence=0.9)]
    out = tracker.update(dets, 1080, 1920,
                         frame=np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert all(isinstance(tid, int) and tid > 0 for _, tid in out)
