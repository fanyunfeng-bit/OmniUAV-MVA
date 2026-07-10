"""Unit tests for mva.datasets.

Mock filesystem layouts via tmp_path fixtures so tests run without the
real ~50 GB datasets. Real-data validation happens in manual `mva ...`
runs (see PROGRESS.md "如何手动测试").
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mva.datasets import (
    DatasetAdapter,
    IndexUnit,
    MatrixDataset,
    MVUEvalDataset,
    QAPair,
    get_adapter,
    list_known,
)
from mva.datasets.reservoir import ReservoirDataset
from mva.datasets.visdrone_mdmt import VisDroneMDMTDataset


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


def test_registry_lists_known():
    known = list_known()
    assert "matrix" in known
    assert "mvu-eval" in known


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        get_adapter("does-not-exist", root="/tmp/nope")


# ----------------------------------------------------------------------
# MATRIX adapter — synthetic mini-MATRIX
# ----------------------------------------------------------------------


def _make_mini_matrix(root: Path) -> Path:
    """Create a 1-scene, 2-view, 3-frame MATRIX-like layout."""
    scene = root / "MINI"
    for view in ("D1", "D2"):
        view_dir = scene / "image_subsets" / view
        view_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            cv2.imwrite(
                str(view_dir / f"{i:04d}.png"),
                np.full((8, 8, 3), i * 50, dtype=np.uint8),
            )
    # Matchings dir for cross-view GT (optional, but exercise the loader)
    matchings = scene / "matchings" / "Pedestrians"
    matchings.mkdir(parents=True, exist_ok=True)
    (matchings / "3d_0000.txt").write_text(
        "0 12345 1.0 2.0 0.0\n1 67890 3.0 4.0 0.0\n"
    )
    return scene


def test_matrix_lists_one_scene(tmp_path):
    _make_mini_matrix(tmp_path)
    ds = MatrixDataset(root=tmp_path)
    scenes = list(ds.list_scenes())
    assert len(scenes) == 1
    assert scenes[0].scene_id == "MINI"
    assert set(scenes[0].view_ids) == {"D1", "D2"}


def test_matrix_open_view_yields_frames(tmp_path):
    _make_mini_matrix(tmp_path)
    ds = MatrixDataset(root=tmp_path)
    src = ds.open_view("MINI", "D1")
    frames = list(src)
    assert len(frames) == 3
    assert frames[0].view_id == "D1"


def test_matrix_iter_indexable_requires_store(tmp_path):
    _make_mini_matrix(tmp_path)
    ds = MatrixDataset(root=tmp_path)
    with pytest.raises(ValueError):
        list(ds.iter_indexable_units("MINI"))


def test_matrix_cross_view_gt_parses(tmp_path):
    _make_mini_matrix(tmp_path)
    ds = MatrixDataset(root=tmp_path)
    gt = ds.load_cross_view_gt("MINI", 0)
    assert len(gt) == 2
    assert gt[0]["global_id"] == 12345
    assert gt[1]["x"] == 3.0


def test_matrix_qa_not_supported(tmp_path):
    _make_mini_matrix(tmp_path)
    ds = MatrixDataset(root=tmp_path)
    with pytest.raises(NotImplementedError):
        list(ds.load_qa_pairs())


def test_matrix_satisfies_protocol(tmp_path):
    _make_mini_matrix(tmp_path)
    ds = MatrixDataset(root=tmp_path)
    assert isinstance(ds, DatasetAdapter)
    assert ds.supports_cross_view_linking is True
    assert ds.cross_view_linking_mode == "synchronized"
    assert ds.supports_qa_eval is False


# ----------------------------------------------------------------------
# MVU-Eval adapter — synthetic mini-MVU
# ----------------------------------------------------------------------


def _make_mini_mvu(root: Path) -> None:
    """Create a 3-QA, 2-video MVU-Eval-like layout."""
    root.mkdir(parents=True, exist_ok=True)
    # Two stub mp4s (we just write empty files; loaders only care about names
    # for the path-resolution tests, not real video reads)
    for name in ("video1.mp4", "video2.mp4"):
        (root / name).write_bytes(b"")
    qa = {
        "0": {
            "video_paths": ["video1.mp4", "video2.mp4"],
            "question": "Which is brighter?",
            "options": ["A. Video 1", "B. Video 2"],
            "ground_truth": "A",
            "task": "Counting",
        },
        "1": {
            "video_paths": ["video1.mp4"],
            "question": "What do you see?",
            "options": ["A. Cat", "B. Dog"],
            "ground_truth": "B",
            "task": "Object Recognition",
        },
        "2": {
            "video_paths": ["video2.mp4"],
            "question": "How many?",
            "options": ["A. 1", "B. 2", "C. 3"],
            "ground_truth": "C",
            "task": "Counting",
        },
    }
    (root / "MVU_Eval_QAs.json").write_text(json.dumps(qa))


def test_mvu_lists_scenes(tmp_path):
    _make_mini_mvu(tmp_path)
    ds = MVUEvalDataset(root=tmp_path)
    scenes = list(ds.list_scenes())
    assert len(scenes) == 3
    assert scenes[0].scene_id == "qa-0"


def test_mvu_list_scenes_task_filter(tmp_path):
    _make_mini_mvu(tmp_path)
    ds = MVUEvalDataset(root=tmp_path)
    counting = list(ds.list_scenes(filter={"task": "Counting"}))
    assert len(counting) == 2
    objrec = list(ds.list_scenes(filter={"task": "Object Recognition"}))
    assert len(objrec) == 1
    assert objrec[0].scene_id == "qa-1"


def test_mvu_list_scenes_limit(tmp_path):
    _make_mini_mvu(tmp_path)
    ds = MVUEvalDataset(root=tmp_path)
    first2 = list(ds.list_scenes(limit=2))
    assert len(first2) == 2


def test_mvu_get_scene_invalid_id(tmp_path):
    _make_mini_mvu(tmp_path)
    ds = MVUEvalDataset(root=tmp_path)
    with pytest.raises(KeyError):
        ds.get_scene("not-a-qa")
    with pytest.raises(KeyError):
        ds.get_scene("qa-9999")


def test_mvu_qa_pairs_yield_full_records(tmp_path):
    _make_mini_mvu(tmp_path)
    ds = MVUEvalDataset(root=tmp_path)
    qa_list = list(ds.load_qa_pairs())
    assert len(qa_list) == 3
    qa0 = qa_list[0]
    assert qa0.qa_id == "0"
    assert qa0.scene_id == "qa-0"
    assert qa0.ground_truth == "A"
    assert qa0.options == ["A. Video 1", "B. Video 2"]
    assert qa0.task == "Counting"
    assert qa0.metadata["video_paths"] == ["video1.mp4", "video2.mp4"]


def test_mvu_qa_pairs_task_filter_and_limit(tmp_path):
    _make_mini_mvu(tmp_path)
    ds = MVUEvalDataset(root=tmp_path)
    only_counting = list(ds.load_qa_pairs(tasks=["Counting"]))
    assert len(only_counting) == 2
    just_one = list(ds.load_qa_pairs(limit=1))
    assert len(just_one) == 1


def test_mvu_satisfies_protocol(tmp_path):
    _make_mini_mvu(tmp_path)
    ds = MVUEvalDataset(root=tmp_path)
    assert isinstance(ds, DatasetAdapter)
    # M3.0: MVU-Eval supports cross-view linking via appearance mode
    # (videos within a QA are not time-synchronized but often share
    # objects — esp. video_editing and Ordering tasks).
    assert ds.supports_cross_view_linking is True
    assert ds.cross_view_linking_mode == "appearance"
    assert ds.supports_qa_eval is True


def test_mvu_missing_qa_file_raises(tmp_path):
    (tmp_path / "video1.mp4").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        MVUEvalDataset(root=tmp_path)


# ----------------------------------------------------------------------
# IndexUnit / Scene / QAPair dataclasses
# ----------------------------------------------------------------------


def test_index_unit_construction():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    u = IndexUnit(
        unit_id="x", scene_id="s", view_id="v",
        kind="image", data=img, vector_type="reid",
        metadata={"t": 0.0},
    )
    assert u.vector_type == "reid"
    assert u.document is None
    # Segment fields default to None for non-segmented (ROI-crop) units
    assert u.start_sec is None
    assert u.end_sec is None
    assert u.segment_idx is None


def test_index_unit_segment_fields_round_trip():
    """For video segment units, start_sec / end_sec / segment_idx must
    travel as first-class fields (used by `mva index` to build ChromaDB
    metadata that lets retrieval map back to the original clip)."""
    u = IndexUnit(
        unit_id="vid::seg0003", scene_id="qa-0", view_id="vid.mp4",
        kind="image_seq",
        data=[np.zeros((4, 4, 3), dtype=np.uint8)],
        vector_type="frame",
        start_sec=30.0, end_sec=40.0, segment_idx=3,
    )
    assert u.start_sec == 30.0
    assert u.end_sec == 40.0
    assert u.segment_idx == 3


def test_qa_pair_open_ended():
    """ground_truth=None and options=None mean open-ended; eval should skip
    accuracy scoring but still record model output."""
    qa = QAPair(qa_id="x", scene_id="s", question="Q")
    assert qa.options is None
    assert qa.ground_truth is None


# ----------------------------------------------------------------------
# MVU-Eval video segmentation — exercise sentrysearch-style sliding window
# ----------------------------------------------------------------------


def _write_synthetic_mp4(path: Path, duration_sec: float, fps: float = 10.0) -> None:
    """Write a tiny synthetic mp4 of solid-color frames so cv2 can read
    duration + seek by msec. Frame content doesn't matter — segmentation
    only cares about timing."""
    total = max(1, int(duration_sec * fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (16, 16))
    if not writer.isOpened():
        pytest.skip("cv2 mp4 writer unavailable in this environment")
    try:
        for i in range(total):
            shade = int(255 * (i / max(1, total - 1)))
            frame = np.full((16, 16, 3), shade, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _make_mvu_with_real_videos(root: Path, duration_sec: float) -> None:
    """MVU-Eval layout with two REAL synthetic mp4s (vs. empty stubs)."""
    root.mkdir(parents=True, exist_ok=True)
    for name in ("video_a.mp4", "video_b.mp4"):
        _write_synthetic_mp4(root / name, duration_sec)
    qa = {
        "0": {
            "video_paths": ["video_a.mp4", "video_b.mp4"],
            "question": "Q?",
            "options": ["A.", "B."],
            "ground_truth": "A",
            "task": "Counting",
        },
    }
    (root / "MVU_Eval_QAs.json").write_text(json.dumps(qa))


def test_mvu_iter_indexable_units_segments_video(tmp_path):
    """A 25-second video at window=10s/stride=10s yields ≥ 2 segments."""
    _make_mvu_with_real_videos(tmp_path, duration_sec=25.0)
    ds = MVUEvalDataset(root=tmp_path)
    units = list(ds.iter_indexable_units(
        "qa-0", view_id="video_a.mp4",
        window_sec=10.0, stride_sec=10.0, nframes_per_segment=2,
    ))
    assert len(units) >= 2, f"expected ≥2 segments, got {len(units)}"
    # First segment starts at 0; segment_idx is monotonically increasing
    assert units[0].start_sec == 0.0
    assert units[0].segment_idx == 0
    for i in range(1, len(units)):
        assert units[i].segment_idx == units[i - 1].segment_idx + 1
        assert units[i].start_sec > units[i - 1].start_sec
    # Every yielded unit carries the segment metadata + at least one frame
    for u in units:
        assert u.kind == "image_seq"
        assert u.end_sec > u.start_sec
        assert isinstance(u.data, list) and len(u.data) >= 1
        assert u.metadata["video_path"].endswith("video_a.mp4")


def test_mvu_iter_indexable_units_unique_unit_ids(tmp_path):
    """Per-segment unit_ids must be unique within a view so ChromaDB can use
    them as id components without collision."""
    _make_mvu_with_real_videos(tmp_path, duration_sec=22.0)
    ds = MVUEvalDataset(root=tmp_path)
    units = list(ds.iter_indexable_units(
        "qa-0", view_id="video_a.mp4",
        window_sec=10.0, stride_sec=10.0, nframes_per_segment=2,
    ))
    ids = [u.unit_id for u in units]
    assert len(ids) == len(set(ids))


def test_mvu_iter_indexable_units_overlapping_stride(tmp_path):
    """stride < window yields more (overlapping) segments than non-overlap."""
    _make_mvu_with_real_videos(tmp_path, duration_sec=20.0)
    ds = MVUEvalDataset(root=tmp_path)
    no_overlap = list(ds.iter_indexable_units(
        "qa-0", view_id="video_a.mp4",
        window_sec=10.0, stride_sec=10.0, nframes_per_segment=2,
    ))
    overlap = list(ds.iter_indexable_units(
        "qa-0", view_id="video_a.mp4",
        window_sec=10.0, stride_sec=5.0, nframes_per_segment=2,
    ))
    assert len(overlap) > len(no_overlap)


def test_mvu_iter_indexable_units_max_frames_caps(tmp_path):
    """`max_frames` (legacy CLI knob meaning 'max units across views') still
    caps total yield even with segmentation."""
    _make_mvu_with_real_videos(tmp_path, duration_sec=30.0)
    ds = MVUEvalDataset(root=tmp_path)
    units = list(ds.iter_indexable_units(
        "qa-0", max_frames=2,
        window_sec=5.0, stride_sec=5.0, nframes_per_segment=2,
    ))
    assert len(units) == 2


def test_mvu_iter_indexable_units_rejects_bad_window(tmp_path):
    _make_mini_mvu(tmp_path)
    ds = MVUEvalDataset(root=tmp_path)
    with pytest.raises(ValueError):
        list(ds.iter_indexable_units("qa-0", window_sec=0.0))


# ----------------------------------------------------------------------
# VisDrone-MDMT
# ----------------------------------------------------------------------


def _make_visdrone(tmp_path, n_frames=30):
    """Create a minimal VisDrone-MDMT test layout."""
    base = tmp_path / "Multi-Drone-Multi-Object-Detection-and-Tracking" / "test"
    for drone, suffix in [("1", "1"), ("2", "2")]:
        view_dir = base / drone / f"26-{suffix}"
        view_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_frames):
            cv2.imwrite(
                str(view_dir / f"{i+1:08d}.jpg"),
                np.full((16, 16, 3), i % 256, dtype=np.uint8),
            )
    return tmp_path


def test_visdrone_list_scenes(tmp_path):
    _make_visdrone(tmp_path)
    ds = VisDroneMDMTDataset(root=tmp_path)
    scenes = list(ds.list_scenes())
    assert len(scenes) == 1
    assert scenes[0].scene_id == "scene-26"
    assert sorted(scenes[0].view_ids) == ["D1", "D2"]


def test_visdrone_get_scene(tmp_path):
    _make_visdrone(tmp_path)
    ds = VisDroneMDMTDataset(root=tmp_path)
    scene = ds.get_scene("scene-26")
    assert scene.view_ids == ["D1", "D2"]
    assert scene.metadata["source_fps"] == 30.0
    assert scene.metadata["default_window_sec"] == 3.0


def test_visdrone_iter_segments(tmp_path):
    from mva.segmentation import SegmenterConfig
    _make_visdrone(tmp_path, n_frames=90)  # 90 frames @30fps = 3s → 1 segment at window=3
    ds = VisDroneMDMTDataset(root=tmp_path)
    config = SegmenterConfig(window_sec=3.0, stride_sec=3.0, nframes_per_segment=4)
    segs = list(ds.iter_segments("scene-26", "D1", config))
    assert len(segs) >= 1
    assert segs[0].view_id == "D1"
    assert len(segs[0].frames) == 4


def test_visdrone_cross_view_mode(tmp_path):
    _make_visdrone(tmp_path)
    ds = VisDroneMDMTDataset(root=tmp_path)
    assert ds.cross_view_linking_mode == "appearance"


def test_visdrone_satisfies_protocol(tmp_path):
    _make_visdrone(tmp_path)
    ds = VisDroneMDMTDataset(root=tmp_path)
    assert isinstance(ds, DatasetAdapter)


# ----------------------------------------------------------------------
# Reservoir / PCL-Simulation adapter — synthetic file-backed scene
# ----------------------------------------------------------------------


def _make_reservoir(root: Path, dur1: float, dur2: float) -> None:
    """Create <root>/Reservoir/{view1,view2}.mp4 with the given durations."""
    scene = root / "Reservoir"
    scene.mkdir(parents=True, exist_ok=True)
    _write_synthetic_mp4(scene / "view1.mp4", duration_sec=dur1)
    _write_synthetic_mp4(scene / "view2.mp4", duration_sec=dur2)


def test_reservoir_registered():
    assert "pcl-sim" in list_known()


def test_reservoir_discovers_scene_and_views(tmp_path):
    _make_reservoir(tmp_path, dur1=12.0, dur2=10.0)
    ds = ReservoirDataset(root=tmp_path)
    scenes = list(ds.list_scenes())
    assert [s.scene_id for s in scenes] == ["Reservoir"]
    assert ds.get_scene("Reservoir").view_ids == ["view1", "view2"]


def test_reservoir_satisfies_protocol(tmp_path):
    _make_reservoir(tmp_path, dur1=12.0, dur2=10.0)
    ds = ReservoirDataset(root=tmp_path)
    assert isinstance(ds, DatasetAdapter)
    assert ds.cross_view_linking_mode == "appearance"
    assert ds.supports_cross_view_linking is True


def test_reservoir_iter_segments_aligns_to_shortest(tmp_path):
    """The longer view is capped to the shorter view's duration so both
    yield the same number of time-aligned segments (the '取最短对齐' rule)."""
    from mva.segmentation import SegmenterConfig

    _make_reservoir(tmp_path, dur1=20.0, dur2=12.0)
    ds = ReservoirDataset(root=tmp_path)
    # Shortest view (12s) is the alignment bound exposed in metadata.
    assert ds.get_scene("Reservoir").metadata["aligned_duration_sec"] == pytest.approx(
        12.0, abs=1.0
    )
    config = SegmenterConfig(window_sec=5.0, stride_sec=5.0, nframes_per_segment=4)
    segs1 = list(ds.iter_segments("Reservoir", "view1", config))
    segs2 = list(ds.iter_segments("Reservoir", "view2", config))
    # Despite view1 being 20s, it is truncated to 12s → equal segment counts.
    assert len(segs1) == len(segs2)
    assert segs1[-1].start_t == segs2[-1].start_t
    assert all(s.start_t < 12.0 for s in segs1)


def test_reservoir_unknown_scene_raises(tmp_path):
    _make_reservoir(tmp_path, dur1=12.0, dur2=10.0)
    ds = ReservoirDataset(root=tmp_path)
    with pytest.raises(KeyError):
        ds.get_scene("NoSuchScene")


def test_reservoir_qa_not_supported(tmp_path):
    _make_reservoir(tmp_path, dur1=12.0, dur2=10.0)
    ds = ReservoirDataset(root=tmp_path)
    with pytest.raises(NotImplementedError):
        list(ds.load_qa_pairs())


def test_reservoir_empty_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReservoirDataset(root=tmp_path)  # no <scene>/*.mp4 anywhere
