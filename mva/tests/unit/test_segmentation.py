"""Unit tests for mva.segmentation (Phase A of M2.8).

Exercises the two source-shape segmenters against synthetic fixtures so
tests run without the real ~50 GB datasets:
  - `iter_video_segments` against tiny cv2-written mp4s
  - `iter_image_dir_segments` against tmp_path-written PNGs

Also covers the adapter-level `iter_segments` Protocol method to confirm
both MVUEvalDataset and MatrixDataset delegate correctly.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mva.datasets import MatrixDataset, MVUEvalDataset
from mva.segmentation import (
    Segment,
    SegmenterConfig,
    iter_image_dir_segments,
    iter_video_segments,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _write_synthetic_mp4(
    path: Path, duration_sec: float, fps: float = 10.0,
) -> bool:
    """Write a tiny solid-color mp4 of the given duration. Returns False
    if cv2's mp4v writer is unavailable in this env (tests skip then)."""
    total = max(1, int(duration_sec * fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (16, 16))
    if not writer.isOpened():
        return False
    try:
        for i in range(total):
            shade = int(255 * (i / max(1, total - 1)))
            writer.write(np.full((16, 16, 3), shade, dtype=np.uint8))
    finally:
        writer.release()
    return True


def _write_synthetic_png_dir(
    dir_path: Path, n_frames: int = 30,
) -> None:
    """Write `n_frames` zero-padded PNGs into dir_path."""
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        cv2.imwrite(
            str(dir_path / f"{i:04d}.png"),
            np.full((16, 16, 3), (i * 8) % 256, dtype=np.uint8),
        )


# ----------------------------------------------------------------------
# SegmenterConfig
# ----------------------------------------------------------------------


def test_segmenter_config_defaults_match_sentrysearch():
    """10 s window, 10 s stride (non-overlap), 4 frames per segment."""
    c = SegmenterConfig()
    assert c.window_sec == 10.0
    assert c.stride_sec == 10.0
    assert c.nframes_per_segment == 4


def test_segmenter_config_validate_rejects_zero():
    with pytest.raises(ValueError):
        SegmenterConfig(window_sec=0).validate()
    with pytest.raises(ValueError):
        SegmenterConfig(stride_sec=0).validate()
    with pytest.raises(ValueError):
        SegmenterConfig(nframes_per_segment=0).validate()


# ----------------------------------------------------------------------
# iter_video_segments — synthetic mp4
# ----------------------------------------------------------------------


def test_video_segments_25s_yields_three_segments(tmp_path):
    mp4 = tmp_path / "clip.mp4"
    if not _write_synthetic_mp4(mp4, duration_sec=25.0):
        pytest.skip("cv2 mp4 writer unavailable")
    config = SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                             nframes_per_segment=2)
    segs = list(iter_video_segments(mp4, view_id="vid", config=config))
    # 25 s / 10 s stride → segments at [0,10) [10,20) [20,25); the last
    # one is 5 s ≥ MIN_SEGMENT_SEC, so it survives.
    assert len(segs) == 3
    assert segs[0].start_t == 0.0
    assert segs[0].end_t == 10.0
    assert segs[2].start_t == 20.0
    assert segs[2].end_t == 25.0


def test_video_segments_carry_source_uri_and_view_id(tmp_path):
    mp4 = tmp_path / "clip.mp4"
    if not _write_synthetic_mp4(mp4, duration_sec=12.0):
        pytest.skip("cv2 mp4 writer unavailable")
    segs = list(iter_video_segments(
        mp4, view_id="my_view",
        config=SegmenterConfig(nframes_per_segment=2),
    ))
    assert all(s.view_id == "my_view" for s in segs)
    assert all(s.source_uri == str(mp4) for s in segs)


def test_video_segments_frames_match_nframes_per_segment(tmp_path):
    mp4 = tmp_path / "clip.mp4"
    if not _write_synthetic_mp4(mp4, duration_sec=15.0):
        pytest.skip("cv2 mp4 writer unavailable")
    config = SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                             nframes_per_segment=4)
    segs = list(iter_video_segments(mp4, view_id="v", config=config))
    for seg in segs:
        assert isinstance(seg.frames, list)
        # Allow ≤ K in case some seeks land past EOF on the tail segment
        assert 0 < len(seg.frames) <= config.nframes_per_segment
        assert len(seg.frames) == len(seg.frame_indices)


def test_video_segments_segment_idx_is_monotonic(tmp_path):
    mp4 = tmp_path / "clip.mp4"
    if not _write_synthetic_mp4(mp4, duration_sec=30.0):
        pytest.skip("cv2 mp4 writer unavailable")
    segs = list(iter_video_segments(
        mp4, view_id="v",
        config=SegmenterConfig(window_sec=5.0, stride_sec=5.0,
                               nframes_per_segment=2),
    ))
    assert [s.segment_idx for s in segs] == list(range(len(segs)))


def test_video_segments_overlap_yields_more(tmp_path):
    mp4 = tmp_path / "clip.mp4"
    if not _write_synthetic_mp4(mp4, duration_sec=20.0):
        pytest.skip("cv2 mp4 writer unavailable")
    no_overlap = list(iter_video_segments(
        mp4, view_id="v",
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=2),
    ))
    overlap = list(iter_video_segments(
        mp4, view_id="v",
        config=SegmenterConfig(window_sec=10.0, stride_sec=5.0,
                               nframes_per_segment=2),
    ))
    assert len(overlap) > len(no_overlap)


def test_video_segments_short_video_yields_one_segment(tmp_path):
    """A 0.5 s clip < MIN_SEGMENT_SEC must still yield one segment so
    we don't silently drop legitimately tiny inputs."""
    mp4 = tmp_path / "tiny.mp4"
    if not _write_synthetic_mp4(mp4, duration_sec=0.5, fps=20.0):
        pytest.skip("cv2 mp4 writer unavailable")
    segs = list(iter_video_segments(
        mp4, view_id="v",
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=2),
    ))
    assert len(segs) == 1
    assert segs[0].start_t == 0.0


def test_video_segments_missing_file_returns_empty(tmp_path):
    """Bogus path → empty iterator, no exception (matches M2.7 behavior)."""
    segs = list(iter_video_segments(
        tmp_path / "nope.mp4", view_id="v", config=SegmenterConfig(),
    ))
    assert segs == []


def test_video_segments_invalid_config_raises(tmp_path):
    mp4 = tmp_path / "clip.mp4"
    if not _write_synthetic_mp4(mp4, duration_sec=10.0):
        pytest.skip("cv2 mp4 writer unavailable")
    with pytest.raises(ValueError):
        list(iter_video_segments(
            mp4, view_id="v",
            config=SegmenterConfig(window_sec=0.0),
        ))


# ----------------------------------------------------------------------
# iter_image_dir_segments — synthetic PNG dir
# ----------------------------------------------------------------------


def test_image_dir_segments_matrix_style(tmp_path):
    """40 PNG @ 2fps = 20 s; 10 s window/stride → 2 segments."""
    png_dir = tmp_path / "D1"
    _write_synthetic_png_dir(png_dir, n_frames=40)
    segs = list(iter_image_dir_segments(
        png_dir, view_id="D1", source_fps=2.0,
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=4),
    ))
    assert len(segs) == 2
    assert segs[0].start_t == 0.0
    assert segs[0].end_t == 10.0
    assert segs[1].start_t == 10.0
    # All sampled frames came from the first half of the file list
    assert all(0 <= idx < 20 for idx in segs[0].frame_indices)
    assert all(20 <= idx < 40 for idx in segs[1].frame_indices)


def test_image_dir_segments_carry_metadata(tmp_path):
    png_dir = tmp_path / "D1"
    _write_synthetic_png_dir(png_dir, n_frames=20)
    segs = list(iter_image_dir_segments(
        png_dir, view_id="D1", source_fps=2.0,
        config=SegmenterConfig(nframes_per_segment=2),
    ))
    assert segs[0].metadata["source_fps"] == 2.0
    assert segs[0].metadata["total_files"] == 20
    assert segs[0].source_uri == str(png_dir)


def test_image_dir_segments_empty_dir_yields_nothing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    segs = list(iter_image_dir_segments(
        empty, view_id="D1", source_fps=2.0, config=SegmenterConfig(),
    ))
    assert segs == []


def test_image_dir_segments_short_sequence_one_segment(tmp_path):
    """4 PNG @ 2fps = 2 s, single short segment."""
    png_dir = tmp_path / "short"
    _write_synthetic_png_dir(png_dir, n_frames=4)
    segs = list(iter_image_dir_segments(
        png_dir, view_id="D1", source_fps=2.0,
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=4),
    ))
    assert len(segs) == 1
    assert len(segs[0].frame_indices) == 4


def test_image_dir_segments_bad_source_fps_raises(tmp_path):
    png_dir = tmp_path / "D1"
    _write_synthetic_png_dir(png_dir, n_frames=10)
    with pytest.raises(ValueError):
        list(iter_image_dir_segments(
            png_dir, view_id="D1", source_fps=0.0,
            config=SegmenterConfig(),
        ))


# ----------------------------------------------------------------------
# Adapter-level integration: iter_segments via Protocol
# ----------------------------------------------------------------------


def _make_mvu_with_video(root: Path, duration_sec: float) -> None:
    """MVU-Eval layout with one REAL synthetic mp4."""
    root.mkdir(parents=True, exist_ok=True)
    _write_synthetic_mp4(root / "video_a.mp4", duration_sec=duration_sec)
    qa = {"0": {"video_paths": ["video_a.mp4"], "question": "Q?",
                "options": ["A.", "B."], "ground_truth": "A",
                "task": "Counting"}}
    (root / "MVU_Eval_QAs.json").write_text(json.dumps(qa))


def test_mvu_adapter_iter_segments_yields_segments(tmp_path):
    _make_mvu_with_video(tmp_path, duration_sec=15.0)
    ds = MVUEvalDataset(root=tmp_path)
    segs = list(ds.iter_segments(
        "qa-0", view_id="video_a.mp4",
        config=SegmenterConfig(nframes_per_segment=2),
    ))
    assert len(segs) >= 1
    assert all(isinstance(s, Segment) for s in segs)
    assert segs[0].view_id == "video_a.mp4"


def test_mvu_adapter_iter_segments_missing_video_silent(tmp_path):
    """A QA referencing a missing video file → empty iterator, no raise."""
    qa = {"0": {"video_paths": ["does_not_exist.mp4"], "question": "Q",
                "options": ["A.", "B."], "ground_truth": "A",
                "task": "Counting"}}
    (tmp_path / "MVU_Eval_QAs.json").write_text(json.dumps(qa))
    ds = MVUEvalDataset(root=tmp_path)
    segs = list(ds.iter_segments(
        "qa-0", view_id="does_not_exist.mp4",
        config=SegmenterConfig(nframes_per_segment=2),
    ))
    assert segs == []


def test_matrix_adapter_iter_segments_yields_segments(tmp_path):
    """MATRIX-style scene: PNG sequence at 2 fps."""
    scene_dir = tmp_path / "MINI"
    for view in ("D1", "D2"):
        _write_synthetic_png_dir(
            scene_dir / "image_subsets" / view, n_frames=40,
        )
    ds = MatrixDataset(root=tmp_path)
    segs = list(ds.iter_segments(
        "MINI", view_id="D1",
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=4),
    ))
    # 40 PNG / 2 fps = 20 s → 2 segments at default config
    assert len(segs) == 2
    assert segs[0].view_id == "D1"


def test_matrix_adapter_iter_segments_missing_view_silent(tmp_path):
    scene_dir = tmp_path / "MINI"
    _write_synthetic_png_dir(
        scene_dir / "image_subsets" / "D1", n_frames=10,
    )
    ds = MatrixDataset(root=tmp_path)
    segs = list(ds.iter_segments(
        "MINI", view_id="D99",  # nonexistent
        config=SegmenterConfig(nframes_per_segment=2),
    ))
    assert segs == []
