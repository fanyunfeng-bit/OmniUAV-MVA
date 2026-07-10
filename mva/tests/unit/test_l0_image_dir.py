"""Unit tests for ImageDirStreamSource.

Synthetic frames written to a tmp_path; verifies sorted-name iteration,
downsampling via sample_fps, ext filtering, missing-dir error, and the
Frame.telemetry passthrough invariant (§3.4 #1) is preserved.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from mva.contracts import Frame
from mva.l0_stream import ImageDirStreamSource


def _write_frame(path, color):
    img = np.full((4, 4, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_yields_frames_in_sorted_order(tmp_path):
    for i in range(3):
        _write_frame(tmp_path / f"{i:04d}.png", (i * 50, 0, 0))

    src = ImageDirStreamSource(tmp_path, view_id="D1", source_fps=10.0)
    frames = list(src)
    assert len(frames) == 3
    assert [f.view_id for f in frames] == ["D1", "D1", "D1"]
    # Timestamps spaced by 1/source_fps
    assert [round(f.t, 3) for f in frames] == [0.0, 0.1, 0.2]
    # All frames are valid Frame objects with telemetry=None default
    for f in frames:
        assert isinstance(f, Frame)
        assert f.telemetry is None
        assert f.image.shape == (4, 4, 3)


def test_sample_fps_downsamples(tmp_path):
    for i in range(10):
        _write_frame(tmp_path / f"{i:04d}.png", (0, 0, 0))

    # 10 frames at source 10 FPS, sampled at 5 FPS → keep every 2nd
    src = ImageDirStreamSource(
        tmp_path, view_id="D1", source_fps=10.0, sample_fps=5.0
    )
    frames = list(src)
    assert len(frames) == 5


def test_extension_filter(tmp_path):
    _write_frame(tmp_path / "good.png", (0, 0, 0))
    _write_frame(tmp_path / "alsogood.jpg", (0, 0, 0))
    (tmp_path / "ignore.txt").write_text("not an image")

    src = ImageDirStreamSource(tmp_path, view_id="D1")
    assert len(src) == 2


def test_missing_dir_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        ImageDirStreamSource(tmp_path / "nope", view_id="D1")


def test_invalid_source_fps_raises(tmp_path):
    with pytest.raises(ValueError):
        ImageDirStreamSource(tmp_path, view_id="D1", source_fps=0)


def test_corrupt_image_skipped(tmp_path):
    _write_frame(tmp_path / "0000.png", (0, 0, 0))
    # Fake .png that cv2 will fail to decode
    (tmp_path / "0001.png").write_bytes(b"not actually a png")
    _write_frame(tmp_path / "0002.png", (0, 0, 0))

    src = ImageDirStreamSource(tmp_path, view_id="D1", source_fps=2.0)
    frames = list(src)
    # The middle frame is skipped, only the two valid ones come through
    assert len(frames) == 2
