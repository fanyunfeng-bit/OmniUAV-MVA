"""Minimal L0 unit tests."""
from __future__ import annotations

import pytest

from mva.l0_stream import FileStreamSource, StreamSource


def test_file_source_raises_on_missing_file(tmp_path):
    missing = tmp_path / "nonexistent.mp4"
    with pytest.raises(FileNotFoundError):
        FileStreamSource(missing, view_id="drone-1")


def test_file_source_satisfies_protocol():
    assert hasattr(FileStreamSource, "__iter__")
    assert StreamSource is not None
