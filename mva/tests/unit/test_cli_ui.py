"""Unit tests for `mva ui` plumbing (M5.4).

Covers the gradio-free helpers (attachment dispatch, segment hit
extraction, label formatting, ffmpeg invocation) plus a `build_app`
smoke gated by `importorskip("gradio")` so CI without [ui] extra
still passes.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from mva.cli.ui import (
    _build_attachments,
    _extract_segment_hits,
    _is_segment_hit,
    _load_demo_answers,
    _normalize_question,
    _parse_segment_idx_from_label,
    _segment_label,
    extract_clip,
)
from mva.contracts import Attachment


# -------- demo-answers soft override --------

def test_normalize_question_strips_punct_and_space():
    # full/half-width '?' , spaces → same key
    a = _normalize_question("有几艘船？")
    assert a == _normalize_question("有几艘船?")
    assert a == _normalize_question("  有几艘船 ")
    assert a == "有几艘船"


def test_normalize_question_is_exact_not_substring():
    # view-scoped question must NOT collapse to the bare one
    assert _normalize_question("view1 里有几艘船") != _normalize_question("有几艘船")


def test_load_demo_answers_dict(tmp_path):
    f = tmp_path / "demo.json"
    f.write_text('{"有几艘船": "共 10 艘船。"}', encoding="utf-8")
    m = _load_demo_answers(str(f))
    assert m[_normalize_question("有几艘船?")] == "共 10 艘船。"


def test_load_demo_answers_list_form(tmp_path):
    f = tmp_path / "demo.json"
    f.write_text('[{"match": "都看到了哪些目标", "answer": "x"}]', encoding="utf-8")
    m = _load_demo_answers(str(f))
    assert m[_normalize_question("都看到了哪些目标？")] == "x"


def test_load_demo_answers_none_and_bad_are_empty(tmp_path):
    assert _load_demo_answers(None) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert _load_demo_answers(str(bad)) == {}      # non-fatal


# -------- _build_attachments --------

def test_build_attachments_dispatches_image_by_ext():
    out = _build_attachments(["/tmp/foo.jpg", "/tmp/bar.png"])
    assert len(out) == 2
    assert all(isinstance(a, Attachment) for a in out)
    assert [a.kind for a in out] == ["image", "image"]
    assert out[0].path == Path("/tmp/foo.jpg")
    assert out[0].label == "foo.jpg"


def test_build_attachments_dispatches_video_by_ext():
    out = _build_attachments(["/tmp/x.mp4", "/tmp/y.mov", "/tmp/z.webm"])
    assert [a.kind for a in out] == ["video", "video", "video"]


def test_build_attachments_unknown_ext_defaults_to_image():
    """Unknown extensions default to image (Qwen-VL processes as image)."""
    out = _build_attachments(["/tmp/data.bin"])
    assert out[0].kind == "image"


def test_build_attachments_empty_and_none_safe():
    assert _build_attachments([]) == []
    assert _build_attachments(None) == []  # type: ignore[arg-type]


def test_build_attachments_accepts_filedata_like_objects():
    """Gradio MultimodalTextbox may yield FileData-like objects with .name."""
    class FakeFile:
        name = "/tmp/uploaded.mp4"
    out = _build_attachments([FakeFile()])
    assert out[0].kind == "video"
    assert out[0].path == Path("/tmp/uploaded.mp4")


# -------- _is_segment_hit / _extract_segment_hits --------

def test_is_segment_hit_requires_source_uri_and_times():
    assert _is_segment_hit({
        "source_uri": "/v.mp4", "start_t": 0.0, "end_t": 10.0,
    }) is True
    assert _is_segment_hit({"source_uri": None, "start_t": 0, "end_t": 10}) is False
    assert _is_segment_hit({"start_t": 0, "end_t": 10}) is False
    assert _is_segment_hit({"source_uri": "/v.mp4", "end_t": 10}) is False


def test_extract_segment_hits_walks_list_results():
    class _Inv:
        def __init__(self, result):
            self.result = result
    class _Res:
        invocations = [
            _Inv([
                {"source_uri": "/a.mp4", "start_t": 0.0, "end_t": 10.0, "view_id": "v1"},
                {"source_uri": "/b.mp4", "start_t": 5.0, "end_t": 15.0, "view_id": "v2"},
            ]),
        ]
    hits = _extract_segment_hits(_Res())
    assert len(hits) == 2
    assert hits[0]["source_uri"] == "/a.mp4"


def test_extract_segment_hits_dedupes_across_invocations():
    """If find_segment_by_description AND get_segment_clip both return the
    same segment, list it once."""
    class _Inv:
        def __init__(self, result):
            self.result = result
    seg = {"source_uri": "/a.mp4", "start_t": 0.0, "end_t": 10.0, "view_id": "v1"}
    class _Res:
        invocations = [_Inv([seg]), _Inv(seg)]
    hits = _extract_segment_hits(_Res())
    assert len(hits) == 1


def test_extract_segment_hits_skips_non_segments():
    class _Inv:
        def __init__(self, result):
            self.result = result
    class _Res:
        invocations = [
            _Inv("just a string"),
            _Inv(42),
            _Inv([{"answer": "no segment fields"}]),
            _Inv([{"source_uri": "/v.mp4", "start_t": 0, "end_t": 5}]),
        ]
    hits = _extract_segment_hits(_Res())
    assert len(hits) == 1


# -------- _segment_label / _parse_segment_idx_from_label --------

def test_segment_label_with_classes_list():
    label = _segment_label({
        "view_id": "qa805::cam-front",
        "source_uri": "/data/qa805/front.mp4",
        "start_t": 12.0,
        "end_t": 22.0,
        "detected_classes": ["person", "car"],
    }, idx=2)
    assert label.startswith("#2  front.mp4")
    assert "[qa805::cam-front]" in label
    assert "12.0-22.0s" in label
    assert "(person,car)" in label


def test_segment_label_without_classes():
    label = _segment_label({
        "view_id": "v1",
        "source_uri": "/x.mp4",
        "start_t": 0.0,
        "end_t": 10.0,
    }, idx=0)
    assert label == "#0  x.mp4 [v1] 0.0-10.0s"


def test_parse_segment_idx_from_label_round_trip():
    seg = {"view_id": "v1", "source_uri": "/x.mp4", "start_t": 0, "end_t": 1}
    label = _segment_label(seg, idx=7)
    assert _parse_segment_idx_from_label(label) == 7


def test_parse_segment_idx_from_label_rejects_garbage():
    assert _parse_segment_idx_from_label("") is None
    assert _parse_segment_idx_from_label("no hash here") is None
    assert _parse_segment_idx_from_label("#abc foo") is None
    assert _parse_segment_idx_from_label(None) is None  # type: ignore[arg-type]


# -------- extract_clip --------

def test_extract_clip_invokes_ffmpeg_with_seek_and_duration(tmp_path):
    """Happy path: source exists, ffmpeg present, subprocess called with
    expected args."""
    source = tmp_path / "src.mp4"
    source.write_bytes(b"fake mp4")  # exists check only — ffmpeg is mocked

    with patch("mva.cli.ui.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("mva.cli.ui.subprocess.run") as run_mock:
        out = extract_clip(str(source), 5.0, 15.0, output_dir=str(tmp_path))

    run_mock.assert_called_once()
    cmd = run_mock.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "-ss" in cmd
    assert "5.000" in cmd  # start
    assert "10.000" in cmd  # duration end-start
    assert "-c" in cmd and "copy" in cmd
    assert out.endswith(".mp4")
    assert str(tmp_path) in out


def test_extract_clip_missing_ffmpeg_raises():
    with patch("mva.cli.ui.shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="ffmpeg"):
            extract_clip("/whatever.mp4", 0, 10)


def test_extract_clip_missing_source_raises(tmp_path):
    with patch("mva.cli.ui.shutil.which", return_value="/usr/bin/ffmpeg"):
        with pytest.raises(FileNotFoundError, match="source"):
            extract_clip(str(tmp_path / "nope.mp4"), 0, 10)


def test_extract_clip_image_dir_calls_ffmpeg(tmp_path):
    """MATRIX-style PNG sequence source → stitched into video via ffmpeg."""
    import cv2
    import numpy as np
    src_dir = tmp_path / "matrix-d1"
    src_dir.mkdir()
    for i in range(10):
        cv2.imwrite(
            str(src_dir / f"{i:04d}.png"),
            np.full((16, 16, 3), i * 25, dtype=np.uint8),
        )
    with patch("mva.cli.ui.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("mva.cli.ui.subprocess.run") as run_mock:
        extract_clip(str(src_dir), 0, 5, output_dir=str(tmp_path))
    assert run_mock.called
    cmd = run_mock.call_args[0][0]
    assert "-framerate" in cmd
    assert "libx264" in cmd


def test_extract_clip_zero_duration_floors_to_100ms(tmp_path):
    """end_t == start_t shouldn't crash — bump to 0.1s min to satisfy ffmpeg."""
    source = tmp_path / "src.mp4"
    source.write_bytes(b"fake")
    with patch("mva.cli.ui.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("mva.cli.ui.subprocess.run") as run_mock:
        extract_clip(str(source), 5.0, 5.0, output_dir=str(tmp_path))
    cmd = run_mock.call_args[0][0]
    t_idx = cmd.index("-t")
    assert cmd[t_idx + 1] == "0.100"


def test_extract_clip_ffmpeg_failure_propagates(tmp_path):
    source = tmp_path / "src.mp4"
    source.write_bytes(b"fake")
    with patch("mva.cli.ui.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("mva.cli.ui.subprocess.run",
               side_effect=subprocess.CalledProcessError(1, ["ffmpeg"])):
        with pytest.raises(subprocess.CalledProcessError):
            extract_clip(str(source), 0, 5)


# -------- build_app (gradio-gated) --------

def test_build_app_constructs_blocks_smoke():
    """Smoke: build_app returns a gr.Blocks without crashing.
    Skipped if gradio not installed."""
    gr = pytest.importorskip("gradio")
    # Minimal fake service — build_app only reads .llm_model + .vstore
    class _FakeService:
        llm_model = "fake/Model"
        vstore = None
    app = None
    try:
        from mva.cli.ui import build_app
        app = build_app(_FakeService(), db_path="/tmp/fake.duckdb")
    finally:
        if app is not None:
            assert isinstance(app, gr.Blocks)


def test_build_app_with_memory_state_smoke():
    """Smoke: build_app still returns gr.Blocks after gr.State(ConversationMemory()) is added.
    Skipped if gradio not installed."""
    gr = pytest.importorskip("gradio")
    # Minimal fake service — build_app only reads .llm_model + .vstore
    class _FakeService:
        llm_model = "fake/Model"
        vstore = None
    app = None
    try:
        from mva.cli.ui import build_app
        app = build_app(_FakeService(), db_path="/tmp/fake.duckdb")
    finally:
        if app is not None:
            assert isinstance(app, gr.Blocks)


# -------- LiveCapture (Phase 1 streaming panels) --------


def _write_mp4(path: Path, duration_sec: float, fps: float = 10.0) -> bool:
    """Tiny solid-color mp4 so cv2 can read duration + seek. False if the
    mp4v writer is unavailable (test skips then)."""
    import cv2
    import numpy as np
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    if not writer.isOpened():
        return False
    try:
        for i in range(max(1, int(duration_sec * fps))):
            writer.write(np.full((24, 32, 3), (i * 7) % 255, dtype=np.uint8))
    finally:
        writer.release()
    return True


class _FakeStore:
    def __init__(self, rows):
        self._rows = rows

    def execute_readonly(self, _sql):
        return self._rows


class _ServiceWithStore:
    llm_model = "fake/Model"
    vstore = None

    def __init__(self, rows):
        self.store = _FakeStore(rows)


def test_live_capture_empty_without_store():
    from mva.cli.ui import LiveCapture

    class _NoStore:
        pass
    assert LiveCapture.from_service(_NoStore()).views == []


def test_live_capture_discovers_views_from_segments(tmp_path):
    from mva.cli.ui import LiveCapture
    a, b = tmp_path / "view1.mp4", tmp_path / "view2.mp4"
    if not (_write_mp4(a, 3.0) and _write_mp4(b, 2.0)):
        pytest.skip("cv2 mp4 writer unavailable")
    svc = _ServiceWithStore([
        {"view_id": "view1", "source_uri": str(a)},
        {"view_id": "view2", "source_uri": str(b)},
    ])
    live = LiveCapture.from_service(svc)
    assert live.views == ["view1", "view2"]


def test_live_capture_skips_missing_files(tmp_path):
    from mva.cli.ui import LiveCapture
    a = tmp_path / "view1.mp4"
    if not _write_mp4(a, 2.0):
        pytest.skip("cv2 mp4 writer unavailable")
    svc = _ServiceWithStore([
        {"view_id": "view1", "source_uri": str(a)},
        {"view_id": "ghost", "source_uri": str(tmp_path / "nope.mp4")},
    ])
    assert LiveCapture.from_service(svc).views == ["view1"]


def test_live_capture_loop_duration_is_shortest(tmp_path):
    from mva.cli.ui import LiveCapture
    a, b = tmp_path / "v1.mp4", tmp_path / "v2.mp4"
    if not (_write_mp4(a, 3.0) and _write_mp4(b, 2.0)):
        pytest.skip("cv2 mp4 writer unavailable")
    live = LiveCapture.from_service(_ServiceWithStore([
        {"view_id": "v1", "source_uri": str(a)},
        {"view_id": "v2", "source_uri": str(b)},
    ]))
    assert live.loop_duration == pytest.approx(2.0, abs=0.5)  # shortest


def test_live_capture_current_frames_returns_rgb_arrays(tmp_path):
    from mva.cli.ui import LiveCapture
    a = tmp_path / "v1.mp4"
    if not _write_mp4(a, 3.0):
        pytest.skip("cv2 mp4 writer unavailable")
    live = LiveCapture.from_service(_ServiceWithStore([
        {"view_id": "v1", "source_uri": str(a)},
    ]))
    frames = live.current_frames()
    assert set(frames) == {"v1"}
    assert frames["v1"] is not None and frames["v1"].ndim == 3


def test_live_capture_playhead_wraps_on_shared_clock(monkeypatch):
    from mva.cli import ui
    live = ui.LiveCapture({})
    live._loop = 10.0
    live._t0 = 100.0
    monkeypatch.setattr(ui.time, "time", lambda: 125.0)
    assert live._playhead() == pytest.approx(5.0)   # (125-100) % 10


def test_live_capture_status_line_marks_live():
    from mva.cli.ui import LiveCapture
    live = LiveCapture({})
    assert "LIVE" in live.status_line()


def test_build_app_with_live_panels_smoke(tmp_path):
    """build_app wires gr.Timer + per-view gr.Image when views are present."""
    gr = pytest.importorskip("gradio")
    from mva.cli.ui import build_app
    a = tmp_path / "view1.mp4"
    if not _write_mp4(a, 2.0):
        pytest.skip("cv2 mp4 writer unavailable")
    svc = _ServiceWithStore([{"view_id": "view1", "source_uri": str(a)}])
    app = build_app(svc, db_path=str(tmp_path / "x.duckdb"))
    assert isinstance(app, gr.Blocks)


# -------- fresh-DB live-ingest source resolution (--dataset/--scene) --------


def _make_pcl_sim_scene(root: Path) -> bool:
    scene = root / "Reservoir"
    scene.mkdir(parents=True, exist_ok=True)
    return _write_mp4(scene / "view1.mp4", 3.0) and _write_mp4(scene / "view2.mp4", 2.0)


def test_adapter_view_paths_empty_without_dataset():
    import argparse
    from mva.cli.ui import _adapter_view_paths
    ns = argparse.Namespace(dataset=None, scene=None, dataset_root=None)
    assert _adapter_view_paths(ns) == {}


def test_adapter_view_paths_resolves_from_scene(tmp_path):
    import argparse
    from mva.cli.ui import _adapter_view_paths
    if not _make_pcl_sim_scene(tmp_path):
        pytest.skip("cv2 mp4 writer unavailable")
    ns = argparse.Namespace(dataset="pcl-sim", scene="Reservoir", dataset_root=tmp_path)
    paths = _adapter_view_paths(ns)
    assert set(paths) == {"view1", "view2"}
    assert all(Path(p).is_file() for p in paths.values())


def test_resolve_live_capture_fresh_db_falls_back_to_adapter(tmp_path):
    import argparse
    from mva.cli.ui import _resolve_live_capture
    from mva.l5_state import WorldStateStore
    if not _make_pcl_sim_scene(tmp_path):
        pytest.skip("cv2 mp4 writer unavailable")

    class _Svc:
        store = WorldStateStore(db_path=":memory:")          # empty DB → no views
    ns = argparse.Namespace(dataset="pcl-sim", scene="Reservoir",
                            dataset_root=tmp_path, live_ingest=True)
    live = _resolve_live_capture(_Svc(), ns)
    assert live.views == ["view1", "view2"]                  # sourced from adapter


# -------- CLI registration --------

def test_cli_ui_subparser_registered():
    """`mva ui --help` builds without crash and includes --port flag."""
    from mva.cli.__main__ import build_parser
    parser = build_parser()
    # Parse a valid --help-style invocation; the subparser must be wired.
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["ui", "--help"])
    assert exc_info.value.code == 0
