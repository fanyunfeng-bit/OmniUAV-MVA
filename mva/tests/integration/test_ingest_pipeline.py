"""Integration test for `mva ingest` core loop (Phase C of M2.8).

Drives the full pipeline with:
  - Mock MultimodalEmbedder (deterministic SHA-256-seeded vectors, no
    16 GB model download)
  - FakeDetector (canned Detection lists, no ultralytics dependency)
  - Real WorldStateStore (`:memory:` DuckDB) + real VectorStore (tmp
    PersistentClient)
  - Synthetic MATRIX + MVU-Eval mini-fixtures via cv2 writes

Asserts the cross-store invariants that the M2.8 design depends on:
  1. Every DuckDB segment row has a `embed_chroma_id` pointing at an
     existing ChromaDB segment-kind vector.
  2. Every DuckDB tracklet row's `embedding_ref` (when set) points at an
     existing ChromaDB bbox-kind vector with matching segment_idx.
  3. `--segments-per-view` caps work per view independently, ensuring
     equal temporal coverage across views.
  4. `--no-detect` skips detection rows; `--no-embed-bboxes` keeps
     detection rows but leaves `embedding_ref=NULL`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from mva.cli.ingest import ingest_scene
from mva.datasets import MatrixDataset, MVUEvalDataset
from mva.l1_perception import Detection
from mva.l5_state import MultimodalEmbedder, VectorStore, WorldStateStore
from mva.segmentation import SegmenterConfig


# ----------------------------------------------------------------------
# Fixtures: tiny synthetic datasets + fake detector
# ----------------------------------------------------------------------


class FakeDetector:
    """Returns canned Detection lists per frame. Deterministic for tests."""

    def __init__(self, per_frame: List[List[Detection]]):
        self._per_frame = per_frame
        self._call = 0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        out = self._per_frame[self._call % len(self._per_frame)]
        self._call += 1
        return out


def _det(cls: str, conf: float = 0.9) -> Detection:
    return Detection(
        bbox=(2.0, 2.0, 10.0, 10.0), class_id=0,
        class_name=cls, confidence=conf,
    )


def _write_mp4(path: Path, duration_sec: float, fps: float = 10.0) -> bool:
    total = max(1, int(duration_sec * fps))
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (16, 16),
    )
    if not writer.isOpened():
        return False
    try:
        for i in range(total):
            shade = int(255 * (i / max(1, total - 1)))
            writer.write(np.full((16, 16, 3), shade, dtype=np.uint8))
    finally:
        writer.release()
    return True


def _make_mvu(root: Path, duration_sec: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if not _write_mp4(root / "video_a.mp4", duration_sec=duration_sec):
        pytest.skip("cv2 mp4 writer unavailable")
    qa = {"0": {"video_paths": ["video_a.mp4"], "question": "Q?",
                "options": ["A.", "B."], "ground_truth": "A",
                "task": "Counting"}}
    (root / "MVU_Eval_QAs.json").write_text(json.dumps(qa))


def _make_matrix(root: Path, n_frames_per_view: int = 40) -> None:
    """Synthetic MATRIX with D1 + D2, n_frames PNGs each."""
    scene = root / "MINI"
    for view in ("D1", "D2"):
        view_dir = scene / "image_subsets" / view
        view_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_frames_per_view):
            cv2.imwrite(
                str(view_dir / f"{i:04d}.png"),
                np.full((16, 16, 3), i * 4 % 256, dtype=np.uint8),
            )


@pytest.fixture
def stores(tmp_path):
    """Fresh DuckDB :memory: + tmp ChromaDB."""
    store = WorldStateStore(":memory:")
    vstore = VectorStore(persist_dir=str(tmp_path / "chroma"))
    yield store, vstore
    store.close()


@pytest.fixture
def mock_embedder():
    return MultimodalEmbedder(model_path=None, dim=64)


# ----------------------------------------------------------------------
# MVU-Eval video path
# ----------------------------------------------------------------------


def test_ingest_video_writes_segments_and_links_to_chroma(
    tmp_path, stores, mock_embedder,
):
    """Happy path: 25 s video, default config → 3 segments; every
    segment row in DuckDB references an existing ChromaDB vector."""
    _make_mvu(tmp_path, duration_sec=25.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    fake = FakeDetector([[_det("person")], [_det("car"), _det("person")]])

    stats = ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    assert stats["segments"] == 3, f"expected 3 segments, got {stats}"
    assert stats["detections"] > 0
    # M3.1: one embedding per *track* (mean-pool over frames), not per
    # detection — so bbox_embeddings == tracklets and ≤ detections.
    assert stats["tracklets"] > 0
    assert stats["bbox_embeddings"] == stats["tracklets"]
    assert stats["bbox_embeddings"] <= stats["detections"]

    # Every DuckDB segment row has a chroma id that resolves
    segs = store.query_segments(view_id="video_a.mp4")
    assert len(segs) == 3
    for s in segs:
        assert s["embed_chroma_id"] is not None
        rev = store.get_segment_by_chroma_id(s["embed_chroma_id"])
        assert rev is not None
        assert rev["segment_idx"] == s["segment_idx"]
        assert s["nframes_sampled"] == 2
        assert s["detected_classes"] is not None
        assert s["detected_counts"]


def test_ingest_no_detect_writes_segments_but_no_tracklets(
    tmp_path, stores, mock_embedder,
):
    """--no-detect: segments still embedded, no tracklet rows written."""
    _make_mvu(tmp_path, duration_sec=15.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    stats = ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=None, embed_bboxes=False,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    assert stats["segments"] >= 1
    assert stats["detections"] == 0
    assert stats["bbox_embeddings"] == 0

    tracklets = store.query_tracklets("video_a.mp4")
    assert tracklets == []
    # Segments still landed
    assert store.query_segments(view_id="video_a.mp4")


def test_ingest_no_embed_bboxes_keeps_detections_but_no_bbox_vectors(
    tmp_path, stores, mock_embedder,
):
    """--no-embed-bboxes: detections recorded, but embedding_ref stays NULL
    and no bbox vector lands in ChromaDB."""
    _make_mvu(tmp_path, duration_sec=12.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    fake = FakeDetector([[_det("person"), _det("car")]])
    initial_chroma = vstore.collection.count()
    stats = ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=False,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    assert stats["segments"] >= 1
    assert stats["detections"] > 0
    assert stats["bbox_embeddings"] == 0

    tracklets = store.query_tracklets("video_a.mp4")
    # M3.1: row count is one-per-track (≤ detections), and all have NULL
    # embedding_ref because --no-embed-bboxes was passed.
    assert len(tracklets) == stats["tracklets"]
    assert stats["tracklets"] > 0
    assert all(r["embedding_ref"] is None for r in tracklets)
    # ChromaDB gained exactly stats["segments"] entries (no bbox vectors)
    assert vstore.collection.count() - initial_chroma == stats["segments"]


def test_segments_per_view_caps_each_view_independently(
    tmp_path, stores, mock_embedder,
):
    """Per-view cap ensures each view gets the same number of segments."""
    _make_matrix(tmp_path, n_frames_per_view=200)  # 100s @ 2fps → 10 seg/view
    ds = MatrixDataset(root=tmp_path)
    store, vstore = stores

    fake = FakeDetector([[_det("person")]])

    stats = ingest_scene(
        adapter=ds, scene_id="MINI", view_ids=["D1", "D2"],
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=3,
    )
    assert stats["segments"] == 6  # 3 per view × 2 views
    d1 = store.query_segments(view_id="D1")
    d2 = store.query_segments(view_id="D2")
    assert len(d1) == 3, f"expected 3 segments in D1, got {len(d1)}"
    assert len(d2) == 3, f"expected 3 segments in D2, got {len(d2)}"


def test_segments_per_view_short_view_doesnt_affect_long_view(
    tmp_path, stores, mock_embedder,
):
    """If one view has fewer segments than the cap, other views still
    get their full allocation — per-view caps are independent."""
    # D1 = 4 PNGs @ 2fps = 2s = 1 segment max
    # D2 = 100 PNGs = 50s = 5 segments
    scene = tmp_path / "MIXED"
    for view, n in (("D1", 4), ("D2", 100)):
        view_dir = scene / "image_subsets" / view
        view_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cv2.imwrite(
                str(view_dir / f"{i:04d}.png"),
                np.full((16, 16, 3), i % 256, dtype=np.uint8),
            )
    ds = MatrixDataset(root=tmp_path)
    store, vstore = stores

    fake = FakeDetector([[_det("person")]])
    stats = ingest_scene(
        adapter=ds, scene_id="MIXED", view_ids=["D1", "D2"],
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=5,
    )
    # D1 yields only 1 (exhausted), D2 gets its full 5
    assert stats["segments"] == 6
    d1 = store.query_segments(view_id="D1")
    d2 = store.query_segments(view_id="D2")
    assert len(d1) == 1
    assert len(d2) == 5


# ----------------------------------------------------------------------
# MATRIX PNG path
# ----------------------------------------------------------------------


def test_ingest_matrix_yields_segments_per_view(tmp_path, stores, mock_embedder):
    _make_matrix(tmp_path, n_frames_per_view=40)  # 20 s @ 2fps
    ds = MatrixDataset(root=tmp_path)
    store, vstore = stores

    fake = FakeDetector([[_det("person")]])
    stats = ingest_scene(
        adapter=ds, scene_id="MINI", view_ids=["D1"],
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=4),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    assert stats["segments"] == 2
    # PNG source_uri ends with the view directory path
    segs = store.query_segments(view_id="D1")
    assert all("image_subsets/D1" in s["source_uri"] for s in segs)


def test_ingest_bbox_chroma_metadata_carries_class_and_bbox(
    tmp_path, stores, mock_embedder,
):
    """Per-bbox ChromaDB entries must carry the class name + bbox coords +
    parent segment_idx — that's what makes retrieval results actionable."""
    _make_mvu(tmp_path, duration_sec=12.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    fake = FakeDetector([[_det("person", conf=0.88)]])
    ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    bbox_rows = vstore.query(
        query_vector=mock_embedder.encode_image(np.zeros((8, 8, 3), dtype=np.uint8)),
        vector_type="reid", top_k=10,
    )
    assert len(bbox_rows) > 0
    md = bbox_rows[0]["metadata"]
    assert md["vector_kind"] == "bbox"
    assert md["class_name"] == "person"
    assert "bbox_x1" in md and "bbox_y1" in md
    assert "segment_idx" in md
    assert md["confidence"] == pytest.approx(0.88)


def test_ingest_segment_chroma_metadata_carries_time_window(
    tmp_path, stores, mock_embedder,
):
    """Segment-level ChromaDB entries must carry (start_t, end_t, source_uri)
    so retrieval can `ffmpeg -ss start_t -t (end_t - start_t)` back to clip."""
    _make_mvu(tmp_path, duration_sec=15.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=None, embed_bboxes=False,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    seg_rows = vstore.query(
        query_vector=mock_embedder.encode_text("anything"),
        vector_type="frame", top_k=10,
    )
    assert len(seg_rows) >= 1
    md = seg_rows[0]["metadata"]
    assert md["vector_kind"] == "segment"
    assert "start_t" in md and "end_t" in md
    assert md["source_uri"].endswith("video_a.mp4")
    assert md["end_t"] > md["start_t"]


# ----------------------------------------------------------------------
# M3.0 — cross-view linking integrated into ingest (dual mode)
# ----------------------------------------------------------------------


def test_ingest_matrix_writes_synchronized_cross_view_links(
    tmp_path, stores, mock_embedder,
):
    """MATRIX `cross_view_linking_mode="synchronized"` → ingest's L2
    post-pass writes `cross_view_links` rows with `created_by` ∈
    {geometric, geometric+appearance}. The mock embedder returns
    SHA-256-seeded vectors so cosine between distinct detections varies
    — usually below the 0.6 appearance threshold, so we expect a mix
    (or all `geometric` if no pair clears the cosine bar)."""
    _make_matrix(tmp_path, n_frames_per_view=40)
    ds = MatrixDataset(root=tmp_path)
    store, vstore = stores

    # FakeDetector returns one person per frame so D1 and D2 each get
    # 4 detections per segment (K=4) — Hungarian has work to do.
    fake = FakeDetector([[_det("person")]])

    stats = ingest_scene(
        adapter=ds, scene_id="MINI", view_ids=["D1", "D2"],
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    assert "cross_view_links" in stats
    assert stats["cross_view_links"] >= 0   # at least the L2 step ran
    links = store.query_cross_view_links()
    for link in links:
        assert link.created_by in ("geometric", "geometric+appearance")
        # Cross-view link is between distinct views (Pydantic invariant)
        assert len({v for v, _ in link.view_observations}) == 2


def test_ingest_mvu_writes_appearance_cross_view_links(
    tmp_path, stores, mock_embedder,
):
    """MVU-Eval `cross_view_linking_mode="appearance"` → ingest's L2
    post-pass uses pure-cosine matching. Tested with two videos that
    share class labels but mock embeddings; we lower threshold via
    smaller --segments-per-view here so the test focuses on plumbing not
    quality."""
    # Two videos at the same QA → cross-video appearance matching can fire
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    for name in ("video_a.mp4", "video_b.mp4"):
        if not _write_mp4(root / name, duration_sec=12.0):
            pytest.skip("cv2 mp4 writer unavailable")
    qa = {"0": {"video_paths": ["video_a.mp4", "video_b.mp4"],
                "question": "Q?", "options": ["A.", "B."],
                "ground_truth": "A", "task": "Counting"}}
    (root / "MVU_Eval_QAs.json").write_text(json.dumps(qa))

    ds = MVUEvalDataset(root=root)
    store, vstore = stores

    fake = FakeDetector([[_det("person")]])
    stats = ingest_scene(
        adapter=ds, scene_id="qa-0",
        view_ids=["video_a.mp4", "video_b.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    assert "cross_view_links" in stats
    links = store.query_cross_view_links()
    # Whether any links fire depends on the mock embedder's seed; the
    # invariant we lock is "if any link exists, it's appearance-typed".
    for link in links:
        assert link.created_by == "appearance"
        assert len({v for v, _ in link.view_observations}) == 2


def test_ingest_mvu_skips_l2_when_no_bbox_embeddings(
    tmp_path, stores, mock_embedder, capsys,
):
    """MVU-Eval appearance mode requires bbox embeddings. With
    --no-embed-bboxes we should warn + skip L2, not crash."""
    _make_mvu(tmp_path, duration_sec=12.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    fake = FakeDetector([[_det("person")]])
    stats = ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=False,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    assert stats["cross_view_links"] == 0
    # Single-view QA → no link possible anyway; the WARN only fires when
    # multi-view + no embeddings. Just ensure the path is exercised.


def test_ingest_appearance_threshold_param_overrides_default(
    tmp_path, stores, mock_embedder,
):
    """`appearance_threshold` is plumbed through CLI → ingest_scene →
    _link_cross_views → linker constructor. Same data, low threshold
    must yield ≥ as many links as high threshold."""
    _make_mvu(tmp_path, duration_sec=12.0)
    # Add a second video so cross-video matching has candidates
    if not _write_mp4(tmp_path / "video_b.mp4", duration_sec=12.0):
        pytest.skip("cv2 mp4 writer unavailable")
    qa = json.load(open(tmp_path / "MVU_Eval_QAs.json"))
    qa["0"]["video_paths"] = ["video_a.mp4", "video_b.mp4"]
    (tmp_path / "MVU_Eval_QAs.json").write_text(json.dumps(qa))

    ds = MVUEvalDataset(root=tmp_path)
    fake = FakeDetector([[_det("person")]])

    # Strict threshold — mock embedder cosines are random, very few will pass
    store_strict, vstore_strict = (
        WorldStateStore(":memory:"),
        VectorStore(persist_dir=str(tmp_path / "chroma_strict")),
    )
    stats_strict = ingest_scene(
        adapter=ds, scene_id="qa-0",
        view_ids=["video_a.mp4", "video_b.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store_strict, vstore=vstore_strict,
        segments_per_view=math.inf, appearance_threshold=0.99,
    )

    # Permissive threshold — should match strictly more (with the same data)
    store_loose, vstore_loose = (
        WorldStateStore(":memory:"),
        VectorStore(persist_dir=str(tmp_path / "chroma_loose")),
    )
    stats_loose = ingest_scene(
        adapter=ds, scene_id="qa-0",
        view_ids=["video_a.mp4", "video_b.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store_loose, vstore=vstore_loose,
        segments_per_view=math.inf, appearance_threshold=-1.0,  # accept anything
    )
    assert stats_loose["cross_view_links"] >= stats_strict["cross_view_links"]
    store_strict.close()
    store_loose.close()


def test_ingest_unknown_mode_skipped_gracefully(
    tmp_path, stores, mock_embedder, monkeypatch,
):
    """Defensive: if a future adapter sets
    `cross_view_linking_mode="something_new"`, ingest should print a
    warning and continue, not raise."""
    _make_mvu(tmp_path, duration_sec=12.0)
    ds = MVUEvalDataset(root=tmp_path)
    monkeypatch.setattr(ds, "cross_view_linking_mode", "future-mode")
    store, vstore = stores

    fake = FakeDetector([[_det("person")]])
    stats = ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    assert stats["cross_view_links"] == 0


# ----------------------------------------------------------------------
# M3.1 — ByteTrack per-segment aggregation
# ----------------------------------------------------------------------


class _StaticBboxDetector:
    """Returns the same bbox + class on every frame so the tracker
    sees an unmoving target → all K frames consolidate into one
    track. Used to verify per-track aggregation."""

    def __init__(self, n_objects: int = 1):
        self._n = n_objects

    def detect(self, frame: np.ndarray):
        # n distinct non-overlapping bboxes; one detection per object,
        # same bbox every call → tracker keeps each id stable.
        out = []
        for i in range(self._n):
            x = 2.0 + i * 4.0
            out.append(Detection(
                bbox=(x, 2.0, x + 3.0, 6.0),
                class_id=0, class_name="person", confidence=0.9,
            ))
        return out


def test_ingest_track_on_consolidates_K_frames_into_one_tracklet(
    tmp_path, stores, mock_embedder,
):
    """With --track ON and a static-bbox detector, K=4 sampled frames
    of one person collapse to a single tracklet row whose bboxes JSON
    contains K entries — P1-02 fixed: no longer N tracklet rows per
    same-identity-across-frames."""
    _make_mvu(tmp_path, duration_sec=10.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    stats = ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=4),
        embedder=mock_embedder, detector=_StaticBboxDetector(n_objects=1),
        embed_bboxes=True, store=store, vstore=vstore,
        segments_per_view=math.inf, track=True,
    )
    assert stats["segments"] == 1
    # K=4 frames, 1 person each → 4 raw detections, 1 track after
    # consolidation
    assert stats["detections"] == 4
    assert stats["tracklets"] == 1
    # And one bbox embedding (mean-pool over the 4 crops)
    assert stats["bbox_embeddings"] == 1

    tracklets = store.query_tracklets("video_a.mp4")
    assert len(tracklets) == 1
    bboxes = json.loads(tracklets[0]["bboxes"]) if isinstance(
        tracklets[0]["bboxes"], str) else tracklets[0]["bboxes"]
    assert len(bboxes) == 4, f"expected K=4 bbox rows in JSON, got {bboxes}"


def test_ingest_no_track_falls_back_to_per_detection_rows(
    tmp_path, stores, mock_embedder,
):
    """Regression guard: --no-track must preserve M3.0 behavior — one
    tracklet row + one bbox embedding per detection, no aggregation."""
    _make_mvu(tmp_path, duration_sec=10.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    stats = ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=4),
        embedder=mock_embedder, detector=_StaticBboxDetector(n_objects=1),
        embed_bboxes=True, store=store, vstore=vstore,
        segments_per_view=math.inf, track=False,
    )
    assert stats["detections"] == 4
    # M3.0 baseline: one tracklet per detection
    assert stats["tracklets"] == 4
    assert stats["bbox_embeddings"] == 4
    tracklets = store.query_tracklets("video_a.mp4")
    assert len(tracklets) == 4
    # Each row's bboxes JSON has exactly 1 entry (the single frame)
    for r in tracklets:
        rows = json.loads(r["bboxes"]) if isinstance(r["bboxes"], str) \
            else r["bboxes"]
        assert len(rows) == 1


def test_ingest_track_on_collapses_bbox_embeddings_p3_11(
    tmp_path, stores, mock_embedder,
):
    """P3-11 (same person across K frames produced K near-duplicate
    ChromaDB rows). With --track ON the count should be N_tracks, not
    N_tracks * K."""
    _make_mvu(tmp_path, duration_sec=10.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    stats = ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=4),
        embedder=mock_embedder, detector=_StaticBboxDetector(n_objects=2),
        embed_bboxes=True, store=store, vstore=vstore,
        segments_per_view=math.inf, track=True,
    )
    # 2 objects, K=4 frames → 8 raw detections, 2 tracks
    assert stats["detections"] == 8
    assert stats["tracklets"] == 2
    # ChromaDB delta: 1 segment + 2 bbox embeddings (NOT 8)
    bbox_rows = vstore.query(
        query_vector=mock_embedder.encode_image(
            np.zeros((8, 8, 3), dtype=np.uint8)),
        vector_type="reid", top_k=10,
    )
    bbox_count = len([r for r in bbox_rows
                      if r["metadata"].get("vector_kind") == "bbox"])
    assert bbox_count == 2
    # And the metadata indicates which frames were mean-pooled
    for r in bbox_rows[:2]:
        if r["metadata"].get("vector_kind") == "bbox":
            assert r["metadata"]["n_frames_in_track"] == 4


def test_ingest_track_vs_no_track_tracklet_count_delta_on_matrix(
    tmp_path, stores, mock_embedder,
):
    """End-to-end: same MATRIX fixture under --track ON vs OFF — track
    ON produces strictly ≤ tracklet rows. Mock embedder + static bbox
    detector keeps the comparison deterministic."""
    _make_matrix(tmp_path, n_frames_per_view=20)
    ds = MatrixDataset(root=tmp_path)

    # No-track path
    store_off = WorldStateStore(":memory:")
    vstore_off = VectorStore(persist_dir=str(tmp_path / "chroma_off"))
    stats_off = ingest_scene(
        adapter=ds, scene_id="MINI", view_ids=["D1"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=_StaticBboxDetector(n_objects=2),
        embed_bboxes=True, store=store_off, vstore=vstore_off,
        segments_per_view=math.inf, track=False,
    )

    # Track path
    store_on = WorldStateStore(":memory:")
    vstore_on = VectorStore(persist_dir=str(tmp_path / "chroma_on"))
    stats_on = ingest_scene(
        adapter=ds, scene_id="MINI", view_ids=["D1"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=_StaticBboxDetector(n_objects=2),
        embed_bboxes=True, store=store_on, vstore=vstore_on,
        segments_per_view=math.inf, track=True,
    )

    # Same raw detection count (track is post-detect aggregation)
    assert stats_on["detections"] == stats_off["detections"]
    # But strictly fewer tracklets when tracking
    assert stats_on["tracklets"] < stats_off["tracklets"]
    assert stats_on["bbox_embeddings"] < stats_off["bbox_embeddings"]
    store_off.close()
    store_on.close()


def test_ingest_idempotent_on_rerun(tmp_path, stores, mock_embedder):
    """Re-running ingest with the same args is fully idempotent end-to-end.
    M3.4: VectorStore.add now defaults to `upsert=True`, so duplicate ids
    no longer raise; segment counts on both stores stay stable across
    re-runs."""
    _make_mvu(tmp_path, duration_sec=12.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    kwargs = dict(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=None, embed_bboxes=False,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    ingest_scene(**kwargs)
    seg_count_after_first = len(store.query_segments(view_id="video_a.mp4"))
    chroma_count_after_first = vstore.collection.count()
    # Second call must not crash (P2-04 fix) and must leave both stores
    # at the same row count (DuckDB INSERT OR REPLACE + ChromaDB upsert).
    ingest_scene(**kwargs)
    seg_count_after_second = len(store.query_segments(view_id="video_a.mp4"))
    chroma_count_after_second = vstore.collection.count()
    assert seg_count_after_first == seg_count_after_second
    assert chroma_count_after_first == chroma_count_after_second


def test_ingest_rerun_does_not_duplicate_cross_view_links(
    tmp_path, stores, mock_embedder,
):
    """Post-M3.4 follow-up: cross_view_links must be idempotent on rerun.

    Before the fix, every L2 link got a fresh uuid4 link_id and
    insert_cross_view_link used plain INSERT, so a rerun stacked
    duplicate logical links (real data: MATRIX 199 → 398, qa-805 94 →
    188). After the fix, deterministic link_id from sorted observations
    + INSERT OR REPLACE means the row count stays flat across reruns.
    """
    # Two videos so MVU-Eval's appearance mode actually produces links
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    for name in ("video_a.mp4", "video_b.mp4"):
        if not _write_mp4(root / name, duration_sec=12.0):
            pytest.skip("cv2 mp4 writer unavailable")
    qa = {"0": {"video_paths": ["video_a.mp4", "video_b.mp4"],
                "question": "Q?", "options": ["A.", "B."],
                "ground_truth": "A", "task": "Counting"}}
    (root / "MVU_Eval_QAs.json").write_text(json.dumps(qa))

    ds = MVUEvalDataset(root=root)
    store, vstore = stores
    fake = FakeDetector([[_det("person")]])
    kwargs = dict(
        adapter=ds, scene_id="qa-0",
        view_ids=["video_a.mp4", "video_b.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore,
        segments_per_view=math.inf,
        # Permissive threshold so the mock-embedder fixture has a real
        # chance to produce links to lock the invariant on.
        appearance_threshold=-1.0,
    )
    stats_first = ingest_scene(**kwargs)
    n_first = len(store.query_cross_view_links())
    assert stats_first["cross_view_links"] >= 1, (
        "test needs ≥1 link to lock the no-dup invariant"
    )
    assert n_first == stats_first["cross_view_links"]

    ingest_scene(**kwargs)
    n_second = len(store.query_cross_view_links())
    assert n_second == n_first, (
        f"cross_view_links count drifted across reruns "
        f"({n_first} → {n_second}) — link_id is not deterministic "
        f"or the insert path skipped INSERT OR REPLACE"
    )


# ----------------------------------------------------------------------
# M3.6.D — bbox ChromaDB metadata carries classes_in_track multiset
# ----------------------------------------------------------------------


def test_ingest_bbox_metadata_records_classes_in_track_for_mixed_track(
    tmp_path, stores, mock_embedder,
):
    """P3-12 partial mitigation: when the class-agnostic IoU tracker
    merges two YOLO labels (cat, dog) at the same bbox position into one
    track, ChromaDB bbox metadata must surface the multiset in
    `classes_in_track` (CSV, sorted). The legacy `class_name` field still
    holds the rep frame's class (so existing retrieval paths are
    unchanged); the new field is an additive bbox-level fallback for
    downstream renderers."""
    _make_mvu(tmp_path, duration_sec=10.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    # Frame 0 → cat, frame 1 → dog, both at the same bbox → IoU=1.0 →
    # tracker keeps them on the same track (class-agnostic by design).
    fake = FakeDetector([[_det("cat")], [_det("dog")]])
    ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf, track=True,
    )
    bbox_rows = vstore.query(
        query_vector=mock_embedder.encode_image(np.zeros((8, 8, 3), dtype=np.uint8)),
        vector_type="reid", top_k=10,
    )
    assert len(bbox_rows) == 1, (
        f"expected 1 merged track row, got {len(bbox_rows)}"
    )
    md = bbox_rows[0]["metadata"]
    assert md["vector_kind"] == "bbox"
    # Sorted CSV multiset — order is alphabetical regardless of detection order
    assert md["classes_in_track"] == "cat,dog"
    # `class_name` is the rep frame's class (rep_idx = K // 2 = 1 → dog)
    assert md["class_name"] in ("cat", "dog")
    assert md["n_frames_in_track"] == 2


def test_ingest_segments_per_view_warns_when_cap_hit(
    tmp_path, stores, mock_embedder, capsys,
):
    """When --segments-per-view cap clips work on a view, ingest warns
    so the user knows the run was partial."""
    # 200 PNGs/view @ 2 fps = 100s = 10 segments per view at window=10s.
    # Cap at 3/view → each view has 7 more segments available.
    _make_matrix(tmp_path, n_frames_per_view=200)
    ds = MatrixDataset(root=tmp_path)
    store, vstore = stores
    fake = FakeDetector([[_det("person")]])

    ingest_scene(
        adapter=ds, scene_id="MINI", view_ids=["D1", "D2"],
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=3,
    )
    out = capsys.readouterr().out
    assert "hit --segments-per-view=3 cap" in out


def test_ingest_no_segments_per_view_warning_when_below_cap(
    tmp_path, stores, mock_embedder, capsys,
):
    """When the dataset fits comfortably under the cap, there's nothing
    to warn about — the warning must be silent (no false alarms)."""
    _make_matrix(tmp_path, n_frames_per_view=40)  # 2 seg/view → fits in 100
    ds = MatrixDataset(root=tmp_path)
    store, vstore = stores
    fake = FakeDetector([[_det("person")]])

    ingest_scene(
        adapter=ds, scene_id="MINI", view_ids=["D1", "D2"],
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=100,
    )
    out = capsys.readouterr().out
    assert "hit --segments-per-view" not in out


def test_ingest_no_warning_with_no_cap(
    tmp_path, stores, mock_embedder, capsys,
):
    """`segments_per_view=inf` (i.e. CLI --segments-per-view 0) means
    "no cap" — the warning machinery must short-circuit."""
    _make_matrix(tmp_path, n_frames_per_view=40)
    ds = MatrixDataset(root=tmp_path)
    store, vstore = stores
    fake = FakeDetector([[_det("person")]])

    ingest_scene(
        adapter=ds, scene_id="MINI", view_ids=["D1", "D2"],
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf,
    )
    out = capsys.readouterr().out
    assert "hit --segments-per-view" not in out


def test_ingest_llm_fallback_upgrades_low_confidence_link(
    tmp_path, stores, mock_embedder,
):
    """M4.3 end-to-end: --enable-llm-fallback + scripted LLM →
    low-confidence links get a `created_by='llm'` upgrade in the
    DuckDB row.

    Engineering predictably-low-conf links from FakeDetector alone is
    brittle (Geometric mode produces high conf for any tight pair). We
    sidestep by setting `fallback_confidence_threshold=1.0` — every
    link counts as "below threshold" → every link should be visited
    by the LLM. The scripted LLM always says "same object, 0.88" so
    each visited link gets re-stamped as `created_by='llm'`.

    Throttle (max_per_view=1 default) caps the actual upgrades, so we
    just check ≥1 LLM call happened and ≥1 link ended up `created_by='llm'`.

    LLM response uses confidence=1.0 so it clears the same threshold value
    we use to trigger the fallback — otherwise the inner LLMCrossViewLinker
    threshold drops it too.
    """
    import json as _json

    class _ScriptedLLM:
        def __init__(self, resp):
            self.resp = resp
            self.calls = 0

        def complete(self, prompt, images=None, **_kw):
            self.calls += 1
            return self.resp

        def unload(self):
            pass

    _make_matrix(tmp_path, n_frames_per_view=40)
    ds = MatrixDataset(root=tmp_path)
    store, vstore = stores
    rois_dir = tmp_path / "fallback_rois"
    rois_dir.mkdir()

    # Round-robin call pattern for K=2, 2 views: detector is called
    # for D1seg0(2 frames), D2seg0(2), D1seg1(2), D2seg1(2)...
    # counter // 2 % 2 == 0 → D1 (calls 0-1, 4-5, ...); == 1 → D2.
    # Different bboxes per view → geometric distance > 0 → conf < 1.0.
    class _PerViewDetector:
        def __init__(self):
            self._n = 0

        def detect(self, _frame):
            view_idx = (self._n // 2) % 2
            self._n += 1
            # D1 bbox at left half; D2 bbox at center — overlapping enough
            # to stay under the 0.3 distance threshold but offset enough
            # to drop confidence well under 1.0.
            x = 2.0 if view_idx == 0 else 6.0
            return [Detection(
                bbox=(x, 2.0, x + 4.0, 8.0),
                class_id=0, class_name="person", confidence=0.9,
            )]

    scripted_llm = _ScriptedLLM(_json.dumps({
        "same_object": True, "confidence": 1.0,
    }))

    ingest_scene(
        adapter=ds, scene_id="MINI", view_ids=["D1", "D2"],
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=2),
        embedder=mock_embedder, detector=_PerViewDetector(),
        embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf,
        rois_dir=str(rois_dir),
        appearance_threshold=-1.0,    # disable appearance filter so links survive
        enable_llm_fallback=True,
        fallback_llm_client=scripted_llm,
        fallback_confidence_threshold=0.95,
    )
    links = store.query_cross_view_links()
    assert scripted_llm.calls >= 1, (
        "fallback wiring not passing the LLM client through — "
        "no .complete() calls recorded"
    )
    llm_links = [link for link in links if link.created_by == "llm"]
    assert llm_links, (
        f"expected ≥1 created_by='llm' link after fallback, got "
        f"{[link.created_by for link in links]}"
    )
    # Sanity: at least one upgraded link carries the LLM's own confidence
    assert any(abs(link.confidence - 1.0) < 1e-6 for link in llm_links)


def test_ingest_llm_fallback_does_not_fire_without_flag(
    tmp_path, stores, mock_embedder,
):
    """Default (--enable-llm-fallback OFF): the scripted LLM is never
    called, all links stay at their original created_by."""
    class _SpyLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt, images=None, **_kw):
            self.calls += 1
            return "should not be invoked"

    _make_matrix(tmp_path, n_frames_per_view=40)
    ds = MatrixDataset(root=tmp_path)
    store, vstore = stores
    fake = FakeDetector([[_det("person")]])
    spy = _SpyLLM()

    ingest_scene(
        adapter=ds, scene_id="MINI", view_ids=["D1", "D2"],
        config=SegmenterConfig(window_sec=10.0, stride_sec=10.0,
                               nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf,
        # enable_llm_fallback omitted = default False
        fallback_llm_client=spy,
    )
    assert spy.calls == 0
    links = store.query_cross_view_links()
    assert all(link.created_by != "llm" for link in links)


def test_ingest_cache_rois_writes_jpeg_per_track_and_populates_roi_uri(
    tmp_path, stores, mock_embedder,
):
    """M4.1: when `rois_dir` is set, ingest writes one JPEG per bbox
    embedding (rep-frame crop) and the path is reachable from the
    DuckDB tracklet's embedding_ref → ChromaDB metadata. Without this
    the LLMCrossViewLinker fallback has to pay the delayed-decode tax
    on every pair."""
    _make_mvu(tmp_path, duration_sec=12.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores
    rois_dir = tmp_path / "rois"
    rois_dir.mkdir()

    fake = FakeDetector([[_det("person", conf=0.9)]])
    ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf,
        rois_dir=str(rois_dir),
    )
    # The bbox chroma row should have written a corresponding JPEG
    rois = sorted(rois_dir.glob("*.jpg"))
    assert len(rois) >= 1, "expected at least one ROI JPEG written"
    # Each tracklet's embedding_ref points at the bbox chroma id; the
    # corresponding JPEG must exist on disk
    tracklets = store.query_tracklets("video_a.mp4")
    for tk in tracklets:
        ref = tk["embedding_ref"]
        if ref is None:
            continue
        roi_path = rois_dir / f"{ref}.jpg"
        assert roi_path.exists(), (
            f"tracklet {tk['tracklet_id']} embedding_ref={ref} but "
            f"{roi_path} doesn't exist on disk"
        )
        # The JPEG is readable and has positive size
        img = cv2.imread(str(roi_path))
        assert img is not None and img.size > 0


def test_ingest_without_cache_rois_writes_no_jpegs(
    tmp_path, stores, mock_embedder,
):
    """Default path: --cache-rois OFF → no JPEGs anywhere. We don't
    want ROI caching to be on by default and silently bloat
    runs/."""
    _make_mvu(tmp_path, duration_sec=12.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores
    rois_dir = tmp_path / "should_stay_empty"
    rois_dir.mkdir()

    fake = FakeDetector([[_det("person")]])
    ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf,
        # rois_dir omitted = default None
    )
    assert list(rois_dir.glob("*")) == []


def test_ingest_bbox_metadata_classes_in_track_single_class_track(
    tmp_path, stores, mock_embedder,
):
    """For single-class tracks the field is still populated (just equal to
    `class_name`) so downstream renderers don't need to handle missing."""
    _make_mvu(tmp_path, duration_sec=10.0)
    ds = MVUEvalDataset(root=tmp_path)
    store, vstore = stores

    fake = FakeDetector([[_det("person")]])
    ingest_scene(
        adapter=ds, scene_id="qa-0", view_ids=["video_a.mp4"],
        config=SegmenterConfig(window_sec=10.0, nframes_per_segment=2),
        embedder=mock_embedder, detector=fake, embed_bboxes=True,
        store=store, vstore=vstore, segments_per_view=math.inf, track=True,
    )
    bbox_rows = vstore.query(
        query_vector=mock_embedder.encode_image(np.zeros((8, 8, 3), dtype=np.uint8)),
        vector_type="reid", top_k=10,
    )
    assert len(bbox_rows) >= 1
    md = bbox_rows[0]["metadata"]
    assert md["classes_in_track"] == "person"
    assert "," not in md["classes_in_track"]
