"""Unit tests for Phase 2 live incremental ingest.

Covers the L5 eviction primitives (delete_segment cascade + VectorStore.delete),
the one-window adapter, and the LiveIngestor control flow (monotonic seg_idx,
cursor advance, FIFO prune). The full run_cycle's detect+embed is GPU and is
validated manually; here ingest_scene + Detector are monkeypatched so we test
the orchestration without models.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from mva.l5_state import VectorStore, WorldStateStore
from mva.cli.live_ingest import LiveIngestor, _OneWindowAdapter, _decode_window


def _write_mp4(path: Path, duration_sec: float, fps: float = 10.0) -> bool:
    import cv2
    import numpy as np
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    if not w.isOpened():
        return False
    try:
        for i in range(max(1, int(duration_sec * fps))):
            w.write(np.full((24, 32, 3), (i * 9) % 255, dtype=np.uint8))
    finally:
        w.release()
    return True


# ---- L5 eviction primitives ------------------------------------------------


def test_store_delete_segment_cascades_and_returns_chroma_ids():
    store = WorldStateStore(db_path=":memory:")
    store.insert_segment("view1", 0, 0.0, 5.0, "/v.mp4", embed_chroma_id="seg-c0")
    store.insert_tracklet("view1", "view1-seg0-t1", 0.0, 5.0,
                          [[1.0, 0, 0, 9, 9, "boat", 0.8]],
                          embedding_ref="bbox-c0", segment_idx=0)
    assert store.list_segment_indices("view1") == [0]

    evicted = store.delete_segment("view1", 0)
    assert set(evicted) == {"seg-c0", "bbox-c0"}        # both vectors flagged
    assert store.list_segment_indices("view1") == []    # segment row gone
    assert store.query_tracklets("view1") == []         # tracklets cascaded


def test_store_delete_segment_missing_is_idempotent():
    store = WorldStateStore(db_path=":memory:")
    assert store.delete_segment("view1", 99) == []


def test_vector_store_delete_removes_rows():
    vs = VectorStore(persist_dir=None)
    vec = [0.1] * 8
    cid = vs.add(vec, vector_type="frame", view_id="v1", tracklet_id="t1")
    assert vs.collection.count() == 1
    vs.delete([cid])
    assert vs.collection.count() == 0
    vs.delete([])                                        # no-op, no crash


# ---- one-window adapter ----------------------------------------------------


def test_one_window_adapter_yields_single_segment(tmp_path):
    mp4 = tmp_path / "v1.mp4"
    if not _write_mp4(mp4, 6.0):
        pytest.skip("cv2 mp4 writer unavailable")
    adapter = _OneWindowAdapter({"v1": str(mp4)}, seg_idx=7,
                                w_start=0.0, w_end=5.0, nframes=4)
    segs = list(adapter.iter_segments("live", "v1", None))
    assert len(segs) == 1
    assert segs[0].segment_idx == 7
    assert segs[0].view_id == "v1"
    assert segs[0].start_t == 0.0 and segs[0].end_t == 5.0
    assert len(segs[0].frames) >= 1
    assert adapter.cross_view_linking_mode == "appearance"


def test_decode_window_returns_frames(tmp_path):
    mp4 = tmp_path / "v1.mp4"
    if not _write_mp4(mp4, 4.0):
        pytest.skip("cv2 mp4 writer unavailable")
    frames, idxs = _decode_window(str(mp4), 0.0, 2.0, 4)
    assert len(frames) >= 1 and len(frames) == len(idxs)


# ---- LiveIngestor control flow (no GPU) ------------------------------------


class _FakeStore:
    def __init__(self, indices):
        self._idx = {v: list(i) for v, i in indices.items()}
        self.deleted = []
        self.dropped_idx = False

    def drop_secondary_indexes(self):
        self.dropped_idx = True

    def list_segment_indices(self, view_id):
        return sorted(self._idx.get(view_id, []))

    def delete_segment(self, view_id, segment_idx):
        self._idx.get(view_id, []).remove(segment_idx)
        self.deleted.append((view_id, segment_idx))
        return [f"{view_id}-c{segment_idx}"]


class _FakeVStore:
    def __init__(self):
        self.deleted = []

    def delete(self, ids):
        self.deleted.extend(ids)


def _ingestor(store, vstore, **kw):
    return LiveIngestor(
        embedder=object(), store=store, vstore=vstore,
        sources={"view1": "/v1.mp4", "view2": "/v2.mp4"},
        loop_duration=20.0, gpu_lock=threading.Lock(), **kw)


def test_next_seg_idx_continues_above_existing():
    store = _FakeStore({"view1": [5, 6, 7], "view2": [5, 6]})
    ing = _ingestor(store, _FakeVStore())
    assert ing._seg_idx == 8                              # max(7) + 1


def test_ingestor_drops_secondary_index_on_init():
    store = _FakeStore({"view1": [], "view2": []})
    _ingestor(store, _FakeVStore())
    assert store.dropped_idx is True            # avoids the FIFO-delete crash


def test_prune_evicts_oldest_beyond_keep_n():
    store = _FakeStore({"view1": [0, 1, 2, 3, 4], "view2": [0, 1, 2, 3, 4]})
    vstore = _FakeVStore()
    ing = _ingestor(store, vstore, keep_n=2)
    ing._prune()
    # keep last 2 per view → evict 0,1,2 from each
    assert store.list_segment_indices("view1") == [3, 4]
    assert sorted(d for v, d in store.deleted if v == "view1") == [0, 1, 2]
    assert len(vstore.deleted) == 6                       # 3 per view × 2 views


def test_run_cycle_increments_seg_idx_and_advances_cursor(monkeypatch):
    import mva.cli.ingest as ingest_mod
    import mva.l1_perception as l1

    calls = {}

    def _fake_ingest_scene(**kw):
        calls["kw"] = kw

    class _DummyDetector:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr(ingest_mod, "ingest_scene", _fake_ingest_scene)
    monkeypatch.setattr(l1, "Detector", _DummyDetector)

    store = _FakeStore({"view1": [], "view2": []})
    ing = _ingestor(store, _FakeVStore(), window_sec=5.0)
    assert ing._seg_idx == 0 and ing._video_cursor == 0.0

    ing.run_cycle()

    # ingest_scene was driven with our one-window adapter at seg 0, window [0,5]
    adapter = calls["kw"]["adapter"]
    assert adapter._seg_idx == 0
    assert adapter._w_start == 0.0 and adapter._w_end == 5.0
    assert calls["kw"]["view_ids"] == ["view1", "view2"]
    # state advanced for the next cycle
    assert ing._seg_idx == 1
    assert ing._video_cursor == 5.0
    assert ing.cycles == 1


def test_run_cycle_wraps_cursor_at_loop_end(monkeypatch):
    import mva.cli.ingest as ingest_mod
    import mva.l1_perception as l1
    monkeypatch.setattr(ingest_mod, "ingest_scene", lambda **kw: None)
    monkeypatch.setattr(l1, "Detector", lambda **kw: object())

    store = _FakeStore({"view1": [], "view2": []})
    ing = _ingestor(store, _FakeVStore(), window_sec=5.0)
    ing.loop = 8.0
    ing._video_cursor = 5.0
    ing.run_cycle()                                       # window [5,8], then wrap
    assert ing._video_cursor == 0.0
