"""Unit tests for L5 VectorStore (single-collection ChromaDB).

Verifies the §3.2 L5 design (Eng Review 1A/1B):
  - single collection `tracklets_embeddings`
  - metadata: vector_type {text|frame|reid} + view_id + tracklet_id
  - metadata-filter queries: by vector_type, by view_id, by both
  - reid-by-vector (the folded-in L1.5 lookup)

Tests use a temp directory + PersistentClient (in-process, no remote). All
queries pass embeddings explicitly so the default ChromaDB embedder is never
invoked (avoids triggering a model download on first run).
"""
from __future__ import annotations

import numpy as np
import pytest

from mva.l5_state.chromadb_store import (
    VECTOR_TYPE_FRAME,
    VECTOR_TYPE_REID,
    VECTOR_TYPE_TEXT,
    VectorStore,
)


@pytest.fixture
def store(tmp_path):
    s = VectorStore(persist_dir=str(tmp_path / "chroma"))
    yield s


def _vec(seed: int, dim: int = 8) -> list[float]:
    """Reproducible random unit-ish vector for tests."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


class TestVectorTypeFilter:
    def test_three_vector_types_coexist_one_collection(self, store):
        store.add(_vec(1), VECTOR_TYPE_TEXT, "drone-1", "tk-1")
        store.add(_vec(2), VECTOR_TYPE_FRAME, "drone-1", "tk-1")
        store.add(_vec(3), VECTOR_TYPE_REID, "drone-1", "tk-1")

        # All three live in one collection
        assert store.collection.count() == 3

    def test_vector_type_filter_returns_only_matching(self, store):
        store.add(_vec(1), VECTOR_TYPE_TEXT, "drone-1", "tk-1")
        store.add(_vec(2), VECTOR_TYPE_REID, "drone-1", "tk-2")
        store.add(_vec(3), VECTOR_TYPE_REID, "drone-2", "tk-3")

        # Query with a reid filter — text entry must not appear
        results = store.query(
            query_vector=_vec(10), vector_type=VECTOR_TYPE_REID, top_k=5
        )
        assert len(results) == 2
        for r in results:
            assert r["metadata"]["vector_type"] == VECTOR_TYPE_REID


class TestViewIdFilter:
    def test_view_id_filter_isolates_per_view(self, store):
        store.add(_vec(1), VECTOR_TYPE_TEXT, "drone-1", "tk-1")
        store.add(_vec(2), VECTOR_TYPE_TEXT, "drone-2", "tk-2")

        results = store.query(query_vector=_vec(10), view_id="drone-1", top_k=5)
        assert len(results) == 1
        assert results[0]["metadata"]["view_id"] == "drone-1"
        assert results[0]["metadata"]["tracklet_id"] == "tk-1"

    def test_combined_view_and_vector_type_filter(self, store):
        # Same view, two vector_types
        store.add(_vec(1), VECTOR_TYPE_TEXT, "drone-1", "tk-1")
        store.add(_vec(2), VECTOR_TYPE_REID, "drone-1", "tk-1")
        # Different view, same vector_type
        store.add(_vec(3), VECTOR_TYPE_REID, "drone-2", "tk-2")

        results = store.query(
            query_vector=_vec(10),
            vector_type=VECTOR_TYPE_REID,
            view_id="drone-1",
            top_k=5,
        )
        assert len(results) == 1
        assert results[0]["metadata"]["view_id"] == "drone-1"
        assert results[0]["metadata"]["vector_type"] == VECTOR_TYPE_REID


class TestReidByVectorLookup:
    """The folded-in L1.5 use case: find tracklets across views by appearance."""

    def test_query_returns_closest_match(self, store):
        # Two visually similar tracklets across views + one far-away
        target = _vec(42)
        store.add(target, VECTOR_TYPE_REID, "drone-1", "tk-1")
        store.add(_vec(43), VECTOR_TYPE_REID, "drone-2", "tk-2")  # close-ish
        store.add(_vec(999), VECTOR_TYPE_REID, "drone-3", "tk-3")  # far

        results = store.query(
            query_vector=target, vector_type=VECTOR_TYPE_REID, top_k=3
        )
        assert len(results) == 3
        # Distances should be sorted ascending (closest first)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)
        # The exact-match vector should be the closest
        assert results[0]["metadata"]["tracklet_id"] == "tk-1"


class TestPersistenceAndShape:
    def test_persistence_across_reopen(self, tmp_path):
        path = str(tmp_path / "chroma")
        s1 = VectorStore(persist_dir=path)
        s1.add(_vec(1), VECTOR_TYPE_TEXT, "drone-1", "tk-1")
        assert s1.collection.count() == 1

        # Reopen the same dir — entry must survive
        s2 = VectorStore(persist_dir=path)
        assert s2.collection.count() == 1
        results = s2.query(query_vector=_vec(1), top_k=5)
        assert len(results) == 1

    def test_add_returns_id(self, store):
        eid = store.add(_vec(1), VECTOR_TYPE_TEXT, "drone-1", "tk-1")
        assert isinstance(eid, str)
        assert len(eid) > 0

    def test_query_without_vector_or_text_raises(self, store):
        with pytest.raises(ValueError):
            store.query()

    def test_extra_metadata_preserved(self, store):
        store.add(
            _vec(1),
            VECTOR_TYPE_TEXT,
            "drone-1",
            "tk-1",
            extra_metadata={"t_start": 1.0, "t_end": 5.0},
        )
        results = store.query(query_vector=_vec(1), top_k=1)
        md = results[0]["metadata"]
        assert md["t_start"] == 1.0
        assert md["t_end"] == 5.0

    def test_segment_metadata_round_trip(self, store):
        """Video-segment fields (start_sec / end_sec / segment_idx +
        detected_classes) must survive a write→query round-trip — the LLM
        relies on them to map an embedding hit back to the original clip."""
        store.add(
            _vec(1),
            VECTOR_TYPE_FRAME,
            view_id="qa-0::vid.mp4",
            tracklet_id="vid::seg0002",
            extra_metadata={
                "video_path": "/data/vid.mp4",
                "start_sec": 20.0,
                "end_sec": 30.0,
                "segment_idx": 2,
                "chunk_id": 2,
                "detected_classes": "car,person",
                "detected_counts_json": '{"car": 1, "person": 4}',
            },
        )
        results = store.query(query_vector=_vec(1), top_k=1)
        md = results[0]["metadata"]
        assert md["start_sec"] == 20.0
        assert md["end_sec"] == 30.0
        assert md["segment_idx"] == 2
        assert md["video_path"] == "/data/vid.mp4"
        assert md["detected_classes"] == "car,person"

    def test_segment_chunk_id_keeps_ids_unique(self, store):
        """Two segments sharing (view, tracklet, vector_type) must get
        distinct ChromaDB ids via the chunk_id suffix path."""
        eid_a = store.add(
            _vec(1), VECTOR_TYPE_FRAME, "qa-0::vid.mp4", "vid",
            extra_metadata={"chunk_id": 0, "segment_idx": 0},
        )
        eid_b = store.add(
            _vec(2), VECTOR_TYPE_FRAME, "qa-0::vid.mp4", "vid",
            extra_metadata={"chunk_id": 1, "segment_idx": 1},
        )
        assert eid_a != eid_b
        assert store.collection.count() == 2


class TestUpsertSemantics:
    """M3.4 — `VectorStore.add` defaults to `upsert=True` so re-running
    `mva ingest` is fully idempotent. PROBLEMS P2-04 fix.

    Note: chromadb 1.x's `collection.add()` is itself a silent no-op on
    duplicate ids (older versions raised "ID already exists"). The
    distinction we lock in here is the *semantic* one: `upsert=True`
    overwrites (new data wins) while `upsert=False` keeps the existing
    entry (old data wins) — either way no exception.
    """

    def test_duplicate_add_default_upserts_overwrites(self, store):
        store.add(
            _vec(1), VECTOR_TYPE_FRAME, "drone-1", "tk-1",
            extra_metadata={"detected_classes": "car"},
        )
        # Same id, new metadata — upsert should overwrite.
        store.add(
            _vec(99), VECTOR_TYPE_FRAME, "drone-1", "tk-1",
            extra_metadata={"detected_classes": "person,car"},
        )
        assert store.collection.count() == 1
        results = store.query(query_vector=_vec(99), top_k=1)
        # New vector wins → near-zero distance to itself
        assert results[0]["distance"] < 0.01
        # New metadata wins
        assert results[0]["metadata"]["detected_classes"] == "person,car"

    def test_explicit_upsert_false_keeps_original(self, store):
        """`upsert=False` falls through to chromadb's `collection.add()`,
        which in 1.x silently keeps the first row."""
        store.add(
            _vec(1), VECTOR_TYPE_FRAME, "drone-1", "tk-1",
            extra_metadata={"detected_classes": "car"},
        )
        store.add(
            _vec(99), VECTOR_TYPE_FRAME, "drone-1", "tk-1",
            extra_metadata={"detected_classes": "REPLACED"},
            upsert=False,
        )
        assert store.collection.count() == 1
        results = store.query(query_vector=_vec(1), top_k=1)
        # Original vector still there
        assert results[0]["distance"] < 0.01
        assert results[0]["metadata"]["detected_classes"] == "car"

    def test_rerun_at_scale_is_idempotent(self, store):
        """Many duplicates → row count stable across re-runs."""
        for i in range(10):
            store.add(
                _vec(i), VECTOR_TYPE_REID, f"v-{i % 2}", f"tk-{i}",
            )
        assert store.collection.count() == 10
        # Replay: same ids, slightly different vectors.
        for i in range(10):
            store.add(
                _vec(i + 100), VECTOR_TYPE_REID, f"v-{i % 2}", f"tk-{i}",
            )
        assert store.collection.count() == 10  # not 20
