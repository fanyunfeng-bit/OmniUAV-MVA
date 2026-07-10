"""Unit tests for the L6 ToolRegistry wiring.

Verifies each registered tool reaches the right L5 method with the right
args. WorldStateStore is in-memory (DuckDB :memory:), VectorStore uses a
tmp dir to avoid persistence side-effects.
"""
from __future__ import annotations

import pytest

from mva.contracts import CrossViewLink, Event
from mva.l5_state import VectorStore, WorldStateStore
from mva.l6_interaction import ToolRegistry, ToolSpec, build_default_registry


@pytest.fixture
def populated_store():
    store = WorldStateStore(db_path=":memory:")
    store.insert_tracklet("drone-1", "tk-1", 0.0, 2.0, [(0.0, 10, 20, 30, 40)])
    store.insert_tracklet("drone-1", "tk-2", 1.0, 3.0, [(1.0, 50, 60, 70, 80)])
    store.insert_tracklet("drone-2", "tk-3", 0.0, 2.0, [(0.0, 15, 25, 35, 45)])
    store.insert_caption("drone-1", "c-1", frame_idx=0, t=0.5, caption_text="red car")
    store.insert_event(
        Event(event_id="e-1", type="loitering", t=1.0, view_id="drone-1",
              tracklet_ids=["tk-1"], summary_text="stationary > 30s")
    )
    store.insert_cross_view_link(
        CrossViewLink(
            link_id="link-1",
            view_observations=[("drone-1", "tk-1"), ("drone-2", "tk-3")],
            confidence=0.85,
            created_by="geometric",
            created_at=0.0,
        )
    )
    return store


# ---- query_db tool -------------------------------------------------------


def test_query_db_select_tracklets(populated_store):
    reg = build_default_registry(populated_store)
    rows = reg.call("query_db", {"sql": "SELECT * FROM tracklets_drone_1"})
    assert len(rows) == 2
    ids = {r["tracklet_id"] for r in rows}
    assert ids == {"tk-1", "tk-2"}


def test_query_db_count(populated_store):
    reg = build_default_registry(populated_store)
    rows = reg.call("query_db", {
        "sql": "SELECT COUNT(*) AS cnt FROM tracklets_drone_1"
    })
    assert rows[0]["cnt"] == 2


def test_query_db_cross_view_links(populated_store):
    reg = build_default_registry(populated_store)
    rows = reg.call("query_db", {
        "sql": "SELECT * FROM cross_view_links WHERE confidence > 0.5"
    })
    assert len(rows) == 1
    assert rows[0]["confidence"] == 0.85


def test_query_db_rejects_non_select(populated_store):
    reg = build_default_registry(populated_store)
    with pytest.raises(ValueError, match="Only SELECT"):
        reg.call("query_db", {"sql": "DROP TABLE cross_view_links"})
    with pytest.raises(ValueError, match="Only SELECT"):
        reg.call("query_db", {"sql": "DELETE FROM cross_view_links"})


def test_query_db_segments(populated_store):
    populated_store.insert_segment(
        view_id="v1", segment_idx=0, start_t=0.0, end_t=10.0,
        source_uri="x.mp4", embed_chroma_id="c1", nframes_sampled=4,
        detected_classes="person", detected_counts={"person": 2},
    )
    populated_store.insert_segment(
        view_id="v2", segment_idx=0, start_t=0.0, end_t=10.0,
        source_uri="y.mp4", embed_chroma_id="c2", nframes_sampled=4,
        detected_classes=None, detected_counts=None,
    )
    reg = build_default_registry(populated_store)
    rows = reg.call("query_db", {
        "sql": "SELECT DISTINCT view_id FROM segments ORDER BY view_id"
    })
    assert [r["view_id"] for r in rows] == ["v1", "v2"]


def test_query_db_aggregation(populated_store):
    """LLM can write GROUP BY / aggregate queries."""
    populated_store.insert_segment(
        view_id="D1", segment_idx=0, start_t=0.0, end_t=10.0,
        source_uri="a.mp4", embed_chroma_id="c1", nframes_sampled=4,
    )
    populated_store.insert_segment(
        view_id="D1", segment_idx=1, start_t=10.0, end_t=20.0,
        source_uri="a.mp4", embed_chroma_id="c2", nframes_sampled=4,
    )
    populated_store.insert_segment(
        view_id="D3", segment_idx=0, start_t=0.0, end_t=10.0,
        source_uri="b.mp4", embed_chroma_id="c3", nframes_sampled=4,
    )
    reg = build_default_registry(populated_store)
    rows = reg.call("query_db", {
        "sql": "SELECT view_id, COUNT(*) AS n FROM segments GROUP BY view_id ORDER BY view_id"
    })
    assert rows == [
        {"view_id": "D1", "n": 2},
        {"view_id": "D3", "n": 1},
    ]


# ---- schema summary ------------------------------------------------------


def test_get_schema_summary(populated_store):
    schema = populated_store.get_schema_summary()
    assert "segments" in schema
    assert "cross_view_links" in schema
    assert "tracklets_drone_1" in schema


# ---- filter_by_distance --------------------------------------------------


def test_filter_by_distance_drops_weak_matches():
    from mva.l6_interaction.tools import _filter_by_distance
    hits = [
        {"id": "a", "distance": 0.40},
        {"id": "b", "distance": 0.84},
        {"id": "c", "distance": 0.90},
        {"id": "d", "distance": 1.10},
        {"id": "e", "distance": None},
    ]
    out = _filter_by_distance(hits, max_distance=0.85)
    kept = {h["id"] for h in out}
    assert kept == {"a", "b", "e"}, f"got {kept}"
    assert _filter_by_distance(hits, max_distance=None) == hits


def test_find_by_description_tool_respects_max_distance(populated_store, tmp_path):
    from mva.l5_state import MultimodalEmbedder, VectorStore
    embedder = MultimodalEmbedder(model_path=None, dim=64)
    vstore = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=embedder.as_chromadb_embedding_function(),
    )
    for i, text in enumerate(["person walking", "car driving", "red backpack"]):
        vec = embedder.encode_text(text)
        vstore.add(
            vec, vector_type="reid", view_id="v1",
            tracklet_id=f"t{i}", document=text,
        )
    reg = build_default_registry(populated_store, vstore=vstore)
    all_hits = reg.call("find_by_description", {"text": "completely unrelated", "top_k": 3})
    assert len(all_hits) >= 1
    none_hits = reg.call(
        "find_by_description",
        {"text": "completely unrelated", "top_k": 3, "max_distance": 0.01},
    )
    assert none_hits == [] or all(
        h["distance"] is not None and h["distance"] <= 0.01 for h in none_hits
    )


# ---- registry basics ------------------------------------------------------


def test_unknown_tool_raises(populated_store):
    reg = build_default_registry(populated_store)
    with pytest.raises(KeyError):
        reg.call("not_a_tool", {})


def test_custom_tool_can_be_registered(populated_store):
    reg = ToolRegistry()
    reg.register(ToolSpec(name="echo", description="echo", fn=lambda **kw: kw))
    out = reg.call("echo", {"x": 1})
    assert out == {"x": 1}


def test_vector_tools_not_registered_without_vstore(populated_store):
    reg = build_default_registry(populated_store)
    assert "find_by_description" not in reg
    assert "find_segment_by_description" not in reg
    assert "find_bbox_by_description" not in reg


def test_vector_tools_registered_when_vstore_supplied(populated_store, tmp_path):
    vstore = VectorStore(persist_dir=str(tmp_path / "chroma"))
    reg = build_default_registry(populated_store, vstore=vstore)
    assert "find_by_description" in reg
    assert "find_segment_by_description" in reg
    assert "find_bbox_by_description" in reg


# ---- M2.8: segment-level retrieval tools --------------------------------


def _segment_fixture_store_and_vstore(tmp_path, populated_store):
    from mva.l5_state import MultimodalEmbedder

    emb = MultimodalEmbedder(model_path=None, dim=64)
    vstore = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=emb.as_chromadb_embedding_function(),
    )
    seg_meta_a = {
        "vector_kind": "segment", "scene_id": "qa-0",
        "view_id_raw": "video_a.mp4", "segment_idx": 0,
        "start_t": 0.0, "end_t": 10.0,
        "source_uri": "/data/video_a.mp4", "nframes_sampled": 4,
        "chunk_id": 0,
    }
    seg_meta_b = {**seg_meta_a, "segment_idx": 1, "start_t": 10.0,
                  "end_t": 20.0, "chunk_id": 1}
    cid_a = vstore.add(
        emb.encode_text("a car drives by"), "frame",
        "qa-0::video_a.mp4", "seg0000",
        extra_metadata=seg_meta_a, document="video_a [0-10s]",
    )
    cid_b = vstore.add(
        emb.encode_text("a person walking"), "frame",
        "qa-0::video_a.mp4", "seg0001",
        extra_metadata=seg_meta_b, document="video_a [10-20s]",
    )
    populated_store.insert_segment(
        "video_a.mp4", 0, 0.0, 10.0, "/data/video_a.mp4",
        embed_chroma_id=cid_a, nframes_sampled=4,
        detected_classes="car", detected_counts={"car": 2},
    )
    populated_store.insert_segment(
        "video_a.mp4", 1, 10.0, 20.0, "/data/video_a.mp4",
        embed_chroma_id=cid_b, nframes_sampled=4,
        detected_classes="person", detected_counts={"person": 3},
    )
    return vstore, cid_a, cid_b


def test_find_segment_by_description_returns_enriched_hits(
    populated_store, tmp_path,
):
    vstore, _, _ = _segment_fixture_store_and_vstore(tmp_path, populated_store)
    reg = build_default_registry(populated_store, vstore=vstore)
    assert "find_segment_by_description" in reg

    hits = reg.call(
        "find_segment_by_description",
        {"text": "person walking", "top_k": 2},
    )
    assert len(hits) >= 1
    for h in hits:
        assert h["metadata"]["vector_kind"] == "segment"
        assert h["segment"] is not None
        assert h["segment"]["embed_chroma_id"] == h["id"]
        assert "source_uri" in h["segment"]
        assert h["segment"]["end_t"] > h["segment"]["start_t"]


def test_find_segment_by_description_skips_non_segment_frame_vectors(
    populated_store, tmp_path,
):
    from mva.l5_state import MultimodalEmbedder
    emb = MultimodalEmbedder(model_path=None, dim=64)
    vstore = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=emb.as_chromadb_embedding_function(),
    )
    vstore.add(
        emb.encode_text("legacy video embedding"), "frame",
        "old-scene::old-view", "old-vid",
        extra_metadata={"nframes_sampled": 8},
        document="legacy",
    )
    reg = build_default_registry(populated_store, vstore=vstore)
    hits = reg.call(
        "find_segment_by_description",
        {"text": "anything", "top_k": 5},
    )
    assert hits == []


def test_find_bbox_by_description_filters_to_reid_type(
    populated_store, tmp_path,
):
    from mva.l5_state import MultimodalEmbedder
    emb = MultimodalEmbedder(model_path=None, dim=64)
    vstore = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=emb.as_chromadb_embedding_function(),
    )
    vstore.add(emb.encode_text("seg"), "frame", "view", "seg0",
               extra_metadata={"vector_kind": "segment"})
    vstore.add(emb.encode_text("bbox"), "reid", "view", "bbox0",
               extra_metadata={
                   "vector_kind": "bbox", "class_name": "person",
                   "bbox_x1": 1.0, "bbox_y1": 2.0,
                   "bbox_x2": 5.0, "bbox_y2": 6.0,
                   "segment_idx": 0, "confidence": 0.9,
               })
    reg = build_default_registry(populated_store, vstore=vstore)
    hits = reg.call(
        "find_bbox_by_description",
        {"text": "person", "top_k": 5},
    )
    assert all(h["metadata"]["vector_kind"] == "bbox" for h in hits)
    assert hits and hits[0]["metadata"]["class_name"] == "person"
