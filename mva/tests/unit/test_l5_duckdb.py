"""Unit tests for L5 DuckDB WorldStateStore.

Covers:
- Per-view table lazy creation
- Tracklet / event / caption / cross-view-link / telemetry roundtrips
- Time-range queries
- Per-view isolation (one view does not leak into another)
- CrossViewLink Pydantic roundtrip
- §3.4 #8 still returns 501
"""
from __future__ import annotations

import pytest

from mva.contracts import CrossViewLink, Event
from mva.l5_state.duckdb_store import WorldStateStore, _sanitize_view_id


# ---- schema / sanitization ------------------------------------------------


def test_sanitize_view_id_replaces_dash():
    assert _sanitize_view_id("drone-1") == "drone_1"


def test_sanitize_view_id_prefixes_digit_leading():
    assert _sanitize_view_id("1cam") == "v_1cam"


def test_sanitize_view_id_rejects_empty():
    with pytest.raises(ValueError):
        _sanitize_view_id("")


# ---- tracklets ------------------------------------------------------------


def test_tracklet_insert_and_query_roundtrip():
    with WorldStateStore() as store:
        store.insert_tracklet(
            view_id="drone-1",
            tracklet_id="tk-1",
            t_start=0.0,
            t_end=5.0,
            bboxes=[[0.0, 10.0, 20.0, 30.0, 40.0]],
            embedding_ref="chroma:emb-1",
        )
        rows = store.query_tracklets("drone-1")
        assert len(rows) == 1
        assert rows[0]["tracklet_id"] == "tk-1"
        assert rows[0]["t_start"] == 0.0
        assert rows[0]["t_end"] == 5.0
        assert rows[0]["bboxes"] == [[0.0, 10.0, 20.0, 30.0, 40.0]]
        assert rows[0]["embedding_ref"] == "chroma:emb-1"


def test_tracklet_time_range_filter():
    with WorldStateStore() as store:
        for i in range(5):
            store.insert_tracklet(
                view_id="drone-1",
                tracklet_id=f"tk-{i}",
                t_start=float(i * 10),
                t_end=float(i * 10 + 5),
                bboxes=[],
            )
        # Window [12, 28] overlaps tk-1 (10-15), tk-2 (20-25), tk-3 (30-35? no)
        hits = store.query_tracklets("drone-1", t_start=12.0, t_end=28.0)
        ids = {r["tracklet_id"] for r in hits}
        assert ids == {"tk-1", "tk-2"}


def test_tracklet_query_unknown_view_returns_empty():
    with WorldStateStore() as store:
        assert store.query_tracklets("never-touched") == []


def test_tracklet_per_view_isolation():
    with WorldStateStore() as store:
        store.insert_tracklet("drone-1", "tk-a", 0.0, 1.0, [])
        store.insert_tracklet("drone-2", "tk-b", 0.0, 1.0, [])
        ids_a = {r["tracklet_id"] for r in store.query_tracklets("drone-1")}
        ids_b = {r["tracklet_id"] for r in store.query_tracklets("drone-2")}
        assert ids_a == {"tk-a"}
        assert ids_b == {"tk-b"}


# ---- events ---------------------------------------------------------------


def test_event_insert_and_query_roundtrip():
    with WorldStateStore() as store:
        ev = Event(
            event_id="e-1",
            type="loitering",
            t=100.0,
            view_id="drone-1",
            tracklet_ids=["tk-1", "tk-2"],
            summary_text="Person near gate",
        )
        store.insert_event(ev)
        out = store.query_events("drone-1")
        assert len(out) == 1
        assert out[0].event_id == "e-1"
        assert out[0].tracklet_ids == ["tk-1", "tk-2"]
        assert out[0].type == "loitering"
        assert out[0].summary_text == "Person near gate"


def test_event_filter_by_type():
    with WorldStateStore() as store:
        store.insert_event(Event(event_id="e1", type="loitering", t=1.0, view_id="v1"))
        store.insert_event(Event(event_id="e2", type="collision", t=2.0, view_id="v1"))
        hits = store.query_events("v1", type="collision")
        assert len(hits) == 1 and hits[0].event_id == "e2"


# ---- captions -------------------------------------------------------------


def test_caption_insert_and_query():
    with WorldStateStore() as store:
        store.insert_caption("drone-1", "c1", 0, 0.0, "white van in lot")
        store.insert_caption("drone-1", "c2", 30, 1.0, "two cars passing")
        hits = store.query_captions("drone-1", t_start=0.5, t_end=2.0)
        assert len(hits) == 1
        assert hits[0]["caption_id"] == "c2"


# ---- cross-view links -----------------------------------------------------


def test_cross_view_link_roundtrip_preserves_pydantic_shape():
    with WorldStateStore() as store:
        link = CrossViewLink(
            link_id="lnk-1",
            view_observations=[("drone-1", "tk-1"), ("drone-2", "tk-9")],
            confidence=0.82,
            created_by="geometric",
            created_at=1000.0,
        )
        store.insert_cross_view_link(link)

        hits = store.query_cross_view_links()
        assert len(hits) == 1
        assert isinstance(hits[0], CrossViewLink)
        assert hits[0].link_id == "lnk-1"
        assert hits[0].view_observations == [("drone-1", "tk-1"), ("drone-2", "tk-9")]
        assert hits[0].confidence == 0.82
        assert hits[0].created_by == "geometric"


def test_cross_view_link_min_confidence_filter():
    with WorldStateStore() as store:
        for i, conf in enumerate([0.3, 0.6, 0.9]):
            store.insert_cross_view_link(
                CrossViewLink(
                    link_id=f"l{i}",
                    view_observations=[("v1", "a"), ("v2", "b")],
                    confidence=conf,
                    created_by="geometric",
                    created_at=float(i),
                )
            )
        hits = store.query_cross_view_links(min_confidence=0.5)
        ids = {h.link_id for h in hits}
        assert ids == {"l1", "l2"}
        # Verify descending confidence order
        assert hits[0].confidence >= hits[1].confidence


def test_cross_view_link_view_id_filter_single_string():
    """M3.7 (P1-09): view_id='D1' returns only links that include D1."""
    with WorldStateStore() as store:
        for i, observations in enumerate([
            [("D1", "tk-a"), ("D2", "tk-b")],
            [("D2", "tk-c"), ("D3", "tk-d")],
            [("D1", "tk-e"), ("D3", "tk-f")],
            [("D4", "tk-g"), ("D5", "tk-h")],
        ]):
            store.insert_cross_view_link(
                CrossViewLink(
                    link_id=f"link-{i}",
                    view_observations=observations,
                    confidence=0.7,
                    created_by="geometric",
                    created_at=float(i),
                )
            )
        # D1 appears in links 0 and 2
        d1_links = store.query_cross_view_links(view_id="D1")
        d1_ids = {link.link_id for link in d1_links}
        assert d1_ids == {"link-0", "link-2"}, (
            f"expected D1 in link-0 + link-2, got {d1_ids}"
        )
        # D4 appears only in link-3
        d4_links = store.query_cross_view_links(view_id="D4")
        assert {link.link_id for link in d4_links} == {"link-3"}
        # D9 is in no link
        assert store.query_cross_view_links(view_id="D9") == []


def test_cross_view_link_view_id_filter_list_requires_all():
    """M3.7 (P1-09): view_id=['D1', 'D3'] requires the link to span BOTH
    views. List filter is set superset semantics — strict and unambiguous."""
    with WorldStateStore() as store:
        for i, observations in enumerate([
            [("D1", "tk-a"), ("D2", "tk-b")],      # has D1 only of {D1,D3}
            [("D2", "tk-c"), ("D3", "tk-d")],      # has D3 only
            [("D1", "tk-e"), ("D3", "tk-f")],      # has BOTH
            [("D1", "tk-g"), ("D3", "tk-h")],      # has BOTH (duplicate-shape link)
        ]):
            store.insert_cross_view_link(
                CrossViewLink(
                    link_id=f"link-{i}",
                    view_observations=observations,
                    confidence=0.7,
                    created_by="geometric",
                    created_at=float(i),
                )
            )
        # Both D1 and D3 must appear in link
        d1_d3 = store.query_cross_view_links(view_id=["D1", "D3"])
        d1_d3_ids = {link.link_id for link in d1_d3}
        assert d1_d3_ids == {"link-2", "link-3"}, (
            f"expected only links containing BOTH D1 and D3, got {d1_d3_ids}"
        )
        # Order-independent + dedup
        same = store.query_cross_view_links(view_id=["D3", "D1"])
        assert {link.link_id for link in same} == d1_d3_ids


def test_cross_view_link_view_id_filter_with_limit_applies_post_filter():
    """When view_id filter is active, `limit` must apply AFTER filtering
    so we don't lose results to the SQL LIMIT."""
    with WorldStateStore() as store:
        # 3 D1 links + 2 non-D1 links, all confidence 0.7
        link_data = [
            ("link-d1a", [("D1", "a"), ("D2", "b")], 0.9),
            ("link-d1b", [("D1", "c"), ("D3", "d")], 0.8),
            ("link-nope1", [("D2", "e"), ("D3", "f")], 0.7),
            ("link-nope2", [("D2", "g"), ("D3", "h")], 0.6),
            ("link-d1c", [("D1", "i"), ("D5", "j")], 0.5),
        ]
        for lid, obs, conf in link_data:
            store.insert_cross_view_link(CrossViewLink(
                link_id=lid, view_observations=obs, confidence=conf,
                created_by="geometric", created_at=0.0,
            ))
        # view_id="D1" + limit=2 should return the TOP 2 D1 links by
        # default sort (confidence DESC): d1a (0.9), d1b (0.8)
        out = store.query_cross_view_links(view_id="D1", limit=2)
        assert [link.link_id for link in out] == ["link-d1a", "link-d1b"]
        # limit=10 still returns all 3 D1 links (not all 5)
        out_all = store.query_cross_view_links(view_id="D1", limit=10)
        assert {link.link_id for link in out_all} == {"link-d1a", "link-d1b", "link-d1c"}


def test_cross_view_link_view_id_filter_validation():
    """Empty list / wrong type → raise rather than silently no-op."""
    import pytest
    with WorldStateStore() as store:
        store.insert_cross_view_link(CrossViewLink(
            link_id="x", view_observations=[("D1", "a"), ("D2", "b")],
            confidence=0.7, created_by="geometric", created_at=0.0,
        ))
        with pytest.raises(ValueError, match="view_id="):
            store.query_cross_view_links(view_id=[])
        with pytest.raises(TypeError):
            store.query_cross_view_links(view_id=123)
        with pytest.raises(TypeError):
            store.query_cross_view_links(view_id=["D1", 42])


def test_cross_view_link_sort_by_and_limit_top_k():
    """M3.6.A (P1-04): sort_by + limit lets the LLM ask 'top-k highest
    confidence link' directly instead of mis-translating '最高' as
    min_confidence=1.0 (which always returns [])."""
    import pytest

    with WorldStateStore() as store:
        for i, conf in enumerate([0.3, 0.6, 0.9, 0.55]):
            store.insert_cross_view_link(
                CrossViewLink(
                    link_id=f"l{i}",
                    view_observations=[("v1", f"a{i}"), ("v2", f"b{i}")],
                    confidence=conf,
                    created_by="geometric",
                    created_at=float(i),
                )
            )
        # confidence_desc + limit=1 → the 0.9 one
        top1 = store.query_cross_view_links(sort_by="confidence_desc", limit=1)
        assert len(top1) == 1 and top1[0].confidence == 0.9

        # confidence_asc + limit=2 → 0.3, 0.55
        bottom2 = store.query_cross_view_links(sort_by="confidence_asc", limit=2)
        assert [link.confidence for link in bottom2] == [0.3, 0.55]

        # created_at_desc + limit=1 → latest (i=3)
        latest = store.query_cross_view_links(sort_by="created_at_desc", limit=1)
        assert len(latest) == 1 and latest[0].link_id == "l3"

        # sort_by=None preserves M0-M3.5 default (confidence DESC) — back-compat
        all_default = store.query_cross_view_links()
        assert [link.confidence for link in all_default] == [0.9, 0.6, 0.55, 0.3]

        # invalid sort_by raises (so the LLM gets a hard error, not silent
        # wrong-order)
        with pytest.raises(ValueError):
            store.query_cross_view_links(sort_by="totally_bogus")

        # invalid limit raises
        with pytest.raises(ValueError):
            store.query_cross_view_links(limit=-1)


# ---- telemetry ------------------------------------------------------------


def test_telemetry_columns_and_extras():
    with WorldStateStore() as store:
        store.insert_telemetry(
            t=100.0,
            view_id="drone-1",
            telemetry={
                "gps_lat": 39.9,
                "gps_lon": 116.4,
                "alt": 120.0,
                "gimbal_qx": 0.0,
                "gimbal_qy": 0.0,
                "gimbal_qz": 0.0,
                "gimbal_qw": 1.0,
                "imu_ax": 0.1,  # extra → JSON blob
                "battery_pct": 78,
            },
        )
        rows = store.query_telemetry("drone-1")
        assert len(rows) == 1
        r = rows[0]
        assert r["gps_lat"] == 39.9
        assert r["alt"] == 120.0
        # Extras roundtripped
        assert r["imu_ax"] == 0.1
        assert r["battery_pct"] == 78


def test_telemetry_view_filter():
    with WorldStateStore() as store:
        store.insert_telemetry(1.0, "drone-1", {"gps_lat": 1.0})
        store.insert_telemetry(2.0, "drone-2", {"gps_lat": 2.0})
        assert len(store.query_telemetry("drone-1")) == 1
        assert len(store.query_telemetry("drone-2")) == 1
        assert store.query_telemetry("drone-1")[0]["gps_lat"] == 1.0


# ---- §3.4 #8 still 501 ----------------------------------------------------


def test_human_correction_endpoint_returns_501_until_m5():
    with WorldStateStore() as store:
        assert store.human_correction_endpoint("lnk-1", "correct") == 501


# ---- persistence ----------------------------------------------------------


# ---- idempotency (re-runnable demos) --------------------------------------


def test_tracklet_reinsert_same_id_overwrites():
    """Re-inserting the same tracklet_id replaces the prior row, no crash.

    Demos generate deterministic ids like 'D1-f0-d0'; re-running on a
    persisted DB used to raise PRIMARY KEY violation. Now it's an upsert.
    """
    with WorldStateStore() as store:
        store.insert_tracklet("drone-1", "tk-1", 0.0, 1.0, [(0.0, 1, 2, 3, 4)])
        # Second insert with the same id but different values — must not raise
        store.insert_tracklet("drone-1", "tk-1", 5.0, 6.0, [(5.0, 9, 9, 9, 9)])

        rows = store.query_tracklets("drone-1")
        assert len(rows) == 1
        assert rows[0]["t_start"] == 5.0
        assert rows[0]["bboxes"] == [[5.0, 9, 9, 9, 9]]


def test_caption_reinsert_same_id_overwrites():
    with WorldStateStore() as store:
        store.insert_caption("drone-1", "c-1", frame_idx=0, t=0.5, caption_text="red car")
        store.insert_caption("drone-1", "c-1", frame_idx=0, t=0.5, caption_text="blue car")
        rows = store.query_captions("drone-1")
        assert len(rows) == 1
        assert rows[0]["caption_text"] == "blue car"


def test_event_reinsert_same_id_overwrites():
    with WorldStateStore() as store:
        store.insert_event(
            Event(event_id="e-1", type="loitering", t=1.0, view_id="drone-1",
                  tracklet_ids=["tk-1"], summary_text="v1")
        )
        store.insert_event(
            Event(event_id="e-1", type="speeding", t=2.0, view_id="drone-1",
                  tracklet_ids=["tk-9"], summary_text="v2")
        )
        events = store.query_events("drone-1")
        assert len(events) == 1
        assert events[0].type == "speeding"
        assert events[0].summary_text == "v2"


# ---- persistence ----------------------------------------------------------


def test_persistence_across_open_close(tmp_path):
    db = tmp_path / "world.duckdb"
    with WorldStateStore(str(db)) as s1:
        s1.insert_tracklet("drone-1", "tk-1", 0.0, 1.0, [])
        s1.insert_cross_view_link(
            CrossViewLink(
                link_id="lnk-1",
                view_observations=[("drone-1", "a"), ("drone-2", "b")],
                confidence=0.5,
                created_by="geometric",
                created_at=0.0,
            )
        )

    with WorldStateStore(str(db)) as s2:
        # Shared table survived
        links = s2.query_cross_view_links()
        assert len(links) == 1 and links[0].link_id == "lnk-1"
        # Per-view tables discovered on init — can read without re-touch
        rows = s2.query_tracklets("drone-1")
        ids = {r["tracklet_id"] for r in rows}
        assert ids == {"tk-1"}
        # Append more, both records present
        s2.insert_tracklet("drone-1", "tk-2", 1.0, 2.0, [])
        assert {r["tracklet_id"] for r in s2.query_tracklets("drone-1")} == {"tk-1", "tk-2"}


# ---- M2.8: segments + tracklet.segment_idx --------------------------------


def test_insert_segment_and_get_segment_roundtrip():
    """Round-trip a fully-populated segment row + reverse-lookup by chroma id."""
    with WorldStateStore() as store:
        store.insert_segment(
            view_id="vid.mp4",
            segment_idx=3,
            start_t=30.0,
            end_t=40.0,
            source_uri="/data/vid.mp4",
            embed_chroma_id="qa-0::vid.mp4::vid::seg0003::frame::3",
            nframes_sampled=4,
            detected_classes="car,person",
            detected_counts={"car": 1, "person": 12},
        )
        # Direct fetch
        seg = store.get_segment("vid.mp4", 3)
        assert seg is not None
        assert seg["start_t"] == 30.0
        assert seg["end_t"] == 40.0
        assert seg["source_uri"] == "/data/vid.mp4"
        assert seg["embed_chroma_id"].endswith("::seg0003::frame::3")
        assert seg["nframes_sampled"] == 4
        assert seg["detected_classes"] == "car,person"
        assert seg["detected_counts"] == {"car": 1, "person": 12}

        # Reverse lookup (ChromaDB hit → segment row)
        rev = store.get_segment_by_chroma_id(seg["embed_chroma_id"])
        assert rev is not None and rev["segment_idx"] == 3


def test_insert_segment_idempotent_on_replace():
    """Re-inserting the same (view_id, segment_idx) overwrites — supports
    re-running `mva ingest` without ChromaDB id clashes (P2-04)."""
    with WorldStateStore() as store:
        store.insert_segment("v", 0, 0.0, 10.0, "/u", embed_chroma_id="a")
        store.insert_segment("v", 0, 0.0, 10.0, "/u", embed_chroma_id="b")
        seg = store.get_segment("v", 0)
        assert seg["embed_chroma_id"] == "b"


def test_query_segments_view_and_time_filter():
    with WorldStateStore() as store:
        store.insert_segment("A", 0, 0.0, 10.0, "/A")
        store.insert_segment("A", 1, 10.0, 20.0, "/A")
        store.insert_segment("B", 0, 0.0, 10.0, "/B")
        # All
        assert len(store.query_segments()) == 3
        # By view
        a_only = store.query_segments(view_id="A")
        assert len(a_only) == 2 and all(s["view_id"] == "A" for s in a_only)
        # By time window
        early = store.query_segments(t_start=0.0, t_end=5.0)
        assert len(early) == 2  # A:seg0 + B:seg0
        late = store.query_segments(view_id="A", t_start=15.0, t_end=25.0)
        assert len(late) == 1 and late[0]["segment_idx"] == 1


def test_query_segments_empty_table_returns_list():
    with WorldStateStore() as store:
        assert store.query_segments() == []
        assert store.query_segments(view_id="ghost") == []


def test_get_segment_missing_returns_none():
    with WorldStateStore() as store:
        assert store.get_segment("ghost", 0) is None
        assert store.get_segment_by_chroma_id("bogus") is None


def test_tracklet_carries_segment_idx_when_provided():
    with WorldStateStore() as store:
        store.insert_tracklet(
            "drone-1", "D1-seg0001-f0-d0", 0.0, 10.0,
            bboxes=[(0.0, 10, 10, 20, 20, "person", 0.9)],
            embedding_ref="bbox-emb-1",
            segment_idx=1,
        )
        rows = store.query_tracklets("drone-1")
        assert len(rows) == 1
        assert rows[0]["segment_idx"] == 1
        assert rows[0]["embedding_ref"] == "bbox-emb-1"


def test_tracklet_segment_idx_filter():
    """`query_tracklets(segment_idx=N)` narrows to one parent segment —
    used by retrieval to enumerate detections within a hit segment."""
    with WorldStateStore() as store:
        store.insert_tracklet("v", "t0", 0.0, 10.0, [], segment_idx=0)
        store.insert_tracklet("v", "t1", 0.0, 10.0, [], segment_idx=0)
        store.insert_tracklet("v", "t2", 10.0, 20.0, [], segment_idx=1)
        seg0 = store.query_tracklets("v", segment_idx=0)
        assert {r["tracklet_id"] for r in seg0} == {"t0", "t1"}
        seg1 = store.query_tracklets("v", segment_idx=1)
        assert {r["tracklet_id"] for r in seg1} == {"t2"}


def test_legacy_tracklet_has_null_segment_idx():
    """Tracklets written without segment_idx (legacy `mva perceive` path)
    must round-trip with segment_idx=None, not crash."""
    with WorldStateStore() as store:
        store.insert_tracklet("v", "legacy", 0.0, 1.0, [])
        rows = store.query_tracklets("v")
        assert rows[0]["segment_idx"] is None


def test_segments_persist_across_reopen(tmp_path):
    db = tmp_path / "world.duckdb"
    with WorldStateStore(str(db)) as s1:
        s1.insert_segment(
            "v", 0, 0.0, 10.0, "/u",
            embed_chroma_id="cid",
            detected_classes="person",
            detected_counts={"person": 5},
        )
    with WorldStateStore(str(db)) as s2:
        seg = s2.get_segment("v", 0)
        assert seg is not None
        assert seg["detected_counts"] == {"person": 5}
