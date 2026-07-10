from mva.service.retrieval import strip_scene, parse_hits, enrich_segment_time


def test_strip_scene():
    assert strip_scene("Reservoir::view1") == "view1"
    assert strip_scene("view2") == "view2"


def test_parse_hits():
    raw = [
        {"id": "a", "distance": 0.1, "document": "view1 [0.0-10.0s]",
         "metadata": {"view_id": "Reservoir::view1", "segment_idx": 0,
                      "vector_kind": "segment", "class_name": None}},
        {"id": "b", "distance": 0.4, "document": "airplane @ view1",
         "metadata": {"view_id": "Reservoir::view1", "segment_idx": 0,
                      "vector_kind": "bbox", "class_name": "airplane",
                      "tracklet_id": "seg0000-track1"}},
    ]
    hits = parse_hits(raw)
    assert hits[0]["view_id"] == "view1"
    assert hits[0]["kind"] == "segment"
    assert abs(hits[0]["score"] - 0.9) < 1e-6
    assert hits[1]["kind"] == "bbox"
    assert hits[1]["class_name"] == "airplane"


class _FakeStore:
    def __init__(self, rows): self._rows = rows
    def execute_readonly(self, sql, *a, **k): return self._rows


def test_enrich_segment_time():
    hit = {"view_id": "view1", "segment_idx": 0, "kind": "segment"}
    store = _FakeStore([{"start_t": 0.0, "source_uri": "/x/view1.mp4"}])
    out = enrich_segment_time(hit, store)
    assert out["t"] == 0.0
    assert out["source_uri"] == "/x/view1.mp4"
