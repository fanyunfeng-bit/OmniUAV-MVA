from types import SimpleNamespace
from mva.cli.ingest import _add_bbox_vector


class _CapVStore:
    def __init__(self): self.extra = None
    def add(self, vector, vector_type, view_id, tracklet_id,
            extra_metadata=None, document=None):
        self.extra = extra_metadata
        return "chroma-id-1"


def test_bbox_vector_carries_segment_time():
    seg = SimpleNamespace(start_t=0.0, end_t=10.0, view_id="view1",
                          segment_idx=0, source_uri="/x/view1.mp4")
    det = SimpleNamespace(bbox=(1.0, 2.0, 3.0, 4.0), class_name="car", confidence=0.9)
    vs = _CapVStore()
    _add_bbox_vector(vs, [0.1] * 8, "Scene", seg, "track1", 0, 0, det,
                     n_frames=2, classes_in_track="car")
    assert vs.extra["start_t"] == 0.0
    assert vs.extra["end_t"] == 10.0
    assert vs.extra["view_id_raw"] == "view1"     # 既有字段未破坏
