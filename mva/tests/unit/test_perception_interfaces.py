import numpy as np
from mva.perception.pipeline import (
    Track, PassthroughTracker, DensePerceptionPipeline,
)
from mva.perception.relation import NullRelationModeler


class _FakeFrameSource:
    fps = 2.0
    def iter_frames(self):
        for t in (0.0, 0.5):
            yield (t, np.zeros((8, 8, 3), np.uint8))


class _FakeDetector:
    def detect(self, frame):
        return [((0, 0, 4, 4), "boat", 0.9)]


def test_passthrough_tracker_gives_ids():
    tr = PassthroughTracker()
    out = tr.update([((0, 0, 4, 4), "boat", 0.9)], t=0.0)
    assert len(out) == 1 and out[0].class_name == "boat" and out[0].track_id


def test_dense_pipeline_runs_over_framesource():
    pipe = DensePerceptionPipeline()
    tracks = pipe.run(_FakeFrameSource(), view_id="view1",
                      detector=_FakeDetector(), tracker=PassthroughTracker())
    assert len(tracks) == 2                      # 2 帧各 1 个检测
    assert all(isinstance(t, Track) for t in tracks)
    assert {t.t for t in tracks} == {0.0, 0.5}   # 保留绝对时间
    assert all(t.view_id == "view1" for t in tracks)


def test_null_relation_modeler_is_placeholder():
    assert NullRelationModeler().model([]) == []
