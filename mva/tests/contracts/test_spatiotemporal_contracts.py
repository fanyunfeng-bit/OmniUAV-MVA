import pytest
from pydantic import ValidationError
from mva.contracts import SceneGraphEdge, SituationEvent, GlobalPrediction


def test_scene_graph_edge():
    e = SceneGraphEdge(t=1.0, subj_global_id="g1", rel="near", obj="g2", confidence=0.7)
    assert e.rel == "near"


def test_situation_event_defaults_and_validator():
    ev = SituationEvent(event_id="e1", kind="gathering", t_start=0.0, t_end=5.0,
                        confidence=0.6)
    assert ev.global_ids == [] and ev.region is None
    with pytest.raises(ValidationError):
        SituationEvent(event_id="e2", kind="x", t_start=5.0, t_end=1.0, confidence=0.5)


def test_global_prediction():
    p = GlobalPrediction(global_id="g1", t_future=2.0, x=10.0, y=11.0, confidence=0.5)
    assert (p.x, p.y) == (10.0, 11.0)
