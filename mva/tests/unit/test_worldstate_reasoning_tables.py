from mva.l5_state.duckdb_store import WorldStateStore
from mva.contracts import SceneGraphEdge, SituationEvent, GlobalPrediction


def test_scene_graph_and_event_and_prediction_roundtrip():
    s = WorldStateStore(":memory:")
    s.insert_scene_graph_edge(SceneGraphEdge(t=1.0, subj_global_id="g1", rel="near",
                                             obj="g2", confidence=0.7))
    s.insert_situation_event(SituationEvent(event_id="e1", kind="gathering",
                              t_start=0.0, t_end=5.0, global_ids=["g1", "g2"],
                              confidence=0.6))
    s.insert_global_prediction(GlobalPrediction(global_id="g1", t_future=2.0,
                               x=10.0, y=11.0, confidence=0.5))
    assert s.query_scene_graph_edges()[0]["rel"] == "near"
    assert s.query_situation_events()[0]["kind"] == "gathering"
    assert s.query_global_predictions("g1")[0]["x"] == 10.0
    s.close()
