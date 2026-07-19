from mva.l5_state.duckdb_store import WorldStateStore
from mva.contracts import CameraPose, GlobalObject, GlobalObservation, GlobalTrajectory


def _store():
    return WorldStateStore(":memory:")


def test_camera_pose_roundtrip():
    s = _store()
    s.insert_camera_pose(CameraPose(view_id="cam01", t=0.0, fx=600, fy=600,
                                    cx=320, cy=240, quat=(0, 0, 0, 1),
                                    translation=(1, 2, 20)))
    rows = s.query_camera_poses("cam01")
    assert len(rows) == 1 and rows[0]["tz"] == 20.0
    s.close()


def test_global_object_roundtrip():
    s = _store()
    s.insert_global_object(GlobalObject(global_id="g1", class_name="car",
                                        first_t=0, last_t=5, n_views=2, confidence=0.8))
    rows = s.query_global_objects()
    assert rows[0]["global_id"] == "g1" and rows[0]["n_views"] == 2
    s.close()


def test_global_observation_and_trajectory():
    s = _store()
    s.insert_global_observation(GlobalObservation(global_id="g1", view_id="cam01",
                                view_track_id="t1", t=0.0, bbox=(1, 2, 3, 4),
                                world_xyz=(5, 6, 0)))
    s.insert_global_trajectory(GlobalTrajectory(global_id="g1", t=0.0, x=5, y=6))
    assert s.query_global_observations("g1")[0]["wx"] == 5.0
    assert s.query_global_trajectory("g1")[0]["y"] == 6.0
    s.close()
