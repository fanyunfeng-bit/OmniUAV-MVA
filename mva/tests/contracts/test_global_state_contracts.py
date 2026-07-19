import pytest
from pydantic import ValidationError
from mva.contracts import GlobalObject, GlobalObservation, GlobalTrajectory


def test_global_object_valid():
    g = GlobalObject(global_id="g1", class_name="car", first_t=0.0, last_t=10.0,
                     n_views=2, confidence=0.8)
    assert g.n_views == 2


def test_global_object_rejects_last_before_first():
    with pytest.raises(ValidationError):
        GlobalObject(global_id="g", class_name="car", first_t=5.0, last_t=1.0,
                     n_views=1, confidence=0.5)


def test_global_object_rejects_bad_confidence():
    with pytest.raises(ValidationError):
        GlobalObject(global_id="g", class_name="car", first_t=0, last_t=1,
                     n_views=1, confidence=1.5)


def test_observation_world_xyz_optional():
    o = GlobalObservation(global_id="g1", view_id="cam01", view_track_id="t1",
                          t=0.0, bbox=(1, 2, 3, 4))
    assert o.world_xyz is None


def test_trajectory_defaults():
    p = GlobalTrajectory(global_id="g1", t=0.0, x=5.0, y=6.0)
    assert (p.z, p.vx, p.vy) == (0.0, None, None)
