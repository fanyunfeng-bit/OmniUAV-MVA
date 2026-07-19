import pytest
from pydantic import ValidationError
from mva.contracts import WorldPoint, Ray, CameraPose


def test_world_point_defaults_z():
    p = WorldPoint(x=1.0, y=2.0)
    assert (p.x, p.y, p.z) == (1.0, 2.0, 0.0)


def test_ray_holds_origin_and_direction():
    r = Ray(origin=WorldPoint(x=0, y=0, z=10), direction=(0.0, 0.0, -1.0))
    assert r.origin.z == 10
    assert r.direction == (0.0, 0.0, -1.0)


def test_camera_pose_roundtrip():
    c = CameraPose(view_id="cam01", t=1.0, fx=600, fy=600, cx=320, cy=240,
                   quat=(0, 0, 0, 1), translation=(5, 6, 20))
    d = c.model_dump()
    assert d["view_id"] == "cam01" and d["translation"] == (5, 6, 20)


def test_camera_pose_quat_must_be_length_4():
    with pytest.raises(ValidationError):
        CameraPose(view_id="c", t=0, fx=1, fy=1, cx=0, cy=0,
                   quat=(0, 0, 1), translation=(0, 0, 0))
