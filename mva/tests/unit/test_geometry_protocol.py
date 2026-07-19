from mva.geometry import PoseProvider, Projector, TimeSync
from mva.geometry.fakes import StaticPoseProvider, DownwardProjector, NearestTimeSync
from mva.contracts import CameraPose, Ray, WorldPoint


_POSE = CameraPose(view_id="cam01", t=0.0, fx=600, fy=600, cx=320, cy=240,
                   quat=(0, 0, 0, 1), translation=(0, 0, 20))


def test_fakes_satisfy_protocols():
    assert isinstance(StaticPoseProvider(_POSE), PoseProvider)
    assert isinstance(DownwardProjector(StaticPoseProvider(_POSE)), Projector)
    assert isinstance(NearestTimeSync(), TimeSync)


def test_static_pose_provider_returns_pose():
    p = StaticPoseProvider(_POSE).pose("cam01", 0.0)
    assert p.translation == (0, 0, 20)


def test_downward_projector_backprojects_to_ground():
    proj = DownwardProjector(StaticPoseProvider(_POSE))
    wp = proj.backproject("cam01", (320.0, 240.0), 0.0, ground_z=0.0)
    assert isinstance(wp, WorldPoint) and wp.z == 0.0


def test_nearest_time_sync_groups_by_tolerance():
    sets = NearestTimeSync().align({"a": [0.0, 1.0], "b": [0.02, 1.03]}, tol=0.05)
    assert len(sets) == 2
    assert set(sets[0].keys()) == {"a", "b"}
