import json
from mva.datasets.airsim_gt import AirSimGT
from mva.contracts import CameraPose, GlobalObject, WorldPoint


def _write_gt(tmp_path):
    gt = {
        "cameras": [{"view_id": "cam01", "t": 0.0, "fx": 600, "fy": 600,
                     "cx": 320, "cy": 240, "quat": [0, 0, 0, 1],
                     "translation": [1, 2, 20]}],
        "objects": [{"global_id": "obj1", "class_name": "car", "t": 0.0,
                     "world": [5.0, 6.0, 0.0]}],
    }
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(gt))
    return str(p)


def test_camera_poses_parsed(tmp_path):
    a = AirSimGT(_write_gt(tmp_path))
    poses = a.camera_poses()
    assert len(poses) == 1
    assert isinstance(poses[0], CameraPose)
    assert poses[0].translation == (1, 2, 20)


def test_object_positions_parsed(tmp_path):
    a = AirSimGT(_write_gt(tmp_path))
    objs = a.object_positions()
    assert len(objs) == 1
    obj, wp = objs[0]
    assert isinstance(obj, GlobalObject) and isinstance(wp, WorldPoint)
    assert obj.global_id == "obj1" and (wp.x, wp.y) == (5.0, 6.0)
