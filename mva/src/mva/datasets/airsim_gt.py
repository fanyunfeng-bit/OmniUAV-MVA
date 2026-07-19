"""AirSim 真值适配器：读真值相机位姿 + 目标 3D 位置。

既做 M2 的位姿来源起步（不必先啃 SLAM/GPS 标定），又当 M3/M4/预测的评测 GT。
真实无人机换 GPS/IMU+VO 时，只需另写一个提供相同 CameraPose 契约的适配器，下游不变。

GT JSON 格式::

    {
      "cameras": [{"view_id","t","fx","fy","cx","cy","quat":[4],"translation":[3]}, ...],
      "objects": [{"global_id","class_name","t","world":[x,y,z]}, ...]
    }
"""
from __future__ import annotations

import json
from pathlib import Path

from mva.contracts import CameraPose, GlobalObject, WorldPoint


class AirSimGT:
    def __init__(self, path: str):
        self._data = json.loads(Path(path).read_text())

    def camera_poses(self) -> list[CameraPose]:
        out = []
        for c in self._data.get("cameras", []):
            out.append(CameraPose(
                view_id=c["view_id"], t=float(c["t"]),
                fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"],
                quat=tuple(c["quat"]), translation=tuple(c["translation"]),
            ))
        return out

    def object_positions(self) -> list[tuple[GlobalObject, WorldPoint]]:
        out = []
        for o in self._data.get("objects", []):
            t = float(o["t"])
            obj = GlobalObject(global_id=o["global_id"], class_name=o["class_name"],
                               first_t=t, last_t=t, n_views=1, confidence=1.0)
            wx, wy, wz = o["world"]
            out.append((obj, WorldPoint(x=wx, y=wy, z=wz)))
        return out
