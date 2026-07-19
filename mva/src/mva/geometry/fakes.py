"""M2 桩实现：仅为 Phase 0 并行开工，几何不求准确。真实算法由 M2 owner 替换。"""
from __future__ import annotations

from mva.contracts import CameraPose, Ray, WorldPoint


class StaticPoseProvider:
    """恒返回同一个位姿（忽略 view_id/t）。"""
    def __init__(self, pose: CameraPose):
        self._pose = pose

    def pose(self, view_id: str, t: float) -> CameraPose:
        return self._pose


class DownwardProjector:
    """桩：把任意像素当作相机正下方一条竖直射线，与地平面 z=ground_z 求交。"""
    def __init__(self, pose_provider: StaticPoseProvider):
        self._pp = pose_provider

    def ray(self, view_id: str, pixel: tuple[float, float], t: float) -> Ray:
        cam = self._pp.pose(view_id, t)
        return Ray(origin=WorldPoint(x=cam.translation[0], y=cam.translation[1],
                                     z=cam.translation[2]),
                   direction=(0.0, 0.0, -1.0))

    def backproject(self, view_id: str, pixel: tuple[float, float], t: float,
                    ground_z: float = 0.0) -> WorldPoint:
        cam = self._pp.pose(view_id, t)
        return WorldPoint(x=cam.translation[0], y=cam.translation[1], z=ground_z)


class NearestTimeSync:
    """桩：以第一路的时间戳为锚，其余路各取 tol 内最近的一帧，凑成同刻集合。"""
    def align(self, view_timestamps: dict[str, list[float]],
              tol: float = 0.05) -> list[dict[str, float]]:
        if not view_timestamps:
            return []
        anchor_view = next(iter(view_timestamps))
        out: list[dict[str, float]] = []
        for at in view_timestamps[anchor_view]:
            group = {anchor_view: at}
            for v, ts in view_timestamps.items():
                if v == anchor_view:
                    continue
                near = [x for x in ts if abs(x - at) <= tol]
                if near:
                    group[v] = min(near, key=lambda x: abs(x - at))
            out.append(group)
        return out
