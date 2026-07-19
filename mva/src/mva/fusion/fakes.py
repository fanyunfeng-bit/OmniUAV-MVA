"""M3 桩实现：Phase 0 并行用，关联/三角化不求准确。真实算法由 M3 owner 替换。"""
from __future__ import annotations

from typing import Any

from mva.contracts import GlobalObject, Ray, WorldPoint


class SingletonAssociator:
    """桩：不做跨视角关联，每个观测各自成一组。"""
    def associate(self, view_tracklets_at_t: list[Any], geometry: Any) -> list[list[Any]]:
        return [[obs] for obs in view_tracklets_at_t]


class CentroidTriangulator:
    """桩：取所有射线原点的均值当作 3D 位置（真实实现应做射线求交）。"""
    def triangulate(self, rays: list[Ray]) -> WorldPoint:
        if not rays:
            return WorldPoint(x=0.0, y=0.0, z=0.0)
        n = len(rays)
        sx = sum(r.origin.x for r in rays) / n
        sy = sum(r.origin.y for r in rays) / n
        sz = sum(r.origin.z for r in rays) / n
        return WorldPoint(x=sx, y=sy, z=sz)


class CountingGlobalTracker:
    """桩：每组发一个 GlobalObject，global_id 用序号。"""
    def step(self, groups_at_t: list[Any], t: float) -> list[GlobalObject]:
        return [
            GlobalObject(global_id=f"g{i}", class_name="unknown",
                         first_t=t, last_t=t, n_views=max(1, len(g)), confidence=0.5)
            for i, g in enumerate(groups_at_t)
        ]
