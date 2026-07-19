"""M3 全局融合 Protocol：跨视角关联、三角化、时序全局跟踪。"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mva.contracts import GlobalObject, Ray, WorldPoint


@runtime_checkable
class CrossViewAssociator(Protocol):
    def associate(self, view_tracklets_at_t: list[Any],
                  geometry: Any) -> list[list[Any]]: ...


@runtime_checkable
class Triangulator(Protocol):
    def triangulate(self, rays: list[Ray]) -> WorldPoint: ...


@runtime_checkable
class GlobalTracker(Protocol):
    def step(self, groups_at_t: list[list[Any]], t: float) -> list[GlobalObject]: ...
