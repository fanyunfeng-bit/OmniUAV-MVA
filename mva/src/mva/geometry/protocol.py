"""M2 几何 Protocol：位姿、投影、时序同步。换算法=换实现，不动调用方。"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from mva.contracts import CameraPose, Ray, WorldPoint


@runtime_checkable
class PoseProvider(Protocol):
    def pose(self, view_id: str, t: float) -> CameraPose: ...


@runtime_checkable
class Projector(Protocol):
    def ray(self, view_id: str, pixel: tuple[float, float], t: float) -> Ray: ...
    def backproject(self, view_id: str, pixel: tuple[float, float], t: float,
                    ground_z: float = 0.0) -> WorldPoint: ...


@runtime_checkable
class TimeSync(Protocol):
    def align(self, view_timestamps: dict[str, list[float]],
              tol: float = 0.05) -> list[dict[str, float]]: ...
