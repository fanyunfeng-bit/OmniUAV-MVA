"""M2 几何契约：世界点、射线、相机位姿。设计见
docs/superpowers/specs/2026-07-17-modular-architecture-global-3d-fusion-design.md §5。"""
from __future__ import annotations

from pydantic import BaseModel, field_validator


class WorldPoint(BaseModel):
    x: float
    y: float
    z: float = 0.0


class Ray(BaseModel):
    origin: WorldPoint                          # 相机中心，世界系
    direction: tuple[float, float, float]       # 单位方向向量，世界系


class CameraPose(BaseModel):
    view_id: str
    t: float
    fx: float
    fy: float
    cx: float
    cy: float
    quat: tuple[float, float, float, float]     # world<-cam 旋转 (qx,qy,qz,qw)
    translation: tuple[float, float, float]     # 相机中心，世界系

    @field_validator("quat")
    @classmethod
    def _quat_len4(cls, v):
        if len(v) != 4:
            raise ValueError("quat must be length-4 (qx,qy,qz,qw)")
        return v
