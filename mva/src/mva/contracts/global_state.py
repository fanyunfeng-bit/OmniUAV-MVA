"""M3 全局对象契约：全局对象注册表 / 观测 / 轨迹。§5。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class GlobalObject(BaseModel):
    global_id: str
    class_name: str
    first_t: float
    last_t: float
    n_views: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _last_after_first(self):
        if self.last_t < self.first_t:
            raise ValueError("last_t must be >= first_t")
        return self


class GlobalObservation(BaseModel):
    global_id: str
    view_id: str
    view_track_id: str
    t: float
    bbox: tuple[float, float, float, float]
    world_xyz: Optional[tuple[float, float, float]] = None   # 未三角化时为 None


class GlobalTrajectory(BaseModel):
    global_id: str
    t: float
    x: float
    y: float
    z: float = 0.0
    vx: Optional[float] = None
    vy: Optional[float] = None
