"""M4 时空契约：场景图边 / 态势事件 / 全局点预测。§5。
命名避让既有 contracts.Event(dataclass) 与 contracts.TrajectoryPrediction。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class SceneGraphEdge(BaseModel):
    t: float
    subj_global_id: str
    rel: str                       # near / left_of / approaching / inside_region ...
    obj: str                       # global_id 或 region 名
    confidence: float = Field(ge=0.0, le=1.0)


class SituationEvent(BaseModel):
    event_id: str
    kind: str                      # gathering/dispersal/intrusion/loitering/collision/anomaly/change
    t_start: float
    t_end: float
    global_ids: list[str] = []
    region: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _end_after_start(self):
        if self.t_end < self.t_start:
            raise ValueError("t_end must be >= t_start")
        return self


class GlobalPrediction(BaseModel):
    global_id: str
    t_future: float
    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)
