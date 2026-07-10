"""L3 output contracts.

- TrajectoryPrediction: produced by `Reasoner.predict_trajectory`
- Anomaly: produced by `Reasoner.detect_anomaly`

Both algorithmic mode (M3) and LLM mode (M4) must conform — verified by
`tests/contracts/`.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TrajectoryPrediction(BaseModel):
    view_id: str
    tracklet_id: str
    horizon_seconds: float = Field(gt=0)
    waypoints: list[tuple[float, float, float]]  # [(t_offset, x, y), ...]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("waypoints")
    @classmethod
    def at_least_one_waypoint(
        cls, v: list[tuple[float, float, float]]
    ) -> list[tuple[float, float, float]]:
        if len(v) < 1:
            raise ValueError("TrajectoryPrediction must have at least one waypoint")
        return v


class Anomaly(BaseModel):
    event_id: str
    tracklet_ids: list[str]
    t: float
    type: str
    severity: Literal["low", "medium", "high"]
    explanation: Optional[str] = None  # LLM mode may fill; algorithmic mode usually None
