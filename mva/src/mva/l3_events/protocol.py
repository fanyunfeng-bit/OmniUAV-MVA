"""L3 Reasoner Protocol — algorithmic (default) and LLM modes share this surface."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from mva.contracts import Anomaly, TrajectoryPrediction


@runtime_checkable
class Reasoner(Protocol):
    def predict_trajectory(
        self, view_id: str, tracklet_id: str, horizon: float
    ) -> Optional[TrajectoryPrediction]:
        ...

    def classify_behavior(
        self, view_id: str, tracklet_id: str, context: dict
    ) -> str:
        ...

    def detect_anomaly(
        self, view_id: str, tracklet_id: str
    ) -> Optional[Anomaly]:
        ...
