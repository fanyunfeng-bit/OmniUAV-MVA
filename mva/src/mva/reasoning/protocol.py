"""M4 时空推理 Protocol：事件检测、轨迹预测。关系建模复用既有 RelationModeler。"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from mva.contracts import GlobalPrediction, GlobalTrajectory, SituationEvent


@runtime_checkable
class EventDetector(Protocol):
    def detect(self, trajectories: list[GlobalTrajectory],
               t_window: tuple[float, float]) -> list[SituationEvent]: ...


@runtime_checkable
class TrajectoryPredictor(Protocol):
    def predict(self, trajectory: list[GlobalTrajectory],
                horizon_s: float) -> list[GlobalPrediction]: ...
