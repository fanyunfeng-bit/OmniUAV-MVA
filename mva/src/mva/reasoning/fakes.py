"""M4 桩实现：Phase 0 并行用。真实算法由 M4 owner 替换。"""
from __future__ import annotations

from mva.contracts import GlobalPrediction, GlobalTrajectory, SituationEvent


class NullEventDetector:
    """桩：不检测任何事件。"""
    def detect(self, trajectories, t_window) -> list[SituationEvent]:
        return []


class ConstantVelocityPredictor:
    """桩：用轨迹末两点的速度做常速外推一个点。"""
    def predict(self, trajectory: list[GlobalTrajectory],
                horizon_s: float) -> list[GlobalPrediction]:
        if len(trajectory) < 2:
            return []
        a, b = trajectory[-2], trajectory[-1]
        dt = (b.t - a.t) or 1.0
        vx, vy = (b.x - a.x) / dt, (b.y - a.y) / dt
        return [GlobalPrediction(
            global_id=b.global_id, t_future=b.t + horizon_s,
            x=b.x + vx * horizon_s, y=b.y + vy * horizon_s, confidence=0.5)]
