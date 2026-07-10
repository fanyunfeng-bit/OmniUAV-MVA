"""AlgorithmicReasoner — default L3 implementation (M3.2).

Constant-velocity trajectory prediction + rule-based anomaly detection:

- `predict_trajectory(view_id, tracklet_id, horizon)`: fits a constant-velocity
  model over the tracklet's bbox-center series and extrapolates `horizon`
  seconds forward at fixed waypoint spacing. Confidence drops with
  per-axis residual variance (well-aligned linear motion → ~1.0, jittery
  motion → ~0).

- `detect_anomaly(view_id, tracklet_id)`: two rules per PLAN §6.1 M3.2:
    * **loitering** — tracklet duration ≥ `loiter_min_seconds` AND max
      displacement from start < `loiter_max_displacement_px`.
    * **speed_spike** — tracklet's mean speed > `scene_mean + N·σ` of
      sibling tracklets in the same view + time window (N defaults to 3).
  Returns the first triggered rule (severity "medium" by default) or None.

Needs at least 2 bbox observations to do anything meaningful. M3.1 tracker
on MATRIX K=4 sparse sampling produces mostly single-frame tracklets →
`predict_trajectory` / `detect_anomaly` return None for those (degenerate
input, not a bug). MVU-Eval segments + tightly-sampled fixtures exercise
the rules properly.

LLM-mode bulk anomaly explanation lives in `llm_mode.py` (M4).
"""
from __future__ import annotations

import json
import math
from typing import Optional

from mva.contracts import Anomaly, TrajectoryPrediction


class AlgorithmicReasoner:
    """Algorithmic L3 reasoner. Construct with a `WorldStateStore` to give
    the methods access to tracklet bbox histories.

    Parameters
    ----------
    store : WorldStateStore
        Source of tracklet bboxes. The reasoner reads only; never writes.
    loiter_min_seconds : float
        Minimum tracklet duration to even consider a loitering call
        (default 30.0 per PLAN §6.1 M3.2). For short-segment experiments
        (10 s windows) you may want to lower this — but at risk of false
        positives on stationary detections of short duration.
    loiter_max_displacement_px : float
        If a tracklet that long has max bbox-center displacement smaller
        than this, it counts as loitering (default 20 px).
    speed_spike_sigma : float
        Speed-spike fires when a tracklet's mean speed exceeds
        `scene_mean + sigma·scene_std` (default 3.0 per PLAN).
    n_waypoints : int
        How many evenly-spaced waypoints `predict_trajectory` emits
        (default 5). The waypoint at index i corresponds to
        `horizon * (i+1) / N` seconds forward.
    """

    def __init__(
        self,
        store=None,
        *,
        loiter_min_seconds: float = 30.0,
        loiter_max_displacement_px: float = 20.0,
        speed_spike_sigma: float = 3.0,
        n_waypoints: int = 5,
    ) -> None:
        self.store = store
        self.loiter_min_seconds = loiter_min_seconds
        self.loiter_max_displacement_px = loiter_max_displacement_px
        self.speed_spike_sigma = speed_spike_sigma
        self.n_waypoints = n_waypoints

    # ------------------------------------------------------------------ Protocol

    def predict_trajectory(
        self, view_id: str, tracklet_id: str, horizon: float
    ) -> Optional[TrajectoryPrediction]:
        if horizon <= 0 or self.store is None:
            return None
        centers = self._tracklet_centers(view_id, tracklet_id)
        if len(centers) < 2:
            return None
        # Velocity from the last two observations — robust to short tracks
        (t_prev, x_prev, y_prev) = centers[-2]
        (t_last, x_last, y_last) = centers[-1]
        dt = t_last - t_prev
        if dt <= 0:
            return None
        vx = (x_last - x_prev) / dt
        vy = (y_last - y_prev) / dt
        # Evenly spaced waypoints forward of t_last (encoded as t_offset
        # from the trajectory's last observation, per contract).
        waypoints: list[tuple[float, float, float]] = []
        for i in range(1, self.n_waypoints + 1):
            t_offset = horizon * (i / self.n_waypoints)
            waypoints.append((
                float(t_offset),
                float(x_last + vx * t_offset),
                float(y_last + vy * t_offset),
            ))
        confidence = self._linear_fit_confidence(centers)
        return TrajectoryPrediction(
            view_id=view_id,
            tracklet_id=tracklet_id,
            horizon_seconds=float(horizon),
            waypoints=waypoints,
            confidence=confidence,
        )

    def classify_behavior(
        self, view_id: str, tracklet_id: str, context: dict
    ) -> str:
        # Behavior classification deferred to M5 — needs ROI-level VLM
        # captioning or a SlowFast-style action recognizer. Until then
        # we report "unknown" so callers can branch off it.
        return "unknown"

    def detect_anomaly(
        self, view_id: str, tracklet_id: str
    ) -> Optional[Anomaly]:
        if self.store is None:
            return None
        centers = self._tracklet_centers(view_id, tracklet_id)
        if len(centers) < 2:
            return None
        t_first, _, _ = centers[0]
        t_last_obs = centers[-1][0]
        duration = t_last_obs - t_first

        # Rule 1: loitering — long duration + tight cluster
        if duration >= self.loiter_min_seconds:
            x0, y0 = centers[0][1], centers[0][2]
            max_disp = 0.0
            for _, x, y in centers:
                d = math.hypot(x - x0, y - y0)
                if d > max_disp:
                    max_disp = d
            if max_disp < self.loiter_max_displacement_px:
                return Anomaly(
                    event_id=f"loitering-{view_id}-{tracklet_id}",
                    tracklet_ids=[tracklet_id],
                    t=float(t_first),
                    type="loitering",
                    severity="medium",
                )

        # Rule 2: speed spike — compared to peers in same window
        my_speed = self._mean_speed(centers)
        if my_speed is None:
            return None
        sibling_speeds = self._sibling_speeds(
            view_id, tracklet_id, t_first, t_last_obs,
        )
        if len(sibling_speeds) >= 3:
            n = len(sibling_speeds)
            mean = sum(sibling_speeds) / n
            var = sum((s - mean) ** 2 for s in sibling_speeds) / n
            std = math.sqrt(var)
            # Synthetic / homogeneous scenes give σ=0 (everyone same
            # speed); the σ-rule then can't define "outlier". Use a
            # 10%-of-mean floor so a 10×-faster track still fires.
            effective_std = max(std, abs(mean) * 0.1)
            if effective_std > 0 and my_speed > mean + self.speed_spike_sigma * effective_std:
                return Anomaly(
                    event_id=f"speed-spike-{view_id}-{tracklet_id}",
                    tracklet_ids=[tracklet_id],
                    t=float(t_first),
                    type="speed_spike",
                    severity="medium",
                )
        return None

    # ------------------------------------------------------------------ Helpers

    def _tracklet_centers(
        self, view_id: str, tracklet_id: str,
    ) -> list[tuple[float, float, float]]:
        """Fetch this tracklet's per-frame bbox centers as (t, cx, cy)."""
        tracklets = self.store.query_tracklets(view_id)
        for tk in tracklets:
            if tk["tracklet_id"] != tracklet_id:
                continue
            return _bboxes_to_centers(tk["bboxes"])
        return []

    def _sibling_speeds(
        self,
        view_id: str,
        exclude_id: str,
        t_start: float,
        t_end: float,
    ) -> list[float]:
        """Mean speeds of every other tracklet whose window overlaps
        [t_start, t_end] in this view."""
        out: list[float] = []
        for tk in self.store.query_tracklets(view_id, t_start=t_start, t_end=t_end):
            if tk["tracklet_id"] == exclude_id:
                continue
            centers = _bboxes_to_centers(tk["bboxes"])
            spd = self._mean_speed(centers)
            if spd is not None:
                out.append(spd)
        return out

    @staticmethod
    def _mean_speed(
        centers: list[tuple[float, float, float]],
    ) -> Optional[float]:
        """Total path length / total elapsed time. None if degenerate."""
        if len(centers) < 2:
            return None
        total_dist = 0.0
        total_dt = 0.0
        for (t0, x0, y0), (t1, x1, y1) in zip(centers[:-1], centers[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            total_dist += math.hypot(x1 - x0, y1 - y0)
            total_dt += dt
        if total_dt <= 0:
            return None
        return total_dist / total_dt

    @staticmethod
    def _linear_fit_confidence(
        centers: list[tuple[float, float, float]],
    ) -> float:
        """Map per-axis fit residual to a [0, 1] confidence. Few
        observations → use a conservative default."""
        n = len(centers)
        if n < 3:
            return 0.5  # 2 obs: velocity defined, variance is not
        ts = [c[0] for c in centers]
        xs = [c[1] for c in centers]
        ys = [c[2] for c in centers]
        t_mean = sum(ts) / n
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        den = sum((t - t_mean) ** 2 for t in ts)
        if den <= 0:
            return 0.5
        sx = sum((t - t_mean) * (x - x_mean) for t, x in zip(ts, xs)) / den
        sy = sum((t - t_mean) * (y - y_mean) for t, y in zip(ts, ys)) / den
        resid_sq = 0.0
        for t, x, y in zip(ts, xs, ys):
            ex = x - x_mean - sx * (t - t_mean)
            ey = y - y_mean - sy * (t - t_mean)
            resid_sq += ex * ex + ey * ey
        rmse = math.sqrt(resid_sq / n)
        # 0 px residual → 1.0; 20 px → 0.5; 60 px → ~0.25
        conf = 1.0 / (1.0 + rmse / 20.0)
        return max(0.0, min(1.0, conf))


def _bboxes_to_centers(
    bboxes,
) -> list[tuple[float, float, float]]:
    """Parse the stored bboxes JSON / list into [(t, cx, cy), ...]."""
    if isinstance(bboxes, str):
        bboxes = json.loads(bboxes)
    out: list[tuple[float, float, float]] = []
    for row in bboxes:
        # row layout from cli/ingest.py: [t, x1, y1, x2, y2, class, conf]
        if len(row) < 5:
            continue
        t, x1, y1, x2, y2 = row[0], row[1], row[2], row[3], row[4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        out.append((float(t), float(cx), float(cy)))
    return out
