"""Unit tests for `AlgorithmicReasoner` (M3.2).

Synthetic in-memory fixtures: insert tracklets with known bbox histories
into a `:memory:` WorldStateStore, then check predictions + anomalies.
"""
from __future__ import annotations

import math

import pytest

from mva.contracts import Anomaly, TrajectoryPrediction
from mva.l3_events import AlgorithmicReasoner
from mva.l5_state import WorldStateStore


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _insert_track(
    store: WorldStateStore,
    view_id: str,
    tracklet_id: str,
    centers: list[tuple[float, float, float]],
    *,
    half: float = 5.0,
    cls: str = "person",
) -> None:
    """Insert a tracklet whose bbox-center series is `centers` (list of
    (t, cx, cy)). The bbox JSON layout matches ingest's writer:
    [t, x1, y1, x2, y2, class, conf]."""
    bboxes = []
    for t, cx, cy in centers:
        bboxes.append([
            float(t),
            float(cx - half), float(cy - half),
            float(cx + half), float(cy + half),
            cls, 0.9,
        ])
    t_start = centers[0][0]
    t_end = centers[-1][0]
    store.insert_tracklet(
        view_id=view_id,
        tracklet_id=tracklet_id,
        t_start=t_start, t_end=t_end,
        bboxes=bboxes,
    )


@pytest.fixture
def store():
    s = WorldStateStore(":memory:")
    yield s
    s.close()


# ----------------------------------------------------------------------
# predict_trajectory
# ----------------------------------------------------------------------


class TestPredictTrajectory:
    def test_constant_velocity_returns_valid_prediction(self, store):
        # 4 obs straight-line at 10 px/sec eastward starting at (100, 200)
        centers = [(t, 100.0 + 10.0 * t, 200.0) for t in (0.0, 1.0, 2.0, 3.0)]
        _insert_track(store, "drone-1", "tk-1", centers)
        r = AlgorithmicReasoner(store=store, n_waypoints=5)
        pred = r.predict_trajectory("drone-1", "tk-1", horizon=2.0)
        assert isinstance(pred, TrajectoryPrediction)
        assert pred.horizon_seconds == pytest.approx(2.0)
        assert len(pred.waypoints) == 5
        # Last waypoint at t_offset=2.0: extrapolated from last obs (t=3, x=130)
        # → (3+2, 130+20, 200) actually waypoint encodes t_offset only: (2.0, 150.0, 200.0)
        t_off, wx, wy = pred.waypoints[-1]
        assert t_off == pytest.approx(2.0)
        assert wx == pytest.approx(150.0, abs=1e-3)
        assert wy == pytest.approx(200.0, abs=1e-3)

    def test_ade_under_5px_for_clean_constant_velocity(self, store):
        """ADE acceptance per PLAN §6.1 M3.2: clean constant-velocity
        track → average displacement error < 5 px against GT."""
        # Build 5 historical obs at 8 px/sec, then GT 5 future obs at
        # the same velocity. The reasoner should extrapolate ~perfectly.
        v = 8.0
        hist = [(t, 50.0 + v * t, 100.0) for t in (0.0, 1.0, 2.0, 3.0, 4.0)]
        _insert_track(store, "v", "tk", hist)
        r = AlgorithmicReasoner(store=store, n_waypoints=5)
        pred = r.predict_trajectory("v", "tk", horizon=5.0)
        assert pred is not None
        last_t, last_x, _ = hist[-1]
        # Ground truth at t_offset i is (last_x + v * t_offset, 100)
        errors = []
        for (t_off, wx, wy) in pred.waypoints:
            gt_x = last_x + v * t_off
            errors.append(math.hypot(wx - gt_x, wy - 100.0))
        ade = sum(errors) / len(errors)
        assert ade < 5.0, f"ADE {ade:.2f}px exceeded 5px budget"

    def test_single_observation_returns_none(self, store):
        _insert_track(store, "v", "tk", [(0.0, 50.0, 50.0)])
        r = AlgorithmicReasoner(store=store)
        assert r.predict_trajectory("v", "tk", horizon=2.0) is None

    def test_unknown_tracklet_returns_none(self, store):
        r = AlgorithmicReasoner(store=store)
        assert r.predict_trajectory("v", "nonexistent", horizon=2.0) is None

    def test_zero_horizon_rejected(self, store):
        centers = [(0.0, 0.0, 0.0), (1.0, 10.0, 0.0)]
        _insert_track(store, "v", "tk", centers)
        r = AlgorithmicReasoner(store=store)
        assert r.predict_trajectory("v", "tk", horizon=0.0) is None

    def test_confidence_high_for_perfectly_linear_motion(self, store):
        centers = [(t, t * 5.0, t * 3.0) for t in (0.0, 1.0, 2.0, 3.0, 4.0)]
        _insert_track(store, "v", "tk", centers)
        r = AlgorithmicReasoner(store=store)
        pred = r.predict_trajectory("v", "tk", horizon=1.0)
        assert pred is not None
        assert pred.confidence > 0.95

    def test_confidence_low_for_jittery_motion(self, store):
        # Same direction trend but lots of axis-orthogonal jitter
        centers = [
            (0.0, 0.0, 0.0),
            (1.0, 10.0, 50.0),
            (2.0, 20.0, -50.0),
            (3.0, 30.0, 50.0),
            (4.0, 40.0, -50.0),
        ]
        _insert_track(store, "v", "tk", centers)
        r = AlgorithmicReasoner(store=store)
        pred = r.predict_trajectory("v", "tk", horizon=1.0)
        assert pred is not None
        assert pred.confidence < 0.7  # Significant y-axis residuals → lower


# ----------------------------------------------------------------------
# detect_anomaly — loitering
# ----------------------------------------------------------------------


class TestDetectAnomalyLoitering:
    def test_loitering_triggers_when_stationary_for_30s(self, store):
        # Person at ~(100, 100) for 35 seconds, small jitter
        centers = [(t, 100.0 + (t % 2) * 1.0, 100.0) for t in range(0, 36)]
        _insert_track(store, "v", "tk-loiter", centers)
        r = AlgorithmicReasoner(store=store)
        a = r.detect_anomaly("v", "tk-loiter")
        assert isinstance(a, Anomaly)
        assert a.type == "loitering"
        assert "tk-loiter" in a.tracklet_ids

    def test_loitering_fires_for_at_least_90_percent_of_stationary_tracks(self, store):
        """Synthetic stress: 20 stationary 30-s tracks with small jitter
        → loitering should trigger on ≥ 90% (acceptance per PLAN)."""
        r = AlgorithmicReasoner(store=store, loiter_min_seconds=30.0,
                                loiter_max_displacement_px=20.0)
        n = 20
        for i in range(n):
            centers = [(t, 100.0 + (t * 0.3), 100.0) for t in range(0, 35)]
            # max disp here = 0.3 * 34 ≈ 10 px → well under 20 px bar
            _insert_track(store, "v", f"tk-{i}", centers)
        triggered = 0
        for i in range(n):
            a = r.detect_anomaly("v", f"tk-{i}")
            if a is not None and a.type == "loitering":
                triggered += 1
        # Each tracklet is identical → all should fire; allow 90% margin
        # for any test-fixture noise.
        assert triggered >= int(0.9 * n), f"only {triggered}/{n} tracks triggered"

    def test_short_duration_does_not_trigger_loitering(self, store):
        # 5 seconds is well below the 30 s min
        centers = [(t, 100.0, 100.0) for t in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)]
        _insert_track(store, "v", "tk-short", centers)
        r = AlgorithmicReasoner(store=store, loiter_min_seconds=30.0)
        # Without sibling tracks, speed-spike also can't fire → expect None
        assert r.detect_anomaly("v", "tk-short") is None

    def test_moving_track_does_not_trigger_loitering(self, store):
        # 35 s but moving 5 px/s → 175 px total displacement
        centers = [(t, 100.0 + 5.0 * t, 100.0) for t in range(0, 36)]
        _insert_track(store, "v", "tk-walker", centers)
        r = AlgorithmicReasoner(store=store)
        a = r.detect_anomaly("v", "tk-walker")
        # Loitering should NOT fire. Without 3+ siblings, speed-spike also
        # can't fire → expect None.
        if a is not None:
            assert a.type != "loitering"

    def test_false_positive_rate_under_5_percent_on_moving_tracks(self, store):
        """Acceptance per PLAN: no-loiter FP rate ≤ 5% on moving tracks."""
        r = AlgorithmicReasoner(store=store, loiter_min_seconds=30.0,
                                loiter_max_displacement_px=20.0)
        n = 20
        false_pos = 0
        for i in range(n):
            # Walking at 3 px/s for 35 s → total 105 px displacement
            centers = [(t, 100.0 + 3.0 * t, 100.0) for t in range(0, 35)]
            _insert_track(store, "v", f"walker-{i}", centers)
        for i in range(n):
            a = r.detect_anomaly("v", f"walker-{i}")
            if a is not None and a.type == "loitering":
                false_pos += 1
        assert false_pos <= 1, f"{false_pos}/{n} FP (budget 1)"


# ----------------------------------------------------------------------
# detect_anomaly — speed spike
# ----------------------------------------------------------------------


class TestDetectAnomalySpeedSpike:
    def test_speed_spike_triggers_for_outlier(self, store):
        # 5 slow walkers (~1 px/s) + 1 fast runner (~30 px/s) over 5 s
        for i in range(5):
            centers = [(t, 50.0 + i * 30.0 + t * 1.0, 100.0)
                       for t in range(0, 6)]
            _insert_track(store, "v", f"walker-{i}", centers)
        runner_centers = [(t, 500.0 + t * 30.0, 100.0) for t in range(0, 6)]
        _insert_track(store, "v", "runner", runner_centers)
        r = AlgorithmicReasoner(
            store=store, speed_spike_sigma=2.5, loiter_min_seconds=999,
        )
        a = r.detect_anomaly("v", "runner")
        assert a is not None
        assert a.type == "speed_spike"
        assert "runner" in a.tracklet_ids

    def test_speed_spike_does_not_trigger_when_track_matches_peers(self, store):
        # All 6 tracks at the same speed → no spike
        for i in range(6):
            centers = [(t, 50.0 + i * 30.0 + t * 3.0, 100.0)
                       for t in range(0, 6)]
            _insert_track(store, "v", f"walker-{i}", centers)
        r = AlgorithmicReasoner(
            store=store, speed_spike_sigma=3.0, loiter_min_seconds=999,
        )
        for i in range(6):
            a = r.detect_anomaly("v", f"walker-{i}")
            # No anomaly expected. If any fires, it must NOT be speed_spike.
            if a is not None:
                assert a.type != "speed_spike"

    def test_speed_spike_needs_three_siblings(self, store):
        # 1 fast + 1 slow → only 1 sibling → not enough for σ estimate
        _insert_track(store, "v", "fast",
                      [(t, t * 50.0, 100.0) for t in range(0, 6)])
        _insert_track(store, "v", "slow",
                      [(t, t * 1.0, 50.0) for t in range(0, 6)])
        r = AlgorithmicReasoner(
            store=store, speed_spike_sigma=1.0, loiter_min_seconds=999,
        )
        # Either tracklet should not fire speed-spike (only 1 peer).
        assert r.detect_anomaly("v", "fast") is None
        assert r.detect_anomaly("v", "slow") is None


# ----------------------------------------------------------------------
# classify_behavior — placeholder, deferred to M5
# ----------------------------------------------------------------------


def test_classify_behavior_returns_unknown(store):
    r = AlgorithmicReasoner(store=store)
    assert r.classify_behavior("v", "tk", context={}) == "unknown"


# ----------------------------------------------------------------------
# Reasoner Protocol structural check
# ----------------------------------------------------------------------


def test_algorithmic_reasoner_satisfies_protocol():
    """`isinstance(reasoner, Reasoner)` works because the Protocol is
    `@runtime_checkable` — any drift in the public API breaks here."""
    from mva.l3_events import Reasoner
    r = AlgorithmicReasoner()
    assert isinstance(r, Reasoner)
