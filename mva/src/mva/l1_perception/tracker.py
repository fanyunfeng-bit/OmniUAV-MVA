"""L1 multi-object tracker — consolidates same-identity detections across
the K sampled frames of a segment.

Two algorithms available:

- `"iou_greedy"` (default): greedy IoU matching against the previous
  frame, no Kalman, no two-stage matching. Stateless beyond "what was
  in the last frame," deterministic, no external deps. **The right
  choice for our segment-local setup** (K=4 frames, reset between
  segments) because every detection emits a track id immediately on
  first sighting.

- `"bytetrack"`: wraps `boxmot.trackers.bytetrack.bytetrack.ByteTrack`.
  This is the famous ByteTrack paper's two-stage matching + Kalman
  prediction implementation. **Not recommended for K ≤ 4**: new tracks
  are not output until they are re-matched in the next frame
  (`STrack.is_activated` is only set to True on frame_id=1 OR after a
  successful re-match). Combined with our segment-reset, that drops
  ~95% of detections in real MATRIX D1 (empirically: 846 → 24
  tracklets). Available so we can ablate / use it when K is large
  (≥10) and segments don't reset (a v2+ change).

A `ByteTracker` instance is stateful per view: caller creates one per
view, calls `.update(detections, h, w)` for each frame in a segment,
and calls `.reset()` at segment boundaries (segment-local track IDs;
cross-segment merging is M5 work).
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from mva.l1_perception.detector import Detection


Algorithm = Literal["iou_greedy", "bytetrack", "botsort"]


def _try_import(path: str, name: str):
    try:
        mod = __import__(path, fromlist=[name])
        return getattr(mod, name)
    except (ImportError, AttributeError):
        return None


_BOXMOT_BYTETRACK = _try_import("boxmot.trackers.bytetrack.bytetrack", "ByteTrack")
_BOXMOT_BOTSORT = _try_import("boxmot.trackers.botsort.botsort", "BotSort")


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    ub = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = ua + ub - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class ByteTracker:
    """Per-view stateful multi-object tracker. See module docstring for
    algorithm trade-offs.

    Parameters
    ----------
    conf_threshold : float
        Drop detections with confidence below this.
    iou_threshold : float
        IoU bar for associating a new bbox with a previous track.
    algorithm : str
        ``"iou_greedy"`` (default) or ``"bytetrack"``.
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        algorithm: Algorithm = "iou_greedy",
    ) -> None:
        if algorithm not in ("iou_greedy", "bytetrack", "botsort"):
            raise ValueError(
                f"algorithm must be 'iou_greedy', 'bytetrack' or 'botsort', "
                f"got {algorithm!r}"
            )
        if algorithm == "bytetrack" and _BOXMOT_BYTETRACK is None:
            raise ImportError(
                "algorithm='bytetrack' requires boxmot. "
                "Install with: pip install 'mva[detection]'"
            )
        if algorithm == "botsort" and _BOXMOT_BOTSORT is None:
            raise ImportError(
                "algorithm='botsort' requires boxmot. "
                "Install with: pip install 'mva[detection]'"
            )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.algorithm = algorithm
        self._next_id = 1
        self._prev: list[tuple[int, tuple[float, float, float, float]]] = []
        self._impl: Optional[object] = None  # lazy boxmot.ByteTrack

    @property
    def is_simple(self) -> bool:
        """True when running the IoU-greedy path (the default).

        Retained for compatibility with the M3.1 unit tests' `is_mock`
        assertion semantics (renamed)."""
        return self.algorithm == "iou_greedy"

    def reset(self) -> None:
        """Clear all state. Use at segment boundaries — next update()
        starts fresh with track id 1."""
        self._next_id = 1
        self._prev = []
        if self._impl is not None and hasattr(self._impl, "reset"):
            self._impl.reset()

    def update(
        self,
        detections: list[Detection],
        frame_h: int,
        frame_w: int,
        frame: Optional[np.ndarray] = None,
    ) -> list[tuple[Detection, int]]:
        """Feed one frame's detections; return [(det, track_id), ...]
        for the dets that the tracker accepted (length may be ≤ input;
        low-confidence dets can be dropped).

        `frame` (BGR image) is optional and only used by `botsort` for camera
        motion compensation (CMC) — pass it for moving-drone footage. Without
        it, botsort falls back to a blank frame (CMC inert)."""
        if not detections:
            self._prev = []
            return []
        if self.algorithm == "iou_greedy":
            return self._update_iou_greedy(detections)
        return self._update_boxmot(detections, frame_h, frame_w, frame)

    # --- IoU greedy (default) --------------------------------------------

    def _update_iou_greedy(
        self,
        detections: list[Detection],
    ) -> list[tuple[Detection, int]]:
        """Greedy IoU matching against last frame's tracks. Deterministic.

        For each detection (in input order):
          1. Compute IoU vs every unmatched previous track.
          2. If best IoU ≥ iou_threshold, inherit that track's id.
          3. Otherwise, assign a fresh id from `_next_id`.

        Detections below `conf_threshold` are dropped. Previous-frame
        tracks that don't match any current detection are forgotten on
        the next call (they don't persist beyond one frame — appropriate
        for our segment-reset model where K ≤ 10 frames at coarse 2 fps
        sampling, so motion-based prediction would be wrong anyway).
        """
        prev_used = [False] * len(self._prev)
        result: list[tuple[Detection, int]] = []
        for det in detections:
            if det.confidence < self.conf_threshold:
                continue
            best_iou = 0.0
            best_idx = -1
            for i, (_, prev_box) in enumerate(self._prev):
                if prev_used[i]:
                    continue
                iou = _bbox_iou(det.bbox, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                prev_used[best_idx] = True
                tid = self._prev[best_idx][0]
            else:
                tid = self._next_id
                self._next_id += 1
            result.append((det, tid))
        self._prev = [(tid, det.bbox) for det, tid in result]
        return result

    # --- boxmot ByteTrack / BoTSORT (opt-in) ----------------------------

    def _make_boxmot_impl(self):
        """Construct the boxmot tracker for the selected algorithm."""
        if self.algorithm == "bytetrack":
            assert _BOXMOT_BYTETRACK is not None  # narrowed by ctor
            return _BOXMOT_BYTETRACK(
                min_conf=self.conf_threshold,
                track_thresh=self.conf_threshold,
                match_thresh=self.iou_threshold,
                min_hits=1,
            )
        assert _BOXMOT_BOTSORT is not None
        # with_reid=False → motion + CMC only (no ReID weights to download);
        # CMC (camera-motion-comp) is the win on moving-drone footage.
        return _BOXMOT_BOTSORT(
            reid_model=None,
            with_reid=False,
            new_track_thresh=self.conf_threshold,
            track_buffer=30,
        )

    def _update_boxmot(
        self,
        detections: list[Detection],
        frame_h: int,
        frame_w: int,
        frame: Optional[np.ndarray] = None,
    ) -> list[tuple[Detection, int]]:
        if self._impl is None:
            self._impl = self._make_boxmot_impl()
        dets = np.array(
            [
                [
                    d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                    d.confidence, d.class_id,
                ]
                for d in detections
            ],
            dtype=np.float32,
        )
        # botsort's CMC needs the real frame; bytetrack ignores it. Fall back to
        # a blank canvas of the right size when no frame is supplied.
        if frame is not None:
            img = frame
        else:
            img = np.zeros((max(1, frame_h), max(1, frame_w), 3), dtype=np.uint8)
        out = self._impl.update(dets, img)  # type: ignore[union-attr]
        result: list[tuple[Detection, int]] = []
        if out.size == 0:
            return result
        # TrackResults columns: [x1, y1, x2, y2, track_id, conf, cls, input_idx]
        for row in out:
            input_idx = int(row[-1])
            tid = int(row[4])
            if 0 <= input_idx < len(detections):
                result.append((detections[input_idx], tid))
        return result
