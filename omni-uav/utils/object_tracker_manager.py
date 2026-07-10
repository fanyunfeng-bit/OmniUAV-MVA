"""Object tracker manager for handling multiple tracked objects."""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from PyQt5 import QtGui

from .kcf_tracker import KCFTracker


class TrackedObject:
    """Represents a single tracked object."""

    def __init__(self, tracker_id: int, bbox: Tuple[int, int, int, int], label: str = ""):
        """Initialize tracked object.

        Args:
            tracker_id: Unique ID for this tracker
            bbox: Initial bounding box [x, y, w, h]
            label: Optional label for the object
        """
        self.tracker_id = tracker_id
        self.bbox = bbox  # [x, y, w, h]
        self.label = label
        self.tracker = KCFTracker(padding=2.5, features='color', kernel='linear')
        self.active = True
        self.color = self._generate_color(tracker_id)

    def _generate_color(self, tracker_id: int) -> Tuple[int, int, int]:
        """Generate a unique color for this tracker."""
        np.random.seed(tracker_id)
        return tuple(np.random.randint(50, 255, 3).tolist())


class ObjectTrackerManager:
    """Manages multiple object trackers."""

    def __init__(self):
        self.trackers: Dict[int, TrackedObject] = {}
        self.next_id = 0

    def add_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], label: str = "") -> int:
        """Add a new tracker for an object.

        Args:
            frame: Current frame (BGR format)
            bbox: Bounding box [x, y, w, h]
            label: Optional label for the object

        Returns:
            Tracker ID
        """
        tracker_id = self.next_id
        self.next_id += 1

        tracked_obj = TrackedObject(tracker_id, bbox, label)
        tracked_obj.tracker.init(frame, bbox)
        self.trackers[tracker_id] = tracked_obj

        return tracker_id

    def remove_tracker(self, tracker_id: int):
        """Remove a tracker."""
        if tracker_id in self.trackers:
            del self.trackers[tracker_id]

    def clear_all(self):
        """Remove all trackers."""
        self.trackers.clear()
        self.next_id = 0

    def update(self, frame: np.ndarray):
        """Update all trackers with new frame.

        Args:
            frame: New frame (BGR format)
        """
        to_remove = []
        for tracker_id, tracked_obj in self.trackers.items():
            if not tracked_obj.active:
                continue

            result = tracked_obj.tracker.update(frame)
            if result is None:
                # Tracking failed
                tracked_obj.active = False
                to_remove.append(tracker_id)
            else:
                x1, y1, x2, y2 = result
                tracked_obj.bbox = (x1, y1, x2 - x1, y2 - y1)

        # Remove failed trackers
        for tracker_id in to_remove:
            self.remove_tracker(tracker_id)

    def draw_trackers(self, frame: np.ndarray) -> np.ndarray:
        """Draw all active trackers on frame.

        Args:
            frame: Frame to draw on (BGR format)

        Returns:
            Frame with trackers drawn
        """
        result = frame.copy()

        for tracked_obj in self.trackers.values():
            if not tracked_obj.active:
                continue

            x, y, w, h = tracked_obj.bbox
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), tracked_obj.color, 3)

            # Draw label
            label_text = f"Track #{tracked_obj.tracker_id}"
            if tracked_obj.label:
                label_text += f": {tracked_obj.label}"

            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                result,
                (x1, y1 - text_height - 8),
                (x1 + text_width, y1),
                tracked_obj.color,
                -1
            )

            # Draw label text
            cv2.putText(
                result,
                label_text,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return result

    def get_active_count(self) -> int:
        """Get number of active trackers."""
        return sum(1 for t in self.trackers.values() if t.active)

    def get_active_tracks(self) -> Dict[int, Dict]:
        """Get active tracker information in dictionary format.

        Returns:
            Dict mapping tracker_id to dict with bbox and label
        """
        active_tracks = {}
        for tracker_id, tracked_obj in self.trackers.items():
            if tracked_obj.active:
                active_tracks[tracker_id] = {
                    "bbox": tracked_obj.bbox,
                    "label": tracked_obj.label
                }
        return active_tracks
