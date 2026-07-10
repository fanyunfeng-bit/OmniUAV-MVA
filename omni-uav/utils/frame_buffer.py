"""
Frame buffer system for storing recent frames from multiple cameras.
Supports temporal analysis by maintaining a circular buffer of frames with timestamps.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from PyQt5 import QtGui, QtCore


@dataclass
class FrameEntry:
    """Single frame entry with metadata."""
    frame: QtGui.QImage
    timestamp: float  # Milliseconds since epoch
    frame_number: int  # Sequential frame number for this camera


class FrameBuffer:
    """Circular buffer for storing recent frames from multiple cameras."""

    def __init__(self, max_frames_per_camera: int = 100):
        """Initialize frame buffer.

        Args:
            max_frames_per_camera: Maximum number of frames to store per camera
        """
        self.max_frames = max_frames_per_camera
        self.buffers: Dict[str, deque] = {}
        self.frame_counters: Dict[str, int] = {}

    def add_frame(self, camera_id: str, frame: QtGui.QImage, timestamp: Optional[float] = None):
        """Add a frame to the buffer for a specific camera.

        Args:
            camera_id: Unique identifier for the camera (e.g., "无人机-01")
            frame: QImage frame to store
            timestamp: Timestamp in milliseconds (uses current time if None)
        """
        if timestamp is None:
            timestamp = QtCore.QDateTime.currentMSecsSinceEpoch()

        # Initialize buffer for this camera if needed
        if camera_id not in self.buffers:
            self.buffers[camera_id] = deque(maxlen=self.max_frames)
            self.frame_counters[camera_id] = 0

        # Increment frame counter
        frame_number = self.frame_counters[camera_id]
        self.frame_counters[camera_id] += 1

        # Create frame entry and add to buffer
        entry = FrameEntry(
            frame=frame.copy(),  # Make a copy to avoid reference issues
            timestamp=timestamp,
            frame_number=frame_number
        )
        self.buffers[camera_id].append(entry)

    def get_recent_frames(
        self,
        camera_id: str,
        count: int = 5,
        interval: int = 1
    ) -> List[Tuple[QtGui.QImage, float, int]]:
        """Get recent frames from a camera with specified interval.

        Args:
            camera_id: Camera identifier
            count: Number of frames to retrieve
            interval: Frame interval (1 = consecutive, 2 = every other frame, etc.)

        Returns:
            List of tuples (frame, timestamp, frame_number) in chronological order
            Returns empty list if not enough frames available
        """
        if camera_id not in self.buffers:
            return []

        buffer = self.buffers[camera_id]
        buffer_size = len(buffer)

        # Calculate required buffer size
        required_size = (count - 1) * interval + 1

        if buffer_size < required_size:
            return []  # Not enough frames yet

        # Extract frames with interval, starting from the most recent
        result = []
        for i in range(count):
            # Calculate index from the end of the buffer
            idx = buffer_size - 1 - (i * interval)
            if idx >= 0:
                entry = buffer[idx]
                result.append((entry.frame, entry.timestamp, entry.frame_number))

        # Reverse to get chronological order (oldest to newest)
        result.reverse()
        return result

    def get_frames_by_time(
        self,
        camera_id: str,
        start_ago_sec: float,
        end_ago_sec: float,
        max_frames: int = 5,
    ) -> List[Tuple[QtGui.QImage, float, int]]:
        """[MOD 2026-07-09 | 步骤3] 按"多少秒前"的时间窗取历史帧。

        取时间戳落在 [now-end_ago, now-start_ago] 内的帧(start_ago<=end_ago，都表示"多少秒前")，
        均匀采样至多 max_frames 帧。返回 (frame, timestamp, frame_number)，按时间旧->新。
        """
        if camera_id not in self.buffers or len(self.buffers[camera_id]) == 0:
            return []
        now = QtCore.QDateTime.currentMSecsSinceEpoch()
        lo = now - end_ago_sec * 1000.0    # 更早的边界(时间戳更小)
        hi = now - start_ago_sec * 1000.0  # 更近的边界(时间戳更大)
        sel = [e for e in self.buffers[camera_id] if lo <= e.timestamp <= hi]
        if not sel:
            return []
        if len(sel) > max_frames and max_frames > 0:
            step = len(sel) / float(max_frames)
            sel = [sel[min(len(sel) - 1, int(i * step))] for i in range(max_frames)]
        return [(e.frame, e.timestamp, e.frame_number) for e in sel]

    def get_latest_frame(self, camera_id: str) -> Optional[Tuple[QtGui.QImage, float, int]]:
        """Get the most recent frame from a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            Tuple of (frame, timestamp, frame_number) or None if no frames
        """
        if camera_id not in self.buffers or len(self.buffers[camera_id]) == 0:
            return None

        entry = self.buffers[camera_id][-1]
        return (entry.frame, entry.timestamp, entry.frame_number)

    def get_buffer_size(self, camera_id: str) -> int:
        """Get the current number of frames stored for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            Number of frames in buffer
        """
        if camera_id not in self.buffers:
            return 0
        return len(self.buffers[camera_id])

    def clear_camera(self, camera_id: str):
        """Clear all frames for a specific camera.

        Args:
            camera_id: Camera identifier
        """
        if camera_id in self.buffers:
            self.buffers[camera_id].clear()
            self.frame_counters[camera_id] = 0

    def clear_all(self):
        """Clear all frames from all cameras."""
        self.buffers.clear()
        self.frame_counters.clear()

    def get_camera_ids(self) -> List[str]:
        """Get list of all camera IDs with frames in buffer.

        Returns:
            List of camera identifiers
        """
        return list(self.buffers.keys())

    def has_sufficient_frames(self, camera_id: str, count: int, interval: int) -> bool:
        """Check if buffer has enough frames for the requested retrieval.

        Args:
            camera_id: Camera identifier
            count: Number of frames needed
            interval: Frame interval

        Returns:
            True if sufficient frames available
        """
        required_size = (count - 1) * interval + 1
        return self.get_buffer_size(camera_id) >= required_size
