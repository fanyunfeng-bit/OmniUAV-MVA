"""
Cross-camera tracking manager for multi-UAV object tracking.
Maintains consistent object IDs across multiple camera views.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .feature_extractor import FeatureExtractor


@dataclass
class GlobalTrack:
    """Represents a globally tracked object across cameras."""
    global_id: int
    features: np.ndarray
    camera_tracks: Dict[str, int]  # camera_id -> local_tracker_id
    last_seen_frame: int
    color: Tuple[int, int, int]
    label: str = ""
    active: bool = True


class CrossCameraTracker:
    """Manages object tracking across multiple camera views."""

    def __init__(self, similarity_threshold: float = 0.7,
                 feature_type: str = "color_histogram",
                 max_frames_missing: int = 30):
        """Initialize cross-camera tracker.

        Args:
            similarity_threshold: Minimum similarity for matching (0-1)
            feature_type: Type of features to use
            max_frames_missing: Max frames before removing inactive track
        """
        self.similarity_threshold = similarity_threshold
        self.feature_extractor = FeatureExtractor(feature_type)
        self.max_frames_missing = max_frames_missing

        self.global_tracks: Dict[int, GlobalTrack] = {}
        self.next_global_id = 1
        self.current_frame = 0

    def update(self, camera_id: str, frame: np.ndarray,
               local_tracks: List[Tuple[int, Tuple[int, int, int, int], str]]):
        """Update with new tracks from a camera.

        Args:
            camera_id: Identifier for the camera
            frame: Current frame from camera (BGR format)
            local_tracks: List of (tracker_id, bbox, label) tuples
        """
        self.current_frame += 1

        # Extract features for all local tracks
        track_features = []
        for tracker_id, bbox, label in local_tracks:
            features = self.feature_extractor.extract(frame, bbox)
            track_features.append((tracker_id, bbox, label, features))

        # Match to existing global tracks
        matched_globals = set()
        for tracker_id, bbox, label, features in track_features:
            best_match_id = None
            best_similarity = self.similarity_threshold

            # Find best matching global track
            for global_id, global_track in self.global_tracks.items():
                if not global_track.active:
                    continue

                similarity = self.feature_extractor.compute_similarity(
                    features, global_track.features
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = global_id

            if best_match_id is not None:
                # Update existing global track
                self._update_global_track(best_match_id, camera_id, tracker_id,
                                         features, label)
                matched_globals.add(best_match_id)
            else:
                # Create new global track
                self._create_global_track(camera_id, tracker_id, features, label)

        # Cleanup inactive tracks
        self._cleanup_inactive_tracks()

    def _update_global_track(self, global_id: int, camera_id: str,
                            tracker_id: int, features: np.ndarray, label: str):
        """Update an existing global track with new observation.

        Args:
            global_id: Global track ID
            camera_id: Camera identifier
            tracker_id: Local tracker ID
            features: Extracted features
            label: Object label
        """
        track = self.global_tracks[global_id]
        track.camera_tracks[camera_id] = tracker_id
        track.last_seen_frame = self.current_frame
        track.active = True

        # Update features with exponential moving average
        alpha = 0.3
        track.features = alpha * features + (1 - alpha) * track.features

        # Update label if provided
        if label:
            track.label = label

    def _create_global_track(self, camera_id: str, tracker_id: int,
                            features: np.ndarray, label: str):
        """Create a new global track.

        Args:
            camera_id: Camera identifier
            tracker_id: Local tracker ID
            features: Extracted features
            label: Object label
        """
        global_id = self.next_global_id
        self.next_global_id += 1

        # Generate a unique color for this global track
        np.random.seed(global_id)
        color = tuple(np.random.randint(50, 255, 3).tolist())

        track = GlobalTrack(
            global_id=global_id,
            features=features.copy(),
            camera_tracks={camera_id: tracker_id},
            last_seen_frame=self.current_frame,
            color=color,
            label=label,
            active=True
        )

        self.global_tracks[global_id] = track

    def _cleanup_inactive_tracks(self):
        """Remove tracks that haven't been seen recently."""
        to_remove = []
        for global_id, track in self.global_tracks.items():
            frames_missing = self.current_frame - track.last_seen_frame
            if frames_missing > self.max_frames_missing:
                track.active = False
                to_remove.append(global_id)

        for global_id in to_remove:
            del self.global_tracks[global_id]

    def get_global_tracks(self) -> Dict[int, GlobalTrack]:
        """Get all active global tracks.

        Returns:
            Dictionary of global_id -> GlobalTrack
        """
        return {gid: track for gid, track in self.global_tracks.items()
                if track.active}

    def get_global_id_for_camera_track(self, camera_id: str,
                                       tracker_id: int) -> Optional[int]:
        """Get global ID for a local camera track.

        Args:
            camera_id: Camera identifier
            tracker_id: Local tracker ID

        Returns:
            Global ID if found, None otherwise
        """
        for global_id, track in self.global_tracks.items():
            if track.active and camera_id in track.camera_tracks:
                if track.camera_tracks[camera_id] == tracker_id:
                    return global_id
        return None
