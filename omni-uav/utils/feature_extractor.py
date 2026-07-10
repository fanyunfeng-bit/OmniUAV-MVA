"""
Feature extraction utilities for object re-identification.
Supports color histogram and deep feature extraction.
"""
import numpy as np
import cv2
from typing import Tuple


class FeatureExtractor:
    """Extract appearance features from object bounding boxes."""

    def __init__(self, feature_type: str = "color_histogram"):
        """Initialize feature extractor.

        Args:
            feature_type: Type of features to extract ("color_histogram" or "deep_features")
        """
        self.feature_type = feature_type

    def extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract features from a bounding box region.

        Args:
            frame: Input frame (BGR format)
            bbox: Bounding box as (x1, y1, x2, y2)

        Returns:
            Feature vector as numpy array
        """
        x1, y1, x2, y2 = bbox

        # Clip to frame boundaries
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            # Invalid bbox, return zero features
            if self.feature_type == "color_histogram":
                return np.zeros(256, dtype=np.float32)
            else:
                return np.zeros(512, dtype=np.float32)

        # Extract region
        region = frame[y1:y2, x1:x2]

        if self.feature_type == "color_histogram":
            return self._extract_color_histogram(region)
        elif self.feature_type == "deep_features":
            return self._extract_deep_features(region)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    def _extract_color_histogram(self, region: np.ndarray) -> np.ndarray:
        """Extract normalized color histogram features.

        Args:
            region: Image region (BGR format)

        Returns:
            Normalized histogram feature vector
        """
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Compute histogram for H and S channels
        h_hist = cv2.calcHist([hsv], [0], None, [128], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [128], [0, 256])

        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()

        # Concatenate histograms
        features = np.concatenate([h_hist, s_hist])

        return features.astype(np.float32)

    def _extract_deep_features(self, region: np.ndarray) -> np.ndarray:
        """Extract deep learning features (placeholder for future implementation).

        Args:
            region: Image region (BGR format)

        Returns:
            Feature vector
        """
        # Placeholder: For now, use color histogram as fallback
        # In future, could use a pretrained CNN (ResNet, MobileNet, etc.)
        return self._extract_color_histogram(region)

    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute similarity between two feature vectors.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if self.feature_type == "color_histogram":
            # Use histogram correlation
            return float(cv2.compareHist(
                features1.reshape(-1, 1),
                features2.reshape(-1, 1),
                cv2.HISTCMP_CORREL
            ))
        else:
            # Use cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
