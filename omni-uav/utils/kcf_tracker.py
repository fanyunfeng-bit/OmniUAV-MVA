"""Kernelized Correlation Filter (KCF) tracker implementation.

Based on: High-Speed Tracking with Kernelized Correlation Filters
Henriques et al., TPAMI 2015
"""

import numpy as np
import cv2
import math
from .tracker_utils import clip_bbox, get_cosine_window, fft2, ifft2


class KCFTracker:
    """KCF object tracker using correlation filters."""

    def __init__(self, padding=2.5, features='color', kernel='linear'):
        """Initialize KCF tracker.

        Args:
            padding: Tracked region size multiplier (default: 2.5x target size)
            features: Feature type ('gray' or 'color')
            kernel: Kernel type ('linear' or 'gaussian')
        """
        self.padding = padding
        self.lambda_r = 1e-4  # regularization
        self.features = features
        self.kernel = kernel
        self.output_sigma_factor = 0.01

        if self.features == 'gray' or self.features == 'color':
            self.learning_rate = 0.15
            self.feature_bandwidth = 0.2
            self.cell_size = 1

        self.initialized = False

    def init(self, frame, bbox):
        """Initialize tracker with first frame and bounding box.

        Args:
            frame: First frame (BGR format)
            bbox: Bounding box [x, y, w, h]
        """
        self.w, self.h = bbox[2], bbox[3]
        self.x0, self.y0 = bbox[0], bbox[1]

        # Calculate padded region size
        self.padded_w = math.floor(self.w * (1 + self.padding)) // self.cell_size
        self.padded_h = math.floor(self.h * (1 + self.padding)) // self.cell_size

        # Clip to frame boundaries
        self.x1_clip, self.y1_clip, self.x2_clip, self.y2_clip = clip_bbox(
            self.x0 - self.w * self.padding // 2,
            self.y0 - self.h * self.padding // 2,
            math.floor(self.w * (1 + self.padding)),
            math.floor(self.h * (1 + self.padding)),
            frame.shape[:2]
        )

        f = frame[self.y1_clip:self.y2_clip, self.x1_clip:self.x2_clip, :]

        # Create cosine window
        self.cosine_window = get_cosine_window((self.padded_w, self.padded_h))
        self.cosine_window = np.expand_dims(self.cosine_window, axis=2)

        self.spatial_bandwidth = self.output_sigma_factor * np.sqrt(
            self.padded_w * self.padded_h
        )

        # Create Gaussian regression target
        self.yf = fft2(self._get_cyclic_gaussian_map())

        # Extract features and train
        self.xf = self._preprocess_frame(f)
        self.alphaf = self._train(self.xf, self.yf)

        self.initialized = True

    def update(self, frame):
        """Track object in new frame.

        Args:
            frame: New frame (BGR format)

        Returns:
            Bounding box [x1, y1, x2, y2] or None if tracking failed
        """
        if not self.initialized:
            return None

        f = frame[self.y1_clip:self.y2_clip, self.x1_clip:self.x2_clip, :]

        z = self._preprocess_frame(f)
        responses = self._detect(self.alphaf, self.xf, z)

        # Find maximum response
        max_yx = np.where(responses == np.max(responses))
        max_yx = (max_yx[0][0], max_yx[1][0])

        # Calculate displacement
        if max_yx[0] + 1 > self.padded_h / 2:
            dy = max_yx[0] - self.padded_h
        else:
            dy = max_yx[0]
        if max_yx[1] + 1 > self.padded_w / 2:
            dx = max_yx[1] - self.padded_w
        else:
            dx = max_yx[1]

        dy, dx = dy * self.cell_size, dx * self.cell_size
        self.x0 += dx
        self.y0 += dy

        # Update clipping region
        self.x1_clip, self.y1_clip, self.x2_clip, self.y2_clip = clip_bbox(
            self.x0 - self.w * self.padding // 2,
            self.y0 - self.h * self.padding // 2,
            math.floor(self.w * (1 + self.padding)),
            math.floor(self.h * (1 + self.padding)),
            frame.shape[:2]
        )

        f = frame[self.y1_clip:self.y2_clip, self.x1_clip:self.x2_clip, :]

        # Update model
        new_x = self._preprocess_frame(f)
        new_alphaf = self._train(new_x, self.yf)

        self.alphaf = self.learning_rate * new_alphaf + (1 - self.learning_rate) * self.alphaf
        self.xf = self.learning_rate * new_x + (1 - self.learning_rate) * self.xf

        return int(self.x0), int(self.y0), int(self.x0 + self.w), int(self.y0 + self.h)

    def _kernel_correlation(self, x1f, x2f):
        """Compute kernel correlation."""
        if self.kernel == 'gaussian':
            N = x1f.shape[0] * x1f.shape[1]

            xx = np.dot(x1f.flatten().conj().T, x1f.flatten()) / N
            yy = np.dot(x2f.flatten().conj().T, x2f.flatten()) / N
            cf = x1f * np.conj(x2f)
            c = np.sum(np.real(ifft2(cf)), axis=2)

            kf = fft2(
                np.exp(-1 / self.feature_bandwidth ** 2 * np.abs(xx + yy - 2 * c) / np.size(x1f))
            )

        elif self.kernel == 'linear':
            kf = np.sum(x1f * np.conj(x2f), axis=2) / np.size(x1f)

        return kf

    def _extract_features(self, x):
        """Extract features from image patch."""
        if self.features == 'gray' or self.features == 'color':
            x = x / 255
            x = x - np.mean(x)

        return x

    def _train(self, xf, yf):
        """Train correlation filter."""
        kf = self._kernel_correlation(xf, xf)
        alphaf = yf / (kf + self.lambda_r)
        return alphaf

    def _detect(self, alphaf, xf, zf):
        """Detect object in new frame."""
        kf = self._kernel_correlation(zf, xf)
        responses = np.real(ifft2(alphaf * kf))
        return responses

    def _preprocess_frame(self, frame):
        """Preprocess frame for tracking."""
        x = cv2.resize(frame, (self.padded_w * self.cell_size, self.padded_h * self.cell_size))
        x = self._extract_features(x)
        return fft2(x * self.cosine_window)

    def _get_cyclic_gaussian_map(self):
        """Create cyclic Gaussian response map."""
        xx, yy = np.meshgrid(
            np.arange(self.padded_w) - self.padded_w // 2,
            np.arange(self.padded_h) - self.padded_h // 2
        )

        dist = (xx ** 2 + yy ** 2) / (self.spatial_bandwidth ** 2)
        response = np.exp(-0.5 * dist)

        # Roll to put maximum at top-left corner
        response = np.roll(response, -math.floor(self.padded_w / 2), axis=1)
        response = np.roll(response, -math.floor(self.padded_h / 2), axis=0)

        return response
