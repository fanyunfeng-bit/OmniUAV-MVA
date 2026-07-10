"""Utility functions for object tracking."""

import numpy as np
import cv2


def get_cosine_window(sz, method='hanning'):
    """Create cosine window for tracking.

    The image is multiplied by a cosine window which gradually reduces
    the pixel values near the edge to zero. This puts more emphasis
    near the center of the target.
    """
    w, h = sz

    if method == 'blackman':
        w_win = np.blackman(w)
        h_win = np.blackman(h)
    elif method == 'hanning':
        w_win = np.hanning(w)
        h_win = np.hanning(h)
    elif method == 'hamming':
        w_win = np.hamming(w)
        h_win = np.hamming(h)
    else:
        raise ValueError(f"Unknown window method: {method}")

    w_msk, h_msk = np.meshgrid(w_win, h_win)
    win = w_msk * h_msk

    return win


def clip_bbox(x, y, w, h, sz):
    """Clip bounding box to image boundaries."""
    x1 = int(np.clip(x, 0, sz[1]))
    x2 = int(np.clip(x + w, 0, sz[1]))
    y1 = int(np.clip(y, 0, sz[0]))
    y2 = int(np.clip(y + h, 0, sz[0]))

    return x1, y1, x2, y2


def fft2(x):
    """2D Fast Fourier Transform."""
    return np.fft.fft(np.fft.fft(x, axis=1), axis=0).astype(np.complex64)


def ifft2(x):
    """2D Inverse Fast Fourier Transform."""
    return np.fft.ifft(np.fft.ifft(x, axis=1), axis=0).astype(np.complex64)
