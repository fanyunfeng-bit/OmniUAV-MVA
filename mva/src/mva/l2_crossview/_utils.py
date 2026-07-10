"""Shared helpers for L2 cross-view linkers."""
from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for L2-normalized vectors == dot product.

    Caller (the embedder) is expected to L2-normalize upstream.
    Returns 0.0 if either side is empty or shapes mismatch.
    """
    if not a or not b:
        return 0.0
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if va.shape != vb.shape:
        return 0.0
    return float(np.dot(va, vb))
