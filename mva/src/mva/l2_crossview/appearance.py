"""AppearanceCrossViewLinker — L2 linker for non-synchronized multi-video
sources (M3.0, the MVU-Eval-style use case).

Unlike `GeometricCrossViewLinker` (which buckets by `(t, class_name)`
and matches on bbox-center geometric distance), this linker buckets by
`(class_name, segment_idx)` and matches on **cosine distance of
appearance embeddings**. No bbox geometric comparability is required —
which is the whole point, since two unrelated videos have no shared
coordinate frame.

When to use:
  - MVU-Eval (mode="appearance"): "video_editing" QAs are edited
    variants of the same source clip; "Ordering" QAs are temporal
    fragments of the same event; these share objects/scenes by
    appearance even though wall-clock times don't align.
  - Future non-synchronized multi-camera setups where each camera has
    its own time origin.

When NOT to use:
  - MATRIX-style synchronized capture — `GeometricCrossViewLinker` with
    `appearance_threshold` is strictly more informative (geometry +
    appearance dual signal).

Caveats baked into the design:
  - **False positives are common** (two distinct people who look alike
    across unrelated clips) — the appearance threshold is the only line
    of defense. Default 0.75 (vs 0.6 for MATRIX) is conservative.
  - `created_by="appearance"` flags this link as "looks like the same
    object, but cannot guarantee same physical identity" — L6 tool
    descriptions tell the LLM to phrase answers accordingly.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from mva.contracts import CrossViewLink, ViewObservation, make_link_id


_DEFAULT_APPEARANCE_THRESHOLD = 0.75


from mva.l2_crossview._utils import cosine_similarity as _cosine_similarity


class AppearanceCrossViewLinker:
    """Pure-cosine cross-view linker for non-synchronized sources.

    Parameters
    ----------
    appearance_threshold : float
        Minimum cosine similarity for a Hungarian-matched pair to
        survive. Default 0.75 — more conservative than MATRIX's 0.6 in
        the synchronized linker, because pure-appearance has no
        geometric cross-check to suppress false positives.
    """

    def __init__(
        self,
        appearance_threshold: float = _DEFAULT_APPEARANCE_THRESHOLD,
    ) -> None:
        if not (-1.0 <= appearance_threshold <= 1.0):
            raise ValueError(
                f"appearance_threshold must be in [-1, 1] (cosine range), "
                f"got {appearance_threshold}"
            )
        self.appearance_threshold = float(appearance_threshold)

    def link(
        self, observations: Iterable[ViewObservation],
    ) -> list[CrossViewLink]:
        observations = [
            o for o in observations
            if o.appearance_embedding is not None
        ]
        if not observations:
            return []

        # Bucket by class_name only. The earlier (class, segment_idx)
        # bucketing was a premature optimization that blocked the
        # legitimate "same object appears at different times in each
        # video" case — the typical pattern of MVU-Eval's OR / Counting
        # tasks (non-overlapping scenes, shared object identity).
        #
        # The by-view grouping inside `_link_bucket` still prevents
        # same-view self-matches. Cost is O((Σ obs/class)²) cosine per
        # class — for typical MVU-Eval QA (4-6 videos × 1-3 segments ×
        # a few detections per frame), well under a millisecond.
        buckets: dict[str, list[ViewObservation]] = defaultdict(list)
        for o in observations:
            buckets[o.class_name].append(o)

        links: list[CrossViewLink] = []
        now = time.time()
        for _cls, bucket in buckets.items():
            links.extend(self._link_bucket(bucket, now))

        links.sort(key=lambda link: link.confidence, reverse=True)
        return links

    # ------------------------------------------------------------------

    def _link_bucket(
        self, bucket: list[ViewObservation], created_at: float,
    ) -> list[CrossViewLink]:
        by_view: dict[str, list[ViewObservation]] = defaultdict(list)
        for o in bucket:
            by_view[o.view_id].append(o)
        view_ids = list(by_view.keys())
        if len(view_ids) < 2:
            return []

        out: list[CrossViewLink] = []
        for i in range(len(view_ids)):
            for j in range(i + 1, len(view_ids)):
                va, vb = view_ids[i], view_ids[j]
                out.extend(
                    self._pairwise_link(by_view[va], by_view[vb], created_at)
                )
        return out

    def _pairwise_link(
        self,
        obs_a: list[ViewObservation],
        obs_b: list[ViewObservation],
        created_at: float,
    ) -> list[CrossViewLink]:
        if not obs_a or not obs_b:
            return []

        # Cost matrix: 1 - cosine similarity (Hungarian minimizes cost).
        cost = np.ones((len(obs_a), len(obs_b)), dtype=np.float64)
        for i, a in enumerate(obs_a):
            for j, b in enumerate(obs_b):
                sim = _cosine_similarity(
                    a.appearance_embedding, b.appearance_embedding,
                )
                cost[i, j] = 1.0 - sim

        row_ind, col_ind = linear_sum_assignment(cost)
        links: list[CrossViewLink] = []
        for i, j in zip(row_ind, col_ind):
            sim = 1.0 - float(cost[i, j])
            if sim < self.appearance_threshold:
                continue
            a, b = obs_a[i], obs_b[j]
            # Confidence == cosine similarity itself, clipped to [0,1]
            # (some embeddings can land slightly out-of-range with FP rounding).
            confidence = max(0.0, min(1.0, sim))
            observations = [
                (a.view_id, a.tracklet_id),
                (b.view_id, b.tracklet_id),
            ]
            links.append(
                CrossViewLink(
                    # Deterministic id so reruns are idempotent
                    # (post-M3.4 follow-up).
                    link_id=make_link_id(observations),
                    view_observations=observations,
                    confidence=confidence,
                    created_by="appearance",
                    created_at=created_at,
                )
            )
        return links
