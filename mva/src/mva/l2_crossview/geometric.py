"""GeometricCrossViewLinker — default L2 implementation (M1 W3).

Algorithm: bbox-center distance in normalized image coords + Hungarian
assignment per (timestamp, class_name) bucket. No external calibration
required — this is a baseline placeholder ahead of M3 BEV / M4 ReID.

Pairwise across distinct views: with N views we run N·(N-1)/2 Hungarian
solves and emit one CrossViewLink per matched pair. Multi-view (3+) chained
links are an M4+ refinement.

Confidence is the linear normalization `1 - dist/threshold`, capped at [0,1].
Pairs above the threshold are dropped (PLAN §3.5: empty → []; do not
auto-trigger LLM fallback).

M2.8 — appearance consistency secondary filter (PROBLEMS P1-03 修法草案 #2):
when both observations in a Hungarian-matched pair carry an
`appearance_embedding`, optionally drop the pair if their cosine
similarity falls below `appearance_threshold`. Keeps the geometric
fast-path unchanged when no embeddings are available.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from mva.contracts import CrossViewLink, ViewObservation, make_link_id


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


from mva.l2_crossview._utils import cosine_similarity as _cosine_similarity


class GeometricCrossViewLinker:
    """Default L2 mode: bbox-center Hungarian matching across views.

    Parameters
    ----------
    distance_threshold : float
        Maximum L2 distance (in normalized [0,1] coords) for a pair to be
        considered a candidate. Above this → dropped.
    appearance_threshold : float | None
        🆕 M2.8 — when both members of a Hungarian-matched pair carry an
        `appearance_embedding` (cosine-suitable L2-normalized vector),
        require their cosine similarity ≥ this value or drop the link.
        None disables the filter (default = back-compat with M1/M2.7).
        Reasonable values: 0.5-0.7 for Qwen3-VL-Embedding-8B on same-class
        crops.
    """

    def __init__(
        self,
        distance_threshold: float = 0.3,
        appearance_threshold: Optional[float] = None,
    ) -> None:
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")
        if appearance_threshold is not None and not (
            -1.0 <= appearance_threshold <= 1.0
        ):
            raise ValueError(
                f"appearance_threshold must be in [-1, 1] (cosine range), "
                f"got {appearance_threshold}"
            )
        self.distance_threshold = float(distance_threshold)
        self.appearance_threshold = (
            float(appearance_threshold)
            if appearance_threshold is not None else None
        )

    def link(
        self, observations: Iterable[ViewObservation]
    ) -> list[CrossViewLink]:
        observations = list(observations)
        if not observations:
            return []

        # Bucket by (t, class_name): cross-view links must be co-temporal
        # AND same-class. Small float tolerance on t.
        buckets: dict[tuple[float, str], list[ViewObservation]] = defaultdict(list)
        for o in observations:
            buckets[(round(o.t, 6), o.class_name)].append(o)

        links: list[CrossViewLink] = []
        now = time.time()
        for (t_key, _cls), bucket in buckets.items():
            links.extend(self._link_bucket(bucket, now))

        links.sort(key=lambda link: link.confidence, reverse=True)
        return links

    # ------------------------------------------------------------------

    def _link_bucket(
        self, bucket: list[ViewObservation], created_at: float
    ) -> list[CrossViewLink]:
        # Group by view inside the bucket
        by_view: dict[str, list[ViewObservation]] = defaultdict(list)
        for o in bucket:
            by_view[o.view_id].append(o)
        view_ids = list(by_view.keys())
        if len(view_ids) < 2:
            return []

        out: list[CrossViewLink] = []
        # Pairwise across distinct views
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

        # Cost matrix: L2 distance of normalized bbox centers
        cost = np.zeros((len(obs_a), len(obs_b)), dtype=np.float64)
        for i, a in enumerate(obs_a):
            ax, ay = _bbox_center(a.bbox)
            for j, b in enumerate(obs_b):
                bx, by = _bbox_center(b.bbox)
                cost[i, j] = math.hypot(ax - bx, ay - by)

        row_ind, col_ind = linear_sum_assignment(cost)
        links: list[CrossViewLink] = []
        for i, j in zip(row_ind, col_ind):
            d = float(cost[i, j])
            if d > self.distance_threshold:
                continue
            a, b = obs_a[i], obs_b[j]

            # M2.8 appearance-consistency secondary filter. Only applies
            # when both sides carry an embedding AND a threshold is set;
            # otherwise we degrade to pure geometric (M1 behavior).
            if (
                self.appearance_threshold is not None
                and a.appearance_embedding is not None
                and b.appearance_embedding is not None
            ):
                cos = _cosine_similarity(
                    a.appearance_embedding, b.appearance_embedding,
                )
                if cos < self.appearance_threshold:
                    continue
                # Blend geometric + appearance into confidence so the LLM
                # sees a number that reflects both signals. Equal weight
                # is a reasonable starting point; tune in M4 with eval data.
                geom_conf = 1.0 - d / self.distance_threshold
                confidence = max(0.0, min(1.0, 0.5 * geom_conf + 0.5 * cos))
                created_by = "geometric+appearance"
            else:
                confidence = max(
                    0.0, min(1.0, 1.0 - d / self.distance_threshold),
                )
                created_by = "geometric"

            observations = [
                (a.view_id, a.tracklet_id),
                (b.view_id, b.tracklet_id),
            ]
            links.append(
                CrossViewLink(
                    # Deterministic id so re-running the same scene
                    # overwrites the previous link row instead of stacking
                    # duplicates (post-M3.4 fix).
                    link_id=make_link_id(observations),
                    view_observations=observations,
                    confidence=confidence,
                    created_by=created_by,
                    created_at=created_at,
                )
            )
        return links
