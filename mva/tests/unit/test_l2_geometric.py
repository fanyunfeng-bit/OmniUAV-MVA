"""Unit tests for L2 GeometricCrossViewLinker.

Per PROGRESS.md M1 W3 + PLAN.md §3.2 L2:
  - Algorithm: bbox-center L2 distance over normalized coords → Hungarian
  - Distance threshold gates matches; far-apart bboxes do not link
  - Class mismatch never matches
  - Output is list[CrossViewLink], DESC sorted by confidence
  - Pairwise across distinct views; same-view tracklets do not form a link

These tests bind the algorithm choice. The parametrized contract test in
`tests/contracts/test_cross_view_link.py` already covers the empty-input case
for both Geometric and LLM modes; here we lock the non-empty behavior of the
Geometric mode specifically.
"""
from __future__ import annotations

from mva.contracts import CrossViewLink, ViewObservation
from mva.l2_crossview import GeometricCrossViewLinker


def _obs(view: str, tk: str, t: float, cx: float, cy: float, cls: str = "car"):
    """Build a tiny 0.1×0.1 bbox centered at (cx, cy) for tests."""
    half = 0.05
    return ViewObservation(
        view_id=view,
        tracklet_id=tk,
        t=t,
        bbox=(cx - half, cy - half, cx + half, cy + half),
        class_name=cls,
    )


class TestGeometricLinkerBasics:
    def test_empty_input(self):
        assert GeometricCrossViewLinker().link([]) == []

    def test_single_view_no_link(self):
        # Two tracklets in the SAME view should never form a cross-view link.
        obs = [
            _obs("drone-1", "tk-1", 0.0, 0.3, 0.3),
            _obs("drone-1", "tk-2", 0.0, 0.7, 0.7),
        ]
        assert GeometricCrossViewLinker().link(obs) == []

    def test_two_views_co_located_pair_links(self):
        # Same target seen at the same normalized location across 2 views.
        obs = [
            _obs("drone-1", "tk-1", 0.0, 0.5, 0.5),
            _obs("drone-2", "tk-A", 0.0, 0.5, 0.5),
        ]
        out = GeometricCrossViewLinker().link(obs)
        assert len(out) == 1
        link = out[0]
        assert isinstance(link, CrossViewLink)
        assert link.created_by == "geometric"
        assert link.confidence == 1.0    # zero distance
        view_ids = {v for v, _ in link.view_observations}
        assert view_ids == {"drone-1", "drone-2"}


class TestClassAndDistanceGating:
    def test_class_mismatch_does_not_link(self):
        obs = [
            _obs("drone-1", "tk-1", 0.0, 0.5, 0.5, cls="car"),
            _obs("drone-2", "tk-A", 0.0, 0.5, 0.5, cls="person"),
        ]
        assert GeometricCrossViewLinker().link(obs) == []

    def test_far_apart_above_threshold_does_not_link(self):
        # Default threshold is 0.3 (normalized). Place centers 0.8 apart.
        obs = [
            _obs("drone-1", "tk-1", 0.0, 0.1, 0.1),
            _obs("drone-2", "tk-A", 0.0, 0.9, 0.9),
        ]
        assert GeometricCrossViewLinker().link(obs) == []

    def test_borderline_distance_yields_low_confidence(self):
        # Center distance ≈ 0.15 with threshold 0.3 → confidence ≈ 0.5
        linker = GeometricCrossViewLinker(distance_threshold=0.3)
        obs = [
            _obs("drone-1", "tk-1", 0.0, 0.5, 0.5),
            _obs("drone-2", "tk-A", 0.0, 0.5 + 0.15, 0.5),
        ]
        out = linker.link(obs)
        assert len(out) == 1
        assert 0.4 < out[0].confidence < 0.6

    def test_different_timestamps_do_not_match(self):
        # Same location but different t — L2 only links co-temporal observations.
        obs = [
            _obs("drone-1", "tk-1", 0.0, 0.5, 0.5),
            _obs("drone-2", "tk-A", 10.0, 0.5, 0.5),
        ]
        assert GeometricCrossViewLinker().link(obs) == []


class TestHungarianAssignment:
    def test_optimal_assignment_beats_greedy(self):
        # Two tracklets per view. Greedy nearest-neighbor would mismatch.
        # View-1 t1 is closer to View-2 tA than View-1 t2 is, AND
        # View-1 t1 is also closer to View-2 tB. Greedy would pair t1↔tA
        # then leave t2 to pick the worse remaining option. Hungarian
        # globally minimizes total cost.
        obs = [
            _obs("drone-1", "tk-1", 0.0, 0.30, 0.30),
            _obs("drone-1", "tk-2", 0.0, 0.45, 0.45),
            _obs("drone-2", "tk-A", 0.0, 0.32, 0.32),  # ← clearly tk-1's match
            _obs("drone-2", "tk-B", 0.0, 0.47, 0.47),  # ← clearly tk-2's match
        ]
        out = GeometricCrossViewLinker(distance_threshold=0.4).link(obs)
        # Both pairs should appear as links
        assert len(out) == 2

        pairs = {
            frozenset(o[1] for o in link.view_observations) for link in out
        }
        assert {"tk-1", "tk-A"} in [set(p) for p in pairs]
        assert {"tk-2", "tk-B"} in [set(p) for p in pairs]

    def test_output_sorted_desc_by_confidence(self):
        obs = [
            # Pair 1: near-zero distance → conf ≈ 1.0
            _obs("drone-1", "tk-1", 0.0, 0.5, 0.5),
            _obs("drone-2", "tk-A", 0.0, 0.5, 0.5),
            # Pair 2: ~0.15 distance → conf ≈ 0.5
            _obs("drone-1", "tk-2", 0.0, 0.2, 0.2),
            _obs("drone-2", "tk-B", 0.0, 0.2 + 0.15, 0.2),
        ]
        out = GeometricCrossViewLinker(distance_threshold=0.3).link(obs)
        assert len(out) == 2
        confidences = [link.confidence for link in out]
        assert confidences == sorted(confidences, reverse=True)
        assert confidences[0] > 0.9
        assert 0.4 < confidences[1] < 0.6


class TestPydanticConformance:
    def test_all_outputs_are_cross_view_link_instances(self):
        obs = [
            _obs("drone-1", "tk-1", 0.0, 0.5, 0.5),
            _obs("drone-2", "tk-A", 0.0, 0.5, 0.5),
        ]
        out = GeometricCrossViewLinker().link(obs)
        for link in out:
            assert isinstance(link, CrossViewLink)
            assert 0.0 <= link.confidence <= 1.0
            assert len({v for v, _ in link.view_observations}) >= 2
            assert link.created_by == "geometric"


# --------------------------------------------------------------------------
# M2.8 — appearance-consistency secondary filter
# --------------------------------------------------------------------------


import numpy as np
import pytest


def _obs_with_embedding(
    view: str, tk: str, t: float, cx: float, cy: float,
    embedding: list[float], cls: str = "car",
):
    """A ViewObservation that also carries an L2-normalized embedding."""
    half = 0.05
    return ViewObservation(
        view_id=view, tracklet_id=tk, t=t,
        bbox=(cx - half, cy - half, cx + half, cy + half),
        class_name=cls,
        appearance_embedding=embedding,
    )


def _unit(seed: int, dim: int = 64) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


class TestAppearanceFilter:
    def test_threshold_none_falls_back_to_pure_geometric(self):
        """When `appearance_threshold` is None (default), the linker must
        ignore any embeddings present and behave exactly like M1."""
        emb_a = _unit(0)
        emb_b = _unit(999)        # very different from emb_a
        obs = [
            _obs_with_embedding("drone-1", "tk-1", 0.0, 0.5, 0.5, emb_a),
            _obs_with_embedding("drone-2", "tk-A", 0.0, 0.5, 0.5, emb_b),
        ]
        linker = GeometricCrossViewLinker(distance_threshold=0.3)
        out = linker.link(obs)
        assert len(out) == 1
        assert out[0].created_by == "geometric"

    def test_low_cosine_drops_pair(self):
        """Geometrically perfect match but embeddings disagree → drop."""
        emb_a = _unit(0)
        emb_b = _unit(999)        # near-orthogonal to emb_a
        # Sanity-check the embeddings really are dissimilar
        cos = float(np.dot(np.asarray(emb_a), np.asarray(emb_b)))
        assert cos < 0.5, f"fixture cosine {cos} too high; pick new seeds"

        obs = [
            _obs_with_embedding("drone-1", "tk-1", 0.0, 0.5, 0.5, emb_a),
            _obs_with_embedding("drone-2", "tk-A", 0.0, 0.5, 0.5, emb_b),
        ]
        linker = GeometricCrossViewLinker(
            distance_threshold=0.3, appearance_threshold=0.7,
        )
        assert linker.link(obs) == []

    def test_high_cosine_keeps_pair_and_blends_confidence(self):
        """Geometrically perfect + appearance-similar → kept, with
        created_by='geometric+appearance' and confidence blending both."""
        emb = _unit(0)
        emb_copy = list(emb)                 # cosine = 1.0
        obs = [
            _obs_with_embedding("drone-1", "tk-1", 0.0, 0.5, 0.5, emb),
            _obs_with_embedding("drone-2", "tk-A", 0.0, 0.5, 0.5, emb_copy),
        ]
        linker = GeometricCrossViewLinker(
            distance_threshold=0.3, appearance_threshold=0.7,
        )
        out = linker.link(obs)
        assert len(out) == 1
        assert out[0].created_by == "geometric+appearance"
        # Both geometric (~1.0) and appearance (~1.0) high → blended ~1.0
        assert out[0].confidence > 0.95

    def test_missing_embedding_on_one_side_uses_geometric_only(self):
        """If one side lacks an embedding, the filter is skipped — keep
        the pair using pure geometric confidence (avoid silent drops on
        partial coverage)."""
        emb = _unit(0)
        obs = [
            _obs_with_embedding("drone-1", "tk-1", 0.0, 0.5, 0.5, emb),
            # Side B has no embedding
            ViewObservation(
                view_id="drone-2", tracklet_id="tk-A", t=0.0,
                bbox=(0.45, 0.45, 0.55, 0.55), class_name="car",
            ),
        ]
        linker = GeometricCrossViewLinker(
            distance_threshold=0.3, appearance_threshold=0.99,
        )
        out = linker.link(obs)
        assert len(out) == 1
        assert out[0].created_by == "geometric"

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            GeometricCrossViewLinker(appearance_threshold=1.5)
        with pytest.raises(ValueError):
            GeometricCrossViewLinker(appearance_threshold=-2.0)
