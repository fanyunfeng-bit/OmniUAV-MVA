"""Unit tests for AppearanceCrossViewLinker (M3.0).

This linker is the L2 mode for non-synchronized multi-video sources
(MVU-Eval-style). Unlike `GeometricCrossViewLinker`, it does NOT bucket
by time and does NOT use bbox geometric distance — only cosine
similarity of `appearance_embedding` within (class_name, segment_idx)
buckets.

Tests cover:
  - Empty / single-view inputs return []
  - High-cosine pairs cross views become links with `created_by="appearance"`
  - Low-cosine pairs (< threshold) are dropped
  - Different class_names never link
  - Different segment_idx never link (same class, same view OK; same
    class, same seg, different view → candidate)
  - Hungarian assigns 1-to-1 (no duplicate use of same observation)
  - Confidence equals the cosine similarity (clipped to [0,1])
  - Output is sorted DESC by confidence
  - Threshold validation
"""
from __future__ import annotations

import numpy as np
import pytest

from mva.contracts import CrossViewLink, ViewObservation
from mva.l2_crossview import AppearanceCrossViewLinker


def _unit(seed: int, dim: int = 64) -> list[float]:
    """Reproducible L2-normalized vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


def _obs(view: str, tk: str, cls: str, embedding: list[float],
         seg_idx: int = 0) -> ViewObservation:
    return ViewObservation(
        view_id=view, tracklet_id=tk, t=0.0,
        bbox=(0.0, 0.0, 0.0, 0.0),     # ignored in appearance mode
        class_name=cls,
        appearance_embedding=embedding,
        segment_idx=seg_idx,
    )


# ----------------------------------------------------------------------
# Basic linker behavior
# ----------------------------------------------------------------------


def test_empty_returns_empty():
    assert AppearanceCrossViewLinker().link([]) == []


def test_no_embedding_filtered_out():
    """Observations without an embedding are silently skipped — required
    by appearance-only matching."""
    obs = [
        ViewObservation("v1", "tk1", 0.0, (0,0,0,0), "person",
                        appearance_embedding=None, segment_idx=0),
        ViewObservation("v2", "tk2", 0.0, (0,0,0,0), "person",
                        appearance_embedding=None, segment_idx=0),
    ]
    assert AppearanceCrossViewLinker().link(obs) == []


def test_single_view_returns_empty():
    """A cross-view link requires ≥ 2 distinct view_ids."""
    obs = [
        _obs("v1", "tk1", "person", _unit(0)),
        _obs("v1", "tk2", "person", _unit(1)),
    ]
    assert AppearanceCrossViewLinker().link(obs) == []


# ----------------------------------------------------------------------
# Threshold + class + segment bucketing
# ----------------------------------------------------------------------


def test_high_cosine_creates_link():
    emb = _unit(0)
    obs = [
        _obs("v1", "tk1", "person", emb),
        _obs("v2", "tk2", "person", list(emb)),  # identical → cos = 1.0
    ]
    out = AppearanceCrossViewLinker(appearance_threshold=0.7).link(obs)
    assert len(out) == 1
    assert out[0].created_by == "appearance"
    assert out[0].confidence > 0.99
    views = {v for v, _ in out[0].view_observations}
    assert views == {"v1", "v2"}


def test_low_cosine_dropped():
    obs = [
        _obs("v1", "tk1", "person", _unit(0)),
        _obs("v2", "tk2", "person", _unit(999)),  # near-orthogonal
    ]
    # Sanity-check seeds are dissimilar
    cos = float(np.dot(np.asarray(obs[0].appearance_embedding),
                       np.asarray(obs[1].appearance_embedding)))
    assert cos < 0.7

    assert AppearanceCrossViewLinker(appearance_threshold=0.7).link(obs) == []


def test_class_mismatch_never_links():
    """Even with identical embeddings, different class_name → no link."""
    emb = _unit(0)
    obs = [
        _obs("v1", "tk1", "person", emb, seg_idx=0),
        _obs("v2", "tk2", "car",    list(emb), seg_idx=0),
    ]
    assert AppearanceCrossViewLinker(appearance_threshold=0.7).link(obs) == []


def test_segment_idx_mismatch_can_now_link():
    """M3.0 update (2026-05-22): segment_idx is no longer in the bucket
    key. Same class + matching embeddings across different segments now
    produces a link — required for MVU-Eval's OR / Counting tasks where
    the same object appears at different times in each video."""
    emb = _unit(0)
    obs = [
        _obs("v1", "tk1", "person", emb, seg_idx=0),
        _obs("v2", "tk2", "person", list(emb), seg_idx=1),
    ]
    out = AppearanceCrossViewLinker(appearance_threshold=0.7).link(obs)
    assert len(out) == 1
    assert out[0].created_by == "appearance"


def test_class_separates_buckets_even_after_seg_drop():
    """Bucket key is class_name only now — but different class still
    must NOT link, even with identical embeddings."""
    emb = _unit(0)
    obs = [
        _obs("v1", "tk1", "person", emb, seg_idx=0),
        _obs("v2", "tk2", "car",    list(emb), seg_idx=0),
    ]
    assert AppearanceCrossViewLinker(appearance_threshold=0.7).link(obs) == []


def test_same_view_self_match_still_blocked():
    """`_link_bucket` requires ≥ 2 distinct view_ids inside a bucket.
    Same view + same class + multiple segments must NOT produce a self-
    match even with the wider class-only bucket."""
    emb_a = _unit(0)
    emb_b = _unit(1)
    obs = [
        _obs("v1", "tk-seg0", "person", emb_a, seg_idx=0),
        _obs("v1", "tk-seg1", "person", emb_b, seg_idx=1),
        _obs("v1", "tk-seg2", "person", list(emb_a), seg_idx=2),  # cos=1 vs tk-seg0
    ]
    assert AppearanceCrossViewLinker(appearance_threshold=0.7).link(obs) == []


# ----------------------------------------------------------------------
# Hungarian semantics
# ----------------------------------------------------------------------


def test_hungarian_assigns_one_to_one():
    """Two views, two persons each, identical pairs (v1 tk1 ↔ v2 tk1,
    v1 tk2 ↔ v2 tk2). Hungarian must pair correctly, not duplicate one
    observation."""
    emb_a = _unit(0)
    emb_b = _unit(100)
    obs = [
        _obs("v1", "v1-tk1", "person", emb_a),
        _obs("v1", "v1-tk2", "person", emb_b),
        _obs("v2", "v2-tk1", "person", list(emb_a)),  # matches v1-tk1
        _obs("v2", "v2-tk2", "person", list(emb_b)),  # matches v1-tk2
    ]
    out = AppearanceCrossViewLinker(appearance_threshold=0.7).link(obs)
    assert len(out) == 2
    # Each tracklet appears exactly once
    tracklets_used = [tk for link in out for _, tk in link.view_observations]
    assert sorted(tracklets_used) == ["v1-tk1", "v1-tk2", "v2-tk1", "v2-tk2"]


def test_output_sorted_desc_by_confidence():
    emb_a = _unit(0)
    emb_b = _unit(100)
    obs = [
        _obs("v1", "v1-tk-near", "person", emb_a),
        _obs("v2", "v2-tk-near", "person", list(emb_a)),         # cos = 1.0
        _obs("v1", "v1-tk-mid",  "person", emb_b),
        # build a vector that has known cos with emb_b
        _obs("v2", "v2-tk-mid",  "person", _mix(emb_b, _unit(200), 0.8)),
    ]
    out = AppearanceCrossViewLinker(appearance_threshold=0.5).link(obs)
    assert len(out) == 2
    confidences = [link.confidence for link in out]
    assert confidences == sorted(confidences, reverse=True)


def _mix(a: list[float], b: list[float], alpha: float) -> list[float]:
    """Convex combo, re-normalized — produces a controllable cosine."""
    va = np.asarray(a)
    vb = np.asarray(b)
    mixed = alpha * va + (1 - alpha) * vb
    mixed /= np.linalg.norm(mixed) + 1e-9
    return mixed.tolist()


# ----------------------------------------------------------------------
# Pydantic conformance
# ----------------------------------------------------------------------


def test_all_outputs_are_cross_view_link_instances():
    emb = _unit(0)
    obs = [
        _obs("v1", "tk1", "person", emb),
        _obs("v2", "tk2", "person", list(emb)),
    ]
    for link in AppearanceCrossViewLinker(appearance_threshold=0.5).link(obs):
        assert isinstance(link, CrossViewLink)
        assert 0.0 <= link.confidence <= 1.0
        assert len({v for v, _ in link.view_observations}) >= 2
        assert link.created_by == "appearance"


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        AppearanceCrossViewLinker(appearance_threshold=1.5)
    with pytest.raises(ValueError):
        AppearanceCrossViewLinker(appearance_threshold=-2.0)


# ----------------------------------------------------------------------
# Contract: MVU-Eval-style use case (video_editing variants)
# ----------------------------------------------------------------------


def test_three_views_pairwise_links():
    """Three videos of the same scene — appearance linker should produce
    three pairwise links per matching identity (v1↔v2, v1↔v3, v2↔v3)."""
    emb = _unit(42)
    obs = [
        _obs("video_a.mp4", "tk_a", "person", emb),
        _obs("video_b.mp4", "tk_b", "person", list(emb)),
        _obs("video_c.mp4", "tk_c", "person", list(emb)),
    ]
    out = AppearanceCrossViewLinker(appearance_threshold=0.7).link(obs)
    assert len(out) == 3
    pairs = {tuple(sorted(v for v, _ in link.view_observations))
             for link in out}
    assert pairs == {
        ("video_a.mp4", "video_b.mp4"),
        ("video_a.mp4", "video_c.mp4"),
        ("video_b.mp4", "video_c.mp4"),
    }
    for link in out:
        assert link.created_by == "appearance"
