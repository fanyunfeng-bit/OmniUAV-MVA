"""Unit tests for `_llm_fallback_upgrade_links` (M4.3).

The function lives in `mva.cli.ingest` (private to keep its API
surface tight); tests reach in via name. Each test constructs:
- A small list of pre-algorithmic CrossViewLinks with controlled
  confidences
- The matching ViewObservations (with roi_uri pointing at tmp JPEGs so
  the default ROI loader works without VLM downloads)
- A scripted LLM client that records every prompt + returns canned JSON

Covers the four behaviors PLAN §6.2 M4.3 requires:
1. High-conf links pass through (LLM never invoked)
2. Low-conf + LLM confirms → link replaced (`created_by="llm"`)
3. Low-conf + LLM rejects → link dropped
4. Per-view throttle: only `max_per_view` LLM calls per view total
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from mva.cli.ingest import _llm_fallback_upgrade_links
from mva.contracts import (
    CrossViewLink,
    ViewObservation,
    make_link_id,
)


# ----------------------------------------------------------------------
# Test scaffolding
# ----------------------------------------------------------------------


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0
        self.prompts_seen = []
        self.calls = 0

    def complete(self, prompt, images=None, **_kw):
        self.calls += 1
        self.prompts_seen.append(prompt)
        if self._i >= len(self._responses):
            return self._responses[-1]
        out = self._responses[self._i]
        self._i += 1
        return out


def _make_roi(tmp_path, name):
    path = tmp_path / f"{name}.jpg"
    cv2.imwrite(str(path), np.full((16, 16, 3), 120, dtype=np.uint8))
    return str(path)


def _obs(tmp_path, view, tk, name=None):
    name = name or f"{view}-{tk}"
    return ViewObservation(
        view_id=view, tracklet_id=tk, t=0.0,
        bbox=(0.1, 0.1, 0.5, 0.5),
        class_name="person",
        roi_uri=_make_roi(tmp_path, name),
    )


def _link(v_a, tk_a, v_b, tk_b, conf, created_by="geometric"):
    observations = [(v_a, tk_a), (v_b, tk_b)]
    return CrossViewLink(
        link_id=make_link_id(observations),
        view_observations=observations,
        confidence=conf,
        created_by=created_by,
        created_at=0.0,
    )


# ----------------------------------------------------------------------
# 1. High-confidence links pass through, LLM never invoked
# ----------------------------------------------------------------------


def test_high_confidence_links_skip_llm_entirely(tmp_path):
    """Any link with confidence ≥ threshold must NOT trigger the LLM."""
    obs = [_obs(tmp_path, "v1", "a"), _obs(tmp_path, "v2", "b")]
    links = [_link("v1", "a", "v2", "b", conf=0.8)]
    llm = ScriptedLLM(["if you see this the test is broken"])

    out = _llm_fallback_upgrade_links(
        links, obs, llm, threshold=0.5, max_per_view=1,
    )
    assert llm.calls == 0
    assert len(out) == 1
    assert out[0].created_by == "geometric"   # untouched
    assert out[0].confidence == pytest.approx(0.8)


# ----------------------------------------------------------------------
# 2. Low-confidence + LLM confirms → upgrade
# ----------------------------------------------------------------------


def test_low_confidence_link_upgraded_when_llm_confirms(tmp_path):
    """LLM says same_object=true with confidence ≥ threshold → original
    geometric link replaced with the LLM version (same link_id, new
    created_by + confidence)."""
    obs = [_obs(tmp_path, "v1", "a"), _obs(tmp_path, "v2", "b")]
    links = [_link("v1", "a", "v2", "b", conf=0.3)]
    llm = ScriptedLLM([
        json.dumps({"same_object": True, "confidence": 0.85})
    ])

    out = _llm_fallback_upgrade_links(
        links, obs, llm, threshold=0.5, max_per_view=1,
    )
    assert llm.calls == 1
    assert len(out) == 1
    assert out[0].created_by == "llm"
    assert out[0].confidence == pytest.approx(0.85)
    # link_id identical → DuckDB upsert just updates the row
    assert out[0].link_id == links[0].link_id


# ----------------------------------------------------------------------
# 3. Low-confidence + LLM rejects → link dropped
# ----------------------------------------------------------------------


def test_low_confidence_link_dropped_when_llm_says_not_same(tmp_path):
    """LLM judges 'not same object' → drop the algorithmic link entirely.
    The algorithmic linker was uncertain; the VLM second opinion says
    these aren't the same person → trust the VLM."""
    obs = [_obs(tmp_path, "v1", "a"), _obs(tmp_path, "v2", "b")]
    links = [_link("v1", "a", "v2", "b", conf=0.3)]
    llm = ScriptedLLM([
        json.dumps({"same_object": False, "confidence": 0.95})
    ])

    out = _llm_fallback_upgrade_links(
        links, obs, llm, threshold=0.5, max_per_view=1,
    )
    assert llm.calls == 1
    assert out == []


def test_low_confidence_link_dropped_when_llm_below_threshold(tmp_path):
    """LLM says same_object=true but its own confidence is below the
    fallback threshold → drop the link. The whole point of the
    threshold is "don't accept fuzzy answers"."""
    obs = [_obs(tmp_path, "v1", "a"), _obs(tmp_path, "v2", "b")]
    links = [_link("v1", "a", "v2", "b", conf=0.3)]
    llm = ScriptedLLM([
        json.dumps({"same_object": True, "confidence": 0.2})
    ])

    out = _llm_fallback_upgrade_links(
        links, obs, llm, threshold=0.5, max_per_view=1,
    )
    assert llm.calls == 1
    assert out == []


# ----------------------------------------------------------------------
# 4. Throttle: at most `max_per_view` LLM calls per view across the run
# ----------------------------------------------------------------------


def test_throttle_caps_llm_invocations_per_view(tmp_path):
    """3 low-conf links all touching v1; with max_per_view=1 only the
    FIRST gets LLM judgment. The rest pass through unchanged (still
    low-conf, still created_by='geometric')."""
    obs = [
        _obs(tmp_path, "v1", "a"), _obs(tmp_path, "v2", "x"),
        _obs(tmp_path, "v1", "b"), _obs(tmp_path, "v2", "y"),
        _obs(tmp_path, "v1", "c"), _obs(tmp_path, "v2", "z"),
    ]
    links = [
        _link("v1", "a", "v2", "x", conf=0.3),
        _link("v1", "b", "v2", "y", conf=0.35),
        _link("v1", "c", "v2", "z", conf=0.4),
    ]
    # Only first response matters — others should never get asked
    llm = ScriptedLLM([
        json.dumps({"same_object": True, "confidence": 0.9}),
        # If the throttle is broken these would be used:
        json.dumps({"same_object": True, "confidence": 0.95}),
        json.dumps({"same_object": True, "confidence": 0.99}),
    ])

    out = _llm_fallback_upgrade_links(
        links, obs, llm, threshold=0.5, max_per_view=1,
    )
    assert llm.calls == 1, (
        f"throttle broken: expected 1 LLM call (max_per_view=1 across v1), "
        f"got {llm.calls}"
    )
    # Output: one upgraded link + two original low-conf links
    created_by_set = {link.created_by for link in out}
    assert "llm" in created_by_set
    assert "geometric" in created_by_set
    llm_links = [link for link in out if link.created_by == "llm"]
    assert len(llm_links) == 1
    geometric_links = [link for link in out if link.created_by == "geometric"]
    assert len(geometric_links) == 2


def test_throttle_independent_per_view(tmp_path):
    """3 low-conf links each on DISJOINT view pairs: 1 call per view is
    allowed for each, so all 3 should get LLM second opinions."""
    obs = [
        _obs(tmp_path, "v1", "a"), _obs(tmp_path, "v2", "x"),
        _obs(tmp_path, "v3", "b"), _obs(tmp_path, "v4", "y"),
        _obs(tmp_path, "v5", "c"), _obs(tmp_path, "v6", "z"),
    ]
    links = [
        _link("v1", "a", "v2", "x", conf=0.3),
        _link("v3", "b", "v4", "y", conf=0.35),
        _link("v5", "c", "v6", "z", conf=0.4),
    ]
    llm = ScriptedLLM([
        json.dumps({"same_object": True, "confidence": 0.9}),
        json.dumps({"same_object": True, "confidence": 0.85}),
        json.dumps({"same_object": True, "confidence": 0.95}),
    ])

    out = _llm_fallback_upgrade_links(
        links, obs, llm, threshold=0.5, max_per_view=1,
    )
    assert llm.calls == 3
    assert all(link.created_by == "llm" for link in out)
    assert len(out) == 3


# ----------------------------------------------------------------------
# 5. Output ordering invariant — DESC by confidence
# ----------------------------------------------------------------------


def test_upgraded_output_is_sorted_desc_by_confidence(tmp_path):
    """LLM upgrades can shift confidences; output must remain sorted
    DESC so the existing DuckDB query_cross_view_links contract is
    preserved."""
    obs = [
        _obs(tmp_path, "v1", "a"), _obs(tmp_path, "v2", "x"),
        _obs(tmp_path, "v3", "b"), _obs(tmp_path, "v4", "y"),
    ]
    links = [
        _link("v1", "a", "v2", "x", conf=0.4),    # low → will upgrade to 0.95
        _link("v3", "b", "v4", "y", conf=0.7),    # high, passes through
    ]
    llm = ScriptedLLM([
        json.dumps({"same_object": True, "confidence": 0.95})
    ])
    out = _llm_fallback_upgrade_links(
        links, obs, llm, threshold=0.5, max_per_view=1,
    )
    # Expected: 2 links, sorted [0.95 llm, 0.7 geometric]
    assert [link.confidence for link in out] == sorted(
        [link.confidence for link in out], reverse=True,
    )
    assert out[0].created_by == "llm"
    assert out[1].created_by == "geometric"


# ----------------------------------------------------------------------
# 6. Defensive: missing observation in the lookup table
# ----------------------------------------------------------------------


def test_missing_observation_keeps_original_link(tmp_path):
    """If a link references a (view, tracklet) we don't have an obs
    for (shouldn't happen in practice; defensive guard), we keep the
    original link rather than crashing the whole ingest."""
    # Only one observation; link references a tracklet not in obs
    obs = [_obs(tmp_path, "v1", "a")]
    links = [_link("v1", "a", "v2", "nobody", conf=0.3)]
    llm = ScriptedLLM([json.dumps({"same_object": True, "confidence": 0.9})])

    out = _llm_fallback_upgrade_links(
        links, obs, llm, threshold=0.5, max_per_view=1,
    )
    assert llm.calls == 0
    assert len(out) == 1
    assert out[0].created_by == "geometric"
