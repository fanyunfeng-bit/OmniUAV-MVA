"""Unit tests for `mva.l2_crossview.llm_mode.LLMCrossViewLinker` (M4.1).

Covers prompt-driven dispatch + JSON parsing + retry + ROI hybrid loader
+ confidence threshold gating. All tests use a scripted fake LLM
client + tmp JPEG / mp4 / PNG-dir ROI sources — no GPU, no model
download.

Contract-level invariants (empty input → [], output is CrossViewLink,
DESC ordering, created_by enum) are in
`tests/contracts/test_cross_view_link.py`, parametrized over all 3 modes.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from mva.contracts import CrossViewLink, ViewObservation
from mva.l2_crossview.llm_mode import (
    LLMCrossViewLinker,
    _parse_json_block,
    default_roi_loader,
)


# ----------------------------------------------------------------------
# Test scaffolding
# ----------------------------------------------------------------------


class ScriptedLLM:
    """Fake LLM client returning canned responses in sequence. Records
    every prompt + images received so tests can assert on prompt shape."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._i = 0
        self.prompts_seen: list[str] = []
        self.images_seen: list[list[np.ndarray]] = []

    def complete(
        self, prompt: str, images=None, **_kw,
    ) -> str:
        self.prompts_seen.append(prompt)
        self.images_seen.append(list(images) if images else [])
        if self._i >= len(self._responses):
            return self._responses[-1]
        out = self._responses[self._i]
        self._i += 1
        return out


def _obs(view: str, tk: str, *, t: float = 0.0, cls: str = "person",
         bbox=(0.1, 0.1, 0.3, 0.3), segment_idx=None,
         roi_uri=None, frame_idx=None, source_uri=None) -> ViewObservation:
    return ViewObservation(
        view_id=view, tracklet_id=tk, t=t, bbox=bbox, class_name=cls,
        segment_idx=segment_idx,
        roi_uri=roi_uri, frame_idx=frame_idx, source_uri=source_uri,
    )


def _stub_roi(_obs):
    """Synthetic 8x8 BGR crop — keeps tests from needing real files when
    we're only exercising prompt / parse / threshold paths."""
    return np.full((8, 8, 3), 128, dtype=np.uint8)


# ----------------------------------------------------------------------
# link() — happy path + bucketing + thresholds
# ----------------------------------------------------------------------


def test_link_empty_input_returns_empty_list():
    """Contract: empty input → []. Doubly covered (also in contract test)
    because unit-side regressions tend to surface here first."""
    linker = LLMCrossViewLinker(llm_client=ScriptedLLM([]))
    assert linker.link([]) == []


def test_link_no_client_returns_empty_list_for_real_observations():
    """No LLM wired → never call any loader, just []. This is the
    contract-fixture's zero-arg ctor path."""
    linker = LLMCrossViewLinker()
    obs = [_obs("v1", "t1"), _obs("v2", "t2")]
    assert linker.link(obs) == []


def test_link_returns_link_when_llm_says_same_object_above_threshold():
    llm = ScriptedLLM([
        json.dumps({
            "same_object": True, "confidence": 0.8,
            "reasoning": "matching clothing + pose",
        })
    ])
    linker = LLMCrossViewLinker(llm_client=llm, roi_loader=_stub_roi)
    links = linker.link([_obs("v1", "t1"), _obs("v2", "t2")])
    assert len(links) == 1
    link = links[0]
    assert isinstance(link, CrossViewLink)
    assert link.created_by == "llm"
    assert link.confidence == pytest.approx(0.8)
    assert set(link.view_observations) == {("v1", "t1"), ("v2", "t2")}


def test_link_drops_pair_when_same_object_false():
    """LLM says 'not same' → no CrossViewLink emitted (even with high
    confidence — the boolean is the gate)."""
    llm = ScriptedLLM([
        json.dumps({
            "same_object": False, "confidence": 0.9,
            "reasoning": "totally different clothing",
        })
    ])
    linker = LLMCrossViewLinker(llm_client=llm, roi_loader=_stub_roi)
    assert linker.link([_obs("v1", "t1"), _obs("v2", "t2")]) == []


def test_link_drops_pair_below_confidence_threshold():
    llm = ScriptedLLM([
        json.dumps({
            "same_object": True, "confidence": 0.3,
            "reasoning": "very unsure",
        })
    ])
    linker = LLMCrossViewLinker(
        llm_client=llm, roi_loader=_stub_roi, confidence_threshold=0.5,
    )
    assert linker.link([_obs("v1", "t1"), _obs("v2", "t2")]) == []


def test_link_only_pairs_same_class_observations():
    """Class mismatch → never ask the LLM (waste of tokens)."""
    llm = ScriptedLLM([
        json.dumps({"same_object": True, "confidence": 0.9})
    ])
    linker = LLMCrossViewLinker(llm_client=llm, roi_loader=_stub_roi)
    out = linker.link([
        _obs("v1", "t1", cls="person"),
        _obs("v2", "t2", cls="car"),
    ])
    assert out == []
    assert llm.prompts_seen == []   # never bothered to ask


def test_link_skips_pair_when_roi_loader_returns_none():
    """ROI missing → skip LLM call, drop the pair. Never feed the LLM
    blank input."""
    def empty_loader(_obs):
        return None
    llm = ScriptedLLM([json.dumps({"same_object": True, "confidence": 0.9})])
    linker = LLMCrossViewLinker(llm_client=llm, roi_loader=empty_loader)
    assert linker.link([_obs("v1", "t1"), _obs("v2", "t2")]) == []
    assert llm.prompts_seen == []


def test_link_retries_once_on_malformed_then_recovers():
    """First response unparseable; second (stricter) succeeds → link
    emitted with the retry's confidence."""
    llm = ScriptedLLM([
        "Sure, let me think about this... uncertain.",
        json.dumps({"same_object": True, "confidence": 0.7}),
    ])
    linker = LLMCrossViewLinker(
        llm_client=llm, roi_loader=_stub_roi, max_retries=1,
    )
    links = linker.link([_obs("v1", "t1"), _obs("v2", "t2")])
    assert len(links) == 1
    assert llm.prompts_seen[1].endswith(
        "上一次回复无法解析为 JSON。请只输出一行 JSON，不要任何其他字符。"
    )


def test_link_gives_up_after_retry_exhausted():
    """Both calls unparseable → no link, no Pydantic error raised."""
    llm = ScriptedLLM(["um...", "still um..."])
    linker = LLMCrossViewLinker(
        llm_client=llm, roi_loader=_stub_roi, max_retries=1,
    )
    assert linker.link([_obs("v1", "t1"), _obs("v2", "t2")]) == []
    assert len(llm.prompts_seen) == 2


def test_link_sorts_multiple_pairs_by_confidence_desc():
    llm = ScriptedLLM([
        json.dumps({"same_object": True, "confidence": 0.6}),
        json.dumps({"same_object": True, "confidence": 0.9}),
        json.dumps({"same_object": True, "confidence": 0.7}),
        json.dumps({"same_object": True, "confidence": 0.8}),
    ])
    linker = LLMCrossViewLinker(llm_client=llm, roi_loader=_stub_roi)
    # 2 obs in v1 × 2 obs in v2 = 4 pairs (same class, same bucket)
    out = linker.link([
        _obs("v1", "a"), _obs("v1", "b"),
        _obs("v2", "c"), _obs("v2", "d"),
    ])
    assert [link.confidence for link in out] == sorted(
        [link.confidence for link in out], reverse=True,
    )
    assert len(out) == 4


def test_link_clamps_out_of_range_confidence():
    """Qwen sometimes returns confidence > 1.0 or < 0 — we clamp before
    Pydantic sees it so the model raise doesn't escape."""
    llm = ScriptedLLM([
        json.dumps({"same_object": True, "confidence": 1.5})
    ])
    linker = LLMCrossViewLinker(llm_client=llm, roi_loader=_stub_roi)
    out = linker.link([_obs("v1", "t1"), _obs("v2", "t2")])
    assert len(out) == 1
    assert out[0].confidence == pytest.approx(1.0)


def test_link_rejects_non_numeric_confidence():
    """LLM returns 'high' instead of a number — we drop rather than
    crash. (Same rule as the L3 reasoner.)"""
    llm = ScriptedLLM([
        json.dumps({"same_object": True, "confidence": "high"})
    ])
    linker = LLMCrossViewLinker(
        llm_client=llm, roi_loader=_stub_roi, max_retries=0,
    )
    assert linker.link([_obs("v1", "t1"), _obs("v2", "t2")]) == []


def test_link_requires_at_least_two_views():
    """Two observations from the same view → no candidate (the
    CrossViewLink Pydantic invariant already enforces this; the linker
    pre-filters to avoid wasted LLM calls)."""
    llm = ScriptedLLM([json.dumps({"same_object": True, "confidence": 0.9})])
    linker = LLMCrossViewLinker(llm_client=llm, roi_loader=_stub_roi)
    out = linker.link([
        _obs("v1", "t1"), _obs("v1", "t2"),    # same view
    ])
    assert out == []
    assert llm.prompts_seen == []


def test_link_invalid_confidence_threshold_raises():
    with pytest.raises(ValueError):
        LLMCrossViewLinker(confidence_threshold=1.5)
    with pytest.raises(ValueError):
        LLMCrossViewLinker(confidence_threshold=-0.1)


# ----------------------------------------------------------------------
# default_roi_loader — hybrid cache/decode behavior
# ----------------------------------------------------------------------


def test_roi_loader_cache_hit_reads_jpeg(tmp_path):
    """(b) path: when roi_uri exists and is readable, we return its
    pixels without decoding the parent source. The cheapest path."""
    img = np.full((20, 30, 3), 200, dtype=np.uint8)
    roi_path = tmp_path / "cached.jpg"
    cv2.imwrite(str(roi_path), img)
    obs = _obs("v1", "t1", roi_uri=str(roi_path))
    out = default_roi_loader(obs)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] > 0 and out.shape[1] > 0


def test_roi_loader_cache_miss_falls_back_to_image_dir(tmp_path):
    """(a) fallback: roi_uri unset → use source_uri (an image dir) +
    frame_idx to imread + crop. Models the MATRIX PNG-sequence path."""
    src = tmp_path / "frames"
    src.mkdir()
    for i in range(4):
        cv2.imwrite(
            str(src / f"{i:04d}.png"),
            np.full((40, 60, 3), 50 + i * 20, dtype=np.uint8),
        )
    obs = _obs(
        "v1", "t1",
        bbox=(0.1, 0.1, 0.5, 0.5),
        source_uri=str(src), frame_idx=2,
    )
    out = default_roi_loader(obs)
    assert isinstance(out, np.ndarray)
    assert out.size > 0


def test_roi_loader_returns_none_when_no_source(tmp_path):
    """Neither cache nor source_uri → None. Caller skips the LLM call."""
    obs = _obs("v1", "t1")
    assert default_roi_loader(obs) is None


def test_roi_loader_cache_path_missing_falls_through_to_decode(tmp_path):
    """If roi_uri is set but the file is gone (cache deleted between
    ingest and query), we should still try the delayed-decode path
    rather than silently failing."""
    src = tmp_path / "frames"
    src.mkdir()
    cv2.imwrite(
        str(src / "0000.png"), np.full((40, 60, 3), 30, dtype=np.uint8),
    )
    obs = _obs(
        "v1", "t1",
        bbox=(0.0, 0.0, 0.5, 0.5),
        roi_uri=str(tmp_path / "does_not_exist.jpg"),
        source_uri=str(src), frame_idx=0,
    )
    out = default_roi_loader(obs)
    assert isinstance(out, np.ndarray)
    assert out.size > 0


def test_roi_loader_frame_idx_out_of_range_returns_none(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    cv2.imwrite(
        str(src / "0000.png"), np.full((10, 10, 3), 0, dtype=np.uint8),
    )
    obs = _obs(
        "v1", "t1", source_uri=str(src), frame_idx=99,
    )
    assert default_roi_loader(obs) is None


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


class TestParseJsonBlockL2:
    def test_clean_json(self):
        assert _parse_json_block('{"a": 1}') == {"a": 1}

    def test_wordy_with_embedded_json(self):
        assert _parse_json_block(
            'Sure: {"same_object": true, "confidence": 0.8} done'
        ) == {"same_object": True, "confidence": 0.8}

    def test_unparseable_returns_none(self):
        assert _parse_json_block("not valid at all") is None

    def test_empty_returns_none(self):
        assert _parse_json_block("") is None
