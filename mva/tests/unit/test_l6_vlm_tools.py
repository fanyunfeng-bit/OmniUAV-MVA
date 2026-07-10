"""Unit tests for L6 G-2 VLM-native `look_at` tool + orchestrator cascade.

Covers the honesty gates (retrieval-distance gate, VLM explicit-abstain), the
provenance label, and the cascade where a typed structured tool that comes back
empty *for an existence/identification question* falls back to look_at.

Fakes supply retrieval + a stub VLM; `_decode_frames` is monkeypatched so no
real video / model is touched.
"""
from __future__ import annotations

import numpy as np
import pytest

from mva.l6_interaction import vlm_tools
from mva.l6_interaction.vlm_tools import (
    _look_at, _render_look_at, is_empty_structured, wants_visual_check,
    register_vlm_tools,
)
from mva.l6_interaction.tools import ToolRegistry
from mva.l6_interaction.structured_tools import register_structured_tools


@pytest.fixture(autouse=True)
def _fake_frames(monkeypatch):
    """Avoid real ffmpeg/cv2 decode — hand look_at one fake BGR frame."""
    monkeypatch.setattr(
        vlm_tools, "_decode_frames",
        lambda *a, **k: [np.zeros((8, 8, 3), dtype=np.uint8)],
    )


class FakeVStore:
    def __init__(self, distance):
        self._distance = distance

    def query(self, **_):
        if self._distance is None:
            return []
        return [{
            "id": "view1::seg0::frame::0",
            "distance": self._distance,
            "metadata": {"vector_kind": "segment"},
        }]


class FakeStore:
    def get_segment_by_chroma_id(self, _id):
        return {
            "view_id": "view1", "segment_idx": 0,
            "start_t": 0.0, "end_t": 5.0, "source_uri": "/x.mp4",
        }


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, prompt, **_):
        self.calls += 1
        return self.reply


# ---- look_at gates ---------------------------------------------------------


def test_look_at_grounded_answer():
    res = _look_at(FakeStore(), FakeVStore(0.30), FakeLLM("一艘白色的小船"), "船什么颜色")
    assert res["abstained"] is False
    assert res["answer"] == "一艘白色的小船"
    rendered = _render_look_at(res, {})
    assert "白色" in rendered and "根据画面判断" in rendered


def test_look_at_weak_match_abstains_before_vlm():
    llm = FakeLLM("不该被调用")
    res = _look_at(FakeStore(), FakeVStore(0.95), llm, "船什么颜色")  # > 0.85 gate
    assert res["abstained"] is True and res["reason"] == "weak_match"
    assert llm.calls == 0                       # gated before the VLM call
    assert "无法确认" in _render_look_at(res, {})


def test_look_at_no_segment_abstains():
    res = _look_at(FakeStore(), FakeVStore(None), FakeLLM("x"), "船什么颜色")
    assert res["abstained"] is True and res["reason"] == "no_segment"


def test_look_at_vlm_explicit_abstain():
    res = _look_at(FakeStore(), FakeVStore(0.30), FakeLLM("画面看不清"), "船什么颜色")
    assert res["abstained"] is True and res["reason"] == "vlm_abstain"


# ---- cascade helpers -------------------------------------------------------


def test_wants_visual_check_excludes_counts():
    assert wants_visual_check("有没有船") is True
    assert wants_visual_check("水里漂的是什么") is True
    assert wants_visual_check("有几艘船") is False        # count → trust structured 0
    assert wants_visual_check("view1 有多少目标") is False


def test_is_empty_structured():
    assert is_empty_structured("count_objects", {"total_distinct_tracks": 0}) is True
    assert is_empty_structured("count_objects", {"total_distinct_tracks": 3}) is False
    assert is_empty_structured("which_views", {"views": []}) is True
    assert is_empty_structured("scene_stats", {"views": []}) is False  # not a cascading tool


# ---- orchestrator cascade --------------------------------------------------


class _EmptyStore(FakeStore):
    """Structured-tool surface with NO tracks (so count_objects == empty)."""

    def execute_readonly(self, _sql):
        return [{"view_id": "view1"}]

    def query_tracklets(self, *_a, **_k):
        return []

    def query_segments(self, *_a, **_k):
        return [{"view_id": "view1", "segment_idx": 0, "end_t": 5.0}]

    def query_cross_view_links(self, **_):
        return []


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def complete(self, prompt, **_):
        self.prompts.append(prompt)
        return self._responses.pop(0)


def _registry(store, llm):
    reg = ToolRegistry()
    register_structured_tools(reg, store)
    register_vlm_tools(reg, store, vstore=FakeVStore(0.30), llm=llm)
    return reg


def test_cascade_empty_count_existence_falls_back_to_vlm():
    from mva.l6_interaction import QueryPlanner
    from mva.l6_interaction.orchestrator import Orchestrator

    store = _EmptyStore()
    # planner picks count_objects(boat); then look_at VLM is invoked in cascade.
    llm = ScriptedLLM([
        '{"intent":"exist","tool_calls":[{"tool":"count_objects",'
        '"args":{"class_name":"boat"}}],"rationale":"存在性"}',
        "画面里有一艘小船",         # VLM grounded answer
    ])
    reg = _registry(store, llm)
    orch = Orchestrator(llm=llm, planner=QueryPlanner(llm, reg), registry=reg)
    result = orch.run("有没有船？")

    assert len(llm.prompts) == 2                      # planner + VLM (no answer synth)
    assert "根据画面判断" in result.answer
    assert "小船" in result.answer
    assert result.invocations[-1].tool == "look_at"


def test_cascade_skipped_for_count_question():
    """'有几艘船' with empty result is trusted — no VLM fallback, honest '没有'."""
    from mva.l6_interaction import QueryPlanner
    from mva.l6_interaction.orchestrator import Orchestrator

    store = _EmptyStore()
    llm = ScriptedLLM([
        '{"intent":"count","tool_calls":[{"tool":"count_objects",'
        '"args":{"class_name":"boat"}}],"rationale":"计数"}',
    ])  # only ONE response: if cascade fired, VLM call would IndexError
    reg = _registry(store, llm)
    orch = Orchestrator(llm=llm, planner=QueryPlanner(llm, reg), registry=reg)
    result = orch.run("有几艘船？")

    assert len(llm.prompts) == 1                      # no cascade
    assert "没有检测到" in result.answer


# ---- G-2.1: object mode (bbox crop) ----------------------------------------


class _ObjStore:
    def get_segment(self, view_id, seg_idx):
        return {"view_id": "view1", "segment_idx": 0, "start_t": 0.0,
                "end_t": 5.0, "source_uri": "/x.mp4"}

    def get_segment_by_chroma_id(self, _id):
        return self.get_segment("view1", 0)


class _BBoxVStore:
    def __init__(self, persist_dir, dist=0.3):
        self.persist_dir = persist_dir
        self._d = dist

    def query(self, query_text=None, vector_type=None, view_id=None, top_k=5, **_):
        if vector_type == "reid":
            return [{"id": "bid1", "distance": self._d, "metadata": {
                "vector_kind": "bbox", "view_id_raw": "view1", "segment_idx": 0,
                "bbox_x1": 0.0, "bbox_y1": 0.0, "bbox_x2": 5.0, "bbox_y2": 5.0,
                "source_uri": "/x.mp4", "class_name": "boat"}}]
        return []                                   # no scene hits


def test_is_object_question_routing():
    from mva.l6_interaction.vlm_tools import _is_object_question
    assert _is_object_question("船什么颜色") is True
    assert _is_object_question("水里漂的是什么") is True
    assert _is_object_question("船在做什么") is False      # activity → scene
    assert _is_object_question("画面里发生了什么") is False


def test_look_at_object_uses_cached_roi_crop(tmp_path):
    import cv2
    (tmp_path / "chroma-rois").mkdir()
    cv2.imwrite(str(tmp_path / "chroma-rois" / "bid1.jpg"),
                np.full((6, 6, 3), 200, np.uint8))
    res = _look_at(_ObjStore(), _BBoxVStore(str(tmp_path / "chroma")),
                   FakeLLM("一艘白色的船"), "船什么颜色")
    assert res["abstained"] is False
    assert res["mode"] == "object"
    assert "白色" in res["answer"]


def test_look_at_object_crops_from_frame_when_no_cache(tmp_path):
    # No -rois dir → falls back to cropping the object out of the source frame
    # (the autouse fixture hands _decode_frames a fake frame to crop).
    res = _look_at(_ObjStore(), _BBoxVStore(str(tmp_path / "chroma")),
                   FakeLLM("白色小船"), "船什么颜色")
    assert res["abstained"] is False and res["mode"] == "object"


def test_object_question_falls_back_to_scene_when_no_bbox():
    # FakeVStore yields only segment-kind hits → object mode finds no bbox →
    # dispatcher falls back to whole-frame scene mode.
    res = _look_at(FakeStore(), FakeVStore(0.3), FakeLLM("一片水面"), "船什么颜色")
    assert res["abstained"] is False
    assert res["mode"] == "scene"
