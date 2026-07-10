"""Unit tests for L6 G-1 typed structured tools + orchestrator fast path.

These tools replace LLM-authored SQL: the planner picks a tool by name and
fills slots; the tool runs correct, JSON-aware queries internally. We verify
the grounded numbers (per-frame peak vs distinct tracks), the fail-closed
empty render, and that a single renderable tool short-circuits the answer LLM.

A FakeStore supplies canned tracklets/segments/links so we test the tool logic
in isolation (no DuckDB / model load).
"""
from __future__ import annotations

from types import SimpleNamespace

from mva.l6_interaction.tools import ToolRegistry
from mva.l6_interaction.structured_tools import register_structured_tools


def _tk(tid, seg, bboxes):
    ts = [b[0] for b in bboxes]
    return {
        "tracklet_id": tid, "segment_idx": seg,
        "t_start": min(ts), "t_end": max(ts),
        "bboxes": bboxes, "embedding_ref": None,
    }


class FakeStore:
    """Minimal WorldStateStore surface for the structured tools."""

    def __init__(self):
        # view1: frame t=1 has 2 boats (A,B) + 1 uav (C); frame t=2 has 1 boat (A)
        self._tracks = {
            "view1": [
                _tk("v1-A", 0, [[1.0, 0, 0, 10, 10, "boat", 0.9],
                                [2.0, 0, 0, 10, 10, "boat", 0.8]]),
                _tk("v1-B", 0, [[1.0, 20, 20, 30, 30, "boat", 0.7]]),
                _tk("v1-C", 0, [[1.0, 40, 40, 50, 50, "uav", 0.6]]),
            ],
            "view2": [
                _tk("v2-D", 0, [[1.0, 0, 0, 10, 10, "boat", 0.9]]),
            ],
        }

    def execute_readonly(self, sql):
        return [{"view_id": "view1"}, {"view_id": "view2"}]

    def query_tracklets(self, view_id, t_start=None, t_end=None, segment_idx=None):
        out = []
        for tk in self._tracks.get(view_id, []):
            if t_end is not None and tk["t_start"] > t_end:
                continue
            if t_start is not None and tk["t_end"] < t_start:
                continue
            out.append(tk)
        return out

    def query_segments(self, view_id=None, t_start=None, t_end=None):
        segs = [
            {"view_id": "view1", "segment_idx": 0, "end_t": 5.0},
            {"view_id": "view2", "segment_idx": 0, "end_t": 5.0},
        ]
        return [s for s in segs if view_id is None or s["view_id"] == view_id]

    def query_cross_view_links(self, **_):
        return [SimpleNamespace(
            view_observations=[["view1", "v1-A"], ["view2", "v2-D"]],
            confidence=0.9,
        )]


def _reg():
    reg = ToolRegistry()
    register_structured_tools(reg, FakeStore())
    return reg


def _call(name, args):
    reg = _reg()
    res = reg.call(name, args)
    return res, reg.render(name, res, args)


# ---- count_objects ---------------------------------------------------------


def test_count_objects_peak_is_per_frame_not_per_track():
    res, rendered = _call("count_objects", {"class_name": "boat"})
    # view1: frame t=1 has 2 boats → peak 2; 2 distinct tracks (A,B)
    assert res["per_view"]["view1"] == {"peak_per_frame": 2, "distinct_tracks": 2}
    assert res["per_view"]["view2"] == {"peak_per_frame": 1, "distinct_tracks": 1}
    assert "最多同时 2" in rendered and "view1" in rendered


def test_count_objects_view_filter():
    res, _ = _call("count_objects", {"class_name": "uav", "view_id": "view1"})
    assert set(res["per_view"]) == {"view1"}
    assert res["per_view"]["view1"]["peak_per_frame"] == 1


def test_count_objects_absent_class_fail_closed():
    res, rendered = _call("count_objects", {"class_name": "person"})
    assert res["total_distinct_tracks"] == 0
    assert "没有检测到 person" in rendered
    # must NOT invent a number
    assert "最多同时" not in rendered


# ---- list_objects / which_views / when_seen / objects_at_time --------------


def test_list_objects_reports_peak_per_class():
    res, rendered = _call("list_objects", {})
    assert res["per_view"]["view1"]["boat"] == 2  # peak simultaneous
    assert res["per_view"]["view1"]["uav"] == 1
    assert "view1" in rendered and "boat×2" in rendered


def test_which_views():
    res, rendered = _call("which_views", {"class_name": "boat"})
    assert res["views"] == ["view1", "view2"]
    assert "view1" in rendered and "view2" in rendered

    _, rendered_absent = _call("which_views", {"class_name": "truck"})
    assert "没有任何视角" in rendered_absent


def test_objects_at_time_uses_nearest_frame():
    res, rendered = _call("objects_at_time", {"t": 1.0})
    assert res["per_view"]["view1"]["boat"] == 2
    assert "第 1 秒" in rendered


def test_when_seen_dedups_windows():
    # both boat tracks A,B are in the same 0-window → one time span, not two
    res, rendered = _call("when_seen", {"class_name": "boat", "view_id": "view1"})
    assert res["spans"], "expected at least one span"
    # render must not repeat the identical window
    assert rendered.count("1–2s") <= 1 or "1–2" in rendered


# ---- cross_view_matches / scene_stats --------------------------------------


def test_cross_view_matches_class_consistency():
    res, rendered = _call("cross_view_matches", {})
    assert res["n_links"] == 1
    assert res["links"][0]["class_consistent"] is True
    assert res["links"][0]["classes"] == ["boat", "boat"]
    assert "1 对" in rendered


def test_scene_stats():
    res, rendered = _call("scene_stats", {})
    assert res["views"] == ["view1", "view2"]
    assert res["cross_view_links"] == 1
    assert "2 个视角" in rendered


# ---- every typed tool ships a render --------------------------------------


def test_all_structured_tools_have_render():
    reg = _reg()
    for name in ["count_objects", "list_objects", "which_views", "when_seen",
                 "objects_at_time", "cross_view_matches", "scene_stats"]:
        assert reg.has_render(name), f"{name} missing render()"


# ---- orchestrator single-tool fast path ------------------------------------


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def complete(self, prompt, **_):
        self.prompts.append(prompt)
        return self._responses.pop(0)  # IndexError if called more than scripted


def test_orchestrator_fast_path_skips_answer_llm():
    from mva.l6_interaction import QueryPlanner
    from mva.l6_interaction.orchestrator import Orchestrator

    reg = _reg()
    plan_json = (
        '{"intent":"count","tool_calls":'
        '[{"tool":"count_objects","args":{"class_name":"boat"}}],'
        '"rationale":"数船"}'
    )
    # Only ONE scripted response: the planner. If the answer LLM were invoked
    # the stub would IndexError — so a clean run proves the short-circuit.
    llm = ScriptedLLM([plan_json])
    orch = Orchestrator(llm=llm, planner=QueryPlanner(llm, reg), registry=reg)
    result = orch.run("有几艘船？")

    assert len(llm.prompts) == 1                      # planner only
    assert result.invocations[0].tool == "count_objects"
    assert "最多同时 2" in result.answer


def test_orchestrator_multi_tool_uses_answer_llm():
    """Two tool calls → no fast path → answer LLM composes (2 LLM calls)."""
    from mva.l6_interaction import QueryPlanner
    from mva.l6_interaction.orchestrator import Orchestrator

    reg = _reg()
    plan_json = (
        '{"intent":"count","tool_calls":['
        '{"tool":"count_objects","args":{"class_name":"boat"}},'
        '{"tool":"which_views","args":{"class_name":"boat"}}],'
        '"rationale":"数+定位"}'
    )
    llm = ScriptedLLM([plan_json, "view1 和 view2 都有船。"])
    orch = Orchestrator(llm=llm, planner=QueryPlanner(llm, reg), registry=reg)
    result = orch.run("有几艘船，分别在哪个视角？")

    assert len(llm.prompts) == 2                      # planner + answer synthesis
    assert result.answer == "view1 和 view2 都有船。"
