"""Unit tests for the M2.8 type-aware result renderer in
`Orchestrator._render_invocations` / `_render_tool_result`.

The old code did blanket `repr() + truncate to 500 chars` which buried
M2.8 segment enrichment (source_uri / start_t / end_t) at the tail of a
repr() and lost it. The new renderer recognizes:
  - empty list  → '[] (no matches)'
  - segment hit (dict with `segment` key) → compact line with view /
    segment_idx / time window / source_uri tail / detected_classes
  - bbox hit (dict with metadata.vector_kind='bbox') → compact line
    with class_name / confidence / parent segment_idx
  - everything else → repr + 500-char fallback

These tests pin those shapes against the answer-synthesis prompt so the
LLM never sees the truncated version of the high-value fields.
"""
from __future__ import annotations

from mva.l6_interaction.orchestrator import (
    ToolInvocation,
    _render_list_result,
    _render_one_hit,
    _render_tool_result,
)


def _seg_hit(dist: float, view: str, idx: int, start: float, end: float,
             classes: str = "person,car",
             counts: dict | None = None,
             source: str = "/data/long_path_to_some_video_a.mp4") -> dict:
    return {
        "id": f"{view}::seg{idx}::frame::{idx}",
        "distance": dist,
        "metadata": {"vector_kind": "segment", "segment_idx": idx},
        "document": f"{view} [{start}-{end}s]",
        "segment": {
            "view_id": view,
            "segment_idx": idx,
            "start_t": start,
            "end_t": end,
            "source_uri": source,
            "embed_chroma_id": f"{view}::seg{idx}::frame::{idx}",
            "nframes_sampled": 4,
            "detected_classes": classes,
            "detected_counts": counts or {"person": 3, "car": 1},
        },
    }


def _bbox_hit(dist: float, cls: str, conf: float, seg_idx: int) -> dict:
    return {
        "id": f"v::seg{seg_idx}-f0-d0::reid::{seg_idx}",
        "distance": dist,
        "metadata": {
            "vector_kind": "bbox", "class_name": cls,
            "confidence": conf, "segment_idx": seg_idx,
            "view_id_raw": "video_a.mp4",
            "bbox_x1": 1.0, "bbox_y1": 2.0,
            "bbox_x2": 5.0, "bbox_y2": 6.0,
        },
        "document": f"{cls} (conf={conf:.2f})",
    }


# ----------------------------------------------------------------------
# _render_tool_result — dispatch
# ----------------------------------------------------------------------


def test_empty_list_explicit_marker():
    rendered = _render_tool_result([]).lower()
    assert "no matches" in rendered
    # M3.9 (P3-13): empty result MUST tell the LLM "do not invent"
    assert "do not invent" in rendered


def test_none_result_gets_no_signal_annotation():
    """M3.9 (P3-13): bare None used to render as 'None' which Qwen
    interpreted as license to guess. Now we hand it a clear no-signal
    annotation so the anti-hallucination prompt has a token to grip."""
    rendered = _render_tool_result(None)
    assert "None" in rendered
    assert "no signal" in rendered.lower()
    assert "do not invent" in rendered.lower()


def test_empty_string_result_gets_no_signal_annotation():
    """describe_scene / query_captions can legitimately return '' when
    no captions are indexed for the scene. Without M3.9 P3-13 this
    rendered as `''` and Qwen filled in based on scene knowledge —
    the exact MATRIX 室内/室外 hallucination case (self-024)."""
    rendered = _render_tool_result("")
    assert "empty" in rendered.lower()
    assert "no signal" in rendered.lower()
    assert "do not invent" in rendered.lower()


def test_non_empty_string_passes_through():
    """A real caption string should NOT get the no-signal annotation
    (we only flag the bare empty case)."""
    rendered = _render_tool_result("两个人在户外平台上行走")
    assert "no signal" not in rendered.lower()
    assert "户外平台" in rendered


def test_answer_prompt_contains_fallback_rule():
    """Answer prompt must keep a fail-closed directive (G-1): the LLM is told
    to say "无法确认"/"没有找到" rather than fabricate when data is missing."""
    from mva.l6_interaction.orchestrator import ANSWER_PROMPT_TEMPLATE
    tmpl = ANSWER_PROMPT_TEMPLATE
    assert "无法确认" in tmpl
    assert "严禁编造" in tmpl          # anti-fabrication is now explicit
    assert "db_context" in tmpl


def test_scalar_int_repr_passthrough():
    assert _render_tool_result(42) == "42"


def test_long_string_truncated_at_500():
    s = "x" * 1000
    out = _render_tool_result(s)
    assert out.endswith("...")
    assert len(out) <= 510


# ----------------------------------------------------------------------
# Segment-hit rendering — the M2.8 use case
# ----------------------------------------------------------------------


def test_segment_hit_surfaces_time_window_and_source():
    hit = _seg_hit(0.18, "video_a.mp4", 3, 30.0, 40.0)
    line = _render_one_hit(hit)
    # All M2.8 enrichment fields must appear inline (not buried in repr)
    assert "video_a.mp4" in line
    assert "idx=3" in line
    assert "[30.0,40.0]s" in line
    assert "person,car" in line              # detected_classes
    assert "0.180" in line                   # distance, 3 decimals


def test_segment_hit_handles_missing_distance():
    hit = _seg_hit(0.0, "v", 0, 0.0, 10.0)
    hit["distance"] = None
    line = _render_one_hit(hit)
    assert "dist=n/a" in line


def test_list_of_segment_hits_shows_total_count():
    hits = [_seg_hit(0.1 * i, "v", i, i*10.0, (i+1)*10.0) for i in range(8)]
    rendered = _render_list_result(hits)
    assert "total=8" in rendered
    assert "more" in rendered                # "and N more..." for >5 items


def test_list_of_segment_hits_under_threshold_no_more_marker():
    hits = [_seg_hit(0.1 * i, "v", i, i*10.0, (i+1)*10.0) for i in range(3)]
    rendered = _render_list_result(hits)
    assert "total=3" in rendered
    assert "more" not in rendered


# ----------------------------------------------------------------------
# Bbox-hit rendering
# ----------------------------------------------------------------------


def test_bbox_hit_surfaces_class_and_segment():
    hit = _bbox_hit(0.22, "person", 0.88, 7)
    line = _render_one_hit(hit)
    assert "class='person'" in line
    assert "seg_idx=7" in line
    assert "0.220" in line                   # distance


def test_bbox_hit_single_class_track_no_track_classes_note():
    """A single-class track populates `classes_in_track` (M3.6.D) but the
    renderer suppresses the `track_classes=...` annotation when there's
    no multiset to surface — otherwise the LLM prompt gets noisy with
    `track_classes='person'` on every single-class hit."""
    hit = _bbox_hit(0.22, "person", 0.88, 7)
    hit["metadata"]["classes_in_track"] = "person"
    line = _render_one_hit(hit)
    assert "track_classes" not in line


def test_segment_hit_strong_match_no_weak_note():
    """M3.8 (P2-12): strong match (dist ≤ 0.85) → no `weak_match` flag.
    The LLM should treat this as a confident hit."""
    hit = _seg_hit(0.45, "v", 0, 0.0, 10.0)
    line = _render_one_hit(hit)
    assert "weak_match" not in line


def test_segment_hit_weak_match_flagged():
    """M3.8 (P2-12): dist > 0.85 → `weak_match=true` flag in the line
    so the LLM caveat answers like '可能没找到红色背包' rather than
    confidently misreporting."""
    hit = _seg_hit(0.92, "v", 0, 0.0, 10.0)
    line = _render_one_hit(hit)
    assert "weak_match=true" in line
    assert "0.920" in line


def test_bbox_hit_weak_match_flagged():
    """Same gating on bbox hits — typical 'asked for red backpack, got
    person crop' scenario where distance saturates around 0.9+."""
    hit = _bbox_hit(0.93, "person", 0.8, 5)
    line = _render_one_hit(hit)
    assert "weak_match=true" in line


def test_bbox_hit_multi_class_track_surfaces_classes_in_track():
    """M3.6.D: when the class-agnostic IoU tracker merged two YOLO labels
    (e.g. cat+dog at same bbox) into one track, the renderer surfaces the
    multiset so the LLM doesn't dismiss the minority class. Without this
    the bbox hit only shows `rep_det.class_name`, hiding P3-12 mergers."""
    hit = _bbox_hit(0.22, "dog", 0.88, 7)
    hit["metadata"]["classes_in_track"] = "cat,dog"
    line = _render_one_hit(hit)
    assert "class='dog'" in line                 # rep class still shown
    assert "track_classes='cat,dog'" in line     # multiset surfaced too


def test_mixed_list_renders_each_hit_in_its_own_style():
    """A list that mixes segment + bbox + generic dicts (legacy) — each
    item must render with its own style, not crash."""
    mixed = [
        _seg_hit(0.1, "v", 0, 0.0, 10.0),
        _bbox_hit(0.2, "car", 0.7, 0),
        {"id": "old", "distance": 0.5, "metadata": {}},   # M2.7 legacy shape
    ]
    out = _render_list_result(mixed)
    assert "<segment" in out
    assert "<bbox" in out
    assert "total=3" in out


# ----------------------------------------------------------------------
# Invocation rendering (wraps result)
# ----------------------------------------------------------------------


def test_render_invocations_preserves_error_path():
    """Errors keep the M2.7 error path; the new renderer only kicks in
    on successful results."""
    from mva.l6_interaction.orchestrator import Orchestrator
    invs = [ToolInvocation(
        tool="query_db", args={"sql": "SELECT COUNT(*) FROM tracklets_x"},
        error="KeyError: missing", result=None,
    )]
    rendered = Orchestrator._render_invocations(invs)
    assert "ERROR" in rendered
    assert "KeyError" in rendered


def test_render_invocations_segment_hit_full_pipeline():
    """End-to-end: a successful invocation returning a segment list goes
    through `_render_invocations` and produces a prompt-ready string
    that still carries source_uri + time window."""
    from mva.l6_interaction.orchestrator import Orchestrator
    invs = [ToolInvocation(
        tool="find_segment_by_description",
        args={"text": "people walking", "top_k": 2},
        result=[_seg_hit(0.18, "video_a.mp4", 3, 30.0, 40.0)],
    )]
    rendered = Orchestrator._render_invocations(invs)
    assert "find_segment_by_description" in rendered
    assert "video_a.mp4" in rendered
    assert "[30.0,40.0]s" in rendered


def test_run_injects_history_into_planner_and_answer():
    from mva.l6_interaction.orchestrator import Orchestrator
    from mva.l6_interaction.planner import QueryPlanner
    from mva.l6_interaction.tools import ToolRegistry

    class RecordingLLM:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt, *a, **k):
            self.prompts.append(prompt)
            return '{"intent": "x", "tool_calls": [], "rationale": "r"}'

    llm = RecordingLLM()
    reg = ToolRegistry()
    planner = QueryPlanner(llm, reg, db_schema="S")
    orch = Orchestrator(llm, planner, reg, db_context="DBC")

    hb = "[对话历史]\n用户: 红车在哪\n助手: 在 D1 视角"
    orch.run("它呢", history_block=hb)

    assert len(llm.prompts) == 2                       # planner + answer
    assert all("红车在哪" in p for p in llm.prompts)   # history in BOTH prompts


def test_run_history_defaults_empty():
    from mva.l6_interaction.orchestrator import Orchestrator
    from mva.l6_interaction.planner import QueryPlanner
    from mva.l6_interaction.tools import ToolRegistry

    class RecordingLLM:
        def complete(self, prompt, *a, **k):
            return '{"intent": "x", "tool_calls": [], "rationale": "r"}'

    llm = RecordingLLM()
    reg = ToolRegistry()
    planner = QueryPlanner(llm, reg, db_schema="S")
    # No history_block kwarg → backward compat, must not raise.
    Orchestrator(llm, planner, reg, db_context="D").run("有几辆车")
