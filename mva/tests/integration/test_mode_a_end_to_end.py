"""End-to-end integration test for L6 Mode A.

Populates a real WorldStateStore (DuckDB :memory:) + real VectorStore (tmp
ChromaDB), wires up a scripted-LLM Orchestrator, and asserts the full
NL→plan→tools→answer pipeline works.

The LLM is a ScriptedLLM stub: it pops one canned response for the planning
turn and another for the answer-synthesis turn. This exercises every
component except the real Qwen2.5-VL model load.

NOTE: The orchestrator auto-injects find_segment_by_description on every
query (7B model workaround). Tests account for this extra invocation.
"""
from __future__ import annotations

import pytest

from mva.contracts import CrossViewLink, Event
from mva.l5_state import VectorStore, WorldStateStore
from mva.l6_interaction import (
    Orchestrator,
    QueryPlanner,
    build_default_registry,
)


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("ScriptedLLM out of responses")
        return self._responses.pop(0)


@pytest.fixture
def populated_world(tmp_path):
    store = WorldStateStore(db_path=":memory:")
    for i in range(3):
        store.insert_tracklet(
            view_id="drone-1",
            tracklet_id=f"v1-tk-{i}",
            t_start=float(i * 20),
            t_end=float(i * 20 + 10),
            bboxes=[(float(i * 20), 10, 10, 20, 20)],
        )
    store.insert_tracklet(
        "drone-2", "v2-tk-0", 0.0, 10.0, [(0.0, 30, 30, 40, 40)]
    )
    store.insert_caption(
        "drone-1", "c-1", frame_idx=0, t=5.0, caption_text="一辆红色小车"
    )
    store.insert_event(
        Event(
            event_id="e-1",
            type="loitering",
            t=10.0,
            view_id="drone-1",
            tracklet_ids=["v1-tk-0"],
            summary_text="停留 > 30s",
        )
    )
    store.insert_cross_view_link(
        CrossViewLink(
            link_id="L-1",
            view_observations=[("drone-1", "v1-tk-0"), ("drone-2", "v2-tk-0")],
            confidence=0.9,
            created_by="geometric",
            created_at=0.0,
        )
    )
    vstore = VectorStore(persist_dir=str(tmp_path / "chroma"))
    return store, vstore


def test_count_query_end_to_end(populated_world):
    """LLM writes SQL to count tracklets in a time window."""
    store, vstore = populated_world
    sql = "SELECT COUNT(*) AS cnt FROM tracklets_drone_1 WHERE t_start >= 0.0 AND t_end <= 60.0"
    plan_json = (
        '{"intent": "count", '
        '"tool_calls": [{"tool": "query_db", '
        f'"args": {{"sql": "{sql}"}}}}], '
        '"rationale": "SQL count"}'
    )
    answer = "drone-1 在过去 60 秒里共出现 3 个 tracklet。"

    llm = ScriptedLLM([plan_json, answer])
    registry = build_default_registry(store, vstore=vstore)
    planner = QueryPlanner(llm, registry)
    orch = Orchestrator(llm, planner, registry)

    result = orch.run("过去 60 秒里 drone-1 有几辆车？")

    assert result.plan.intent == "count"
    # Auto-inject adds find_segment_by_description; planner adds query_db
    tools_called = [inv.tool for inv in result.invocations]
    assert "find_segment_by_description" in tools_called
    assert "query_db" in tools_called
    query_db_inv = next(i for i in result.invocations if i.tool == "query_db")
    assert query_db_inv.error == ""
    assert query_db_inv.result[0]["cnt"] == 3
    assert result.answer == answer


def test_multi_tool_plan(populated_world):
    """Composite: SQL for tracklets + SQL for cross-view links."""
    store, vstore = populated_world
    plan_json = (
        '{"intent": "summary", "tool_calls": ['
        '{"tool": "query_db", "args": {"sql": "SELECT * FROM tracklets_drone_1"}}, '
        '{"tool": "query_db", "args": {"sql": "SELECT * FROM cross_view_links WHERE confidence > 0.5"}}'
        '], "rationale": "tracklets + 跨视图链接"}'
    )
    answer = "drone-1 有 3 个 tracklet；存在 1 个跨视图链接。"

    llm = ScriptedLLM([plan_json, answer])
    registry = build_default_registry(store, vstore=vstore)
    orch = Orchestrator(llm, QueryPlanner(llm, registry), registry)

    result = orch.run("drone-1 的概况 + 跨视图情况")

    # 1 auto-inject + 2 planner calls = 3
    tools_called = [inv.tool for inv in result.invocations]
    assert tools_called.count("query_db") == 2
    # Find the cross_view_links query result
    cv_inv = [i for i in result.invocations
              if i.tool == "query_db" and "cross_view" in i.args.get("sql", "")]
    assert len(cv_inv) == 1
    assert cv_inv[0].result[0]["confidence"] == 0.9


def test_tool_error_does_not_crash_orchestrator(populated_world):
    store, vstore = populated_world
    plan_json = (
        '{"intent": "x", '
        '"tool_calls": [{"tool": "query_db", '
        '"args": {"sql": "DROP TABLE cross_view_links"}}], '
        '"rationale": "bad query"}'
    )
    answer = "出错了。"

    llm = ScriptedLLM([plan_json, answer])
    registry = build_default_registry(store, vstore=vstore)
    orch = Orchestrator(llm, QueryPlanner(llm, registry), registry)
    result = orch.run("...")

    # query_db with DROP should error
    query_db_inv = next(i for i in result.invocations if i.tool == "query_db")
    assert query_db_inv.error != ""
    assert "ValueError" in query_db_inv.error
    assert result.answer == answer


def test_no_tool_calls_still_synthesizes_answer(populated_world):
    store, vstore = populated_world
    plan_json = '{"intent": "smalltalk", "tool_calls": [], "rationale": "聊天"}'
    answer = "你好，请问需要查询哪一路视频？"

    llm = ScriptedLLM([plan_json, answer])
    registry = build_default_registry(store, vstore=vstore)
    orch = Orchestrator(llm, QueryPlanner(llm, registry), registry)
    result = orch.run("你好")

    # Even with no planner calls, auto-inject fires segment search
    assert any(i.tool == "find_segment_by_description" for i in result.invocations)
    assert result.answer == answer
