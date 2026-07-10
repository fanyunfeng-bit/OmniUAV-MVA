"""Unit tests for the L6 QueryPlanner.

Locks down the JSON-parsing tolerance (plain / fenced / embedded) and the
hard-failure behavior on malformed responses. Uses a ScriptedLLM stub so no
model load is involved.
"""
from __future__ import annotations

import pytest

from mva.l5_state import WorldStateStore
from mva.l6_interaction import QueryPlanner, build_default_registry


class ScriptedLLM:
    """LLM stub returning a queued response per `.complete()` call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("ScriptedLLM out of responses")
        return self._responses.pop(0)


@pytest.fixture
def registry():
    store = WorldStateStore(db_path=":memory:")
    return build_default_registry(store=store)


def test_parses_plain_json(registry):
    llm = ScriptedLLM(['{"intent": "count", "tool_calls": [], "rationale": "ok"}'])
    plan = QueryPlanner(llm, registry).plan("有几辆车？")
    assert plan.intent == "count"
    assert plan.tool_calls == []


def test_parses_fenced_json_block(registry):
    response = """好的，这是计划：
```json
{
  "intent": "find",
  "tool_calls": [{"tool": "query_db", "args": {"sql": "SELECT * FROM cross_view_links"}}],
  "rationale": "查询跨视图链接"
}
```
其他闲聊。
"""
    plan = QueryPlanner(ScriptedLLM([response]), registry).plan("找一下")
    assert plan.intent == "find"
    assert len(plan.tool_calls) == 1
    assert plan.tool_calls[0].tool == "query_db"


def test_parses_embedded_json(registry):
    response = '解释: {"intent": "x", "tool_calls": [], "rationale": "r"} 完毕'
    plan = QueryPlanner(ScriptedLLM([response]), registry).plan("Q")
    assert plan.intent == "x"


def test_raises_on_unparseable(registry):
    llm = ScriptedLLM(["完全自由文本，没有任何 JSON"])
    with pytest.raises(ValueError):
        QueryPlanner(llm, registry).plan("Q")


def test_prompt_includes_tool_descriptions(registry):
    llm = ScriptedLLM(['{"intent": "x", "tool_calls": [], "rationale": ""}'])
    QueryPlanner(llm, registry).plan("Q")
    prompt = llm.prompts[0]
    assert "query_db" in prompt
    assert "Q" in prompt


def test_prompt_includes_db_schema(registry):
    store = WorldStateStore(db_path=":memory:")
    store.insert_segment(
        view_id="D1", segment_idx=0, start_t=0.0, end_t=10.0,
        source_uri="a.mp4", embed_chroma_id="c1", nframes_sampled=4,
    )
    reg = build_default_registry(store)
    schema = store.get_schema_summary()
    llm = ScriptedLLM(['{"intent": "x", "tool_calls": [], "rationale": ""}'])
    QueryPlanner(llm, reg, db_schema=schema).plan("how many views?")
    prompt = llm.prompts[0]
    assert "segments" in prompt
    assert "cross_view_links" in prompt


# ----------------------------------------------------------------------
# Tool-name validation + typo correction
# ----------------------------------------------------------------------


def test_validate_tools_typo_corrects_close_match(registry, capsys):
    response = (
        '{"intent": "count", "tool_calls": ['
        '{"tool": "query_d", "args": {"sql": "SELECT 1"}}],'
        '"rationale": "typo"}'
    )
    plan = QueryPlanner(ScriptedLLM([response]), registry).plan("Q")
    assert len(plan.tool_calls) == 1
    assert plan.tool_calls[0].tool == "query_db"
    msg = capsys.readouterr().out
    assert "tool name typo" in msg


def test_validate_tools_unknown_dropped(registry, capsys):
    response = (
        '{"intent": "x", "tool_calls": ['
        '{"tool": "totally_made_up_tool_zzz", "args": {}},'
        '{"tool": "query_db", "args": {"sql": "SELECT 1"}}],'
        '"rationale": ""}'
    )
    plan = QueryPlanner(ScriptedLLM([response]), registry).plan("Q")
    assert len(plan.tool_calls) == 1
    assert plan.tool_calls[0].tool == "query_db"
    msg = capsys.readouterr().out
    assert "dropping unknown tool" in msg


def test_validate_tools_exact_match_passthrough(registry):
    response = (
        '{"intent": "x", "tool_calls": ['
        '{"tool": "query_db", "args": {"sql": "SELECT 1"}}],'
        '"rationale": ""}'
    )
    plan = QueryPlanner(ScriptedLLM([response]), registry).plan("Q")
    assert plan.tool_calls[0].tool == "query_db"


def test_validate_tools_preserves_intent_and_rationale(registry):
    response = (
        '{"intent": "custom_intent_name", "tool_calls": [],'
        '"rationale": "my reasoning here"}'
    )
    plan = QueryPlanner(ScriptedLLM([response]), registry).plan("Q")
    assert plan.intent == "custom_intent_name"
    assert plan.rationale == "my reasoning here"


def test_plan_includes_history_block():
    from mva.l6_interaction.planner import QueryPlanner
    from mva.l6_interaction.tools import ToolRegistry

    class RecordingLLM:
        def __init__(self):
            self.last_prompt = ""

        def complete(self, prompt, *a, **k):
            self.last_prompt = prompt
            return '{"intent": "x", "tool_calls": [], "rationale": "r"}'

    llm = RecordingLLM()
    planner = QueryPlanner(llm, ToolRegistry(), db_schema="SCHEMA")
    planner.plan("它在哪个视角", history_block="[对话历史]\n用户: 红车在哪\n助手: 在 D1")

    assert "红车在哪" in llm.last_prompt        # history reached the planner prompt
    assert "它在哪个视角" in llm.last_prompt     # current question still present


def test_plan_history_defaults_empty():
    from mva.l6_interaction.planner import QueryPlanner
    from mva.l6_interaction.tools import ToolRegistry

    class RecordingLLM:
        def complete(self, prompt, *a, **k):
            return '{"intent": "x", "tool_calls": [], "rationale": "r"}'

    # No history_block kwarg → must not raise (backward compat).
    QueryPlanner(RecordingLLM(), ToolRegistry(), db_schema="S").plan("有几辆车")
