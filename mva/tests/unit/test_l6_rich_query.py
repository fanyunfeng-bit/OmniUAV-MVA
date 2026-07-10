"""Tests for the L6 Orchestrator's RichQuery + attachment-tool routing.

ScriptedLLM stubs in for both planner LLM calls (plan + answer). Mock
MultimodalEmbedder used so no 16 GB load happens. Tests verify:

- Orchestrator accepts both str and RichQuery
- No attachments → base registry tools are visible to planner; no
  attachment-bound tools registered
- With attachments → describe_attachment / find_similar_to_attachment /
  compare_attachments are registered for THIS run only, base registry is
  untouched
- The planner prompt receives a `[已附加文件]` block with idx + label
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from mva.contracts import Attachment, RichQuery
from mva.l5_state import MultimodalEmbedder, VectorStore, WorldStateStore
from mva.l6_interaction import (
    Orchestrator,
    QueryPlanner,
    build_default_registry,
)


class ScriptedLLM:
    """Tiny LLM stub: pops responses in order, records prompts."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("ScriptedLLM out of responses")
        return self._responses.pop(0)


@pytest.fixture
def base_setup():
    """Shared store + registry + mock embedder for all the tests below."""
    store = WorldStateStore(db_path=":memory:")
    registry = build_default_registry(store)
    embedder = MultimodalEmbedder(model_path=None)  # mock
    return store, registry, embedder


@pytest.fixture
def image_attachment(tmp_path):
    img_path = tmp_path / "target.jpg"
    cv2.imwrite(str(img_path), np.full((16, 16, 3), 200, dtype=np.uint8))
    return Attachment(kind="image", path=img_path, label="target.jpg")


def test_orchestrator_accepts_plain_str(base_setup):
    _, registry, _ = base_setup
    plan = '{"intent": "x", "tool_calls": [], "rationale": ""}'
    answer = "OK"
    llm = ScriptedLLM([plan, answer])
    orch = Orchestrator(llm, QueryPlanner(llm, registry), registry)
    result = orch.run("hello")
    assert result.answer == "OK"
    assert result.question == "hello"


def test_orchestrator_accepts_rich_query_no_attachments(base_setup):
    _, registry, _ = base_setup
    plan = '{"intent": "x", "tool_calls": [], "rationale": ""}'
    answer = "OK"
    llm = ScriptedLLM([plan, answer])
    orch = Orchestrator(llm, QueryPlanner(llm, registry), registry)
    result = orch.run(RichQuery(text="hi"))
    assert result.answer == "OK"


def test_attachment_present_adds_tools_to_planner_prompt(
    base_setup, image_attachment, tmp_path,
):
    store, registry, embedder = base_setup
    vstore = VectorStore(persist_dir=str(tmp_path / "chroma"))
    plan = '{"intent": "describe", "tool_calls": [{"tool": "describe_attachment", "args": {"idx": 0}}], "rationale": "look at the image"}'
    describe_answer = "MOCK-DESCRIPTION"
    final = "The image shows a thing."
    # Planner gets plan, describe_attachment uses llm to "describe", then final answer
    llm = ScriptedLLM([plan, describe_answer, final])

    orch = Orchestrator(
        llm, QueryPlanner(llm, registry), registry,
        embedder=embedder, vstore=vstore,
    )
    result = orch.run(RichQuery(text="describe this", attachments=[image_attachment]))

    # Planner prompt should mention the attachment
    plan_prompt = llm.prompts[0]
    assert "describe_attachment" in plan_prompt
    assert "已附加文件" in plan_prompt
    assert "target.jpg" in plan_prompt

    # Auto-inject fires find_similar_to_attachment, then planner's
    # describe_attachment also fires
    tools_called = [inv.tool for inv in result.invocations]
    assert "find_similar_to_attachment" in tools_called
    assert "describe_attachment" in tools_called
    desc_inv = next(i for i in result.invocations if i.tool == "describe_attachment")
    assert desc_inv.result == "MOCK-DESCRIPTION"
    assert result.answer == "The image shows a thing."


def test_base_registry_untouched_by_per_query_attachment_tools(
    base_setup, image_attachment, tmp_path,
):
    store, registry, embedder = base_setup
    vstore = VectorStore(persist_dir=str(tmp_path / "chroma"))
    pre_tool_names = set(registry.names())
    assert "describe_attachment" not in pre_tool_names

    plan = '{"intent": "x", "tool_calls": [], "rationale": ""}'
    llm = ScriptedLLM([plan, "OK"])
    orch = Orchestrator(
        llm, QueryPlanner(llm, registry), registry,
        embedder=embedder, vstore=vstore,
    )
    orch.run(RichQuery(text="hi", attachments=[image_attachment]))

    # Base registry must still NOT contain attachment tools (no leak)
    post_tool_names = set(registry.names())
    assert post_tool_names == pre_tool_names


def test_find_similar_to_attachment_routes_through_embedder(
    base_setup, image_attachment, tmp_path,
):
    store, registry, embedder = base_setup
    vstore = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=embedder.as_chromadb_embedding_function(),
    )
    # Seed the vstore with a known reid vector — should be the top hit when
    # we encode the same image and query.
    img = cv2.imread(str(image_attachment.path))
    vec = embedder.encode_image(img)
    vstore.add(vec, "reid", "test-view", "tk-target")

    plan = (
        '{"intent": "find", '
        '"tool_calls": [{"tool": "find_similar_to_attachment", '
        '"args": {"idx": 0, "top_k": 3}}], '
        '"rationale": "look up similar"}'
    )
    final = "Found 1 match."
    llm = ScriptedLLM([plan, final])

    orch = Orchestrator(
        llm, QueryPlanner(llm, registry), registry,
        embedder=embedder, vstore=vstore,
    )
    result = orch.run(RichQuery(
        text="find similar", attachments=[image_attachment],
    ))
    inv = result.invocations[0]
    assert inv.tool == "find_similar_to_attachment"
    assert inv.error == ""
    assert isinstance(inv.result, list)
    assert len(inv.result) >= 1
    assert inv.result[0]["metadata"]["tracklet_id"] == "tk-target"


def test_orchestrator_result_has_next_step_hint_default_none(base_setup):
    _, registry, _ = base_setup
    plan = '{"intent": "x", "tool_calls": [], "rationale": ""}'
    llm = ScriptedLLM([plan, "OK"])
    orch = Orchestrator(llm, QueryPlanner(llm, registry), registry)
    result = orch.run("anything")
    assert result.next_step_hint is None
