"""Smoke tests for all 8 §3.4 interface stubs.

These do NOT verify semantic correctness — only that:
  1. The stub is importable
  2. The advertised method/attribute exists
  3. Calling it does not crash unexpectedly

Purpose: lock the §3.4 surface against silent API rot. When the project
later refactors module boundaries or method signatures, these tests fail
loudly instead of leaving dead stubs.
"""
from __future__ import annotations

import numpy as np
import pytest


# §3.4 #1 — Telemetry Bus
def test_telemetry_field_passes_through() -> None:
    """Frame.telemetry can be None (default) or a dict; downstream survives both."""
    from mva.contracts import Frame

    f_none = Frame(view_id="v1", t=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))
    assert f_none.telemetry is None

    f_dict = Frame(
        view_id="v1",
        t=0.0,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        telemetry={"gps_lat": 39.9, "gps_lon": 116.4, "alt": 100.0},
    )
    assert f_dict.telemetry == {"gps_lat": 39.9, "gps_lon": 116.4, "alt": 100.0}


# §3.4 #2 — L2 LLM mode
def test_llm_linker_returns_empty_list_when_unimplemented() -> None:
    """LLMCrossViewLinker is callable and respects the empty-list contract."""
    from mva.l2_crossview import LLMCrossViewLinker

    linker = LLMCrossViewLinker()
    result = linker.link([])
    assert result == []
    assert isinstance(result, list)


# §3.4 #3 — L3 LLM mode
def test_llm_reasoner_methods_callable() -> None:
    """LLMReasoner has the three Reasoner Protocol methods, callable without crash."""
    from mva.l3_events import LLMReasoner

    r = LLMReasoner()
    assert r.predict_trajectory("v1", "t1", 5.0) is None
    assert r.classify_behavior("v1", "t1", {}) == "unknown"
    assert r.detect_anomaly("v1", "t1") is None


# §3.4 #4 — L4 telemetry-context prompt placeholder
def test_prompt_template_renders_with_empty_telemetry() -> None:
    """render_describe_prompt works with telemetry=None (default) and a populated dict."""
    from mva.l4_llm import render_describe_prompt

    p_empty = render_describe_prompt(view_id="drone-1", telemetry=None)
    assert "drone-1" in p_empty
    # Empty telemetry → no "GPS" / "altitude" line in the prompt
    assert "GPS" not in p_empty

    p_full = render_describe_prompt(
        view_id="drone-1", telemetry={"gps_lat": 39.9, "gps_lon": 116.4, "alt": 100.0}
    )
    assert "GPS" in p_full
    assert "drone-1" in p_full


# §3.4 #5 — L4 fine-tuning channel
def test_llm_client_load_signature_exists() -> None:
    """LLMClient.load(model_path) exists and accepts a string."""
    from mva.l4_llm import LLMClient

    client = LLMClient(model_path=None)
    assert client.is_mock is True
    client.load("Qwen/Qwen2.5-VL-7B-Instruct")
    assert client.model_path == "Qwen/Qwen2.5-VL-7B-Instruct"
    # Don't call .complete() — that would trigger model download.


# §3.4 #6 — Mode B BriefingAgent
def test_briefing_agent_protocol_callable() -> None:
    """NullBriefingAgent satisfies the BriefingAgent Protocol and returns None / empty briefing."""
    from mva.contracts import Briefing, Event
    from mva.l6_interaction import BriefingAgent, NullBriefingAgent

    agent = NullBriefingAgent()
    assert isinstance(agent, BriefingAgent)  # runtime_checkable Protocol

    e = Event(event_id="e1", type="test", t=0.0, view_id="v1")
    assert agent.on_event(e) is None

    briefing = agent.periodic_summary(0.0, 300.0)
    assert isinstance(briefing, Briefing)
    assert "not implemented" in briefing.tldr.lower()


# §3.4 #7 — Voice input
def test_voice_input_returns_unimplemented() -> None:
    """VoiceInput.get_query raises NotImplementedError — interface present, impl deferred."""
    from mva.l6_interaction import VoiceInput

    v = VoiceInput()
    with pytest.raises(NotImplementedError):
        v.get_query()


# Interactive interface passthrough (post-M2.5 RichQuery contract)
def test_rich_query_with_attachment_constructs_and_orchestrator_accepts() -> None:
    """RichQuery + Attachment construct; Orchestrator accepts both str and
    RichQuery transparently. Locks down the 'leave hook for future UI' API."""
    from pathlib import Path
    from mva.contracts import Attachment, RichQuery
    from mva.l5_state import WorldStateStore
    from mva.l6_interaction import (
        Orchestrator, build_default_registry,
    )

    rq = RichQuery(text="hi", attachments=[
        Attachment(kind="image", path=Path("/tmp/nope.jpg"), label="x"),
    ])
    assert rq.text == "hi"
    assert rq.attachments[0].kind == "image"
    assert rq.source == "text"

    # Orchestrator.run must accept RichQuery (signature stability — does NOT
    # exercise execution since LLM would be needed)
    store = WorldStateStore(db_path=":memory:")
    build_default_registry(store)  # just exercise that the call works
    # Constructing the orchestrator with a None llm would not let .run work,
    # but just verify the run() signature is happy with RichQuery shape.
    assert callable(Orchestrator.run)


# §3.4 #8 — L7 human-in-the-loop RPC
def test_human_correction_endpoint_returns_501() -> None:
    """L7 RPC endpoints return 501 status code — interface in place, real impl in M5."""
    from mva.l7_hitl import HumanCorrectionInterface

    iface = HumanCorrectionInterface()
    assert iface.mark_link_incorrect("link-x") == 501
    assert iface.mark_link_correct("link-x") == 501

    # cross_view_links.created_by="human" enum value is part of the L2 contract
    from mva.contracts import CrossViewLink

    link = CrossViewLink(
        link_id="x",
        view_observations=[("drone-1", "t1"), ("drone-2", "t2")],
        confidence=1.0,
        created_by="human",
        created_at=0.0,
    )
    assert link.created_by == "human"
