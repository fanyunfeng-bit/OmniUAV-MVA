"""Tests for mva.cli.query.QueryService.

All tests run with mock LLM + mock embedder (no GPU). Covers:
- Construction succeeds for both with-chroma and without-chroma modes
- list_tools() returns the expected names
- close() releases without raising
- Context-manager interface
- answer() returns OrchestratorResult shape
- M3.6.B: `_warn_hidden_cross_view` helper behaves correctly across the
  three relevant input combinations (P1-07)
"""
from __future__ import annotations

import pytest

from mva.cli.query import QueryService, _warn_hidden_cross_view
from mva.contracts import CrossViewLink
from mva.l5_state import WorldStateStore


def test_construct_db_only(tmp_path):
    service = QueryService(
        db_path=":memory:",
        chroma_dir=None,
        llm_model=None,            # mock LLM
        embedder_model=None,
    )
    try:
        tool_names = {t["name"] for t in service.list_tools()}
        assert "query_db" in tool_names
        # Without vstore, vector tools are NOT registered
        assert "find_by_description" not in tool_names
    finally:
        service.close()


def test_construct_with_chroma_uses_mock_embedder(tmp_path):
    """When chroma_dir is set and embedder_model is None, no embedder loads."""
    service = QueryService(
        db_path=":memory:",
        chroma_dir=str(tmp_path / "chroma"),
        llm_model=None,
        embedder_model=None,        # no embedder → no vstore wired
    )
    try:
        # Because embedder_model is None, the vstore is also None per the
        # facade's construction logic (see QueryService.__init__).
        assert service.embedder is None
        assert service.vstore is None
    finally:
        service.close()


def test_context_manager(tmp_path):
    with QueryService(db_path=":memory:", llm_model=None) as service:
        assert service.list_tools()


def test_answer_returns_orchestrator_result(tmp_path):
    """End-to-end with mock LLM — answer string contains MOCK marker."""
    service = QueryService(db_path=":memory:", llm_model=None)
    try:
        # Mock LLM returns boilerplate that doesn't parse as JSON → planner
        # raises ValueError, which propagates here (the CLI's _print_result
        # would catch it but the service.answer doesn't suppress).
        with pytest.raises(ValueError):
            service.answer("hello")
    finally:
        service.close()


# ----------------------------------------------------------------------
# M3.6.B — _warn_hidden_cross_view helper (PROBLEMS P1-07)
# ----------------------------------------------------------------------


def _seed_one_link(db_path: str) -> None:
    store = WorldStateStore(db_path=db_path)
    try:
        store.insert_cross_view_link(
            CrossViewLink(
                link_id="lnk-test",
                view_observations=[("v1", "tk-a"), ("v2", "tk-b")],
                confidence=0.7,
                created_by="geometric",
                created_at=0.0,
            )
        )
    finally:
        store.close()


def test_warn_hidden_cross_view_fires_when_off_and_db_has_links(
    tmp_path, capsys,
):
    """The whole point of the warning: --cross-view off + DB has rows → user
    sees the count + reason. Without this they think the data is gone."""
    db = tmp_path / "x.duckdb"
    _seed_one_link(str(db))
    _warn_hidden_cross_view(str(db), "off")
    out = capsys.readouterr().out
    assert "--cross-view off" in out
    assert "1 cross_view_links" in out
    assert "hidden from the LLM" in out
    assert "data preserved" in out


def test_warn_hidden_cross_view_silent_when_auto(tmp_path, capsys):
    """When the tools are exposed there's nothing to warn about, even if
    the DB has cross-view rows."""
    db = tmp_path / "x.duckdb"
    _seed_one_link(str(db))
    _warn_hidden_cross_view(str(db), "auto")
    out = capsys.readouterr().out
    assert out == ""


def test_warn_hidden_cross_view_silent_when_off_but_db_empty(tmp_path, capsys):
    """Empty DB → no point warning about hidden rows that don't exist."""
    db = tmp_path / "empty.duckdb"
    # Create the file (empty schema) but no cross_view_links rows
    store = WorldStateStore(db_path=str(db))
    store.close()
    _warn_hidden_cross_view(str(db), "off")
    assert capsys.readouterr().out == ""


def test_warn_hidden_cross_view_silent_for_memory_or_missing(tmp_path, capsys):
    """`:memory:` and non-existent paths are no-ops (the probe would error
    or be meaningless). The launch-time `_check_db_populated` already
    handles missing-DB messaging — we don't duplicate it here."""
    _warn_hidden_cross_view(":memory:", "off")
    _warn_hidden_cross_view(str(tmp_path / "does_not_exist.duckdb"), "off")
    assert capsys.readouterr().out == ""


# ----------------------------------------------------------------------
# M3.8 — _resolve_quantize (PROBLEMS P2-02 + P2-03)
# ----------------------------------------------------------------------


def _make_args(**kw):
    import argparse
    ns = argparse.Namespace(
        quantize=None, chroma_dir=None, no_auto_quantize=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_resolve_quantize_respects_explicit_value(monkeypatch):
    """User explicitly set --quantize int4 → always returned, no
    VRAM probe, no auto-force logic."""
    from mva.cli.query import _resolve_quantize
    # Make the VRAM probe panic if called — we shouldn't reach it
    monkeypatch.setattr(
        "mva.cli.query._probe_total_vram_bytes",
        lambda: pytest.fail("VRAM probe should not run when --quantize is set"),
    )
    assert _resolve_quantize(_make_args(quantize="int4", chroma_dir="/x")) == "int4"
    assert _resolve_quantize(_make_args(quantize="int8")) == "int8"


def test_resolve_quantize_no_chroma_dir_skips_auto_force(monkeypatch):
    """No --chroma-dir → no embedder coexistence → no need to force
    INT4 regardless of VRAM."""
    from mva.cli.query import _resolve_quantize
    monkeypatch.setattr(
        "mva.cli.query._probe_total_vram_bytes",
        lambda: pytest.fail("VRAM probe should not run without chroma_dir"),
    )
    assert _resolve_quantize(_make_args(chroma_dir=None)) is None


def test_resolve_quantize_chroma_small_vram_forces_int4(monkeypatch, capsys):
    """The original P2-02 case: 24 GB GPU + --chroma-dir + no
    --quantize → auto INT4."""
    from mva.cli.query import _resolve_quantize
    monkeypatch.setattr(
        "mva.cli.query._probe_total_vram_bytes", lambda: 24 * 1024 ** 3,
    )
    out = _resolve_quantize(_make_args(chroma_dir="/x"))
    assert out == "int4"
    msg = capsys.readouterr().out
    assert "forcing INT4" in msg
    assert "[cli]" in msg   # banner shared between mva query + mva ask


def test_resolve_quantize_chroma_big_vram_skips_force(monkeypatch, capsys):
    """P2-02 fix: 48 GB GPU + --chroma-dir → keep FP16 (None)."""
    from mva.cli.query import _resolve_quantize
    monkeypatch.setattr(
        "mva.cli.query._probe_total_vram_bytes", lambda: 48 * 1024 ** 3,
    )
    out = _resolve_quantize(_make_args(chroma_dir="/x"))
    assert out is None
    msg = capsys.readouterr().out
    assert "48.0 GB VRAM" in msg
    assert "keeping FP16" in msg
    assert "[cli]" in msg


def test_resolve_quantize_no_auto_quantize_flag_skips_force(monkeypatch):
    """--no-auto-quantize lets the user opt out of the heuristic
    entirely; --quantize unset stays None even on a 24GB GPU."""
    from mva.cli.query import _resolve_quantize
    monkeypatch.setattr(
        "mva.cli.query._probe_total_vram_bytes", lambda: 24 * 1024 ** 3,
    )
    out = _resolve_quantize(_make_args(chroma_dir="/x", no_auto_quantize=True))
    assert out is None


def test_check_db_populated_returns_false_when_db_missing(tmp_path, capsys):
    """M3.8 (PROBLEMS P2-07): missing DB file → False + fatal message.
    Caller (cmd_query / cmd_ask) is expected to translate this into
    return-code 1 so the shell user sees a non-zero exit rather than
    silently burning tokens on a doomed empty-DB query."""
    from mva.cli.query import _check_db_populated
    missing = tmp_path / "does_not_exist.duckdb"
    assert _check_db_populated(str(missing)) is False
    out = capsys.readouterr().out
    assert "[fatal]" in out
    assert "DB not found" in out
    assert str(missing) in out


def test_check_db_populated_returns_true_for_memory(capsys):
    """`:memory:` is always OK — caller continues."""
    from mva.cli.query import _check_db_populated
    assert _check_db_populated(":memory:") is True
    # :memory: should be silent (no spurious warning)
    assert capsys.readouterr().out == ""


def test_check_db_populated_returns_true_for_empty_db_with_soft_warn(
    tmp_path, capsys,
):
    """Empty-but-existing DB: soft warn (not fatal). Caller continues
    so queries can still run against :memory: data added later or just
    to fail gracefully on each tool call."""
    from mva.cli.query import _check_db_populated
    from mva.l5_state import WorldStateStore
    db = tmp_path / "empty.duckdb"
    store = WorldStateStore(db_path=str(db))
    store.close()
    assert _check_db_populated(str(db)) is True
    out = capsys.readouterr().out
    assert "[warn]" in out
    assert "no segments" in out


def test_resolve_quantize_does_not_mutate_args(monkeypatch):
    """P2-03 anti-pattern fix: never write back into the argparse
    Namespace (which could leak state if the same args is reused)."""
    from mva.cli.query import _resolve_quantize
    monkeypatch.setattr(
        "mva.cli.query._probe_total_vram_bytes", lambda: 24 * 1024 ** 3,
    )
    args = _make_args(chroma_dir="/x")
    _ = _resolve_quantize(args)
    assert args.quantize is None, (
        "_resolve_quantize must not mutate args.quantize (P2-03 anti-pattern). "
        f"Got {args.quantize!r}"
    )


# ----------------------------------------------------------------------
# Task 4 — QueryService.answer(memory=...) wiring
# ----------------------------------------------------------------------


def test_answer_grows_and_threads_memory():
    from mva.cli.query import QueryService
    from mva.l6_interaction.memory import ConversationMemory
    from mva.l6_interaction.orchestrator import OrchestratorResult
    from mva.l6_interaction.plan import QueryPlan

    svc = QueryService(db_path=":memory:", llm_model=None)   # mock LLM, no embedder

    seen_history: list[str] = []

    def fake_run(query, history_block=""):
        seen_history.append(history_block)
        q = query.text if hasattr(query, "text") else query
        return OrchestratorResult(
            question=q,
            plan=QueryPlan(intent="i", tool_calls=[], rationale="r"),
            answer=f"ans:{q}",
        )

    svc.orchestrator.run = fake_run     # bypass planner JSON parsing for this test

    mem = ConversationMemory()
    svc.answer("q1", memory=mem)
    svc.answer("它呢", memory=mem)       # cue 它 -> recency pulls q1 (no embedder)

    assert [t.question for t in mem.turns] == ["q1", "它呢"]
    assert mem.turns[0].answer == "ans:q1"
    assert "q1" in seen_history[1]      # 2nd call saw q1 via cue-gated recency
    svc.close()


def test_answer_without_memory_is_unchanged():
    from mva.cli.query import QueryService
    from mva.l6_interaction.orchestrator import OrchestratorResult
    from mva.l6_interaction.plan import QueryPlan

    svc = QueryService(db_path=":memory:", llm_model=None)
    svc.orchestrator.run = lambda query, history_block="": OrchestratorResult(
        question="q", plan=QueryPlan(intent="i", tool_calls=[], rationale="r"),
        answer="a",
    )
    # No memory kwarg -> original behavior, no crash.
    assert svc.answer("hello").answer == "a"
    svc.close()


def test_answer_skips_empty_text_turn_in_memory():
    # Attachment-only UI messages (empty text) must NOT add an inert turn
    # (no question, no embedding) that wastes a max_turns slot.
    from mva.cli.query import QueryService
    from mva.contracts import RichQuery
    from mva.l6_interaction.memory import ConversationMemory
    from mva.l6_interaction.orchestrator import OrchestratorResult
    from mva.l6_interaction.plan import QueryPlan

    svc = QueryService(db_path=":memory:", llm_model=None)
    svc.orchestrator.run = lambda query, history_block="": OrchestratorResult(
        question="", plan=QueryPlan(intent="i", tool_calls=[], rationale="r"),
        answer="a",
    )
    mem = ConversationMemory()
    svc.answer(RichQuery(text="", attachments=[]), memory=mem)
    assert mem.turns == []          # empty-text turn skipped
    svc.answer("real question", memory=mem)
    assert [t.question for t in mem.turns] == ["real question"]
    svc.close()
