import math

from mva.l6_interaction.memory import ConversationMemory, ConversationTurn


def _unit(*vals: float) -> list[float]:
    """Build an L2-normalized vector from leading values (mirrors embedder contract)."""
    norm = math.sqrt(sum(x * x for x in vals)) or 1.0
    return [x / norm for x in vals]


def test_empty_memory_returns_empty():
    m = ConversationMemory()
    assert m.select_relevant("它呢", [1.0, 0.0]) == []


def test_referential_cue_detection():
    from mva.l6_interaction.memory import _has_referential_cue
    assert _has_referential_cue("它呢")
    assert _has_referential_cue("再数一次")
    assert _has_referential_cue("刚才那个在哪")
    assert _has_referential_cue("行人呢？")           # sentence-final 呢
    assert not _has_referential_cue("scene-26 有几辆车")
    assert not _has_referential_cue("D1 视角行人数量")
    assert not _has_referential_cue("")


def test_self_contained_query_gets_no_recency():
    # No referential cue + nothing similar enough → clean, history-free.
    m = ConversationMemory(recent_keep=2, retrieve_top_m=3, min_score=0.5)
    m.add_turn("红车在哪", "在 D1", embedding=_unit(1, 0))
    m.add_turn("天气如何", "晴", embedding=_unit(0, 1))
    sel = m.select_relevant("场景里有几个行人", _unit(0, 0, 1))   # orthogonal, no cue
    assert sel == []


def test_referential_cue_pulls_recency():
    m = ConversationMemory(recent_keep=2, retrieve_top_m=0)   # relevance off
    m.add_turn("q1", "a1", embedding=_unit(1, 0))
    m.add_turn("红车在哪", "在 D1", embedding=_unit(0, 1))
    sel = m.select_relevant("它呢", _unit(0, 0, 1))           # cue 它 → recency in
    assert [t.question for t in sel] == ["q1", "红车在哪"]    # last recent_keep=2


def test_cue_recency_works_without_embedding():
    # Coreference path must work even with no embedder (mock / no --chroma-dir).
    m = ConversationMemory(recent_keep=1)
    m.add_turn("红车在哪", "在 D1")
    sel = m.select_relevant("再数一次", None)                # cue 再 → recency, no emb
    assert [t.question for t in sel] == ["红车在哪"]


def test_no_cue_no_embedding_is_empty():
    m = ConversationMemory(recent_keep=2)
    m.add_turn("红车在哪", "在 D1")
    m.add_turn("行人几个", "5 个")
    sel = m.select_relevant("场景里有公交车吗", None)        # no cue, no emb → empty
    assert sel == []


def test_relevance_retrieves_related_old_turn_without_cue():
    # Topic recall: a self-contained query (no cue) that relates to an OLD turn
    # still pulls it via cosine; the dissimilar recent turn is NOT forced in.
    m = ConversationMemory(recent_keep=1, retrieve_top_m=1, min_score=0.3)
    m.add_turn("数了多少辆车", "9 辆", embedding=_unit(1, 0))    # idx0
    m.add_turn("天气怎样", "晴", embedding=_unit(0, 1))          # idx1 recent, no cue
    sel = m.select_relevant("D2 视角车辆总数", _unit(1, 0.05))   # no cue, close to idx0
    qs = [t.question for t in sel]
    assert "数了多少辆车" in qs                  # pulled by relevance
    assert "天气怎样" not in qs                  # recent NOT forced (no cue) + dissimilar


def test_relevance_and_cue_recency_merge_chronologically():
    m = ConversationMemory(recent_keep=1, retrieve_top_m=1, min_score=0.3)
    m.add_turn("数车", "9", embedding=_unit(1, 0))           # idx0 (relevance)
    m.add_turn("无关", "y", embedding=_unit(0, 1))           # idx1
    m.add_turn("它呢", "...", embedding=_unit(0, 0, 1))      # idx2 (recent + cue)
    sel = m.select_relevant("它的数量 D2", _unit(1, 0.05))   # cue 它 + close to idx0
    qs = [t.question for t in sel]
    # idx0 by relevance + idx2 by cue-recency → chronological order
    assert qs == ["数车", "它呢"]


def test_min_score_drops_weak_matches():
    m = ConversationMemory(recent_keep=0, retrieve_top_m=2, min_score=0.5)
    m.add_turn("orthogonal", "x", embedding=_unit(0, 1))
    sel = m.select_relevant("something", _unit(1, 0))   # score 0 < 0.5, no cue
    assert sel == []


def test_render_block_format_and_truncation():
    turns = [
        ConversationTurn("有几辆车", "一共 9 辆车。"),
        ConversationTurn("行人呢", "x" * 250),
    ]
    block = ConversationMemory.render_block(turns)
    assert "[对话历史" in block
    assert "用户: 有几辆车" in block
    assert "助手: 一共 9 辆车。" in block
    assert "用户: 行人呢" in block
    assert "…" in block                         # long answer truncated


def test_render_block_empty():
    assert ConversationMemory.render_block([]) == ""


def test_max_turns_cap_drops_oldest():
    m = ConversationMemory(max_turns=2)
    m.add_turn("q1", "a1")
    m.add_turn("q2", "a2")
    m.add_turn("q3", "a3")
    assert [t.question for t in m.turns] == ["q2", "q3"]


def test_calibrated_defaults_pinned():
    # The min_score default is hard-won: the VisDrone scene-26 dogfood
    # (eval/memory-dogfood.md) caught min_score=0.35 leaking unrelated turns
    # (cosine 0.52-0.64) into self-contained queries → wrong answers. Pin the
    # calibrated defaults so a silent revert is caught by CI.
    m = ConversationMemory()
    assert m.min_score == 0.70
    assert m.recent_keep == 2
    assert m.retrieve_top_m == 3
    assert m.max_turns == 100
