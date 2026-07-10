"""L6 conversation memory — session-scoped, content-gated recall.

Lives in the UI's gr.State (one instance per browser session); NEVER stored on
the QueryService singleton (shared across sessions via a lock — storing memory
there would let two tabs corrupt each other's context).

Design (2026-06-01 plan, cue-gated revision):
  - Each completed turn is stored with its QUESTION embedding (reused from the
    vector QueryService already computes for selection — exactly one encode per
    turn is dedicated to memory).
  - select_relevant(query_text, query_embedding) decides PER QUERY whether to
    include history, via two independent gates — returns turns in chronological
    order:
      * RECENCY (coreference): the last `recent_keep` turns are included ONLY
        when the query text carries a referential / continuation cue
        (它/刚才/再/…/sentence-final 呢). "它呢" is a follow-up but is
        semantically DISSIMILAR to its antecedent — so recency must be gated by
        a textual cue, never by similarity.
      * RELEVANCE (topic recall): the top `retrieve_top_m` turns over ALL turns
        by cosine ≥ `min_score`. Self-gating — a fresh, unrelated query matches
        nothing and contributes no history.
    A self-contained query (no cue + nothing similar) → empty history → the
    planner gets a clean context.
  - With no query embedding (mock embedder / no --chroma-dir), only the cue gate
    runs: cued queries still get recency; uncued queries get empty history.
  - Embeddings are L2-normalized (MultimodalEmbedder contract) → cosine == dot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_MAX_ANSWER_CHARS = 200   # truncate long answers in the rendered history block

# Referential / continuation cues that mark a query as a follow-up needing the
# recent turns. PRECISION-leaning: anaphoric pronouns + explicit back-references
# + safe continuations. Ambiguous demonstratives (这个/那个 — usually
# determiners, e.g. "这个场景里有几辆车") are deliberately EXCLUDED to avoid
# dragging history into self-contained queries (the whole point of gating).
_REFERENTIAL_CUES = (
    "它", "他", "她", "它们", "他们", "她们",       # anaphoric pronouns
    "上一个", "上述", "上面", "前面", "之前",        # explicit back-reference
    "刚才", "刚刚",
    "再", "继续", "同样", "同上",                    # continuation
)


def _has_referential_cue(text: str) -> bool:
    """Cheap heuristic: does the query look like a follow-up needing recency?

    True on any cue in `_REFERENTIAL_CUES`, or a sentence-final 呢 ("行人呢?").
    Errs toward recall: a false positive only injects recent turns the answer
    prompt is told to treat as reference-only; a false negative drops
    coreference. Microsecond cost (substring scan)."""
    if not text:
        return False
    if text.rstrip().rstrip("?？").endswith("呢"):
        return True
    return any(cue in text for cue in _REFERENTIAL_CUES)


@dataclass
class ConversationTurn:
    question: str
    answer: str
    embedding: Optional[list[float]] = None   # L2-normalized question vector


@dataclass
class ConversationMemory:
    recent_keep: int = 2
    retrieve_top_m: int = 3
    max_turns: int = 100
    min_score: float = 0.70        # relevance self-gate cosine threshold.
    #   Calibrated on VisDrone scene-26 (2026-06-01 dogfood): Qwen3-VL-Emb
    #   text-text cosine runs HIGH — loosely-related/unrelated query pairs land
    #   0.52-0.64, genuine same-topic recall lands 0.72-0.85. 0.70 cleanly
    #   separates them (self-contained queries → empty history). Lower it if a
    #   real topic-recall follow-up isn't pulled back; raise it if unrelated
    #   turns leak in. Per-deployment tunable.
    turns: list[ConversationTurn] = field(default_factory=list)

    def add_turn(
        self,
        question: str,
        answer: str,
        embedding: Optional[list[float]] = None,
    ) -> None:
        self.turns.append(ConversationTurn(question, answer, embedding))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def select_relevant(
        self,
        query_text: str,
        query_embedding: Optional[list[float]] = None,
    ) -> list[ConversationTurn]:
        n = len(self.turns)
        if n == 0:
            return []
        selected: set[int] = set()

        # RELEVANCE gate (self-gating): cosine top-M over ALL turns ≥ min_score.
        if query_embedding is not None:
            scored: list[tuple[int, float]] = []
            for i, turn in enumerate(self.turns):
                if turn.embedding is None:
                    continue
                scored.append((i, _dot(query_embedding, turn.embedding)))
            scored.sort(key=lambda pair: pair[1], reverse=True)
            for i, score in scored[: self.retrieve_top_m]:
                if score >= self.min_score:
                    selected.add(i)

        # RECENCY gate (cue-gated): only follow-up queries pull the recent turns.
        if self.recent_keep > 0 and _has_referential_cue(query_text):
            keep = min(self.recent_keep, n)
            selected.update(range(n - keep, n))

        return [self.turns[i] for i in sorted(selected)]

    @staticmethod
    def render_block(turns: list[ConversationTurn]) -> str:
        if not turns:
            return ""
        lines = ["[对话历史 — 供指代消解与上下文承接，不是工具返回的数据]"]
        for t in turns:
            ans = t.answer
            if len(ans) > _MAX_ANSWER_CHARS:
                ans = ans[:_MAX_ANSWER_CHARS] + "…"
            lines.append(f"用户: {t.question}")
            lines.append(f"助手: {ans}")
        return "\n".join(lines)


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product == cosine for L2-normalized vectors. Pure-python (dim≈768,
    a handful of calls per turn → microseconds; keeps this module numpy-free)."""
    return sum(x * y for x, y in zip(a, b))
