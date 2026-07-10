"""L6 tool registry: callable handles the Query Planner can request.

Two categories of tools:

  Generic SQL:  query_db — LLM writes its own SELECT against DuckDB.
                Replaces the previous 10+ specialized wrapper tools.
  Multimodal:   find_segment_by_description, find_bbox_by_description,
                find_by_description — ChromaDB vector search (cannot
                be replaced by SQL).
  Attachment:   describe_attachment, find_similar_to_attachment,
                compare_attachments — registered per-query by the
                Orchestrator when the user attaches files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from mva.l5_state import VectorStore, WorldStateStore


@dataclass
class ToolSpec:
    """One tool registered against the orchestrator.

    `render` (G-1) is an optional deterministic answer-templater for the
    simple case: `render(result, args) -> str | None`. When the planner
    issues exactly ONE call to a tool whose render returns a string, the
    orchestrator returns that string directly and skips the answer LLM —
    faster and literally un-hallucinatable (the number printed is the one
    the tool computed). Returning None means "needs LLM composition".
    """

    name: str
    description: str
    fn: Callable[..., Any]
    render: Optional[Callable[[Any, dict], Optional[str]]] = None


class ToolRegistry:
    """Dict-of-tools with a description rendering for prompt injection."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def call(self, name: str, args: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name!r}")
        return self._tools[name].fn(**args)

    def has_render(self, name: str) -> bool:
        return name in self._tools and self._tools[name].render is not None

    def render(self, name: str, result: Any, args: dict[str, Any]) -> Optional[str]:
        """Template the simple-case answer for `name`, or None to defer to the LLM."""
        spec = self._tools.get(name)
        if spec is None or spec.render is None:
            return None
        return spec.render(result, args)

    def descriptions(self) -> str:
        """Render the tool list for embedding into an LLM prompt."""
        return "\n".join(
            f"- {spec.name}: {spec.description}"
            for spec in self._tools.values()
        )


def build_default_registry(
    store: WorldStateStore,
    vstore: Optional[VectorStore] = None,
    llm: Optional[Any] = None,
    enable_cross_view: bool = True,
) -> ToolRegistry:
    """Wire L6 tools onto the supplied L5 stores.

    Core tool: `query_db` — the LLM writes read-only SQL against DuckDB
    to answer any structured data question (counts, metadata, cross-view
    links, filtering, aggregation).

    Vector tools: `find_*_by_description` — semantic / multimodal search
    that cannot be expressed as SQL (requires ChromaDB embeddings).
    """
    reg = ToolRegistry()

    # G-1: typed structured tools (count_objects / list_objects / which_views /
    # when_seen / objects_at_time / cross_view_matches / scene_stats). These are
    # the PRIMARY path for structured questions — the planner picks one by name
    # and fills typed slots; the tool runs correct, JSON-aware queries internally
    # so the 7B never authors (and never mis-authors) SQL. Each ships a render()
    # so the orchestrator can template the answer and skip the answer LLM.
    from mva.l6_interaction.structured_tools import register_structured_tools
    register_structured_tools(reg, store)

    reg.register(ToolSpec(
        name="query_db",
        description=(
            "【兜底】仅当上面的 typed 工具都不适用、需要自定义聚合时才用。"
            "Execute a read-only SQL SELECT against DuckDB. "
            "NOTE: object class lives inside the `bboxes` JSON of tracklets_*, "
            "NOT a column — prefer count_objects/list_objects over hand-written SQL. "
            "args: sql (str, required). Returns a list of row dicts."
        ),
        fn=lambda sql: store.execute_readonly(sql),
    ))

    if vstore is not None:
        reg.register(ToolSpec(
            name="find_by_description",
            description=(
                "Free-text vector search over the multimodal embedding index. "
                "Backed by Qwen3-VL-Embedding-8B (cross-modal: text query "
                "matches image-embedded ROI crops). Use for semantic search "
                "like '找穿红衣服的人' / 'find a red car'. "
                "args: text (str, required); "
                "view_id (str, optional — null = all views); "
                "top_k (int, default 5); "
                "max_distance (float, optional — drop hits above this)."
            ),
            fn=lambda text, view_id=None, top_k=5, max_distance=None, **_: (
                _filter_by_distance(
                    vstore.query(
                        query_text=text,
                        vector_type=None,
                        view_id=view_id,
                        top_k=top_k,
                    ),
                    max_distance,
                )
            ),
        ))

        reg.register(ToolSpec(
            name="find_segment_by_description",
            description=(
                "Free-text search over video segments (10s windows). "
                "Returns ChromaDB hits enriched with DuckDB segment metadata "
                "(source_uri, start_t, end_t, detected_classes, detected_counts). "
                "Use for time-localization questions like '什么时候出现了 X'. "
                "args: text (str, required); "
                "view_id (str, optional); "
                "top_k (int, default 5); "
                "max_distance (float, optional)."
            ),
            fn=lambda text, view_id=None, top_k=5, max_distance=None, **_: (
                _filter_by_distance(
                    _find_segments(store, vstore, text, view_id, top_k),
                    max_distance,
                )
            ),
        ))

        reg.register(ToolSpec(
            name="find_bbox_by_description",
            description=(
                "Free-text search over per-detection bbox crops. "
                "Returns hits with class_name + bbox coords + parent segment. "
                "Use when you need object-level granularity. "
                "args: text (str, required); top_k (int, default 5); "
                "max_distance (float, optional)."
            ),
            fn=lambda text, top_k=5, max_distance=None, **_: (
                _filter_by_distance(
                    vstore.query(
                        query_text=text, vector_type="reid", top_k=top_k,
                    ),
                    max_distance,
                )
            ),
        ))

    # G-2: VLM-native `look_at` (retrieve segment → Qwen-VL reads real frames).
    # Needs both vstore (retrieval) and llm (vision). No-op if either is absent
    # (e.g. mock query path) — register_vlm_tools guards internally.
    from mva.l6_interaction.vlm_tools import register_vlm_tools
    register_vlm_tools(reg, store, vstore, llm)

    return reg


# ----------------------------------------------------------------------
# Attachment-bound tools (registered per-query when RichQuery has files)
# ----------------------------------------------------------------------


def register_attachment_tools(
    registry: ToolRegistry,
    attachments: list,
    llm: Any = None,
    embedder: Any = None,
    vstore: Any = None,
) -> None:
    """Add tools that operate on the current query's attachments."""
    if not attachments:
        return

    n = len(attachments)
    catalog = "; ".join(
        f"idx={i} kind={a.kind} path={a.path.name}"
        for i, a in enumerate(attachments)
    )

    if llm is not None:
        registry.register(ToolSpec(
            name="describe_attachment",
            description=(
                f"Use the VLM to describe one of the user's attached files. "
                f"Currently attached: {catalog}. "
                f"args: idx (int, 0..{n-1})."
            ),
            fn=lambda idx: _describe_attachment(llm, attachments[int(idx)]),
        ))

    if embedder is not None and vstore is not None:
        registry.register(ToolSpec(
            name="find_similar_to_attachment",
            description=(
                f"Encode an attachment via Qwen3-VL-Embedding-8B then search "
                f"the multimodal index for the most similar stored items "
                f"across all views. "
                f"Currently attached: {catalog}. "
                f"args: idx (int, 0..{n-1}); top_k (int, default 5)."
            ),
            fn=lambda idx, top_k=5: _find_similar_to_attachment(
                embedder, vstore, attachments[int(idx)], int(top_k),
            ),
        ))

    if embedder is not None and n >= 2:
        registry.register(ToolSpec(
            name="compare_attachments",
            description=(
                "Cosine similarity between two attachments' embeddings. "
                f"args: idx_a (int), idx_b (int)  — both in 0..{n-1}."
            ),
            fn=lambda idx_a, idx_b: _compare_attachments(
                embedder, attachments[int(idx_a)], attachments[int(idx_b)],
            ),
        ))


def _describe_attachment(llm: Any, attachment) -> str:
    if attachment.kind == "image":
        import cv2
        img = cv2.imread(str(attachment.path))
        if img is None:
            return f"[ERROR] could not read image {attachment.path}"
        return llm.complete("请用一两句话描述这张图片。", images=[img])
    if attachment.kind == "video":
        import cv2
        cap = cv2.VideoCapture(str(attachment.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total == 0:
            cap.release()
            return f"[ERROR] could not read video {attachment.path}"
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, img = cap.read()
        cap.release()
        if not ok:
            return f"[ERROR] could not read mid-frame of {attachment.path}"
        return llm.complete(
            "这是用户上传视频的中间一帧，请用一两句话描述。", images=[img]
        )
    return f"[ERROR] unsupported attachment kind: {attachment.kind!r}"


def _find_similar_to_attachment(embedder, vstore, attachment, top_k: int):
    if attachment.kind == "image":
        import cv2
        img = cv2.imread(str(attachment.path))
        if img is None:
            return [{"error": f"could not read image {attachment.path}"}]
        vec = embedder.encode_image(img)
    elif attachment.kind == "video":
        from mva.datasets.mvu_eval import _sample_video_frames
        frames = _sample_video_frames(attachment.path, n=8)
        if not frames:
            return [{"error": f"could not read video {attachment.path}"}]
        vec = embedder.encode_images(frames)
    else:
        return [{"error": f"unsupported attachment kind: {attachment.kind!r}"}]
    return vstore.query(query_vector=vec, top_k=top_k)


def _compare_attachments(embedder, a, b) -> dict:
    import cv2
    import numpy as np

    def _encode(att):
        if att.kind == "image":
            img = cv2.imread(str(att.path))
            if img is None:
                return None
            return embedder.encode_image(img)
        from mva.datasets.mvu_eval import _sample_video_frames
        frames = _sample_video_frames(att.path, n=8)
        if not frames:
            return None
        return embedder.encode_images(frames)

    va = _encode(a)
    vb = _encode(b)
    if va is None or vb is None:
        return {"error": "could not encode one or both attachments",
                "a": a.path.name, "b": b.path.name}

    va = np.asarray(va)
    vb = np.asarray(vb)
    sim = float(np.dot(va, vb))
    return {"cosine_similarity": sim, "a": a.path.name, "b": b.path.name}


# --------------------------------------------------------------------------
# M2.8 segment-level retrieval helpers
# --------------------------------------------------------------------------


def _find_segments(
    store: WorldStateStore,
    vstore: VectorStore,
    text: str,
    view_id: Optional[str],
    top_k: int,
) -> list[dict]:
    """Free-text search over segment vectors + enrich each hit with the
    DuckDB segments row."""
    hits = vstore.query(
        query_text=text, vector_type="frame",
        view_id=view_id, top_k=top_k,
    )
    enriched: list[dict] = []
    for hit in hits:
        md = hit.get("metadata", {}) or {}
        if md.get("vector_kind") != "segment":
            continue
        seg = store.get_segment_by_chroma_id(hit["id"])
        enriched.append({**hit, "segment": seg})
    return enriched


# --------------------------------------------------------------------------
# M3.8 (PROBLEMS P2-12) — distance gating
# --------------------------------------------------------------------------

WEAK_MATCH_DISTANCE_THRESHOLD = 0.85


def _filter_by_distance(
    hits: list[dict], max_distance: Optional[float],
) -> list[dict]:
    if max_distance is None:
        return hits
    return [
        h for h in hits
        if h.get("distance") is None
        or h["distance"] <= float(max_distance)
    ]
