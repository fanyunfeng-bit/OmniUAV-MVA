"""L6 Mode A Orchestrator — the Reactive NL→answer loop.

Flow per PLAN.md §3.2 L6:
    NL Query → Query Planner → tool calls → result aggregation → NL Answer

Accepts either a plain `str` question or a `RichQuery` (text + multimodal
attachments). When attachments are present, an extra layer of
attachment-specific tools is registered for this run only — the base
registry passed at construction is left untouched.

Disambiguation: when a tool returns multiple results above a similarity
threshold (e.g. find_by_description with 2 candidates), the orchestrator
opens a clarification round. In the current sync API this surfaces as a
`needs_disambiguation` result returned to the caller; the user-facing CLI
shell decides how to ask.

Failure handling per §3.5: tool errors are caught + reported to the answer
prompt as `ERROR: ...` so the LLM can phrase a polite fallback. Hard
exceptions only escape on planner-parse failure (handled by caller).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from mva.contracts import Attachment, RichQuery
from mva.l6_interaction.plan import QueryPlan
from mva.l6_interaction.planner import QueryPlanner
from mva.l6_interaction.tools import (
    ToolRegistry,
    register_attachment_tools,
)


ANSWER_PROMPT_TEMPLATE = """你是多路视频态势助理（L6 Mode A）。
{history_block}
用户问题：{question}
{attachment_block}

数据库概况：
{db_context}

工具调用与结果：
{results}

请用简洁的中文回答用户的问题。

规则（**grounding 优先，宁可说没有，绝不编造**）：
1. 只能依据"工具调用与结果"里的**实际数据**作答。数字、类别、视角、时间一律必须来自
   结果，**严禁编造或脑补**任何工具没给出的数值/类别（例如工具没返回数量，就不要自己
   编一个数）。
2. 需要的数据没拿到（工具返回空 []、ERROR，或结果里根本没有该信息）→ 如实说
   "没有找到 X" 或 "根据当前数据无法确认"，**不要猜数字**。
3. find_segment_by_description 按语义相似度排序：第一个 hit 的 view_id / 时间窗可作为
   "哪个视频/什么时候有 X" 类**定位**问题的答案；但它**不能**用来报具体数量或类别计数。
4. cross_view_links / cross_view_matches 的一行代表两个视角共享同一目标。
5. [对话历史]（若有）只用于理解指代与承接，不要把旧结论当成本轮事实；本轮事实以
   "工具调用与结果"为准。
6. 用简洁中文回答。
"""


@dataclass
class ToolInvocation:
    tool: str
    args: dict[str, Any]
    result: Any = None
    error: str = ""


@dataclass
class OrchestratorResult:
    question: str
    plan: QueryPlan
    invocations: list[ToolInvocation] = field(default_factory=list)
    answer: str = ""
    needs_disambiguation: bool = False
    disambiguation_candidates: list[Any] = field(default_factory=list)
    # 🔌 future multi-turn / ReAct hook (M3+): planner emits this when a
    # second pass should refine the plan based on intermediate tool results.
    # None in single-turn current behavior.
    next_step_hint: Optional[str] = None


class Orchestrator:
    """L6 Mode A entry point: orchestrate(NL | RichQuery) → OrchestratorResult."""

    def __init__(
        self,
        llm: Any,
        planner: QueryPlanner,
        registry: ToolRegistry,
        disambiguation_threshold: int = 2,
        embedder: Any = None,
        vstore: Any = None,
        db_context: str = "",
    ) -> None:
        self.llm = llm
        self.planner = planner
        self.registry = registry
        self.disambiguation_threshold = disambiguation_threshold
        self.embedder = embedder
        self.vstore = vstore
        self.db_context = db_context

    def run(self, query: str | RichQuery, history_block: str = "") -> OrchestratorResult:
        rich = query if isinstance(query, RichQuery) else RichQuery(text=query)
        question = rich.text

        # If attachments present, build a per-call registry that adds the
        # describe_attachment / find_similar_to_attachment / compare_attachments
        # tools bound to THIS query's files. Base registry untouched.
        active_registry = self.registry
        if rich.attachments:
            active_registry = self._registry_for_query(rich.attachments)
            print(f"[orchestrator] attachments={len(rich.attachments)} "
                  f"tools={active_registry.names()}")

        plan = self._plan_with_attachments(
            question, rich.attachments, active_registry, history_block,
        )
        print(f"[orchestrator] plan: intent={plan.intent} "
              f"tool_calls={[(c.tool, c.args) for c in plan.tool_calls]}")
        result = OrchestratorResult(question=question, plan=plan)

        # --- G-1 FAST PATH: single renderable typed tool → templated answer ---
        # If the planner issued exactly one call to a tool with a render() and
        # that render produces a string, return it directly: skip both the
        # auto-inject vector search AND the answer LLM. Faster and literally
        # un-hallucinatable (the answer IS the number the tool computed). On
        # tool error or render→None we fall through to the full LLM path.
        fast = self._try_fast_path(plan, active_registry, rich)
        if fast is not None:
            inv, rendered = fast
            result.invocations.append(inv)
            result.answer = rendered
            print(f"[orchestrator] fast-path {inv.tool} → templated answer (no LLM)")
            return result

        # --- AUTO-INJECT: always-inject policy for 7B model workaround ---
        #
        # WHY (both blocks below): Qwen2.5-VL-7B cannot reliably choose
        # the correct tool from the registry. It defaults to query_db for
        # almost every question, even when vector search is clearly needed.
        # These auto-inject blocks guarantee the right tools fire regardless
        # of planner output.
        #
        # TRADEOFF: ~50-100ms extra per query (one vector search). The
        # answer LLM ignores irrelevant results, so quality is unaffected.
        #
        # FUTURE: When upgrading to a 14B+ model that reliably picks tools,
        # both auto-inject blocks can be removed — the planner will do the
        # right thing. If planner also picks the same tool, results are
        # identical (same embedding, same index), no harm from duplication.

        # 1) Attachment search: encode uploaded image/video → find similar
        if rich.attachments and "find_similar_to_attachment" in active_registry:
            for i in range(len(rich.attachments)):
                inv = ToolInvocation(
                    tool="find_similar_to_attachment",
                    args={"idx": i, "top_k": 5},
                )
                try:
                    inv.result = active_registry.call(inv.tool, inv.args)
                    print(f"[orchestrator] (auto-inject) "
                          f"find_similar_to_attachment idx={i} → "
                          f"{len(inv.result or [])} hits")
                except Exception as exc:
                    inv.error = f"{type(exc).__name__}: {exc}"
                result.invocations.append(inv)

        # 2) Segment search: use the question text as semantic query to find
        #    relevant video segments. Covers "which video has X" / "找到有Y的
        #    画面" style questions where SQL on detected_classes fails (YOLO
        #    class names don't cover all concepts, e.g. "bucket").
        if "find_segment_by_description" in active_registry:
            inv = ToolInvocation(
                tool="find_segment_by_description",
                args={"text": question, "top_k": 5},
            )
            try:
                inv.result = active_registry.call(inv.tool, inv.args)
                print(f"[orchestrator] (auto-inject) "
                      f"find_segment_by_description → "
                      f"{len(inv.result or [])} hits")
            except Exception as exc:
                inv.error = f"{type(exc).__name__}: {exc}"
            result.invocations.append(inv)

        # Dispatch planner's tool calls
        for call in plan.tool_calls:
            inv = ToolInvocation(tool=call.tool, args=call.args)
            try:
                inv.result = active_registry.call(call.tool, call.args)
                print(f"[orchestrator] {call.tool} → {inv.result!r:.200}")
            except Exception as exc:
                inv.error = f"{type(exc).__name__}: {exc}"
            result.invocations.append(inv)

        # Disambiguation: find_by_description with 2 close candidates → flag
        for inv in result.invocations:
            if inv.tool == "find_by_description" and isinstance(inv.result, list):
                if len(inv.result) >= self.disambiguation_threshold:
                    distances = [
                        r.get("distance") for r in inv.result if isinstance(r, dict)
                    ]
                    distances = [d for d in distances if d is not None]
                    if (
                        len(distances) >= 2
                        and abs(distances[0] - distances[1]) < 0.05
                    ):
                        result.needs_disambiguation = True
                        result.disambiguation_candidates = inv.result[
                            : self.disambiguation_threshold
                        ]
                        break

        # NL answer synthesis
        rendered = self._render_invocations(result.invocations)
        attachment_block = self._format_attachment_block(rich.attachments)
        answer_prompt = ANSWER_PROMPT_TEMPLATE.format(
            question=question,
            history_block=history_block,
            attachment_block=attachment_block,
            db_context=self.db_context or "(empty)",
            results=rendered,
        )
        result.answer = self.llm.complete(answer_prompt)
        return result

    # ------------------------------------------------------------------
    # G-1 fast path
    # ------------------------------------------------------------------

    def _try_fast_path(
        self, plan: QueryPlan, registry: ToolRegistry, rich: RichQuery,
    ) -> Optional[tuple[ToolInvocation, str]]:
        """If plan is exactly one renderable typed tool that returns a string,
        run it and return (invocation, templated_answer). Else None (caller
        falls back to the full auto-inject + answer-LLM path)."""
        if rich.attachments or len(plan.tool_calls) != 1:
            return None
        call = plan.tool_calls[0]
        if not registry.has_render(call.tool):
            return None
        inv = ToolInvocation(tool=call.tool, args=call.args)
        try:
            inv.result = registry.call(call.tool, call.args)
        except Exception:
            return None  # error → full path handles it (fail-closed answer)

        # G-2 cascade: a typed structured tool that came back EMPTY *for an
        # existence/identification question* → double-check the pixels with
        # look_at (detection recall is limited, so empty might be a miss). A
        # plain count of 0 is trusted — wants_visual_check excludes "有几/多少".
        from mva.l6_interaction.vlm_tools import (
            is_empty_structured, wants_visual_check,
        )
        if (is_empty_structured(call.tool, inv.result)
                and wants_visual_check(rich.text)
                and "look_at" in registry):
            vlm = registry.call("look_at", {"text": rich.text})
            vlm_answer = registry.render("look_at", vlm, {"text": rich.text})
            if (isinstance(vlm, dict) and not vlm.get("abstained")
                    and isinstance(vlm_answer, str)):
                print("[orchestrator] cascade: typed tool empty → look_at "
                      "grounded answer (根据画面判断)")
                vinv = ToolInvocation(
                    tool="look_at", args={"text": rich.text}, result=vlm)
                return vinv, vlm_answer
            # VLM also abstained → fall through to the honest typed "没有" render

        rendered = registry.render(call.tool, inv.result, call.args)
        if not isinstance(rendered, str):
            return None
        return inv, rendered

    # ------------------------------------------------------------------
    # Per-query registry build
    # ------------------------------------------------------------------

    def _registry_for_query(self, attachments: list[Attachment]) -> ToolRegistry:
        """Return a new registry = base registry + attachment-bound tools."""
        from mva.l6_interaction.tools import ToolRegistry as _TR
        scratch = _TR()
        # Copy base tool specs
        for name in self.registry.names():
            scratch.register(self.registry._tools[name])     # type: ignore[attr-defined]
        # Add attachment tools (no-op if embedder / llm is None — described as unavailable)
        register_attachment_tools(
            scratch, attachments,
            llm=self.llm,
            embedder=self.embedder,
            vstore=self.vstore,
        )
        return scratch

    # ------------------------------------------------------------------
    # Planner prompt extension for attachments
    # ------------------------------------------------------------------

    def _plan_with_attachments(
        self,
        question: str,
        attachments: list[Attachment],
        registry: ToolRegistry,
        history_block: str = "",
    ) -> QueryPlan:
        """Run the planner with an optional attachment summary appended.

        The base planner only sees the question + tool descriptions. When
        attachments are present we pre-pend a short summary to the question
        so the LLM can pick attachment-bound tools."""
        if not attachments:
            return self.planner.plan(question, history_block=history_block)
        # Temporarily switch the planner's registry to our active one so its
        # tool list reflects the bound attachment tools.
        original_registry = self.planner.registry
        self.planner.registry = registry
        try:
            summary = _format_attachment_summary(attachments)
            framed = f"{question}\n\n[已附加文件]\n{summary}"
            return self.planner.plan(framed, history_block=history_block)
        finally:
            self.planner.registry = original_registry

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_attachment_block(attachments: list[Attachment]) -> str:
        if not attachments:
            return ""
        return "\n[用户附件]\n" + _format_attachment_summary(attachments)

    @staticmethod
    def _render_invocations(invocations: list[ToolInvocation]) -> str:
        if not invocations:
            return "(planner produced no tool calls)"
        lines: list[str] = []
        for inv in invocations:
            if inv.error:
                lines.append(f"- {inv.tool}({inv.args}) → ERROR: {inv.error}")
                continue
            rendered = _render_tool_result(inv.result)
            lines.append(f"- {inv.tool}({inv.args}) → {rendered}")
        return "\n".join(lines)


def _format_attachment_summary(attachments: list[Attachment]) -> str:
    """Render a brief enumeration of attachments for prompt injection."""
    lines = []
    for idx, att in enumerate(attachments):
        label = f" ({att.label})" if att.label else ""
        lines.append(f"  - idx={idx}  kind={att.kind}{label}  path={att.path.name}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Result rendering (M2.8: type-aware to preserve M2.8 segment enrichment)
# --------------------------------------------------------------------------

_MAX_INLINE_ITEMS = 5      # list head shown verbatim before "and N more..."
_MAX_REPR_CHARS = 500      # fallback truncation for scalar/dict payloads


def _render_tool_result(result: Any) -> str:
    """Render a tool result for the answer-synthesis prompt.

    Special cases avoid the M2.7-style blanket `repr() + truncate to 500`:

    1. **None** → explicit no-signal annotation (M3.9 P3-13 fix; was
       rendered as bare 'None' which Qwen sometimes interpreted as
       "the result happened to be None, fine to guess").
    2. **Empty string** → same — describe_scene / query_captions can
       legitimately return "" when there are no captions indexed.
    3. **Empty list** → explicit "[] (no matches — do not invent)" so
       the answer prompt's anti-hallucination rule has a token to grip.
    4. **List of dicts** (typical of find_*_by_description) → report
       length + render each of the first K hits compactly. M2.8 segment
       hits surface (source_uri, start_t, end_t, detected_classes)
       prominently instead of being buried in repr().
    5. **Anything else** → `repr` + 500-char fallback truncation.

    Fixes PROBLEMS P1-05 (M2.8) + P3-13 (M3.9).
    """
    if result is None:
        return "None (no signal — do not invent details)"
    if isinstance(result, str) and result == "":
        return '"" (empty — no signal, do not invent details)'
    if isinstance(result, list):
        return _render_list_result(result)
    payload = repr(result)
    if len(payload) > _MAX_REPR_CHARS:
        payload = payload[:_MAX_REPR_CHARS] + "..."
    return payload


def _render_list_result(items: list) -> str:
    n = len(items)
    if n == 0:
        # M3.9 (PROBLEMS P3-13): explicit no-signal annotation so the
        # ANSWER_PROMPT anti-hallucination rule has a token to anchor
        # on — "no matches" alone is too neutral and Qwen sometimes
        # reads it as "matches exist but were filtered, let me guess".
        return "[] (no matches — no signal, do not invent details)"
    head = items[:_MAX_INLINE_ITEMS]
    rendered = [_render_one_hit(it) for it in head]
    body = "; ".join(rendered)
    if n > _MAX_INLINE_ITEMS:
        body += f"; ... ({n - _MAX_INLINE_ITEMS} more, total={n})"
    else:
        body += f"  (total={n})"
    return body


# M3.8 (PROBLEMS P2-12): mirror the L6 tool constant here for the renderer.
# Imported indirectly to avoid an L6→L6 cycle at module-import time.
_WEAK_MATCH_DISTANCE_THRESHOLD = 0.85


def _weak_match_note(distance: Any) -> str:
    """Return ' weak_match=true' when distance is above the weak-match
    threshold, '' otherwise. Empty / non-numeric distances are silent."""
    if isinstance(distance, (int, float)) and distance > _WEAK_MATCH_DISTANCE_THRESHOLD:
        return " weak_match=true"
    return ""


def _render_one_hit(item: Any) -> str:
    """Render one element of a list result. Recognizes M2.8 segment hits
    (dict with `segment` key) and bbox hits (dict with `metadata` carrying
    `vector_kind=bbox`)."""
    if not isinstance(item, dict):
        return repr(item)[:120]

    # M2.8 segment hit (from find_segment_by_description): has both
    # ChromaDB fields AND the joined `segment` row.
    seg = item.get("segment")
    if isinstance(seg, dict):
        dist = item.get("distance")
        dist_str = f"{dist:.3f}" if isinstance(dist, (int, float)) else "n/a"
        start_t = seg.get("start_t")
        end_t = seg.get("end_t")
        t_str = (
            f"[{start_t:.1f},{end_t:.1f}]s"
            if isinstance(start_t, (int, float)) and isinstance(end_t, (int, float))
            else "[?,?]s"
        )
        return (
            f"<segment view={seg.get('view_id')!r} "
            f"idx={seg.get('segment_idx')} "
            f"t={t_str} "
            f"src={(seg.get('source_uri') or '')[-40:]!r} "
            f"classes={seg.get('detected_classes')!r} "
            f"counts={seg.get('detected_counts')} "
            f"dist={dist_str}{_weak_match_note(dist)}>"
        )

    # M2.8 bbox hit (from find_bbox_by_description): metadata.vector_kind="bbox"
    md = item.get("metadata") if "metadata" in item else None
    if isinstance(md, dict) and md.get("vector_kind") == "bbox":
        dist = item.get("distance")
        dist_str = f"{dist:.3f}" if isinstance(dist, (int, float)) else "n/a"
        # M3.6.D: when the class-agnostic IoU tracker merged multiple
        # YOLO labels into the same track, surface the full multiset so
        # the LLM doesn't dismiss the minority class (PROBLEMS P3-12).
        classes_csv = md.get("classes_in_track")
        multi_class_note = ""
        if isinstance(classes_csv, str) and "," in classes_csv:
            multi_class_note = f" track_classes={classes_csv!r}"
        return (
            f"<bbox class={md.get('class_name')!r} "
            f"conf={md.get('confidence')} "
            f"seg_idx={md.get('segment_idx')} "
            f"view={md.get('view_id_raw')!r} "
            f"dist={dist_str}{_weak_match_note(dist)}"
            f"{multi_class_note}>"
        )

    # Generic dict (e.g. query_db SQL result) — render key=value pairs
    # compactly, prioritizing fields useful for answering questions.
    parts = []
    for k, v in item.items():
        v_str = repr(v) if not isinstance(v, str) else v
        if len(v_str) > 60:
            v_str = v_str[:60] + "..."
        parts.append(f"{k}={v_str}")
    rendered = "{" + ", ".join(parts) + "}"
    if len(rendered) > 300:
        rendered = rendered[:300] + "...}"
    return rendered
