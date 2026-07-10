"""L6 Query Planner — NL question → structured QueryPlan via the LLM.

The planner is LLM-mode-agnostic: it accepts anything with a `.complete(prompt)
-> str` method. Tests inject a scripted stub; production wires in
mva.l4_llm.LLMClient configured with Qwen2.5-VL-7B.

Parsing tolerates three response shapes:
  1. Plain JSON object
  2. JSON in a fenced ```json ... ``` block
  3. JSON object embedded somewhere in prose

PLAN.md §3.5 L6 rule: on parse failure, raise so the orchestrator can fall
back to a clarification round (do not silently swallow).
"""
from __future__ import annotations

import difflib
import re
from typing import Any

from mva.l6_interaction.plan import QueryPlan
from mva.l6_interaction.tools import ToolRegistry


PLAN_PROMPT_TEMPLATE = """你是多路视频态势引擎的查询规划器（L6 Mode A）。
{history_block}
用户问题：{question}

当前数据库 schema（DuckDB）：
{db_schema}

可用工具：
{tools}

请输出一个 JSON 对象，描述完成该问题需要调用哪些工具。Schema：
{{
  "intent": "意图分类",
  "tool_calls": [
    {{"tool": "工具名", "args": {{...参数键值对...}}}}
  ],
  "rationale": "一句话说明为什么这样规划"
}}

{keyword_hint}决策指南（**优先选下面的 typed 工具，不要自己写 SQL**）：
- 若问题出现"它/他们/那个/上一个/再…一次/同样地"等**指代或省略**，先用上方 [对话历史]
  把指代对象还原成具体名词（如 它→红色卡车），再选工具填参数。历史里没有可还原对象时
  按字面处理，不要臆造。
- 类名用**检测类名**填：船→boat、轮船→ship、无人机/飞机→uav 或 drone、人→person、车→car。
- "有几X / 多少X / 有没有X / 数一下X" → `count_objects`，args: class_name（必填）, view_id（可选）。
- "有哪些目标 / 都看到了什么" → `list_objects`，args: view_id（可选）。
- "X 在哪个视角 / 哪架无人机看到了 X" → `which_views`，args: class_name。
- "X 什么时候出现 / X 出现在哪些时间" → `when_seen`，args: class_name, view_id（可选）。
- "第 N 秒有什么" → `objects_at_time`，args: t=N。
- "两个视角是同一个 X 吗 / 哪些目标被两机同时看到" → `cross_view_matches`，args: class_name（可选）。
- "有几个视角 / 覆盖多长时间 / 总体统计" → `scene_stats`，args: 无。
- **描述/属性/动作/识别**（"那艘船什么颜色"、"在做什么"、"水里漂的是什么"、"画面里
  发生了什么"、"长什么样"）→ `look_at`，args: text=用户问题原文, view_id（可选）。
  这类问题要看真实画面，检测类名覆盖不到。
- 纯外观语义检索（"找穿红衣服的人"、"定位有 X 的画面"）→ `find_*` 向量检索。
- 用户上传了图片/视频附件 + 问"这是什么/在哪出现/有没有相似的" →
  `find_similar_to_attachment`（向量检索）/ `describe_attachment`（描述附件）。
- `query_db` 仅在以上 typed 工具都不适用、确需自定义聚合时才用（注意：目标类名在
  tracklets_* 的 bboxes JSON 里，不是列，别 SELECT detected_classes from tracklets_*）。
- 整个回复只能是 JSON 对象本身，不要任何前后缀文字。
"""


# Thin keyword prior (G-1): biases the planner toward the right typed tool for
# unambiguous phrasings. NOT a bypass — it only injects a hint line; the planner
# LLM still produces the final plan. (The aggressive planner-bypass is deferred.)
_KEYWORD_PRIOR: list[tuple[tuple[str, ...], str]] = [
    # Visual/description cues first — they must win over the generic count cues
    # below (e.g. "在做什么" contains "什么" which a count rule might otherwise grab).
    (("什么颜色", "在做什么", "在干什么", "长什么样", "什么样子", "描述",
      "是什么", "什么东西", "漂的是", "发生了什么"), "look_at"),
    (("几个视角", "多少视角", "覆盖多长", "覆盖多久", "总体统计"), "scene_stats"),
    (("第", "秒有", "时刻"), "objects_at_time"),
    (("什么时候", "何时", "哪些时间", "出现在哪些"), "when_seen"),
    (("哪个视角", "哪架", "哪个无人机", "在哪些视角"), "which_views"),
    (("同一个", "同一目标", "同一艘", "同时看到", "跨视角"), "cross_view_matches"),
    (("有哪些目标", "都看到了什么", "有什么目标", "哪些类别"), "list_objects"),
    (("几", "多少", "有没有", "数一下", "数量"), "count_objects"),
]


def _keyword_hint(question: str) -> str:
    """Return a one-line tool hint for obvious phrasings, else ''."""
    q = question or ""
    for keys, tool in _KEYWORD_PRIOR:
        if any(k in q for k in keys):
            return f"【提示】该问题很可能用 `{tool}` 工具，请优先考虑。\n"
    return ""


class QueryPlanner:
    """Plan NL queries by prompting an LLM for structured JSON output."""

    _TYPO_CUTOFF = 0.6

    def __init__(
        self, llm: Any, registry: ToolRegistry,
        db_schema: str = "",
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.db_schema = db_schema

    def plan(self, question: str, history_block: str = "") -> QueryPlan:
        prompt = PLAN_PROMPT_TEMPLATE.format(
            question=question,
            history_block=history_block,
            keyword_hint=_keyword_hint(question),
            db_schema=self.db_schema or "(no schema available)",
            tools=self.registry.descriptions(),
        )
        response = self.llm.complete(prompt)
        plan = self._parse(response)
        return self._validate_tools(plan)

    def _validate_tools(self, plan: QueryPlan) -> QueryPlan:
        """Rewrite or drop tool_calls whose tool name isn't in the registry."""
        valid_names = set(self.registry.names())
        fixed_calls = []
        for call in plan.tool_calls:
            if call.tool in valid_names:
                fixed_calls.append(call)
                continue
            close = difflib.get_close_matches(
                call.tool, valid_names, n=1, cutoff=self._TYPO_CUTOFF,
            )
            if close:
                corrected = close[0]
                print(
                    f"[planner] tool name typo: {call.tool!r} → {corrected!r} "
                    f"(close match, args preserved)"
                )
                fixed_calls.append(
                    call.model_copy(update={"tool": corrected})
                )
            else:
                print(
                    f"[planner] dropping unknown tool {call.tool!r} "
                    f"(no close match in {sorted(valid_names)[:6]}...)"
                )
        return plan.model_copy(update={"tool_calls": fixed_calls})

    @staticmethod
    def _parse(response: str) -> QueryPlan:
        text = response.strip()

        # 1) Plain JSON
        try:
            return QueryPlan.model_validate_json(text)
        except Exception:
            pass

        # 2) Fenced ```json``` block
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return QueryPlan.model_validate_json(m.group(1))

        # 3) First {...} run anywhere in the response
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return QueryPlan.model_validate_json(m.group(0))

        raise ValueError(
            f"QueryPlanner cannot extract a QueryPlan JSON object from LLM "
            f"response: {response!r}"
        )
