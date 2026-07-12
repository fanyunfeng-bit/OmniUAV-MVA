"""Query 约束解析：从检索 query 抽取「视角 / 时间段」+ 剩余语义文本。

纯逻辑、无 GPU/网络（LLM 解析器只在被注入 llm 时才联网）。设计见
docs/superpowers/specs/2026-07-12-query-conditioned-retrieval-design.md。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class QueryConstraints:
    view_ref: Optional[str] = None       # 抽出的视角数字(字符串), 如 "1"; None=未指定
    time_start: Optional[float] = None   # 绝对秒(从视频起点); None=开放
    time_end: Optional[float] = None
    relative_to_end: bool = False        # True: time_* 是"距末尾偏移量", 引擎用时长换算
    semantic_text: str = ""              # 剥离约束后的剩余文本, 用于嵌入
    source: str = "none"                 # "rule" | "llm" | "none"

    @property
    def has_constraint(self) -> bool:
        return (self.view_ref is not None
                or self.time_start is not None
                or self.time_end is not None)


@runtime_checkable
class ConstraintParser(Protocol):
    def parse(self, text: str) -> QueryConstraints: ...


_VIEW_PATTERNS = [
    re.compile(r"视角\s*(\d+)"),
    re.compile(r"第\s*(\d+)\s*(?:个|号|路)?\s*(?:视角|无人机|镜头|摄像头|相机|画面|机位)"),
    re.compile(r"(?:无人机|drone|uav)\s*#?\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*号\s*(?:无人机|视角|镜头|摄像头)"),
    re.compile(r"view\s*(\d+)", re.IGNORECASE),
    re.compile(r"cam(?:era)?\s*0*(\d+)", re.IGNORECASE),
    re.compile(r"channel\s*(\d+)", re.IGNORECASE),
]

_RE_RANGE = re.compile(r"第?\s*(\d+)\s*秒?\s*(?:到|至|-|~|—|–)\s*第?\s*(\d+)\s*秒")
_RE_FIRST_N = re.compile(r"(?:前|头)\s*(\d+)\s*秒")
_RE_LAST_N = re.compile(r"(?:最后|末尾|后)\s*(\d+)\s*秒")
_RE_POINT = re.compile(r"第\s*(\d+)\s*秒")
_RE_ANCHOR_START = re.compile(r"开头|一开始|刚开始|起初")
_RE_ANCHOR_END = re.compile(r"结尾|末尾|快结束|最后")

# 多字词放前面, 保证"那艘"整体被删而不是只删"那"
_CONNECTIVES = re.compile(
    r"(?:那|这)(?:辆|艘|架|个|只|台|列|群)|一(?:辆|艘|架|个|只|台|列|群)|"
    r"里面|里的|中的|那个|这个|画面|里|中|那|这|的"
)
_ANCHOR_DEFAULT_WINDOW = 10.0
_STRIP_CHARS = " \t\n，。、,.!！?？;；:：\"'“”‘’()（）"


class RuleBasedConstraintParser:
    """正则基线解析。零延迟、离线。source ∈ {"rule","none"}。"""

    def parse(self, text: str) -> QueryConstraints:
        c = QueryConstraints(semantic_text=text)
        spans: list[tuple[int, int]] = []

        for pat in _VIEW_PATTERNS:
            m = pat.search(text)
            if m:
                c.view_ref = m.group(1)
                spans.append(m.span())
                break

        t = self._parse_time(text)
        if t is not None:
            c.time_start, c.time_end, c.relative_to_end, span = t
            spans.append(span)

        c.semantic_text = self._residual(text, spans)
        c.source = "rule" if c.has_constraint else "none"
        return c

    @staticmethod
    def _parse_time(text: str):
        m = _RE_RANGE.search(text)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return (min(a, b), max(a, b), False, m.span())
        m = _RE_FIRST_N.search(text)
        if m:
            return (0.0, float(m.group(1)), False, m.span())
        m = _RE_LAST_N.search(text)
        if m:
            return (float(m.group(1)), 0.0, True, m.span())
        m = _RE_POINT.search(text)
        if m:
            v = float(m.group(1))
            return (v, v, False, m.span())
        m = _RE_ANCHOR_START.search(text)
        if m:
            return (0.0, _ANCHOR_DEFAULT_WINDOW, False, m.span())
        m = _RE_ANCHOR_END.search(text)
        if m:
            return (_ANCHOR_DEFAULT_WINDOW, 0.0, True, m.span())
        return None

    @staticmethod
    def _residual(text: str, spans: list[tuple[int, int]]) -> str:
        s = text
        for a, b in sorted(spans, key=lambda x: x[0], reverse=True):
            s = s[:a] + " " + s[b:]
        s = _CONNECTIVES.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip(_STRIP_CHARS)
        return s if s else text


import json


_LLM_PROMPT = (
    "你是检索查询解析器。从用户查询中抽取视角编号、时间范围和剩余的语义描述。\n"
    "只输出一个 JSON 对象，不要解释、不要代码围栏。字段：\n"
    '{"view": 整数或null, "time_start": 秒(数字)或null, '
    '"time_end": 秒(数字)或null, "semantic_text": "剥离视角/时间后的检索关键词"}\n'
    "示例：查询「靠近水库那个无人机里的白色SUV」→ "
    '{"view": null, "time_start": null, "time_end": null, "semantic_text": "白色SUV"}\n'
    "查询：「%s」"
)

_LLM_TRIGGER = re.compile(
    r"无人机|视角|镜头|摄像头|相机|画面|机位|drone|view|cam|channel|"
    r"秒|开头|一开始|刚开始|结尾|末尾|最后|前面|之后|快结束",
    re.IGNORECASE,
)


def _extract_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1 and b > a:
        s = s[a:b + 1]
    return json.loads(s)


def _as_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


class LLMConstraintParser:
    """云端 LLM 兜底解析。任何异常/垃圾输出 → 空约束(source='none')。"""

    def __init__(self, llm):
        self._llm = llm

    def parse(self, text: str) -> QueryConstraints:
        try:
            raw = self._llm.complete(_LLM_PROMPT % text, max_new_tokens=200)
            data = _extract_json(raw)
            view = data.get("view")
            c = QueryConstraints(
                view_ref=str(int(view)) if view is not None else None,
                time_start=_as_float(data.get("time_start")),
                time_end=_as_float(data.get("time_end")),
                semantic_text=(str(data.get("semantic_text") or "").strip() or text),
                source="llm",
            )
            if not c.has_constraint:
                c.source = "none"
            return c
        except Exception:                                # noqa: BLE001
            return QueryConstraints(semantic_text=text, source="none")


class HybridConstraintParser:
    """规则优先；规则未命中且句子疑似含视角/时间指代时才调 LLM 兜底。"""

    def __init__(self, rule: ConstraintParser, llm: Optional[ConstraintParser] = None):
        self._rule = rule
        self._llm = llm

    def parse(self, text: str) -> QueryConstraints:
        c = self._rule.parse(text)
        if c.has_constraint:
            return c
        if self._llm is not None and _LLM_TRIGGER.search(text or ""):
            return self._llm.parse(text)
        return c
