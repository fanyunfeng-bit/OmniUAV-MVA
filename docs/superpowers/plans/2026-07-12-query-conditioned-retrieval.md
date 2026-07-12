# Query 条件化检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `retrieve` 先从 query 解析出「视角 / 时间段」约束，用剩余语义文本做嵌入检索、用约束做 chroma 元数据硬过滤（空则回退全库），确保"从合适的视角、合适的时间段检索合适的内容"。

**Architecture:** 新增纯逻辑解析层 `query_understanding.py`（规则优先→LLM 兜底的 `HybridConstraintParser`），在 `retrieval.py` 加纯函数把「解析结果 + 库内视角/时长」解析成 chroma `where`，给 `VectorStore.query` 加通用 `where` 逃生口，`AnalysisEngine.retrieve` 串起来并在响应里回传 `applied` 透明化字段，OmniUAV 检索面板展示它。

**Tech Stack:** Python 3.10、pydantic、FastAPI、ChromaDB(`where` 支持 `$and`/`$lte`/`$gte`)、DuckDB(`execute_readonly`)、PyQt5、pytest。

## Global Constraints

- MVA 引擎代码在 `mva/src/mva/`，测试在 `mva/tests/`；用 conda 环境 `mva`：`/home/fyf/miniconda3/envs/mva/bin/python`。
- OmniUAV 前端在 `omni-uav/`，用 conda 环境 `simsys`：`/home/fyf/miniconda3/envs/simsys/bin/python`；GUI 测试加 `QT_QPA_PLATFORM=offscreen`。
- MVA 测试命令：`cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest -m "not gpu" -q`（现基线 547 passed）。
- OmniUAV 测试命令：`cd omni-uav && QT_QPA_PLATFORM=offscreen /home/fyf/miniconda3/envs/simsys/bin/python -m pytest tests/ -q`（现基线 9 passed）。
- 解析层纯逻辑、无 GPU/网络依赖，可单测。LLM 解析走已有 `DashScopeLLMClient.complete(prompt, max_new_tokens=...)`（文本即可）。
- API key 只存本地 gitignored 配置；**每次 commit 前必须** `git grep --cached -nE "sk-[A-Za-z0-9]{20,}"` 确认无密钥后再提交。
- git commit 结尾附：`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- chroma 段向量元数据已含：`view_id`(=`scene::viewX`)、`view_id_raw`(=`viewX`)、`segment_idx`、`start_t`/`end_t`(float)、`source_uri`。视角过滤用 **`view_id_raw`**（免拼 scene 前缀）；时间过滤用 `start_t`/`end_t`。

---

### Task 1: 规则约束解析器 `RuleBasedConstraintParser`

**Files:**
- Create: `mva/src/mva/service/query_understanding.py`
- Test: `mva/tests/unit/test_query_understanding_rule.py`

**Interfaces:**
- Produces:
  - `@dataclass QueryConstraints{view_ref: Optional[str], time_start: Optional[float], time_end: Optional[float], relative_to_end: bool, semantic_text: str, source: str}`，属性 `has_constraint: bool`（view_ref 或任一 time 非空）。
  - `ConstraintParser` Protocol：`parse(self, text: str) -> QueryConstraints`。
  - `RuleBasedConstraintParser().parse(text) -> QueryConstraints`：`source ∈ {"rule","none"}`。

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_query_understanding_rule.py`:
```python
from mva.service.query_understanding import RuleBasedConstraintParser, QueryConstraints


P = RuleBasedConstraintParser()


def test_view_and_residual_basic():
    c = P.parse("视角1里的黄车")
    assert c.view_ref == "1"
    assert c.semantic_text == "黄车"
    assert c.time_start is None and c.time_end is None
    assert c.source == "rule"
    assert c.has_constraint is True


def test_view_english_and_drone_forms():
    assert P.parse("view3 的船").view_ref == "3"
    assert P.parse("cam04 里的人").view_ref == "4"     # 前导0
    assert P.parse("第2个无人机").view_ref == "2"
    assert P.parse("3号镜头的卡车").view_ref == "3"


def test_time_first_n_and_range_and_point():
    assert (P.parse("前10秒的船").time_start,
            P.parse("前10秒的船").time_end) == (0.0, 10.0)
    r = P.parse("第5秒到第15秒的车")
    assert (r.time_start, r.time_end, r.relative_to_end) == (5.0, 15.0, False)
    p = P.parse("第30秒的人")
    assert (p.time_start, p.time_end) == (30.0, 30.0)


def test_time_relative_to_end():
    c = P.parse("最后20秒的红色卡车")
    assert c.relative_to_end is True
    assert (c.time_start, c.time_end) == (20.0, 0.0)
    assert c.semantic_text == "红色卡车"


def test_time_anchors():
    a = P.parse("开头那艘船")
    assert (a.time_start, a.time_end, a.relative_to_end) == (0.0, 10.0, False)
    b = P.parse("结尾的车")
    assert b.relative_to_end is True


def test_view_and_time_together():
    c = P.parse("开头那艘船在view2")
    assert c.view_ref == "2"
    assert (c.time_start, c.time_end) == (0.0, 10.0)
    assert "船" in c.semantic_text


def test_plain_query_no_constraint():
    c = P.parse("黄车")
    assert c.view_ref is None
    assert c.time_start is None
    assert c.semantic_text == "黄车"
    assert c.source == "none"
    assert c.has_constraint is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_query_understanding_rule.py -q`
Expected: FAIL（`ModuleNotFoundError: mva.service.query_understanding`）

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/service/query_understanding.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_query_understanding_rule.py -q`
Expected: PASS（7 passed）

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo "SECRET-ABORT" || true
git add mva/src/mva/service/query_understanding.py mva/tests/unit/test_query_understanding_rule.py
git commit -m "feat(retrieval): rule-based query constraint parser (view/time + residual text)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: LLM 兜底 + 混合解析器 `LLMConstraintParser` / `HybridConstraintParser`

**Files:**
- Modify: `mva/src/mva/service/query_understanding.py`（在 Task 1 文件末尾追加）
- Test: `mva/tests/unit/test_query_understanding_hybrid.py`

**Interfaces:**
- Consumes: `QueryConstraints`, `RuleBasedConstraintParser`（Task 1）。
- Produces:
  - `LLMConstraintParser(llm).parse(text) -> QueryConstraints`：`llm` 需有 `complete(prompt: str, max_new_tokens: int) -> str`。解析失败优雅降级为空约束（`source="none"`）。
  - `HybridConstraintParser(rule, llm=None).parse(text) -> QueryConstraints`：规则命中即返回；否则若 `llm` 非空且句子含触发词才调 llm 解析器。

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_query_understanding_hybrid.py`:
```python
from mva.service.query_understanding import (
    RuleBasedConstraintParser, LLMConstraintParser, HybridConstraintParser,
)


class _FakeLLM:
    """有 .complete() 的假云端 LLM。"""
    def __init__(self, reply): self.reply = reply; self.calls = 0
    def complete(self, prompt, max_new_tokens=200):
        self.calls += 1
        return self.reply


def test_llm_parser_parses_json_with_fences():
    llm = _FakeLLM('```json\n{"view": 2, "time_start": null, '
                   '"time_end": null, "semantic_text": "白色SUV"}\n```')
    c = LLMConstraintParser(llm).parse("水库那个无人机里的白色SUV")
    assert c.view_ref == "2"
    assert c.semantic_text == "白色SUV"
    assert c.source == "llm"


def test_llm_parser_garbage_degrades():
    c = LLMConstraintParser(_FakeLLM("抱歉我无法理解")).parse("随便一句")
    assert c.view_ref is None
    assert c.semantic_text == "随便一句"
    assert c.source == "none"


def test_hybrid_rule_hit_skips_llm():
    llm = _FakeLLM('{"view":9,"time_start":null,"time_end":null,"semantic_text":"x"}')
    h = HybridConstraintParser(RuleBasedConstraintParser(), LLMConstraintParser(llm))
    c = h.parse("视角1里的黄车")
    assert c.view_ref == "1"          # 规则命中
    assert llm.calls == 0             # 不调 LLM


def test_hybrid_rule_miss_with_trigger_calls_llm():
    llm = _FakeLLM('{"view":3,"time_start":null,"time_end":null,"semantic_text":"人"}')
    h = HybridConstraintParser(RuleBasedConstraintParser(), LLMConstraintParser(llm))
    c = h.parse("水库那个无人机里的人")     # 规则抓不到数字, 但含触发词"无人机"
    assert llm.calls == 1
    assert c.view_ref == "3"


def test_hybrid_rule_miss_no_trigger_skips_llm():
    llm = _FakeLLM('{"view":3,"time_start":null,"time_end":null,"semantic_text":"人"}')
    h = HybridConstraintParser(RuleBasedConstraintParser(), LLMConstraintParser(llm))
    c = h.parse("黄色的车")               # 无触发词
    assert llm.calls == 0
    assert c.source == "none"


def test_hybrid_no_llm_ok():
    h = HybridConstraintParser(RuleBasedConstraintParser(), None)
    c = h.parse("水库那个无人机里的人")
    assert c.source == "none"          # 没 llm 就退回规则空结果
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_query_understanding_hybrid.py -q`
Expected: FAIL（`ImportError: cannot import name 'LLMConstraintParser'`）

- [ ] **Step 3: Write minimal implementation**

在 `mva/src/mva/service/query_understanding.py` 末尾追加：
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_query_understanding_hybrid.py -q`
Expected: PASS（6 passed）

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo "SECRET-ABORT" || true
git add mva/src/mva/service/query_understanding.py mva/tests/unit/test_query_understanding_hybrid.py
git commit -m "feat(retrieval): LLM-fallback + hybrid constraint parser

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 约束→where 纯解析函数（retrieval.py）

**Files:**
- Modify: `mva/src/mva/service/retrieval.py`（追加 3 个纯函数）
- Test: `mva/tests/unit/test_retrieval_constraints.py`

**Interfaces:**
- Consumes: `QueryConstraints`（Task 1）。
- Produces（均为纯函数）：
  - `resolve_view_ref(view_ref: Optional[str], views: list[str]) -> Optional[str]`：数字→具体 raw view_id。
  - `resolve_time(c: QueryConstraints, duration: Optional[float]) -> tuple[Optional[float], Optional[float]]`：返回 `(qs, qe)` 绝对秒。
  - `build_metadata_where(view_id_raw: Optional[str], t_start: Optional[float], t_end: Optional[float]) -> Optional[dict]`：chroma `where`。

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_retrieval_constraints.py`:
```python
from mva.service.retrieval import (
    resolve_view_ref, resolve_time, build_metadata_where,
)
from mva.service.query_understanding import QueryConstraints


def test_resolve_view_cam_and_view_forms():
    assert resolve_view_ref("1", ["cam01", "cam02", "cam03", "cam04"]) == "cam01"
    assert resolve_view_ref("3", ["cam01", "cam02", "cam03", "cam04"]) == "cam03"
    assert resolve_view_ref("2", ["view1", "view2", "view3"]) == "view2"


def test_resolve_view_disambiguates_view1_vs_view11():
    assert resolve_view_ref("1", ["view1", "view11"]) == "view1"


def test_resolve_view_nth_fallback_and_bounds():
    assert resolve_view_ref("1", ["left", "right"]) == "left"      # 无数字→排序第N个
    assert resolve_view_ref("9", ["view1", "view2"]) is None       # 越界
    assert resolve_view_ref("x", ["view1"]) is None                # 非数字
    assert resolve_view_ref("1", []) is None                       # 无 view


def test_resolve_time_absolute_passthrough():
    c = QueryConstraints(time_start=5.0, time_end=15.0, relative_to_end=False)
    assert resolve_time(c, duration=180.0) == (5.0, 15.0)


def test_resolve_time_relative_to_end():
    c = QueryConstraints(time_start=20.0, time_end=0.0, relative_to_end=True)
    assert resolve_time(c, duration=180.0) == (160.0, 180.0)


def test_resolve_time_relative_needs_duration():
    c = QueryConstraints(time_start=20.0, time_end=0.0, relative_to_end=True)
    assert resolve_time(c, duration=None) == (None, None)


def test_resolve_time_none():
    assert resolve_time(QueryConstraints(), duration=180.0) == (None, None)


def test_build_where_view_only():
    assert build_metadata_where("view1", None, None) == {"view_id_raw": "view1"}


def test_build_where_time_only():
    w = build_metadata_where(None, 0.0, 10.0)
    assert w == {"$and": [{"start_t": {"$lte": 10.0}}, {"end_t": {"$gte": 0.0}}]}


def test_build_where_view_and_time():
    w = build_metadata_where("view2", 0.0, 10.0)
    assert w == {"$and": [{"view_id_raw": "view2"},
                          {"start_t": {"$lte": 10.0}},
                          {"end_t": {"$gte": 0.0}}]}


def test_build_where_none():
    assert build_metadata_where(None, None, None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_retrieval_constraints.py -q`
Expected: FAIL（`ImportError: cannot import name 'resolve_view_ref'`）

- [ ] **Step 3: Write minimal implementation**

在 `mva/src/mva/service/retrieval.py` 顶部 `from typing import Any` 下加 `import re`，并在文件末尾追加：
```python
def resolve_view_ref(view_ref, views):
    """'1' → `views` 里的具体 raw view_id；无法解析返回 None。

    1) 唯一数字子串匹配('1'↔'view1'/'cam01'/'drone1')；
    2) 否则按名排序取 1-indexed 第 N 个；
    3) 子串多命中时取排序首个；越界/无 view → None。
    """
    if not view_ref or not views:
        return None
    n = str(view_ref).strip()
    if not n.isdigit():
        return None
    num = int(n)
    pat = re.compile(rf"(?<!\d)0*{num}(?!\d)")
    matches = [v for v in views if pat.search(v)]
    if len(matches) == 1:
        return matches[0]
    ordered = sorted(views)
    if 1 <= num <= len(ordered):
        return ordered[num - 1]
    if matches:
        return sorted(matches)[0]
    return None


def resolve_time(c, duration):
    """把 QueryConstraints 的时间解析成绝对 (qs, qe) 秒。

    relative_to_end 时 time_* 是"距末尾偏移量"，用 duration 换算：real = dur - offset。
    duration 缺失(None)时相对时间无法换算 → (None, None)。
    """
    if c.time_start is None and c.time_end is None:
        return None, None
    if not c.relative_to_end:
        return c.time_start, c.time_end
    if duration is None:
        return None, None
    qs = max(0.0, duration - (c.time_start or 0.0))
    qe = max(0.0, duration - (c.time_end or 0.0))
    return (min(qs, qe), max(qs, qe))


def build_metadata_where(view_id_raw, t_start, t_end):
    """chroma `where`(段向量)：view_id_raw 等值 + 时间重叠。

    query [t_start,t_end] 与段 [start_t,end_t] 重叠 ⇔ start_t≤t_end AND end_t≥t_start。
    """
    clauses = []
    if view_id_raw:
        clauses.append({"view_id_raw": view_id_raw})
    if t_end is not None:
        clauses.append({"start_t": {"$lte": float(t_end)}})
    if t_start is not None:
        clauses.append({"end_t": {"$gte": float(t_start)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_retrieval_constraints.py -q`
Expected: PASS（11 passed）

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo "SECRET-ABORT" || true
git add mva/src/mva/service/retrieval.py mva/tests/unit/test_retrieval_constraints.py
git commit -m "feat(retrieval): view/time resolvers + chroma where builder (pure)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `VectorStore.query` 通用 `where` 逃生口

**Files:**
- Modify: `mva/src/mva/l5_state/chromadb_store.py`（`query` 签名 + `_build_where`）
- Test: `mva/tests/unit/test_l5_chromadb_where.py`

**Interfaces:**
- Produces: `VectorStore.query(..., where: Optional[dict] = None)`——把 `where` 与内建 `vector_type`/`view_id` 子句用 `$and` 合并（`where` 若自身为 `{"$and":[...]}` 则展开，不二次嵌套）。向后兼容既有调用。

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_l5_chromadb_where.py`:
```python
import numpy as np
import pytest
from mva.l5_state.chromadb_store import VectorStore, VECTOR_TYPE_FRAME


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma"))


def _vec(seed, dim=8):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-9)).tolist()


def _add(store, seed, view_raw, start_t, end_t):
    store.add(_vec(seed), VECTOR_TYPE_FRAME, f"Scene::{view_raw}", f"seg{seed}",
              extra_metadata={"vector_kind": "segment", "view_id_raw": view_raw,
                              "start_t": start_t, "end_t": end_t})


def test_build_where_merges_extra_flat_keys():
    w = VectorStore._build_where(VECTOR_TYPE_FRAME, None, {"view_id_raw": "view1"})
    assert w == {"$and": [{"vector_type": VECTOR_TYPE_FRAME},
                          {"view_id_raw": "view1"}]}


def test_build_where_expands_extra_and_no_double_nest():
    extra = {"$and": [{"start_t": {"$lte": 10.0}}, {"end_t": {"$gte": 0.0}}]}
    w = VectorStore._build_where(VECTOR_TYPE_FRAME, None, extra)
    assert w == {"$and": [{"vector_type": VECTOR_TYPE_FRAME},
                          {"start_t": {"$lte": 10.0}},
                          {"end_t": {"$gte": 0.0}}]}


def test_query_where_filters_by_view(store):
    _add(store, 1, "view1", 0.0, 10.0)
    _add(store, 2, "view2", 0.0, 10.0)
    res = store.query(query_vector=_vec(9), vector_type=VECTOR_TYPE_FRAME,
                      top_k=5, where={"view_id_raw": "view1"})
    assert len(res) == 1
    assert res[0]["metadata"]["view_id_raw"] == "view1"


def test_query_where_filters_by_time_overlap(store):
    _add(store, 1, "view1", 0.0, 10.0)
    _add(store, 2, "view1", 100.0, 110.0)
    where = {"$and": [{"start_t": {"$lte": 10.0}}, {"end_t": {"$gte": 0.0}}]}
    res = store.query(query_vector=_vec(9), vector_type=VECTOR_TYPE_FRAME,
                      top_k=5, where=where)
    assert len(res) == 1
    assert res[0]["metadata"]["start_t"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_l5_chromadb_where.py -q`
Expected: FAIL（`query() got an unexpected keyword argument 'where'`）

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/l5_state/chromadb_store.py` —— 修改 `query` 签名与体内 where 构造，并扩展 `_build_where`。

把 `def query(...)` 签名（第 135-142 行附近）改为：
```python
    def query(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        vector_type: Optional[str] = None,
        view_id: Optional[str] = None,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
```

把体内 `where = self._build_where(vector_type, view_id)` 这一行改为：
```python
        combined = self._build_where(vector_type, view_id, where)
        kwargs: dict[str, Any] = {"n_results": top_k}
        if combined is not None:
            kwargs["where"] = combined
```

把 `_build_where` 整个替换为：
```python
    @staticmethod
    def _build_where(
        vector_type: Optional[str],
        view_id: Optional[str],
        extra: Optional[dict] = None,
    ) -> Optional[dict]:
        clauses: list[dict] = []
        if vector_type is not None:
            clauses.append({"vector_type": vector_type})
        if view_id is not None:
            clauses.append({"view_id": view_id})
        if extra:
            if "$and" in extra:
                clauses.extend(extra["$and"])
            else:
                for k, v in extra.items():
                    clauses.append({k: v})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_l5_chromadb_where.py tests/unit/test_l5_chromadb.py -q`
Expected: PASS（新 4 passed + 原有全过，无回归）

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo "SECRET-ABORT" || true
git add mva/src/mva/l5_state/chromadb_store.py mva/tests/unit/test_l5_chromadb_where.py
git commit -m "feat(l5): VectorStore.query generic where escape hatch (merged with builtin clauses)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: 响应透明化模型 `RetrieveConstraints` + `RetrieveResponse.applied`

**Files:**
- Modify: `mva/src/mva/service/models.py`
- Test: `mva/tests/unit/test_service_models_applied.py`

**Interfaces:**
- Produces:
  - `RetrieveConstraints(BaseModel){view_id: Optional[str], time_start: Optional[float], time_end: Optional[float], semantic_text: Optional[str], source: str="none", fell_back: bool=False}`
  - `RetrieveResponse` 新增 `applied: Optional[RetrieveConstraints] = None`。

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_service_models_applied.py`:
```python
from mva.service.models import RetrieveResponse, RetrieveConstraints, RetrieveHit


def test_retrieve_response_applied_roundtrip():
    r = RetrieveResponse(
        hits=[RetrieveHit(view_id="view1", score=0.9)],
        n_vectors_searched=10,
        applied=RetrieveConstraints(view_id="view1", time_start=0.0, time_end=10.0,
                                    semantic_text="黄车", source="rule", fell_back=False),
    )
    d = r.model_dump()
    assert d["applied"]["view_id"] == "view1"
    assert d["applied"]["source"] == "rule"
    assert d["applied"]["fell_back"] is False


def test_retrieve_response_applied_optional():
    r = RetrieveResponse(hits=[], n_vectors_searched=0)
    assert r.applied is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_service_models_applied.py -q`
Expected: FAIL（`ImportError: cannot import name 'RetrieveConstraints'`）

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/service/models.py` —— 在 `RetrieveResponse` 定义**之前**插入：
```python
class RetrieveConstraints(BaseModel):
    view_id: Optional[str] = None        # 解析出的 raw view, 如 "cam01"; None=未限定视角
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    semantic_text: Optional[str] = None  # 实际用于嵌入的文本
    source: str = "none"                 # rule | llm | none
    fell_back: bool = False              # 约束 0 命中 → 已扩展到全库
```
并把 `RetrieveResponse` 改为：
```python
class RetrieveResponse(BaseModel):
    hits: list[RetrieveHit] = []
    n_vectors_searched: int = 0
    applied: Optional[RetrieveConstraints] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_service_models_applied.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo "SECRET-ABORT" || true
git add mva/src/mva/service/models.py mva/tests/unit/test_service_models_applied.py
git commit -m "feat(service): RetrieveResponse.applied transparency field

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 引擎串联——`AnalysisEngine.retrieve` 条件化 + 空回退 + applied

**Files:**
- Modify: `mva/src/mva/service/engine.py`（`__init__` 注入 parser + 缓存；新增 `_get_parser`/`_scene_views`/`_scene_duration`；重写 `retrieve`；`select_scene`/`_inprocess_ingest` 失效缓存）
- Test: `mva/tests/unit/test_engine_retrieve_constraints.py`

**Interfaces:**
- Consumes: `HybridConstraintParser`/`RuleBasedConstraintParser`/`LLMConstraintParser`（Task 1/2）、`resolve_view_ref`/`resolve_time`/`build_metadata_where`（Task 3）、`VectorStore.query(where=...)`（Task 4）、`RetrieveConstraints`（Task 5）。
- Produces: `AnalysisEngine(..., parser=None)`；`retrieve(req) -> RetrieveResponse`（含 `applied`）。

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_engine_retrieve_constraints.py`:
```python
from mva.service.engine import AnalysisEngine
from mva.service.models import RetrieveRequest
from mva.service.query_understanding import RuleBasedConstraintParser


class _FakeCollection:
    def __init__(self, n): self._n = n
    def count(self): return self._n


class _FakeVStore:
    def __init__(self, result_fn):
        self.collection = _FakeCollection(100)
        self.calls = []
        self._result_fn = result_fn
    def query(self, query_text=None, vector_type=None, top_k=10, where=None):
        self.calls.append({"query_text": query_text, "where": where})
        return self._result_fn(where)


class _FakeStore:
    def __init__(self, views, dur): self._views = views; self._dur = dur
    def execute_readonly(self, sql, *a, **k):
        if "DISTINCT view_id" in sql:
            return [{"view_id": v} for v in self._views]
        if "max(end_t)" in sql:
            return [{"dur": self._dur}]
        return [{"start_t": 0.0, "source_uri": None}]      # enrich_segment_time


class _FakeSvc:
    def __init__(self, vstore, store): self.vstore = vstore; self.store = store


def _seg_hit(view_raw="view1"):
    return [{"id": "x", "distance": 0.1, "document": f"{view_raw} [0-10s]",
             "metadata": {"view_id": f"Scene::{view_raw}", "view_id_raw": view_raw,
                          "segment_idx": 0, "vector_kind": "segment"}}]


def _engine(vstore, store):
    e = AnalysisEngine(db_path="/tmp/qcr/world.duckdb",
                       chroma_dir="/tmp/qcr/chroma", defer_query_service=True)
    e._svc = _FakeSvc(vstore, store)
    e._parser = RuleBasedConstraintParser()
    return e


def test_view_constraint_filters_and_embeds_residual():
    vs = _FakeVStore(lambda where: _seg_hit("view1"))
    e = _engine(vs, _FakeStore(["view1", "view2"], 180.0))
    out = e.retrieve(RetrieveRequest(text="视角1里的黄车", top_k=3))
    assert vs.calls[0]["where"] == {"view_id_raw": "view1"}
    assert vs.calls[0]["query_text"] == "黄车"
    assert out.applied.view_id == "view1"
    assert out.applied.source == "rule"
    assert out.applied.fell_back is False


def test_relative_time_uses_duration():
    vs = _FakeVStore(lambda where: _seg_hit("view1"))
    e = _engine(vs, _FakeStore(["view1"], 180.0))
    out = e.retrieve(RetrieveRequest(text="最后20秒的红色卡车", top_k=3))
    w = vs.calls[0]["where"]
    assert {"start_t": {"$lte": 180.0}} in w["$and"]
    assert {"end_t": {"$gte": 160.0}} in w["$and"]
    assert vs.calls[0]["query_text"] == "红色卡车"


def test_empty_hit_falls_back_to_full_library():
    # 带 where 返回空, 去 where 返回命中
    vs = _FakeVStore(lambda where: [] if where is not None else _seg_hit("view2"))
    e = _engine(vs, _FakeStore(["view1", "view2"], 180.0))
    out = e.retrieve(RetrieveRequest(text="视角1里的飞机", top_k=3))
    assert len(vs.calls) == 2
    assert vs.calls[0]["where"] == {"view_id_raw": "view1"}
    assert vs.calls[1]["where"] is None
    assert out.applied.fell_back is True
    assert len(out.hits) == 1


def test_plain_query_no_constraint_single_call():
    vs = _FakeVStore(lambda where: _seg_hit("view1"))
    e = _engine(vs, _FakeStore(["view1"], 180.0))
    out = e.retrieve(RetrieveRequest(text="黄车", top_k=3))
    assert len(vs.calls) == 1
    assert vs.calls[0]["where"] is None
    assert vs.calls[0]["query_text"] == "黄车"
    assert out.applied.source == "none"
    assert out.applied.fell_back is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_engine_retrieve_constraints.py -q`
Expected: FAIL（`where` 未被透传 / `applied` 为 None → AssertionError）

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/service/engine.py` 改动：

(a) `__init__` 增加参数与缓存。把签名里 `ingest_runner: Optional[IngestRunner] = None,` 后加一行 `parser=None,`；并在 `self._svc = None` 前加两行：
```python
        self._parser = parser
        self._views_cache: Optional[list[str]] = None
```

(b) 新增三个方法（放在 `select_scene` 之后、`_inprocess_ingest` 之前）：
```python
    def _get_parser(self):
        if self._parser is None:
            from mva.service.query_understanding import (
                RuleBasedConstraintParser, LLMConstraintParser, HybridConstraintParser,
            )
            llm_parser = (LLMConstraintParser(self._llm)
                          if self._llm is not None else None)
            self._parser = HybridConstraintParser(
                RuleBasedConstraintParser(), llm_parser)
        return self._parser

    def _scene_views(self) -> list[str]:
        if self._views_cache is not None:
            return self._views_cache
        svc = self._ensure_service()
        try:
            rows = svc.store.execute_readonly("SELECT DISTINCT view_id FROM segments")
            views = sorted({r["view_id"] for r in rows if r.get("view_id")})
        except Exception:                                # noqa: BLE001
            views = []
        self._views_cache = views
        return views

    def _scene_duration(self, raw_view: Optional[str] = None) -> Optional[float]:
        svc = self._ensure_service()
        sql = "SELECT max(end_t) AS dur FROM segments"
        if raw_view:
            sql += f" WHERE view_id = '{raw_view}'"
        try:
            rows = svc.store.execute_readonly(sql)
            return rows[0].get("dur") if rows else None
        except Exception:                                # noqa: BLE001
            return None
```

(c) `select_scene` 切库时失效视角缓存。在 `select_scene` 里 `self._build_svc(db, chroma, embedder=embedder)` 这一行**前**加：
```python
        self._views_cache = None
```

(d) `_inprocess_ingest` 末尾（`progress(processed_segments=...)` 之后）追加一行，让新入库的视角/时长刷新：
```python
        self._views_cache = None
```

(e) 重写 `retrieve` 整个方法为：
```python
    def retrieve(self, req):
        # [query 条件化] 解析 view/time → chroma where 硬过滤(空则回退全库) → 嵌入剩余语义文本
        from mva.service.models import RetrieveResponse, RetrieveHit, RetrieveConstraints
        from mva.service.retrieval import (
            parse_hits, enrich_segment_time,
            resolve_view_ref, resolve_time, build_metadata_where,
        )
        from mva.service.thumbnails import extract_frame
        svc = self._ensure_service()
        if getattr(svc, "vstore", None) is None:
            return RetrieveResponse(hits=[], n_vectors_searched=0)
        n_total = svc.vstore.collection.count()

        c = self._get_parser().parse(req.text or "")
        raw_view = (resolve_view_ref(c.view_ref, self._scene_views())
                    if c.view_ref else None)
        duration = self._scene_duration(raw_view) if c.relative_to_end else None
        qs, qe = resolve_time(c, duration)
        where = build_metadata_where(raw_view, qs, qe)

        query_text = c.semantic_text or req.text
        raw = svc.vstore.query(query_text=query_text, vector_type=req.vector_type,
                               top_k=int(req.top_k), where=where)
        fell_back = False
        if not raw and where is not None:
            fell_back = True
            raw = svc.vstore.query(query_text=query_text,
                                   vector_type=req.vector_type,
                                   top_k=int(req.top_k), where=None)

        hits = [enrich_segment_time(h, svc.store) for h in parse_hits(raw)]
        out = []
        for i, h in enumerate(hits):
            thumb = None
            if i == 0 and h.get("source_uri") and h.get("t") is not None:
                import hashlib
                key = hashlib.md5(
                    f"{h['source_uri']}:{h['t']}".encode()).hexdigest()[:10]
                thumb = extract_frame(h["source_uri"], float(h["t"]),
                                      f"/tmp/mva_thumbs/{key}.jpg")
            out.append(RetrieveHit(
                view_id=h["view_id"], t=h.get("t"), segment_idx=h.get("segment_idx"),
                score=h["score"], kind=h["kind"], class_name=h.get("class_name"),
                doc=h.get("doc"), thumbnail_path=thumb,
            ))
        applied = RetrieveConstraints(
            view_id=raw_view, time_start=qs, time_end=qe,
            semantic_text=query_text, source=c.source, fell_back=fell_back,
        )
        return RetrieveResponse(hits=out, n_vectors_searched=n_total, applied=applied)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_engine_retrieve_constraints.py tests/unit/test_service_retrieve.py -q`
Expected: PASS（新 4 passed + 原 `test_service_retrieve` 仍过）

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo "SECRET-ABORT" || true
git add mva/src/mva/service/engine.py mva/tests/unit/test_engine_retrieve_constraints.py
git commit -m "feat(engine): query-conditioned retrieve (view/time filter + empty fallback + applied)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: bbox 向量补 `start_t/end_t` 元数据（前瞻，为将来目标级时间过滤）

**Files:**
- Modify: `mva/src/mva/cli/ingest.py`（`_add_bbox_vector` 的 `extra_metadata`）
- Test: `mva/tests/unit/test_ingest_bbox_time_meta.py`

**Interfaces:**
- Produces: `_add_bbox_vector` 写入的 `extra_metadata` 新增 `start_t`/`end_t`（float）。不改变既有行为；旧库需重新入库才生效。

- [ ] **Step 1: Write the failing test**

`mva/tests/unit/test_ingest_bbox_time_meta.py`:
```python
from types import SimpleNamespace
from mva.cli.ingest import _add_bbox_vector


class _CapVStore:
    def __init__(self): self.extra = None
    def add(self, vector, vector_type, view_id, tracklet_id,
            extra_metadata=None, document=None):
        self.extra = extra_metadata
        return "chroma-id-1"


def test_bbox_vector_carries_segment_time():
    seg = SimpleNamespace(start_t=0.0, end_t=10.0, view_id="view1",
                          segment_idx=0, source_uri="/x/view1.mp4")
    det = SimpleNamespace(bbox=(1.0, 2.0, 3.0, 4.0), class_name="car", confidence=0.9)
    vs = _CapVStore()
    _add_bbox_vector(vs, [0.1] * 8, "Scene", seg, "track1", 0, 0, det,
                     n_frames=2, classes_in_track="car")
    assert vs.extra["start_t"] == 0.0
    assert vs.extra["end_t"] == 10.0
    assert vs.extra["view_id_raw"] == "view1"     # 既有字段未破坏
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_ingest_bbox_time_meta.py -q`
Expected: FAIL（`KeyError: 'start_t'`）

- [ ] **Step 3: Write minimal implementation**

`mva/src/mva/cli/ingest.py` —— 在 `_add_bbox_vector` 的 `extra_metadata` 字典里、`"source_uri": seg.source_uri,` 这一行**后**加两行：
```python
            "start_t": float(seg.start_t),
            "end_t": float(seg.end_t),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_ingest_bbox_time_meta.py -q`
Expected: PASS（1 passed）

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo "SECRET-ABORT" || true
git add mva/src/mva/cli/ingest.py mva/tests/unit/test_ingest_bbox_time_meta.py
git commit -m "feat(ingest): add start_t/end_t to bbox vector metadata (future object-level time filter)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: OmniUAV 检索面板展示 `applied` 透明化

**Files:**
- Modify: `omni-uav/tabs/retrieval_tab.py`（`render` 追加 applied 行 + 新增 `_format_applied`）
- Test: `omni-uav/tests/test_retrieval_tab_applied.py`

**Interfaces:**
- Consumes: 后端 `RetrieveResponse.applied`（Task 5/6 通过 `MvaClient.retrieve` 原样返回的 dict 里的 `applied` 键）。
- Produces: `RetrievalTab._format_applied(applied: Optional[dict]) -> str`；`render` 在原透明化行下追加一行（无约束时不加）。

- [ ] **Step 1: Write the failing test**

`omni-uav/tests/test_retrieval_tab_applied.py`:
```python
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets
from tabs.retrieval_tab import RetrievalTab

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _render(applied):
    tab = RetrievalTab(mva_client=None)
    tab.render({"hits": [], "n_vectors_searched": 5, "applied": applied})
    return tab.transparency_label.text()


def test_shows_view_time_semantic_and_source():
    txt = _render({"view_id": "view1", "time_start": 0.0, "time_end": 10.0,
                   "semantic_text": "黄车", "source": "rule", "fell_back": False})
    assert "视角 view1" in txt
    assert "黄车" in txt
    assert "规则" in txt


def test_shows_fallback_note():
    txt = _render({"view_id": "view1", "time_start": None, "time_end": None,
                   "semantic_text": "飞机", "source": "rule", "fell_back": True})
    assert "无命中" in txt


def test_plain_query_adds_no_applied_line():
    txt = _render({"view_id": None, "time_start": None, "time_end": None,
                   "semantic_text": "黄车", "source": "none", "fell_back": False})
    assert "已限定" not in txt
    assert "5" in txt          # 基础透明化行仍在


def test_missing_applied_is_safe():
    tab = RetrievalTab(mva_client=None)
    tab.render({"hits": [], "n_vectors_searched": 3})    # 无 applied 键
    assert "3" in tab.transparency_label.text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd omni-uav && QT_QPA_PLATFORM=offscreen /home/fyf/miniconda3/envs/simsys/bin/python -m pytest tests/test_retrieval_tab_applied.py -q`
Expected: FAIL（`视角 view1` 不在文本里 → AssertionError）

- [ ] **Step 3: Write minimal implementation**

`omni-uav/tabs/retrieval_tab.py` —— 把 `render` 开头的 `self.transparency_label.setText(...)` 这段替换为下面，并在类里新增 `_format_applied` 静态方法。

`render` 里原来的：
```python
        self.transparency_label.setText(
            f"检索透明化:查了 {n} 个向量 · 命中 {len(hits)} 条(显示 top-{min(3, len(hits))})"
        )
```
改为：
```python
        base = (f"检索透明化:查了 {n} 个向量 · 命中 {len(hits)} 条"
                f"(显示 top-{min(3, len(hits))})")
        extra = self._format_applied(res.get("applied"))
        self.transparency_label.setText(base + (("\n" + extra) if extra else ""))
```

在类末尾（`_on_item_clicked` 之后）新增：
```python
    @staticmethod
    def _format_applied(applied) -> str:
        """把后端 applied 约束渲染成一行人读透明化；无约束返回空串。"""
        if not applied:
            return ""
        parts = []
        if applied.get("view_id"):
            parts.append(f"视角 {applied['view_id']}")
        ts, te = applied.get("time_start"), applied.get("time_end")
        if ts is not None or te is not None:
            a = f"{ts:.0f}" if isinstance(ts, (int, float)) else "…"
            b = f"{te:.0f}" if isinstance(te, (int, float)) else "…"
            parts.append(f"时间 {a}–{b}s")
        src_cn = {"rule": "规则", "llm": "LLM"}.get(applied.get("source", "none"), "")
        if not parts and not src_cn:
            return ""                       # 纯语义查询：不加额外行
        line = "已限定 " + " · ".join(parts) if parts else "语义检索"
        sem = applied.get("semantic_text")
        if sem:
            line += f' · 语义"{sem}"'
        if src_cn:
            line += f"({src_cn})"
        if applied.get("fell_back"):
            vid = applied.get("view_id") or "该约束"
            line += f" — {vid} 无命中，已扩展到全部"
        return line
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd omni-uav && QT_QPA_PLATFORM=offscreen /home/fyf/miniconda3/envs/simsys/bin/python -m pytest tests/test_retrieval_tab_applied.py tests/test_retrieval_tab.py -q`
Expected: PASS（新 4 passed + 原 `test_retrieval_tab` 仍过）

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo "SECRET-ABORT" || true
git add omni-uav/tabs/retrieval_tab.py omni-uav/tests/test_retrieval_tab_applied.py
git commit -m "feat(ui): retrieval panel shows applied view/time constraints + fallback note

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: 全量回归 + 文档收尾

**Files:**
- Modify: `MODIFICATIONS.md`、`README.md`（功能一览表 + 检索说明）

**Interfaces:** 无新接口，仅回归与文档。

- [ ] **Step 1: 跑 MVA 全量(非 GPU)确认无回归**

Run: `cd mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest -m "not gpu" -q`
Expected: PASS（≥ 547 + 本计划新增用例，全绿）

- [ ] **Step 2: 跑 OmniUAV 全量确认无回归**

Run: `cd omni-uav && QT_QPA_PLATFORM=offscreen /home/fyf/miniconda3/envs/simsys/bin/python -m pytest tests/ -q`
Expected: PASS（≥ 9 + 本计划新增用例，全绿）

- [ ] **Step 3: 更新 MODIFICATIONS.md**

在 `MODIFICATIONS.md` 的"## 7. 修复"之后、"## 当前使用速览"之前插入一节（commit 哈希按实际填）：
```markdown
## 8. Query 条件化检索
设计见 `docs/superpowers/specs/2026-07-12-query-conditioned-retrieval-design.md`，
计划见 `docs/superpowers/plans/2026-07-12-query-conditioned-retrieval.md`。
检索时先从 query 解析「视角/时间段」约束，用剩余语义文本嵌入、用约束做 chroma 硬过滤(空则回退全库)。
- 约束解析器 `mva/service/query_understanding.py`：规则优先 → LLM 兜底(`HybridConstraintParser`)。
- 约束→where 纯函数 + view 数字解析 + 相对末尾时间换算(`mva/service/retrieval.py`)。
- `VectorStore.query` 通用 `where` 逃生口(`mva/l5_state/chromadb_store.py`)。
- `AnalysisEngine.retrieve` 串联 + 空回退 + `applied` 透明化(`mva/service/engine.py`)。
- bbox 向量补 `start_t/end_t`(前瞻目标级时间过滤，需重新入库生效)。
- OmniUAV 检索面板展示 `applied`(视角/时间/语义/来源/回退)。
```

- [ ] **Step 4: 更新 README.md 功能一览**

把 `README.md` 功能一览表里"多视角检索"那一行说明改为：
```markdown
| 多视角检索 | ✅ | 文字查询→解析「视角/时间段」约束(规则优先→LLM兜底)硬过滤+空回退→top-3命中+top-1缩略图+透明化+跳帧 |
```

- [ ] **Step 5: Commit**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && echo "SECRET-ABORT" || true
git add MODIFICATIONS.md README.md
git commit -m "docs: record query-conditioned retrieval (MODIFICATIONS + README)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

## 手动验证（实现完成后，可选）

1. 起 sidecar + 用已入库的 airsim 场景：`bash scripts/start_mva_sidecar.sh`，OmniUAV 选 `~/OmniUAV-MVA-data/airsim_downtown_4view` 并入库。
2. 检索 tab 输入 `视角1里的黄车` → 透明化行应显示 `已限定 视角 cam01 · 语义"黄车"(规则)`，命中应只来自 cam01。
3. 输入 `最后10秒的车` → 应显示 `时间 …–…s` 且命中集中在末段。
4. 输入纯 `黄车` → 无"已限定"行，全库检索（保持现状）。
5. 输入一个某视角必然无命中的组合（如 `视角1里的火箭`）→ 应回退并显示 `— cam01 无命中，已扩展到全部`。
