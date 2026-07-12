# Query 条件化检索 (Query-Conditioned Retrieval) — 设计

日期：2026-07-12
状态：已批准，待写实现计划
关联：`2026-07-10-omniuav-mva-integration-design.md`（§3.3 检索粒度）、
`2026-07-10-p1-retrieval-panel-and-perception-interface.md`（P1 检索面板）

## 1. 问题

当前 `AnalysisEngine.retrieve` 把用户整句（如 `"视角1里的黄车"`）**原样嵌入**并在**全库**近邻搜索，
不做任何视角/时间过滤。后果：

1. `"视角1"` 这几个字污染了嵌入向量（本该只嵌入 `"黄车"`）。
2. 命中来自**所有视角**，用户明确指定的 `视角1` 不起任何约束作用。

目标：**根据 query 解析出「视角 / 时间段」约束，用剩余语义文本做嵌入检索，用约束做元数据过滤**，
确保"从合适的视角、合适的时间段检索合适的内容"。

## 2. 已定决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 约束解析方式 | **混合：规则优先 → LLM 兜底** | 显式表述（"视角1"）走规则零延迟；模糊指代（"水库那个无人机"）才调云端 LLM |
| 过滤强度 | **硬过滤 + 空则回退** | 符合"确保从合适的视角"；但某视角/时段 0 命中时回退全库并提示，避免"什么都搜不到" |

## 3. 现状事实（决定可行性）

ChromaDB 每条向量已存可过滤的元数据（见 `mva/cli/ingest.py::_add_segment_vector / _add_bbox_vector`）：

- **段向量**（`vector_kind="segment"`, `vector_type="frame"`，本轮检索 MVP 用它）：
  `view_id`(=`scene::view1`)、`view_id_raw`(=`view1`)、`segment_idx`、
  **`start_t` / `end_t`（float）**、`source_uri`、`scene_id`。
- **bbox 向量**（`vector_kind="bbox"`, `vector_type="reid"`）：
  同上但**无 `start_t/end_t`**，只有 `segment_idx`、`class_name`、`classes_in_track`。

`VectorStore.query`（`mva/l5_state/chromadb_store.py`）**已支持** `view_id` 与 `vector_type`
两个内建等值子句（`_build_where` 用 `$and` 合并），但：服务层 `retrieve` 从不传 `view_id`；
且不支持数值范围（时间）过滤，也没有通用 `where` 逃生口。

## 4. 架构

```
query "视角1里的黄车"
  │
  ▼  HybridConstraintParser.parse()               [mva/service/query_understanding.py]
  │     ├─ RuleBasedConstraintParser  (正则, 零延迟)
  │     └─ LLMConstraintParser        (兜底, 云端 qwen3-vl-plus → JSON)
  ▼
QueryConstraints{ view_ref="1", time_*=None, semantic_text="黄车", source="rule" }
  │
  ▼  AnalysisEngine.retrieve()                      [mva/service/engine.py]
  │     ├─ 解析视角:  "1" → 具体 raw view_id (子串匹配 / 排序第N个)   ← segments 表 DISTINCT view_id
  │     ├─ 解析时间:  绝对/相对(用 max(end_t) 求时长) → (qs, qe)
  │     ├─ 拼 metadata where:  {view_id_raw=..., start_t≤qe, end_t≥qs}
  │     ├─ 嵌入 semantic_text ("黄车")  →  vstore.query(query_text, where=...)
  │     └─ 硬过滤: 若 0 命中 → 去约束重查全库, fell_back=True
  ▼
RetrieveResponse{ hits[], applied{view_id,time_*,semantic_text,source,fell_back} }
  │
  ▼  OmniUAV 检索面板 "透明化" 行                   [omni-uav/tabs/retrieval_tab.py]
        "已限定 视角 cam01 · 时间 0–30s · 语义"黄车"(规则)"
        回退时: "view1 无命中 → 已扩展到全部视角"
```

## 5. 组件

### 5.1 `mva/service/query_understanding.py`（新，纯逻辑、无 GPU、可单测）

```python
@dataclass
class QueryConstraints:
    view_ref: Optional[str] = None        # 抽出的视角数字(字符串), 如 "1"; None=未指定
    time_start: Optional[float] = None    # 绝对秒(从视频起点); None=开放
    time_end: Optional[float] = None
    relative_to_end: bool = False         # True: time_start/end 是"距末尾的偏移量"(秒), 引擎用时长换算
    semantic_text: str = ""               # 剥离约束后的剩余文本, 用于嵌入
    source: str = "none"                  # "rule" | "llm" | "none" — 供透明化展示

class ConstraintParser(Protocol):
    def parse(self, text: str) -> QueryConstraints: ...
```

**`RuleBasedConstraintParser`**
- 视角正则（中英，取第一处命中的数字）：
  `视角\s*(\d+)`、`第\s*(\d+)\s*(?:个|号|路)?\s*(?:视角|无人机|镜头|摄像头|相机|画面|机位)`、
  `(?:无人机|drone|uav)\s*[#]?\s*(\d+)`、`view\s*(\d+)`、`cam(?:era)?\s*0*(\d+)`、
  `(\d+)\s*号\s*(?:无人机|视角|镜头)`、`channel\s*(\d+)`。
- 时间正则：
  `前\s*(\d+)\s*秒`→(0,N)；`第?\s*(\d+)\s*秒\s*(?:到|至|-|~|—)\s*第?\s*(\d+)\s*秒?`→(A,B)；
  `第\s*(\d+)\s*秒`(单点)→(N,N)；`最后\s*(\d+)\s*秒|末尾\s*(\d+)\s*秒`→`relative_to_end`,(N,0)；
  锚点 `开头|一开始|刚开始|起初`→(0,10)；`结尾|末尾|最后(?!\d)|快结束`→`relative_to_end`,(10,0)。
- 剩余语义文本：原句去掉所有命中的视角/时间片段 → 清理悬挂连接词/量词
  （`里的|里面|中的|里|中|的|画面|那个|这个|那|一(辆|艘|架|个|只)`）→ trim 空白与首尾标点。
  若清理后为空 → 退回原句（保证有东西可嵌入）。
- `source="rule"` 当且仅当解析出 view_ref 或任一时间；否则调用方按"none"处理。

**`LLMConstraintParser(llm)`**
- prompt 让云端 LLM 严格输出 JSON：
  `{"view": <int|null>, "time_start": <float|null>, "time_end": <float|null>, "semantic_text": "<str>"}`。
- 健壮解析：剥 ```json 代码围栏、取首个 `{...}`；任何异常 → 返回空约束（`semantic_text`=原句，`source="none"`）。
- 成功且含 view/time → `source="llm"`。

**`HybridConstraintParser(rule, llm=None)`**
- 先 `rule.parse`；若解析出 view_ref 或时间 → 直接返回（`source="rule"`）。
- 否则若 `llm is not None` **且** 原句疑似含视角/时间指代（触发词集合命中：
  `无人机|视角|镜头|摄像头|相机|画面|机位|drone|view|cam|秒|开头|结尾|末尾|最后|前面|一开始`）
  → 调 `llm.parse`。
- 否则 → 返回规则的空结果（全句为 semantic_text，`source="none"`），**不调 LLM**（避免每次纯语义查询都打云端）。

### 5.2 `mva/l5_state/chromadb_store.py::VectorStore.query` — 加通用 `where` 逃生口

- 新增参数 `where: Optional[dict] = None`。
- `_build_where(vector_type, view_id, extra=None)`：把内建 `vector_type`/`view_id` 子句与 `extra`
  的所有键**平铺进同一个 `$and` 列表**再返回（0 子句→None，1 子句→裸 dict，≥2→`$and`）。
  注意 `extra` 自身可能已是 `{"$and":[...]}`（时间重叠），需并入而非嵌套二次 `$and`：
  若 `extra` 含 `$and` 则展开其列表项，否则把 `extra` 的每个 `k:v` 作为一项。
- 向后兼容：`view_id` 参数保留；现有调用不受影响。

### 5.3 `mva/service/engine.py::AnalysisEngine`

- `__init__` 注入 `parser: Optional[ConstraintParser] = None`；默认在 `_ensure_service` 后
  构造 `HybridConstraintParser(RuleBasedConstraintParser(), LLMConstraintParser(self._llm) if self._llm else None)`。
- 新增 `_scene_views() -> list[str]`：`SELECT DISTINCT view_id FROM segments`（当前活动库），
  按名排序；结果**按 scene 缓存**（切库时失效）。
- 新增 `_resolve_view(view_ref: str, views: list[str]) -> Optional[str]`：
  1. 数字子串匹配：优先返回**唯一**一个 `view_id` 使 `re.search(rf'0*{n}\b'或含该数字)` 命中
     （`1`↔`view1`/`cam01`/`drone1`）；
  2. 否则按排序 1-indexed 取第 N 个（`1≤N≤len` 时 `sorted(views)[N-1]`）；
  3. 越界/无 view → None（视角约束作废，等价不加视角过滤）。
- 新增 `_resolve_time(c, views_filter) -> tuple[Optional[float], Optional[float]]`：
  绝对量直接用；`relative_to_end` 用 `SELECT max(end_t) FROM segments [WHERE view_id=...]` 求 `dur`，
  换算 `real = dur - offset`（下限 clamp 到 0）。
- `retrieve(req)` 新流程：
  1. `c = self._parser.parse(req.text)`。
  2. `raw_view = _resolve_view(c.view_ref, views)`（若 view_ref 非空）。
  3. `(qs, qe) = _resolve_time(c, raw_view)`（若有时间）。
  4. 拼 `where`：`view_id_raw` 等值 + 时间重叠 `{"$and":[{"start_t":{"$lte":qe}},{"end_t":{"$gte":qs}}]}`
     （单边则单子句；两者都无则 `where=None`）。
  5. 嵌入 `c.semantic_text`（空则 `req.text`）→ `vstore.query(query_text=..., vector_type=req.vector_type, top_k, where=where)`。
  6. **空则回退**：命中 0 且 `where` 非空 → 用同一 `semantic_text` 去 `where` 重查，`fell_back=True`。
  7. 富化段时间 + top-1 缩略图（不变）。
  8. 返回 `RetrieveResponse(hits, n_vectors_searched, applied=RetrieveConstraints(...))`。
- **时间过滤仅作用于段向量**（其元数据有 `start_t/end_t`）。bbox 级检索（`vector_type="reid"`，未来）
  的时间过滤留待：给 bbox 元数据补 `start_t/end_t`（本设计顺带在 `_add_bbox_vector` 加，需重新入库生效），
  当前 MVP 不依赖。

### 5.4 `mva/service/models.py` — 响应加透明化字段

```python
class RetrieveConstraints(BaseModel):
    view_id: Optional[str] = None        # 解析出的 raw view, 如 "cam01"; None=未限定视角
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    semantic_text: Optional[str] = None  # 实际用于嵌入的文本
    source: str = "none"                 # rule | llm | none
    fell_back: bool = False              # 约束 0 命中 → 已扩展到全库

class RetrieveResponse(BaseModel):
    hits: list[RetrieveHit] = []
    n_vectors_searched: int = 0
    applied: Optional[RetrieveConstraints] = None   # 新增
```

### 5.5 `omni-uav/tabs/retrieval_tab.py` — 面板展示 applied

- `render(res)` 读取 `res.get("applied")`，在现有"透明化"标签追加一行人读描述：
  - 有约束：`已限定 视角 cam01 · 时间 0–30s · 语义"黄车"(规则)`（缺项省略对应片段）。
  - `fell_back`：追加 `— view1 无命中，已扩展到全部视角`。
  - 无约束（`source="none"` 且无 view/time）：不显示额外行（保持现状）。
- `MvaClient.retrieve` 已返回整个 dict，无需改动客户端。

### 5.6 `mva/cli/ingest.py::_add_bbox_vector` — 补 bbox 时间元数据（前瞻）

- `extra_metadata` 增加 `"start_t": float(seg.start_t), "end_t": float(seg.end_t)`。
- 不改变现有行为；仅为将来目标级时间过滤铺路（需重新入库才对旧库生效）。

## 6. 数据流示例

| query | view_ref→raw | time | semantic_text | where |
|---|---|---|---|---|
| `视角1里的黄车` | `1`→`view1` | — | `黄车` | `{view_id_raw:view1}` |
| `开头那艘船在view2` | `2`→`view2` | (0,10) | `船` | `{$and:[view_id_raw:view2, start_t≤10, end_t≥0]}` |
| `最后20秒的红色卡车` | — | rel(20,0)→(dur-20,dur) | `红色卡车` | `{$and:[start_t≤dur, end_t≥dur-20]}` |
| `黄车`（纯语义） | — | — | `黄车` | `None`（全库，现状） |
| `水库那个无人机里的人`（规则未命中数字）| LLM 兜底 | LLM | LLM | 依 LLM 结果 |

## 7. 测试

- **规则解析**：中英多种视角表述（视角1/第2个无人机/view3/cam04/3号镜头）、时间表述
  （前10秒/第5秒到第15秒/最后20秒/开头/结尾/单点第30秒）、剩余文本剥离（连接词/量词清理、空退回原句）。
- **view 解析**：数字子串匹配（`1`↔`view1`/`cam01`）、排序第 N 个、越界→None、多义歧义处理。
- **时间解析**：`relative_to_end` 用假 `dur` 换算、单边范围、clamp 到 0。
- **`VectorStore.query` 逃生口**：`where` 与内建子句合并（含 `$and` 展开不二次嵌套）。
- **引擎级**（假 vstore/store）：视角+时间 where 正确透传、空命中→回退、嵌入的是 semantic_text 而非原句、`applied` 字段正确。
- **`LLMConstraintParser`**（假 llm 返回 JSON / 返回垃圾 → 优雅降级）。
- **`HybridConstraintParser`**：规则命中不调 LLM、规则未命中且含触发词才调 LLM、无触发词不调 LLM。
- OmniUAV：`retrieval_tab.render` 对 `applied`/`fell_back` 的展示（offscreen）。

## 8. 保留的升级接口（YAGNI 边界）

- `ConstraintParser` Protocol：将来换更强的解析（多模态 query、指代消解、更细时间语法）。
- `VectorStore.query(where=...)` 逃生口：将来加类别、tracklet_id、地理范围等任意元数据过滤。
- bbox 向量补 `start_t/end_t`：将来目标级（`vector_type="reid"`）时间过滤。
- **本轮不做**：跨视角联合约束（"在 view1 出现又在 view2 出现的车"）、模糊时间（"傍晚"）、
  相对物体（"船旁边的人"）、按检测类别硬过滤（现走语义嵌入即可）。

## 9. 非目标

- 不改动无约束（纯语义）查询的现状路径与全库回退语义。
- 不引入新的持久化 schema（仅在既有 chroma 元数据上过滤 + 一处前瞻性加字段）。
- 不要求实时；离线本地库检索场景。
