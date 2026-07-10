# P1: 多视角检索面板 + 感知流接口 —— Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 OmniUAV 加一个"多视角检索"面板：文字查询 → 返回 top-3 命中(view/时刻/分数)+ top-1 缩略图 + 检索透明化，点击命中跳到对应视角那一帧；同时把"感知流(密集)"接口(FrameSource / Tracker / PerceptionPipeline / RelationModeler)定义好并给基线实现，为后续密集跟踪/时空关系留口。

**Architecture:** Part A：sidecar 新增 `/retrieve`(包 ChromaDB 段向量检索 + 从 DuckDB 富化时间/视频源 + top-1 抽帧缩略图)，OmniUAV 新增 `RetrievalTab`。Part B：MVA 新增 `mva.perception` 模块，定义双流解耦的接口 + 基线(不实现密集跑，只留口 + 契约测试)。检索走**语义流(段级)**；感知流接口独立于嵌入采样(见 spec §3.2 D10)。

**Tech Stack:** Python 3.10；FastAPI(sidecar)；MVA `QueryService.vstore`(ChromaDB)/`store`(DuckDB)；OpenCV 抽帧；OmniUAV PyQt5 + `requests`；pytest。

## Global Constraints

- **单一 monorepo**：`/home/fyf/fyf/PCL/OmniUAV-MVA`(remote `git@github.com:fanyunfeng-bit/OmniUAV-MVA.git`)。所有改动提交到这里。MVA 代码在 `mva/`，OmniUAV 在 `omni-uav/`。
- **MVA 命令用** `/home/fyf/miniconda3/envs/mva/bin/python`；从 `mva/` 目录跑 `pytest`。OmniUAV 测试用 `/home/fyf/miniconda3/envs/simsys/bin/python` + `QT_QPA_PLATFORM=offscreen`，从 `omni-uav/` 目录跑。
- **新测试默认无 GPU、无网络可过**(用 fake/synthetic 数据；不加载 16G 嵌入)。GPU/重活标 `@pytest.mark.gpu`。
- **每次提交前密钥硬门**：`cd /home/fyf/fyf/PCL/OmniUAV-MVA && git grep --cached -nE "sk-[A-Za-z0-9]{20,}"` 必须为空。
- **git 提交尾行**：`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`；`git -c user.name=fanyunfeng-bit -c user.email=fan_yun_feng@163.com commit`。
- **检索粒度(本轮 MVP)**：段级(10s)。命中来自 ChromaDB `vector_type="frame"`(段向量)。目标/帧级(bbox)检索列为**下一增量**(tracklet_id 前缀映射待处理，见 Task 4 备注)。
- **ChromaDB view_id 带 scene 前缀**：如 `Reservoir::view1`；DuckDB `segments.view_id` 是 `view1`。富化时须**剥掉 `<scene>::` 前缀**。
- 参考 spec：`docs/superpowers/specs/2026-07-10-omniuav-mva-integration-design.md`(§3.2 双流、§3.3 检索粒度、§7.1 检索面板)。

---

## 文件结构

**MVA(新增)**
- `mva/src/mva/service/thumbnails.py` — `extract_frame(video_path, t_sec, out_path)` 抽帧。
- `mva/src/mva/service/retrieval.py` — 纯逻辑：`parse_hits()`(chroma 结果→中间结构)、`enrich_segment_hits()`(从 DuckDB 补 start_t/source_uri)。
- `mva/src/mva/perception/__init__.py`、`frame_source.py`、`pipeline.py`、`relation.py` — 感知流接口 + 基线。
- 测试：`mva/tests/unit/{test_retrieval.py,test_thumbnails.py,test_service_retrieve.py,test_perception_frame_source.py,test_perception_interfaces.py}`。

**MVA(修改)**
- `mva/src/mva/service/models.py` — 加 `RetrieveRequest/RetrieveHit/RetrieveResponse` + `EngineProtocol.retrieve`。
- `mva/src/mva/service/app.py` — 加 `POST /retrieve`。
- `mva/src/mva/service/engine.py` — 加 `AnalysisEngine.retrieve()`。
- `mva/tests/unit/_fakes.py` — `FakeEngine` 加 `retrieve()`。

**OmniUAV(新增/修改)**
- `omni-uav/utils/mva_client.py`(改)— 加 `retrieve()`。
- `omni-uav/tabs/retrieval_tab.py`(新)— `RetrievalTab`。
- `omni-uav/tabs/__init__.py`(改)— 导出 `RetrievalTab`。
- `omni-uav/app.py`(改)— 注册检索 tab + 连接"跳到帧"。
- `omni-uav/tabs/camera_tab.py`(改)— `seek_to(cam_id, t_sec)`。
- 测试：`omni-uav/tests/{test_mva_client_retrieve.py,test_retrieval_tab.py}`。

---

# Part A — 多视角检索

## Task 1: 检索 RPC 模型 + EngineProtocol.retrieve

**Files:**
- Modify: `mva/src/mva/service/models.py`
- Test: `mva/tests/unit/test_service_models.py`(追加)

**Interfaces:**
- Produces: `RetrieveRequest{text:Optional[str], image_path:Optional[str], top_k:int=3, vector_type:str="frame"}`；`RetrieveHit{view_id, t, segment_idx, score, kind, class_name:Optional[str], doc:Optional[str], thumbnail_path:Optional[str]}`；`RetrieveResponse{hits:list[RetrieveHit], n_vectors_searched:int}`；`EngineProtocol.retrieve(RetrieveRequest)->RetrieveResponse`。

- [ ] **Step 1: 写失败测试**

```python
# mva/tests/unit/test_service_models.py 追加
def test_retrieve_models():
    from mva.service.models import RetrieveRequest, RetrieveHit, RetrieveResponse
    req = RetrieveRequest(text="airplane")
    assert req.top_k == 3 and req.vector_type == "frame"
    resp = RetrieveResponse(
        hits=[RetrieveHit(view_id="view1", t=0.0, segment_idx=0, score=0.9, kind="segment")],
        n_vectors_searched=28,
    )
    d = resp.model_dump()
    assert d["hits"][0]["view_id"] == "view1"
    assert d["n_vectors_searched"] == 28
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_service_models.py::test_retrieve_models -q -o addopts=""`
Expected: FAIL (`ImportError: cannot import name 'RetrieveRequest'`)

- [ ] **Step 3: 加模型**

在 `mva/src/mva/service/models.py` 末尾(`EngineProtocol` 之前)加：

```python
class RetrieveRequest(BaseModel):
    text: Optional[str] = None
    image_path: Optional[str] = None
    top_k: int = 3
    vector_type: str = "frame"           # "frame"=段级(本轮 MVP)；"reid"=目标级(后续)


class RetrieveHit(BaseModel):
    view_id: str
    t: Optional[float] = None            # 段起点(秒)，用于跳帧/抽缩略图
    segment_idx: Optional[int] = None
    score: float                         # 越大越相关(= 1 - 距离)
    kind: str = "segment"                # "segment" | "bbox"
    class_name: Optional[str] = None
    doc: Optional[str] = None            # 人读描述(如 "view1 [0.0-10.0s]")
    thumbnail_path: Optional[str] = None # 仅 top-1 有


class RetrieveResponse(BaseModel):
    hits: list[RetrieveHit] = []
    n_vectors_searched: int = 0
```

并在 `EngineProtocol` 里加一行方法(放 `answer` 之后)：

```python
    def retrieve(self, req: RetrieveRequest) -> RetrieveResponse: ...
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_service_models.py -q -o addopts=""`
Expected: PASS (4 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git add mva/src/mva/service/models.py mva/tests/unit/test_service_models.py
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && exit 1 || true
git -c user.name=fanyunfeng-bit -c user.email=fan_yun_feng@163.com commit -q -m "feat(service): retrieval RPC models + EngineProtocol.retrieve

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: 检索纯逻辑(解析 chroma 命中 + 富化段时间)

**Files:**
- Create: `mva/src/mva/service/retrieval.py`
- Test: `mva/tests/unit/test_retrieval.py`

**Interfaces:**
- Consumes: 无(纯函数)。ChromaDB `vstore.query()` 返回 `list[{"id","distance","metadata","document"}]`(见 `chromadb_store.py`)。
- Produces:
  - `strip_scene(view_id: str) -> str`：`"Reservoir::view1"` → `"view1"`。
  - `parse_hits(raw: list[dict]) -> list[dict]`：每条 → `{view_id(已剥前缀), segment_idx, tracklet_id, class_name, kind, score, doc}`；`score = 1 - distance`；`kind` 由 metadata `vector_kind`("segment"→"segment"，"bbox"→"bbox")。
  - `enrich_segment_time(hit: dict, store) -> dict`：对 `kind=="segment"` 的 hit，用 `store.execute_readonly` 查 `segments` 表补 `t`(start_t)与 `source_uri`；查不到则保持 None。

- [ ] **Step 1: 写失败测试**

```python
# mva/tests/unit/test_retrieval.py
from mva.service.retrieval import strip_scene, parse_hits, enrich_segment_time


def test_strip_scene():
    assert strip_scene("Reservoir::view1") == "view1"
    assert strip_scene("view2") == "view2"


def test_parse_hits():
    raw = [
        {"id": "a", "distance": 0.1, "document": "view1 [0.0-10.0s]",
         "metadata": {"view_id": "Reservoir::view1", "segment_idx": 0,
                      "vector_kind": "segment", "class_name": None}},
        {"id": "b", "distance": 0.4, "document": "airplane @ view1",
         "metadata": {"view_id": "Reservoir::view1", "segment_idx": 0,
                      "vector_kind": "bbox", "class_name": "airplane",
                      "tracklet_id": "seg0000-track1"}},
    ]
    hits = parse_hits(raw)
    assert hits[0]["view_id"] == "view1"
    assert hits[0]["kind"] == "segment"
    assert abs(hits[0]["score"] - 0.9) < 1e-6
    assert hits[1]["kind"] == "bbox"
    assert hits[1]["class_name"] == "airplane"


class _FakeStore:
    def __init__(self, rows): self._rows = rows
    def execute_readonly(self, sql, *a, **k): return self._rows


def test_enrich_segment_time():
    hit = {"view_id": "view1", "segment_idx": 0, "kind": "segment"}
    store = _FakeStore([{"start_t": 0.0, "source_uri": "/x/view1.mp4"}])
    out = enrich_segment_time(hit, store)
    assert out["t"] == 0.0
    assert out["source_uri"] == "/x/view1.mp4"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_retrieval.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: No module named 'mva.service.retrieval'`)

- [ ] **Step 3: 实现**

```python
# mva/src/mva/service/retrieval.py
"""检索纯逻辑：解析 ChromaDB 命中 + 从 DuckDB 富化段时间。无 GPU 依赖，可单测。"""
from __future__ import annotations
from typing import Any


def strip_scene(view_id: str) -> str:
    """'Reservoir::view1' -> 'view1'（DuckDB 用无前缀 view_id）。"""
    return view_id.split("::", 1)[-1] if "::" in view_id else view_id


def parse_hits(raw: list[dict]) -> list[dict]:
    hits = []
    for r in raw:
        md = r.get("metadata") or {}
        kind = "segment" if md.get("vector_kind") == "segment" else "bbox"
        dist = r.get("distance")
        score = (1.0 - float(dist)) if dist is not None else 0.0
        hits.append({
            "view_id": strip_scene(md.get("view_id", "")),
            "segment_idx": md.get("segment_idx"),
            "tracklet_id": md.get("tracklet_id"),
            "class_name": md.get("class_name"),
            "kind": kind,
            "score": score,
            "doc": r.get("document"),
        })
    return hits


def enrich_segment_time(hit: dict, store: Any) -> dict:
    """段级命中：查 segments 表补 start_t(=t) 与 source_uri。"""
    if hit.get("kind") != "segment" or hit.get("segment_idx") is None:
        return hit
    sql = (
        "SELECT start_t, source_uri FROM segments "
        f"WHERE view_id = '{hit['view_id']}' AND segment_idx = {int(hit['segment_idx'])} "
        "LIMIT 1"
    )
    try:
        rows = store.execute_readonly(sql)
    except Exception:                       # noqa: BLE001
        rows = []
    if rows:
        row = rows[0]
        hit["t"] = row.get("start_t")
        hit["source_uri"] = row.get("source_uri")
    return hit
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_retrieval.py -q -o addopts=""`
Expected: PASS (3 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git add mva/src/mva/service/retrieval.py mva/tests/unit/test_retrieval.py
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && exit 1 || true
git -c user.name=fanyunfeng-bit -c user.email=fan_yun_feng@163.com commit -q -m "feat(service): retrieval pure logic (parse chroma hits + enrich segment time)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: 抽帧缩略图 helper

**Files:**
- Create: `mva/src/mva/service/thumbnails.py`
- Test: `mva/tests/unit/test_thumbnails.py`

**Interfaces:**
- Produces: `extract_frame(video_path: str, t_sec: float, out_path: str) -> Optional[str]`：用 cv2 seek 到 `t_sec` 抽一帧写 jpg；成功返回 out_path，失败返回 None。

- [ ] **Step 1: 写失败测试(合成视频，无网络)**

```python
# mva/tests/unit/test_thumbnails.py
import cv2, numpy as np
from mva.service.thumbnails import extract_frame


def _make_video(path, n=20, fps=10, wh=(64, 48)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, wh)
    for i in range(n):
        img = np.full((wh[1], wh[0], 3), i * 10 % 255, np.uint8)
        vw.write(img)
    vw.release()


def test_extract_frame(tmp_path):
    vid = str(tmp_path / "v.mp4")
    _make_video(vid)
    out = str(tmp_path / "thumb.jpg")
    res = extract_frame(vid, t_sec=0.5, out_path=out)
    assert res == out
    img = cv2.imread(out)
    assert img is not None and img.shape[0] > 0


def test_extract_frame_bad_path(tmp_path):
    assert extract_frame("/no/such.mp4", 0.0, str(tmp_path / "x.jpg")) is None
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_thumbnails.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: 实现**

```python
# mva/src/mva/service/thumbnails.py
"""从视频在指定时刻抽一帧存 jpg（检索 top-1 缩略图用）。"""
from __future__ import annotations
import os
from typing import Optional

import cv2


def extract_frame(video_path: str, t_sec: float, out_path: str) -> Optional[str]:
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 兜底:取第一帧
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        return out_path if cv2.imwrite(out_path, frame) else None
    finally:
        cap.release()
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_thumbnails.py -q -o addopts=""`
Expected: PASS (2 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git add mva/src/mva/service/thumbnails.py mva/tests/unit/test_thumbnails.py
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && exit 1 || true
git -c user.name=fanyunfeng-bit -c user.email=fan_yun_feng@163.com commit -q -m "feat(service): frame thumbnail extraction helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Engine.retrieve + /retrieve 端点

**Files:**
- Modify: `mva/src/mva/service/engine.py`、`mva/src/mva/service/app.py`、`mva/tests/unit/_fakes.py`
- Test: `mva/tests/unit/test_service_retrieve.py`

**Interfaces:**
- Consumes: `RetrieveRequest/RetrieveResponse`(T1)、`parse_hits/enrich_segment_time`(T2)、`extract_frame`(T3)、`RetrieveHit`。
- Produces: `AnalysisEngine.retrieve(req)->RetrieveResponse`(用 `svc.vstore.query` + 富化 + top-1 缩略图；无 vstore→空结果)；`POST /retrieve`；`FakeEngine.retrieve`。
- 备注(目标/帧级后续)：`vector_type="reid"` 的 bbox 命中，DuckDB tracklet_id 带 `view1-` 前缀而 chroma 无 → 需要额外映射，本轮不做；segment 级已足够 MVP。

- [ ] **Step 1: 写失败测试(FakeEngine 返回固定结果，端点契约)**

```python
# mva/tests/unit/test_service_retrieve.py
from fastapi.testclient import TestClient
from mva.service.app import create_app
from tests.unit._fakes import FakeEngine


def test_retrieve_endpoint():
    c = TestClient(create_app(FakeEngine()))
    r = c.post("/retrieve", json={"text": "airplane", "top_k": 3})
    assert r.status_code == 200
    b = r.json()
    assert b["n_vectors_searched"] == 28
    assert b["hits"][0]["view_id"] == "view1"
    assert b["hits"][0]["thumbnail_path"] == "/tmp/thumb.jpg"
```

在 `mva/tests/unit/_fakes.py` 的 `FakeEngine` 里加：

```python
    def retrieve(self, req):
        from mva.service.models import RetrieveResponse, RetrieveHit
        return RetrieveResponse(
            hits=[RetrieveHit(view_id="view1", t=0.0, segment_idx=0, score=0.9,
                              kind="segment", thumbnail_path="/tmp/thumb.jpg")],
            n_vectors_searched=28,
        )
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_service_retrieve.py -q -o addopts=""`
Expected: FAIL (404 —— 端点未加)

- [ ] **Step 3: 加端点 + Engine.retrieve**

`mva/src/mva/service/app.py`：models import 追加 `RetrieveRequest, RetrieveResponse`，并在 `return app` 前加：

```python
    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(req: RetrieveRequest) -> RetrieveResponse:
        return engine.retrieve(req)
```

`mva/src/mva/service/engine.py` 加方法(在 `answer` 之后)：

```python
    def retrieve(self, req: "IngestRequest") -> "AnswerResponse":  # 占位，见下替换
        raise NotImplementedError
```

实际实现(替换上面占位，注意 import)：

```python
    def retrieve(self, req):
        from mva.service.models import RetrieveResponse, RetrieveHit
        from mva.service.retrieval import parse_hits, enrich_segment_time
        from mva.service.thumbnails import extract_frame
        svc = self._ensure_service()
        if getattr(svc, "vstore", None) is None:
            return RetrieveResponse(hits=[], n_vectors_searched=0)
        n_total = svc.vstore.collection.count()
        raw = svc.vstore.query(query_text=req.text, vector_type=req.vector_type,
                               top_k=int(req.top_k))
        hits = [enrich_segment_time(h, svc.store) for h in parse_hits(raw)]
        out = []
        for i, h in enumerate(hits):
            thumb = None
            if i == 0 and h.get("source_uri") and h.get("t") is not None:
                import os, hashlib
                key = hashlib.md5(f"{h['source_uri']}:{h['t']}".encode()).hexdigest()[:10]
                thumb = extract_frame(h["source_uri"], float(h["t"]),
                                      f"/tmp/mva_thumbs/{key}.jpg")
            out.append(RetrieveHit(
                view_id=h["view_id"], t=h.get("t"), segment_idx=h.get("segment_idx"),
                score=h["score"], kind=h["kind"], class_name=h.get("class_name"),
                doc=h.get("doc"), thumbnail_path=thumb,
            ))
        return RetrieveResponse(hits=out, n_vectors_searched=n_total)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_service_retrieve.py tests/unit/test_service_app.py -q -o addopts=""`
Expected: PASS

- [ ] **Step 5: 真机联调(可选，需 sidecar + 已入库)**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
MVA_DB=/tmp/mva2/world.duckdb MVA_CHROMA=/tmp/mva2/chroma bash scripts/start_mva_sidecar.sh
# 等 engine_ready
curl -s -X POST http://127.0.0.1:8900/retrieve -H 'Content-Type: application/json' -d '{"text":"airplane","top_k":3}' | python3 -m json.tool
bash scripts/stop_mva_sidecar.sh
```
Expected: 返回 3 条段命中(view/t/score)，top-1 有 thumbnail_path。

- [ ] **Step 6: 提交**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git add mva/src/mva/service/engine.py mva/src/mva/service/app.py mva/tests/unit/_fakes.py mva/tests/unit/test_service_retrieve.py
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && exit 1 || true
git -c user.name=fanyunfeng-bit -c user.email=fan_yun_feng@163.com commit -q -m "feat(service): /retrieve endpoint + Engine.retrieve (segment-level + top-1 thumbnail)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: OmniUAV MvaClient.retrieve

**Files:**
- Modify: `omni-uav/utils/mva_client.py`
- Test: `omni-uav/tests/test_mva_client_retrieve.py`

**Interfaces:**
- Produces: `MvaClient.retrieve(text=None, image_path=None, top_k=3, vector_type="frame")->dict`(键 `hits`、`n_vectors_searched`)。

- [ ] **Step 1: 写失败测试**

```python
# omni-uav/tests/test_mva_client_retrieve.py
from utils.mva_client import MvaClient


class _Resp:
    status_code = 200
    def __init__(self, p): self._p = p
    def json(self): return self._p
    def raise_for_status(self): pass


def test_retrieve(monkeypatch):
    c = MvaClient()
    monkeypatch.setattr(c._s, "post",
        lambda *a, **k: _Resp({"hits": [{"view_id": "view1", "t": 0.0, "score": 0.9}],
                               "n_vectors_searched": 28}))
    out = c.retrieve(text="airplane")
    assert out["n_vectors_searched"] == 28
    assert out["hits"][0]["view_id"] == "view1"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/omni-uav && QT_QPA_PLATFORM=offscreen /home/fyf/miniconda3/envs/simsys/bin/python -m pytest tests/test_mva_client_retrieve.py -q`
Expected: FAIL (`AttributeError: 'MvaClient' object has no attribute 'retrieve'`)

- [ ] **Step 3: 实现**

在 `omni-uav/utils/mva_client.py` 的 `MvaClient` 里加：

```python
    def retrieve(self, text=None, image_path=None, top_k: int = 3,
                 vector_type: str = "frame") -> dict:
        r = self._s.post(f"{self.base_url}/retrieve",
                         json={"text": text, "image_path": image_path,
                               "top_k": top_k, "vector_type": vector_type},
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json()
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/omni-uav && QT_QPA_PLATFORM=offscreen /home/fyf/miniconda3/envs/simsys/bin/python -m pytest tests/test_mva_client_retrieve.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git add omni-uav/utils/mva_client.py omni-uav/tests/test_mva_client_retrieve.py
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && exit 1 || true
git -c user.name=fanyunfeng-bit -c user.email=fan_yun_feng@163.com commit -q -m "feat(omni-uav): MvaClient.retrieve

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: 检索面板 RetrievalTab

**Files:**
- Create: `omni-uav/tabs/retrieval_tab.py`
- Modify: `omni-uav/tabs/__init__.py`、`omni-uav/app.py`
- Test: `omni-uav/tests/test_retrieval_tab.py`

**Interfaces:**
- Consumes: `MvaClient`(注入)。
- Produces: `RetrievalTab(mva_client, parent=None)`(QWidget)：搜索框 + "检索"按钮 + 结果区(top-3 文字 + top-1 缩略图) + 透明化行；信号 `jump_requested = pyqtSignal(str, float)`(view_id, t)。方法 `_do_search()`(读框→调 client→渲染)、`render(result: dict)`。

- [ ] **Step 1: 写失败测试(渲染逻辑，无网络；注入假 client)**

```python
# omni-uav/tests/test_retrieval_tab.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets
from tabs.retrieval_tab import RetrievalTab

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeClient:
    def retrieve(self, **k):
        return {"hits": [
            {"view_id": "view1", "t": 0.0, "score": 0.91, "class_name": None,
             "doc": "view1 [0.0-10.0s]", "thumbnail_path": None},
            {"view_id": "view2", "t": 10.0, "score": 0.80, "doc": "view2 [10-20s]"},
            {"view_id": "view1", "t": 20.0, "score": 0.72, "doc": "view1 [20-30s]"},
        ], "n_vectors_searched": 28}


def test_render_top3_and_transparency():
    tab = RetrievalTab(_FakeClient())
    tab.render(_FakeClient().retrieve())
    assert tab.results_list.count() == 3            # top-3 命中
    assert "28" in tab.transparency_label.text()    # 透明化:查了28个向量
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/omni-uav && QT_QPA_PLATFORM=offscreen /home/fyf/miniconda3/envs/simsys/bin/python -m pytest tests/test_retrieval_tab.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'tabs.retrieval_tab'`)

- [ ] **Step 3: 实现面板**

```python
# omni-uav/tabs/retrieval_tab.py
"""多视角检索面板：文字查询 → top-3 命中 + top-1 缩略图 + 透明化；点击命中跳帧。"""
from PyQt5 import QtCore, QtGui, QtWidgets


class RetrievalTab(QtWidgets.QWidget):
    jump_requested = QtCore.pyqtSignal(str, float)   # (view_id, t_sec)

    def __init__(self, mva_client, parent=None):
        super().__init__(parent)
        self.mva_client = mva_client
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.query_edit = QtWidgets.QLineEdit()
        self.query_edit.setPlaceholderText("检索:如 airplane / 一艘船 …")
        self.query_edit.returnPressed.connect(self._do_search)
        self.search_btn = QtWidgets.QPushButton("检索")
        self.search_btn.clicked.connect(self._do_search)
        bar.addWidget(self.query_edit)
        bar.addWidget(self.search_btn)
        root.addLayout(bar)

        self.transparency_label = QtWidgets.QLabel("")
        root.addWidget(self.transparency_label)

        body = QtWidgets.QHBoxLayout()
        self.results_list = QtWidgets.QListWidget()          # top-3 文字
        self.results_list.itemClicked.connect(self._on_item_clicked)
        body.addWidget(self.results_list, 3)
        self.thumb_label = QtWidgets.QLabel("(top-1 缩略图)")  # top-1 缩略图
        self.thumb_label.setAlignment(QtCore.Qt.AlignCenter)
        self.thumb_label.setMinimumSize(240, 180)
        self.thumb_label.setStyleSheet("border:1px solid #555;")
        body.addWidget(self.thumb_label, 2)
        root.addLayout(body)

    def _do_search(self):
        q = self.query_edit.text().strip()
        if not q:
            return
        try:
            res = self.mva_client.retrieve(text=q, top_k=3)
        except Exception as e:                               # noqa: BLE001
            self.transparency_label.setText(f"检索失败: {e}")
            return
        self.render(res)

    def render(self, res: dict):
        hits = res.get("hits") or []
        n = res.get("n_vectors_searched", 0)
        self.transparency_label.setText(
            f"检索透明化:查了 {n} 个向量 · 命中 {len(hits)} 条(显示 top-{min(3,len(hits))})"
        )
        self.results_list.clear()
        for i, h in enumerate(hits[:3]):
            cls = f" · {h.get('class_name')}" if h.get("class_name") else ""
            t = h.get("t")
            tstr = f"{t:.1f}s" if isinstance(t, (int, float)) else "?"
            item = QtWidgets.QListWidgetItem(
                f"#{i+1} {h.get('view_id')} @ {tstr}{cls} · 分数 {h.get('score',0):.2f}\n"
                f"     {h.get('doc') or ''}"
            )
            item.setData(QtCore.Qt.UserRole, (h.get("view_id"), t))
            self.results_list.addItem(item)
        # top-1 缩略图
        thumb = hits[0].get("thumbnail_path") if hits else None
        if thumb:
            pix = QtGui.QPixmap(thumb)
            if not pix.isNull():
                self.thumb_label.setPixmap(
                    pix.scaled(self.thumb_label.size(), QtCore.Qt.KeepAspectRatio,
                               QtCore.Qt.SmoothTransformation))
                return
        self.thumb_label.setText("(top-1 无缩略图)")

    def _on_item_clicked(self, item):
        data = item.data(QtCore.Qt.UserRole)
        if data and data[1] is not None:
            self.jump_requested.emit(str(data[0]), float(data[1]))
```

`omni-uav/tabs/__init__.py` 追加导出：

```python
from .retrieval_tab import RetrievalTab
```
并把 `RetrievalTab` 加进 `__all__`。

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/omni-uav && QT_QPA_PLATFORM=offscreen /home/fyf/miniconda3/envs/simsys/bin/python -m pytest tests/test_retrieval_tab.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: 在 app.py 注册 tab(并接跳帧信号)**

`omni-uav/app.py`：在 `tabs.addTab(self.camera_tab, "多无人机镜头")` 之后加：

```python
        # [MOD | P1] 多视角检索面板
        from tabs.retrieval_tab import RetrievalTab
        self.retrieval_tab = RetrievalTab(self.mva_client)
        self.retrieval_tab.jump_requested.connect(self._on_jump_requested)
        tabs.addTab(self.retrieval_tab, "多视角检索")
```

并加方法(放 `_refresh_engine_status` 附近)：

```python
    def _on_jump_requested(self, view_id: str, t_sec: float):
        # [MOD | P1] 检索命中 → 跳到对应视角那一帧(见 Task 7 camera_tab.seek_to)
        cam = {"view1": "cam01", "view2": "cam02", "view3": "cam03", "view4": "cam04"}.get(view_id, view_id)
        try:
            self.camera_tab.seek_to(cam, t_sec)
            self.system_output.appendPlainText(f"[检索] 跳转 {view_id} → {t_sec:.1f}s")
        except Exception as e:  # noqa: BLE001
            self.system_output.appendPlainText(f"[检索] 跳转失败: {e}")
```

- [ ] **Step 6: 验证 import app + 面板 tab 都在**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/omni-uav && QT_QPA_PLATFORM=offscreen /home/fyf/miniconda3/envs/simsys/bin/python -c "import sys;sys.path.insert(0,'.');import app;print(hasattr(app.MainWindow,'_on_jump_requested'))"`
Expected: 打印 `True`

- [ ] **Step 7: 提交**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git add omni-uav/tabs/retrieval_tab.py omni-uav/tabs/__init__.py omni-uav/app.py omni-uav/tests/test_retrieval_tab.py
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && exit 1 || true
git -c user.name=fanyunfeng-bit -c user.email=fan_yun_feng@163.com commit -q -m "feat(omni-uav): 多视角检索面板 (top-3 + top-1 缩略图 + 透明化 + 跳帧信号)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: 相机 tab 跳帧 seek_to

**Files:**
- Modify: `omni-uav/tabs/camera_tab.py`
- Test: 手动(Qt 视频 seek，不做单测)

**Interfaces:**
- Consumes: `self.video_streams`(dict cam_id→StreamBase)。
- Produces: `MultiUavCameraTab.seek_to(cam_id: str, t_sec: float)`：把对应 `VideoStream` seek 到 `t_sec`，并切到该镜头单视图。

- [ ] **Step 1: 实现 seek_to**

在 `MultiUavCameraTab` 里加(靠近 `_set_data_dir`)：

```python
    def seek_to(self, cam_id: str, t_sec: float):
        # [MOD 2026-07-10 | P1 跳帧] 检索命中跳转:把该 cam 的视频流 seek 到 t_sec，并切单视图
        stream = self.video_streams.get(cam_id)
        if stream is None:
            print(f"[seek] 无该镜头流: {cam_id}")
            return
        cap = getattr(stream, "cap", None)   # VideoStream 内部 cv2.VideoCapture
        if cap is not None:
            try:
                import cv2
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec)) * 1000.0)
            except Exception as e:  # noqa: BLE001
                print(f"[seek] 失败: {e}")
        # 切到单视图看该镜头(view_mode: 0=单视图)
        try:
            self.view_mode.setCurrentIndex(0)
            idx = self.uav_ids.index(next(u for u, c in self.uav_cam_map.items() if c == cam_id))
            self.uav_combo.setCurrentIndex(idx)
        except Exception:  # noqa: BLE001
            pass
```

> 实现前确认 `VideoStream` 暴露的 cv2 capture 属性名(读 `omni-uav/widgets/video_stream.py`；若不是 `cap`，改成实际属性名)。若封装未暴露 capture，则给 `VideoStream` 加一个 `seek(t_sec)` 方法再在此调用。

- [ ] **Step 2: 手动验证**

启动 sidecar(带已入库库)+ OmniUAV(`DISPLAY=:0.0`)：检索 "airplane" → 点 top 命中 → 相机 tab 切到该视角、画面跳到对应时刻附近。

- [ ] **Step 3: 提交**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git add omni-uav/tabs/camera_tab.py
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && exit 1 || true
git -c user.name=fanyunfeng-bit -c user.email=fan_yun_feng@163.com commit -q -m "feat(omni-uav): camera_tab.seek_to for retrieval jump-to-frame

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

# Part B — 感知流接口(留口，不实现密集跑)

## Task 8: FrameSource 接口 + 基线

**Files:**
- Create: `mva/src/mva/perception/__init__.py`、`mva/src/mva/perception/frame_source.py`
- Test: `mva/tests/unit/test_perception_frame_source.py`

**Interfaces:**
- Produces:
  - `FrameSource`(Protocol)：`iter_frames() -> Iterator[tuple[float, "np.ndarray"]]`(yield (t_sec, BGR 帧))；`fps: float`。
  - `UniformFrameSource(video_path, target_fps)`：按 `target_fps` 均匀抽帧(≤ 源 fps)；实现 Protocol。**这是感知流"密集取帧"的基线**，与嵌入的段采样无关(D10)。

- [ ] **Step 1: 写失败测试(合成视频)**

```python
# mva/tests/unit/test_perception_frame_source.py
import cv2, numpy as np
from mva.perception.frame_source import UniformFrameSource, FrameSource


def _make_video(path, n=30, fps=10, wh=(64, 48)):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, wh)
    for i in range(n):
        vw.write(np.full((wh[1], wh[0], 3), i, np.uint8))
    vw.release()


def test_uniform_frame_source_density(tmp_path):
    vid = str(tmp_path / "v.mp4"); _make_video(vid, n=30, fps=10)   # 3s@10fps
    fs = UniformFrameSource(vid, target_fps=5)                      # 期望 ~15 帧
    frames = list(fs.iter_frames())
    assert isinstance(fs, FrameSource)          # 满足 Protocol
    assert 12 <= len(frames) <= 18
    ts = [t for t, _ in frames]
    assert ts == sorted(ts) and ts[0] >= 0.0    # 时间递增、绝对时间基准
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_perception_frame_source.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: 实现**

```python
# mva/src/mva/perception/__init__.py
"""感知流(密集)接口：与嵌入的段采样解耦(spec §3.2 D10)。tracker/关系建模器可插拔(§3.1)。"""
```

```python
# mva/src/mva/perception/frame_source.py
from __future__ import annotations
from typing import Iterator, Protocol, Tuple, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    fps: float
    def iter_frames(self) -> Iterator[Tuple[float, np.ndarray]]: ...


class UniformFrameSource:
    """按 target_fps 均匀抽帧(密集感知流基线)。t 为视频绝对时间(秒)。"""
    def __init__(self, video_path: str, target_fps: float = 5.0):
        self.video_path = video_path
        self.target_fps = float(target_fps)
        self.fps = self.target_fps

    def iter_frames(self) -> Iterator[Tuple[float, np.ndarray]]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        src_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        step = max(1, int(round(src_fps / max(1e-6, self.target_fps))))
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    yield (idx / src_fps, frame)
                idx += 1
        finally:
            cap.release()
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_perception_frame_source.py -q -o addopts=""`
Expected: PASS (1 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git add mva/src/mva/perception/__init__.py mva/src/mva/perception/frame_source.py mva/tests/unit/test_perception_frame_source.py
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && exit 1 || true
git -c user.name=fanyunfeng-bit -c user.email=fan_yun_feng@163.com commit -q -m "feat(perception): FrameSource protocol + UniformFrameSource baseline (dense stream)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Tracker / PerceptionPipeline / RelationModeler 接口 + 基线

**Files:**
- Create: `mva/src/mva/perception/pipeline.py`、`mva/src/mva/perception/relation.py`
- Test: `mva/tests/unit/test_perception_interfaces.py`

**Interfaces:**
- Produces:
  - `Track`(dataclass)：`{track_id:str, view_id:str, t:float, bbox:tuple[float,float,float,float], class_name:str, score:float}`。
  - `Tracker`(Protocol)：`update(dets, t) -> list[Track]`、`reset()`。基线 `PassthroughTracker`(每个检测给独立 id，不做时序关联——占位，后续换 ByteTrack/BoT-SORT)。
  - `PerceptionPipeline`(ABC)：`run(frame_source: FrameSource, view_id, detector, tracker) -> list[Track]`。基线 `DensePerceptionPipeline`(遍历 FrameSource→detector→tracker→Track 列表)。
  - `RelationModeler`(ABC)：`model(tracks: list[Track]) -> list[dict]`(返回关系三元组占位)。基线 `NullRelationModeler`(返回 `[]`，留口)。

- [ ] **Step 1: 写失败测试(用假 detector/tracker，无 GPU)**

```python
# mva/tests/unit/test_perception_interfaces.py
import numpy as np
from mva.perception.pipeline import (
    Track, PassthroughTracker, DensePerceptionPipeline,
)
from mva.perception.relation import NullRelationModeler


class _FakeFrameSource:
    fps = 2.0
    def iter_frames(self):
        for t in (0.0, 0.5):
            yield (t, np.zeros((8, 8, 3), np.uint8))


class _FakeDetector:
    # 返回 (bbox, class_name, score) 列表
    def detect(self, frame):
        return [((0, 0, 4, 4), "boat", 0.9)]


def test_passthrough_tracker_gives_ids():
    tr = PassthroughTracker()
    out = tr.update([((0, 0, 4, 4), "boat", 0.9)], t=0.0)
    assert len(out) == 1 and out[0].class_name == "boat" and out[0].track_id


def test_dense_pipeline_runs_over_framesource():
    pipe = DensePerceptionPipeline()
    tracks = pipe.run(_FakeFrameSource(), view_id="view1",
                      detector=_FakeDetector(), tracker=PassthroughTracker())
    assert len(tracks) == 2                      # 2 帧各 1 个检测
    assert all(isinstance(t, Track) for t in tracks)
    assert {t.t for t in tracks} == {0.0, 0.5}   # 保留绝对时间


def test_null_relation_modeler_is_placeholder():
    assert NullRelationModeler().model([]) == []
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_perception_interfaces.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: 实现**

```python
# mva/src/mva/perception/pipeline.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Tuple, runtime_checkable

Bbox = Tuple[float, float, float, float]
Detection = Tuple[Bbox, str, float]        # (bbox, class_name, score)


@dataclass
class Track:
    track_id: str
    view_id: str
    t: float
    bbox: Bbox
    class_name: str
    score: float


@runtime_checkable
class Tracker(Protocol):
    def update(self, dets: Iterable[Detection], t: float) -> List[Track]: ...
    def reset(self) -> None: ...


class PassthroughTracker:
    """基线:每个检测独立 id，不做时序关联。占位，后续换 ByteTrack/BoT-SORT(§3.1)。"""
    def __init__(self):
        self._n = 0
    def reset(self) -> None:
        self._n = 0
    def update(self, dets: Iterable[Detection], t: float) -> List[Track]:
        out = []
        for bbox, cls, score in dets:
            self._n += 1
            out.append(Track(track_id=f"t{self._n}", view_id="", t=t,
                             bbox=bbox, class_name=cls, score=score))
        return out


class PerceptionPipeline(ABC):
    @abstractmethod
    def run(self, frame_source, view_id: str, detector, tracker: Tracker) -> List[Track]:
        ...


class DensePerceptionPipeline(PerceptionPipeline):
    """基线:遍历 FrameSource(密集帧)→detector→tracker→Track 列表。
    与嵌入段采样解耦(D10)。detector 需有 .detect(frame)->list[(bbox,cls,score)]。"""
    def run(self, frame_source, view_id: str, detector, tracker: Tracker) -> List[Track]:
        tracker.reset()
        tracks: List[Track] = []
        for t, frame in frame_source.iter_frames():
            dets = detector.detect(frame)
            for tr in tracker.update(dets, t):
                tr.view_id = view_id
                tracks.append(tr)
        return tracks
```

```python
# mva/src/mva/perception/relation.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class RelationModeler(ABC):
    """时空关系建模接口(留口)。model: tracks(密集轨迹) → 关系三元组列表。"""
    @abstractmethod
    def model(self, tracks: list) -> List[dict]:
        ...


class NullRelationModeler(RelationModeler):
    """基线占位:返回空。后续换规则/学习式场景图(spec §8, §3.1)。"""
    def model(self, tracks: list) -> List[dict]:
        return []
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests/unit/test_perception_interfaces.py -q -o addopts=""`
Expected: PASS (3 passed)

- [ ] **Step 5: 全量回归 + 提交**

Run: `cd /home/fyf/fyf/PCL/OmniUAV-MVA/mva && /home/fyf/miniconda3/envs/mva/bin/python -m pytest tests -m "not gpu" -q -o addopts="" 2>&1 | tail -2`
Expected: 全绿(原 532 + 本轮新增)。

```bash
cd /home/fyf/fyf/PCL/OmniUAV-MVA
git add mva/src/mva/perception/pipeline.py mva/src/mva/perception/relation.py mva/tests/unit/test_perception_interfaces.py
git grep --cached -nE "sk-[A-Za-z0-9]{20,}" && exit 1 || true
git -c user.name=fanyunfeng-bit -c user.email=fan_yun_feng@163.com commit -q -m "feat(perception): Tracker/PerceptionPipeline/RelationModeler interfaces + baselines

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review（已执行）

- **Spec 覆盖**：Part A 对应 spec §7.1 多视角检索面板(top-3/缩略图 top-1/跳帧/透明化) + §3.3 检索粒度(段级 MVP)；Part B 对应 §3.2 双流解耦(FrameSource/Tracker/PerceptionPipeline/RelationModeler 接口 + 基线)。跨视角/跟踪面板、密集感知流实现、目标级(帧精)检索 = 有意留作 P1b/下一增量(spec §9)，非遗漏。
- **占位符扫描**：无 TODO/待填。Task 4 Step3 有一处"占位再替换"是显式两步实现(先放 NotImplementedError 再替换真身)，Task 7 有"确认 VideoStream capture 属性名"是对既有代码的核对指令(非逻辑占位)。
- **类型一致**：`RetrieveRequest/RetrieveHit/RetrieveResponse` 字段贯穿 T1/T4/T5/T6；`FrameSource.iter_frames`→`(t, frame)` 在 T8/T9 一致；`Track` 字段在 T9 定义并被 pipeline 使用；`MvaClient.retrieve` 返回 dict 键与 `RetrieveResponse` 对齐；`jump_requested(view_id,t)`→`_on_jump_requested`→`camera_tab.seek_to` 串通。

## 已知风险/后续

- **R-2**：`VideoStream` 的 cv2 capture 属性名需核对(Task 7)；若封装未暴露，给 `VideoStream` 加 `seek()`。
- **R-3**：段级检索 MVP 只搜 `vector_type="frame"`；目标/帧级(bbox)需处理 chroma vs DuckDB 的 tracklet_id 前缀映射，列为下一增量。
- **R-4**：缩略图只对 top-1 抽帧(用户定)；命中很多时不额外开销。
- **R-5**：感知流为接口+基线，未接入 ingest；密集跑(P1b)时再把 DensePerceptionPipeline 结果写世界状态,并配可调 fps。

---
*P1 计划完。P1b(密集感知流+跟踪面板)、P2(空间关系)各自单独出计划。*
