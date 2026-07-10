# P0 骨架：MVA Sidecar + Grounded 问答 —— Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 OmniUAV 通过本地 RPC 调用一个 MVA sidecar 服务：选一个数据集 scene → 入库 → 用云端 Qwen3-VL-Plus 做可溯源(grounded)问答；sidecar 不可用时问答自动降级回 OmniUAV 现有直连 VLM 路径。

**Architecture（方案 A）:** 问答的 agent 逻辑跑在 MVA sidecar（贴着世界状态 DuckDB+ChromaDB）。sidecar = FastAPI（`localhost:8900`），封装 MVA `QueryService`（本地 `Qwen3-VL-Embedding-8B` 嵌入 + **新增的云端 DashScope LLM 适配器**做问答合成）和 `mva ingest`（子进程入库）。OmniUAV 侧新增 `MvaClient`，**只把"问答"分支路由到 sidecar**（引擎掉线才降级）；**"指令(跟踪/检测/重建)"分支原封不动，留在 OmniUAV 自己理解+执行**。两个 repo 各自 git 提交。

**Tech Stack:** Python 3.10；FastAPI + uvicorn；MVA `mva` 包（QueryService/ingest，DuckDB+ChromaDB，sentence-transformers）；DashScope 兼容模式(OpenAI 协议) 云端 `qwen3-vl-plus`；OmniUAV PyQt5 + `requests`；pytest。

## Global Constraints

- **单一 monorepo**（2026-07-10 起）：`/home/fyf/fyf/PCL/OmniUAV-MVA`（remote `git@github.com:fanyunfeng-bit/OmniUAV-MVA.git`）。**所有改动都提交到这里**，与旧 MVA(main)/omni-uav(dev_wenjj) 无关。
  - MVA 代码在 `OmniUAV-MVA/mva/`（`import mva` 已 `pip install -e` 指向此处）；OmniUAV 在 `OmniUAV-MVA/omni-uav/`；脚本在 `OmniUAV-MVA/scripts/`。
  - **计划正文里旧的绝对路径 `/home/fyf/fyf/PCL/Multi-Video-Analysis/...` → `OmniUAV-MVA/mva/...`；`/home/fyf/fyf/PCL/Simulation-System/omni-uav/...` → `OmniUAV-MVA/omni-uav/...`；git 命令一律 `cd /home/fyf/fyf/PCL/OmniUAV-MVA`。**
  - MVA 命令用 `/home/fyf/miniconda3/envs/mva/bin/python`。**每次提交前跑密钥硬门**：`git grep --cached -nE "sk-[A-Za-z0-9]{20,}"` 必须为空。
  - Task 10 的启停脚本放 `OmniUAV-MVA/scripts/`（已入库，正常 commit）。
- **MVA 测试**：`pytest`（`pyproject.toml`: `testpaths=["tests"]`, `pythonpath=["src"]`）。重/GPU 用 `@pytest.mark.gpu`；日常 `pytest -m "not gpu"`。**本 P0 新测试必须无 GPU、无网络下可过**（用 Fake/mock）。
- **嵌入(生产)**：本地 `Qwen/Qwen3-VL-Embedding-8B`（768 维 MRL）；测试用 mock（`model_path=None`）。
- **问答(生产)**：云端 DashScope `qwen3-vl-plus`，兼容端点 `https://dashscope.aliyuncs.com/compatible-mode/v1`。**key 从环境变量 `DASHSCOPE_API_KEY` 读，禁止写死/提交**。
- **只改问答分支**：OmniUAV 现有"意图判定→跟踪 vs 问答"结构保留；P0 只把**问答分支**改成走 MVA，跟踪/指令分支不动。
- **图像跨进程**：只传文件路径(同机共享 FS)，不传 base64。
- 参考 spec：`Simulation-System/docs/superpowers/specs/2026-07-10-omniuav-mva-integration-design.md`。

---

## 文件结构

**MVA 仓库（新增）**
- `src/mva/service/__init__.py` — 导出 `create_app`, `AnalysisEngine`。
- `src/mva/service/models.py` — RPC Pydantic 模型 + `EngineProtocol`。
- `src/mva/service/app.py` — FastAPI 工厂 `create_app(engine)` + 端点。
- `src/mva/service/engine.py` — `AnalysisEngine`（包 `QueryService` + `mva ingest` 子进程 + 任务表）。
- `src/mva/service/__main__.py` — uvicorn 入口。
- `src/mva/l4_llm/cloud_client.py` — `DashScopeLLMClient`（云端问答适配器）。
- `tests/unit/{_fakes.py,test_service_models.py,test_service_app.py,test_ingest_jobs.py,test_cloud_client.py,test_queryservice_llm_injection.py}`、`tests/smoke/test_service_smoke.py`。

**MVA 仓库（修改）**
- `pyproject.toml` — 新增 `service` 可选依赖组。
- `src/mva/cli/query.py` — `QueryService.__init__` 新增可选 `llm=` 注入（向后兼容）。

**OmniUAV 仓库（新增/修改）**
- `omni-uav/utils/mva_client.py` — `MvaClient`。
- `omni-uav/tests/{test_mva_client.py,test_qa_routing.py,conftest.py,__init__.py}`。
- `omni-uav/app.py`（修改）— 问答分支路由 + 引擎状态灯。

**Simulation-System（非 git）**
- `start_mva_sidecar.sh`（新增）+ `stop_live_demo.sh`（改）+ `MODIFICATIONS.md`（记录）。

---

## Task 1: MVA — service 依赖 + RPC 模型 + EngineProtocol

**Files:**
- Modify: `Multi-Video-Analysis/pyproject.toml`
- Create: `Multi-Video-Analysis/src/mva/service/__init__.py`
- Create: `Multi-Video-Analysis/src/mva/service/models.py`
- Test: `Multi-Video-Analysis/tests/unit/test_service_models.py`

**Interfaces:**
- Produces: 模型 `HealthResponse`、`IngestRequest`、`IngestStartResponse`、`IngestStatusResponse`、`AnswerRequest`、`Grounding`、`AnswerResponse`；协议 `EngineProtocol`（`health()`, `ingest_start(IngestRequest)`, `ingest_status(str)`, `ingest_stop(str)`, `answer(AnswerRequest)`）。后续所有 MVA service 任务消费这些类型。

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_service_models.py
from mva.service.models import (
    HealthResponse, IngestRequest, IngestStartResponse, IngestStatusResponse,
    AnswerRequest, AnswerResponse, Grounding,
)


def test_ingest_request_defaults():
    r = IngestRequest(source="/data/scene1")
    assert r.mode == "offline"
    assert r.config == {}


def test_answer_response_roundtrip():
    resp = AnswerResponse(answer="3 艘船", groundings=[Grounding(view_id="view1", t=12.0)])
    d = resp.model_dump()
    assert d["answer"] == "3 艘船"
    assert d["groundings"][0]["view_id"] == "view1"


def test_ingest_status_states():
    s = IngestStatusResponse(job_id="j1", state="running", processed_segments=4)
    assert s.state == "running"
    assert s.error is None
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_service_models.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'mva.service'`）

- [ ] **Step 3: 建包与模型**

```python
# src/mva/service/__init__.py
"""MVA sidecar service: FastAPI wrapper over QueryService + ingest.

集成缝：OmniUAV 经本地 RPC 调用；MVA 独占 DuckDB+ChromaDB。
详见 docs/superpowers/specs/2026-07-10-omniuav-mva-integration-design.md
"""
```

```python
# src/mva/service/models.py
from __future__ import annotations
from typing import Any, Literal, Optional, Protocol, runtime_checkable
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    engine_ready: bool
    db_path: Optional[str] = None


class IngestRequest(BaseModel):
    source: str
    dataset: Optional[str] = None
    mode: Literal["offline", "live"] = "offline"
    config: dict[str, Any] = {}


class IngestStartResponse(BaseModel):
    job_id: str


class IngestStatusResponse(BaseModel):
    job_id: str
    state: Literal["pending", "running", "done", "error"]
    processed_segments: int = 0
    total_segments: Optional[int] = None
    current_t: Optional[float] = None
    error: Optional[str] = None


class Grounding(BaseModel):
    view_id: Optional[str] = None
    t: Optional[float] = None
    tracklet_id: Optional[str] = None


class AnswerRequest(BaseModel):
    query: str
    attachments: list[str] = []
    session_id: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    groundings: list[Grounding] = []
    plan: Optional[dict] = None


@runtime_checkable
class EngineProtocol(Protocol):
    def health(self) -> HealthResponse: ...
    def ingest_start(self, req: IngestRequest) -> IngestStartResponse: ...
    def ingest_status(self, job_id: str) -> IngestStatusResponse: ...
    def ingest_stop(self, job_id: str) -> None: ...
    def answer(self, req: AnswerRequest) -> AnswerResponse: ...
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_service_models.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: pyproject 加 service 依赖组**

`[project.optional-dependencies]` 内新增（`dev` 之后）：

```toml
service = ["fastapi>=0.110", "uvicorn>=0.29", "httpx>=0.27", "requests>=2.31"]
```

并把 `all` 改为：

```toml
all = ["mva[detection,llm,storage,ui,dev,service]"]
```

安装：`cd /home/fyf/fyf/PCL/Multi-Video-Analysis && pip install -e '.[service]'`

- [ ] **Step 6: 提交**

```bash
cd /home/fyf/fyf/PCL/Multi-Video-Analysis
git add pyproject.toml src/mva/service/__init__.py src/mva/service/models.py tests/unit/test_service_models.py
git commit -m "feat(service): add sidecar RPC models + EngineProtocol + service deps"
```

---

## Task 2: MVA — FastAPI 应用工厂 + /health

**Files:**
- Create: `Multi-Video-Analysis/src/mva/service/app.py`
- Create: `Multi-Video-Analysis/tests/unit/_fakes.py`
- Test: `Multi-Video-Analysis/tests/unit/test_service_app.py`

**Interfaces:**
- Consumes: `EngineProtocol`, `HealthResponse`（Task 1）。
- Produces: `create_app(engine: EngineProtocol) -> FastAPI`，含 `GET /health`；测试用 `FakeEngine`（`tests/unit/_fakes.py`，实现 `EngineProtocol`，供后续测试复用）。

- [ ] **Step 1: 写失败测试 + Fake 引擎**

```python
# tests/unit/_fakes.py
from mva.service.models import (
    HealthResponse, IngestRequest, IngestStartResponse, IngestStatusResponse,
    AnswerRequest, AnswerResponse, Grounding,
)


class FakeEngine:
    """内存假引擎，满足 EngineProtocol，供 service 端点单测(无 GPU/网络)。"""
    def __init__(self):
        self.answers = {}
        self.jobs = {}
        self._n = 0

    def health(self) -> HealthResponse:
        return HealthResponse(engine_ready=True, db_path="/tmp/fake.duckdb")

    def ingest_start(self, req: IngestRequest) -> IngestStartResponse:
        self._n += 1
        jid = f"job{self._n}"
        self.jobs[jid] = IngestStatusResponse(job_id=jid, state="running")
        return IngestStartResponse(job_id=jid)

    def ingest_status(self, job_id: str) -> IngestStatusResponse:
        return self.jobs[job_id]

    def ingest_stop(self, job_id: str) -> None:
        self.jobs[job_id] = IngestStatusResponse(job_id=job_id, state="done")

    def answer(self, req: AnswerRequest) -> AnswerResponse:
        return self.answers.get(
            req.query,
            AnswerResponse(answer=f"echo:{req.query}",
                           groundings=[Grounding(view_id="view1", t=1.0)]),
        )
```

```python
# tests/unit/test_service_app.py
from fastapi.testclient import TestClient
from mva.service.app import create_app
from tests.unit._fakes import FakeEngine


def _client():
    return TestClient(create_app(FakeEngine()))


def test_health_ok():
    r = _client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine_ready"] is True
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_service_app.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'mva.service.app'`）

- [ ] **Step 3: 写应用工厂**

```python
# src/mva/service/app.py
from __future__ import annotations
from fastapi import FastAPI
from mva.service.models import EngineProtocol, HealthResponse


def create_app(engine: EngineProtocol) -> FastAPI:
    """构造 sidecar FastAPI 应用。engine 满足 EngineProtocol(生产=AnalysisEngine, 测试=FakeEngine)。"""
    app = FastAPI(title="MVA sidecar", version="0.0.1")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return engine.health()

    return app
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_service_app.py -q`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/Multi-Video-Analysis
git add src/mva/service/app.py tests/unit/test_service_app.py tests/unit/_fakes.py
git commit -m "feat(service): FastAPI app factory + /health"
```

---

## Task 3: MVA — /answer 端点

**Files:**
- Modify: `Multi-Video-Analysis/src/mva/service/app.py`
- Test: `Multi-Video-Analysis/tests/unit/test_service_app.py`（追加）

**Interfaces:**
- Consumes: `AnswerRequest`/`AnswerResponse`（Task 1），`FakeEngine`（Task 2）。
- Produces: `POST /answer`（body=`AnswerRequest`→`AnswerResponse`）。

- [ ] **Step 1: 追加失败测试**

```python
# tests/unit/test_service_app.py 追加
def test_answer_echo():
    r = _client().post("/answer", json={"query": "画面里有几艘船"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "echo:画面里有几艘船"
    assert body["groundings"][0]["view_id"] == "view1"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_service_app.py::test_answer_echo -q`
Expected: FAIL（404 Not Found）

- [ ] **Step 3: 加 /answer 端点**

`src/mva/service/app.py` 顶部 import 改为
`from mva.service.models import EngineProtocol, HealthResponse, AnswerRequest, AnswerResponse`，
并在 `return app` 之前加：

```python
    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> AnswerResponse:
        return engine.answer(req)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_service_app.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/Multi-Video-Analysis
git add src/mva/service/app.py tests/unit/test_service_app.py
git commit -m "feat(service): POST /answer endpoint"
```

---

## Task 4: MVA — /ingest/{start,status,stop}

**Files:**
- Modify: `Multi-Video-Analysis/src/mva/service/app.py`
- Test: `Multi-Video-Analysis/tests/unit/test_service_app.py`（追加）

**Interfaces:**
- Consumes: `IngestRequest`/`IngestStartResponse`/`IngestStatusResponse`（Task 1）。
- Produces: `POST /ingest/start`（→`IngestStartResponse`）、`GET /ingest/status?job=<id>`（→`IngestStatusResponse`）、`POST /ingest/stop?job=<id>`（→204）。

- [ ] **Step 1: 追加失败测试**

```python
# tests/unit/test_service_app.py 追加
def test_ingest_start_then_status():
    c = _client()
    r = c.post("/ingest/start", json={"source": "/data/scene1", "mode": "offline"})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    s = c.get("/ingest/status", params={"job": jid})
    assert s.status_code == 200
    assert s.json()["state"] == "running"


def test_ingest_stop():
    c = _client()
    jid = c.post("/ingest/start", json={"source": "/d"}).json()["job_id"]
    assert c.post("/ingest/stop", params={"job": jid}).status_code == 204
    assert c.get("/ingest/status", params={"job": jid}).json()["state"] == "done"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_service_app.py -k ingest -q`
Expected: FAIL（404）

- [ ] **Step 3: 加 ingest 端点**

在 `return app` 之前加（顶部补 `from fastapi import FastAPI, Response`；models import 补 `IngestRequest, IngestStartResponse, IngestStatusResponse`）：

```python
    @app.post("/ingest/start", response_model=IngestStartResponse)
    def ingest_start(req: IngestRequest) -> IngestStartResponse:
        return engine.ingest_start(req)

    @app.get("/ingest/status", response_model=IngestStatusResponse)
    def ingest_status(job: str) -> IngestStatusResponse:
        return engine.ingest_status(job)

    @app.post("/ingest/stop", status_code=204)
    def ingest_stop(job: str) -> Response:
        engine.ingest_stop(job)
        return Response(status_code=204)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_service_app.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/Multi-Video-Analysis
git add src/mva/service/app.py tests/unit/test_service_app.py
git commit -m "feat(service): /ingest start/status/stop endpoints"
```

---

## Task 5: MVA — DashScope 云端 LLM 适配器

**Files:**
- Create: `Multi-Video-Analysis/src/mva/l4_llm/cloud_client.py`
- Test: `Multi-Video-Analysis/tests/unit/test_cloud_client.py`

**Interfaces:**
- Produces: `DashScopeLLMClient(model="qwen3-vl-plus", api_key=None, base_url=None, timeout=60)`，方法 `complete(prompt: str, images=None, max_new_tokens=256) -> str`（问答主路径）与 `complete_messages(messages: list[dict], max_new_tokens=256) -> str`（文本+图像 best-effort，不支持视频段）。**实现前先读 `src/mva/l4_llm/client.py` 确认本地 `LLMClient` 的 `complete`/`complete_messages` 签名并保持一致**。key 从 `DASHSCOPE_API_KEY` 读，禁止写死。

- [ ] **Step 1: 写失败测试（mock HTTP，不真连网）**

```python
# tests/unit/test_cloud_client.py
import numpy as np
from mva.l4_llm.cloud_client import DashScopeLLMClient


class _FakeResp:
    status_code = 200
    def json(self):
        return {"choices": [{"message": {"content": "3 艘船"}}]}
    def raise_for_status(self):
        pass


def test_complete_posts_and_parses(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["auth"] = headers.get("Authorization")
        return _FakeResp()

    monkeypatch.setattr("mva.l4_llm.cloud_client.requests.post", fake_post)
    c = DashScopeLLMClient(model="qwen3-vl-plus", api_key="sk-test")
    out = c.complete("画面里有几艘船", images=[np.zeros((4, 4, 3), dtype=np.uint8)])

    assert out == "3 艘船"
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer sk-test"
    content = captured["json"]["messages"][-1]["content"]
    assert any(part.get("type") == "image_url" for part in content)
    assert captured["json"]["model"] == "qwen3-vl-plus"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_cloud_client.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'mva.l4_llm.cloud_client'`）

- [ ] **Step 3: 实现云端适配器**

```python
# src/mva/l4_llm/cloud_client.py
"""DashScope(通义) 云端 LLM 适配器 —— QueryService 问答用云端 qwen3-vl-plus。

与本地 mva.l4_llm.client.LLMClient 的公开方法(complete / complete_messages)保持一致，
以便直接注入 QueryService。key 从环境变量 DASHSCOPE_API_KEY 读，禁止写死。
"""
from __future__ import annotations
import base64
import os
from typing import Any, Optional

import cv2
import numpy as np
import requests

_DEFAULT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _img_to_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)          # img: BGR/np.uint8 HxWx3
    if not ok:
        raise ValueError("cv2.imencode failed for image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class DashScopeLLMClient:
    def __init__(
        self,
        model: str = "qwen3-vl-plus",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScopeLLMClient 需要 API key：设置环境变量 DASHSCOPE_API_KEY")
        self.base_url = (base_url or os.environ.get("DASHSCOPE_BASE_URL")
                         or _DEFAULT_BASE).rstrip("/")
        self.timeout = timeout

    def _post(self, messages: list[dict], max_new_tokens: int) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "max_tokens": max_new_tokens},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def complete(self, prompt: str, images: Optional[list[np.ndarray]] = None,
                 max_new_tokens: int = 256) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images or []:
            content.append({"type": "image_url",
                            "image_url": {"url": _img_to_data_url(img)}})
        return self._post([{"role": "user", "content": content}], max_new_tokens)

    def complete_messages(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        """把 MVA/Qwen 风格 messages 翻成 OpenAI content(文本+图像)。
        P0 限制：不支持视频段(video)。content 可为 str 或 list[part]；
        part: {"type":"text","text":...} / {"type":"image","image":np.ndarray or path}。"""
        out_msgs: list[dict[str, Any]] = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                out_msgs.append({"role": m["role"], "content": c})
                continue
            parts: list[dict[str, Any]] = []
            for part in c or []:
                ptype = part.get("type")
                if ptype == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "image":
                    img = part.get("image")
                    arr = cv2.imread(img) if isinstance(img, str) else img
                    parts.append({"type": "image_url",
                                  "image_url": {"url": _img_to_data_url(arr)}})
                elif ptype == "video":
                    parts.append({"type": "text", "text": "[视频段在云端适配器 P0 中未支持]"})
            out_msgs.append({"role": m["role"], "content": parts})
        return self._post(out_msgs, max_new_tokens)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_cloud_client.py -q`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/Multi-Video-Analysis
git add src/mva/l4_llm/cloud_client.py tests/unit/test_cloud_client.py
git commit -m "feat(llm): DashScope cloud LLM adapter (qwen3-vl-plus)"
```

---

## Task 6: MVA — QueryService 支持注入 llm

**Files:**
- Modify: `Multi-Video-Analysis/src/mva/cli/query.py`（`QueryService.__init__`）
- Test: `Multi-Video-Analysis/tests/unit/test_queryservice_llm_injection.py`

**Interfaces:**
- Consumes: 现有 `QueryService.__init__(db_path, chroma_dir=None, llm_model=None, embedder_model=..., embed_dim=768, quantization=None, device=None, enable_cross_view=True)`（已勘察 `query.py:55-106`）。
- Produces: 新增可选关键字参数 `llm=None`。`llm` 非空 → 用注入对象作生成 LLM（跳过本地 `LLMClient`）；为空 → 行为不变。

- [ ] **Step 1: 写失败测试（无 GPU）**

```python
# tests/unit/test_queryservice_llm_injection.py
import tempfile
from mva.cli.query import QueryService


class _StubLLM:
    def complete(self, prompt, images=None, max_new_tokens=256):
        return "stub-answer"
    def complete_messages(self, messages, max_new_tokens=256):
        return "stub-answer"


def test_injected_llm_is_used():
    with tempfile.TemporaryDirectory() as d:
        svc = QueryService(db_path=f"{d}/w.duckdb", llm=_StubLLM())  # 无 chroma → 不加载嵌入
        assert svc.llm is not None
        assert svc.llm.complete("x") == "stub-answer"
        svc.close()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_queryservice_llm_injection.py -q`
Expected: FAIL（`TypeError: __init__() got an unexpected keyword argument 'llm'`）

- [ ] **Step 3: 加 llm 注入**

`QueryService.__init__` 参数表末尾加（`enable_cross_view` 之后）：

```python
        enable_cross_view: bool = True,
        llm=None,
    ) -> None:
```

把原 `self.llm = LLMClient(model_path=llm_model, quantization=quantization)` 改为：

```python
        # 允许注入自定义生成 LLM(如云端 DashScopeLLMClient)；否则用本地 LLMClient。
        if llm is not None:
            self.llm = llm
        else:
            self.llm = LLMClient(model_path=llm_model, quantization=quantization)
```

- [ ] **Step 4: 运行确认通过 + 未回归**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_queryservice_llm_injection.py -q && python -m pytest tests -m "not gpu" -q`
Expected: 新测试 PASS；既有 `not gpu` 套件仍全绿。

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/Multi-Video-Analysis
git add src/mva/cli/query.py tests/unit/test_queryservice_llm_injection.py
git commit -m "feat(query): allow injecting a custom generation LLM into QueryService"
```

---

## Task 7: MVA — 真实 AnalysisEngine + uvicorn 入口

**Files:**
- Create: `Multi-Video-Analysis/src/mva/service/engine.py`
- Create: `Multi-Video-Analysis/src/mva/service/__main__.py`
- Modify: `Multi-Video-Analysis/src/mva/service/__init__.py`
- Test: `Multi-Video-Analysis/tests/unit/test_ingest_jobs.py`、`tests/smoke/test_service_smoke.py`

**Interfaces:**
- Consumes: models（Task 1）、`QueryService`（含 Task 6 `llm=`）、`DashScopeLLMClient`（Task 5）。
- Produces:
  - `AnalysisEngine(db_path, chroma_dir=None, embedder_model="Qwen/Qwen3-VL-Embedding-8B", device=None, llm=None, ingest_runner=None, defer_query_service=False)` 实现 `EngineProtocol`。
  - `ingest_runner: Callable[[IngestRequest, progress_cb], None]`（默认=子进程 `mva ingest`；可注入以测试）。`progress_cb(**kw)` 更新 `IngestStatusResponse` 字段。
  - `python -m mva.service --host 127.0.0.1 --port 8900 --db <path> --chroma-dir <path>`。

- [ ] **Step 1: 写失败测试（注入假 runner，纯内存）**

```python
# tests/unit/test_ingest_jobs.py
import time
from mva.service.engine import AnalysisEngine
from mva.service.models import IngestRequest


def test_ingest_job_lifecycle(tmp_path):
    calls = []

    def fake_runner(req, progress):
        progress(processed_segments=1, total_segments=2)
        progress(processed_segments=2, total_segments=2)
        calls.append(req.source)

    eng = AnalysisEngine(db_path=str(tmp_path / "w.duckdb"), chroma_dir=None,
                         ingest_runner=fake_runner, defer_query_service=True)
    start = eng.ingest_start(IngestRequest(source="/data/s1"))
    for _ in range(50):
        if eng.ingest_status(start.job_id).state == "done":
            break
        time.sleep(0.02)
    st = eng.ingest_status(start.job_id)
    assert st.state == "done"
    assert st.processed_segments == 2
    assert calls == ["/data/s1"]
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_ingest_jobs.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'mva.service.engine'`）

- [ ] **Step 3: 实现 AnalysisEngine**

```python
# src/mva/service/engine.py
"""真实引擎：QueryService(本地嵌入 + 云端 LLM) + mva ingest(子进程) + 任务表。"""
from __future__ import annotations
import subprocess
import threading
import uuid
from typing import Callable, Optional

from mva.service.models import (
    HealthResponse, IngestRequest, IngestStartResponse, IngestStatusResponse,
    AnswerRequest, AnswerResponse,
)

ProgressCb = Callable[..., None]
IngestRunner = Callable[[IngestRequest, ProgressCb], None]


class _IngestJob:
    def __init__(self, job_id: str):
        self.status = IngestStatusResponse(job_id=job_id, state="pending")
        self.stop_flag = threading.Event()


def _default_subprocess_runner(req: IngestRequest, progress: ProgressCb) -> None:
    """默认入库：子进程跑已测的 `mva ingest` CLI。
    ⚠️ 执行前先 `mva ingest --help` 确认 dataset/scene/db/chroma 的确切 flag，
    据此微调下面命令拼装(这里给常见形态)。"""
    cfg = req.config or {}
    cmd = ["mva", "ingest"]
    if req.dataset:
        cmd += ["--dataset", req.dataset]
    cmd += ["--scene", req.source]
    if "db" in cfg:
        cmd += ["--db", cfg["db"]]
    if "chroma_dir" in cfg:
        cmd += ["--chroma-dir", cfg["chroma_dir"]]
    if "embedder_model" in cfg:
        cmd += ["--embedder-model", cfg["embedder_model"]]
    cmd += ["--detect", "--track"]
    progress(processed_segments=0)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"mva ingest 失败: {proc.stderr[-2000:]}")
    progress(processed_segments=1, total_segments=1)


class AnalysisEngine:
    def __init__(
        self,
        db_path: str,
        chroma_dir: Optional[str] = None,
        embedder_model: str = "Qwen/Qwen3-VL-Embedding-8B",
        device: Optional[str] = None,
        llm=None,
        ingest_runner: Optional[IngestRunner] = None,
        defer_query_service: bool = False,
    ) -> None:
        self.db_path = db_path
        self.chroma_dir = chroma_dir
        self._runner = ingest_runner or _default_subprocess_runner
        self._jobs: dict[str, _IngestJob] = {}
        self._lock = threading.Lock()
        self._svc = None
        self._svc_kwargs = dict(db_path=db_path, chroma_dir=chroma_dir,
                                embedder_model=embedder_model, embed_dim=768,
                                device=device, llm=llm)
        if not defer_query_service:
            self._ensure_service()

    def _ensure_service(self):
        if self._svc is None:
            from mva.cli.query import QueryService
            self._svc = QueryService(**self._svc_kwargs)
        return self._svc

    def health(self) -> HealthResponse:
        return HealthResponse(engine_ready=self._svc is not None, db_path=self.db_path)

    def ingest_start(self, req: IngestRequest) -> IngestStartResponse:
        job_id = uuid.uuid4().hex[:12]
        job = _IngestJob(job_id)
        with self._lock:
            self._jobs[job_id] = job

        def progress(**kw):
            for k, v in kw.items():
                setattr(job.status, k, v)
            job.status.state = "running"

        def run():
            try:
                self._runner(req, progress)
                job.status.state = "done"
            except Exception as e:                       # noqa: BLE001
                job.status.state = "error"
                job.status.error = str(e)[:500]

        threading.Thread(target=run, daemon=True).start()
        job.status.state = "running"
        return IngestStartResponse(job_id=job_id)

    def ingest_status(self, job_id: str) -> IngestStatusResponse:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return IngestStatusResponse(job_id=job_id, state="error", error="unknown job_id")
        return job.status

    def ingest_stop(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            job.stop_flag.set()
            job.status.state = "done"

    def answer(self, req: AnswerRequest) -> AnswerResponse:
        from mva.contracts import Attachment, RichQuery
        svc = self._ensure_service()
        atts = [Attachment(kind="image", path=p, label=p) for p in req.attachments]
        rich = RichQuery(text=req.query, attachments=atts)
        result = svc.answer(rich)
        plan = None
        try:
            plan = result.plan.model_dump() if hasattr(result.plan, "model_dump") else None
        except Exception:                                # noqa: BLE001
            plan = None
        return AnswerResponse(answer=result.answer, groundings=[], plan=plan)
```

`src/mva/service/__init__.py` 末尾加：

```python
from mva.service.app import create_app          # noqa: E402
from mva.service.engine import AnalysisEngine    # noqa: E402

__all__ = ["create_app", "AnalysisEngine"]
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/unit/test_ingest_jobs.py -q`
Expected: PASS（1 passed）

- [ ] **Step 5: 写 uvicorn 入口**

```python
# src/mva/service/__main__.py
import argparse
import uvicorn
from mva.service.app import create_app
from mva.service.engine import AnalysisEngine


def main() -> None:
    ap = argparse.ArgumentParser("mva.service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--db", required=True, help="DuckDB 世界状态库路径")
    ap.add_argument("--chroma-dir", default=None, help="ChromaDB 目录")
    ap.add_argument("--embedder-model", default="Qwen/Qwen3-VL-Embedding-8B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--qa-model", default="qwen3-vl-plus", help="云端问答模型")
    args = ap.parse_args()

    from mva.l4_llm.cloud_client import DashScopeLLMClient
    llm = DashScopeLLMClient(model=args.qa_model)       # key 从 DASHSCOPE_API_KEY 读
    engine = AnalysisEngine(db_path=args.db, chroma_dir=args.chroma_dir,
                            embedder_model=args.embedder_model, device=args.device, llm=llm)
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 冒烟测试**

```python
# tests/smoke/test_service_smoke.py
def test_service_main_importable():
    import mva.service.__main__ as m
    assert hasattr(m, "main")


def test_engine_health_defer():
    from mva.service.engine import AnalysisEngine
    eng = AnalysisEngine(db_path="/tmp/x.duckdb", chroma_dir=None, defer_query_service=True)
    assert eng.health().engine_ready is False
```

Run: `cd /home/fyf/fyf/PCL/Multi-Video-Analysis && python -m pytest tests/smoke/test_service_smoke.py -q`
Expected: PASS（2 passed）

- [ ] **Step 7: 手动联调（可选，需真环境 + `export DASHSCOPE_API_KEY=...`）**

`python -m mva.service --db /tmp/mva/world.duckdb --chroma-dir /tmp/mva/chroma` → `curl http://127.0.0.1:8900/health` 应返回 `engine_ready`。

- [ ] **Step 8: 提交**

```bash
cd /home/fyf/fyf/PCL/Multi-Video-Analysis
git add src/mva/service/engine.py src/mva/service/__main__.py src/mva/service/__init__.py tests/unit/test_ingest_jobs.py tests/smoke/test_service_smoke.py
git commit -m "feat(service): AnalysisEngine (QueryService+ingest jobs) + uvicorn entrypoint"
```

---

## Task 8: OmniUAV — MvaClient

**Files:**
- Create: `omni-uav/utils/mva_client.py`
- Create: `omni-uav/tests/{__init__.py,conftest.py,test_mva_client.py}`

**Interfaces:**
- Produces: `MvaClient(base_url="http://127.0.0.1:8900", timeout=60)`：`is_alive()->bool`、`ingest_start(source,mode="offline",dataset=None,config=None)->str`、`ingest_status(job_id)->dict`、`answer(query,attachments=None,session_id=None)->dict`（键 `answer`/`groundings`/`plan`）。

- [ ] **Step 1: 写失败测试（monkeypatch requests）**

```python
# omni-uav/tests/test_mva_client.py
from utils.mva_client import MvaClient


class _Resp:
    def __init__(self, payload, code=200):
        self._p = payload; self.status_code = code
    def json(self): return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_is_alive_true(monkeypatch):
    c = MvaClient()
    monkeypatch.setattr(c._s, "get", lambda *a, **k: _Resp({"status": "ok", "engine_ready": True}))
    assert c.is_alive() is True


def test_is_alive_false_on_error(monkeypatch):
    c = MvaClient()
    def boom(*a, **k): raise ConnectionError("refused")
    monkeypatch.setattr(c._s, "get", boom)
    assert c.is_alive() is False


def test_answer_returns_payload(monkeypatch):
    c = MvaClient()
    monkeypatch.setattr(c._s, "post",
                        lambda *a, **k: _Resp({"answer": "3 艘船", "groundings": [], "plan": None}))
    assert c.answer("画面里有几艘船")["answer"] == "3 艘船"
```

- [ ] **Step 2: 运行确认失败**

先 `mkdir -p /home/fyf/fyf/PCL/Simulation-System/omni-uav/tests && touch /home/fyf/fyf/PCL/Simulation-System/omni-uav/tests/__init__.py`
Run: `cd /home/fyf/fyf/PCL/Simulation-System/omni-uav && python -m pytest tests/test_mva_client.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'utils.mva_client'`）

- [ ] **Step 3: 实现 MvaClient + conftest**

```python
# omni-uav/utils/mva_client.py
"""OmniUAV → MVA sidecar 的 RPC 客户端(requests 封装)。契约见 spec §5。图像只传路径。"""
from __future__ import annotations
import os
from typing import Any, Optional

import requests

DEFAULT_BASE = os.environ.get("MVA_SIDECAR_URL", "http://127.0.0.1:8900")


class MvaClient:
    def __init__(self, base_url: str = DEFAULT_BASE, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._s = requests.Session()

    def is_alive(self) -> bool:
        try:
            r = self._s.get(f"{self.base_url}/health", timeout=3)
            r.raise_for_status()
            return bool(r.json().get("engine_ready", False))
        except Exception:                                # noqa: BLE001
            return False

    def ingest_start(self, source: str, mode: str = "offline",
                     dataset: Optional[str] = None, config: Optional[dict] = None) -> str:
        r = self._s.post(f"{self.base_url}/ingest/start",
                         json={"source": source, "mode": mode,
                               "dataset": dataset, "config": config or {}},
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json()["job_id"]

    def ingest_status(self, job_id: str) -> dict[str, Any]:
        r = self._s.get(f"{self.base_url}/ingest/status",
                        params={"job": job_id}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def answer(self, query: str, attachments: Optional[list[str]] = None,
               session_id: Optional[str] = None) -> dict[str, Any]:
        r = self._s.post(f"{self.base_url}/answer",
                         json={"query": query, "attachments": attachments or [],
                               "session_id": session_id},
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json()
```

```python
# omni-uav/tests/conftest.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/Simulation-System/omni-uav && python -m pytest tests/test_mva_client.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/Simulation-System/omni-uav
git add utils/mva_client.py tests/__init__.py tests/conftest.py tests/test_mva_client.py
git commit -m "feat(omni-uav): add MvaClient RPC wrapper for MVA sidecar"
```

---

## Task 9: OmniUAV — 问答分支路由 + 引擎状态灯 + 降级

**Files:**
- Modify: `omni-uav/app.py`（`MainWindow`：构造 `MvaClient`、状态灯、改**问答分支** `_execute_regular_query`）
- Create: `omni-uav/tests/test_qa_routing.py`

**Interfaces:**
- Consumes: `MvaClient`（Task 8）。
- Produces: 纯函数 `decide_qa_route(engine_alive: bool) -> str`（`"sidecar"`/`"local"`）；`MainWindow` 新增 `self.mva_client`、`self.engine_status_label`、`self._engine_alive`。

> **只改问答分支**：`_execute_regular_query`（intent 判定后 `is_tracking=false` 的那支）。跟踪/检测/重建等指令分支**不动**。

- [ ] **Step 1: 写失败测试（纯函数，无 Qt）**

```python
# omni-uav/tests/test_qa_routing.py
from app import decide_qa_route


def test_route_sidecar_when_alive():
    assert decide_qa_route(True) == "sidecar"


def test_route_local_when_dead():
    assert decide_qa_route(False) == "local"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/fyf/fyf/PCL/Simulation-System/omni-uav && QT_QPA_PLATFORM=offscreen python -m pytest tests/test_qa_routing.py -q`
Expected: FAIL（`ImportError: cannot import name 'decide_qa_route'`）

- [ ] **Step 3: 加纯函数 + 接线**

`app.py` 模块顶层（`MainWindow` 定义前）加：

```python
# [MOD 2026-07-10 | P0 grounded问答] 问答路由：引擎在→走 MVA sidecar，否则本地降级。
def decide_qa_route(engine_alive: bool) -> str:
    return "sidecar" if engine_alive else "local"
```

`MainWindow.__init__`（`self.llm_client` 建好后）加：

```python
        # [MOD 2026-07-10 | P0] MVA sidecar 客户端 + 引擎状态灯(仅问答用；指令分支不受影响)
        from utils.mva_client import MvaClient
        self.mva_client = MvaClient()
        self._engine_alive = False
        self.engine_status_label = QtWidgets.QLabel("引擎:检测中…")
        self.statusBar().addPermanentWidget(self.engine_status_label)
        self._engine_timer = QtCore.QTimer(self)
        self._engine_timer.timeout.connect(self._refresh_engine_status)
        self._engine_timer.start(5000)
        self._refresh_engine_status()
```

加方法：

```python
    def _refresh_engine_status(self):
        self._engine_alive = self.mva_client.is_alive()
        self.engine_status_label.setText("引擎●已连接" if self._engine_alive else "引擎○未连接")
```

在 `_execute_regular_query` **开头**（收集 image_paths / `_enqueue_llm_request` 之前）插入 sidecar 分流；失败或引擎不在则继续走下面原有本地逻辑：

```python
        # [MOD 2026-07-10 | P0] 问答分支路由：引擎在→MVA grounded 问答；否则落到原本地路径降级
        if decide_qa_route(getattr(self, "_engine_alive", False)) == "sidecar":
            try:
                result = self.mva_client.answer(self._original_user_prompt)
                ans = result.get("answer", "")
                g = result.get("groundings") or []
                src = ("  [溯源] " + ", ".join(
                    f"{x.get('view_id')}@{x.get('t')}" for x in g)) if g else ""
                self.llm_output.append(f"[grounded] {ans}{src}")
                return
            except Exception as e:                        # noqa: BLE001
                print(f"[P0] sidecar 问答失败，降级本地: {e}")
                # 不 return，继续走下面原有本地路径
```

（`_execute_regular_query` 其余逻辑、以及跟踪/指令分支，全部保持原样。）

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/fyf/fyf/PCL/Simulation-System/omni-uav && QT_QPA_PLATFORM=offscreen python -m pytest tests/test_qa_routing.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
cd /home/fyf/fyf/PCL/Simulation-System/omni-uav
git add app.py tests/test_qa_routing.py
git commit -m "feat(omni-uav): route QA branch through MVA sidecar with local fallback + engine status"
```

---

## Task 10: 启动/停止集成（Simulation-System，非 git）

**Files:**
- Create: `Simulation-System/start_mva_sidecar.sh`
- Modify: `Simulation-System/stop_live_demo.sh`
- Modify: `Simulation-System/MODIFICATIONS.md`

**Interfaces:**
- Produces: `start_mva_sidecar.sh` 在 MVA env 内起 `python -m mva.service`；`stop_live_demo.sh` 用 `pkill -f 'mva[.]service'` 停它（正则打断字面量，避免 pkill 自匹配）。

- [ ] **Step 1: 写 sidecar 启动脚本**

```bash
# Simulation-System/start_mva_sidecar.sh
#!/bin/bash
# [MOD 2026-07-10 | P0] 在 MVA env 内拉起 sidecar(FastAPI :8900)。
# 需先: MVA env 已 pip install -e '.[service,storage,llm]'；export DASHSCOPE_API_KEY=...
set -u
MVA=/home/fyf/fyf/PCL/Multi-Video-Analysis
DB=${MVA_DB:-/tmp/mva/world.duckdb}
CHROMA=${MVA_CHROMA:-/tmp/mva/chroma}
LOGDIR=/tmp/sim_live_logs; mkdir -p "$LOGDIR" "$(dirname "$DB")" "$CHROMA"
: "${DASHSCOPE_API_KEY:?请先 export DASHSCOPE_API_KEY=<你的key>}"

if ( ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null ) | grep -q ":8900"; then
  echo "sidecar 已在运行 (:8900)"; exit 0
fi
cd "$MVA" || exit 1
nohup python -m mva.service --db "$DB" --chroma-dir "$CHROMA" \
  > "$LOGDIR/mva_sidecar.log" 2>&1 &
echo "sidecar 启动中… 日志 $LOGDIR/mva_sidecar.log (探活: curl http://127.0.0.1:8900/health)"
```

`chmod +x Simulation-System/start_mva_sidecar.sh`

- [ ] **Step 2: stop 脚本加停 sidecar**

`Simulation-System/stop_live_demo.sh` 的"停止 OmniUAV"之后加：

```bash
echo "停止 MVA sidecar…"
pkill -9 -f 'mva[.]service' 2>/dev/null   # 正则打断字面量，避免 pkill 自匹配
```

- [ ] **Step 3: 手动验证起停**

```bash
export DASHSCOPE_API_KEY=<你的key>
bash /home/fyf/fyf/PCL/Simulation-System/start_mva_sidecar.sh
sleep 5 && curl -s http://127.0.0.1:8900/health   # 期望 {"status":"ok",...}
bash /home/fyf/fyf/PCL/Simulation-System/stop_live_demo.sh
curl -s http://127.0.0.1:8900/health || echo "已停(连接被拒=符合预期)"
```
Expected: 起后 `/health` 返回 JSON；停后连接被拒。

- [ ] **Step 4: 记录到 MODIFICATIONS.md**

`Simulation-System/MODIFICATIONS.md` 末尾追加：

```markdown
## P0 MVA sidecar 集成（2026-07-10）
- **新增** `start_mva_sidecar.sh`：在 MVA env 内起 FastAPI sidecar(:8900)，封装 grounded 问答/入库。
- **改** `stop_live_demo.sh`：新增 `pkill -9 -f 'mva[.]service'` 停 sidecar。
- OmniUAV 侧：问答分支改走 sidecar(引擎掉线降级)；指令分支不变。
- 依赖：MVA `pip install -e '.[service]'`；env `DASHSCOPE_API_KEY`。
- 详见 `docs/superpowers/plans/2026-07-10-p0-mva-sidecar-grounded-qa.md`。
```

（本目录非 git，无需 commit；MVA/omni-uav 改动已各自提交。）

---

## Self-Review（已执行）

- **Spec 覆盖**：对应 spec §9 P0 骨架 + D4 云端问答（方案 A：agent 在 MVA + 云端适配器 Task 5/6/7；OmniUAV 只路由问答分支 Task 9）。`/retrieve`+`/state/*`（P1 面板）、geo/关系（P2）**有意不在本计划**，属后续计划。
- **占位符扫描**：无 "TODO/待填"。Task 5/7 两处"先读 `client.py` / 先 `mva ingest --help` 确认签名/flag"是对**既有代码/外部 CLI 的确认指令**（非逻辑占位），其余代码完整。
- **类型一致**：`EngineProtocol` 五方法在 `FakeEngine`(Task 2) 与 `AnalysisEngine`(Task 7) 一致；`AnswerResponse{answer,groundings,plan}`、`IngestStatusResponse{state,processed_segments,total_segments,current_t,error}` 全程一致；`MvaClient.answer()` 返回 dict 键与 `AnswerResponse` 对齐；`decide_qa_route` 在 Task 9 定义并被测试与 `_execute_regular_query` 消费。

## 已知风险/后续

- **R-A**：`mva ingest` 的 dataset/scene/db/chroma 精确 flag 需实现者 `--help` 确认后微调 Task 7。
- **R-B**：云端适配器 P0 不支持视频段(`look_at` 走视频降级)；图像/文本问答不受影响。
- **R-C**：P0 `AnswerResponse.groundings` 暂空，后续从 `OrchestratorResult.invocations` 提取(P1)。
- **R-D**：sidecar 加载本地嵌入 8B(~16–18GB)；离线主力模式无仿真不冲突，实时模式需时间片(P3)。

---
*P0 计划完。P1(检索/跨视角面板 + /retrieve + /state/*)、P2(空间关系理解)、P3(近实时)各自单独出计划。*
