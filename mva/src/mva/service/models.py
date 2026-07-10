from __future__ import annotations
from typing import Any, Literal, Optional, Protocol, runtime_checkable
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    engine_ready: bool
    db_path: Optional[str] = None


class IngestRequest(BaseModel):
    source: str                                  # dataset scene id 或视频/图像目录
    dataset: Optional[str] = None                # MVA 数据集适配器名(如 pcl-sim)；None=按目录
    mode: Literal["offline", "live"] = "offline"
    config: dict[str, Any] = {}                  # 透传给 ingest 的旋钮(window_sec 等)


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
    attachments: list[str] = []                  # 图像/视频文件路径
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
