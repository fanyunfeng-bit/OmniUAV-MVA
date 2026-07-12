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
    applied: Optional[RetrieveConstraints] = None


@runtime_checkable
class EngineProtocol(Protocol):
    def health(self) -> HealthResponse: ...
    def ingest_start(self, req: IngestRequest) -> IngestStartResponse: ...
    def ingest_status(self, job_id: str) -> IngestStatusResponse: ...
    def ingest_stop(self, job_id: str) -> None: ...
    def answer(self, req: AnswerRequest) -> AnswerResponse: ...
    def retrieve(self, req: RetrieveRequest) -> RetrieveResponse: ...
    def select_scene(self, scene: str) -> None: ...
