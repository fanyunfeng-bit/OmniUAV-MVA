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

        job.status.state = "running"          # 先置 running，再起线程，避免与线程的 done 竞争
        threading.Thread(target=run, daemon=True).start()
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
