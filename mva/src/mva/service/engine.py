"""真实引擎：QueryService(本地嵌入 + 云端 LLM) + 进程内 ingest + 任务表。

入库在 sidecar 进程内进行(复用 QueryService 已加载的 store/embedder/vstore)，
避开 DuckDB 跨进程读写锁、也免得重复加载 16G 嵌入模型。
"""
from __future__ import annotations
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
        self.device = device
        self._runner = ingest_runner or self._inprocess_ingest
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

    def _inprocess_ingest(self, req: IngestRequest, progress: ProgressCb) -> None:
        """进程内入库：复用 sidecar 已加载的 store/embedder/vstore(避 DuckDB 锁 + 免重载嵌入)。

        config 可含: dataset_root, views, window_sec, stride_sec, nframes_per_segment,
        segments_per_view(0=全部), detect(bool), detect_model, detect_conf, tracker。
        """
        svc = self._ensure_service()
        if getattr(svc, "embedder", None) is None or getattr(svc, "vstore", None) is None:
            raise RuntimeError("sidecar 未加载嵌入/向量库：入库需带 --chroma-dir 启动 sidecar")

        from mva.datasets import get_adapter
        from mva.segmentation import SegmenterConfig
        from mva.cli.ingest import ingest_scene

        cfg = req.config or {}
        adapter = get_adapter(req.dataset or "pcl-sim", root=cfg.get("dataset_root"))
        scene = adapter.get_scene(req.source)
        view_ids = cfg.get("views") or scene.view_ids
        seg = SegmenterConfig(
            window_sec=float(cfg.get("window_sec", 10.0)),
            stride_sec=float(cfg.get("stride_sec", 10.0)),
            nframes_per_segment=int(cfg.get("nframes_per_segment", 4)),
        )

        detector = None
        if cfg.get("detect", True):
            from mva.l1_perception import Detector
            detector = Detector(
                model_name=cfg.get("detect_model", "yolo11n.pt"),
                conf=float(cfg.get("detect_conf", 0.25)),
                device=cfg.get("detect_device") or self.device,
            )

        spv = cfg.get("segments_per_view", 4)
        progress(processed_segments=0)
        stats = ingest_scene(
            adapter=adapter, scene_id=req.source, view_ids=view_ids, config=seg,
            embedder=svc.embedder, detector=detector, embed_bboxes=True,
            store=svc.store, vstore=svc.vstore,
            segments_per_view=(float("inf") if spv in (0, None) else float(spv)),
            track=bool(cfg.get("track", True)),
            tracker_algorithm=cfg.get("tracker", "iou_greedy"),
        )
        progress(processed_segments=int(stats.get("segments", 0)),
                 total_segments=int(stats.get("segments", 0)))

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

    def retrieve(self, req):
        # [P1] 段级检索:vstore.query → 富化段时间 → top-1 抽缩略图
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
                import hashlib
                key = hashlib.md5(f"{h['source_uri']}:{h['t']}".encode()).hexdigest()[:10]
                thumb = extract_frame(h["source_uri"], float(h["t"]),
                                      f"/tmp/mva_thumbs/{key}.jpg")
            out.append(RetrieveHit(
                view_id=h["view_id"], t=h.get("t"), segment_idx=h.get("segment_idx"),
                score=h["score"], kind=h["kind"], class_name=h.get("class_name"),
                doc=h.get("doc"), thumbnail_path=thumb,
            ))
        return RetrieveResponse(hits=out, n_vectors_searched=n_total)
