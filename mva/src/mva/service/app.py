from __future__ import annotations
from fastapi import FastAPI, Response
from mva.service.models import (
    EngineProtocol, HealthResponse, AnswerRequest, AnswerResponse,
    IngestRequest, IngestStartResponse, IngestStatusResponse,
    RetrieveRequest, RetrieveResponse,
)


def create_app(engine: EngineProtocol) -> FastAPI:
    """构造 sidecar FastAPI 应用。engine 满足 EngineProtocol(生产=AnalysisEngine, 测试=FakeEngine)。"""
    app = FastAPI(title="MVA sidecar", version="0.0.1")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return engine.health()

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> AnswerResponse:
        return engine.answer(req)

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

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(req: RetrieveRequest) -> RetrieveResponse:
        return engine.retrieve(req)

    @app.post("/select_scene", status_code=204)
    def select_scene(scene: str) -> Response:
        engine.select_scene(scene)
        return Response(status_code=204)

    return app
