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
