def test_service_main_importable():
    import mva.service.__main__ as m
    assert hasattr(m, "main")


def test_engine_health_defer():
    from mva.service.engine import AnalysisEngine
    eng = AnalysisEngine(db_path="/tmp/x.duckdb", chroma_dir=None, defer_query_service=True)
    assert eng.health().engine_ready is False
