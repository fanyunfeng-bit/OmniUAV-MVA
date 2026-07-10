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


def test_inprocess_ingest_requires_embedder(tmp_path):
    """无 chroma/embedder 时，进程内入库应报错(不是静默成功)。无 GPU 可测。"""
    eng = AnalysisEngine(db_path=str(tmp_path / "w.duckdb"), chroma_dir=None,
                         defer_query_service=True)

    class _FakeSvc:              # 预置一个"未加载嵌入"的 svc，绕过真 QueryService 构造
        embedder = None
        vstore = None
        store = None
    eng._svc = _FakeSvc()

    start = eng.ingest_start(IngestRequest(source="Reservoir", dataset="pcl-sim",
                                           config={"dataset_root": "/x"}))
    for _ in range(50):
        if eng.ingest_status(start.job_id).state in ("done", "error"):
            break
        time.sleep(0.02)
    st = eng.ingest_status(start.job_id)
    assert st.state == "error"
    assert "嵌入" in (st.error or "")
