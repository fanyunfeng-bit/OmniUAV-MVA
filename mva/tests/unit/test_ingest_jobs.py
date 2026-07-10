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
