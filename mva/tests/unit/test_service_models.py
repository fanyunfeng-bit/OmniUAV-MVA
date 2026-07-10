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


def test_retrieve_models():
    from mva.service.models import RetrieveRequest, RetrieveHit, RetrieveResponse
    req = RetrieveRequest(text="airplane")
    assert req.top_k == 3 and req.vector_type == "frame"
    resp = RetrieveResponse(
        hits=[RetrieveHit(view_id="view1", t=0.0, segment_idx=0, score=0.9, kind="segment")],
        n_vectors_searched=28,
    )
    d = resp.model_dump()
    assert d["hits"][0]["view_id"] == "view1"
    assert d["n_vectors_searched"] == 28
