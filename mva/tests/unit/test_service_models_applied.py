from mva.service.models import RetrieveResponse, RetrieveConstraints, RetrieveHit


def test_retrieve_response_applied_roundtrip():
    r = RetrieveResponse(
        hits=[RetrieveHit(view_id="view1", score=0.9)],
        n_vectors_searched=10,
        applied=RetrieveConstraints(view_id="view1", time_start=0.0, time_end=10.0,
                                    semantic_text="黄车", source="rule", fell_back=False),
    )
    d = r.model_dump()
    assert d["applied"]["view_id"] == "view1"
    assert d["applied"]["source"] == "rule"
    assert d["applied"]["fell_back"] is False


def test_retrieve_response_applied_optional():
    r = RetrieveResponse(hits=[], n_vectors_searched=0)
    assert r.applied is None
