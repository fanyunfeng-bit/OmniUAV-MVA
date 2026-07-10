from fastapi.testclient import TestClient
from mva.service.app import create_app
from tests.unit._fakes import FakeEngine


def test_retrieve_endpoint():
    c = TestClient(create_app(FakeEngine()))
    r = c.post("/retrieve", json={"text": "airplane", "top_k": 3})
    assert r.status_code == 200
    b = r.json()
    assert b["n_vectors_searched"] == 28
    assert b["hits"][0]["view_id"] == "view1"
    assert b["hits"][0]["thumbnail_path"] == "/tmp/thumb.jpg"
