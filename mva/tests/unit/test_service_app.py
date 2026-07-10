from fastapi.testclient import TestClient
from mva.service.app import create_app
from tests.unit._fakes import FakeEngine


def _client():
    return TestClient(create_app(FakeEngine()))


def test_health_ok():
    r = _client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine_ready"] is True


def test_answer_echo():
    r = _client().post("/answer", json={"query": "画面里有几艘船"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "echo:画面里有几艘船"
    assert body["groundings"][0]["view_id"] == "view1"
