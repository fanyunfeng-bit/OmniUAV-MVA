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


def test_ingest_start_then_status():
    c = _client()
    r = c.post("/ingest/start", json={"source": "/data/scene1", "mode": "offline"})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    s = c.get("/ingest/status", params={"job": jid})
    assert s.status_code == 200
    assert s.json()["state"] == "running"


def test_ingest_stop():
    c = _client()
    jid = c.post("/ingest/start", json={"source": "/d"}).json()["job_id"]
    assert c.post("/ingest/stop", params={"job": jid}).status_code == 204
    assert c.get("/ingest/status", params={"job": jid}).json()["state"] == "done"


def test_select_scene_endpoint():
    assert _client().post("/select_scene", params={"scene": "sceneX"}).status_code == 204
