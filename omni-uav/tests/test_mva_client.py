from utils.mva_client import MvaClient


class _Resp:
    def __init__(self, payload, code=200):
        self._p = payload; self.status_code = code
    def json(self): return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_is_alive_true(monkeypatch):
    c = MvaClient()
    monkeypatch.setattr(c._s, "get", lambda *a, **k: _Resp({"status": "ok", "engine_ready": True}))
    assert c.is_alive() is True


def test_is_alive_false_on_error(monkeypatch):
    c = MvaClient()
    def boom(*a, **k): raise ConnectionError("refused")
    monkeypatch.setattr(c._s, "get", boom)
    assert c.is_alive() is False


def test_answer_returns_payload(monkeypatch):
    c = MvaClient()
    monkeypatch.setattr(c._s, "post",
                        lambda *a, **k: _Resp({"answer": "3 艘船", "groundings": [], "plan": None}))
    assert c.answer("画面里有几艘船")["answer"] == "3 艘船"
