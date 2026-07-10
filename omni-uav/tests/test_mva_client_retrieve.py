from utils.mva_client import MvaClient


class _Resp:
    status_code = 200
    def __init__(self, p): self._p = p
    def json(self): return self._p
    def raise_for_status(self): pass


def test_retrieve(monkeypatch):
    c = MvaClient()
    monkeypatch.setattr(c._s, "post",
        lambda *a, **k: _Resp({"hits": [{"view_id": "view1", "t": 0.0, "score": 0.9}],
                               "n_vectors_searched": 28}))
    out = c.retrieve(text="airplane")
    assert out["n_vectors_searched"] == 28
    assert out["hits"][0]["view_id"] == "view1"
