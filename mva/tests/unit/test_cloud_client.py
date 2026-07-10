import numpy as np
from mva.l4_llm.cloud_client import DashScopeLLMClient


class _FakeResp:
    status_code = 200
    def json(self):
        return {"choices": [{"message": {"content": "3 艘船"}}]}
    def raise_for_status(self):
        pass


def test_complete_posts_and_parses(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["auth"] = headers.get("Authorization")
        return _FakeResp()

    monkeypatch.setattr("mva.l4_llm.cloud_client.requests.post", fake_post)
    c = DashScopeLLMClient(model="qwen3-vl-plus", api_key="sk-test")
    out = c.complete("画面里有几艘船", images=[np.zeros((4, 4, 3), dtype=np.uint8)])

    assert out == "3 艘船"
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer sk-test"
    content = captured["json"]["messages"][-1]["content"]
    assert any(part.get("type") == "image_url" for part in content)
    assert captured["json"]["model"] == "qwen3-vl-plus"
