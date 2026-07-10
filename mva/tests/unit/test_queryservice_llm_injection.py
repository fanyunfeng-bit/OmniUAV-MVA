import tempfile
from mva.cli.query import QueryService


class _StubLLM:
    def complete(self, prompt, images=None, max_new_tokens=256):
        return "stub-answer"
    def complete_messages(self, messages, max_new_tokens=256):
        return "stub-answer"
    def unload(self):
        return None


def test_injected_llm_is_used():
    with tempfile.TemporaryDirectory() as d:
        svc = QueryService(db_path=f"{d}/w.duckdb", llm=_StubLLM())  # 无 chroma → 不加载嵌入
        assert svc.llm is not None
        assert svc.llm.complete("x") == "stub-answer"
        svc.close()


def test_injected_embedder_not_unloaded_on_close():
    class _Emb:
        def unload(self):
            raise AssertionError("不应 unload 注入的 embedder(engine 复用)")

    with tempfile.TemporaryDirectory() as d:
        svc = QueryService(db_path=f"{d}/w.duckdb", llm=_StubLLM(), embedder=_Emb())
        assert svc.embedder is not None
        assert svc._own_embedder is False
        svc.close()   # 不应对注入的 embedder 调 unload
