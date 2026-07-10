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
