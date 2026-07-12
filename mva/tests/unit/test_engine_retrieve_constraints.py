from mva.service.engine import AnalysisEngine
from mva.service.models import RetrieveRequest
from mva.service.query_understanding import RuleBasedConstraintParser


class _FakeCollection:
    def __init__(self, n): self._n = n
    def count(self): return self._n


class _FakeVStore:
    def __init__(self, result_fn):
        self.collection = _FakeCollection(100)
        self.calls = []
        self._result_fn = result_fn
    def query(self, query_text=None, vector_type=None, top_k=10, where=None):
        self.calls.append({"query_text": query_text, "where": where})
        return self._result_fn(where)


class _FakeStore:
    def __init__(self, views, dur): self._views = views; self._dur = dur
    def execute_readonly(self, sql, *a, **k):
        if "DISTINCT view_id" in sql:
            return [{"view_id": v} for v in self._views]
        if "max(end_t)" in sql:
            return [{"dur": self._dur}]
        return [{"start_t": 0.0, "source_uri": None}]      # enrich_segment_time


class _FakeSvc:
    def __init__(self, vstore, store): self.vstore = vstore; self.store = store


def _seg_hit(view_raw="view1"):
    return [{"id": "x", "distance": 0.1, "document": f"{view_raw} [0-10s]",
             "metadata": {"view_id": f"Scene::{view_raw}", "view_id_raw": view_raw,
                          "segment_idx": 0, "vector_kind": "segment"}}]


def _engine(vstore, store):
    e = AnalysisEngine(db_path="/tmp/qcr/world.duckdb",
                       chroma_dir="/tmp/qcr/chroma", defer_query_service=True)
    e._svc = _FakeSvc(vstore, store)
    e._parser = RuleBasedConstraintParser()
    return e


def test_view_constraint_filters_and_embeds_residual():
    vs = _FakeVStore(lambda where: _seg_hit("view1"))
    e = _engine(vs, _FakeStore(["view1", "view2"], 180.0))
    out = e.retrieve(RetrieveRequest(text="视角1里的黄车", top_k=3))
    assert vs.calls[0]["where"] == {"view_id_raw": "view1"}
    assert vs.calls[0]["query_text"] == "黄车"
    assert out.applied.view_id == "view1"
    assert out.applied.source == "rule"
    assert out.applied.fell_back is False


def test_relative_time_uses_duration():
    vs = _FakeVStore(lambda where: _seg_hit("view1"))
    e = _engine(vs, _FakeStore(["view1"], 180.0))
    out = e.retrieve(RetrieveRequest(text="最后20秒的红色卡车", top_k=3))
    w = vs.calls[0]["where"]
    assert {"start_t": {"$lte": 180.0}} in w["$and"]
    assert {"end_t": {"$gte": 160.0}} in w["$and"]
    assert vs.calls[0]["query_text"] == "红色卡车"


def test_empty_hit_falls_back_to_full_library():
    # 带 where 返回空, 去 where 返回命中
    vs = _FakeVStore(lambda where: [] if where is not None else _seg_hit("view2"))
    e = _engine(vs, _FakeStore(["view1", "view2"], 180.0))
    out = e.retrieve(RetrieveRequest(text="视角1里的飞机", top_k=3))
    assert len(vs.calls) == 2
    assert vs.calls[0]["where"] == {"view_id_raw": "view1"}
    assert vs.calls[1]["where"] is None
    assert out.applied.fell_back is True
    assert len(out.hits) == 1


def test_plain_query_no_constraint_single_call():
    vs = _FakeVStore(lambda where: _seg_hit("view1"))
    e = _engine(vs, _FakeStore(["view1"], 180.0))
    out = e.retrieve(RetrieveRequest(text="黄车", top_k=3))
    assert len(vs.calls) == 1
    assert vs.calls[0]["where"] is None
    assert vs.calls[0]["query_text"] == "黄车"
    assert out.applied.source == "none"
    assert out.applied.fell_back is False
