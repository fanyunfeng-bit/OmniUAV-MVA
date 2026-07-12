import numpy as np
import pytest
from mva.l5_state.chromadb_store import VectorStore, VECTOR_TYPE_FRAME


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma"))


def _vec(seed, dim=8):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-9)).tolist()


def _add(store, seed, view_raw, start_t, end_t):
    store.add(_vec(seed), VECTOR_TYPE_FRAME, f"Scene::{view_raw}", f"seg{seed}",
              extra_metadata={"vector_kind": "segment", "view_id_raw": view_raw,
                              "start_t": start_t, "end_t": end_t})


def test_build_where_merges_extra_flat_keys():
    w = VectorStore._build_where(VECTOR_TYPE_FRAME, None, {"view_id_raw": "view1"})
    assert w == {"$and": [{"vector_type": VECTOR_TYPE_FRAME},
                          {"view_id_raw": "view1"}]}


def test_build_where_expands_extra_and_no_double_nest():
    extra = {"$and": [{"start_t": {"$lte": 10.0}}, {"end_t": {"$gte": 0.0}}]}
    w = VectorStore._build_where(VECTOR_TYPE_FRAME, None, extra)
    assert w == {"$and": [{"vector_type": VECTOR_TYPE_FRAME},
                          {"start_t": {"$lte": 10.0}},
                          {"end_t": {"$gte": 0.0}}]}


def test_query_where_filters_by_view(store):
    _add(store, 1, "view1", 0.0, 10.0)
    _add(store, 2, "view2", 0.0, 10.0)
    res = store.query(query_vector=_vec(9), vector_type=VECTOR_TYPE_FRAME,
                      top_k=5, where={"view_id_raw": "view1"})
    assert len(res) == 1
    assert res[0]["metadata"]["view_id_raw"] == "view1"


def test_query_where_filters_by_time_overlap(store):
    _add(store, 1, "view1", 0.0, 10.0)
    _add(store, 2, "view1", 100.0, 110.0)
    where = {"$and": [{"start_t": {"$lte": 10.0}}, {"end_t": {"$gte": 0.0}}]}
    res = store.query(query_vector=_vec(9), vector_type=VECTOR_TYPE_FRAME,
                      top_k=5, where=where)
    assert len(res) == 1
    assert res[0]["metadata"]["start_t"] == 0.0
