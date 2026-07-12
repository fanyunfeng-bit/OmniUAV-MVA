"""Regression: view-scoped query must match whether the caller passes the
raw view id ('cam02') or the scene-prefixed one ('Scene::cam02').

Ingest writes BOTH `view_id` (=scene::view, prefixed) and `view_id_raw`
(=view). Callers (planner/look_at/find_by_description) pass the RAW id, so
`VectorStore.query(view_id='cam02')` filtering only on the prefixed metadata
matched nothing → look_at abstained with no_segment.
"""
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


def _add_segment(store, seed, scene, view_raw):
    # mirrors ingest._add_segment_vector: prefixed view_id + raw view_id_raw
    store.add(_vec(seed), VECTOR_TYPE_FRAME, f"{scene}::{view_raw}", f"seg{seed}",
              extra_metadata={"vector_kind": "segment", "view_id_raw": view_raw,
                              "start_t": 0.0, "end_t": 10.0})


def test_query_by_raw_view_id_matches_prefixed_metadata(store):
    _add_segment(store, 1, "airsim_downtown_4view", "cam02")
    _add_segment(store, 2, "airsim_downtown_4view", "cam01")
    res = store.query(query_vector=_vec(9), vector_type=VECTOR_TYPE_FRAME,
                      view_id="cam02", top_k=5)          # 调用方传的是 raw 'cam02'
    assert len(res) == 1
    assert res[0]["metadata"]["view_id_raw"] == "cam02"


def test_query_by_prefixed_view_id_still_works(store):
    _add_segment(store, 1, "Scene", "cam02")
    res = store.query(query_vector=_vec(9), vector_type=VECTOR_TYPE_FRAME,
                      view_id="Scene::cam02", top_k=5)   # 传完整前缀也要命中
    assert len(res) == 1
    assert res[0]["metadata"]["view_id_raw"] == "cam02"
