"""Unit tests for L5 MultimodalEmbedder (Qwen3-VL-Embedding wrapper).

All tests run in mock mode (`model_path=None`) so the 16 GB Qwen3-VL-
Embedding-8B model is never loaded in CI. Mock mode produces deterministic
SHA-256-seeded unit vectors, which is enough to verify:
  - return shape (dim) + L2-normalization invariant
  - text vs image vs multi-frame routes produce distinct vectors
  - same input produces same vector (determinism)
  - ChromaDB embedding-function adapter follows the protocol shape
  - dim out-of-range rejected
  - lifecycle (is_loaded / unload)

The end-to-end test that actually loads Qwen3-VL-Embedding-8B lives in
demo_matrix.py (manual run) — not in pytest.
"""
from __future__ import annotations

import numpy as np
import pytest

from mva.l5_state import DEFAULT_DIM, MultimodalEmbedder


def _is_unit(v) -> bool:
    arr = np.asarray(v, dtype=np.float64)
    return abs(np.linalg.norm(arr) - 1.0) < 1e-5


@pytest.fixture
def embedder():
    return MultimodalEmbedder(model_path=None)


# ----------------------------------------------------------------------
# Shape + normalization
# ----------------------------------------------------------------------


def test_default_dim_is_768():
    assert DEFAULT_DIM == 768


def test_encode_text_returns_unit_vector_of_dim(embedder):
    v = embedder.encode_text("red car")
    assert isinstance(v, list)
    assert len(v) == DEFAULT_DIM
    assert _is_unit(v)


def test_encode_text_batch_returns_list_of_unit_vectors(embedder):
    vs = embedder.encode_text(["red car", "blue truck", "person"])
    assert isinstance(vs, list)
    assert len(vs) == 3
    for v in vs:
        assert len(v) == DEFAULT_DIM
        assert _is_unit(v)


def test_encode_image_returns_unit_vector_of_dim(embedder):
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    v = embedder.encode_image(img)
    assert len(v) == DEFAULT_DIM
    assert _is_unit(v)


def test_encode_images_mean_pools_and_normalizes(embedder):
    imgs = [
        np.full((32, 32, 3), c, dtype=np.uint8) for c in (50, 100, 150, 200)
    ]
    v = embedder.encode_images(imgs)
    assert len(v) == DEFAULT_DIM
    assert _is_unit(v)


def test_encode_images_multiple_works(embedder):
    imgs = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    v = embedder.encode_images(imgs)
    assert len(v) == DEFAULT_DIM


def test_encode_images_empty_raises(embedder):
    with pytest.raises(ValueError):
        embedder.encode_images([])


# ----------------------------------------------------------------------
# Determinism + cross-modality distinctness
# ----------------------------------------------------------------------


def test_same_text_same_vector(embedder):
    v1 = embedder.encode_text("hello")
    v2 = embedder.encode_text("hello")
    assert v1 == v2


def test_different_text_different_vector(embedder):
    v1 = embedder.encode_text("hello")
    v2 = embedder.encode_text("world")
    assert v1 != v2


def test_text_and_image_routes_are_distinct(embedder):
    # Even if seed-bytes happened to collide, they should produce different vectors
    # because we prefix with "text:" / "img:" before hashing.
    img = np.full((4, 4, 3), 0, dtype=np.uint8)
    v_text = embedder.encode_text("0")
    v_img = embedder.encode_image(img)
    assert v_text != v_img


# ----------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------


def test_dim_too_small_raises():
    with pytest.raises(ValueError):
        MultimodalEmbedder(dim=32)


def test_dim_too_large_raises():
    with pytest.raises(ValueError):
        MultimodalEmbedder(dim=8192)


def test_custom_dim_supported():
    e = MultimodalEmbedder(model_path=None, dim=256)
    v = e.encode_text("x")
    assert len(v) == 256


# ----------------------------------------------------------------------
# Lifecycle (mock mode — _model stays None)
# ----------------------------------------------------------------------


def test_mock_is_mock(embedder):
    assert embedder.is_mock is True
    assert embedder.is_loaded is False


def test_unload_is_safe_when_not_loaded(embedder):
    embedder.unload()
    embedder.unload()      # idempotent
    assert embedder.is_loaded is False


# ----------------------------------------------------------------------
# ChromaDB EmbeddingFunction adapter
# ----------------------------------------------------------------------


def test_chromadb_adapter_returns_list_of_lists(embedder):
    fn = embedder.as_chromadb_embedding_function()
    out = fn(["red car", "blue truck"])
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(len(v) == DEFAULT_DIM for v in out)
    assert all(_is_unit(v) for v in out)


def test_chromadb_adapter_has_name_method(embedder):
    fn = embedder.as_chromadb_embedding_function()
    assert "qwen" in fn.name().lower()


def test_chromadb_adapter_single_input(embedder):
    # ChromaDB may pass a single-element list
    fn = embedder.as_chromadb_embedding_function()
    out = fn(["only one"])
    assert len(out) == 1
    assert len(out[0]) == DEFAULT_DIM


# ----------------------------------------------------------------------
# Integration with VectorStore (mock embedder + real ChromaDB)
# ----------------------------------------------------------------------


def test_vector_store_uses_mock_embedder_for_query_text(tmp_path):
    """VectorStore.query(query_text=...) routes through the supplied
    embedding function, not ChromaDB's default. Verified by adding two
    text rows + querying with the second one's exact text → it should be
    the top result (distance 0)."""
    from mva.l5_state import VectorStore

    embedder = MultimodalEmbedder(model_path=None)
    vstore = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedding_function=embedder.as_chromadb_embedding_function(),
    )

    # Add via pre-computed vectors so we control what goes in (mock vectors
    # are deterministic, so encoding "red car" elsewhere matches what we
    # add here).
    v_red = embedder.encode_text("red car")
    v_blue = embedder.encode_text("blue truck")
    vstore.add(v_red, "text", "drone-1", "tk-1")
    vstore.add(v_blue, "text", "drone-1", "tk-2")

    results = vstore.query(query_text="red car", top_k=2)
    assert len(results) == 2
    # closest is the row we encoded with the same text
    assert results[0]["metadata"]["tracklet_id"] == "tk-1"
