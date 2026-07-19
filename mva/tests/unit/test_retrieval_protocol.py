from mva.retrieval import Embedder, Retriever
from mva.retrieval.fakes import ZeroEmbedder, EmptyRetriever
from mva.service.models import RetrieveRequest, RetrieveResponse


def test_fakes_satisfy_protocols():
    assert isinstance(ZeroEmbedder(dim=8), Embedder)
    assert isinstance(EmptyRetriever(), Retriever)


def test_zero_embedder_shapes():
    e = ZeroEmbedder(dim=8)
    assert len(e.encode_text("x")) == 8
    assert len(e.encode_image(object())) == 8
    assert len(e.encode_images([object(), object()])) == 8


def test_empty_retriever_returns_response():
    res = EmptyRetriever().retrieve(RetrieveRequest(text="airplane"))
    assert isinstance(res, RetrieveResponse)
    assert res.hits == [] and res.n_vectors_searched == 0


def test_existing_embedder_class_satisfies_interface():
    # 现有 MultimodalEmbedder 结构上满足 Embedder（类可 import，不实例化=不加载 16G）
    from mva.l5_state.embedder import MultimodalEmbedder
    for m in ("encode_text", "encode_image", "encode_images"):
        assert hasattr(MultimodalEmbedder, m)
