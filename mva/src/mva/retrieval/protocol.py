"""M5 信息压缩 + 检索 Protocol：嵌入器（压缩）与检索器。

现有 `mva.l5_state.embedder.MultimodalEmbedder` 结构上满足 `Embedder`；
检索的查询解析已有 `mva.service.query_understanding.ConstraintParser`（另一 Protocol）。
注意与 `mva.service.retrieval`（纯逻辑模块）区分：本包是 M5 的可换接口。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mva.service.models import RetrieveRequest, RetrieveResponse


@runtime_checkable
class Embedder(Protocol):
    def encode_text(self, text: Any) -> list[float]: ...
    def encode_image(self, image: Any) -> list[float]: ...
    def encode_images(self, images: Any) -> list[float]: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, req: RetrieveRequest) -> RetrieveResponse: ...
