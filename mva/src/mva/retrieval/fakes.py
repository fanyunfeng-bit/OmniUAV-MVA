"""M5 桩：Phase 0 并行用。真实嵌入/检索由 M5 owner 提供
（现有 MultimodalEmbedder + service 检索已可用，本桩仅示接口）。"""
from __future__ import annotations

from typing import Any

from mva.service.models import RetrieveRequest, RetrieveResponse


class ZeroEmbedder:
    """桩：任何输入都返回定长零向量。"""
    def __init__(self, dim: int = 768):
        self.dim = dim

    def encode_text(self, text: Any) -> list[float]:
        return [0.0] * self.dim

    def encode_image(self, image: Any) -> list[float]:
        return [0.0] * self.dim

    def encode_images(self, images: Any) -> list[float]:
        return [0.0] * self.dim


class EmptyRetriever:
    """桩：任何查询都返回空命中。"""
    def retrieve(self, req: RetrieveRequest) -> RetrieveResponse:
        return RetrieveResponse(hits=[], n_vectors_searched=0)
