"""Multimodal embedder backed by Qwen3-VL-Embedding-8B.

Feeds vectors into L5 VectorStore (ChromaDB). Replaces ChromaDB's default
onnx-MiniLM-L6-V2 embedder, which is English-leaning and unaware of
images. Aligned with sentrysearch's convention: MRL-truncated to 768
dimensions + L2-normalized, so cosine similarity == inner product and
storage is 3 KB per vector.

Lazy-load + explicit unload — Qwen3-VL-Embedding-8B (~16-18 GB VRAM) cannot
share the RTX 3090 with Qwen2.5-VL-7B-Instruct (~14 GB) the generative
LLM. Workflow is **time-isolated**:

    embed pass: load embedder → encode → write to ChromaDB → embedder.unload()
    gen pass:   load LLMClient → Mode A REPL

Mock mode (`model_path=None`) returns deterministic seeded vectors so
unit tests don't trigger the 16 GB model download.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Optional

import numpy as np


DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_DIM = 768                 # matches sentrysearch (MRL truncated)


def _seeded_vector(seed_bytes: bytes, dim: int) -> list[float]:
    """Deterministic mock vector for tests. L2-normalized."""
    h = hashlib.sha256(seed_bytes).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


class MultimodalEmbedder:
    """Wrap Qwen3-VL-Embedding-8B as a uniform text/image/video encoder.

    Parameters
    ----------
    model_path : str | None
        HuggingFace model id (e.g. "Qwen/Qwen3-VL-Embedding-8B") or local
        path. `None` enables mock mode: every encode returns a
        deterministic vector seeded by SHA-256 of the input, so tests
        and scaffolding work without the 16 GB model.
    dim : int
        MRL truncation dimensionality. 768 matches sentrysearch's
        convention; valid range 64–4096 per Qwen3-VL-Embedding's MRL
        training.
    device : str | None
        Torch device, e.g. "cuda", "cuda:0", "cpu". None lets
        sentence-transformers pick.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        dim: int = DEFAULT_DIM,
        device: Optional[str] = None,
    ) -> None:
        if not (64 <= dim <= 4096):
            raise ValueError(
                f"dim must be in [64, 4096] for Qwen3-VL-Embedding MRL, got {dim}"
            )
        self.model_path = model_path
        self.dim = dim
        self.device = device
        self._model: Any = None

    @property
    def is_mock(self) -> bool:
        return self.model_path is None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ----------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None or self.is_mock:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for the real "
                "MultimodalEmbedder. Install with: pip install 'mva[llm]'"
            ) from exc
        self._model = SentenceTransformer(
            self.model_path,
            truncate_dim=self.dim,
            device=self.device,
        )

    def unload(self) -> None:
        """Free the model's GPU memory. Safe to call multiple times."""
        if self._model is None:
            return
        try:
            del self._model
        finally:
            self._model = None
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    # ----------------------------------------------------------------------
    # Encoding API
    # ----------------------------------------------------------------------

    def encode_text(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Encode one string or a batch of strings → L2-normalized vector(s)."""
        if isinstance(text, str):
            if self.is_mock:
                return _seeded_vector(("text:" + text).encode("utf-8"), self.dim)
            self._ensure_loaded()
            arr = self._model.encode(
                text, normalize_embeddings=True, convert_to_numpy=True,
            )
            return arr.astype(np.float32).tolist()

        # Batch
        if self.is_mock:
            return [_seeded_vector(("text:" + t).encode("utf-8"), self.dim) for t in text]
        self._ensure_loaded()
        arr = self._model.encode(
            list(text), normalize_embeddings=True, convert_to_numpy=True,
        )
        return arr.astype(np.float32).tolist()

    def encode_image(self, image: np.ndarray) -> list[float]:
        """Encode a BGR numpy image → L2-normalized vector."""
        if self.is_mock:
            return _seeded_vector(("img:" + str(image.shape) + str(image.mean())).encode("utf-8"), self.dim)
        self._ensure_loaded()
        from PIL import Image  # type: ignore
        pil = Image.fromarray(image[:, :, ::-1])      # BGR → RGB
        arr = self._model.encode(
            pil, normalize_embeddings=True, convert_to_numpy=True,
        )
        return arr.astype(np.float32).tolist()

    def encode_images(self, images: Iterable[np.ndarray]) -> list[float]:
        """Encode a sequence of frames → one L2-normalized mean-pooled vector.

        Treats a sequence of frames as one "video" / one tracklet — averages
        per-frame embeddings (with the average re-normalized to unit length).
        Use for ReID-by-tracklet or video-chunk retrieval.
        """
        images = list(images)
        if not images:
            raise ValueError("encode_images called with no frames")
        if self.is_mock:
            stacked = np.stack(
                [_seeded_vector(("img:" + str(im.shape) + str(im.mean())).encode("utf-8"), self.dim)
                 for im in images]
            )
        else:
            self._ensure_loaded()
            from PIL import Image  # type: ignore
            pil_list = [Image.fromarray(im[:, :, ::-1]) for im in images]
            stacked = self._model.encode(
                pil_list, normalize_embeddings=False, convert_to_numpy=True,
            )
        mean = stacked.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-9)
        return mean.astype(np.float32).tolist()

    # ----------------------------------------------------------------------
    # ChromaDB adapter
    # ----------------------------------------------------------------------

    def as_chromadb_embedding_function(self) -> Any:
        """Return a ChromaDB-compatible EmbeddingFunction wrapping this embedder.

        Lets VectorStore's `query(query_text=...)` route through Qwen
        instead of the default onnx-MiniLM. Pass the result to
        `VectorStore(embedding_function=embedder.as_chromadb_embedding_function())`.
        """
        return _ChromaTextEmbeddingFunction(self)


class _ChromaTextEmbeddingFunction:
    """ChromaDB EmbeddingFunction adapter.

    ChromaDB 0.5+ split the legacy `__call__(input)` interface into
    `embed_documents(input)` and `embed_query(input)`. We expose all three
    (they all route to encode_text — the document/query split only matters
    for asymmetric models like Instructor; Qwen3-VL-Embedding is symmetric).
    """

    def __init__(self, embedder: "MultimodalEmbedder") -> None:
        self._embedder = embedder

    def _embed(self, input: list[str]) -> list[list[float]]:
        result = self._embedder.encode_text(list(input))
        # encode_text returns a single vector for single str, list-of-list for batch
        if isinstance(result, list) and result and isinstance(result[0], float):
            return [result]                            # type: ignore[list-item]
        return result                                  # type: ignore[return-value]

    def __call__(self, input: list[str]) -> list[list[float]]:    # legacy
        return self._embed(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    @staticmethod
    def name() -> str:
        return "qwen3-vl-embedding-mva"

    def default_space(self) -> str:                    # chromadb 1.x hook
        return "cosine"

    def supported_spaces(self) -> list[str]:           # chromadb 1.x hook
        return ["cosine"]

    def is_legacy(self) -> bool:                       # chromadb >= 0.5 hook
        return False
