"""L5 vector store (ChromaDB, single-collection design per §3.2 L5 / Eng Review 1A).

One collection `tracklets_embeddings`. Every row carries metadata:
  vector_type ∈ {text, frame, reid}    — what the embedding represents
  view_id                              — which stream produced it
  tracklet_id                          — which tracklet inside that stream

Queries combine `query_embeddings` (or `query_texts`) with a metadata `where`
filter, so the same collection serves single-view text search, frame search,
and ReID-by-appearance lookups (the folded-in L1.5 use case per 1B).

Persistence: pass `persist_dir`. Without it we still create a PersistentClient
in a fresh temp directory — chromadb 0.5+ removed EphemeralClient from the
public surface so we just point it at a tmp path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional


VECTOR_TYPE_TEXT = "text"
VECTOR_TYPE_FRAME = "frame"
VECTOR_TYPE_REID = "reid"

_VECTOR_TYPES = {VECTOR_TYPE_TEXT, VECTOR_TYPE_FRAME, VECTOR_TYPE_REID}


class VectorStore:
    """Single-collection multimodal vector store backed by ChromaDB."""

    COLLECTION = "tracklets_embeddings"

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_function: Optional[Any] = None,
    ) -> None:
        """Open or create the `tracklets_embeddings` collection.

        Parameters
        ----------
        persist_dir : str | None
            Directory on disk to persist the collection. If None we use a
            fresh temp directory (state lost when the process exits — useful
            for tests).
        embedding_function : optional ChromaDB embedding function
            Only needed if you intend to call `query(query_text=...)`. For
            `query_vector=...` queries no embedder is invoked.
        """
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "chromadb is required for VectorStore. "
                "Install with: pip install 'mva[storage]'"
            ) from exc

        if persist_dir is None:
            persist_dir = tempfile.mkdtemp(prefix="mva-chroma-")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir

        self.client = chromadb.PersistentClient(path=persist_dir)
        kwargs: dict[str, Any] = {}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self.collection = self.client.get_or_create_collection(
            self.COLLECTION, **kwargs
        )

    # ---- writes ----------------------------------------------------------

    def add(
        self,
        vector: list[float],
        vector_type: str,
        view_id: str,
        tracklet_id: str,
        extra_metadata: Optional[dict] = None,
        document: Optional[str] = None,
        upsert: bool = True,
    ) -> str:
        """Insert one vector. Returns the assigned ChromaDB id.

        Id is deterministic: `{view_id}::{tracklet_id}::{vector_type}` plus a
        suffix if `extra_metadata` carries a `chunk_id` (e.g. multiple frame
        embeddings per tracklet).

        M3.4: default `upsert=True` uses `collection.upsert` so re-running
        `mva ingest` on the same scene replaces the row instead of crashing
        with a duplicate-id error (PROBLEMS P2-04). Set `upsert=False`
        to get the strict M2.x behavior (`collection.add`).
        """
        if vector_type not in _VECTOR_TYPES:
            raise ValueError(
                f"vector_type must be one of {_VECTOR_TYPES}, got {vector_type!r}"
            )
        meta: dict[str, Any] = {
            "vector_type": vector_type,
            "view_id": view_id,
            "tracklet_id": tracklet_id,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        emb_id = f"{view_id}::{tracklet_id}::{vector_type}"
        chunk = meta.get("chunk_id")
        if chunk is not None:
            emb_id = f"{emb_id}::{chunk}"

        add_kwargs: dict[str, Any] = {
            "ids": [emb_id],
            "embeddings": [list(vector)],
            "metadatas": [meta],
        }
        if document is not None:
            add_kwargs["documents"] = [document]
        if upsert:
            self.collection.upsert(**add_kwargs)
        else:
            self.collection.add(**add_kwargs)
        return emb_id

    def delete(self, ids: list[str]) -> None:
        """Delete vectors by id. Used by the live-ingest worker's FIFO
        eviction to keep the rolling window bounded. No-op on empty list."""
        ids = [i for i in (ids or []) if i]
        if ids:
            self.collection.delete(ids=ids)

    # ---- reads -----------------------------------------------------------

    def query(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        vector_type: Optional[str] = None,
        view_id: Optional[str] = None,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Single-collection query with optional metadata filter.

        Exactly one of `query_vector` / `query_text` must be provided.

        For ReID-by-image lookups (the folded-in L1.5 use case), encode the
        crop externally (with the L1 ReID model) and pass the embedding as
        `query_vector` with `vector_type="reid"`.

        Returns a list of dicts shaped:
            {"id": str, "distance": float, "metadata": dict, "document": str|None}
        sorted by distance ascending (closest first).
        """
        if query_vector is None and query_text is None:
            raise ValueError(
                "VectorStore.query requires query_vector or query_text"
            )

        combined = self._build_where(vector_type, view_id, where)
        kwargs: dict[str, Any] = {"n_results": top_k}
        if combined is not None:
            kwargs["where"] = combined
        if query_vector is not None:
            kwargs["query_embeddings"] = [list(query_vector)]
        else:
            kwargs["query_texts"] = [query_text]

        result = self.collection.query(**kwargs)
        ids = result.get("ids", [[]])[0]
        distances = (result.get("distances") or [[None] * len(ids)])[0]
        metadatas = (result.get("metadatas") or [[{}] * len(ids)])[0]
        documents = (result.get("documents") or [[None] * len(ids)])[0]

        return [
            {
                "id": ids[i],
                "distance": distances[i],
                "metadata": metadatas[i] or {},
                "document": documents[i],
            }
            for i in range(len(ids))
        ]

    @staticmethod
    def _build_where(
        vector_type: Optional[str],
        view_id: Optional[str],
        extra: Optional[dict] = None,
    ) -> Optional[dict]:
        clauses: list[dict] = []
        if vector_type is not None:
            clauses.append({"vector_type": vector_type})
        if view_id is not None:
            clauses.append({"view_id": view_id})
        if extra:
            if "$and" in extra:
                clauses.extend(extra["$and"])
            else:
                for k, v in extra.items():
                    clauses.append({k: v})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
