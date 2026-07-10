"""DatasetAdapter Protocol — the contract every dataset plugin must satisfy.

The Protocol intentionally allows two very different data shapes through
the same interface:

  - MATRIX-style: one large Scene with many synchronized drone views, has
    cross-view ID GT and per-frame calibration. supports_cross_view_linking=True.
  - MVU-Eval-style: many small Scenes (one per QA pair) each with 2-6
    transient "views" (video files), no cross-camera link semantics,
    but with multiple-choice QA pairs for accuracy evaluation.
    supports_qa_eval=True.

Adapter authors implement the Protocol and register in mva.datasets.registry.
The CLI / Orchestrator / eval pipeline are dataset-agnostic — they call
through Protocol methods only.

M2.8 adds `iter_segments(scene_id, view_id, config)` as the new primary
ingestion method — yields `mva.segmentation.Segment` objects. The older
`iter_indexable_units` is kept for the frozen `mva index` legacy path
but new datasets need only implement `iter_segments`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Protocol, runtime_checkable

from mva.segmentation import Segment, SegmenterConfig

# Note: StreamSource (from mva.l0_stream) is the return type of `open_view`,
# but typed as `Any` here to avoid a circular import. Adapters import the
# concrete sources directly.


@dataclass
class Scene:
    """A meaningful grouping of views within a dataset.

    MATRIX: scene = "MATRIX_30x30", view_ids = ["D1", ..., "D8"].
    MVU-Eval: scene = "qa-0" (each question), view_ids = the question's
              video paths.
    """

    scene_id: str
    view_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAPair:
    """A multi-video question-answer item (for benchmark datasets like MVU-Eval).

    `options` is None for open-ended questions; non-None for multiple-choice
    (in which case `ground_truth` is the correct letter, e.g. "A").
    `task` is a free-form category tag used to aggregate per-task accuracy
    in the eval report.
    """

    qa_id: str
    scene_id: str
    question: str
    options: Optional[list[str]] = None
    ground_truth: Optional[str] = None
    task: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexUnit:
    """A single embeddable unit produced by an adapter for the index phase.

    The CLI's `mva index` iterates these and pushes each through the
    embedder + into ChromaDB. Different datasets yield different unit
    granularities:

    - MATRIX: one IndexUnit per detected ROI crop (kind="image",
      vector_type="reid").
    - MVU-Eval: one IndexUnit per video **segment** (default ~10s sliding
      window, sentrysearch-style), with N frames per segment mean-pooled
      into a single embedding. (kind="image_seq", vector_type="frame".)

    `data` is preloaded BGR ndarray (or list thereof for image_seq) ready
    for `MultimodalEmbedder.encode_image` / `encode_images`.

    The optional `start_sec` / `end_sec` / `segment_idx` fields are filled
    in by adapters that emit segmented units (e.g. MVU-Eval video slicer).
    They flow through to ChromaDB metadata so retrieval can map an
    embedding back to the original video time-range (`ffmpeg -ss
    start_sec -t (end_sec - start_sec)`).
    """

    unit_id: str               # used as chroma id suffix; must be unique per (scene, view)
    scene_id: str
    view_id: str
    kind: Literal["image", "image_seq"]
    data: Any                  # np.ndarray (image) or list[np.ndarray] (image_seq)
    vector_type: Literal["text", "frame", "reid"] = "frame"
    metadata: dict[str, Any] = field(default_factory=dict)
    document: Optional[str] = None     # optional human-readable text for the row
    # ---- segment-level fields (video slicing) -------------------------------
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    segment_idx: Optional[int] = None


@runtime_checkable
class DatasetAdapter(Protocol):
    """The full Protocol — implementations live in mva.datasets.{name}."""

    name: str
    root: Path

    # Capability flags drive whether the CLI registers certain tools / subcommands
    supports_cross_view_linking: bool
    supports_qa_eval: bool

    # M3.0 — granularity of cross-view linking. Determines which L2 linker
    # `mva ingest` invokes on this adapter's data:
    #   "synchronized" — same-t same-class bucketing + Hungarian on bbox
    #                    geometry (+ optional appearance secondary filter).
    #                    For multi-camera time-synchronized capture (MATRIX).
    #   "appearance"   — bucket by (class, segment_idx); Hungarian on
    #                    cosine distance of appearance embeddings.
    #                    For non-synchronized cross-video object matching
    #                    (MVU-Eval video_editing variants, etc.).
    #   "none"         — skip L2 in ingest (single-view or unrelated videos).
    # `supports_cross_view_linking` is the derived bool (mode != "none").
    cross_view_linking_mode: Literal["synchronized", "appearance", "none"]

    # ------------------------------------------------------------------
    # Scene discovery
    # ------------------------------------------------------------------

    def list_scenes(
        self,
        filter: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Scene]:
        """Yield Scene objects. `filter` is adapter-specific (e.g. {"task": "Counting"});
        `limit` caps how many to yield (None = all)."""
        ...

    def get_scene(self, scene_id: str) -> Scene:
        """Look up a Scene by id (raises KeyError if not found)."""
        ...

    # ------------------------------------------------------------------
    # View streaming (perception phase)
    # ------------------------------------------------------------------

    def open_view(
        self,
        scene_id: str,
        view_id: str,
        sample_fps: Optional[float] = None,
    ) -> Any:                  # returns StreamSource — typed Any to avoid cycle
        """Return a Frame iterator for the given (scene, view)."""
        ...

    # ------------------------------------------------------------------
    # Indexable units (legacy `mva index` path, frozen at M2.7)
    # ------------------------------------------------------------------

    def iter_indexable_units(
        self,
        scene_id: str,
        view_id: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Iterator[IndexUnit]:
        """Yield IndexUnits for embedding. `view_id=None` = all views.

        Deprecated as of M2.8 — kept for back-compat with the frozen
        `mva index` CLI. New code should use `iter_segments`."""
        ...

    # ------------------------------------------------------------------
    # Segments (M2.8 primary ingestion API)
    # ------------------------------------------------------------------

    def iter_segments(
        self,
        scene_id: str,
        view_id: str,
        config: SegmenterConfig,
    ) -> Iterator[Segment]:
        """Yield Segments for ingestion: one Segment per sliding window
        with `config.nframes_per_segment` uniformly-sampled BGR frames.

        Used by `mva ingest`. Adapters typically delegate to
        `mva.segmentation.iter_video_segments` (video files) or
        `iter_image_dir_segments` (PNG sequences)."""
        ...

    # ------------------------------------------------------------------
    # Optional capabilities — implementations raise NotImplementedError
    # ------------------------------------------------------------------

    def load_qa_pairs(
        self,
        tasks: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[QAPair]:
        """Yield QAPairs. Raises NotImplementedError if not supports_qa_eval."""
        ...

    def load_calibration(
        self,
        scene_id: str,
        view_id: str,
        frame_idx: int,
    ) -> Optional[dict[str, Any]]:
        """Return {'K': 3x3 np.ndarray, 'rvec': 3x1, 'tvec': 3x1, ...} or None.
        Used by M3+ BEV projection. Adapters without calibration return None."""
        ...
