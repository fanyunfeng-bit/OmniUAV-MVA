"""MVU-Eval dataset adapter (NJU-LINK/MVU-Eval).

Layout (within the root):

    <root>/
    ├── MVU_Eval_QAs.json              # 1824 QA pairs (multiple-choice)
    ├── README.md  case.pdf  ...
    ├── <video>.mp4                    # ~4636 video files at root
    ├── temp_ordering-videos/          # task-specific extras
    ├── temp_grounding-videos/
    ├── temp_cap_filling-videos/
    └── video_editing-video/{add,replace,remove}/

QA JSON shape (per item):
    {
      "video_paths": ["scene0377_01.mp4", ...],     # N videos to compare
      "question": "...",
      "options": ["A. ...", "B. ...", ...],
      "ground_truth": "A",
      "task": "Counting"
    }

Capabilities: supports_cross_view_linking=False (no synchronized views);
              supports_qa_eval=True (1824 QA with letter GT).

Each QA becomes one Scene whose view_ids are the QA's video filenames.

Indexing convention (sentrysearch-style):
    Each video is sliced into ~10 s sliding-window segments (configurable
    via `window_sec` / `stride_sec`). Each segment yields ONE IndexUnit
    with N=`nframes_per_segment` uniformly-sampled frames that the
    embedder mean-pools into a single vector. The segment's
    `start_sec` / `end_sec` / `segment_idx` flow through to ChromaDB
    metadata so retrieval can locate the original clip.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from mva.datasets.base import IndexUnit, QAPair, Scene
from mva.l0_stream import FileStreamSource
from mva.segmentation import Segment, SegmenterConfig, iter_video_segments

# MVU-Eval reference inference script defaults (DATASETS/MVU-Eval/main_all_MVU_Eval_llama3.py)
MVU_DEFAULT_NFRAMES = 32
MVU_DEFAULT_MAX_PIXELS = 720
QA_FILENAME = "MVU_Eval_QAs.json"

# Sentrysearch-style sliding window defaults. Non-overlapping by default
# (stride == window); the CLI exposes both knobs.
DEFAULT_WINDOW_SEC = 10.0
DEFAULT_STRIDE_SEC = 10.0
DEFAULT_NFRAMES_PER_SEGMENT = 4
# Drop trailing segments shorter than this — avoids embedding a 0.3 s tail.
MIN_SEGMENT_SEC = 1.0


class MVUEvalDataset:
    """DatasetAdapter for MVU-Eval multi-video understanding benchmark."""

    name = "mvu-eval"
    # M3.0: cross-video appearance matching IS meaningful for several MVU-Eval
    # task types (video_editing variants share scenes; Ordering / Counting
    # often share objects). Time is NOT synchronized across the QA's video
    # paths, so we use the appearance-only L2 linker.
    supports_cross_view_linking = True
    supports_qa_eval = True
    cross_view_linking_mode = "appearance"

    def __init__(self, root: Path | str = "DATASETS/MVU-Eval") -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"MVU-Eval root not found: {self.root}")
        self._qa_path = self.root / QA_FILENAME
        if not self._qa_path.is_file():
            raise FileNotFoundError(
                f"QA file not found: {self._qa_path}\n"
                "Download MVU_Eval_QAs.json from HuggingFace "
                "MVU-Eval-Team/MVU-Eval-Data."
            )
        # Lazy-loaded dict: {qa_id: QA dict}
        self._qa_cache: Optional[dict[str, dict[str, Any]]] = None

    def _load_qa(self) -> dict[str, dict[str, Any]]:
        if self._qa_cache is None:
            with self._qa_path.open("r", encoding="utf-8") as f:
                self._qa_cache = json.load(f)
        return self._qa_cache

    def _resolve_video(self, filename: str) -> Path:
        """MVU-Eval ships videos in several sub-dirs; the QA JSON gives
        filenames relative to the root. Search root first, then each
        known sub-dir."""
        candidates = [
            self.root / filename,
            self.root / "temp_ordering-videos" / filename,
            self.root / "temp_grounding-videos" / filename,
            self.root / "temp_cap_filling-videos" / filename,
            self.root / "video_editing-video" / "add" / filename,
            self.root / "video_editing-video" / "replace" / filename,
            self.root / "video_editing-video" / "remove" / filename,
        ]
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(
            f"Video not found in MVU-Eval root or known sub-dirs: {filename}"
        )

    # ------------------------------------------------------------------
    # Scenes — each QA pair is its own Scene
    # ------------------------------------------------------------------

    @staticmethod
    def _make_view_ids(video_paths: list[str]) -> tuple[list[str], dict[str, str]]:
        """Map raw video filenames to short view IDs (V1, V2, ...).
        Returns (view_ids, {short_id: original_filename})."""
        view_ids = [f"V{i+1}" for i in range(len(video_paths))]
        mapping = dict(zip(view_ids, video_paths))
        return view_ids, mapping

    def list_scenes(
        self,
        filter: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Scene]:
        qa = self._load_qa()
        tasks_filter = None
        if filter and "task" in filter:
            t = filter["task"]
            tasks_filter = {t} if isinstance(t, str) else set(t)
        count = 0
        for qa_id, qa_item in qa.items():
            if tasks_filter and qa_item.get("task") not in tasks_filter:
                continue
            video_paths = list(qa_item.get("video_paths", []))
            view_ids, mapping = self._make_view_ids(video_paths)
            yield Scene(
                scene_id=f"qa-{qa_id}",
                view_ids=view_ids,
                metadata={
                    "task": qa_item.get("task"),
                    "question": qa_item.get("question"),
                    "view_mapping": mapping,
                },
            )
            count += 1
            if limit is not None and count >= limit:
                return

    def get_scene(self, scene_id: str) -> Scene:
        qa = self._load_qa()
        if not scene_id.startswith("qa-"):
            raise KeyError(f"MVU-Eval scene_id must start with 'qa-', got {scene_id!r}")
        qa_id = scene_id[len("qa-"):]
        if qa_id not in qa:
            raise KeyError(f"Scene not found: {scene_id}")
        qa_item = qa[qa_id]
        video_paths = list(qa_item.get("video_paths", []))
        view_ids, mapping = self._make_view_ids(video_paths)
        return Scene(
            scene_id=scene_id,
            view_ids=view_ids,
            metadata={
                "task": qa_item.get("task"),
                "question": qa_item.get("question"),
                "view_mapping": mapping,
            },
        )

    # ------------------------------------------------------------------
    # View streaming — each "view" is one mp4 file
    # ------------------------------------------------------------------

    def _resolve_view_file(self, scene_id: str, view_id: str) -> Path:
        """Resolve a short view ID (V1, V2, ...) to the video file path."""
        scene = self.get_scene(scene_id)
        mapping = scene.metadata.get("view_mapping", {})
        filename = mapping.get(view_id, view_id)
        return self._resolve_video(filename)

    def open_view(
        self,
        scene_id: str,
        view_id: str,
        sample_fps: Optional[float] = None,
    ) -> FileStreamSource:
        video_path = self._resolve_view_file(scene_id, view_id)
        return FileStreamSource(
            video_path, view_id=view_id, sample_fps=sample_fps,
        )

    # ------------------------------------------------------------------
    # M2.8 ingestion: yield Segment objects for the unified ingest pipeline
    # ------------------------------------------------------------------

    def iter_segments(
        self,
        scene_id: str,
        view_id: str,
        config: SegmenterConfig,
    ) -> Iterator[Segment]:
        """Delegate to `iter_video_segments`. Skips missing video files
        silently (matches the M2.7 `iter_indexable_units` behavior)."""
        try:
            video_path = self._resolve_view_file(scene_id, view_id)
        except FileNotFoundError:
            return
        yield from iter_video_segments(video_path, view_id, config)

    # ------------------------------------------------------------------
    # Indexable units — one IndexUnit per ~10 s video segment (legacy)
    # ------------------------------------------------------------------

    def iter_indexable_units(
        self,
        scene_id: str,
        view_id: Optional[str] = None,
        max_frames: Optional[int] = None,
        store: Any = None,
        window_sec: float = DEFAULT_WINDOW_SEC,
        stride_sec: float = DEFAULT_STRIDE_SEC,
        nframes_per_segment: int = DEFAULT_NFRAMES_PER_SEGMENT,
    ) -> Iterator[IndexUnit]:
        """Yield one IndexUnit per video segment (sliding window).

        Sentrysearch-style: each video is sliced into `window_sec`-long
        windows advancing by `stride_sec`. Within each window we
        uniformly sample `nframes_per_segment` BGR frames; the embedder
        mean-pools them into one vector. Each yielded IndexUnit carries
        `start_sec` / `end_sec` / `segment_idx` so retrieval can map the
        embedding back to the original clip via
        `ffmpeg -ss start_sec -t (end_sec - start_sec)`.

        `max_frames` caps the **number of yielded units across all views**
        (it's a CLI safety knob, not a per-video frame count).
        """
        del store  # unused for MVU-Eval
        if window_sec <= 0 or stride_sec <= 0:
            raise ValueError(
                f"window_sec / stride_sec must be > 0, got "
                f"window={window_sec}, stride={stride_sec}"
            )
        if nframes_per_segment <= 0:
            raise ValueError(
                f"nframes_per_segment must be > 0, got {nframes_per_segment}"
            )

        scene = self.get_scene(scene_id)
        view_ids = [view_id] if view_id else scene.view_ids
        count = 0
        for vid in view_ids:
            if max_frames is not None and count >= max_frames:
                return
            try:
                video_path = self._resolve_view_file(scene_id, vid)
            except FileNotFoundError:
                continue
            duration = _video_duration_sec(video_path)
            if duration <= 0:
                continue

            seg_idx = 0
            cur = 0.0
            while cur < duration:
                if max_frames is not None and count >= max_frames:
                    return
                seg_end = min(cur + window_sec, duration)
                if seg_end - cur < MIN_SEGMENT_SEC and seg_idx > 0:
                    # Drop tiny trailing segment, but keep the only segment
                    # for very short videos (< MIN_SEGMENT_SEC total).
                    break
                frames = _sample_segment_frames(
                    video_path, cur, seg_end, nframes_per_segment,
                )
                if not frames:
                    seg_idx += 1
                    cur += stride_sec
                    continue
                stem = Path(vid).stem
                yield IndexUnit(
                    unit_id=f"{stem}::seg{seg_idx:04d}",
                    scene_id=scene_id,
                    view_id=vid,
                    kind="image_seq",
                    data=frames,
                    vector_type="frame",
                    metadata={
                        "video_path": str(video_path),
                        "nframes_sampled": len(frames),
                    },
                    document=f"{stem} [{cur:.1f}-{seg_end:.1f}s]",
                    start_sec=float(cur),
                    end_sec=float(seg_end),
                    segment_idx=seg_idx,
                )
                count += 1
                seg_idx += 1
                cur += stride_sec

    # ------------------------------------------------------------------
    # QA pairs
    # ------------------------------------------------------------------

    def load_qa_pairs(
        self,
        tasks: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[QAPair]:
        qa = self._load_qa()
        task_set = set(tasks) if tasks else None
        count = 0
        for qa_id, qa_item in qa.items():
            if task_set and qa_item.get("task") not in task_set:
                continue
            yield QAPair(
                qa_id=qa_id,
                scene_id=f"qa-{qa_id}",
                question=qa_item.get("question", ""),
                options=qa_item.get("options"),
                ground_truth=qa_item.get("ground_truth"),
                task=qa_item.get("task"),
                metadata={
                    "video_paths": qa_item.get("video_paths", []),
                },
            )
            count += 1
            if limit is not None and count >= limit:
                return

    # ------------------------------------------------------------------
    # No calibration
    # ------------------------------------------------------------------

    def load_calibration(
        self,
        scene_id: str,
        view_id: str,
        frame_idx: int,
    ) -> Optional[dict[str, Any]]:
        return None


def _video_duration_sec(video_path: Path) -> float:
    """Return video duration in seconds via OpenCV. 0.0 on read failure."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    if fps <= 0 or total <= 0:
        return 0.0
    return float(total) / float(fps)


def _sample_segment_frames(
    video_path: Path, start_sec: float, end_sec: float, n: int,
) -> list:
    """Sample up to `n` uniformly-spaced BGR frames from [start_sec, end_sec).

    Returns [] on read failure or empty window. Seeks via
    `CAP_PROP_POS_MSEC` rather than frame index so we don't need to know
    fps precisely (some MVU-Eval videos have VFR / metadata oddities).
    """
    import cv2

    if end_sec <= start_sec or n <= 0:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    span = end_sec - start_sec
    # n evenly-spaced samples inside [start, end); first sample at start,
    # last at start + span * (n-1)/n so we never overshoot end_sec.
    timestamps = [
        start_sec + (i + 0.5) * (span / n) for i in range(n)
    ]
    frames = []
    for t_sec in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ok, img = cap.read()
        if ok and img is not None:
            frames.append(img)
    cap.release()
    return frames


# Back-compat shim: a few callers (and any out-of-tree users) imported the
# old whole-video sampler. Kept so old code doesn't break — new code should
# use `_sample_segment_frames` against a specific window.
def _sample_video_frames(video_path: Path, n: int) -> list:
    duration = _video_duration_sec(video_path)
    if duration <= 0:
        return []
    return _sample_segment_frames(video_path, 0.0, duration, n)
