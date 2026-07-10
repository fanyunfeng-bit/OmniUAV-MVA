"""PCL-Simulation dataset adapter (multi-drone simulated scenes).

Layout (within the root):

    <root>/                       ← default DATASETS/PCL-Simulation
    ├── Reservoir/
    │   ├── view1.mp4             ← drone 1
    │   ├── view2.mp4             ← drone 2
    │   └── (view3.mp4 ...)       ← drop in more files → more views
    └── <OtherScene>/             ← more sub-dirs → more scenes
        └── *.mp4

One **scene** per sub-directory; one **view** per `*.mp4` inside it. This
keeps the interface open: a third drone is just another `viewN.mp4`, a new
scene is just another sub-directory — no code change.

Capabilities: supports_cross_view_linking=True; supports_qa_eval=False.

Cross-view mode is **"appearance"** (not "synchronized"): the drones are
simulated independently, are not frame-synchronized, have different
durations / effective fps, and carry no calibration, so geometric
(bbox-center) matching is meaningless — appearance-embedding cosine is the
only sound option. Same rationale as VisDrone-MDMT.

**Temporal alignment:** views in one scene can have different lengths
(Reservoir: view1≈166.8s, view2≈151.2s). `iter_segments` caps every view
to the scene's **shortest** view duration so the two views cover the same
time span (segment_idx i ↔ window [i·stride, i·stride+window] for all
views). The trailing surplus of the longer video is dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional

from mva.datasets.base import IndexUnit, QAPair, Scene
from mva.l0_stream import FileStreamSource
from mva.segmentation import (
    Segment,
    SegmenterConfig,
    iter_video_segments,
)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
DEFAULT_WINDOW_SEC = 10.0
DEFAULT_STRIDE_SEC = 10.0


def _probe_duration(path: Path) -> float:
    """Effective duration in seconds via cv2 (FRAME_COUNT / FPS).

    cv2's CAP_PROP_FPS reports the *effective* fps for these sim clips
    (≈2.6, not the container's nominal 25), so FRAME_COUNT / FPS is the
    true wall-clock span. Returns 0.0 if the file cannot be opened.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return total / fps if fps > 0 and total > 0 else 0.0
    finally:
        cap.release()


class ReservoirDataset:
    """DatasetAdapter for PCL simulated multi-drone scenes (file-backed)."""

    name = "pcl-sim"
    supports_cross_view_linking = True
    supports_qa_eval = False
    cross_view_linking_mode = "appearance"

    def __init__(self, root: Path | str = "DATASETS/PCL-Simulation") -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(
                f"PCL-Simulation root not found: {self.root}\n"
                f"Expected layout: {self.root}/<Scene>/view1.mp4 + view2.mp4"
            )
        self._scenes = self._discover_scenes()
        if not self._scenes:
            raise FileNotFoundError(
                f"No scenes with video files found under {self.root}. "
                f"Expected {self.root}/<Scene>/*.mp4"
            )
        # Lazily-filled {scene_id: aligned (shortest) duration in seconds}.
        self._min_dur_cache: dict[str, float] = {}

    def _discover_scenes(self) -> dict[str, dict[str, Path]]:
        """Build {scene_id: {view_id: video_path}} from the directory tree."""
        scenes: dict[str, dict[str, Path]] = {}
        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir():
                continue
            views = {
                p.stem: p
                for p in sorted(scene_dir.iterdir())
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS
            }
            if views:
                scenes[scene_dir.name] = views
        return scenes

    def _aligned_duration(self, scene_id: str) -> float:
        """Shortest view duration in the scene (the alignment bound)."""
        if scene_id not in self._min_dur_cache:
            views = self._scenes.get(scene_id, {})
            durs = [d for d in (_probe_duration(p) for p in views.values()) if d > 0]
            self._min_dur_cache[scene_id] = min(durs) if durs else 0.0
        return self._min_dur_cache[scene_id]

    # ------------------------------------------------------------------
    # Scenes
    # ------------------------------------------------------------------

    def list_scenes(
        self,
        filter: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Scene]:
        count = 0
        for scene_id in sorted(self._scenes):
            yield self.get_scene(scene_id)
            count += 1
            if limit is not None and count >= limit:
                return

    def get_scene(self, scene_id: str) -> Scene:
        if scene_id not in self._scenes:
            raise KeyError(
                f"Scene not found: {scene_id}. Known: {sorted(self._scenes)}"
            )
        views = self._scenes[scene_id]
        return Scene(
            scene_id=scene_id,
            view_ids=sorted(views.keys()),
            metadata={
                "default_window_sec": DEFAULT_WINDOW_SEC,
                "default_stride_sec": DEFAULT_STRIDE_SEC,
                "aligned_duration_sec": self._aligned_duration(scene_id),
                "view_paths": {v: str(p) for v, p in sorted(views.items())},
            },
        )

    # ------------------------------------------------------------------
    # View streaming (legacy perceive path; ingest uses iter_segments)
    # ------------------------------------------------------------------

    def open_view(
        self,
        scene_id: str,
        view_id: str,
        sample_fps: Optional[float] = None,
    ) -> FileStreamSource:
        video_path = self._scenes[scene_id][view_id]
        return FileStreamSource(
            video_path, view_id=view_id, sample_fps=sample_fps,
        )

    # ------------------------------------------------------------------
    # M2.8 ingestion
    # ------------------------------------------------------------------

    def iter_segments(
        self,
        scene_id: str,
        view_id: str,
        config: SegmenterConfig,
    ) -> Iterator[Segment]:
        views = self._scenes.get(scene_id, {})
        video_path = views.get(view_id)
        if video_path is None or not video_path.is_file():
            return
        # Cap to the scene's shortest view so all views stay time-aligned.
        min_dur = self._aligned_duration(scene_id)
        for seg in iter_video_segments(video_path, view_id, config):
            if min_dur > 0 and seg.start_t >= min_dur:
                break
            yield seg

    # ------------------------------------------------------------------
    # Not supported
    # ------------------------------------------------------------------

    def iter_indexable_units(
        self,
        scene_id: str,
        view_id: Optional[str] = None,
        max_frames: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[IndexUnit]:
        raise NotImplementedError(
            "pcl-sim uses iter_segments (M2.8 ingest path)."
        )

    def load_qa_pairs(
        self, tasks: Optional[list[str]] = None, limit: Optional[int] = None,
    ) -> Iterator[QAPair]:
        raise NotImplementedError(
            "pcl-sim has no QA pairs (supports_qa_eval=False)."
        )

    def load_calibration(
        self, scene_id: str, view_id: str, frame_idx: int,
    ) -> Optional[dict[str, Any]]:
        return None
