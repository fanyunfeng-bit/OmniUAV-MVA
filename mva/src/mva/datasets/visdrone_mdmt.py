"""VisDrone-MDMT dataset adapter (Multi-Drone Multi-Object Tracking).

Layout (within the root):

    <root>/Multi-Drone-Multi-Object-Detection-and-Tracking/
    ├── test/
    │   ├── 1/          ← Drone 1
    │   │   ├── 26-1/   ← scene 26, 300 JPGs (1920×1080)
    │   │   ├── 31-1/
    │   │   └── ...
    │   └── 2/          ← Drone 2
    │       ├── 26-2/
    │       └── ...
    ├── new_xml/        ← GT annotations (CVAT XML, track id + bbox)
    │   ├── 1/26-1.xml
    │   └── 2/26-2.xml
    └── train/, val/    ← not used (test has full dual-drone coverage)

Capabilities: supports_cross_view_linking=True (dual-drone synchronized);
              supports_qa_eval=False (no QA pairs).

Scene durations are short (10-23s @30fps), so the default segmenter
config uses window_sec=3.0 / stride_sec=3.0 instead of 10s.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, Optional

from mva.datasets.base import IndexUnit, QAPair, Scene
from mva.l0_stream import ImageDirStreamSource
from mva.segmentation import (
    Segment,
    SegmenterConfig,
    iter_image_dir_segments,
)

VISDRONE_SOURCE_FPS = 30.0
DEFAULT_WINDOW_SEC = 3.0
DEFAULT_STRIDE_SEC = 3.0


class VisDroneMDMTDataset:
    """DatasetAdapter for VisDrone-MDMT (test split, dual-drone)."""

    name = "visdrone-mdmt"
    supports_cross_view_linking = True
    supports_qa_eval = False
    # "appearance" not "synchronized": both drones are time-synced, but they
    # observe from different positions/altitudes, so the same target appears
    # at completely different pixel coords. Geometric (bbox center distance)
    # matching fails; appearance (embedding cosine) is the correct fallback.
    cross_view_linking_mode = "appearance"

    def __init__(self, root: Path | str = "DATASETS/visdrone-mdmt") -> None:
        self.root = Path(root)
        self._base = self.root / "Multi-Drone-Multi-Object-Detection-and-Tracking"
        self._split_dir = self._base / "test"
        if not self._split_dir.is_dir():
            raise NotADirectoryError(
                f"VisDrone-MDMT test split not found: {self._split_dir}\n"
                f"Expected layout: {self._split_dir}/1/XX-1/ + 2/XX-2/"
            )
        self._scenes = self._discover_scenes()

    def _discover_scenes(self) -> dict[str, dict[str, Path]]:
        """Build {scene_id: {"D1": path, "D2": path}} from directory layout."""
        scenes: dict[str, dict[str, Path]] = {}
        for drone_dir in sorted(self._split_dir.iterdir()):
            if not drone_dir.is_dir() or drone_dir.name not in ("1", "2"):
                continue
            drone_label = f"D{drone_dir.name}"
            for view_dir in sorted(drone_dir.iterdir()):
                if not view_dir.is_dir():
                    continue
                m = re.match(r"(\d+)-[12]", view_dir.name)
                if not m:
                    continue
                scene_id = f"scene-{m.group(1)}"
                scenes.setdefault(scene_id, {})[drone_label] = view_dir
        return scenes

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
                f"Scene not found: {scene_id}. "
                f"Known: {sorted(self._scenes)}"
            )
        views = self._scenes[scene_id]
        return Scene(
            scene_id=scene_id,
            view_ids=sorted(views.keys()),
            metadata={
                "source_fps": VISDRONE_SOURCE_FPS,
                "default_window_sec": DEFAULT_WINDOW_SEC,
                "default_stride_sec": DEFAULT_STRIDE_SEC,
            },
        )

    # ------------------------------------------------------------------
    # View streaming
    # ------------------------------------------------------------------

    def open_view(
        self,
        scene_id: str,
        view_id: str,
        sample_fps: Optional[float] = None,
    ) -> ImageDirStreamSource:
        view_dir = self._scenes[scene_id][view_id]
        return ImageDirStreamSource(
            view_dir,
            view_id=view_id,
            source_fps=VISDRONE_SOURCE_FPS,
            sample_fps=sample_fps,
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
        view_dir = views.get(view_id)
        if view_dir is None or not view_dir.is_dir():
            return
        yield from iter_image_dir_segments(
            view_dir, view_id, VISDRONE_SOURCE_FPS, config,
        )

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
            "VisDrone-MDMT uses iter_segments (M2.8 ingest path)."
        )

    def load_qa_pairs(
        self, tasks: Optional[list[str]] = None, limit: Optional[int] = None,
    ) -> Iterator[QAPair]:
        raise NotImplementedError(
            "VisDrone-MDMT has no QA pairs (supports_qa_eval=False)."
        )

    def load_calibration(
        self, scene_id: str, view_id: str, frame_idx: int,
    ) -> Optional[dict[str, Any]]:
        return None
