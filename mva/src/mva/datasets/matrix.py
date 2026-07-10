"""MATRIX dataset adapter (KostaDakic/MATRIX).

Layout (within the root passed to MatrixDataset):

    <root>/
    ├── MATRIX_30x30/
    │   ├── image_subsets/D{1..8}/{0000..0999}.png    # 8 drones × 1000 PNG frames each, 1920×1080
    │   ├── matchings/Pedestrians/3d_{0000..0999}.txt # per-frame: <local> <global_id> x y z
    │   ├── calibrations/{intrinsic,extrinsic}/       # per-frame per-drone XML
    │   ├── annotations_positions/                    # (usually empty in v1)
    │   └── POMs/                                     # probabilistic occupancy maps
    └── (other scenes if MATRIX adds more)

Capabilities: supports_cross_view_linking=True (full GT in matchings/);
              supports_qa_eval=False (no QA pairs).

For `iter_indexable_units`, MATRIX yields one IndexUnit per detected ROI
crop — but YOLO detections live in the WorldStateStore from a prior
`mva perceive` run, not in MATRIX itself. So MatrixDataset.iter_indexable_units
needs a WorldStateStore handle. The CLI passes it via the `store=` kwarg
on the adapter method (Protocol allows extra kwargs via **kwargs).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from mva.datasets.base import IndexUnit, QAPair, Scene
from mva.l0_stream import ImageDirStreamSource
from mva.segmentation import (
    Segment,
    SegmenterConfig,
    iter_image_dir_segments,
)

# MATRIX native frame rate (2 FPS per the dataset spec)
MATRIX_SOURCE_FPS = 2.0


class MatrixDataset:
    """DatasetAdapter for the MATRIX 30×30 multi-drone benchmark."""

    name = "matrix"
    supports_cross_view_linking = True
    supports_qa_eval = False
    # 8 drones, time-synchronized capture → same physical t across views.
    cross_view_linking_mode = "synchronized"

    def __init__(self, root: Path | str = "DATASETS/matrix") -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(
                f"MATRIX root not found: {self.root}\n"
                f"Expected layout: {self.root}/MATRIX_30x30/image_subsets/D{{1..8}}/"
            )
        # Discover scenes — each immediate sub-dir of root that has image_subsets/ is one scene
        self._scenes: dict[str, Path] = {}
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and (child / "image_subsets").is_dir():
                self._scenes[child.name] = child

    # ------------------------------------------------------------------
    # Scenes
    # ------------------------------------------------------------------

    def list_scenes(
        self,
        filter: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Scene]:
        count = 0
        for scene_id in self._scenes:
            scene = self.get_scene(scene_id)
            if filter:
                if "view_id" in filter and filter["view_id"] not in scene.view_ids:
                    continue
            yield scene
            count += 1
            if limit is not None and count >= limit:
                return

    def get_scene(self, scene_id: str) -> Scene:
        if scene_id not in self._scenes:
            raise KeyError(f"Scene not found: {scene_id}. Known: {list(self._scenes)}")
        scene_root = self._scenes[scene_id]
        view_dirs = sorted(p.name for p in (scene_root / "image_subsets").iterdir() if p.is_dir())
        return Scene(
            scene_id=scene_id,
            view_ids=view_dirs,
            metadata={"root": str(scene_root), "source_fps": MATRIX_SOURCE_FPS},
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
        scene_root = self._scenes[scene_id]
        view_dir = scene_root / "image_subsets" / view_id
        if not view_dir.is_dir():
            raise FileNotFoundError(f"View directory not found: {view_dir}")
        return ImageDirStreamSource(
            view_dir,
            view_id=view_id,
            source_fps=MATRIX_SOURCE_FPS,
            sample_fps=sample_fps,
        )

    # ------------------------------------------------------------------
    # M2.8 ingestion: yield Segment objects from PNG sequence
    # ------------------------------------------------------------------

    def iter_segments(
        self,
        scene_id: str,
        view_id: str,
        config: SegmenterConfig,
    ) -> Iterator[Segment]:
        """Slice the PNG sequence into time-windowed Segments at the
        native MATRIX frame rate (2 fps → 10 s window covers 20 PNGs)."""
        scene_root = self._scenes[scene_id]
        view_dir = scene_root / "image_subsets" / view_id
        if not view_dir.is_dir():
            return
        yield from iter_image_dir_segments(
            view_dir, view_id, MATRIX_SOURCE_FPS, config,
        )

    # ------------------------------------------------------------------
    # Indexable units — requires a populated WorldStateStore (legacy)
    # ------------------------------------------------------------------

    def iter_indexable_units(
        self,
        scene_id: str,
        view_id: Optional[str] = None,
        max_frames: Optional[int] = None,
        store: Any = None,
    ) -> Iterator[IndexUnit]:
        """For MATRIX, IndexUnits are per-detection ROI crops.

        Requires `store` — a WorldStateStore previously populated by
        `mva perceive`. Walks the tracklets, re-opens the source frames,
        crops the bbox region, yields image-kind IndexUnit per detection.
        """
        if store is None:
            raise ValueError(
                "MatrixDataset.iter_indexable_units requires `store=`; "
                "MATRIX indexing reads ROI crops from tracklets written by "
                "`mva perceive` — there are no detections in the dataset itself."
            )
        scene = self.get_scene(scene_id)
        view_ids = [view_id] if view_id else scene.view_ids
        for vid in view_ids:
            view_source = self.open_view(scene_id, vid)
            # Cache frames since we'll need to look them up by frame_idx
            # (cheap for MATRIX — 1000 PNGs per view).
            frames: dict[float, np.ndarray] = {}
            for fr in view_source:
                frames[round(fr.t, 6)] = fr.image
            tracklets = store.query_tracklets(vid)
            count = 0
            for tk in tracklets:
                if max_frames is not None and count >= max_frames:
                    break
                # bboxes was JSON-stored; entries look like
                # [t, x1, y1, x2, y2, class_name, conf] or similar
                for bbox_row in tk["bboxes"]:
                    t = round(float(bbox_row[0]), 6)
                    img = frames.get(t)
                    if img is None:
                        continue
                    x1, y1, x2, y2 = (
                        max(0, int(bbox_row[1])), max(0, int(bbox_row[2])),
                        max(0, int(bbox_row[3])), max(0, int(bbox_row[4])),
                    )
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    class_name = (
                        bbox_row[5] if len(bbox_row) > 5 else "unknown"
                    )
                    conf = float(bbox_row[6]) if len(bbox_row) > 6 else 0.0
                    yield IndexUnit(
                        unit_id=tk["tracklet_id"],
                        scene_id=scene_id,
                        view_id=vid,
                        kind="image",
                        data=crop,
                        vector_type="reid",
                        metadata={
                            "t": t,
                            "class_name": class_name,
                        },
                        document=f"{class_name} (conf={conf:.2f})",
                    )
                    count += 1

    # ------------------------------------------------------------------
    # MATRIX has no QA
    # ------------------------------------------------------------------

    def load_qa_pairs(
        self, tasks: Optional[list[str]] = None, limit: Optional[int] = None,
    ) -> Iterator[QAPair]:
        raise NotImplementedError("MATRIX has no QA pairs (supports_qa_eval=False).")

    # ------------------------------------------------------------------
    # Calibration — for M3 BEV projection
    # ------------------------------------------------------------------

    def load_calibration(
        self,
        scene_id: str,
        view_id: str,
        frame_idx: int,
    ) -> Optional[dict[str, Any]]:
        """Return {K, rvec, tvec} as numpy arrays, or None if missing.

        MATRIX names files as e.g. intr_Drone1_0042.xml (frame_idx 0-padded
        to 4 digits, drone id derived from view_id "D1" → "Drone1").
        """
        scene_root = self._scenes[scene_id]
        drone_num = view_id.lstrip("D")
        try:
            drone_idx = int(drone_num)
        except ValueError:
            return None
        suffix = f"Drone{drone_idx}_{frame_idx:04d}.xml"
        intr_path = scene_root / "calibrations" / "intrinsic" / f"intr_{suffix}"
        extr_path = scene_root / "calibrations" / "extrinsic" / f"extr_{suffix}"
        if not (intr_path.is_file() and extr_path.is_file()):
            return None
        K = _read_opencv_matrix(intr_path, "camera_matrix")
        rvec = _read_opencv_matrix(extr_path, "rvec")
        tvec = _read_opencv_matrix(extr_path, "tvec")
        return {"K": K, "rvec": rvec, "tvec": tvec}

    # ------------------------------------------------------------------
    # MATRIX-specific extension: cross-view ground truth (for M3+ eval)
    # ------------------------------------------------------------------

    def load_cross_view_gt(
        self, scene_id: str, frame_idx: int,
    ) -> list[dict[str, Any]]:
        """Return per-pedestrian global IDs + 3D positions for one timestamp.

        Each entry: {local_idx, global_id, x, y, z}. The global_id is shared
        across drone views for the same person — the cross-camera GT.
        """
        scene_root = self._scenes[scene_id]
        path = scene_root / "matchings" / "Pedestrians" / f"3d_{frame_idx:04d}.txt"
        if not path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            out.append({
                "local_idx": int(parts[0]),
                "global_id": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
            })
        return out


def _read_opencv_matrix(xml_path: Path, tag: str) -> Optional[np.ndarray]:
    """Parse one OpenCV-style matrix node from an XML file."""
    try:
        tree = ET.parse(xml_path)
        node = tree.getroot().find(tag)
        if node is None:
            return None
        rows = int(node.findtext("rows", "0"))
        cols = int(node.findtext("cols", "0"))
        data = node.findtext("data", "").strip().split()
        if not data:
            return None
        vals = [float(v) for v in data]
        if rows * cols != len(vals):
            return None
        return np.array(vals, dtype=np.float64).reshape(rows, cols)
    except Exception:
        return None
