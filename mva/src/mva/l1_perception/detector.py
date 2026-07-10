"""L1 detector — wraps ultralytics YOLOv11 / YOLOE / YOLO-World.

ultralytics is an optional dependency; if not installed, calling `Detector`
raises a helpful ImportError. The class structure stays importable so smoke
tests can still verify the module surface.

Three model families, auto-detected from `model_name`:

- closed-set **YOLOv11** (`yolo11n.pt`, ...): fixed 80 COCO classes. Fast.
- open-vocab **YOLOE** (`yoloe-11l-seg.pt`, ...): text-prompted classes via
  `set_classes(names, get_text_pe(names))`, + instance masks. Needs a higher
  `imgsz` (1280/1920) to see small aerial objects.
- open-vocab **YOLO-World** (`yolov8x-worldv2.pt`, ...): text-prompted via
  `set_classes(names)`.

Pass `classes=["car", "person", "三轮车"]` to drive the open-vocab models with
natural-language target names. For closed-set YOLO, `classes` filters the output
to those COCO names (no open-vocab — unknown names just match nothing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """A single detection: bbox + class + confidence."""

    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) pixels
    class_id: int
    class_name: str
    confidence: float


def _model_family(model_name: str) -> str:
    """'yoloe' | 'world' | 'yolo' from the weights filename."""
    n = model_name.lower()
    if "yoloe" in n:
        return "yoloe"
    if "world" in n:
        return "world"
    return "yolo"


class Detector:
    """YOLO-family detector wrapping ultralytics (closed-set or open-vocab).

    Parameters
    ----------
    model_name : str
        ultralytics weights id. `yolo11n.pt` (closed 80-class, default),
        `yoloe-11l-seg.pt` (open-vocab + seg), `yolov8x-worldv2.pt` (open-vocab).
    conf : float
        Min confidence threshold.
    device : str | None
        Torch device, e.g. "cuda", "cuda:0", "cpu". None lets ultralytics decide.
    classes : list[str] | None
        Natural-language target class names. For YOLOE/YOLO-World these become the
        open-vocab detection vocabulary (set_classes). For closed-set YOLO they
        post-filter the output by class_name. None → model's native classes.
    imgsz : int | None
        Inference image size. None → ultralytics default (640). Open-vocab models
        on small aerial objects want 1280/1920.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf: float = 0.25,
        device: Optional[str] = None,
        classes: Optional[list[str]] = None,
        imgsz: Optional[int] = None,
    ) -> None:
        self.conf = conf
        self.device = device
        self.imgsz = imgsz
        self.classes = list(classes) if classes else None
        self.family = _model_family(model_name)

        try:
            if self.family == "yoloe":
                from ultralytics import YOLOE  # type: ignore
                self.model = YOLOE(model_name)
                if self.classes:
                    self.model.set_classes(
                        self.classes, self.model.get_text_pe(self.classes)
                    )
                self.open_vocab = True
            elif self.family == "world":
                from ultralytics import YOLOWorld  # type: ignore
                self.model = YOLOWorld(model_name)
                if self.classes:
                    self.model.set_classes(self.classes)
                self.open_vocab = True
            else:
                from ultralytics import YOLO  # type: ignore
                self.model = YOLO(model_name)
                self.open_vocab = False
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for L1 Detector. "
                "Install with: pip install 'mva[detection]'"
            ) from exc

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR image and return a list of Detection."""
        kwargs: dict = dict(conf=self.conf, device=self.device, verbose=False)
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        results = self.model(image, **kwargs)
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names                                     # {int: str}
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                out.append(
                    Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        class_id=cls_id,
                        class_name=names.get(cls_id, str(cls_id)),
                        confidence=float(box.conf[0]),
                    )
                )
        # Closed-set YOLO: post-filter to the requested names (open-vocab models
        # already restricted their vocabulary via set_classes).
        if self.classes and not self.open_vocab:
            wanted = set(self.classes)
            out = [d for d in out if d.class_name in wanted]
        return out
