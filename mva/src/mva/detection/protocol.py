"""M1 检测/分割 Protocol：换检测器/分割器不动下游。

现有 `mva.l1_perception.Detector`（YOLO/YOLOE/YOLO-World）结构上满足 `ObjectDetector`。
返回逐目标 `Detection`（bbox/class_name/confidence），见 `mva.l1_perception.Detection`。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ObjectDetector(Protocol):
    def detect(self, image: Any) -> list[Any]: ...      # image: np.ndarray → list[Detection]


@runtime_checkable
class Segmenter(Protocol):
    """（可选）逐帧实例/语义分割。"""
    def segment(self, image: Any) -> list[Any]: ...
