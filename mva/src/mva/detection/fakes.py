"""M1 桩：Phase 0 并行用。真实检测器由 M1 owner 提供
（现有 `mva.l1_perception.Detector` 已结构满足 `ObjectDetector`）。"""
from __future__ import annotations

from typing import Any


class NullDetector:
    """桩：什么都不检测。"""
    def detect(self, image: Any) -> list[Any]:
        return []


class NullSegmenter:
    """桩：不产任何掩码。"""
    def segment(self, image: Any) -> list[Any]:
        return []
