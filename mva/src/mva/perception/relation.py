from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class RelationModeler(ABC):
    """时空关系建模接口(留口)。model: tracks(密集轨迹) → 关系三元组列表。"""
    @abstractmethod
    def model(self, tracks: list) -> List[dict]:
        ...


class NullRelationModeler(RelationModeler):
    """基线占位:返回空。后续换规则/学习式场景图(spec §8, §3.1)。"""
    def model(self, tracks: list) -> List[dict]:
        return []
