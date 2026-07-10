"""Pluggable dataset adapters.

Each adapter implements `DatasetAdapter` Protocol from `mva.datasets.base`
and is registered in `mva.datasets.registry`. The rest of the codebase
(CLI / Orchestrator / eval) is dataset-agnostic — talks only to the
Protocol surface.

Adding a new dataset:
    1. New file `mva/datasets/<name>.py`, class implementing DatasetAdapter
    2. Add `(YourClass, "DATASETS/<dir>")` to ADAPTERS in registry.py
    3. Done — `mva ... --dataset <name>` works everywhere
"""
from mva.datasets.base import DatasetAdapter, IndexUnit, QAPair, Scene
from mva.datasets.matrix import MatrixDataset
from mva.datasets.mvu_eval import MVUEvalDataset
from mva.datasets.registry import ADAPTERS, get_adapter, list_known

__all__ = [
    "ADAPTERS",
    "DatasetAdapter",
    "IndexUnit",
    "MatrixDataset",
    "MVUEvalDataset",
    "QAPair",
    "Scene",
    "get_adapter",
    "list_known",
]
