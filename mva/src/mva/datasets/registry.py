"""Adapter registry: name → class.

Adding a new dataset is two lines: import the class + add to ADAPTERS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Type

from mva.datasets.base import DatasetAdapter
from mva.datasets.matrix import MatrixDataset
from mva.datasets.mvu_eval import MVUEvalDataset
from mva.datasets.reservoir import ReservoirDataset
from mva.datasets.visdrone_mdmt import VisDroneMDMTDataset


# {short-name: (adapter-class, default-root-relative-to-cwd)}
ADAPTERS: dict[str, tuple[Type[DatasetAdapter], str]] = {
    "matrix":         (MatrixDataset, "DATASETS/matrix"),
    "mvu-eval":       (MVUEvalDataset, "DATASETS/MVU-Eval"),
    "visdrone-mdmt":  (VisDroneMDMTDataset, "DATASETS/visdrone-mdmt"),
    "pcl-sim":        (ReservoirDataset, "DATASETS/PCL-Simulation"),
}


def get_adapter(name: str, root: Optional[Path | str] = None) -> DatasetAdapter:
    """Look up an adapter by name, constructing with the given root.

    `root=None` uses the adapter's default (relative to cwd). CLI always
    passes an explicit root resolved against the repo root."""
    if name not in ADAPTERS:
        raise KeyError(
            f"Unknown dataset: {name!r}. Known: {sorted(ADAPTERS)}. "
            "Register new datasets in mva/datasets/registry.py."
        )
    cls, default_root = ADAPTERS[name]
    return cls(Path(root) if root else Path(default_root))


def list_known() -> list[str]:
    return sorted(ADAPTERS.keys())
