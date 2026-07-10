"""L2 CrossViewLinker Protocol — shared by Geometric (default) and LLM modes.

Both implementations are enforced to satisfy the CrossViewLink contract
(see `tests/contracts/test_cross_view_link.py`):

- confidence ∈ [0, 1]
- multi-candidate output sorted DESC by confidence
- empty result is [] (not None, not exception)
- view_observations has >= 2 distinct view_ids
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mva.contracts import CrossViewLink


@runtime_checkable
class CrossViewLinker(Protocol):
    """Produces cross-view links from per-view tracklet observations.

    Parameters of `link` use `Any` for the observation shape because the
    Tracklet type lives in L1 and is still evolving. M1+ will narrow this
    to a concrete TypedDict / dataclass.
    """

    def link(self, observations: list[Any]) -> list[CrossViewLink]:
        ...
