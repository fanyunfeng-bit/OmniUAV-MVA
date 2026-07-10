"""CrossViewLink — the L2 output contract.

Both Geometric mode (default, M1) and LLM mode (M4) must produce instances
of this class. `tests/contracts/test_cross_view_link.py` enforces these
invariants on every implementation:

- confidence ∈ [0, 1]
- multi-candidate output sorted by confidence DESC
- empty result is [] (not None, not exception)
- view_observations has at least 2 distinct view_ids
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Literal

from pydantic import BaseModel, Field, field_validator


def make_link_id(view_observations: Iterable[tuple[str, str]]) -> str:
    """Deterministic link_id from the sorted set of `(view_id, tracklet_id)`
    observations the link spans.

    Same logical link (same observation set) → same id, regardless of run.
    Paired with `WorldStateStore.insert_cross_view_link`'s INSERT OR
    REPLACE this makes `mva ingest` reruns fully idempotent on the
    `cross_view_links` table (M3.4 follow-up — the original M3.4 fix only
    covered ChromaDB + segments/tracklets, not L2 link rows).

    Format: first 16 hex chars of SHA-1. ~10^19 search space across same
    scene is overkill for collision avoidance; the short form keeps DB
    indexes compact."""
    key = "|".join(f"{v}::{t}" for v, t in sorted(view_observations))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class CrossViewLink(BaseModel):
    """A link asserting that observations from 2+ views correspond to the same target."""

    link_id: str
    view_observations: list[tuple[str, str]]            # [(view_id, tracklet_id), ...]
    confidence: float = Field(ge=0.0, le=1.0)
    created_by: Literal[
        "geometric",            # M1 default: bbox-center Hungarian only
        "geometric+appearance", # M2.8: + appearance-embedding cosine filter
        "appearance",           # M3.0: pure cosine matching across non-synchronized views
        "llm",                  # M4 LLM cross-view mode
        "human",                # §3.4 #8 HITL correction (M5)
    ]
    created_at: float                                    # unix timestamp seconds

    @field_validator("view_observations")
    @classmethod
    def at_least_two_distinct_views(
        cls, v: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        if len(v) < 2:
            raise ValueError("CrossViewLink requires at least 2 view observations")
        view_ids = {obs[0] for obs in v}
        if len(view_ids) < 2:
            raise ValueError(
                "CrossViewLink requires at least 2 DISTINCT view_ids "
                "(observations from the same view do not form a cross-view link)"
            )
        return v
