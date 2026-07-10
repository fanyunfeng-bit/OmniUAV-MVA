"""L7 Human-in-the-loop correction interface — 🔌 §3.4 #8 stub.

Provides RPC endpoint placeholders for users to mark cross-view links as
incorrect (or correct). Writes flow back to L5 cross_view_links with
created_by="human" (the enum value already exists in mva.contracts.CrossViewLink).

M0 / v0.x: endpoint returns 501 Not Implemented.
M5: real UI + weak-supervision training data pipeline.
"""
from __future__ import annotations

from typing import Optional


class HumanCorrectionInterface:
    """RPC-shaped façade. M0 returns 501 for everything (smoke-test verifiable)."""

    def mark_link_incorrect(
        self, link_id: str, user_id: Optional[str] = None
    ) -> int:
        """Returns HTTP-style status code. 501 = Not Implemented (M0 stub)."""
        return 501

    def mark_link_correct(
        self, link_id: str, user_id: Optional[str] = None
    ) -> int:
        return 501
