"""L6 BriefingAgent — 🔌 §3.4 #6 stub for Mode B proactive summaries.

M0 / v0.x: NullBriefingAgent returns None / empty briefings; event bus hook
exists so L3 can call it without conditional logic.
v2+: implement saliency double-gate (novelty × anomaly) per §3.2 L6.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from mva.contracts import Briefing, Event


@runtime_checkable
class BriefingAgent(Protocol):
    def on_event(self, event: Event) -> Optional[Briefing]:
        ...

    def periodic_summary(
        self, t_start: float, t_end: float
    ) -> Briefing:
        ...


class NullBriefingAgent:
    """No-op BriefingAgent used in v0.x.

    Keeps the L3 → L6 event-bus wire alive so M0 can verify the hook fires
    without producing actual briefings.
    """

    def on_event(self, event: Event) -> Optional[Briefing]:
        return None

    def periodic_summary(
        self, t_start: float, t_end: float
    ) -> Briefing:
        return Briefing(
            t_start=t_start,
            t_end=t_end,
            tldr="(Mode B briefing not implemented in v0.x)",
            events=[],
            cross_view_insights=[],
        )
