"""Data contracts between layers.

Pydantic models in this package are enforced by `tests/contracts/` and must
be satisfied by both algorithmic and LLM-mode implementations of L2/L3.
Lightweight dataclasses (Frame, NLQuery, RichQuery, Attachment, Event,
Briefing, ViewObservation) carry transport shape without validation overhead.
"""
from mva.contracts.cross_view import CrossViewLink, make_link_id
from mva.contracts.events import Anomaly, TrajectoryPrediction
from mva.contracts.geometry import CameraPose, Ray, WorldPoint
from mva.contracts.global_state import GlobalObject, GlobalObservation, GlobalTrajectory
from mva.contracts.stream import (
    Attachment,
    Briefing,
    Event,
    Frame,
    NLQuery,
    RichQuery,
    ViewObservation,
)

__all__ = [
    "Anomaly",
    "Attachment",
    "Briefing",
    "CrossViewLink",
    "Event",
    "make_link_id",
    "Frame",
    "NLQuery",
    "RichQuery",
    "TrajectoryPrediction",
    "ViewObservation",
    "WorldPoint",
    "Ray",
    "CameraPose",
    "GlobalObject",
    "GlobalObservation",
    "GlobalTrajectory",
]
