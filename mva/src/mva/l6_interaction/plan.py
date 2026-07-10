"""L6 Mode A query-plan schemas.

`QueryPlan` is what the QueryPlanner extracts from the NL question via the
LLM. The orchestrator iterates `tool_calls` and dispatches them through the
ToolRegistry. Keeping this schema small and Pydantic-validated means
malformed LLM output fails loud at parse time, not silently mid-execution.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation requested by the planner."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class QueryPlan(BaseModel):
    """The full plan derived from one NL question."""

    intent: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    rationale: str = ""
