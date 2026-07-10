"""L0 StreamSource Protocol — abstracts over RTSP, file, in-memory."""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from mva.contracts import Frame


@runtime_checkable
class StreamSource(Protocol):
    """Yields Frame instances for one logical view (e.g. one drone)."""

    def __iter__(self) -> Iterator[Frame]:
        ...
