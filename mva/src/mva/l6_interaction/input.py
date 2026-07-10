"""L6 input sources: TextInput (M0 working) + VoiceInput (🔌 §3.4 #7 stub).

Voice input is deferred to v2+ but its interface lives here so the rest of
L6 can be written against `InputSource` without baking in any input modality
assumption.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from mva.contracts import NLQuery


@runtime_checkable
class InputSource(Protocol):
    def get_query(self) -> NLQuery:
        ...


class TextInput:
    """M0 working implementation: reads from a Python str (or stdin)."""

    def __init__(self, text: str = "") -> None:
        self.text = text

    def get_query(self) -> NLQuery:
        if not self.text:
            self.text = input("> ")
        q = NLQuery(text=self.text, source="text")
        # one-shot per construction; reset for next call
        self.text = ""
        return q


class VoiceInput:
    """🔌 §3.4 #7 — voice input stub.

    M0: raise NotImplementedError when get_query is called.
    v2+: integrate ASR pipeline (e.g. whisper.cpp / Paraformer).
    """

    def get_query(self) -> NLQuery:
        raise NotImplementedError(
            "Voice input ships in v2+. Use TextInput for now."
        )
