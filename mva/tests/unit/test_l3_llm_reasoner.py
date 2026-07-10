"""Unit tests for `mva.l3_events.llm_mode.LLMReasoner` (M4.2).

Covers the prompt-building / JSON-parsing / retry / strict-vocab paths
end-to-end with a scripted fake LLM client. Runs in <1s, no GPU.

Contract-level invariants (Protocol satisfaction, return types) are
in `tests/contracts/test_events.py` — kept separate so they parametrize
cleanly over both Reasoner modes.
"""
from __future__ import annotations

import json

import pytest

from mva.contracts import Anomaly
from mva.l3_events.llm_mode import (
    LLMReasoner,
    _extract_behavior_label,
    _parse_json_block,
)
from mva.l5_state import WorldStateStore


# ----------------------------------------------------------------------
# Test scaffolding
# ----------------------------------------------------------------------


class ScriptedLLM:
    """Fake LLMClient that returns canned strings in sequence + records
    every prompt it received (for assertions on prompt content)."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._i = 0
        self.prompts_seen: list[str] = []

    def complete(self, prompt: str, **_kw) -> str:
        self.prompts_seen.append(prompt)
        if self._i >= len(self._responses):
            return self._responses[-1]
        out = self._responses[self._i]
        self._i += 1
        return out


@pytest.fixture
def populated_store():
    store = WorldStateStore(":memory:")
    bboxes = []
    for i in range(8):
        t = float(i) * 2.0
        cx = 100.0 + (i % 3 - 1) * 3.0
        cy = 100.0
        bboxes.append([
            t, cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0,
            "person", 0.9,
        ])
    store.insert_tracklet(
        "drone-1", "tk-loiter", 0.0, 14.0, bboxes,
        embedding_ref=None,
    )
    yield store
    store.close()


# ----------------------------------------------------------------------
# detect_anomaly — happy path + parse fallbacks + retry
# ----------------------------------------------------------------------


def test_detect_anomaly_parses_clean_json_response(populated_store):
    llm = ScriptedLLM([
        json.dumps({
            "type": "loitering", "severity": "medium",
            "explanation": "stationary in tight cluster",
        })
    ])
    reasoner = LLMReasoner(store=populated_store, llm_client=llm)
    out = reasoner.detect_anomaly("drone-1", "tk-loiter")
    assert isinstance(out, Anomaly)
    assert out.type == "loitering"
    assert out.severity == "medium"
    assert "stationary" in (out.explanation or "")
    assert out.tracklet_ids == ["tk-loiter"]
    # Prompt should carry the actual track data
    assert "tk-loiter" in llm.prompts_seen[0]
    assert "drone-1" in llm.prompts_seen[0]


def test_detect_anomaly_extracts_json_from_wordy_response(populated_store):
    """LLM sometimes prefixes with prose; we still extract the JSON block."""
    llm = ScriptedLLM([
        'Sure! Here is the result: {"type": "speed_spike", '
        '"severity": "high", "explanation": "well above peers"} OK?'
    ])
    reasoner = LLMReasoner(store=populated_store, llm_client=llm)
    out = reasoner.detect_anomaly("drone-1", "tk-loiter")
    assert isinstance(out, Anomaly)
    assert out.type == "speed_spike"
    assert out.severity == "high"


def test_detect_anomaly_type_none_returns_no_anomaly(populated_store):
    """When the LLM judges 'no anomaly', the reasoner returns None
    (not a Pydantic Anomaly with type='none')."""
    llm = ScriptedLLM([
        json.dumps({"type": "none", "severity": "low", "explanation": "normal"})
    ])
    reasoner = LLMReasoner(store=populated_store, llm_client=llm)
    assert reasoner.detect_anomaly("drone-1", "tk-loiter") is None


def test_detect_anomaly_retries_once_on_malformed_then_gives_up(populated_store):
    """First response is unparseable; second (stricter prompt) also bad
    → return None and don't raise. PLAN §3.5 L4 retry-then-degrade rule."""
    llm = ScriptedLLM([
        "I don't know how to answer this question.",
        "Still no JSON here.",
    ])
    reasoner = LLMReasoner(
        store=populated_store, llm_client=llm, max_retries=1,
    )
    assert reasoner.detect_anomaly("drone-1", "tk-loiter") is None
    # Confirm we did call complete twice (initial + 1 retry)
    assert len(llm.prompts_seen) == 2
    # Retry prompt should contain the stricter "JSON-only" reminder
    assert "JSON" in llm.prompts_seen[1]


def test_detect_anomaly_retry_recovers_after_first_bad_response(populated_store):
    """First call returns prose; second call (stricter) returns clean
    JSON → we accept the retry result."""
    llm = ScriptedLLM([
        "Hmm let me think about this carefully...",
        json.dumps({"type": "loitering", "severity": "medium"}),
    ])
    reasoner = LLMReasoner(
        store=populated_store, llm_client=llm, max_retries=1,
    )
    out = reasoner.detect_anomaly("drone-1", "tk-loiter")
    assert isinstance(out, Anomaly)
    assert out.type == "loitering"


def test_detect_anomaly_rejects_bogus_type(populated_store):
    """LLM hallucinates a type not in {loitering, speed_spike, none} →
    return None rather than letting Pydantic raise downstream."""
    llm = ScriptedLLM([
        json.dumps({"type": "alien_invasion", "severity": "high"})
    ])
    reasoner = LLMReasoner(store=populated_store, llm_client=llm)
    assert reasoner.detect_anomaly("drone-1", "tk-loiter") is None


def test_detect_anomaly_normalizes_invalid_severity(populated_store):
    """Severity outside {low, medium, high} → fall back to 'medium' rather
    than letting Pydantic raise."""
    llm = ScriptedLLM([
        json.dumps({"type": "loitering", "severity": "critical"})
    ])
    reasoner = LLMReasoner(store=populated_store, llm_client=llm)
    out = reasoner.detect_anomaly("drone-1", "tk-loiter")
    assert isinstance(out, Anomaly)
    assert out.severity == "medium"


def test_detect_anomaly_no_store_returns_none():
    """Degenerate path — no store wired → return None without crashing
    (used by the contract fixture)."""
    llm = ScriptedLLM([json.dumps({"type": "loitering", "severity": "medium"})])
    reasoner = LLMReasoner(store=None, llm_client=llm)
    assert reasoner.detect_anomaly("drone-1", "tk-loiter") is None
    # The LLM should not have been called when there's no store to query
    assert llm.prompts_seen == []


def test_detect_anomaly_unknown_tracklet_returns_none(populated_store):
    llm = ScriptedLLM([json.dumps({"type": "loitering", "severity": "medium"})])
    reasoner = LLMReasoner(store=populated_store, llm_client=llm)
    assert reasoner.detect_anomaly("drone-1", "no-such-tracklet") is None
    assert llm.prompts_seen == []


# ----------------------------------------------------------------------
# classify_behavior — strict vocab
# ----------------------------------------------------------------------


def test_classify_behavior_clean_label(populated_store):
    llm = ScriptedLLM(["stationary"])
    reasoner = LLMReasoner(store=populated_store, llm_client=llm)
    out = reasoner.classify_behavior(
        "drone-1", "tk-loiter", context={"class": "person"},
    )
    assert out == "stationary"


def test_classify_behavior_strips_wordy_response(populated_store):
    """LLM wraps the label in a sentence; we extract the substring."""
    llm = ScriptedLLM(["The person appears to be walking forward steadily."])
    reasoner = LLMReasoner(store=populated_store, llm_client=llm)
    out = reasoner.classify_behavior("drone-1", "tk-loiter", context={})
    assert out == "walking"


def test_classify_behavior_rejects_out_of_vocab(populated_store):
    """LLM hallucinates 'dancing' (not in our 6-label vocab) → fall back
    to 'unknown' so callers can branch on it."""
    llm = ScriptedLLM(["dancing"])
    reasoner = LLMReasoner(store=populated_store, llm_client=llm)
    assert reasoner.classify_behavior(
        "drone-1", "tk-loiter", context={},
    ) == "unknown"


def test_classify_behavior_no_client_returns_unknown(populated_store):
    reasoner = LLMReasoner(store=populated_store, llm_client=None)
    assert reasoner.classify_behavior(
        "drone-1", "tk-loiter", context={},
    ) == "unknown"


# ----------------------------------------------------------------------
# predict_trajectory — locked-stub behavior
# ----------------------------------------------------------------------


def test_predict_trajectory_is_locked_to_none(populated_store):
    """M4.2 design lock (2026-05-23): LLM trajectory prediction
    deferred to M5+. Even with a fully wired reasoner the method must
    return None — calling LLM would waste tokens producing fake
    waypoints from K=4 sparse data."""
    llm = ScriptedLLM([json.dumps({"waypoints": [[1, 100, 200]]})])
    reasoner = LLMReasoner(store=populated_store, llm_client=llm)
    assert reasoner.predict_trajectory(
        "drone-1", "tk-loiter", horizon=2.0,
    ) is None
    assert llm.prompts_seen == []


# ----------------------------------------------------------------------
# Module-level helper unit tests (intentionally exposed for testing)
# ----------------------------------------------------------------------


class TestParseJsonBlock:
    def test_clean_json_dict(self):
        assert _parse_json_block('{"a": 1}') == {"a": 1}

    def test_json_with_leading_prose(self):
        out = _parse_json_block('Here is the answer: {"a": 1} done.')
        assert out == {"a": 1}

    def test_unparseable_returns_none(self):
        assert _parse_json_block("no json here at all") is None

    def test_empty_string_returns_none(self):
        assert _parse_json_block("") is None

    def test_non_dict_top_level_rejected(self):
        # Array at the top level shouldn't be coerced to dict
        assert _parse_json_block("[1, 2, 3]") is None


class TestExtractBehaviorLabel:
    def test_clean_label(self):
        assert _extract_behavior_label("walking") == "walking"

    def test_label_in_sentence(self):
        assert _extract_behavior_label(
            "The subject is running fast."
        ) == "running"

    def test_unknown_when_out_of_vocab(self):
        assert _extract_behavior_label("dancing") == "unknown"

    def test_empty_response_is_unknown(self):
        assert _extract_behavior_label("") == "unknown"

    def test_prefers_more_specific_label(self):
        """'vehicle_moving' beats 'unknown' even though both appear."""
        out = _extract_behavior_label(
            "I'd call this vehicle_moving, though I'm not 100% certain unknown."
        )
        assert out == "vehicle_moving"
