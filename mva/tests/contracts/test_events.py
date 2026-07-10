"""Pydantic contract tests for L3 outputs: TrajectoryPrediction, Anomaly.

Both Mode A (algorithmic) and Mode B (LLM-mode) implementations must produce
instances that satisfy these contracts. See PLAN.md §3.4 #3 and Eng Review 1C.

M3.2 adds parametrized behavioral tests on top of the Pydantic schema tests:
both modes must return Optional[Pydantic-validated] (never bare dicts /
exceptions) for degenerate / unknown inputs.

M4.2 (2026-05-23) promotes `LLMReasoner` from stub to real impl and adds an
**active-path** parametrized contract: both modes must also honor the
contract when given a populated store + (for LLM) a working LLM client.
Per the M4.2 design lock, `predict_trajectory` in LLM mode returns None
(deferred to M5+); the contract test treats None as a pass.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mva.contracts import Anomaly, TrajectoryPrediction
from mva.l3_events import AlgorithmicReasoner, LLMReasoner, Reasoner
from mva.l5_state import WorldStateStore


# ----------------------------------------------------------------------
# Behavioral contract — parametrized over both modes (degenerate path)
# ----------------------------------------------------------------------


@pytest.fixture(
    params=[AlgorithmicReasoner, LLMReasoner],
    ids=["algorithmic", "llm"],
)
def reasoner(request):
    # Both ctors accept zero args (AlgorithmicReasoner accepts an
    # optional `store` kwarg; LLMReasoner accepts optional store + llm_client).
    # Absent → no-data path that satisfies the "return Optional[T]" contract.
    return request.param()


class TestReasonerContract:
    def test_satisfies_protocol(self, reasoner) -> None:
        assert isinstance(reasoner, Reasoner)

    def test_predict_trajectory_returns_optional_pydantic(self, reasoner) -> None:
        out = reasoner.predict_trajectory("drone-1", "unknown-tk", horizon=2.0)
        assert out is None or isinstance(out, TrajectoryPrediction)

    def test_detect_anomaly_returns_optional_pydantic(self, reasoner) -> None:
        out = reasoner.detect_anomaly("drone-1", "unknown-tk")
        assert out is None or isinstance(out, Anomaly)

    def test_classify_behavior_returns_string(self, reasoner) -> None:
        out = reasoner.classify_behavior("drone-1", "tk", context={})
        assert isinstance(out, str)


# ----------------------------------------------------------------------
# M4.2 active-path contract — both modes with a populated store
# ----------------------------------------------------------------------


class _ScriptedLLM:
    """Minimal `complete` shim that returns canned strings in sequence.

    Lets the contract test exercise LLMReasoner's real code paths
    (prompt building + JSON parse + Pydantic assembly) without spinning
    up a 14 GB model. Last canned string is reused if the test calls
    `complete()` more times than scripted responses (so retry paths
    don't crash)."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._i = 0

    def complete(self, prompt: str, **_kw) -> str:
        if self._i >= len(self._responses):
            return self._responses[-1]
        out = self._responses[self._i]
        self._i += 1
        return out


def _populated_store() -> WorldStateStore:
    store = WorldStateStore(":memory:")
    # A 40s loitering tracklet at (100, 100) ± 5px — enough for both the
    # algorithmic loitering rule AND a coherent prompt for the LLM.
    bboxes = []
    for i in range(20):
        t = float(i) * 2.0
        cx = 100.0 + (i % 3 - 1) * 4.0
        cy = 100.0 + (i % 2) * 4.0
        bboxes.append([
            t, cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0,
            "person", 0.9,
        ])
    store.insert_tracklet(
        "drone-1", "tk-active", 0.0, 38.0, bboxes,
        embedding_ref=None,
    )
    return store


@pytest.fixture(
    params=["algorithmic", "llm"],
)
def active_reasoner(request):
    store = _populated_store()
    if request.param == "algorithmic":
        yield AlgorithmicReasoner(store=store)
    else:
        # Scripted LLM: answer detect_anomaly as loitering JSON, behavior
        # as a clean label. Reused for any extra retry calls.
        llm = _ScriptedLLM([
            json.dumps({
                "type": "loitering", "severity": "medium",
                "explanation": "stationary 40s in tight cluster",
            }),
            "stationary",   # classify_behavior label
        ])
        yield LLMReasoner(store=store, llm_client=llm)
    store.close()


class TestReasonerActiveContract:
    """Contract still holds when each mode has real work to do.

    Treats `predict_trajectory` in LLM mode as "may return None" per the
    2026-05-23 design lock (PLAN §6.2 M4.2). Algorithmic mode must
    return a real prediction on this fixture (20 points across 38s)."""

    def test_predict_trajectory_active(self, active_reasoner) -> None:
        out = active_reasoner.predict_trajectory("drone-1", "tk-active", horizon=2.0)
        if isinstance(active_reasoner, LLMReasoner):
            # M4.2 LOCKED: LLM predict_trajectory deferred to M5+ → None pass
            assert out is None
        else:
            assert isinstance(out, TrajectoryPrediction)
            assert out.tracklet_id == "tk-active"
            assert 0.0 <= out.confidence <= 1.0

    def test_detect_anomaly_active(self, active_reasoner) -> None:
        out = active_reasoner.detect_anomaly("drone-1", "tk-active")
        # Both modes should detect SOMETHING (algorithmic via rule,
        # LLM via scripted JSON) — contract is "Anomaly or None"; we
        # additionally check that on a manifestly-loitering tracklet
        # neither mode returns None.
        assert isinstance(out, Anomaly)
        assert out.tracklet_ids == ["tk-active"]
        assert out.severity in ("low", "medium", "high")

    def test_classify_behavior_active(self, active_reasoner) -> None:
        out = active_reasoner.classify_behavior(
            "drone-1", "tk-active", context={"class": "person"},
        )
        assert isinstance(out, str)
        # The algorithmic baseline still returns "unknown" (M5 work); the
        # LLM mode must produce a known-vocab label on this clean fixture.
        if isinstance(active_reasoner, LLMReasoner):
            assert out != "unknown"


class TestTrajectoryPrediction:
    def test_valid_prediction_accepted(self):
        t = TrajectoryPrediction(
            view_id="drone-1",
            tracklet_id="tk-42",
            horizon_seconds=5.0,
            waypoints=[(0.0, 100.0, 200.0), (1.0, 105.0, 210.0)],
            confidence=0.8,
        )
        assert len(t.waypoints) == 2

    def test_horizon_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrajectoryPrediction(
                view_id="drone-1",
                tracklet_id="tk-42",
                horizon_seconds=0.0,
                waypoints=[(0.0, 100.0, 200.0)],
                confidence=0.8,
            )

    def test_empty_waypoints_rejected(self):
        with pytest.raises(ValidationError):
            TrajectoryPrediction(
                view_id="drone-1",
                tracklet_id="tk-42",
                horizon_seconds=5.0,
                waypoints=[],
                confidence=0.8,
            )

    def test_confidence_in_unit_range(self):
        with pytest.raises(ValidationError):
            TrajectoryPrediction(
                view_id="drone-1",
                tracklet_id="tk-42",
                horizon_seconds=5.0,
                waypoints=[(0.0, 100.0, 200.0)],
                confidence=1.5,
            )


class TestAnomaly:
    def test_valid_anomaly_accepted(self):
        a = Anomaly(
            event_id="a-1",
            tracklet_ids=["tk-42"],
            t=1234567890.0,
            type="loitering",
            severity="medium",
        )
        assert a.severity == "medium"
        assert a.explanation is None

    def test_severity_enum(self):
        for sev in ("low", "medium", "high"):
            Anomaly(
                event_id="a-1",
                tracklet_ids=["tk-42"],
                t=0.0,
                type="x",
                severity=sev,
            )

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError):
            Anomaly(
                event_id="a-1",
                tracklet_ids=["tk-42"],
                t=0.0,
                type="x",
                severity="critical",
            )

    def test_explanation_may_be_filled_by_llm_mode(self):
        a = Anomaly(
            event_id="a-1",
            tracklet_ids=["tk-42"],
            t=0.0,
            type="loitering",
            severity="medium",
            explanation="Person stationary for > 5 minutes near restricted gate.",
        )
        assert a.explanation is not None
