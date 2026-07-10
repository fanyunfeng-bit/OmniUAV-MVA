"""Pydantic contract tests for L2 CrossViewLink and its implementations.

Parametrized over Geometric + LLM modes: both must satisfy the same
behavioral contract (empty input → [], output shape, ordering).

Pydantic-level validation tests stand alone — they verify the schema itself
catches invalid input regardless of producer.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mva.contracts import CrossViewLink, make_link_id
from mva.l2_crossview import (
    AppearanceCrossViewLinker,
    GeometricCrossViewLinker,
    LLMCrossViewLinker,
)


# ----------------------------------------------------------------------
# Behavioral contract tests — parametrized over all three modes
# (PLAN §6.2 M4.1: `params=[Geometric, Appearance, LLM]`)
# ----------------------------------------------------------------------


@pytest.fixture(
    params=[
        GeometricCrossViewLinker,
        AppearanceCrossViewLinker,
        LLMCrossViewLinker,
    ],
    ids=["geometric", "appearance", "llm"],
)
def linker(request):
    return request.param()


class TestCrossViewLinkerContract:
    def test_empty_input_returns_empty_list(self, linker) -> None:
        """Both modes return [] (not None) when no observations are provided."""
        result = linker.link([])
        assert isinstance(result, list)
        assert result == []

    def test_all_outputs_are_pydantic_validated(self, linker) -> None:
        """Any output entries must be CrossViewLink instances."""
        # M0 stubs return []; once M1 / M4 produce real entries this catches
        # any drift between modes (e.g. dict vs Pydantic).
        result = linker.link([])
        assert all(isinstance(item, CrossViewLink) for item in result)

    def test_output_sorted_by_confidence_descending(self, linker) -> None:
        """Multi-candidate output must be DESC sorted by confidence."""
        result = linker.link([])
        confidences = [link.confidence for link in result]
        assert confidences == sorted(confidences, reverse=True)


# ----------------------------------------------------------------------
# Schema-level Pydantic validation tests (independent of implementations)
# ----------------------------------------------------------------------


class TestCrossViewLinkSchema:
    def test_valid_link_accepted(self) -> None:
        link = CrossViewLink(
            link_id="x",
            view_observations=[("drone-1", "t1"), ("drone-2", "t2")],
            confidence=0.75,
            created_by="geometric",
            created_at=1234567890.0,
        )
        assert link.confidence == 0.75
        assert link.created_by == "geometric"

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CrossViewLink(
                link_id="x",
                view_observations=[("drone-1", "t1"), ("drone-2", "t2")],
                confidence=1.5,
                created_by="geometric",
                created_at=0.0,
            )

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CrossViewLink(
                link_id="x",
                view_observations=[("drone-1", "t1"), ("drone-2", "t2")],
                confidence=-0.1,
                created_by="geometric",
                created_at=0.0,
            )

    def test_single_view_observation_rejected(self) -> None:
        """Cross-view link must have >= 2 observations."""
        with pytest.raises(ValidationError):
            CrossViewLink(
                link_id="x",
                view_observations=[("drone-1", "t1")],
                confidence=0.5,
                created_by="geometric",
                created_at=0.0,
            )

    def test_same_view_twice_rejected(self) -> None:
        """Two observations from the same view do NOT form a cross-view link."""
        with pytest.raises(ValidationError):
            CrossViewLink(
                link_id="x",
                view_observations=[("drone-1", "t1"), ("drone-1", "t2")],
                confidence=0.5,
                created_by="geometric",
                created_at=0.0,
            )

    def test_created_by_enum_includes_human(self) -> None:
        """L7 needs 'human' as a valid created_by value (§3.4 #8)."""
        link = CrossViewLink(
            link_id="x",
            view_observations=[("drone-1", "t1"), ("drone-2", "t2")],
            confidence=1.0,
            created_by="human",
            created_at=0.0,
        )
        assert link.created_by == "human"

    def test_created_by_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CrossViewLink(
                link_id="x",
                view_observations=[("drone-1", "t1"), ("drone-2", "t2")],
                confidence=1.0,
                created_by="random_invalid",  # type: ignore[arg-type]
                created_at=0.0,
            )


class TestMakeLinkIdDeterminism:
    """`make_link_id` is the foundation of cross_view_links idempotency
    on `mva ingest` rerun. Same logical link → same id, regardless of
    observation order or call timing."""

    def test_same_observations_same_id(self) -> None:
        obs = [("drone-1", "tk-a"), ("drone-2", "tk-b")]
        assert make_link_id(obs) == make_link_id(obs)

    def test_order_independent(self) -> None:
        a = [("drone-1", "tk-a"), ("drone-2", "tk-b")]
        b = [("drone-2", "tk-b"), ("drone-1", "tk-a")]
        assert make_link_id(a) == make_link_id(b)

    def test_different_observations_different_id(self) -> None:
        a = [("drone-1", "tk-a"), ("drone-2", "tk-b")]
        b = [("drone-1", "tk-a"), ("drone-2", "tk-different")]
        assert make_link_id(a) != make_link_id(b)

    def test_id_is_short_hex_string(self) -> None:
        out = make_link_id([("v1", "t1"), ("v2", "t2")])
        assert isinstance(out, str)
        assert len(out) == 16
        int(out, 16)  # raises if not valid hex
