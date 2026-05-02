"""Unit tests for NexusState and related schemas."""

from nexus.schemas.state import (
    NexusState,
    PipelineStage,
    ResearchFinding,
    create_initial_state,
)


def test_create_initial_state_sets_defaults() -> None:
    """create_initial_state should return a valid NexusState with correct defaults."""
    state: NexusState = create_initial_state("Who are the top competitors in cloud AI?")

    assert state["query"] == "Who are the top competitors in cloud AI?"
    assert state["stage"] == PipelineStage.RECEIVED
    assert state["run_id"]  # should be a non-empty UUID string
    assert state["messages"] == []
    assert state["research_findings"] == []
    assert state["analysis_summary"] == ""
    assert state["fact_check_results"] == []
    assert state["report_markdown"] == ""
    assert state["errors"] == []


def test_create_initial_state_run_id_is_unique() -> None:
    """Each call should produce a distinct run_id."""
    s1 = create_initial_state("query A")
    s2 = create_initial_state("query B")
    assert s1["run_id"] != s2["run_id"]


def test_pipeline_stage_values() -> None:
    """All expected stages should exist."""
    assert PipelineStage.RECEIVED == "received"
    assert PipelineStage.RESEARCHING == "researching"
    assert PipelineStage.FAILED == "failed"
    assert PipelineStage.COMPLETE == "complete"


def test_research_finding_defaults() -> None:
    """ResearchFinding should accept minimal args and set defaults."""
    finding = ResearchFinding(source="test", content="some finding")
    assert finding.relevance_score == 0.0
    assert finding.retrieved_at is not None


def test_research_finding_validation() -> None:
    """Relevance score should be bounded [0, 1]."""
    import pytest

    with pytest.raises(Exception):  # noqa: B017 — Pydantic validation error
        ResearchFinding(source="test", content="bad score", relevance_score=1.5)
