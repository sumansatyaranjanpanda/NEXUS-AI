"""Unit tests for the supervisor graph construction and routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.agents.supervisor import (
    _route_after_node,
    build_graph,
    compile_graph,
    hitl_gate_node,
    report_node,
)
from nexus.schemas.state import NexusState, PipelineStage, create_initial_state


class TestRouting:
    """Tests for the supervisor routing function."""

    def test_route_continues_on_non_failed(self) -> None:
        """Should return 'continue' when stage is not FAILED."""
        state = create_initial_state("test")
        state["stage"] = PipelineStage.RESEARCHING
        assert _route_after_node(state) == "continue"

    def test_route_ends_on_failure(self) -> None:
        """Should return END when stage is FAILED."""
        from langgraph.graph import END

        state = create_initial_state("test")
        state["stage"] = PipelineStage.FAILED
        assert _route_after_node(state) == END


class TestHitlGateNode:
    """Tests for the HITL gate node."""

    @pytest.mark.asyncio
    async def test_hitl_sets_review_stage(self) -> None:
        """After HITL approval, stage should be HITL_REVIEW."""
        state = create_initial_state("test query")
        state["stage"] = PipelineStage.FACT_CHECKING
        result = await hitl_gate_node(state)
        assert result["stage"] == PipelineStage.HITL_REVIEW


class TestReportNode:
    """Tests for the report generation node."""

    @pytest.mark.asyncio
    async def test_report_contains_query_and_analysis(self) -> None:
        """Report markdown should include the query and analysis."""
        state = create_initial_state("Who competes with AWS?")
        state["analysis_summary"] = "AWS faces competition from Azure and GCP."
        state["fact_check_results"] = ["1. AWS claim - VERIFIED"]

        result = await report_node(state)

        assert result["stage"] == PipelineStage.COMPLETE
        assert "Who competes with AWS?" in result["report_markdown"]
        assert "AWS faces competition" in result["report_markdown"]
        assert "VERIFIED" in result["report_markdown"]

    @pytest.mark.asyncio
    async def test_report_handles_empty_analysis(self) -> None:
        """Report should handle empty analysis gracefully."""
        state = create_initial_state("test")
        state["analysis_summary"] = ""
        state["fact_check_results"] = []

        result = await report_node(state)
        assert result["stage"] == PipelineStage.COMPLETE
        assert "No analysis available" in result["report_markdown"]


class TestGraphConstruction:
    """Tests for graph build and compile."""

    def test_build_graph_has_all_nodes(self) -> None:
        """Graph should contain all expected nodes."""
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        expected = {"research", "kb_build", "analysis", "factcheck", "hitl_gate", "report"}
        assert expected.issubset(node_names)

    def test_compile_graph_returns_runnable(self) -> None:
        """Compiled graph should have ainvoke method."""
        compiled = compile_graph()
        assert hasattr(compiled, "ainvoke")
        assert hasattr(compiled, "aget_state")
