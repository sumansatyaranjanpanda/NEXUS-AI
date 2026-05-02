"""Unit tests for the analysis agent — mocks OpenRouter/OpenAI and retriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.agents.analysis_agent import analysis_node
from nexus.schemas.state import PipelineStage, ResearchFinding, create_initial_state


class TestAnalysisNode:
    """Tests for the analysis_node LangGraph node."""

    @pytest.mark.asyncio
    async def test_falls_back_to_raw_findings(self) -> None:
        """When KB retrieval fails, should use raw research findings."""
        state = create_initial_state("test query")
        state["research_findings"] = [
            ResearchFinding(source="test", content="Finding about market share", relevance_score=0.8),
        ]

        from nexus.schemas.report import CompetitiveReport

        mock_report = MagicMock(spec=CompetitiveReport)
        mock_report.executive_summary = "Analysis: market share is growing."
        mock_report.sections = []
        mock_report.model_dump.return_value = {"executive_summary": "Analysis: market share is growing."}

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_report)

        with (
            patch("nexus.agents.analysis_agent.NexusRetriever") as mock_ret_cls,
            patch("nexus.agents.analysis_agent.instructor.from_openai", return_value=mock_client),
            patch("nexus.agents.analysis_agent.AsyncOpenAI"),
            patch("nexus.agents.analysis_agent.Settings"),
        ):
            # Make retriever search raise so it falls back to raw findings
            mock_ret = MagicMock()
            mock_ret.search = AsyncMock(side_effect=RuntimeError("Qdrant down"))
            mock_ret_cls.return_value = mock_ret

            result = await analysis_node(state)

        assert result["stage"] == PipelineStage.ANALYZING
        assert "market share" in result["analysis_summary"].lower()

    @pytest.mark.asyncio
    async def test_fails_with_no_context(self) -> None:
        """Should fail if no findings and no KB results."""
        state = create_initial_state("test query")
        state["research_findings"] = []

        with (
            patch("nexus.agents.analysis_agent.NexusRetriever") as mock_ret_cls,
            patch("nexus.agents.analysis_agent.Settings"),
        ):
            mock_ret = MagicMock()
            mock_ret.search = AsyncMock(return_value=[])
            mock_ret_cls.return_value = mock_ret

            result = await analysis_node(state)

        assert result["stage"] == PipelineStage.FAILED
        assert len(result["errors"]) > 0
