"""Unit tests for the fact-check agent — mocks OpenRouter/OpenAI."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.agents.factcheck_agent import factcheck_node
from nexus.schemas.state import PipelineStage, create_initial_state


def _make_chat_response(text: str) -> MagicMock:
    """Build a mock matching OpenAI's chat.completions.create() return shape."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestFactcheckNode:
    """Tests for the factcheck_node LangGraph node."""

    @pytest.mark.asyncio
    async def test_successful_factcheck(self) -> None:
        """Should return fact_check_results on success."""
        state = create_initial_state("test query")
        state["analysis_summary"] = "AWS has 32% market share in cloud computing."

        mock_response = _make_chat_response("1. AWS 32% claim - VERIFIED\nOverall: PASS")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("nexus.agents.factcheck_agent.AsyncOpenAI", return_value=mock_client),
            patch("nexus.agents.factcheck_agent.Settings"),
        ):
            result = await factcheck_node(state)

        assert result["stage"] == PipelineStage.FACT_CHECKING
        assert len(result["fact_check_results"]) == 2
        assert any("VERIFIED" in r for r in result["fact_check_results"])

    @pytest.mark.asyncio
    async def test_fails_with_empty_analysis(self) -> None:
        """Should fail when analysis_summary is empty."""
        state = create_initial_state("test query")
        state["analysis_summary"] = ""

        result = await factcheck_node(state)

        assert result["stage"] == PipelineStage.FAILED
        assert len(result["errors"]) > 0
