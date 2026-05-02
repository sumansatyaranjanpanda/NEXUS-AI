"""Unit tests for the research agent — uses mocked OpenRouter/OpenAI client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.agents.research_agent import _parse_findings, research_node
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


class TestParseFindings:
    """Tests for the _parse_findings helper."""

    def test_numbered_list_parsing(self) -> None:
        """Should split a numbered list into individual findings."""
        raw = (
            "1. First finding about market share\n"
            "   Additional context for first finding\n"
            "2. Second finding about pricing\n"
            "3. Third finding about technology"
        )
        findings = _parse_findings(raw, "test query")
        assert len(findings) == 3
        assert "market share" in findings[0].content
        assert "pricing" in findings[1].content
        assert "technology" in findings[2].content

    def test_single_block_fallback(self) -> None:
        """When text has no numbered items, wrap the entire text."""
        raw = "This is just a paragraph with no numbered items."
        findings = _parse_findings(raw, "test query")
        assert len(findings) == 1
        assert findings[0].relevance_score == 0.5  # fallback score

    def test_empty_text(self) -> None:
        """Empty input should still produce one finding."""
        findings = _parse_findings("", "test query")
        assert len(findings) == 1


class TestResearchNode:
    """Tests for the research_node LangGraph node."""

    @pytest.mark.asyncio
    async def test_successful_research(self) -> None:
        """Should return findings and RESEARCHING stage on success."""
        mock_response = _make_chat_response("1. Finding one\n2. Finding two")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("nexus.agents.research_agent.AsyncOpenAI", return_value=mock_client),
            patch("nexus.agents.research_agent.Settings"),
        ):
            state = create_initial_state("test query")
            result = await research_node(state)

        assert result["stage"] == PipelineStage.RESEARCHING
        assert len(result["research_findings"]) == 2
        assert "errors" not in result

    @pytest.mark.asyncio
    async def test_api_connection_error(self) -> None:
        """Should return FAILED stage on connection error."""
        import openai

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        with (
            patch("nexus.agents.research_agent.AsyncOpenAI", return_value=mock_client),
            patch("nexus.agents.research_agent.Settings"),
        ):
            state = create_initial_state("test query")
            result = await research_node(state)

        assert result["stage"] == PipelineStage.FAILED
        assert len(result["errors"]) == 1
        assert "connection" in result["errors"][0].lower()
