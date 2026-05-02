"""Unit tests for the HyDE module — mocks OpenRouter/OpenAI client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.rag.hyde import generate_hypothetical_document


def _make_chat_response(text: str) -> MagicMock:
    """Build a mock matching OpenAI's chat.completions.create() return shape."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestHyDE:
    """Tests for generate_hypothetical_document."""

    @pytest.mark.asyncio
    async def test_successful_generation(self) -> None:
        """Should return hypothetical document text on success."""
        mock_response = _make_chat_response("This is a hypothetical competitive intelligence finding.")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch("nexus.rag.hyde.AsyncOpenAI", return_value=mock_client),
            patch("nexus.rag.hyde.Settings"),
        ):
            result = await generate_hypothetical_document("Who are AWS competitors?")

        assert "hypothetical" in result.lower()
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_error_raises_runtime(self) -> None:
        """Should raise RuntimeError on API connection failure."""
        import openai

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        with (
            patch("nexus.rag.hyde.AsyncOpenAI", return_value=mock_client),
            patch("nexus.rag.hyde.Settings"),
        ):
            with pytest.raises(RuntimeError, match="connection"):
                await generate_hypothetical_document("test query")
