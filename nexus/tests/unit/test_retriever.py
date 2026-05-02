"""Unit tests for the retriever — mocks Qdrant, embedder, and Cohere."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.rag.retriever import NexusRetriever, RetrievalResult


class TestRetrievalResult:
    """Tests for RetrievalResult data class."""

    def test_to_dict(self) -> None:
        """Should serialize all fields to a dict."""
        result = RetrievalResult(
            chunk_id="abc-123",
            content="Some content",
            score=0.95,
            source_id="doc-1",
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert d["chunk_id"] == "abc-123"
        assert d["content"] == "Some content"
        assert d["score"] == 0.95
        assert d["source_id"] == "doc-1"
        assert d["metadata"] == {"key": "value"}

    def test_defaults(self) -> None:
        """Optional fields should default safely."""
        result = RetrievalResult(chunk_id="x", content="y", score=0.5)
        assert result.source_id == ""
        assert result.metadata == {}


class TestNexusRetrieverRerank:
    """Tests for the _rerank method."""

    def test_rerank_reorders_results(self) -> None:
        """Cohere rerank should reorder and trim results."""
        mock_settings = MagicMock()
        mock_settings.cohere_api_key = "test-key"
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_collection = "test"
        mock_settings.embedding_model = "BAAI/bge-small-en-v1.5"
        mock_settings.sparse_model = "prithivida/Splade_PP_en_v1"
        mock_settings.retrieval_top_k = 10
        mock_settings.rerank_top_k = 2

        with patch("nexus.rag.retriever.NexusEmbedder"), patch("nexus.rag.retriever.QdrantClient"):
            retriever = NexusRetriever(settings=mock_settings)

        results = [
            RetrievalResult(chunk_id="1", content="First", score=0.9),
            RetrievalResult(chunk_id="2", content="Second", score=0.8),
            RetrievalResult(chunk_id="3", content="Third", score=0.7),
        ]

        mock_rerank_item_1 = MagicMock()
        mock_rerank_item_1.index = 2  # Third result is now first
        mock_rerank_item_1.relevance_score = 0.99

        mock_rerank_item_2 = MagicMock()
        mock_rerank_item_2.index = 0  # First result is now second
        mock_rerank_item_2.relevance_score = 0.85

        mock_rerank_response = MagicMock()
        mock_rerank_response.results = [mock_rerank_item_1, mock_rerank_item_2]

        with patch("nexus.rag.retriever.cohere.Client") as mock_cohere_cls:
            mock_cohere_cls.return_value.rerank.return_value = mock_rerank_response
            reranked = retriever._rerank("test query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].chunk_id == "3"  # Was third, now first
        assert reranked[0].score == 0.99
        assert reranked[1].chunk_id == "1"  # Was first, now second

    def test_rerank_fallback_on_error(self) -> None:
        """If Cohere fails, should return unranked results truncated to top_k."""
        mock_settings = MagicMock()
        mock_settings.cohere_api_key = "test-key"
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_collection = "test"
        mock_settings.embedding_model = "BAAI/bge-small-en-v1.5"
        mock_settings.sparse_model = "prithivida/Splade_PP_en_v1"

        with patch("nexus.rag.retriever.NexusEmbedder"), patch("nexus.rag.retriever.QdrantClient"):
            retriever = NexusRetriever(settings=mock_settings)

        results = [
            RetrievalResult(chunk_id="1", content="First", score=0.9),
            RetrievalResult(chunk_id="2", content="Second", score=0.8),
            RetrievalResult(chunk_id="3", content="Third", score=0.7),
        ]

        with patch("nexus.rag.retriever.cohere.Client") as mock_cohere_cls:
            mock_cohere_cls.return_value.rerank.side_effect = Exception("Cohere API down")
            fallback = retriever._rerank("test query", results, top_k=2)

        assert len(fallback) == 2
        assert fallback[0].chunk_id == "1"  # Original order preserved


class TestNexusRetrieverIngest:
    """Tests for the ingest method."""

    def test_ingest_empty_returns_zero(self) -> None:
        """Ingesting an empty list should return 0."""
        mock_settings = MagicMock()
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_collection = "test"
        mock_settings.embedding_model = "BAAI/bge-small-en-v1.5"
        mock_settings.sparse_model = "prithivida/Splade_PP_en_v1"

        with patch("nexus.rag.retriever.NexusEmbedder"), patch("nexus.rag.retriever.QdrantClient"):
            retriever = NexusRetriever(settings=mock_settings)

        assert retriever.ingest([]) == 0
