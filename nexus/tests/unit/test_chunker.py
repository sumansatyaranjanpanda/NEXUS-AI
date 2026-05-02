"""Unit tests for the chunker module."""

from nexus.rag.chunker import TextChunk, chunk_documents, chunk_text


class TestChunkText:
    """Tests for chunk_text function."""

    def test_basic_chunking(self) -> None:
        """Should split text into chunks of the configured size."""
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
        assert len(chunks) == 4  # 300, 300, 300, 100
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(len(c.content) <= 300 for c in chunks)

    def test_overlap_is_present(self) -> None:
        """Adjacent chunks should share overlapping content."""
        text = "abcdefghijklmnopqrst"  # 20 chars
        chunks = chunk_text(text, chunk_size=10, chunk_overlap=3)
        # First chunk: "abcdefghij" (0-10)
        # Second chunk: "hijklmnopq" (7-17) — overlaps with first by 3
        assert chunks[0].content[-3:] == chunks[1].content[:3]

    def test_empty_text_returns_empty(self) -> None:
        """Empty or whitespace-only input should return no chunks."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_source_id_propagation(self) -> None:
        """Source ID should be attached to every chunk."""
        chunks = chunk_text("Hello world test text", chunk_size=10, chunk_overlap=2, source_id="doc-42")
        assert all(c.source_id == "doc-42" for c in chunks)

    def test_metadata_propagation(self) -> None:
        """Metadata dict should be attached to every chunk."""
        meta = {"category": "analysis", "year": "2025"}
        chunks = chunk_text("Some text here", chunk_size=100, chunk_overlap=10, metadata=meta)
        assert all(c.metadata == meta for c in chunks)

    def test_content_hash_is_stable(self) -> None:
        """Same content should produce same hash."""
        chunks_a = chunk_text("Exactly the same text", chunk_size=100, chunk_overlap=10)
        chunks_b = chunk_text("Exactly the same text", chunk_size=100, chunk_overlap=10)
        assert chunks_a[0].content_hash == chunks_b[0].content_hash

    def test_content_hash_differs_for_different_content(self) -> None:
        """Different content should produce different hashes."""
        chunks_a = chunk_text("Text version A", chunk_size=100, chunk_overlap=10)
        chunks_b = chunk_text("Text version B", chunk_size=100, chunk_overlap=10)
        assert chunks_a[0].content_hash != chunks_b[0].content_hash

    def test_index_is_sequential(self) -> None:
        """Chunk indices should be sequential starting from 0."""
        chunks = chunk_text("A" * 500, chunk_size=100, chunk_overlap=10)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_overlap_must_be_less_than_size(self) -> None:
        """Should raise ValueError if overlap >= chunk_size."""
        import pytest

        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("some text", chunk_size=10, chunk_overlap=10)

    def test_small_text_single_chunk(self) -> None:
        """Text smaller than chunk_size should produce exactly one chunk."""
        chunks = chunk_text("Short", chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0].content == "Short"


class TestChunkDocuments:
    """Tests for chunk_documents batch function."""

    def test_multiple_documents(self) -> None:
        """Should chunk multiple documents into a flat list."""
        docs = [
            {"text": "A" * 200, "source_id": "doc-1"},
            {"text": "B" * 300, "source_id": "doc-2"},
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        # doc-1 → 3 chunks (200 / 90 step ≈ 3), doc-2 → 4 chunks (300 / 90 step ≈ 4)
        assert len(chunks) > 4
        assert any(c.source_id == "doc-1" for c in chunks)
        assert any(c.source_id == "doc-2" for c in chunks)

    def test_empty_document_list(self) -> None:
        """Empty document list should return empty chunks."""
        assert chunk_documents([]) == []
