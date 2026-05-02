"""Chunker — splits documents into overlapping text chunks for embedding."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

import structlog
from pydantic import BaseModel, Field

logger = structlog.stdlib.get_logger(__name__)


class TextChunk(BaseModel):
    """A single chunk of text ready for embedding and storage."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(description="The chunk text")
    index: int = Field(description="Position of this chunk within the source document")
    source_id: str = Field(default="", description="Identifier of the source document")
    metadata: dict[str, str] = Field(default_factory=dict)
    content_hash: str = Field(default="", description="SHA-256 of content for dedup")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    source_id: str = "",
    metadata: dict[str, str] | None = None,
) -> list[TextChunk]:
    """Split text into overlapping chunks by character count.

    Args:
        text: The full document text to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
        source_id: Identifier for the source document.
        metadata: Optional metadata to attach to every chunk.

    Returns:
        List of TextChunk objects ready for embedding.
    """
    if not text or not text.strip():
        logger.warning("chunker.empty_input", source_id=source_id)
        return []

    if chunk_overlap >= chunk_size:
        logger.error("chunker.invalid_config", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

    clean_text = text.strip()
    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap
    start = 0
    index = 0

    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunk_content = clean_text[start:end].strip()

        if chunk_content:
            content_hash = hashlib.sha256(chunk_content.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                TextChunk(
                    content=chunk_content,
                    index=index,
                    source_id=source_id,
                    metadata=metadata or {},
                    content_hash=content_hash,
                )
            )
            index += 1

        start += step

    logger.info(
        "chunker.complete",
        source_id=source_id,
        num_chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunks


def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[TextChunk]:
    """Chunk multiple documents. Each dict must have 'text' and optionally 'source_id' and 'metadata'.

    Args:
        documents: List of dicts with keys 'text', 'source_id' (optional), 'metadata' (optional).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        Flat list of TextChunk objects from all documents.
    """
    all_chunks: list[TextChunk] = []

    for doc in documents:
        text = doc.get("text", "")
        source_id = doc.get("source_id", "")
        # metadata values should be strings for Qdrant payload compatibility
        raw_meta = doc.get("metadata", {})
        meta = {k: str(v) for k, v in raw_meta.items()} if raw_meta else {}

        doc_chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_id=source_id,
            metadata=meta,
        )
        all_chunks.extend(doc_chunks)

    logger.info("chunker.batch_complete", num_documents=len(documents), total_chunks=len(all_chunks))
    return all_chunks
