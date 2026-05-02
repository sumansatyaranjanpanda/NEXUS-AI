"""Knowledge Base router — document ingestion and collection statistics."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from nexus.rag.chunker import chunk_documents
from nexus.rag.retriever import NexusRetriever
from nexus.schemas.config import Settings

logger = structlog.stdlib.get_logger(__name__)
router = APIRouter(prefix="/kb", tags=["knowledge-base"])


class IngestDocument(BaseModel):
    """A single document to ingest."""

    text: str = Field(min_length=10, description="Full document text")
    title: str = Field(default="", description="Optional title prepended to text before chunking")
    url: str = Field(default="", description="Source URL for attribution in RAG citations")


class IngestRequest(BaseModel):
    """Batch of documents to add to the knowledge base."""

    documents: list[IngestDocument]
    source_label: str = Field(
        default="manual-ingest",
        description="Label used as source_id when no URL is provided",
    )


class IngestResponse(BaseModel):
    source_label: str
    num_documents: int
    num_chunks: int
    num_upserted: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """Ingest documents into the Qdrant knowledge base.

    Use this to pre-seed the KB with company reports, articles, competitor data etc.
    Pre-seeding dramatically improves RAG retrieval quality because the analysis
    agent has real, curated context instead of only the current run's findings.
    """
    settings = Settings()
    log = logger.bind(num_docs=len(request.documents), source=request.source_label)
    log.info("kb.ingest_request")

    raw_docs = [
        {
            "text": f"{doc.title}\n\n{doc.text}" if doc.title else doc.text,
            "source_id": doc.url or f"{request.source_label}:{i}",
            "metadata": {
                "source": doc.url or request.source_label,
                "title": doc.title,
            },
        }
        for i, doc in enumerate(request.documents)
    ]

    chunks = chunk_documents(
        raw_docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    log.info("kb.chunked", num_chunks=len(chunks))

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced — check that documents have sufficient text")

    try:
        retriever = NexusRetriever(settings)
        retriever.ensure_collection(dense_dim=settings.embedding_dim)
        num_upserted = retriever.ingest(chunks)
    except RuntimeError as exc:
        log.error("kb.ingest_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"KB ingestion failed: {exc}") from exc

    log.info("kb.ingest_complete", num_upserted=num_upserted)
    return IngestResponse(
        source_label=request.source_label,
        num_documents=len(request.documents),
        num_chunks=len(chunks),
        num_upserted=num_upserted,
    )


@router.get("/stats", response_model=dict)
async def kb_stats() -> dict:
    """Return Qdrant collection statistics: point count, status, vector dimensions."""
    settings = Settings()
    try:
        from qdrant_client import QdrantClient

        qc = QdrantClient(url=settings.qdrant_url, timeout=30)
        info = qc.get_collection(collection_name=settings.qdrant_collection)
        return {
            "collection": settings.qdrant_collection,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
