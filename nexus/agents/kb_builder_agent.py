"""KB Builder Agent — chunks research findings and ingests them into Qdrant."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from nexus.rag.chunker import chunk_documents
from nexus.rag.retriever import NexusRetriever
from nexus.schemas.config import Settings
from nexus.schemas.state import NexusState, PipelineStage
from nexus.utils.progress import push_progress

logger = structlog.stdlib.get_logger(__name__)


from langfuse import observe

@observe()
async def kb_builder_node(state: NexusState) -> dict:
    """LangGraph node: takes research findings and ingests them into the vector KB.

    Args:
        state: Current pipeline state with research_findings populated.

    Returns:
        Partial state update with updated stage.
    """
    run_id = state["run_id"]
    findings = state.get("research_findings", [])
    query = state["query"]
    log = logger.bind(run_id=run_id, num_findings=len(findings))
    log.info("kb_builder.start")

    if not findings:
        log.warning("kb_builder.no_findings")
        return {
            "stage": PipelineStage.BUILDING_KB,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Convert findings into documents for chunking
    documents = []
    for i, finding in enumerate(findings):
        finding_dict = finding if isinstance(finding, dict) else finding.model_dump()
        documents.append({
            "text": finding_dict.get("content", ""),
            "source_id": f"{run_id}:finding:{i}",
            "metadata": {
                "query": query[:200],
                "source": finding_dict.get("source", "unknown"),
                "run_id": run_id,
            },
        })

    settings = Settings()

    await push_progress(run_id, f"Chunking {len(findings)} research findings (size={settings.chunk_size}, overlap={settings.chunk_overlap})...", stage="building_kb")
    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    log.info("kb_builder.chunked", num_chunks=len(chunks))
    await push_progress(run_id, f"Created {len(chunks)} chunks — embedding with bge-small-en-v1.5 + SPLADE...", stage="building_kb")

    if not chunks:
        log.warning("kb_builder.no_chunks_produced")
        return {
            "stage": PipelineStage.BUILDING_KB,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Ingest into Qdrant
    try:
        retriever = NexusRetriever(settings)
        retriever.ensure_collection(dense_dim=settings.embedding_dim)
        num_upserted = retriever.ingest(chunks)
        log.info("kb_builder.ingested", num_upserted=num_upserted)
        await push_progress(run_id, f"Stored {num_upserted} vectors in Qdrant (hybrid: dense + sparse)", stage="building_kb")
    except RuntimeError as exc:
        log.error("kb_builder.ingest_error", error=str(exc))
        return {
            "stage": PipelineStage.FAILED,
            "errors": [f"KB ingestion failed: {exc}"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "stage": PipelineStage.BUILDING_KB,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
