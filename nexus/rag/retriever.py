"""Retriever — Qdrant hybrid search (dense + sparse) with Cohere reranking."""

from __future__ import annotations

import cohere
import structlog
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from nexus.rag.chunker import TextChunk
from nexus.rag.embedder import NexusEmbedder
from nexus.rag.hyde import generate_hypothetical_document
from nexus.schemas.config import Settings

logger = structlog.stdlib.get_logger(__name__)


class RetrievalResult:
    """A single search result from the retriever."""

    __slots__ = ("chunk_id", "content", "score", "source_id", "metadata")

    def __init__(
        self,
        chunk_id: str,
        content: str,
        score: float,
        source_id: str = "",
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.source_id = source_id
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "source_id": self.source_id,
            "metadata": self.metadata,
        }


class NexusRetriever:
    """Handles Qdrant collection management, ingestion, hybrid search, and reranking.

    Uses named vectors:
      - 'dense': bge-m3 dense embeddings
      - 'sparse': SPLADE sparse embeddings

    Search flow:
      1. (Optional) HyDE: generate hypothetical document from query
      2. Embed the query (or hypothetical doc) with both dense + sparse models
      3. Hybrid search in Qdrant using both vector types
      4. (Optional) Cohere rerank on the results
      5. Return top-k results
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._embedder = NexusEmbedder(self._settings)
        self._qdrant = QdrantClient(url=self._settings.qdrant_url, timeout=30)
        self._collection = self._settings.qdrant_collection

    def ensure_collection(self, dense_dim: int = 1024) -> None:
        """Create the Qdrant collection if it doesn't exist.

        Args:
            dense_dim: Dimensionality of the dense embedding model.
                       bge-m3 = 1024, bge-small-en-v1.5 = 384.
        """
        try:
            collections = self._qdrant.get_collections().collections
            existing = {c.name for c in collections}
            if self._collection in existing:
                logger.info("retriever.collection_exists", collection=self._collection)
                return
        except Exception as exc:
            logger.error("retriever.list_collections_error", error=str(exc))
            raise RuntimeError(f"Failed to list Qdrant collections: {exc}") from exc

        try:
            self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dense_dim,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )
            logger.info(
                "retriever.collection_created",
                collection=self._collection,
                dense_dim=dense_dim,
            )
        except Exception as exc:
            logger.error("retriever.create_collection_error", error=str(exc))
            raise RuntimeError(f"Failed to create Qdrant collection: {exc}") from exc

    def ingest(self, chunks: list[TextChunk]) -> int:
        """Embed and store chunks in Qdrant.

        Args:
            chunks: List of TextChunk objects to ingest.

        Returns:
            Number of points successfully upserted.
        """
        if not chunks:
            logger.warning("retriever.ingest_empty")
            return 0

        texts = [c.content for c in chunks]
        log = logger.bind(num_chunks=len(texts))
        log.info("retriever.ingest_start")

        # Generate embeddings
        try:
            dense_embeddings = self._embedder.embed_documents_dense(texts)
            sparse_embeddings = self._embedder.embed_documents_sparse(texts)
        except RuntimeError as exc:
            log.error("retriever.embedding_error", error=str(exc))
            raise

        # Build Qdrant points
        points = []
        for i, chunk in enumerate(chunks):
            point = models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    "dense": dense_embeddings[i],
                },
                payload={
                    "content": chunk.content,
                    "source_id": chunk.source_id,
                    "index": chunk.index,
                    "content_hash": chunk.content_hash,
                    **chunk.metadata,
                },
            )
            # Attach sparse vector
            point.vector["sparse"] = models.SparseVector(
                indices=sparse_embeddings[i]["indices"],
                values=sparse_embeddings[i]["values"],
            )
            points.append(point)

        # Upsert in batches of 100
        batch_size = 100
        total_upserted = 0
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start : batch_start + batch_size]
            try:
                self._qdrant.upsert(
                    collection_name=self._collection,
                    points=batch,
                )
                total_upserted += len(batch)
            except UnexpectedResponse as exc:
                log.error("retriever.upsert_error", batch_start=batch_start, error=str(exc))
                raise RuntimeError(f"Qdrant upsert failed at batch {batch_start}: {exc}") from exc

        log.info("retriever.ingest_complete", total_upserted=total_upserted)
        return total_upserted

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        use_hyde: bool = False,
        use_rerank: bool = True,
    ) -> list[RetrievalResult]:
        """Hybrid search with optional HyDE and Cohere reranking.

        Args:
            query: The user's search query.
            top_k: Number of final results to return (defaults to settings.rerank_top_k).
            use_hyde: Whether to use HyDE for the query embedding.
            use_rerank: Whether to apply Cohere reranking.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        top_k = top_k or self._settings.rerank_top_k
        fetch_k = self._settings.retrieval_top_k
        log = logger.bind(query=query[:100], top_k=top_k, use_hyde=use_hyde, use_rerank=use_rerank)
        log.info("retriever.search_start")

        # Step 1: Optionally generate hypothetical document
        embed_text = query
        if use_hyde:
            try:
                embed_text = await generate_hypothetical_document(query, self._settings)
                log.info("retriever.hyde_applied", hyde_length=len(embed_text))
            except RuntimeError as exc:
                log.warning("retriever.hyde_fallback", error=str(exc))
                embed_text = query  # fall back to raw query

        # Step 2: Embed the query (or HyDE doc) with both models
        try:
            dense_vector = self._embedder.embed_query_dense(embed_text)
            sparse_vector = self._embedder.embed_query_sparse(embed_text)
        except RuntimeError as exc:
            log.error("retriever.query_embedding_error", error=str(exc))
            raise

        # Step 3: Hybrid search — prefetch sparse, then combine with dense
        try:
            results = self._qdrant.query_points(
                collection_name=self._collection,
                prefetch=[
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vector["indices"],
                            values=sparse_vector["values"],
                        ),
                        using="sparse",
                        limit=fetch_k,
                    ),
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=fetch_k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=fetch_k,
            )
        except Exception as exc:
            log.error("retriever.search_error", error=str(exc))
            raise RuntimeError(f"Qdrant search failed: {exc}") from exc

        # Convert to RetrievalResult
        search_results = []
        for point in results.points:
            payload = point.payload or {}
            search_results.append(
                RetrievalResult(
                    chunk_id=str(point.id),
                    content=payload.get("content", ""),
                    score=point.score if point.score is not None else 0.0,
                    source_id=payload.get("source_id", ""),
                    metadata={k: str(v) for k, v in payload.items() if k not in ("content", "source_id")},
                )
            )

        log.info("retriever.search_results", num_results=len(search_results))

        # Step 4: Optionally rerank with Cohere
        if use_rerank and self._settings.cohere_api_key and search_results:
            search_results = self._rerank(query, search_results, top_k)
        else:
            search_results = search_results[:top_k]

        log.info("retriever.search_complete", num_final=len(search_results))
        return search_results

    def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank search results using Cohere Rerank.

        Args:
            query: Original user query (not HyDE).
            results: Search results to rerank.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked and trimmed list of RetrievalResult objects.
        """
        log = logger.bind(num_results=len(results), top_k=top_k)
        log.info("retriever.rerank_start")

        documents = [r.content for r in results]

        try:
            co = cohere.Client(api_key=self._settings.cohere_api_key)
            rerank_response = co.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model="rerank-english-v3.0",
            )
        except Exception as exc:
            log.warning("retriever.rerank_failed_fallback", error=str(exc))
            return results[:top_k]

        reranked: list[RetrievalResult] = []
        for item in rerank_response.results:
            original = results[item.index]
            reranked.append(
                RetrievalResult(
                    chunk_id=original.chunk_id,
                    content=original.content,
                    score=item.relevance_score,
                    source_id=original.source_id,
                    metadata=original.metadata,
                )
            )

        log.info("retriever.rerank_complete", num_reranked=len(reranked))
        return reranked
