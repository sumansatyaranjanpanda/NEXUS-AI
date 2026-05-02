"""Embedder — dense (bge-m3) and sparse (SPLADE) embeddings via fastembed."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from nexus.schemas.config import Settings

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding, TextEmbedding

logger = structlog.stdlib.get_logger(__name__)


class NexusEmbedder:
    """Wraps fastembed models for dense + sparse embedding generation.

    Lazily initializes models on first use to avoid slow import-time downloads.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._dense_model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None

    def _get_dense_model(self) -> TextEmbedding:
        """Lazy-load the dense embedding model."""
        if self._dense_model is None:
            from fastembed import TextEmbedding

            model_name = self._settings.embedding_model
            logger.info("embedder.loading_dense_model", model=model_name)
            self._dense_model = TextEmbedding(model_name=model_name)
            logger.info("embedder.dense_model_ready", model=model_name)
        return self._dense_model

    def _get_sparse_model(self) -> SparseTextEmbedding:
        """Lazy-load the sparse embedding model."""
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding

            model_name = self._settings.sparse_model
            logger.info("embedder.loading_sparse_model", model=model_name)
            self._sparse_model = SparseTextEmbedding(model_name=model_name)
            logger.info("embedder.sparse_model_ready", model=model_name)
        return self._sparse_model

    def embed_documents_dense(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            return []

        model = self._get_dense_model()
        try:
            embeddings = list(model.embed(texts))
            logger.info("embedder.dense_complete", num_texts=len(texts), dim=len(embeddings[0]))
            return [emb.tolist() for emb in embeddings]
        except Exception as exc:
            logger.error("embedder.dense_error", error=str(exc), num_texts=len(texts))
            raise RuntimeError(f"Dense embedding failed: {exc}") from exc

    def embed_query_dense(self, query: str) -> list[float]:
        """Generate a dense embedding for a single query.

        Args:
            query: The search query text.

        Returns:
            A single embedding vector.
        """
        model = self._get_dense_model()
        try:
            embeddings = list(model.query_embed(query))
            return embeddings[0].tolist()
        except Exception as exc:
            logger.error("embedder.query_dense_error", error=str(exc))
            raise RuntimeError(f"Query dense embedding failed: {exc}") from exc

    def embed_documents_sparse(self, texts: list[str]) -> list[dict[str, list]]:
        """Generate sparse embeddings (SPLADE) for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of dicts with 'indices' and 'values' keys for sparse vectors.
        """
        if not texts:
            return []

        model = self._get_sparse_model()
        try:
            raw_embeddings = list(model.embed(texts))
            sparse_vectors = []
            for emb in raw_embeddings:
                sparse_vectors.append({
                    "indices": emb.indices.tolist(),
                    "values": emb.values.tolist(),
                })
            logger.info("embedder.sparse_complete", num_texts=len(texts))
            return sparse_vectors
        except Exception as exc:
            logger.error("embedder.sparse_error", error=str(exc), num_texts=len(texts))
            raise RuntimeError(f"Sparse embedding failed: {exc}") from exc

    def embed_query_sparse(self, query: str) -> dict[str, list]:
        """Generate a sparse embedding for a single query.

        Args:
            query: The search query text.

        Returns:
            Dict with 'indices' and 'values' keys.
        """
        model = self._get_sparse_model()
        try:
            raw_embeddings = list(model.query_embed(query))
            emb = raw_embeddings[0]
            return {
                "indices": emb.indices.tolist(),
                "values": emb.values.tolist(),
            }
        except Exception as exc:
            logger.error("embedder.query_sparse_error", error=str(exc))
            raise RuntimeError(f"Query sparse embedding failed: {exc}") from exc
