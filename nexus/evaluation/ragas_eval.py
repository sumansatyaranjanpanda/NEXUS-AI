"""RAGAS Evaluation module for the Nexus Pipeline."""

from __future__ import annotations

from typing import List

import structlog
from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy

from nexus.schemas.config import Settings
from nexus.schemas.state import NexusState

logger = structlog.stdlib.get_logger(__name__)

# context_precision requires a 'reference' (ground truth answer) column we never have.
# answer_relevancy strictness=1 means 1 question generated per response (matches what
# OpenRouter returns), avoiding the "returned 1 instead of 3" warning.
answer_relevancy.strictness = 1
_METRICS = [faithfulness, answer_relevancy]


class _FastEmbedWrapper(Embeddings):
    """Minimal fastembed wrapper with a string `model` attribute.

    FastEmbedEmbeddings from langchain_community exposes the internal
    TextEmbedding object as `.model`, which causes a ValidationError in
    RAGAS 0.4.x usage tracking (it expects a string). This wrapper fixes that.
    """

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding
        self.model = model_name          # string — what RAGAS expects
        self._fe = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [e.tolist() for e in self._fe.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(self._fe.embed([text]))[0].tolist()


def evaluate_state(state: NexusState) -> dict | None:
    """Evaluate a completed Nexus pipeline state using RAGAS.

    Args:
        state: The final NexusState containing the query, contexts, and response.

    Returns:
        A dictionary of metric name → float score, or None if evaluation failed.
    """
    run_id = state.get("run_id", "unknown")
    log = logger.bind(run_id=run_id)
    log.info("ragas_eval.start")

    query = state.get("query")
    contexts = state.get("retrieved_context")
    response = state.get("analysis_summary")

    if not query or not contexts or not response:
        log.warning(
            "ragas_eval.missing_data",
            has_query=bool(query),
            has_contexts=bool(contexts),
            has_response=bool(response),
        )
        return None

    settings = Settings()

    try:
        eval_llm = ChatOpenAI(
            model="openai/gpt-4o-mini",  # cheap + fast — sufficient for RAGAS scoring
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            max_tokens=4096,
        )
        ragas_llm = LangchainLLMWrapper(eval_llm)
    except Exception as exc:
        log.error("ragas_eval.llm_init_failed", error=str(exc))
        return None

    try:
        ragas_embeddings = LangchainEmbeddingsWrapper(
            _FastEmbedWrapper(settings.embedding_model)
        )
    except Exception as exc:
        log.error("ragas_eval.embeddings_init_failed", error=str(exc))
        return None

    data = {
        "user_input": [query],
        "retrieved_contexts": [contexts],
        "response": [response],
    }

    try:
        dataset = Dataset.from_dict(data)
    except Exception as exc:
        log.error("ragas_eval.dataset_creation_failed", error=str(exc))
        return None

    try:
        result = evaluate(
            dataset=dataset,
            metrics=_METRICS,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            show_progress=False,
        )
        # result.scores is a list of dicts — one dict per sample.
        # We always have exactly one sample so index [0] is safe.
        scores = {
            k: round(float(v), 4) if v is not None else None
            for k, v in result.scores[0].items()
        }
        log.info("ragas_eval.complete", scores=scores)
        return scores
    except Exception as exc:
        log.error("ragas_eval.execution_failed", error=str(exc))
        return None
