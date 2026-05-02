"""Golden dataset for systematic RAGAS evaluation.

Purpose:
  Competitive intelligence systems are hard to evaluate without ground truth.
  This module provides:
    1. GOLDEN_DATASET — curated query/reference pairs for 3 real-world scenarios.
    2. evaluate_samples() — fast offline eval using pre-written responses (no pipeline run).
    3. evaluate_with_pipeline() — full end-to-end eval that actually runs the pipeline.

Metric used: answer_correctness (measures factual accuracy vs ground truth).
Unlike faithfulness/relevancy, this requires reference answers — so it's only
usable with golden data, not live queries.
"""

from __future__ import annotations

from typing import Any

import structlog
from datasets import Dataset

logger = structlog.stdlib.get_logger(__name__)

# ---------------------------------------------------------------------------
# Curated Q&A pairs. 'reference' is the ground truth; 'sample_response' is a
# manually-written answer used for quick offline testing without running pipeline.
# ---------------------------------------------------------------------------

GOLDEN_DATASET: list[dict[str, Any]] = [
    {
        "user_input": "Compare Tesla and Rivian on EV production volume in 2023",
        "reference": (
            "Tesla produced approximately 1.85 million vehicles in 2023, a 35% YoY increase. "
            "Rivian produced 57,232 vehicles, exceeding their own guidance of 54,000 units. "
            "Tesla's scale is roughly 32x larger than Rivian's."
        ),
        "sample_response": (
            "Tesla led EV production with 1.85 million vehicles in 2023, a 35% increase year-over-year. "
            "Rivian produced approximately 57,000 vehicles, exceeding their 54,000 guidance. "
            "The gap highlights Tesla's significant manufacturing scale advantage of over 30x."
        ),
    },
    {
        "user_input": "What is OpenAI's competitive position against Anthropic in enterprise AI?",
        "reference": (
            "OpenAI leads enterprise AI with ChatGPT Enterprise reaching 100,000+ enterprise users by end 2023. "
            "Anthropic competes with Claude for Business, targeting safety-conscious enterprises. "
            "Anthropic raised $7.3 billion in 2023-2024 to scale enterprise distribution."
        ),
        "sample_response": (
            "OpenAI maintains market leadership in enterprise AI through ChatGPT Enterprise and GPT-4 API. "
            "Anthropic positions Claude as the safety-focused alternative, "
            "backed by $7B+ in recent funding. Both compete for Fortune 500 adoption."
        ),
    },
    {
        "user_input": "How does Stripe compare to Adyen in payment processing for enterprises?",
        "reference": (
            "Stripe processed over $1 trillion in payment volume in 2023, excelling in developer experience. "
            "Adyen processed €845 billion in 2023, dominating unified global enterprise payments. "
            "Stripe leads in startup and mid-market adoption; Adyen leads in multinational enterprise."
        ),
        "sample_response": (
            "Stripe leads in developer-friendly payment processing with $1T+ annual volume. "
            "Adyen dominates enterprise global payments with €845B processed and a unified platform. "
            "They increasingly compete in the mid-market but with distinct strengths."
        ),
    },
]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_eval_dataset(with_responses: bool = True) -> Dataset:
    """Build a RAGAS-compatible HuggingFace Dataset from GOLDEN_DATASET.

    Args:
        with_responses: Include 'sample_response' as the 'response' column.
                        Set False if you plan to fill responses from a pipeline run.
    """
    records = []
    for item in GOLDEN_DATASET:
        record: dict[str, Any] = {
            "user_input": item["user_input"],
            "reference": item["reference"],
        }
        if with_responses:
            record["response"] = item["sample_response"]
        records.append(record)
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------


def evaluate_samples() -> dict | None:
    """Evaluate pre-written sample responses against reference answers.

    This runs entirely offline — no pipeline invocation, no LLM calls for generation.
    The only LLM calls are from RAGAS itself acting as a judge.
    Good for CI pipelines where you want a fast quality signal.
    """
    from ragas import evaluate
    from ragas.metrics._answer_correctness import answer_correctness
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from nexus.evaluation.ragas_eval import _FastEmbedWrapper
    from nexus.schemas.config import Settings

    settings = Settings()
    log = logger.bind(mode="samples", num_items=len(GOLDEN_DATASET))
    log.info("golden_eval.start")

    try:
        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.openrouter_model,
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(_FastEmbedWrapper(settings.embedding_model))
    except Exception as exc:
        log.error("golden_eval.init_failed", error=str(exc))
        return None

    dataset = build_eval_dataset(with_responses=True)

    try:
        result = evaluate(
            dataset=dataset,
            metrics=[answer_correctness],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            show_progress=False,
        )
        all_scores = result.scores
        avg = {
            k: round(sum(float(s.get(k) or 0) for s in all_scores) / len(all_scores), 4)
            for k in all_scores[0]
        }
        log.info("golden_eval.complete", avg_scores=avg)
        return {
            "mode": "samples",
            "num_items": len(all_scores),
            "avg_scores": avg,
            "per_item_scores": all_scores,
        }
    except Exception as exc:
        log.error("golden_eval.ragas_failed", error=str(exc))
        return None


async def evaluate_with_pipeline() -> dict | None:
    """Run each golden query through the full pipeline then evaluate.

    This is the definitive quality measurement: real pipeline output vs ground truth.
    Uses eval_graph (no HITL interrupt) so it runs to completion automatically.
    Expect ~60s per item due to LLM + embedding + search latency.
    """
    from nexus.agents.supervisor import eval_graph
    from nexus.schemas.state import create_initial_state

    log = logger.bind(mode="pipeline", num_items=len(GOLDEN_DATASET))
    log.info("golden_eval.pipeline_start")

    pipeline_responses = []
    for item in GOLDEN_DATASET:
        query = item["user_input"]
        item_log = logger.bind(query=query[:80])
        item_log.info("golden_eval.running_query")

        state = create_initial_state(query)
        thread_config = {"configurable": {"thread_id": state["run_id"]}}

        try:
            result = await eval_graph.ainvoke(state, config=thread_config)
            response = result.get("analysis_summary", "")
            item_log.info("golden_eval.query_done", response_len=len(response))
        except Exception as exc:
            item_log.error("golden_eval.query_failed", error=str(exc))
            response = ""

        pipeline_responses.append({
            "user_input": item["user_input"],
            "reference": item["reference"],
            "response": response,
        })

    valid = [r for r in pipeline_responses if r["response"]]
    if not valid:
        log.error("golden_eval.no_valid_responses")
        return None

    from ragas import evaluate
    from ragas.metrics._answer_correctness import answer_correctness
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from nexus.evaluation.ragas_eval import _FastEmbedWrapper
    from nexus.schemas.config import Settings

    settings = Settings()
    try:
        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.openrouter_model,
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(_FastEmbedWrapper(settings.embedding_model))
        dataset = Dataset.from_list(valid)
        result = evaluate(
            dataset=dataset,
            metrics=[answer_correctness],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            show_progress=False,
        )
        all_scores = result.scores
        avg = {
            k: round(sum(float(s.get(k) or 0) for s in all_scores) / len(all_scores), 4)
            for k in all_scores[0]
        }
        log.info("golden_eval.pipeline_complete", avg_scores=avg)
        return {
            "mode": "pipeline",
            "num_items": len(all_scores),
            "avg_scores": avg,
            "per_item_scores": all_scores,
        }
    except Exception as exc:
        log.error("golden_eval.pipeline_eval_failed", error=str(exc))
        return None
