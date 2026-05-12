"""Analysis Agent — retrieves from KB and produces a competitive analysis summary."""

from __future__ import annotations

from datetime import datetime, timezone

import instructor
import openai
import structlog
from openai import AsyncOpenAI

from nexus.rag.retriever import NexusRetriever
from nexus.schemas.config import Settings
from nexus.schemas.report import CompetitiveReport
from nexus.schemas.state import NexusState, PipelineStage
from nexus.utils.progress import push_progress

logger = structlog.stdlib.get_logger(__name__)

ANALYSIS_SYSTEM_PROMPT = """\
You are a senior competitive intelligence analyst. You will receive:
1. A research query
2. Retrieved knowledge base passages relevant to that query

Produce a thorough competitive analysis summary that:
- Synthesizes the key findings into a coherent narrative
- Identifies competitive advantages, threats, and market dynamics
- Highlights data gaps or areas of uncertainty
- Provides actionable strategic implications

Write in clear, professional prose. Be specific and cite the evidence from the passages.
Do NOT fabricate information beyond what the passages provide."""


from langfuse import observe

@observe(as_type="generation")
async def analysis_node(state: NexusState) -> dict:
    """LangGraph node: retrieves relevant KB content and produces analysis.

    Args:
        state: Current pipeline state with KB populated.

    Returns:
        Partial state update with analysis_summary.
    """
    run_id = state["run_id"]
    query = state["query"]
    log = logger.bind(run_id=run_id, query=query[:100])
    log.info("analysis_agent.start")

    settings = Settings()

    await push_progress(run_id, "Generating HyDE document for query expansion...", stage="analyzing")
    context_text = ""
    retrieved_passages = []
    try:
        retriever = NexusRetriever(settings)
        results = await retriever.search(
            query=query,
            top_k=settings.rerank_top_k,
            use_hyde=True,
            use_rerank=bool(settings.cohere_api_key),
        )
        if results:
            retrieved_passages = [r.content for r in results]
            context_text = "\n\n---\n\n".join(
                f"[Passage {i+1}] (score: {r.score:.2f})\n{r.content}"
                for i, r in enumerate(results)
            )
            log.info("analysis_agent.context_retrieved", num_passages=len(results))
            rerank_note = " → Cohere reranked" if settings.cohere_api_key else ""
            await push_progress(run_id, f"Retrieved {len(results)} passages via hybrid search (dense + BM25){rerank_note}", stage="analyzing")
        else:
            log.warning("analysis_agent.no_kb_results")
            await push_progress(run_id, "No KB results — falling back to raw research findings...", stage="analyzing")
    except RuntimeError as exc:
        log.warning("analysis_agent.retrieval_failed", error=str(exc))
        await push_progress(run_id, f"KB retrieval failed ({exc}) — using raw findings...", stage="analyzing")

    # Fall back to raw research findings if KB retrieval yielded nothing
    if not context_text:
        findings = state.get("research_findings", [])
        if findings:
            parts = []
            for i, f in enumerate(findings):
                f_dict = f if isinstance(f, dict) else f.model_dump()
                content = f_dict.get("content", "")
                parts.append(f"[Finding {i+1}]\n{content}")
                retrieved_passages.append(content)
            context_text = "\n\n---\n\n".join(parts)
            log.info("analysis_agent.using_raw_findings", num_findings=len(findings))

    if not context_text:
        log.error("analysis_agent.no_context")
        return {
            "stage": PipelineStage.FAILED,
            "errors": ["Analysis agent has no context to analyze"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    await push_progress(run_id, f"Sending {len(retrieved_passages)} passages to LLM for structured analysis...", stage="analyzing")

    # Call OpenRouter through instructor to guarantee structured output
    client = instructor.from_openai(
        AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
    )
    user_message = f"Research query: {query}\n\nRetrieved passages:\n\n{context_text}"

    try:
        report: CompetitiveReport = await client.chat.completions.create(
            model=settings.openrouter_model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_model=CompetitiveReport,
        )
        report.query = query
        report.run_id = run_id
    except openai.APIConnectionError as exc:
        log.error("analysis_agent.connection_error", error=str(exc))
        return {
            "stage": PipelineStage.FAILED,
            "errors": [f"Analysis LLM connection error: {exc}"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except openai.RateLimitError as exc:
        log.error("analysis_agent.rate_limit", error=str(exc))
        return {
            "stage": PipelineStage.FAILED,
            "errors": [f"Analysis LLM rate limit: {exc}"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        log.error("analysis_agent.api_error", error=str(exc))
        return {
            "stage": PipelineStage.FAILED,
            "errors": [f"Analysis LLM error: {exc}"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    import guardrails as gd
    from guardrails.validator_base import Validator, register_validator, ValidationResult, PassResult, FailResult

    @register_validator(name="no_disclaimers", data_type="string")
    class NoDisclaimers(Validator):
        """Ensure the model doesn't output unhelpful AI disclaimers."""
        def validate(self, value: str, metadata: dict) -> ValidationResult:
            disclaimers = ["as an ai", "i cannot", "i don't have access"]
            for d in disclaimers:
                if d in value.lower():
                    return FailResult(error_message=f"Contains AI disclaimer: {d}")
            return PassResult()

    log.info("analysis_agent.complete", sections=len(report.sections))
    await push_progress(run_id, f"Analysis complete — {len(report.sections)} sections generated", stage="analyzing")

    guard = gd.Guard.for_string(validators=[NoDisclaimers(on_fail="noop")])
    validation = guard.validate(report.executive_summary)
    if not validation.validation_passed:
        log.warning("analysis_agent.guardrail_failed", error=str(validation.error))
        return {
            "stage": PipelineStage.FAILED,
            "errors": [f"Guardrail failed: {validation.error}"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "stage": PipelineStage.ANALYZING,
        "retrieved_context": retrieved_passages,
        "analysis_summary": report.executive_summary,
        "structured_report": report.model_dump(mode="json"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
