"""Fact-Check Agent — validates the analysis summary for accuracy."""

from __future__ import annotations

from datetime import datetime, timezone

import openai
import structlog
from openai import AsyncOpenAI

from nexus.schemas.config import Settings
from nexus.schemas.state import NexusState, PipelineStage
from nexus.utils.progress import push_progress

logger = structlog.stdlib.get_logger(__name__)

FACTCHECK_SYSTEM_PROMPT = """\
You are a rigorous fact-checker reviewing a competitive intelligence analysis.
You will receive:
1. The original research query
2. The analysis summary to verify

For each major claim in the analysis, evaluate:
- Is the claim supported by the evidence cited?
- Are there logical gaps or unsupported leaps?
- Are numerical claims plausible?

Output a numbered list of verification results. For each item state:
- The claim being checked
- VERIFIED, UNVERIFIED, or QUESTIONABLE
- A brief explanation

End with an overall assessment: PASS, PASS_WITH_CAVEATS, or FAIL."""


from langfuse import observe

@observe(as_type="generation")
async def factcheck_node(state: NexusState) -> dict:
    """LangGraph node: fact-checks the analysis summary.

    Args:
        state: Current pipeline state with analysis_summary populated.

    Returns:
        Partial state update with fact_check_results.
    """
    run_id = state["run_id"]
    query = state["query"]
    analysis = state.get("analysis_summary", "")
    log = logger.bind(run_id=run_id)
    log.info("factcheck_agent.start")

    if not analysis:
        log.error("factcheck_agent.no_analysis")
        return {
            "stage": PipelineStage.FAILED,
            "errors": ["Fact-check agent received empty analysis"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    settings = Settings()
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )

    user_message = (
        f"Original query: {query}\n\n"
        f"Analysis to fact-check:\n\n{analysis}"
    )

    await push_progress(run_id, f"Verifying {len(analysis.split())} word analysis with LLM...", stage="fact_checking")

    try:
        response = await client.chat.completions.create(
            model=settings.openrouter_model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": FACTCHECK_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except openai.APIConnectionError as exc:
        log.error("factcheck_agent.connection_error", error=str(exc))
        return {
            "stage": PipelineStage.FAILED,
            "errors": [f"Fact-check LLM connection error: {exc}"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except openai.RateLimitError as exc:
        log.error("factcheck_agent.rate_limit", error=str(exc))
        return {
            "stage": PipelineStage.FAILED,
            "errors": [f"Fact-check LLM rate limit: {exc}"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except openai.APIStatusError as exc:
        log.error("factcheck_agent.api_error", status=exc.status_code, error=str(exc))
        return {
            "stage": PipelineStage.FAILED,
            "errors": [f"Fact-check LLM error ({exc.status_code}): {exc}"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    raw_text = response.choices[0].message.content or ""
    log.info("factcheck_agent.complete", response_length=len(raw_text))

    # Split into individual check results
    results = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
    await push_progress(run_id, f"Fact-check complete — {len(results)} verification items", stage="fact_checking")

    return {
        "stage": PipelineStage.FACT_CHECKING,
        "fact_check_results": results,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
