"""Research Agent — real web search via Tavily + LLM synthesis.

Flow:
  1. Tavily API → 8 real web results with URLs, titles, content snippets
  2. LLM synthesizes those results into structured ResearchFindings
  3. Each finding gets the source URL from Tavily (real citation, not hallucination)

Graceful fallback: if TAVILY_API_KEY is missing or Tavily fails,
the agent falls back to pure LLM generation (old behaviour).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import openai
import structlog
from langfuse import observe
from openai import AsyncOpenAI

from nexus.schemas.config import Settings
from nexus.schemas.state import NexusState, PipelineStage, ResearchFinding
from nexus.utils.progress import push_progress

logger = structlog.stdlib.get_logger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """\
You are a competitive intelligence analyst. You have real web search results about the query below.
Synthesize them into 5-8 specific findings. For each finding:
- Clearly state the key insight
- Cite the source with its bracket number, e.g. "according to [1]" or "[3] reports that..."
- Explain its competitive significance
Only report what the search results contain. Do not add information from memory.
Format as a numbered list: 1. ... 2. ..."""

FALLBACK_SYSTEM_PROMPT = """\
You are a senior competitive intelligence analyst. Given a research query, produce
detailed findings as a numbered list. For each, state the insight, the evidence basis,
and competitive significance. Be specific."""


@observe(as_type="generation")
async def research_node(state: NexusState) -> dict:
    """LangGraph node: Tavily web search + LLM synthesis → ResearchFinding list."""
    query = state["query"]
    run_id = state["run_id"]
    log = logger.bind(run_id=run_id, query=query[:100])
    log.info("research_agent.start")

    settings = Settings()

    await push_progress(run_id, f"Searching Tavily: {query[:60]}{'...' if len(query) > 60 else ''}", stage="researching")
    tavily_results = await _search_web(query, settings, log)

    if tavily_results:
        await push_progress(run_id, f"Found {len(tavily_results)} web results — synthesizing with Claude...", stage="researching")
    else:
        await push_progress(run_id, "Tavily unavailable — falling back to LLM-only research...", stage="researching")

    findings = await _synthesize(query, tavily_results, settings, log)

    if not findings:
        return {
            "stage": PipelineStage.FAILED,
            "errors": ["Research agent produced no findings"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    await push_progress(run_id, f"Done — {len(findings)} findings extracted", stage="researching")

    return {
        "stage": PipelineStage.RESEARCHING,
        "research_findings": findings,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _search_web(query: str, settings: Settings, log) -> list[dict]:
    """Tavily web search. Returns list of {url, title, content} dicts.

    Returns empty list (not an error) if key is missing — caller handles fallback.
    """
    if not settings.tavily_api_key:
        log.warning("research_agent.tavily_key_missing", note="set TAVILY_API_KEY for real web search")
        return []
    try:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=settings.tavily_api_key)
        response = await client.search(
            query=query,
            max_results=8,
            search_depth="advanced",
        )
        results = response.get("results", [])
        log.info("research_agent.tavily_complete", num_results=len(results))
        return results
    except Exception as exc:
        log.error("research_agent.tavily_error", error=str(exc))
        return []


async def _synthesize(
    query: str,
    tavily_results: list[dict],
    settings: Settings,
    log,
) -> list[ResearchFinding]:
    """Pass Tavily results to LLM for structured synthesis."""
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )

    if tavily_results:
        context_block = "\n\n".join(
            f"[{i + 1}] URL: {r.get('url', 'unknown')}\n"
            f"Title: {r.get('title', '')}\n"
            f"Content: {r.get('content', '')[:600]}"
            for i, r in enumerate(tavily_results)
        )
        system = SYNTHESIS_SYSTEM_PROMPT
        user_msg = f"Research query: {query}\n\nSearch results:\n{context_block}"
        log.info("research_agent.using_tavily_results", num_results=len(tavily_results))
        model=settings.openrouter_model
    else:
        log.warning("research_agent.tavily_unavailable_llm_fallback")
        system = FALLBACK_SYSTEM_PROMPT
        user_msg = f"Research query: {query}"
        model="perplexity/sonar"

    try:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
    except openai.APIConnectionError as exc:
        log.error("research_agent.connection_error", error=str(exc))
        return []
    except openai.RateLimitError as exc:
        log.error("research_agent.rate_limit", error=str(exc))
        return []
    except openai.APIStatusError as exc:
        log.error("research_agent.api_error", status=exc.status_code, error=str(exc))
        return []

    raw_text = response.choices[0].message.content or ""
    log.info("research_agent.synthesis_complete", length=len(raw_text))
    return _parse_findings(raw_text, query, tavily_results)


def _parse_findings(
    raw_text: str,
    query: str,
    tavily_results: list[dict],
) -> list[ResearchFinding]:
    """Parse numbered LLM output → ResearchFinding list.

    URL attribution: the LLM uses [N] citations in its output.
    We extract those and map back to real Tavily URLs.
    """
    # Build 1-indexed URL lookup from Tavily results
    url_by_ref: dict[int, str] = {
        i + 1: r.get("url", "") for i, r in enumerate(tavily_results)
    }

    lines = raw_text.strip().split("\n")
    findings: list[ResearchFinding] = []
    current_block: list[str] = []
    found_numbered = False

    for line in lines:
        stripped = line.strip()
        # Detect "1.", "2.", "10." — start of a new numbered item
        if stripped and stripped[0].isdigit() and "." in stripped[:4]:
            found_numbered = True
            if current_block:
                content = "\n".join(current_block).strip()
                if content:
                    findings.append(_make_finding(content, query, url_by_ref))
            current_block = [stripped[stripped.index(".") + 1 :].strip()]
        else:
            current_block.append(stripped)

    # Flush last block
    if current_block:
        content = "\n".join(current_block).strip()
        if content:
            findings.append(
                _make_finding(content, query, url_by_ref, score=0.5 if not found_numbered else 0.85)
            )

    if not findings:
        findings.append(
            ResearchFinding(
                source=f"research:{query[:80]}",
                content=raw_text.strip() or "(empty response)",
                relevance_score=0.5,
            )
        )

    return findings


def _make_finding(
    content: str,
    query: str,
    url_by_ref: dict[int, str],
    score: float = 0.85,
) -> ResearchFinding:
    """Build a ResearchFinding, extracting a [N] citation from content if present."""
    source = f"research:{query[:80]}"
    match = re.search(r"\[(\d+)\]", content)
    if match:
        ref = int(match.group(1))
        if ref in url_by_ref and url_by_ref[ref]:
            source = url_by_ref[ref]
    return ResearchFinding(source=source, content=content, relevance_score=score)
