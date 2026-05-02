"""HyDE — Hypothetical Document Embedding via OpenRouter."""

from __future__ import annotations

import openai
import structlog
from openai import AsyncOpenAI

from nexus.schemas.config import Settings

logger = structlog.stdlib.get_logger(__name__)

HYDE_SYSTEM_PROMPT = """\
You are a competitive intelligence knowledge base. Given a query, write a short, \
factual passage (150-250 words) that would be the ideal answer found in a well-curated \
competitive intelligence database. Be specific and data-oriented. Do not hedge or \
add disclaimers — write as if this passage already exists in the database."""


async def generate_hypothetical_document(
    query: str,
    settings: Settings | None = None,
) -> str:
    """Generate a hypothetical document for HyDE retrieval.

    HyDE improves retrieval by embedding a hypothetical ideal answer
    instead of the raw query, which often better matches stored passages.

    Args:
        query: The user's search query.
        settings: App settings (uses defaults if not provided).

    Returns:
        A hypothetical document string for embedding.
    """
    settings = settings or Settings()
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    log = logger.bind(query=query[:100])
    log.info("hyde.generating")

    try:
        response = await client.chat.completions.create(
            model=settings.openrouter_model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ],
        )
    except openai.APIConnectionError as exc:
        log.error("hyde.connection_error", error=str(exc))
        raise RuntimeError(f"HyDE generation failed (connection): {exc}") from exc
    except openai.RateLimitError as exc:
        log.error("hyde.rate_limit", error=str(exc))
        raise RuntimeError(f"HyDE generation failed (rate limit): {exc}") from exc
    except openai.APIStatusError as exc:
        log.error("hyde.api_error", status=exc.status_code, error=str(exc))
        raise RuntimeError(f"HyDE generation failed ({exc.status_code}): {exc}") from exc

    hypothetical_doc = response.choices[0].message.content or ""
    log.info("hyde.complete", doc_length=len(hypothetical_doc))
    return hypothetical_doc
