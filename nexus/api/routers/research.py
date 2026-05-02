"""Research router — endpoints for triggering and checking research runs."""

from __future__ import annotations

from pydantic import BaseModel, Field

import structlog
from fastapi import APIRouter

from nexus.agents.research_agent import research_node
from nexus.schemas.state import NexusState, PipelineStage, ResearchFinding, create_initial_state

logger = structlog.stdlib.get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


# --- Request / Response models ---


class ResearchRequest(BaseModel):
    """Incoming research query."""

    query: str = Field(min_length=3, max_length=2000, description="The competitive intelligence question")


class ResearchResponse(BaseModel):
    """Response after research completes."""

    run_id: str
    query: str
    stage: PipelineStage
    findings: list[ResearchFinding]
    errors: list[str]


# --- Endpoints ---


@router.post("/", response_model=ResearchResponse, status_code=200)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    """Run the research agent synchronously and return findings.

    In Phase 3 this will become an async Celery task with polling.
    For now, we run inline so we can validate the full path.
    """
    log = logger.bind(query=request.query)
    log.info("research.request_received")

    state: NexusState = create_initial_state(request.query)
    result = await research_node(state)

    # Merge result into state
    final_stage: PipelineStage = result.get("stage", state["stage"])
    findings: list[ResearchFinding] = result.get("research_findings", [])
    errors: list[str] = state["errors"] + result.get("errors", [])

    log.info("research.request_complete", stage=final_stage, num_findings=len(findings))

    return ResearchResponse(
        run_id=state["run_id"],
        query=state["query"],
        stage=final_stage,
        findings=findings,
        errors=errors,
    )
