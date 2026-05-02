"""NexusState — the LangGraph typed state that flows through the entire pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class PipelineStage(StrEnum):
    """Tracks where a query is in the pipeline."""

    RECEIVED = "received"
    RESEARCHING = "researching"
    BUILDING_KB = "building_kb"
    ANALYZING = "analyzing"
    FACT_CHECKING = "fact_checking"
    HITL_REVIEW = "hitl_review"
    COMPLETE = "complete"
    FAILED = "failed"


class ResearchFinding(BaseModel):
    """A single piece of research returned by the research agent."""

    source: str = Field(description="Where this information came from")
    content: str = Field(description="The actual finding text")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How relevant to the query (0-1)")
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NexusState(TypedDict):
    """Top-level state that flows through the LangGraph pipeline.

    Uses LangGraph's `add_messages` reducer for the messages field
    so agents can append without overwriting.
    """

    # --- identity ---
    run_id: str # Unique identifier for this pipeline run, useful for tracking and debugging.
    query: str #user's original query

    # --- pipeline tracking ---
    stage: PipelineStage # The current stage of the pipeline
    created_at: str # Timestamp for when the pipeline run was created
    updated_at: str # Timestamp for when the pipeline run was last updated

    # --- agent outputs ---
    messages: Annotated[list, add_messages]
    research_findings: list[ResearchFinding] # Output of Research Agent
    retrieved_context: list[str] | None # Output of KB search
    analysis_summary: str # Output of Analysis Agent
    structured_report: dict | None 
    fact_check_results: list[str] # Output of FactCheck Agent

    # --- final output ---
    report_markdown: str # The final report

    # --- errors ---
    errors: list[str] # Anything that went wrong


def create_initial_state(query: str) -> NexusState:
    """Factory for a fresh pipeline state from a user query."""
    now = datetime.now(timezone.utc).isoformat()
    return NexusState(
        run_id=str(uuid.uuid4()),
        query=query,
        stage=PipelineStage.RECEIVED,
        created_at=now,
        updated_at=now,
        messages=[],
        research_findings=[],
        retrieved_context=None,
        analysis_summary="",
        structured_report=None,
        fact_check_results=[],
        report_markdown="",
        errors=[],
    )
