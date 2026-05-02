"""Structured report schema — the final output of the Nexus pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """A citation used in the report."""

    title: str
    url: str = ""
    snippet: str = ""


class ReportSection(BaseModel):
    """One section of a competitive intelligence report."""

    heading: str
    body: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in this section's accuracy")
    sources: list[SourceReference] = Field(default_factory=list)


class CompetitiveReport(BaseModel):
    """The fully structured competitive intelligence report."""

    title: str
    executive_summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    methodology: str = ""
    limitations: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query: str = Field(description="The original user query that produced this report")
    run_id: str = Field(description="Trace ID linking back to the pipeline run")
