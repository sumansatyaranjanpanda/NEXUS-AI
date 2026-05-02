"""Reports router — pipeline run, HITL approval, SSE streaming, and evaluation."""

from __future__ import annotations

import asyncio
import json
import uuid

import redis.asyncio as aioredis

from pydantic import BaseModel, Field

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from nexus.agents.supervisor import async_graph_ctx, compiled_graph
from nexus.schemas.config import Settings
from nexus.schemas.state import NexusState, PipelineStage, create_initial_state

logger = structlog.stdlib.get_logger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])


# ---------------------------------------------------------------------------
# Shared state helper
# ---------------------------------------------------------------------------


async def _read_run_state(run_id: str):
    """Read pipeline state — tries MemorySaver (sync flow), then Redis (Celery flow).

    Two storage backends exist:
      MemorySaver  — used by POST /run (FastAPI in-process, same memory space)
      AsyncRedisSaver — used by Celery tasks (different OS process, shared via Redis)

    We try MemorySaver first (fast, no network), then fall back to Redis.
    Returns the LangGraph state snapshot or None if not found in either.
    """
    thread_config = {"configurable": {"thread_id": run_id}}

    state = await compiled_graph.aget_state(thread_config)
    if state and state.values:
        return state

    try:
        async with async_graph_ctx() as graph:
            state = await graph.aget_state(thread_config)
            if state and state.values:
                return state
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PipelineRequest(BaseModel):
    """Start a new competitive intelligence pipeline run."""

    query: str = Field(min_length=3, max_length=2000, description="The competitive intelligence question")


class PipelineStatusResponse(BaseModel):
    """Current status of a pipeline run."""

    run_id: str
    query: str
    stage: str
    analysis_summary: str = ""
    fact_check_results: list[str] = Field(default_factory=list)
    report_markdown: str = ""
    errors: list[str] = Field(default_factory=list)
    is_waiting_for_approval: bool = False


class ApprovalRequest(BaseModel):
    """Approve or reject a HITL gate."""

    run_id: str
    approved: bool = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run", response_model=PipelineStatusResponse, status_code=200)
async def start_pipeline(request: PipelineRequest) -> PipelineStatusResponse:
    """Start a full pipeline run. Runs until HITL interrupt or completion.

    The pipeline will pause at the HITL gate (after fact-checking) and
    return with is_waiting_for_approval=True. Use the /approve endpoint
    to resume.
    """
    log = logger.bind(query=request.query[:100])
    log.info("pipeline.start")

    state = create_initial_state(request.query)
    run_id = state["run_id"]
    thread_config = {"configurable": {"thread_id": run_id}}

    try:
        result = await compiled_graph.ainvoke(state, config=thread_config)
    except Exception as exc:
        log.error("pipeline.execution_error", error=str(exc), run_id=run_id)
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {exc}") from exc

    # Check if we're paused at HITL
    graph_state = await compiled_graph.aget_state(thread_config)
    is_waiting = "hitl_gate" in (graph_state.next or ())  # non-empty means interrupted

    final_stage = result.get("stage", PipelineStage.FAILED)
    log.info("pipeline.paused_or_complete", stage=str(final_stage), waiting=is_waiting, run_id=run_id)

    return PipelineStatusResponse(
        run_id=run_id,
        query=request.query,
        stage=str(final_stage),
        analysis_summary=result.get("analysis_summary", ""),
        fact_check_results=result.get("fact_check_results", []),
        report_markdown=result.get("report_markdown", ""),
        errors=result.get("errors", []),
        is_waiting_for_approval=is_waiting,
    )


@router.post("/approve", response_model=PipelineStatusResponse, status_code=200)
async def approve_pipeline(request: ApprovalRequest) -> PipelineStatusResponse:
    """Resume a pipeline run after HITL approval.

    If approved=False, the pipeline is marked as failed.
    """
    log = logger.bind(run_id=request.run_id, approved=request.approved)
    log.info("pipeline.approval_received")

    thread_config = {"configurable": {"thread_id": request.run_id}}

    # Check current graph state
    graph_state = await compiled_graph.aget_state(thread_config)
    if not graph_state or not graph_state.next:
        raise HTTPException(
            status_code=404,
            detail=f"No pipeline waiting for approval with run_id={request.run_id}",
        )

    if not request.approved:
        # Human rejected — update state to FAILED
        await compiled_graph.aupdate_state(
            thread_config,
            {
                "stage": PipelineStage.FAILED,
                "errors": ["Rejected by human reviewer"],
            },
        )
        log.info("pipeline.rejected", run_id=request.run_id)
        current = await compiled_graph.aget_state(thread_config)
        vals = current.values
        return PipelineStatusResponse(
            run_id=request.run_id,
            query=vals.get("query", ""),
            stage=str(vals.get("stage", PipelineStage.FAILED)),
            errors=vals.get("errors", []),
            is_waiting_for_approval=False,
        )

    # Resume the graph (pass None to continue from interrupt)
    try:
        result = await compiled_graph.ainvoke(None, config=thread_config)
    except Exception as exc:
        log.error("pipeline.resume_error", error=str(exc), run_id=request.run_id)
        raise HTTPException(status_code=500, detail=f"Pipeline resume error: {exc}") from exc

    log.info("pipeline.complete", run_id=request.run_id, stage=str(result.get("stage")))

    return PipelineStatusResponse(
        run_id=request.run_id,
        query=result.get("query", ""),
        stage=str(result.get("stage", PipelineStage.COMPLETE)),
        analysis_summary=result.get("analysis_summary", ""),
        fact_check_results=result.get("fact_check_results", []),
        report_markdown=result.get("report_markdown", ""),
        errors=result.get("errors", []),
        is_waiting_for_approval=False,
    )


@router.get("/status/{run_id}", response_model=PipelineStatusResponse, status_code=200)
async def get_pipeline_status(run_id: str) -> PipelineStatusResponse:
    """Check the current status of a pipeline run (sync or Celery async)."""
    graph_state = await _read_run_state(run_id)
    if not graph_state or not graph_state.values:
        raise HTTPException(status_code=404, detail=f"No pipeline found with run_id={run_id}")

    vals = graph_state.values
    is_waiting = "hitl_gate" in (graph_state.next or ())

    return PipelineStatusResponse(
        run_id=run_id,
        query=vals.get("query", ""),
        stage=str(vals.get("stage", "")),
        analysis_summary=vals.get("analysis_summary", ""),
        fact_check_results=vals.get("fact_check_results", []),
        report_markdown=vals.get("report_markdown", ""),
        errors=vals.get("errors", []),
        is_waiting_for_approval=is_waiting,
    )


@router.post("/status/{run_id}/evaluate", response_model=dict, status_code=200)
async def evaluate_pipeline(run_id: str) -> dict:
    """Evaluate a completed pipeline run using RAGAS (faithfulness + answer relevancy)."""
    graph_state = await _read_run_state(run_id)
    if not graph_state or not graph_state.values:
        raise HTTPException(status_code=404, detail=f"No pipeline found with run_id={run_id}")

    vals = graph_state.values
    if vals.get("stage") != PipelineStage.COMPLETE:
        raise HTTPException(
            status_code=400,
            detail="Cannot evaluate an incomplete pipeline run. Finish or approve it first.",
        )

    from nexus.evaluation.ragas_eval import evaluate_state

    log = logger.bind(run_id=run_id)
    log.info("reports.evaluate_start")

    scores = await asyncio.to_thread(evaluate_state, vals)
    if scores is None:
        raise HTTPException(status_code=500, detail="Evaluation failed to execute. Check logs.")

    log.info("reports.evaluate_complete", scores=scores)
    return {"run_id": run_id, "scores": scores}


# ---------------------------------------------------------------------------
# Async pipeline via Celery
# ---------------------------------------------------------------------------


@router.post("/run/async", response_model=dict, status_code=202)
async def start_pipeline_async(request: PipelineRequest) -> dict:
    """Start pipeline asynchronously via Celery — returns run_id immediately.

    The pipeline runs in a Celery worker. Poll GET /status/{run_id} or connect
    to GET /stream/{run_id} for real-time updates.

    Requirements:
      - Celery worker: celery -A nexus.workers.tasks worker --loglevel=info
      - redis-stack image (not plain redis) so Celery and FastAPI share graph state
    """
    from nexus.workers.tasks import run_pipeline_task

    run_id = str(uuid.uuid4())
    log = logger.bind(run_id=run_id, query=request.query[:80])
    log.info("pipeline.async_dispatch")

    task = run_pipeline_task.delay(request.query, run_id)

    return {
        "run_id": run_id,
        "task_id": task.id,
        "status": "queued",
        "poll_url": f"/v1/reports/status/{run_id}",
        "stream_url": f"/v1/reports/stream/{run_id}",
    }


@router.post("/approve/async", response_model=dict, status_code=202)
async def approve_pipeline_async(request: ApprovalRequest) -> dict:
    """Resume a HITL-paused pipeline via Celery — returns immediately."""
    from nexus.workers.tasks import resume_pipeline_task

    log = logger.bind(run_id=request.run_id, approved=request.approved)
    log.info("pipeline.async_resume_dispatch")

    task = resume_pipeline_task.delay(request.run_id, request.approved)
    return {
        "run_id": request.run_id,
        "task_id": task.id,
        "status": "queued",
        "poll_url": f"/v1/reports/status/{request.run_id}",
    }


# ---------------------------------------------------------------------------
# Server-Sent Events (SSE) streaming
# ---------------------------------------------------------------------------


@router.get("/stream/{run_id}")
async def stream_pipeline(run_id: str, request: Request) -> StreamingResponse:
    """Stream pipeline stage updates via Server-Sent Events.

    Connect from JavaScript:
        const source = new EventSource('/v1/reports/stream/<run_id>')
        source.onmessage = (e) => { const data = JSON.parse(e.data); ... }
        source.onerror = () => source.close()

    Emits a JSON event whenever the pipeline stage changes.
    Closes when pipeline reaches COMPLETE or FAILED, or pauses at HITL.
    """
    async def _event_generator():
        thread_config = {"configurable": {"thread_id": run_id}}
        last_stage: str | None = None
        last_waiting: bool = False

        settings = Settings()
        progress_key = f"nexus:progress:{run_id}"
        progress_offset = 0
        redis_client: aioredis.Redis | None = None
        try:
            redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
        except Exception:
            pass

        try:
            async with async_graph_ctx() as graph:
                for _ in range(300):  # 5-minute hard timeout (covers HITL pause time)
                    if await request.is_disconnected():
                        break

                    try:
                        graph_state = await graph.aget_state(thread_config)
                    except Exception:
                        await asyncio.sleep(1)
                        continue

                    # Read new progress log entries pushed by agents during this poll interval
                    new_logs: list[dict] = []
                    if redis_client:
                        try:
                            total = await redis_client.llen(progress_key)
                            if total > progress_offset:
                                raw = await redis_client.lrange(progress_key, progress_offset, -1)
                                new_logs = [json.loads(m) for m in raw]
                                progress_offset = total
                        except Exception:
                            pass

                    if graph_state and graph_state.values:
                        vals = graph_state.values
                        current_stage = str(vals.get("stage", ""))
                        is_waiting = "hitl_gate" in (graph_state.next or ())

                        # Emit whenever stage, waiting status, or new log lines arrive
                        if current_stage != last_stage or is_waiting != last_waiting or new_logs:
                            last_stage = current_stage
                            last_waiting = is_waiting
                            event = {
                                "run_id": run_id,
                                "stage": current_stage,
                                "is_waiting_for_approval": is_waiting,
                                "analysis_summary": vals.get("analysis_summary", ""),
                                "errors": vals.get("errors", []),
                                "logs": new_logs,
                            }
                            yield f"data: {json.dumps(event)}\n\n"

                        terminal = current_stage in (
                            str(PipelineStage.COMPLETE),
                            str(PipelineStage.FAILED),
                        )
                        if terminal:
                            break
                        # Do NOT break on is_waiting — SSE stays open during HITL pause.

                    await asyncio.sleep(1)

        finally:
            if redis_client:
                await redis_client.aclose()

        yield f"data: {json.dumps({'event': 'stream_end', 'run_id': run_id})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx response buffering
        },
    )


# ---------------------------------------------------------------------------
# Golden dataset evaluation
# ---------------------------------------------------------------------------


@router.post("/evaluate/golden", response_model=dict, status_code=200)
async def evaluate_golden(full_pipeline: bool = False) -> dict:
    """Evaluate system quality against the curated golden dataset.

    Args:
        full_pipeline: If False (default) — evaluate pre-written sample responses.
                       Fast (~30s, good for CI checks).
                       If True — run each golden query through the real pipeline.
                       Slow (~3-5 min) but gives true end-to-end quality signal.
    """
    from nexus.evaluation.golden_dataset import evaluate_samples, evaluate_with_pipeline

    log = logger.bind(full_pipeline=full_pipeline)
    log.info("reports.golden_eval_start")

    if full_pipeline:
        result = await evaluate_with_pipeline()
    else:
        result = await asyncio.to_thread(evaluate_samples)

    if result is None:
        raise HTTPException(
            status_code=500,
            detail="Golden dataset evaluation failed. Check logs for RAGAS errors.",
        )

    log.info("reports.golden_eval_complete", mode=result.get("mode"))
    return result
