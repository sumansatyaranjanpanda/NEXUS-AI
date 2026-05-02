"""Celery task workers for async pipeline execution.

Architecture:
  FastAPI      → dispatches task → returns run_id immediately (non-blocking)
  Celery worker → runs pipeline  → writes state to AsyncRedisSaver in Redis
  FastAPI      → GET /status/{run_id} reads same Redis via async_graph_ctx()

Why AsyncRedisSaver (not RedisSaver):
  LangGraph's ainvoke() calls checkpointer.aget_tuple() (async).
  RedisSaver is synchronous and raises NotImplementedError there.
  AsyncRedisSaver implements the full async checkpoint interface.

Why asyncio.run() in Celery tasks:
  Celery task functions are synchronous. asyncio.run() creates a fresh event
  loop per task call — correct for Celery workers, which may run multiple
  tasks sequentially in the same thread.
"""

from __future__ import annotations

import asyncio

from celery import Celery

from nexus.schemas.config import Settings

_settings = Settings()

celery_app = Celery(
    "nexus",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, name="nexus.run_pipeline")
def run_pipeline_task(self, query: str, run_id: str) -> dict:
    """Run the full Nexus pipeline for a query.

    Uses async_graph_ctx() so state is written to AsyncRedisSaver (Redis),
    where FastAPI status and SSE endpoints can read it cross-process.
    """
    from nexus.agents.supervisor import async_graph_ctx
    from nexus.schemas.state import PipelineStage, create_initial_state

    async def _run() -> dict:
        state = create_initial_state(query)
        state["run_id"] = run_id
        thread_config = {"configurable": {"thread_id": run_id}}
        async with async_graph_ctx() as graph:
            result = await graph.ainvoke(state, config=thread_config)
        return result

    result = asyncio.run(_run())
    return {
        "run_id": run_id,
        "stage": str(result.get("stage", PipelineStage.FAILED)),
    }


@celery_app.task(bind=True, name="nexus.resume_pipeline")
def resume_pipeline_task(self, run_id: str, approved: bool) -> dict:
    """Resume pipeline from HITL interrupt after human approval/rejection.

    Must open a fresh async_graph_ctx() to read the interrupted state from
    Redis (where run_pipeline_task wrote it) and resume from there.
    """
    from nexus.agents.supervisor import async_graph_ctx
    from nexus.schemas.state import PipelineStage

    async def _resume() -> dict:
        thread_config = {"configurable": {"thread_id": run_id}}
        async with async_graph_ctx() as graph:
            if not approved:
                await graph.aupdate_state(
                    thread_config,
                    {"stage": PipelineStage.FAILED, "errors": ["Rejected by human reviewer"]},
                )
                return {"run_id": run_id, "stage": str(PipelineStage.FAILED)}
            result = await graph.ainvoke(None, config=thread_config)
            return {
                "run_id": run_id,
                "stage": str(result.get("stage", PipelineStage.FAILED)),
            }

    return asyncio.run(_resume())
