"""Live progress messages — agents push here, SSE forwards to browser in real-time.

Pattern:
  Agent calls push_progress(run_id, "Searching Tavily...")
  SSE polls nexus:progress:{run_id} every second
  Browser receives the message and shows it under the active stage
"""

from __future__ import annotations

import json
import time

import redis.asyncio as aioredis
import structlog

from nexus.schemas.config import Settings

logger = structlog.get_logger(__name__)


async def push_progress(run_id: str, message: str, stage: str = "") -> None:
    """Append a live message to this run's Redis progress list.

    Fire-and-forget: failures are logged but never raised, so agents
    never crash because of a missing progress message.
    """
    settings = Settings()
    client: aioredis.Redis | None = None
    try:
        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        key = f"nexus:progress:{run_id}"
        payload: dict = {"msg": message, "ts": time.time()}
        if stage:
            payload["stage"] = stage
        await client.rpush(key, json.dumps(payload))
        await client.expire(key, 3600)
    except Exception as exc:
        logger.warning("progress.push_failed", run_id=run_id, error=str(exc))
    finally:
        if client:
            await client.aclose()
