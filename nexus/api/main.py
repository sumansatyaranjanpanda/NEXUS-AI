"""Nexus FastAPI application — the main entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nexus.api.routers import kb, reports, research
from nexus.logging_config import setup_logging
from nexus.schemas.config import Settings

logger = structlog.stdlib.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup / shutdown lifecycle."""
    settings = Settings()
    setup_logging(settings.log_level)
    logger.info("nexus.startup", env=settings.app_env, model=settings.openrouter_model)
    yield
    logger.info("nexus.shutdown")


app = FastAPI(
    title="Nexus — Competitive Intelligence",
    description="Autonomous multi-agent competitive intelligence system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — wide open for development, lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(research.router, prefix="/v1")
app.include_router(reports.router, prefix="/v1")
app.include_router(kb.router, prefix="/v1")


# --- Health ---


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
