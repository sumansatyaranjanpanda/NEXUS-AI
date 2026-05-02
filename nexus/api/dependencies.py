"""FastAPI dependency injection — settings singleton and shared resources."""

from __future__ import annotations

from functools import lru_cache

from nexus.schemas.config import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance — reads .env once at startup."""
    return Settings()
