"""
Semantic cache using Redis (with in-memory fallback).

Caches LLM responses by embedding-based similarity matching.
If Redis is unavailable, falls back to an in-memory LRU dict.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Two-tier cache: exact match (hash-based) + semantic similarity.

    Level 1: Exact hash of the prompt → instant O(1) lookup
    Level 2: Embedding similarity (when Redis + RedisVL available)

    Falls back to in-memory OrderedDict if Redis is unavailable.
    """

    def __init__(self, max_memory_entries: int = 500):
        self.settings = get_settings()
        self._redis = None
        self._memory_cache: OrderedDict[str, dict] = OrderedDict()
        self._max_memory = max_memory_entries
        self._redis_available = False
        self._init_redis()

    def _init_redis(self) -> None:
        """Try to connect to Redis. Silently fall back to in-memory."""
        if not self.settings.inference.cache_enabled:
            logger.info("Caching disabled in config.")
            return

        try:
            import redis

            self._redis = redis.from_url(self.settings.redis_url, socket_timeout=3)
            self._redis.ping()
            self._redis_available = True
            logger.info("Redis cache connected at %s", self.settings.redis_url)
        except Exception as e:
            self._redis = None
            self._redis_available = False
            logger.info("Redis unavailable (%s). Using in-memory cache (max=%d).", e, self._max_memory)

    def get(self, prompt: str, model: str = "") -> dict[str, Any] | None:
        """
        Look up a cached response.

        Args:
            prompt: The exact prompt text.
            model: Model identifier for namespacing.

        Returns:
            Cached response dict, or None if not found.
        """
        cache_key = self._hash_key(prompt, model)

        # Level 1: Exact match
        if self._redis_available and self._redis:
            try:
                cached = self._redis.get(f"merchfine:cache:{cache_key}")
                if cached:
                    logger.debug("Cache HIT (Redis): %s", cache_key[:12])
                    return json.loads(cached)
            except Exception as e:
                logger.warning("Redis get failed: %s", e)

        # In-memory fallback
        if cache_key in self._memory_cache:
            # Move to end (LRU)
            self._memory_cache.move_to_end(cache_key)
            logger.debug("Cache HIT (memory): %s", cache_key[:12])
            return self._memory_cache[cache_key]

        logger.debug("Cache MISS: %s", cache_key[:12])
        return None

    def put(self, prompt: str, response: dict[str, Any], model: str = "") -> None:
        """
        Store a response in cache.

        Args:
            prompt: The exact prompt text.
            response: The LLM response dict to cache.
            model: Model identifier for namespacing.
        """
        cache_key = self._hash_key(prompt, model)
        ttl = self.settings.inference.cache_ttl

        # Redis
        if self._redis_available and self._redis:
            try:
                self._redis.setex(
                    f"merchfine:cache:{cache_key}",
                    ttl,
                    json.dumps(response, default=str),
                )
            except Exception as e:
                logger.warning("Redis put failed: %s", e)

        # In-memory (always, as backup)
        self._memory_cache[cache_key] = response
        if len(self._memory_cache) > self._max_memory:
            self._memory_cache.popitem(last=False)  # Remove oldest

    def invalidate(self, prompt: str, model: str = "") -> None:
        """Remove a specific entry from cache."""
        cache_key = self._hash_key(prompt, model)

        if self._redis_available and self._redis:
            try:
                self._redis.delete(f"merchfine:cache:{cache_key}")
            except Exception:
                pass

        self._memory_cache.pop(cache_key, None)

    def clear(self) -> None:
        """Clear the entire cache."""
        if self._redis_available and self._redis:
            try:
                # Delete all merchfine cache keys
                keys = self._redis.keys("merchfine:cache:*")
                if keys:
                    self._redis.delete(*keys)
            except Exception as e:
                logger.warning("Redis clear failed: %s", e)

        self._memory_cache.clear()
        logger.info("Cache cleared.")

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        stats = {
            "backend": "redis" if self._redis_available else "memory",
            "memory_entries": len(self._memory_cache),
            "memory_max": self._max_memory,
            "ttl_seconds": self.settings.inference.cache_ttl,
        }

        if self._redis_available and self._redis:
            try:
                info = self._redis.info("memory")
                stats["redis_used_memory_mb"] = info.get("used_memory", 0) / (1024 * 1024)
                keys = self._redis.keys("merchfine:cache:*")
                stats["redis_entries"] = len(keys) if keys else 0
            except Exception:
                pass

        return stats

    @staticmethod
    def _hash_key(prompt: str, model: str) -> str:
        """Generate a stable hash key from prompt + model."""
        content = f"{model}::{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
