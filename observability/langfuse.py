"""
Langfuse Tracing Integration.

Hooks into LiteLLM and Langchain to trace all LLM calls, RAG retrievals,
and agent steps. Traces are correlated using trace_ids.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)


class LangfuseTracker:
    """Manages Langfuse tracing configuration and initialization."""

    def __init__(self):
        self.settings = get_settings()
        self._enabled = bool(
            self.settings.langfuse_public_key and self.settings.langfuse_secret_key
        )
        self._callbacks = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def init_tracing(self) -> None:
        """Initialize globals for Langfuse if enabled."""
        if not self._enabled:
            logger.info("Langfuse tracing disabled (keys not set).")
            return

        try:
            import litellm
            
            # Configure LiteLLM to use Langfuse
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.settings.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.settings.langfuse_secret_key
            os.environ["LANGFUSE_HOST"] = self.settings.langfuse_host
            
            # litellm config
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["langfuse"]
            
            logger.info("Langfuse tracing enabled for LiteLLM.")
        except ImportError:
            logger.warning("litellm missing, tracing cannot be enabled.")

    def get_langchain_callbacks(self) -> list[Any]:
        """Get the Langchain CallbackHandler for graph tracing."""
        if not self._enabled:
            return []

        if self._callbacks is None:
            try:
                from langfuse.callback import CallbackHandler
                self._callbacks = [CallbackHandler(
                    public_key=self.settings.langfuse_public_key,
                    secret_key=self.settings.langfuse_secret_key,
                    host=self.settings.langfuse_host,
                )]
            except ImportError:
                logger.warning("langfuse SDK missing.")
                self._callbacks = []

        return self._callbacks
